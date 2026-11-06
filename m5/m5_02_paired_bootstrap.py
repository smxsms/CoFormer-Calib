#!/usr/bin/env python3
"""m5_02_paired_bootstrap.py —— 患者级配对 bootstrap（审稿意见 2 的核心新增计算）。

输入：m5_01_dump_probs.py 落盘的 probs_<tag>.npz（逐样本概率）。
输出：ΔC 的配对区间（三层）+ 决策翻转分解（McNemar）+ 可直接粘进论文的数字。

估计量（与论文 §3.7 / alarm_cost.py 完全一致）：
  对固定阈值 τ，患者 i 的代价分子  a_i = w_FN·y_i·1[p_i<τ] + (1−y_i)·1[p_i≥τ]
  归一化分母                    D   = w_FN·π + (1−π)，π 用该格测试集阳性率（固定，不随重抽样变）
  逐患者配对付费差              Δc_i = (a_i^method − a_i^none)/D
  ΔC = mean_i Δc_i（负 = 改善）

三层 bootstrap（B=1000，固定种子）：
  L1 逐格        ：格内测试患者配对重抽样        → 每格 ΔC 的 95% CI
  L2a 设计层     ：**同一 split 内各格共用一次患者重抽样**，再在格上取均值
                   → 「这套设计下均值符号稳不稳」（主区间；不用"每格独立"是因为
                     同 split 的各 seed 评估同一批测试患者，独立重抽会把区间压窄）
  L2b 格+患者两级：先重抽格、再在格内重抽患者    → 加入格间异质性的敏感性区间
  L3 split 簇    ：以 split 为簇重抽样（P19: 5 簇；P12: 3 簇）→ 换一批切分的保守界
                   （5 簇下百分位 bootstrap 不稳定，另附 5 个 split 均值的 t 区间 df=4）

为什么必须分层：35 个 P19 格不是独立样本（5 个 split 共享 38,803 病人池，3 个 seed 共享
同一测试集）。把格当独立会把 CI 压窄——正是审稿意见 2「保留重叠再划分 / 独立队列区分」要拦的。

运行（WSL，先跑完 m5_01）：
  cd ~/CoFormer-Calib
  python m5/m5_02_paired_bootstrap.py

产物（默认 results/m5_paired/）：
  paired_cost_cells.csv    逐格 × 方法 × τ 的 ΔC、CI、翻转计数
  paired_cost_summary.json 三层区间 + 逐 split 均值
  paper_numbers.md         可直接抄进 §5.5 / §7 的数字与英文句子
"""
import os
import re
import csv
import glob
import json
import argparse

import numpy as np

W_FN = 10.0
TAUS = [0.05, 0.10, 0.20, 0.30, 0.50]
METHODS = ['temperature', 'vector', 'platt', 'isotonic']


# ---------------------------------------------------------------- 基础量
def ci95(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def excl_zero(lo, hi):
    return bool(lo > 0 or hi < 0)


def t_ci95(vals):
    """小簇数（df=4/2）下的 t 区间，作为百分位 bootstrap 的稳健替代。"""
    from scipy import stats
    v = np.asarray(vals, dtype=float)
    n = len(v)
    if n < 2:
        return float('nan'), float('nan')
    se = v.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return float(v.mean() - h), float(v.mean() + h)


def binom_p(k, n):
    """McNemar 精确二项检验（双侧）。"""
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(int(k), int(n), 0.5, alternative='two-sided').pvalue)
    except Exception:
        from scipy import stats
        return float(2 * min(stats.binom.cdf(k, n, 0.5), stats.binom.sf(k - 1, n, 0.5)))


def fmt_p(p):
    return '<0.001' if p < 1e-3 else '{:.3f}'.format(p)


def contrast_diffs(cell, a_m, b_m, w_fn=W_FN, taus=TAUS):
    """方法间配对对比：Δc = (a_am − a_bm)/D（共用同一分母，配对严格）。
    用于正面检验「排序反转」：temperature − vector < 0 即 temperature 更低代价。"""
    y = cell['y']
    pi = float(y.mean())
    D = w_fn * pi + (1.0 - pi)
    y1 = (y == 1).astype(np.float64)
    y0 = 1.0 - y1
    tau_arr = np.asarray(taus, dtype=np.float64)

    def numer(p):
        alarm = (p[:, None] >= tau_arr[None, :]).astype(np.float64)
        return (y1[:, None] * (1.0 - alarm) * w_fn + y0[:, None] * alarm).T

    return (numer(cell['p'][a_m]) - numer(cell['p'][b_m])) / D


# ---------------------------------------------------------------- 数据
def load_cells(prob_root):
    cells = []
    for f in sorted(glob.glob(os.path.join(prob_root, 'probs_*.npz'))):
        z = np.load(f, allow_pickle=True)
        tag, ds, bb, split_path, ckpt = [str(x) for x in z['meta']]
        m = re.search(r'(split|subset)(\d+)$', tag)
        cells.append(dict(
            tag=tag, dataset=ds, backbone=bb, cluster=int(m.group(2)) if m else -1,
            y=z['y'].astype(np.float64),
            p={k: z['p_' + k].astype(np.float64) for k in ['none'] + METHODS},
            npz=f,
        ))
    return cells


def per_patient_diffs(cell, w_fn=W_FN, taus=TAUS):
    """返回 (T, n) 的逐患者 Δc 矩阵：Δc[t] = (a_m[t] − a_none[t]) / D。"""
    y = cell['y']
    pi = float(y.mean())
    D = w_fn * pi + (1.0 - pi)
    y1 = (y == 1).astype(np.float64)
    y0 = 1.0 - y1
    tau_arr = np.asarray(taus, dtype=np.float64)

    def numer(p):
        alarm = (p[:, None] >= tau_arr[None, :]).astype(np.float64)   # (n, T)
        missed = y1[:, None] * (1.0 - alarm) * w_fn
        false_alarm = y0[:, None] * alarm
        return (missed + false_alarm).T                                # (T, n)

    a_none = numer(cell['p']['none'])
    out = {}
    for m in METHODS:
        out[m] = (numer(cell['p'][m]) - a_none) / D
    return out


# ---------------------------------------------------------------- bootstrap
def boot_patient_only(d, B, rng):
    """格固定，只在格内重抽患者。d: (C, T, n) 列表 → (B, C, T)。"""
    C = len(d)
    T = d[0].shape[0]
    out = np.empty((B, C, T), dtype=np.float64)
    for b in range(B):
        for c in range(C):
            n = d[c].shape[1]
            idx = rng.randint(0, n, n)
            out[b, c] = d[c][:, idx].mean(axis=1)
    return out


def boot_patient_shared(d, clusters, B, rng):
    """同一 split 内的所有格共用一组患者索引重抽样。

    为什么必须这样做：P19 一个 split 内 3 个 CoFormer seed（+2 GRU-D +2 ta-LSTM）评估的是
    同一批 3,881 名测试患者，各格 ΔC 正相关。若每格独立重抽样再平均，等于假定 35 格相互独立，
    会把区间压窄（这正是审稿意见 2 要拦的口径问题）。此函数按「患者嵌套于 split、
    seed 共享测试集」的真实结构重抽样：一个 split 抽一次患者索引，簇内所有格套用同一索引。
    注意：这里只重抽患者，不重抽 split——split 层变异由 boot_cluster 单独给（L3）。

    → (B, C, T)
    """
    T = d[0].shape[0]
    uniq = sorted(set(clusters))
    members = {k: [i for i, c in enumerate(clusters) if c == k] for k in uniq}
    out = np.empty((B, len(d), T), dtype=np.float64)
    for b in range(B):
        for k in uniq:
            sizes = {d[i].shape[1] for i in members[k]}
            if len(sizes) != 1:
                raise ValueError('split {} 内各格 n_test 不一致 {}：无法共享患者索引'.format(k, sizes))
            n = sizes.pop()
            idx = rng.randint(0, n, n)
            for i in members[k]:
                out[b, i] = d[i][:, idx].mean(axis=1)
    return out


def boot_two_stage(d, B, rng):
    """先重抽格、再在格内重抽患者；结果按被选中格数取均值。→ (B, T)"""
    C = len(d)
    T = d[0].shape[0]
    out = np.empty((B, T), dtype=np.float64)
    for b in range(B):
        sel = rng.randint(0, C, C)
        acc = np.zeros(T)
        for c in sel:
            n = d[c].shape[1]
            idx = rng.randint(0, n, n)
            acc += d[c][:, idx].mean(axis=1)
        out[b] = acc / C
    return out


def boot_cluster(d, clusters, B, rng):
    """以 cluster（split）为簇重抽；簇内所有格都进入均值。→ (B, T)"""
    T = d[0].shape[0]
    uniq = sorted(set(clusters))
    members = {k: [i for i, c in enumerate(clusters) if c == k] for k in uniq}
    out = np.empty((B, T), dtype=np.float64)
    for b in range(B):
        sel = rng.randint(0, len(uniq), len(uniq))
        acc = np.zeros(T)
        cnt = 0
        for s in sel:
            for c in members[uniq[s]]:
                n = d[c].shape[1]
                idx = rng.randint(0, n, n)
                acc += d[c][:, idx].mean(axis=1)
                cnt += 1
        out[b] = acc / max(cnt, 1)
    return out


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prob_root', default='results/m5_paired/probs')
    ap.add_argument('--out_root', default='results/m5_paired')
    ap.add_argument('--results_root', default='results',
                    help='用于读取既有 alarm_cost.json 做点估计交叉核对')
    ap.add_argument('--datasets', default='p19,p12')
    ap.add_argument('--taus', default=','.join(str(t) for t in TAUS))
    ap.add_argument('--primary_tau', type=float, default=0.20)
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--w_fn', type=float, default=W_FN)
    ns = ap.parse_args()

    taus = [float(x) for x in ns.taus.split(',') if x.strip()]
    k_primary = int(np.argmin([abs(t - ns.primary_tau) for t in taus]))
    datasets = [s.strip() for s in ns.datasets.split(',') if s.strip()]

    all_cells = load_cells(ns.prob_root)
    if not all_cells:
        print('[error] {} 下没有 probs_*.npz，先跑 m5_01_dump_probs.py'.format(ns.prob_root))
        return 2

    os.makedirs(ns.out_root, exist_ok=True)
    rows, summary = [], {}
    md = ['# 患者级配对不确定性 · 结果汇总', '',
          '- 数据源：`{}`（{} 格）'.format(ns.prob_root, len(all_cells)),
          '- B={}，种子={}，w_FN={}，τ 网格={}'.format(ns.B, ns.seed, ns.w_fn, taus), '']

    for ds in datasets:
        cells = [c for c in all_cells if c['dataset'] == ds]
        if not cells:
            continue
        d_by_method = {m: [per_patient_diffs(c, ns.w_fn, taus)[m] for c in cells]
                       for m in METHODS}
        clusters = [c['cluster'] for c in cells]

        summary[ds] = dict(n_cells=len(cells), n_test_per_cell=[len(c['y']) for c in cells],
                           clusters=sorted(set(clusters)), taus=taus,
                           primary_tau=taus[k_primary], methods={})

        for m in METHODS:
            rng = np.random.RandomState(ns.seed)
            d = d_by_method[m]
            point = np.array([x.mean(axis=1) for x in d])                  # (C, T)

            arr = boot_patient_only(d, ns.B, rng)                          # (B, C, T) 仅用于逐格 CI
            shared = boot_patient_shared(d, clusters, ns.B, np.random.RandomState(ns.seed + 5))
            two = boot_two_stage(d, ns.B, np.random.RandomState(ns.seed + 1))
            clu = boot_cluster(d, clusters, ns.B, np.random.RandomState(ns.seed + 2))

            L1_lo, L1_hi = np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)
            assert L1_lo.shape == (len(cells), len(taus)), \
                'bootstrap 数组形状异常: {} vs ({}, {})'.format(
                    L1_lo.shape, len(cells), len(taus))
            mean_b = shared.mean(axis=1)          # split 内共享患者索引 → L2a（主区间）
            # 注意：ci95 用单一数组，这里必须按主阈值取列（传 2D 会被 flatten）
            L2a = ci95(mean_b[:, k_primary])
            L2a_naive = ci95(arr.mean(axis=1)[:, k_primary])   # 仅作诊断：把格当独立的（偏窄）版本
            L2b = ci95(two[:, k_primary])
            L3 = ci95(clu[:, k_primary])

            # 逐 split 均值与 t 区间（P19 5 簇 / P12 3 簇）
            per_split = {}
            for cl in sorted(set(clusters)):
                idx = [i for i, c in enumerate(clusters) if c == cl]
                per_split[str(cl)] = float(point[idx, k_primary].mean())
            split_vals = [per_split[str(cl)] for cl in sorted(per_split)]

            # 逐格点估计 + 翻转分解（只在主阈值上算）
            flips = dict(n_flip=0, pos_fn_to_tp=0, pos_tp_to_fn=0,
                         neg_tn_to_fp=0, neg_fp_to_tn=0, n_pos=0, n_neg=0)
            for c, dcell in zip(cells, d):
                y = c['y']
                tau = taus[k_primary]
                a_none = (c['p']['none'] >= tau)
                a_m = (c['p'][m] >= tau)
                flip = a_m != a_none
                flips['n_flip'] += int(flip.sum())
                flips['n_pos'] += int((y == 1).sum())
                flips['n_neg'] += int((y == 0).sum())
                flips['pos_fn_to_tp'] += int(((y == 1) & (~a_none) & a_m).sum())
                flips['pos_tp_to_fn'] += int(((y == 1) & a_none & (~a_m)).sum())
                flips['neg_tn_to_fp'] += int(((y == 0) & (~a_none) & a_m).sum())
                flips['neg_fp_to_tn'] += int(((y == 0) & a_none & (~a_m)).sum())
            pos_n = flips['pos_fn_to_tp'] + flips['pos_tp_to_fn']

            summary[ds]['methods'][m] = dict(
                point_dC=[round(float(v), 5) for v in point.mean(axis=0)],
                L2a_by_tau={('tau={:.2f}'.format(t)): dict(
                    mean=round(float(mean_b[:, j].mean()), 5),
                    ci=[round(float(v), 5) for v in ci95(mean_b[:, j])])
                    for j, t in enumerate(taus)},
                # 逐格 CI 是否排除 0 —— 必须在主阈值列上判（早期版本误用了泄漏变量 k）
                L1_cells_excl_zero=int(sum(excl_zero(L1_lo[i, k_primary], L1_hi[i, k_primary])
                                           for i in range(len(cells)))),
                L1_cells_improving=int(sum(np.round(point[:, k_primary], 5) < 0)),
                L2a=dict(mean=round(float(mean_b[:, k_primary].mean()), 5),
                         ci=[round(v, 5) for v in L2a],
                         excl_zero=excl_zero(*L2a)),
                L2a_naive_independent_cells=dict(ci=[round(v, 5) for v in L2a_naive],
                                                 note='把 35 格当独立样本的偏窄版本，仅诊断用，不进论文'),
                L2b=dict(ci=[round(v, 5) for v in L2b], excl_zero=excl_zero(*L2b)),
                L3_cluster=dict(ci=[round(v, 5) for v in L3],
                                excl_zero=excl_zero(*L3),
                                per_split=per_split,
                                t_ci=[round(v, 6) for v in t_ci95(split_vals)]),
                flips_primary_tau=flips,
                mcnemar_pos_p=round(binom_p(flips['pos_fn_to_tp'], pos_n), 4),
            )

            for i, c in enumerate(cells):
                for j, tau in enumerate(taus):
                    rows.append(dict(
                        tag=c['tag'], dataset=ds, backbone=c['backbone'],
                        split=c['cluster'], n_test=len(c['y']), method=m, tau=tau,
                        deltaC=round(float(point[i, j]), 5),
                        ci_lo=round(float(L1_lo[i, j]), 5),
                        ci_hi=round(float(L1_hi[i, j]), 5),
                        excl_zero=excl_zero(L1_lo[i, j], L1_hi[i, j]),
                        improving=bool(round(float(point[i, j]), 5) < 0)))

            # 点估计交叉核对（既有 alarm_cost.json）
            mism = 0
            for c in cells:
                fp = os.path.join(ns.results_root, 'alarm_' + c['tag'], 'alarm_cost.json')
                if not os.path.exists(fp):
                    continue
                ref = json.load(open(fp))['fixed_tau']
                for j, tau in enumerate(taus):
                    key = 'tau={:.2f}'.format(tau)
                    if key not in ref:
                        continue
                    old = ref[key][m]['dcost']
                    if abs(old - point[cells.index(c), j]) > 1e-4:
                        mism += 1
            summary[ds]['methods'][m]['alarm_json_mismatch'] = mism

        # ---- 方法间配对对比：temperature − vector（"排序反转"的正面检验）----
        # 同一批患者、同一 τ、同一分母，差值本身的 CI 比两个方法各自的 CI 更贴合"排序"主张。
        dc = [contrast_diffs(c, 'temperature', 'vector', ns.w_fn, taus) for c in cells]
        c_point = np.array([x.mean(axis=1) for x in dc])                  # (C, T)
        c_arr = boot_patient_shared(dc, clusters, ns.B, np.random.RandomState(ns.seed + 3))
        c_clu = boot_cluster(dc, clusters, ns.B, np.random.RandomState(ns.seed + 4))
        c_mean = c_arr.mean(axis=1)
        c_L2a = ci95(c_mean[:, k_primary])
        c_L3 = ci95(c_clu[:, k_primary])
        c_split = {}
        for cl in sorted(set(clusters)):
            idx = [i for i, x in enumerate(clusters) if x == cl]
            c_split[str(cl)] = float(c_point[idx, k_primary].mean())
        summary[ds]['contrast_temp_minus_vector'] = dict(
            mean=round(float(c_mean[:, k_primary].mean()), 5),
            L2a=dict(ci=[round(v, 5) for v in c_L2a], excl_zero=excl_zero(*c_L2a)),
            L3_cluster=dict(ci=[round(v, 5) for v in c_L3], excl_zero=excl_zero(*c_L3),
                            per_split=c_split),
            cells_negative=int(sum(np.round(c_point[:, k_primary], 5) < 0)),
            n_cells=len(cells),
            note='ΔC(temperature) − ΔC(vector)；负值 = temperature 代价更低')

        # ---- markdown：主阈值的关键数字 ----
        tau = taus[k_primary]
        md += ['## {}（{} 格，τ={}）'.format(ds.upper(), len(cells), tau), '',
               '| 方法 | mean ΔC | L2a 95% CI | 逐格 CI 排除 0 | 逐格改善 | L3 簇 CI | split 均值 |',
               '|---|---|---|---|---|---|---|']
        for m in METHODS:
            s = summary[ds]['methods'][m]
            md.append('| {} | {:+.4f} | [{:+.4f}, {:+.4f}] | {}/{} | {}/{} | [{:+.4f}, {:+.4f}] | {} |'.format(
                m, s['L2a']['mean'], s['L2a']['ci'][0], s['L2a']['ci'][1],
                s['L1_cells_excl_zero'], len(cells),
                s['L1_cells_improving'], len(cells),
                s['L3_cluster']['ci'][0], s['L3_cluster']['ci'][1],
                ', '.join('{:.4f}'.format(v) for v in s['L3_cluster']['per_split'].values())))
        md += ['', '翻转分解（τ={}，全部格合计）：'.format(tau), '',
               '| 方法 | 翻转 | 阳性 FN→TP | 阳性 TP→FN | McNemar p（阳性） | 阴性 TN→FP | 阴性 FP→TN |',
               '|---|---|---|---|---|---|---|']
        for m in METHODS:
            f = summary[ds]['methods'][m]['flips_primary_tau']
            md.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
                m, f['n_flip'], f['pos_fn_to_tp'], f['pos_tp_to_fn'],
                fmt_p(summary[ds]['methods'][m]['mcnemar_pos_p']),
                f['neg_tn_to_fp'], f['neg_fp_to_tn']))
        ct = summary[ds]['contrast_temp_minus_vector']
        md += ['',
               '方法间配对对比（temperature − vector，负 = temperature 代价更低）：'
               '均值 {:+.4f}，L2a 95% CI [{:+.4f}, {:+.4f}]，L3 簇 CI [{:+.4f}, {:+.4f}]，'
               '逐格为负 {}/{}，逐 split 均值 {}。'.format(
                   ct['mean'], ct['L2a']['ci'][0], ct['L2a']['ci'][1],
                   ct['L3_cluster']['ci'][0], ct['L3_cluster']['ci'][1],
                   ct['cells_negative'], ct['n_cells'],
                   ', '.join('{:.4f}'.format(v) for v in ct['L3_cluster']['per_split'].values())),
               '']

    # ---- 可粘贴句子 ----
    if 'p19' in summary:
        s = summary['p19']['methods']
        tau = summary['p19']['primary_tau']
        ct19 = summary['p19']['contrast_temp_minus_vector']
        md += ['## 可直接替换 §5.5 草稿的英文（数值已填）', '',
               '> (a) 主句 —— §5.5：', '',
               '> At τ={:.1f} on P19, the patient-level paired bootstrap gives a mean ΔC of '
               '{:+.4f} [95% CI {:+.4f}, {:+.4f}] for temperature scaling and {:+.4f} '
               '[{:+.4f}, {:+.4f}] for vector scaling (Platt identical in the binary case); the '
               'corresponding per-cell intervals exclude zero in {}/{} and {}/{} cells.'.format(
                   tau,
                   s['temperature']['L2a']['mean'], s['temperature']['L2a']['ci'][0],
                   s['temperature']['L2a']['ci'][1],
                   s['vector']['L2a']['mean'], s['vector']['L2a']['ci'][0],
                   s['vector']['L2a']['ci'][1],
                   s['temperature']['L1_cells_excl_zero'],
                   summary['p19']['n_cells'],
                   s['vector']['L1_cells_excl_zero'],
                   summary['p19']['n_cells']), '',
               '> (b) 设计说明 —— Methods 或图注：', '',
               '> Intervals are patient-level paired bootstraps ({} resamples). Within a split '
               'the same patient resample is shared by every cell, because those cells are '
               'evaluated on the same test patients, and the same resample is applied to the '
               'uncalibrated and calibrated predictions of a cell.'.format('{:,}'.format(ns.B)), '',
               '> (c) split 层敏感性 —— §5.5 或 §7：', '',
               '> As a cluster-level sensitivity bound, resampling the five P19 splits gives '
               '[{:+.4f}, {:+.4f}] for temperature and [{:+.4f}, {:+.4f}] for vector; '
               'the five split means are {}.'.format(
                   s['temperature']['L3_cluster']['ci'][0],
                   s['temperature']['L3_cluster']['ci'][1],
                   s['vector']['L3_cluster']['ci'][0],
                   s['vector']['L3_cluster']['ci'][1],
                   ', '.join('{:.4f}'.format(v) for v in
                             s['temperature']['L3_cluster']['per_split'].values())), '',
               '> (d) 机制 —— §5.5：', '',
               '> The mechanism is a shift of patients across the threshold: temperature '
               'scaling moves {} positive patients from missed to alarmed and {} in the reverse '
               'direction (exact McNemar p={}), while vector scaling moves {} and {} '
               '(p={}).'.format(
                   s['temperature']['flips_primary_tau']['pos_fn_to_tp'],
                   s['temperature']['flips_primary_tau']['pos_tp_to_fn'],
                   fmt_p(s['temperature']['mcnemar_pos_p']),
                   s['vector']['flips_primary_tau']['pos_fn_to_tp'],
                   s['vector']['flips_primary_tau']['pos_tp_to_fn'],
                   fmt_p(s['vector']['mcnemar_pos_p'])), '',
               '> (e) 方法间配对对比 —— §5.5（"排序反转"的正面检验）：', '',
               '> Because both methods are evaluated on the same patients, the between-method '
               'contrast is itself paired: at τ={:.1f} the mean ΔC of temperature scaling minus '
               'that of vector scaling is {:+.4f} [{:+.4f}, {:+.4f}] (split-cluster bound '
               '[{:+.4f}, {:+.4f}]), so the ranking inversion does not depend on reading the two '
               'curves separately.'.format(
                   tau, ct19['mean'], ct19['L2a']['ci'][0], ct19['L2a']['ci'][1],
                   ct19['L3_cluster']['ci'][0], ct19['L3_cluster']['ci'][1]), '']

    # ---- 风险提示 ----
    warn = []
    for ds in summary:
        if 'p19' == ds:
            s = summary[ds]['methods']
            for m in ('temperature', 'vector'):
                if not s[m]['L2a']['excl_zero']:
                    warn.append('{}: {} 的 L2a 均值区间跨 0 —— §5.5 主张需收窄为"方向一致但区间含 0"。'.format(ds, m))
                if not s[m]['L3_cluster']['excl_zero']:
                    warn.append('{}: {} 的 L3 簇区间跨 0 —— 这是预期的保守界；按 §9.5 的写法披露，不要藏。'.format(ds, m))
            c = summary[ds]['contrast_temp_minus_vector']
            if not c['L2a']['excl_zero']:
                warn.append('{}: temperature−vector 配对对比的 L2a 跨 0 —— "排序反转"只剩点估计，§5.5 必须收窄。'.format(ds))
            if not c['L3_cluster']['excl_zero']:
                warn.append('{}: temperature−vector 配对对比的 L3 跨 0（逐 split 均值 {}）—— 反转在 split 层不稳，须如实披露。'.format(
                    ds, ', '.join('{:.4f}'.format(v) for v in c['L3_cluster']['per_split'].values())))
    if warn:
        md += ['## ⚠ 需要改写的风险项', ''] + ['- ' + w for w in warn] + ['']

    # ---- 落盘 ----
    csv_path = os.path.join(ns.out_root, 'paired_cost_cells.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(ns.out_root, 'paired_cost_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(ns.out_root, 'paper_numbers.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')

    print('\n'.join(md))
    print('saved:', csv_path)
    print('saved:', os.path.join(ns.out_root, 'paired_cost_summary.json'))
    print('saved:', os.path.join(ns.out_root, 'paper_numbers.md'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
