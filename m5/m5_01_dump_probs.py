#!/usr/bin/env python3
"""m5_01_dump_probs.py —— 为「患者级配对不确定性」重跑前向、重拟合校准器、落盘逐样本概率。

背景：全仓库没有任何逐样本概率（calibration_suite.json / alarm_cost.json 都只有聚合量），
所以配对 bootstrap 必须先重跑 44 个二分类 ckpt 的 val/test 前向。本脚本：
  1. 直接 import 既有 calibrate_suite.py 的 build_model / collect_logits / fit_binary /
     binary_metrics —— 概率与既有产物同源，不重写校准实现（避免口径分叉）；
  2. 每个格落盘 probs_<tag>.npz：y、test 索引、p_none、p_temperature、p_vector、
     p_platt、p_isotonic（float64，长度 = n_test，顺序 = split 文件 test 部分顺序）；
  3. 逐格回归核对：用新算概率复算 AUROC/AUPRC/ECE/Brier/NLL/classwise-ECE/拟合参数，
     与既有 calibration_suite.json 逐字段比对，输出 regression_report.{csv,md}。
     期望：全部逐位一致（同一实现、同一 ckpt、同一 split）；任何偏差都要先查清再往下走。

运行（WSL）：
  cd ~/CoFormer-Calib
  python m5/m5_01_dump_probs.py --datasets p19,p12

先试跑 2 个格（约 1-2 分钟）：
  python m5/m5_01_dump_probs.py --limit 2

产物（默认 ~/CoFormer-Calib/results/m5_paired/）：
  probs/probs_<tag>.npz        逐格概率
  probs/index.csv              格清单（tag/dataset/backbone/split/n_test）
  regression_report.csv / .md  与既有 JSON 的逐字段核对
"""
import os
import sys
import re
import json
import glob
import time
import argparse
from types import SimpleNamespace

import numpy as np

# ---- 依赖既有工程（必须 cd ~/CoFormer-Calib 后运行）----
sys.path.insert(0, os.getcwd())
import torch
from torch.utils.data import DataLoader

from data.dataloader import (medical_dataloader, medicalp12_dataloader,
                             seq_collate_irregular)
from utils.utils import prepare_seed
from sklearn.metrics import roc_auc_score, average_precision_score

from calibrate_suite import (build_model, collect_logits, fit_binary,
                             binary_metrics)

torch.backends.cudnn.deterministic = True

METHODS = ['temperature', 'vector', 'platt', 'isotonic']
DATA_ROOTS = {
    'p19': '~/data/P19/processed_data/',
    'p12': '~/data/P12/processed_data/',
}
DEFAULTS = {  # 与 calibrate_suite.py / alarm_cost.py 完全一致
    'p19': dict(num_agents=34, static_dim=6, output_dim=2),
    'p12': dict(num_agents=36, static_dim=9, output_dim=2),
}


def parse_split_id(tag):
    m = re.search(r'(split|subset)(\d+)$', tag)
    return int(m.group(2)) if m else -1


def list_cells(data_root, dup_tags, datasets):
    """从既有 cal2_*/calibration_suite.json 反查 ckpt 与 split，保持与产物同源。"""
    cells = []
    for f in sorted(glob.glob(os.path.join(data_root, 'cal2_*', 'calibration_suite.json'))):
        tag = os.path.basename(os.path.dirname(f))[len('cal2_'):]
        if tag in dup_tags:
            continue
        d = json.load(open(f))
        if d['dataset'] not in datasets:
            continue
        cells.append(dict(tag=tag, dataset=d['dataset'], backbone=d['backbone'],
                          ckpt=d['ckpt'], split_path=d['split_path'], ref_json=f))
    return cells


def make_args(dataset, backbone, ns):
    a = SimpleNamespace()
    a.dataset = dataset
    a.backbone = backbone
    a.num_layers = ns.num_layers
    a.d_model = ns.d_model
    a.d_ff = ns.d_ff
    a.heads = ns.heads
    a.num_neighbors = ns.num_neighbors
    a.num_agents = DEFAULTS[dataset]['num_agents']
    a.static_dim = DEFAULTS[dataset]['static_dim']
    a.output_dim = DEFAULTS[dataset]['output_dim']
    return a


def build_loaders(dataset, data_root_dir, split_path, batch_size, workers):
    loader_cls = medical_dataloader if dataset == 'p19' else medicalp12_dataloader
    collate = seq_collate_irregular
    val = DataLoader(loader_cls(data_root_dir, split_path, part=1),
                     batch_size=batch_size, num_workers=workers, collate_fn=collate)
    test = DataLoader(loader_cls(data_root_dir, split_path, part=2),
                      batch_size=batch_size, num_workers=workers, collate_fn=collate)
    return val, test


def run_cell(cell, ns, device):
    tag = cell['tag']
    a = make_args(cell['dataset'], cell['backbone'], ns)
    prepare_seed(1)

    val_loader, test_loader = build_loaders(
        cell['dataset'], cell['data_root'], cell['split_path'],
        ns.batch_size, ns.workers)

    model = build_model(a)
    ck = torch.load(cell['ckpt'], map_location='cpu')
    sd = ck['model_dict'] if isinstance(ck, dict) and 'model_dict' in ck else ck
    model.load_state_dict(sd)
    model.to(device)

    logits_val, gt_val = collect_logits(model, val_loader, device, multiclass=False)
    logits_test, gt_test = collect_logits(model, test_loader, device, multiclass=False)

    z_val = logits_val[:, 1]
    y_val = gt_val.view(-1).float()
    z_test = logits_test[:, 1]
    y_test = gt_test.view(-1).float()

    p = {'none': torch.sigmoid(z_test).numpy().astype(np.float64)}
    params = {}
    for m in METHODS:
        apply_f, prm = fit_binary(z_val, y_val, m)
        p[m] = np.asarray(apply_f(z_test), dtype=np.float64)
        params[m] = prm

    y_np = y_test.numpy().astype(np.int8)
    split = np.load(cell['split_path'], allow_pickle=True)
    test_index = np.asarray(split[2]).astype(np.int64)

    out = dict(y=y_np, test_index=test_index,
               p_none=p['none'],
               p_temperature=p['temperature'], p_vector=p['vector'],
               p_platt=p['platt'], p_isotonic=p['isotonic'],
               meta=np.array([tag, cell['dataset'], cell['backbone'],
                              cell['split_path'], cell['ckpt']], dtype=object),
               params=np.array([json.dumps(params)], dtype=object))
    return out, y_np, p, params


def check_cell(cell, y_np, p, params):
    """与既有 calibration_suite.json 逐字段比对。"""
    ref = json.load(open(cell['ref_json']))
    rows = []

    def add(field, new, old, tol, note=''):
        if new is None or old is None:
            rows.append(dict(tag=cell['tag'], field=field, new=new, old=old,
                             absdiff=None, tol=tol, status='SKIP', note=note))
            return
        ad = abs(float(new) - float(old))
        rows.append(dict(tag=cell['tag'], field=field, new=round(float(new), 6),
                         old=round(float(old), 6), absdiff=round(ad, 9), tol=tol,
                         status='OK' if ad <= tol else 'DIFF', note=note))

    y_bin = y_np.astype(np.float64)
    add('pos_rate_test', y_bin.mean(), ref.get('pos_rate_test'), 1e-6)
    add('auroc', roc_auc_score(y_np, p['none']) * 100, ref.get('auroc'), 1e-3)
    add('auprc', average_precision_score(y_np, p['none']) * 100, ref.get('auprc'), 1e-3)
    add('n_test', len(y_np), ref.get('n_test'), 0)

    for m in METHODS:
        met_new = binary_metrics(p[m], y_bin)
        met_old = ref['methods'].get(m, {})
        for k in ('ece', 'brier', 'nll', 'classwise_ece'):
            add('{}.{}'.format(m, k), met_new[k], met_old.get(k), 2e-6)
        for k, v in params[m].items():
            if isinstance(v, (int, float)):
                add('{}.param.{}'.format(m, k), v, met_old.get('params', {}).get(k), 1e-5)
        # 预注册原料
        base_ece = ref['methods']['none']['ece']
        add('{}.ece_rel_drop_pct'.format(m),
            (base_ece - met_new['ece']) / base_ece * 100,
            met_old.get('ece_rel_drop_pct'), 1e-2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='results',
                    help='CoFormer-Calib/results（读 cal2_*/ 与 master/）')
    ap.add_argument('--master_csv', default='results/master/master_table_v2_20261031.csv')
    ap.add_argument('--out_root', default='results/m5_paired')
    ap.add_argument('--datasets', default='p19,p12')
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 个格（冒烟测试）')
    ap.add_argument('--tags', default='', help='只跑这些 tag（逗号分隔）')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_layers', type=int, default=8)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--d_ff', type=int, default=256)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--num_neighbors', type=int, default=30)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--dry_run', action='store_true')
    ns = ap.parse_args()

    datasets = [s.strip() for s in ns.datasets.split(',') if s.strip()]

    # 重复格（5 个 bit-for-bit 复现 run）不进入任何统计
    dup = set('trans_medical_p19_split%d' % i for i in range(1, 6))

    cells = list_cells(ns.data_root, dup, datasets)
    if ns.tags:
        keep = {t.strip() for t in ns.tags.split(',') if t.strip()}
        cells = [c for c in cells if c['tag'] in keep]
    if ns.limit:
        cells = cells[:ns.limit]
    if not cells:
        print('[error] 没有匹配的格，检查 --data_root / --datasets / --tags')
        return 2

    for c in cells:
        c['data_root'] = os.path.expanduser(DATA_ROOTS[c['dataset']])
        if not os.path.isabs(c['ckpt']):
            c['ckpt'] = os.path.join(os.getcwd(), c['ckpt'])
        if not os.path.exists(c['ckpt']):
            print('[warn] ckpt 缺失:', c['ckpt'])

    print('cells to run: {}, by dataset: {}'.format(
        len(cells), {d: sum(1 for c in cells if c['dataset'] == d) for d in datasets}))
    for c in cells:
        print('  -', c['tag'], c['dataset'], c['backbone'])
    if ns.dry_run:
        return 0

    prob_dir = os.path.join(ns.out_root, 'probs')
    os.makedirs(prob_dir, exist_ok=True)
    device = torch.device('cuda', ns.gpu)

    rows, index_rows = [], []
    t0 = time.time()
    for i, c in enumerate(cells, 1):
        dst = os.path.join(prob_dir, 'probs_{}.npz'.format(c['tag']))
        if os.path.exists(dst) and not ns.overwrite:
            print('[{}/{}] skip（已存在） {}'.format(i, len(cells), c['tag']))
            continue
        t1 = time.time()
        out, y_np, p, params = run_cell(c, ns, device)
        np.savez(dst, **out)
        rows.extend(check_cell(c, y_np, p, params))
        index_rows.append(dict(tag=c['tag'], dataset=c['dataset'], backbone=c['backbone'],
                               split=parse_split_id(c['tag']), n_test=len(y_np),
                               pos_rate=round(float(y_np.mean()), 6), npz=dst))
        bad = [r for r in rows if r['tag'] == c['tag'] and r['status'] == 'DIFF']
        print('[{}/{}] {}  ({:.1f}s)  回归核对: {}'.format(
            i, len(cells), c['tag'], time.time() - t1,
            '全部一致' if not bad else '{} 处偏差'.format(len(bad))))
        for b in bad[:6]:
            print('    DIFF {}: new={} old={} |d|={}'.format(
                b['field'], b['new'], b['old'], b['absdiff']))

    # ---- 续跑/重跑：用既有 npz 补齐核对与清单（否则"全跳过"的一轮会把报告写成 0 格）----
    ran = {r['tag'] for r in index_rows}
    index_rows = []
    for c in cells:
        dst = os.path.join(prob_dir, 'probs_{}.npz'.format(c['tag']))
        if not os.path.exists(dst):
            print('[warn] 缺 npz，跳过：{}'.format(c['tag']))
            continue
        z = np.load(dst, allow_pickle=True)
        y_np = z['y']
        if c['tag'] not in ran:
            p = {k: z['p_' + k] for k in ['none'] + METHODS}
            params = json.loads(str(z['params'][0]))
            rows.extend(check_cell(c, y_np, p, params))
            print('[check] 复用既有 npz 核对 {}'.format(c['tag']))
        index_rows.append(dict(tag=c['tag'], dataset=c['dataset'], backbone=c['backbone'],
                               split=parse_split_id(c['tag']), n_test=len(y_np),
                               pos_rate=round(float(y_np.mean()), 6), npz=dst))

    # ---- 报告 ----
    import csv
    csv_path = os.path.join(ns.out_root, 'regression_report.csv')
    os.makedirs(ns.out_root, exist_ok=True)
    if rows:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    idx_path = os.path.join(prob_dir, 'index.csv')
    if index_rows:
        with open(idx_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            w.writeheader()
            w.writerows(index_rows)

    n_diff = sum(1 for r in rows if r['status'] == 'DIFF')
    n_ok = sum(1 for r in rows if r['status'] == 'OK')
    worst = sorted([r for r in rows if r['absdiff'] is not None],
                   key=lambda r: r['absdiff'], reverse=True)[:5]
    md = ['# 概率重算 · 回归核对报告', '',
          '- 格数：{}'.format(len(index_rows)),
          '- 字段核对：OK {} / DIFF {} / SKIP {}'.format(
              n_ok, n_diff, sum(1 for r in rows if r['status'] == 'SKIP')),
          '- 本轮新算 {} 格（其余复用既有 npz），本轮耗时 {:.1f} 分钟'.format(
              len(ran), (time.time() - t0) / 60), '',
          '## 最大偏差 Top5', '',
          '| tag | 字段 | 新值 | 旧值 | 绝对差 | 容差 |', '|---|---|---|---|---|---|']
    for r in worst:
        md.append('| {} | {} | {} | {} | {} | {} |'.format(
            r['tag'], r['field'], r['new'], r['old'], r['absdiff'], r['tol']))
    if n_diff:
        md += ['', '## 偏差明细（前 40 条）', '',
               '| tag | 字段 | 新值 | 旧值 | 绝对差 |', '|---|---|---|---|---|']
        for r in [r for r in rows if r['status'] == 'DIFF'][:40]:
            md.append('| {} | {} | {} | {} | {} |'.format(
                r['tag'], r['field'], r['new'], r['old'], r['absdiff']))
        md += ['', '> 有偏差时**不要**直接进入下一步：先确认 ckpt/split/库版本是否与既有产物同源。']
    else:
        md += ['', '**结论：全部字段一致，概率可视为与既有 calibration_suite.json 同源。**']
    with open(os.path.join(ns.out_root, 'regression_report.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')

    print('\n===== 完成 =====')
    print('概率目录 :', prob_dir)
    print('回归报告 :', os.path.join(ns.out_root, 'regression_report.md'))
    print('OK/DIFF  : {}/{}'.format(n_ok, n_diff))
    print('耗时     : {:.1f} 分钟'.format((time.time() - t0) / 60))
    return 0 if n_diff == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
