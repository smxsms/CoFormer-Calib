"""主表拼装：聚合全部 cal2 校准产物 + alarm 告警代价产物 -> master_table.csv + 汇总 markdown。

用法（~/CoFormer-Calib 下）：python build_master_table.py [--out results/master]
"""
import json
import glob
import os
import sys
import numpy as np

OUT_DIR = 'results/master'


def verdict_temp(d):
    t = d['methods']['temperature']
    ci = t['params'].get('T_ci95', [1, 1])
    if not (ci[0] > 1 or ci[1] < 1):
        return 'FAIL_primary'
    if t['ece_rel_drop_pct'] < 10:
        return 'FAIL_aux_ece'
    if not t['brier_improved']:
        return 'FAIL_aux_brier'
    return 'GAIN'


def verdict_other(met):
    if met['ece_rel_drop_pct'] >= 10 and met['brier_improved']:
        return 'GAIN'
    return 'FAIL_aux'


def cell_meta(tag):
    if 'grud' in tag:
        bb = 'grud'
    elif 'talstm' in tag:
        bb = 'talstm'
    else:
        bb = 'coformer'
    if 'p19' in tag:
        ds = 'p19'
    elif 'p12' in tag:
        ds = 'p12'
    else:
        ds = 'pam'
    seed = 's1'
    for s in ['s1', 's2', 's3']:
        if '_{}_'.format(s) in '_' + tag + '_':
            seed = s
    return bb, ds, seed


def main():
    out_dir = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for f in sorted(glob.glob('results/cal2_*/calibration_suite.json')):
        tag = f.split('/')[1].replace('cal2_', '')
        d = json.load(open(f))
        bb, ds, seed = cell_meta(tag)
        base = d['methods']['none']
        alarm = None
        af = 'results/alarm_{}/alarm_cost.json'.format(tag)
        if os.path.exists(af):
            alarm = json.load(open(af))
        for m in ['temperature', 'vector', 'platt', 'isotonic']:
            met = d['methods'][m]
            v = verdict_temp(d) if m == 'temperature' else verdict_other(met)
            row = {
                'tag': tag, 'backbone': bb, 'dataset': ds, 'seed': seed,
                'method': m, 'verdict': v,
                'T': met['params'].get('T', ''),
                'T_ci95': met['params'].get('T_ci95', ''),
                'ece_before': base['ece'], 'ece_after': met['ece'],
                'ece_rel_drop_pct': met['ece_rel_drop_pct'],
                'cwECE_before': base['classwise_ece'], 'cwECE_after': met['classwise_ece'],
                'brier_before': base['brier'], 'brier_after': met['brier'],
                'nll_before': base['nll'], 'nll_after': met['nll'],
                'auroc': d.get('auroc', d.get('auroc_macro_ovr', '')),
                'auprc': d.get('auprc', d.get('auprc_macro_ovr', '')),
            }
            if m == 'temperature':
                ci = met['params'].get('T_ci95', [1, 1])
                row['T_ci_excludes_1'] = bool(ci[0] > 1 or ci[1] < 1)
            if alarm is not None:
                fx = alarm['fixed_tau'].get('tau=0.20', {})
                tr = alarm['transfer'].get(m, {})
                row['alarm_dcost_tau0.2'] = fx.get(m, {}).get('dcost', '')
                row['alarm_fliprate_tau0.2'] = fx.get(m, {}).get('flip_rate', '')
                row['transfer_sens_drift'] = tr.get('sens_drift', '')
            rows.append(row)

    # ---- CSV ----
    keys = ['tag', 'backbone', 'dataset', 'seed', 'method', 'verdict', 'T', 'T_ci95',
            'T_ci_excludes_1', 'ece_before', 'ece_after', 'ece_rel_drop_pct',
            'cwECE_before', 'cwECE_after', 'brier_before', 'brier_after',
            'nll_before', 'nll_after', 'auroc', 'auprc',
            'alarm_dcost_tau0.2', 'alarm_fliprate_tau0.2', 'transfer_sens_drift']
    csv_path = os.path.join(out_dir, 'master_table.csv')
    with open(csv_path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
    print('master_table.csv: {} 行（{} 格 × 4 方法 + none 已并入 before 列）'.format(len(rows), len(rows) // 4))

    # ---- 汇总 markdown ----
    md = ['# 主表汇总（{} 格 × 4 方法）\n'.format(len(rows) // 4)]
    md.append('\n## 判定计数（按 骨干 × 数据集，逐格统计）\n')
    md.append('| backbone | dataset | n | temp | vector | platt | isotonic | T 均值±sd |\n|---|---|---|---|---|---|---|---|\n')
    agg = {}
    for r in rows:
        key = (r['backbone'], r['dataset'])
        if key not in agg:
            agg[key] = {'tags': set(), 'temperature': [], 'vector': [], 'platt': [],
                        'isotonic': [], 'Ts': []}
        agg[key]['tags'].add(r['tag'])
        agg[key][r['method']].append(r['verdict'])
        if r['method'] == 'temperature' and r['T'] != '':
            agg[key]['Ts'].append(float(r['T']))
    for (bb, ds), v in sorted(agg.items()):
        n = len(v['tags'])
        md.append('| {} | {} | {} | {} | {} | {} | {} | {:.3f}±{:.3f} |\n'.format(
            bb, ds, n,
            '{}/{}'.format(v['temperature'].count('GAIN'), len(v['temperature'])),
            '{}/{}'.format(v['vector'].count('GAIN'), len(v['vector'])),
            '{}/{}'.format(v['platt'].count('GAIN'), len(v['platt'])),
            '{}/{}'.format(v['isotonic'].count('GAIN'), len(v['isotonic'])),
            np.mean(v['Ts']), np.std(v['Ts'], ddof=1) if len(v['Ts']) > 1 else 0.0))

    # ---- 告警代价汇总（仅二分类格）----
    alarm_rows = [r for r in rows if r.get('alarm_dcost_tau0.2', '') != '']
    if alarm_rows:
        md.append('\n## 告警代价（τ=0.2 固定阈值，dcost<0 = 校准后代价更低；w_fn=10）\n')
        md.append('| backbone | dataset | method | mean dcost | mean flip_rate | cost改善格/总数 | mean |sens_drift| (迁移) |\n|---|---|---|---|---|---|---|\n')
        ag = {}
        for r in alarm_rows:
            ag.setdefault((r['backbone'], r['dataset'], r['method']), []).append(r)
        for (bb, ds, m), rs in sorted(ag.items()):
            dc = [float(r['alarm_dcost_tau0.2']) for r in rs]
            fr = [float(r['alarm_fliprate_tau0.2']) for r in rs]
            sd = [abs(float(r['transfer_sens_drift'])) for r in rs]
            md.append('| {} | {} | {} | {:+.4f} | {:.3f} | {}/{} | {:.3f} |\n'.format(
                bb, ds, m, np.mean(dc), np.mean(fr),
                sum(1 for x in dc if x < 0), len(dc), np.mean(sd)))
    md_path = os.path.join(out_dir, 'master_summary.md')
    with open(md_path, 'w') as f:
        f.write(''.join(md))
    print('master_summary.md 已写入', md_path)
    print(''.join(md))


if __name__ == '__main__':
    main()
