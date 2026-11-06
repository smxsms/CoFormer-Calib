#!/usr/bin/env python3
"""m5_03_plot_ci.py —— 给论文 Figure 2（脚本名 fig4_alarm_cost）叠加配对 bootstrap 置信带。

不做新计算：曲线仍来自既有 alarm_*/alarm_cost.json（与 make_paper2_figures.py 完全同源），
只把 m5_02 产出的 L2a 95% CI 作为半透明带画在 P19 / P12 两个面板上。
输出新文件名 fig4_alarm_cost_ci.{pdf,png}，不覆盖原图——可直接对比取舍。

运行（WSL，先跑完 m5_02）：
  cd ~/CoFormer-Calib
  python m5/m5_03_plot_ci.py

默认输出到 ./figures/。复用论文图模块（make_paper2_figures.py）时，用 --figmod_dir 或环境变量 FIGMOD_DIR 指定其所在目录。
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGMOD_DIR = os.environ.get('FIGMOD_DIR', '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default=os.path.expanduser('~/CoFormer-Calib/results'))
    ap.add_argument('--ci_json', default=os.path.expanduser(
        '~/CoFormer-Calib/results/m5_paired/paired_cost_summary.json'))
    ap.add_argument('--out', default=os.path.join(FIGMOD_DIR, 'figures'))
    ap.add_argument('--figmod_dir', default=FIGMOD_DIR)
    ap.add_argument('--bands', default='temperature,vector',
                    help='给哪些方法画置信带（vector 与 Platt 在二分类数值等价）')
    ns = ap.parse_args()

    sys.path.insert(0, ns.figmod_dir)
    import make_paper2_figures as mpf   # 复用其数据加载、配色与判定口径

    cal, alarm = mpf.load_all(ns.data_root)
    ci = json.load(open(ns.ci_json))
    bands = [m.strip() for m in ns.bands.split(',') if m.strip()]
    print('loaded {} calibration cells, {} alarm cells'.format(len(cal), len(alarm)))

    taus = [0.05, 0.10, 0.20, 0.30, 0.50]
    tau_keys = ['tau={:.2f}'.format(t) for t in taus]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for pi, ds in enumerate(['p19', 'p12']):
        ax = axes[pi]
        cells = [d for t, d in alarm.items() if d['dataset'] == ds]
        for m in mpf.METHODS:
            dc = [np.mean([c['fixed_tau'][k][m]['dcost'] for c in cells]) for k in tau_keys]
            color = mpf.METHOD_COLORS[m]
            ax.plot(taus, dc, '-o', color=color, lw=1.0, ms=3, label=m)
            if m in bands and ds in ci:
                by_tau = ci[ds]['methods'][m]['L2a_by_tau']
                lo = [by_tau[k]['ci'][0] for k in tau_keys]
                hi = [by_tau[k]['ci'][1] for k in tau_keys]
                ax.fill_between(taus, lo, hi, color=color, alpha=0.13, linewidth=0)
        ax.axhline(0, color='#555555', lw=0.6, ls='--')
        ax.set_xlabel(r'Alarm threshold $\tau$')
        if pi == 0:
            ax.set_ylabel(r'Mean $\Delta$ cost')
            ax.set_title('P19 (sepsis, 35 cells)', fontsize=8)
        else:
            ax.set_title('P12 (LOS>3d, 9 cells)', fontsize=8)
        ax.set_xticks([0.05, 0.2, 0.5])
        ax.set_xticklabels(['0.05', '0.2', '0.5'])
        if ds == 'p19':
            ax.set_ylim(-0.010, 0.020)
    # P19 面板原来用 1e-3 科学计数法偏移，会与标题挤在一起；改成直写三位小数
    from matplotlib.ticker import FuncFormatter
    axes[0].set_yticks([-0.010, -0.005, 0.0, 0.005, 0.010, 0.015, 0.020])
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: '{:.3f}'.format(v)))
    axes[1].legend(frameon=False, fontsize=6, loc='lower left', ncol=2,
                   handlelength=1.2, columnspacing=0.8, labelspacing=0.25)

    # Panel C：与 make_paper2_figures.fig4 完全相同的排序反转柱状图
    ax = axes[2]
    p19_tags = [t for t in cal if mpf.cell_meta(t)[1] == 'p19']
    p19_alarm = [t for t in alarm if alarm[t]['dataset'] == 'p19']
    gain_frac, cost_frac = [], []
    for m in mpf.METHODS:
        vs = [mpf.verdict(cal[t]['methods'][m], m == 'temperature') for t in p19_tags]
        gain_frac.append(vs.count('GAIN') / len(vs))
        dc = [alarm[t]['fixed_tau']['tau=0.20'][m]['dcost'] < 0 for t in p19_alarm]
        cost_frac.append(sum(dc) / len(dc))
    x = np.arange(len(mpf.METHODS))
    ax.bar(x - 0.19, gain_frac, 0.36, color='#888888', label='Calibration verdict (GAIN)')
    ax.bar(x + 0.19, cost_frac, 0.36, color=mpf.C_TEMP,
           label=r'Alarm cost improved ($\tau=0.2$)')
    for xi, (g, c) in enumerate(zip(gain_frac, cost_frac)):
        ax.text(xi - 0.19, g + 0.02, '{:.2f}'.format(g), ha='center', fontsize=6)
        ax.text(xi + 0.19, c + 0.02, '{:.2f}'.format(c), ha='center', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(['Temp', 'Vec', 'Platt', 'Iso'], fontsize=7)
    ax.set_ylabel('Fraction of cells (P19)')
    ax.set_ylim(0, 1.42)
    ax.legend(frameon=False, fontsize=6, loc='upper right')
    ax.set_title('Metric view vs decision view', fontsize=8)

    fig.tight_layout()
    os.makedirs(ns.out, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(ns.out, 'fig4_alarm_cost_ci.{}'.format(ext)))
    plt.close(fig)
    print('saved:', os.path.join(ns.out, 'fig4_alarm_cost_ci.pdf'))
    print('saved:', os.path.join(ns.out, 'fig4_alarm_cost_ci.png'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
