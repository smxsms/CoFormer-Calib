#!/usr/bin/env bash
# ===== M5：患者级配对不确定性（审稿意见 2）一键执行 =====
# 位置：~/CoFormer-Calib/m5/   （工程在 WSL，数据在 ~/data）
#
# 用法：
#   bash ~/CoFormer-Calib/m5/run_m5.sh          # P19 35 格 + P12 9 格（44 格，约 70 分钟）
#   bash ~/CoFormer-Calib/m5/run_m5.sh p19      # 只跑 P19 35 格（论文主比较，约 55 分钟）
#
# 三段：
#   1) m5_01 重跑前向 + 重拟合校准器 + 落盘逐样本概率 + 与既有 JSON 回归核对
#   2) m5_02 患者级配对 bootstrap（三层）+ 决策翻转分解 + 生成可粘贴数字
#   3) m5_03 给论文 Figure 2 叠加 95% 置信带（新文件名，不覆盖原图）
#
# 幂等：m5_01 已存在的 npz 自动跳过（中断可直接重跑续算）；加 --overwrite 才重算。
set -eo pipefail          # 注意：不能用 -u——conda activate 的 dgl_libs.sh 会读未定义变量

MODE="${1:-all}"

for p in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
  if [ -f "$p" ]; then source "$p"; break; fi
done
conda activate coformer         # 必须：torch 2.1.2+cu121 / sklearn / scipy 都在这个 env

cd "$HOME/CoFormer-Calib"       # 必须：脚本靠 cwd import 既有 calibrate_suite.py

if [ "$MODE" = "p19" ]; then DS="p19"; else DS="p19,p12"; fi

echo "===== [1/3] 概率导出（数据集：$DS）====="
python m5/m5_01_dump_probs.py --datasets "$DS"

echo
echo "===== [2/3] 配对 bootstrap ====="
python m5/m5_02_paired_bootstrap.py --datasets "$DS"

echo
echo "===== [3/3] Figure 2 置信带 ====="
python m5/m5_03_plot_ci.py

echo
echo "全部完成。看这三个文件："
echo "  ~/CoFormer-Calib/results/m5_paired/regression_report.md   概率与既有产物的逐字段核对（先看这个）"
echo "  ~/CoFormer-Calib/results/m5_paired/paper_numbers.md       可直接抄进 §5.5 的数字与英文句子"
echo "  <out 目录>/fig4_alarm_cost_ci.pdf"
