#!/bin/bash
# =====  第③步(a)：对全部已有 CoFormer ckpt 补跑校准套件（零训练）=====
# 生成日期: 2026-10-26 | 执行环境: WSL, conda env coformer, 工程目录 ~/CoFormer-Calib
# 覆盖 28 个 ckpt × 4 方法（温度/向量/Platt/等渗）+ classwise 指标 + 预注册判定原料：
#   e0_base_s{1,2,3}_p19_split{1..5}   15 个（P19 跨 seed，创新点③）
#   trans_medical_p19_split{1..5}        5 个（trans_medical 基线）
#   trans_medical_p12_subset{1..3}       3 个（P12 首批校准）
#   trans_medical_pam_split{1..5}        5 个（PAM 首批校准，多分类）
# 幂等: 已存在的 calibration_suite.json 自动跳过，中断重跑即续。
# 结束后自动把汇总表追加到 experiment_log.md。
# 预计耗时: 每 ckpt 数分钟（含 1000 次 bootstrap T CI），总计约 1-2 小时。
# 注意: 新骨干 ckpt 到位后重跑本脚本不会覆盖——本脚本只扫固定清单；新骨干用 step4 内置校准。
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate coformer
cd ~/CoFormer-Calib
mkdir -p ~/coformer_calib_logs

DS_P19="p19 $HOME/data/P19/processed_data/"
DS_P12="p12 $HOME/data/P12/processed_data/"
DS_PAM="pam $HOME/data/PAM/processed_data/"

run_cal () {  # <tag> <dataset> <data_root> <split_path>
    local tag=$1 ds=$2 root=$3 sp=$4
    local out=cal2_${tag}
    if [ ! -f "results/${tag}/models/model_0020.p" ]; then
        echo "[miss] results/${tag}/models/model_0020.p 不存在，跳过"
        return 0
    fi
    if [ -f "results/${out}/calibration_suite.json" ]; then
        echo "[skip] results/${out} 已存在"
        return 0
    fi
    echo "===== [cal] ${out} ====="
    mkdir -p "results/${out}"
    if python calibrate_suite.py --dataset "$ds" --backbone coformer \
        --ckpt "results/${tag}/models/model_0020.p" \
        --data_root "$root" --split_path "$sp" \
        --out "results/${out}/calibration_suite.json" \
        2>&1 | tee ~/coformer_calib_logs/${out}.log; then
        echo "[done] ${out}"
    else
        echo "[FAILED] ${out}（详见 ~/coformer_calib_logs/${out}.log）"
    fi
}

# ---- P19: e0_base 跨 seed ×3 ----
for s in 1 2 3; do
    for i in 1 2 3 4 5; do
        run_cal "e0_base_s${s}_p19_split${i}" p19 "$HOME/data/P19/processed_data/" \
            "$HOME/data/P19/splits/phy19_split${i}_new.npy"
    done
done

# ---- P19: trans_medical 基线 ----
for i in 1 2 3 4 5; do
    run_cal "trans_medical_p19_split${i}" p19 "$HOME/data/P19/processed_data/" \
        "$HOME/data/P19/splits/phy19_split${i}_new.npy"
done

# ---- P12 ×3 subsets ----
for i in 1 2 3; do
    run_cal "trans_medical_p12_subset${i}" p12 "$HOME/data/P12/processed_data/" \
        "$HOME/data/P12/splits/phy12_split_subset${i}.npy"
done

# ---- PAM ×5 splits（多分类） ----
for i in 1 2 3 4 5; do
    run_cal "trans_medical_pam_split${i}" pam "$HOME/data/PAM/processed_data/" \
        "$HOME/data/PAM/splits/PAMAP2_split_${i}.npy"
done

echo
echo "===== 汇总 ====="
python log_cal2_summary.py --append
echo
echo "第③步(a)完成。28 个 ckpt 的 4 方法校准 + classwise 指标已写入 results/cal2_*/calibration_suite.json，汇总已追加 experiment_log.md。"
