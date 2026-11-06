#!/bin/bash
# =====  第⑥步(a)：告警代价分析（创新点④，二分类格，零训练）=====
# 生成日期: 2026-10-30 | 执行环境: WSL, conda env coformer, 工程目录 ~/CoFormer-Calib
# 覆盖 44 个二分类格（P19 35 + P12 9；PAM 为 8 类活动识别，告警语义不适用，显式排除）：
#   P19: e0_base_s{1,2,3} ×5（CoFormer，trans_medical 与 s1 同源不重复计）
#        {grud,talstm}_s{1,2} ×5
#   P12: trans_medical ×3 + {grud,talstm}_s1 ×3
# 每格: 固定阈值决策/代价 + 阈值迁移(sens>=0.8) + AUROC 保序验证 -> results/alarm_<tag>/
# 结束后自动拼装主表（master_table.csv + master_summary.md）并追加 experiment_log.md。
# 幂等: alarm_cost.json 已存在则跳过。预计耗时 40-90 分钟。
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate coformer
cd ~/CoFormer-Calib
mkdir -p ~/coformer_calib_logs

FAILED_LOG=~/coformer_calib_logs/step6_failed.txt
: > ${FAILED_LOG}

run_alarm () {  # <tag> <dataset> <backbone> <data_root> <split_path>
    local tag=$1 ds=$2 bb=$3 root=$4 sp=$5
    local out=alarm_${tag}
    if [ ! -f "results/${tag}/models/model_0020.p" ]; then
        echo "[miss] results/${tag} 无 ckpt，跳过"
        return 0
    fi
    if [ -f "results/${out}/alarm_cost.json" ]; then
        echo "[skip] results/${out} 已存在"
        return 0
    fi
    echo "===== [alarm] ${out} ====="
    mkdir -p "results/${out}"
    if python alarm_cost.py --dataset "$ds" --backbone "$bb" \
        --ckpt "results/${tag}/models/model_0020.p" \
        --data_root "$root" --split_path "$sp" \
        --out "results/${out}/alarm_cost.json" \
        2>&1 | tee ~/coformer_calib_logs/${out}.log; then
        echo "[done] ${out}"
    else
        echo "[FAILED] ${out}" | tee -a ${FAILED_LOG}
    fi
}

# ---- P19: CoFormer e0_base s1/s2/s3 ----
for s in 1 2 3; do
    for i in 1 2 3 4 5; do
        run_alarm "e0_base_s${s}_p19_split${i}" p19 coformer \
            "$HOME/data/P19/processed_data/" "$HOME/data/P19/splits/phy19_split${i}_new.npy"
    done
done

# ---- P19: 新骨干 s1/s2 ----
for bb in grud talstm; do
    for s in 1 2; do
        for i in 1 2 3 4 5; do
            run_alarm "${bb}_s${s}_p19_split${i}" p19 $bb \
                "$HOME/data/P19/processed_data/" "$HOME/data/P19/splits/phy19_split${i}_new.npy"
        done
    done
done

# ---- P12: CoFormer + 新骨干 s1 ----
for i in 1 2 3; do
    run_alarm "trans_medical_p12_subset${i}" p12 coformer \
        "$HOME/data/P12/processed_data/" "$HOME/data/P12/splits/phy12_split_subset${i}.npy"
done
for bb in grud talstm; do
    for i in 1 2 3; do
        run_alarm "${bb}_s1_p12_subset${i}" p12 $bb \
            "$HOME/data/P12/processed_data/" "$HOME/data/P12/splits/phy12_split_subset${i}.npy"
    done
done

echo
echo "===== 队列结束 ====="
if [ -s ${FAILED_LOG} ]; then
    echo "失败格（修复后重跑即续）："; cat ${FAILED_LOG}
else
    echo "全部成功。"
fi

# ---- 主表拼装 ----
echo
echo "===== 主表拼装 ====="
python build_master_table.py --out results/master 2>&1 | tee ~/coformer_calib_logs/master_table.log

# ---- 追加 experiment_log.md ----
python - <<'EOF'
import datetime, glob, json, numpy as np
files = sorted(glob.glob('results/alarm_*/alarm_cost.json'))
if files:
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    lines = ['\n\n## {} · 第⑥步(a) 告警代价汇总（{} 格）\n'.format(ts, len(files))]
    lines.append('| 格 | temp dcost(τ=.2) | vec dcost | platt dcost | iso dcost | '
                 'temp |drift| | vec |drift| | AUROC 不变 |\n|---|---|---|---|---|---|---|---|\n')
    for f in files:
        d = json.load(open(f))
        tag = f.split('/')[1].replace('alarm_', '')
        fx = d['fixed_tau'].get('tau=0.20', {})
        tr = d['transfer']
        inv = d['auroc_invariance']
        a0 = inv['none']
        maxdev = max(abs(inv[m] - a0) for m in ['temperature', 'vector', 'platt', 'isotonic'])
        lines.append('| {} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | {:.3f} | {:.3f} | {:.5f} |\n'.format(
            tag,
            fx['temperature']['dcost'], fx['vector']['dcost'],
            fx['platt']['dcost'], fx['isotonic']['dcost'],
            abs(tr['temperature']['sens_drift']), abs(tr['vector']['sens_drift']),
            maxdev))
    with open('experiment_log.md', 'a') as fo:
        fo.write(''.join(lines))
    print('已追加 {} 格告警汇总到 experiment_log.md'.format(len(files)))
EOF
echo
echo "第⑥步(a)完成。请把 results/master/master_summary.md 发给 Buddy 或直接告知，我来回填实验记录并做最终判读。"
