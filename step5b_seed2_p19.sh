#!/bin/bash
# =====  第⑤步(b)：门 B 保险——新骨干 × P19 × seed2（10 runs）=====
# 生成日期: 2026-10-29 | 执行环境: WSL, conda env coformer, 工程目录 ~/CoFormer-Calib
# 目的: 回应"新骨干校准行为是否 seed 稳定"的最可能审稿质疑。
#   训练 GRU-D / ta-LSTM × P19 split1-5 × seed2，逐 run 跑校准套件；
#   结束后自动输出 seed1 vs seed2 的判定稳定性对比报告并追加 experiment_log.md。
# 幂等: results/<TAG>/results/summary.json 存在则跳过训练；cal2 json 存在则跳过校准。
# 预计耗时: 10 runs × 0.5-1 h ≈ 5-10 GPU·h。
# 建议启动: nohup bash step5b_seed2_p19.sh > ~/coformer_calib_logs/step5b.log 2>&1 &
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate coformer
cd ~/CoFormer-Calib
mkdir -p ~/coformer_calib_logs

FAILED_LOG=~/coformer_calib_logs/step5b_failed.txt
: > ${FAILED_LOG}

run_one () {  # <backbone> <N>
    local bb=$1 N=$2
    local tag=${bb}_s2_p19_split${N}
    local out=cal2_${tag}
    local root=$HOME/data/P19/processed_data/
    local sp=$HOME/data/P19/splits/phy19_split${N}_new.npy

    if [ -f "results/${tag}/results/summary.json" ]; then
        echo "[skip-train] results/${tag} 已存在"
    else
        echo
        echo "===== [train] ${tag} ====="
        if python train_medical.py --dataset p19 --gpu 0 \
            --data_root "$root" --split_path "$sp" \
            --num_agents 34 --static_dim 6 --output_dim 2 \
            --batch_size 16 --backbone $bb --seed 2 --exp_name ${tag} \
            2>&1 | tee ~/coformer_calib_logs/${tag}.log; then
            if [ -d results/trans_medical ]; then
                if [ -d "results/${tag}" ]; then
                    mv "results/${tag}" "results/${tag}_prev_$(date '+%m%d_%H%M%S')"
                fi
                mv results/trans_medical "results/${tag}"
                echo "[archive] -> results/${tag}"
            fi
        else
            echo "[FAILED-train] ${tag}" | tee -a ${FAILED_LOG}
            if [ -d results/trans_medical ]; then
                mv results/trans_medical "results/_aborted_$(date '+%m%d_%H%M%S')"
            fi
            return 0
        fi
    fi

    if [ -f "results/${out}/calibration_suite.json" ]; then
        echo "[skip-cal] results/${out} 已存在"
    elif [ -f "results/${tag}/models/model_0020.p" ]; then
        echo "===== [cal] ${out} ====="
        mkdir -p "results/${out}"
        if python calibrate_suite.py --dataset p19 --backbone $bb \
            --ckpt "results/${tag}/models/model_0020.p" \
            --data_root "$root" --split_path "$sp" \
            --out "results/${out}/calibration_suite.json" \
            2>&1 | tee ~/coformer_calib_logs/${out}.log; then
            echo "[done] ${out}"
        else
            echo "[FAILED-cal] ${out}" | tee -a ${FAILED_LOG}
        fi
    else
        echo "[FAILED] results/${tag} 无 model_0020.p" | tee -a ${FAILED_LOG}
    fi
}

for bb in grud talstm; do
    for i in 1 2 3 4 5; do
        run_one $bb $i
    done
done

echo
echo "===== 队列结束 ====="
if [ -s ${FAILED_LOG} ]; then
    echo "以下 run 失败（修复后重跑本脚本即续）："; cat ${FAILED_LOG}
else
    echo "全部 run 成功。"
fi

# ---- 跨 seed 判定稳定性对比（本步骤的核心产出）----
python - <<'EOF'
import json, glob, datetime
import numpy as np

def load(tag):
    try:
        return json.load(open('results/cal2_{}/calibration_suite.json'.format(tag)))
    except Exception:
        return None

def v_temp(d):
    t = d['methods']['temperature']
    ci = t['params'].get('T_ci95', [1, 1])
    if not (ci[0] > 1 or ci[1] < 1):
        return 'F1'
    if t['ece_rel_drop_pct'] < 10:
        return 'F2'
    if not t['brier_improved']:
        return 'F3'
    return 'G'

def v_other(met):
    return 'G' if (met['ece_rel_drop_pct'] >= 10 and met['brier_improved']) else 'F'

lines = ['### 新骨干 P19 跨 seed 稳定性（seed1 vs seed2）\n']
for bb in ['grud', 'talstm']:
    agree = {m: 0 for m in ['temperature', 'vector', 'platt', 'isotonic']}
    n = 0
    lines.append('\n**{}（5 splits）**\n'.format(bb))
    lines.append('| split | T1 (CI) | T2 (CI) | temp1/temp2 | vec1/vec2 | platt1/platt2 | iso1/iso2 |')
    lines.append('|---|---|---|---|---|---|---|')
    for i in range(1, 6):
        d1 = load('{}_s1_p19_split{}'.format(bb, i))
        d2 = load('{}_s2_p19_split{}'.format(bb, i))
        if d1 is None or d2 is None:
            lines.append('| split{} | 缺产物 | 缺产物 | - | - | - | - |'.format(i))
            continue
        n += 1
        t1, t2 = d1['methods']['temperature'], d2['methods']['temperature']
        def fmt_ci(t):
            ci = t['params'].get('T_ci95', [1, 1])
            return '{:.3f} [{:.2f},{:.2f}]'.format(t['params'].get('T', float('nan')), ci[0], ci[1])
        vt1, vt2 = v_temp(d1), v_temp(d2)
        vv1, vv2 = v_other(d1['methods']['vector']), v_other(d2['methods']['vector'])
        vp1, vp2 = v_other(d1['methods']['platt']), v_other(d2['methods']['platt'])
        vi1, vi2 = v_other(d1['methods']['isotonic']), v_other(d2['methods']['isotonic'])
        agree['temperature'] += (vt1 == vt2)
        agree['vector'] += (vv1 == vv2)
        agree['platt'] += (vp1 == vp2)
        agree['isotonic'] += (vi1 == vi2)
        lines.append('| split{} | {} | {} | {}/{} | {}/{} | {}/{} | {}/{} |'.format(
            i, fmt_ci(t1), fmt_ci(t2), vt1, vt2, vv1, vv2, vp1, vp2, vi1, vi2))
    lines.append('\n判定一致率: temperature {}/{}, vector {}/{}, platt {}/{}, isotonic {}/{}'.format(
        agree['temperature'], n, agree['vector'], n, agree['platt'], n, agree['isotonic'], n))
    # T 的 seed 间一致性
    Ts1 = [load('{}_s1_p19_split{}'.format(bb, i))['methods']['temperature']['params']['T'] for i in range(1, 6) if load('{}_s1_p19_split{}'.format(bb, i))]
    Ts2 = [load('{}_s2_p19_split{}'.format(bb, i))['methods']['temperature']['params']['T'] for i in range(1, 5 + 1) if load('{}_s2_p19_split{}'.format(bb, i))]
    if Ts1 and Ts2:
        r = np.corrcoef(Ts1[:len(Ts2)], Ts2[:len(Ts1)])[0, 1] if len(Ts1) == len(Ts2) else float('nan')
        lines.append('T split 间排序 seed 相关: r = {:.3f} | T1 {:.3f}±{:.3f} vs T2 {:.3f}±{:.3f} (ddof=1)'.format(
            r, np.mean(Ts1), np.std(Ts1, ddof=1), np.mean(Ts2), np.std(Ts2, ddof=1)))
report = '\n'.join(lines)
print(report)
ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
with open('experiment_log.md', 'a') as f:
    f.write('\n\n## {} · 第⑤步(b)：新骨干 P19 seed2 跨 seed 稳定性\n\n'.format(ts))
    f.write(report + '\n')
print('\n已追加到 experiment_log.md')
EOF
python log_cal2_summary.py --append
echo
echo "第⑤步(b)完成。请把对比报告发给 Buddy 回填实验记录并更新门 B 判定。"
