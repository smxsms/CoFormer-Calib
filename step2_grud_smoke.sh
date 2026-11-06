#!/bin/bash
# =====  第②步：GRU-D × P19 split1 首跑验证（门 A）=====
# 生成日期: 2026-10-26 | 执行环境: WSL, conda env coformer, 工程目录 ~/CoFormer-Calib
# 内容: 1) 训练 GRU-D × P19 split1 × seed1（--backbone grud，新骨干首个 run）
#       2) 校准套件（4 方法 + classwise + 预注册判定原料）跑该 ckpt
#       3) 输出门 A 判读报告（训练收敛性 / 校准增益），并自动追加到 experiment_log.md
# 幂等: 训练产物 summary.json 已存在则跳过训练；校准 json 已存在则跳过校准。
# 预计耗时: 训练约 1-2 GPU·h + 校准约 5 分钟（含 1000 次 bootstrap）。
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate coformer
cd ~/CoFormer-Calib
mkdir -p ~/coformer_calib_logs

TAG=grud_s1_p19_split1
CAL=cal2_${TAG}

# ---- 1) 训练 ----
if [ -f results/${TAG}/results/summary.json ]; then
    echo "[skip] results/${TAG} 训练产物已存在"
else
    echo "===== [train] ${TAG} ====="
    python train_medical.py --dataset p19 --gpu 0 \
        --data_root $HOME/data/P19/processed_data/ \
        --split_path $HOME/data/P19/splits/phy19_split1_new.npy \
        --num_agents 34 --static_dim 6 --output_dim 2 \
        --batch_size 16 --backbone grud --seed 1 --exp_name ${TAG} \
        2>&1 | tee ~/coformer_calib_logs/${TAG}.log
    if [ -d results/trans_medical ]; then
        if [ -d results/${TAG} ]; then
            mv results/${TAG} "results/${TAG}_prev_$(date '+%m%d_%H%M%S')"
        fi
        mv results/trans_medical results/${TAG}
        echo "[archive] results/trans_medical -> results/${TAG}"
    fi
fi

# ---- 2) 校准套件 ----
if [ -f results/${CAL}/calibration_suite.json ]; then
    echo "[skip] results/${CAL} 校准产物已存在"
else
    echo "===== [calibrate] ${CAL} ====="
    mkdir -p results/${CAL}
    python calibrate_suite.py --dataset p19 --backbone grud \
        --ckpt results/${TAG}/models/model_0020.p \
        --data_root $HOME/data/P19/processed_data/ \
        --split_path $HOME/data/P19/splits/phy19_split1_new.npy \
        --out results/${CAL}/calibration_suite.json \
        2>&1 | tee ~/coformer_calib_logs/${CAL}.log
fi

# ---- 3) 门 A 判读 + 记录 ----
python - <<'EOF'
import json, datetime
summary = json.load(open('results/grud_s1_p19_split1/results/summary.json'))
cal = json.load(open('results/cal2_grud_s1_p19_split1/calibration_suite.json'))
base = cal['methods']['none']
t = cal['methods'].get('temperature', {})

lines = []
lines.append('### 门 A 判读（GRU-D × P19 split1 首跑）\n')
lines.append('- 训练: best_auprc = {} (epoch {}), best_auroc = {} (epoch {})'.format(
    summary.get('best_auprc'), summary.get('best_auprc_epoch'),
    summary.get('best_auroc'), summary.get('best_auroc_epoch')))
lines.append('  - 收敛性参考: CoFormer 基线 AUPRC 57.54 ± 2.49；GRU-D 为新骨干，首跑落点人工判读（建议 >=45 视为工程无 bug）')
lines.append('- 判别: AUROC = {}, AUPRC = {}, pos_rate_test = {}'.format(
    cal.get('auroc'), cal.get('auprc'), cal.get('pos_rate_test')))
if t:
    ci = t.get('params', {}).get('T_ci95')
    lines.append('- 温度缩放: T = {}, T_ci95 = {}, CI 是否跨 1: {}（跨 1 = 无可靠增益，预注册主判定）'.format(
        t.get('params', {}).get('T'), ci, not t.get('params', {}).get('T_ci_excludes_1', False)))
    lines.append('  - ECE {:.5f} -> {:.5f} ({:+.1f}%), Brier {:.5f} -> {:.5f}（improved={}）'.format(
        base['ece'], t.get('ece'), t.get('ece_rel_drop_pct'), base['brier'], t.get('brier'),
        t.get('brier_improved')))
lines.append('- 注意: 门 A 的结论是人工判读项，以上为全部判定原料')
report = '\n'.join(lines)
print(report)

ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
with open('experiment_log.md', 'a') as f:
    f.write('\n\n## {} · 第②步 门 A：GRU-D × P19 split1 首跑验证\n\n'.format(ts))
    f.write(report + '\n')
print('\n已追加到 experiment_log.md')
EOF
python log_cal2_summary.py --append
echo
echo "第②步完成。请把上面门 A 判读报告的结论告知 Buddy，通过后即可启动 step4_matrix.sh（批 2）。"
