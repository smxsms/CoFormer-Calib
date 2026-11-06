#!/bin/bash
# =====  第④步：骨干训练矩阵首波（GRU-D / ta-LSTM × 13 splits × seed1）=====
# 生成日期: 2026-10-26 | 执行环境: WSL, conda env coformer, 工程目录 ~/CoFormer-Calib
# 前提: 门 A 通过（step2_grud_smoke.sh 的 GRU-D 首跑已验证新骨干代码无 bug）。
# 内容: 26 个训练 run（grud 13 + talstm 13；GRU-D × P19 split1 已由第②步完成，自动跳过），
#       每个训练完成后立即跑校准套件，结果自动追加 experiment_log.md。
# 幂等: results/<TAG>/results/summary.json 存在则跳过训练；cal2 json 存在则跳过校准。
#       中断后重跑即续。失败的 run 记录后继续，不阻塞队列。
# 预计耗时: 每 run 约 0.5-2 GPU·h（PAM 较快、P19 较慢），总计约 20-50 GPU·h（数天队列）。
# 建议启动方式: nohup bash step4_matrix.sh > ~/coformer_calib_logs/step4_matrix.log 2>&1 &
#   然后 tail -f ~/coformer_calib_logs/step4_matrix.log 跟踪进度。
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate coformer
cd ~/CoFormer-Calib
mkdir -p ~/coformer_calib_logs

FAILED_LOG=~/coformer_calib_logs/step4_failed.txt
: > ${FAILED_LOG}

run_one () {  # <backbone> <dataset> <splitname> <N> <bs>
    local bb=$1 ds=$2 spname=$3 N=$4 BS=$5
    local tag=${bb}_s1_${ds}_${spname}
    local out=cal2_${tag}
    local root sp agents staticdim outdim
    case $ds in
        p19) root=$HOME/data/P19/processed_data/; sp=$HOME/data/P19/splits/phy19_split${N}_new.npy; agents=34; staticdim=6; outdim=2 ;;
        p12) root=$HOME/data/P12/processed_data/; sp=$HOME/data/P12/splits/phy12_split_subset${N}.npy; agents=36; staticdim=9; outdim=2 ;;
        pam) root=$HOME/data/PAM/processed_data/; sp=$HOME/data/PAM/splits/PAMAP2_split_${N}.npy; agents=17; staticdim=6; outdim=8 ;;
        *) echo "unknown dataset $ds"; return 1 ;;
    esac

    # ---- 训练 ----
    if [ -f "results/${tag}/results/summary.json" ]; then
        echo "[skip-train] results/${tag} 已存在"
    else
        echo
        echo "===== [train] ${tag} ====="
        if python train_medical.py --dataset $ds --gpu 0 \
            --data_root "$root" --split_path "$sp" \
            --num_agents $agents --static_dim $staticdim --output_dim $outdim \
            --batch_size $BS --backbone $bb --seed 1 --exp_name ${tag} \
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
            # 归档可能存在的半成品，防止污染下一个 run
            if [ -d results/trans_medical ]; then
                mv results/trans_medical "results/_aborted_$(date '+%m%d_%H%M%S')"
            fi
            return 0
        fi
    fi

    # ---- 校准套件 ----
    if [ -f "results/${out}/calibration_suite.json" ]; then
        echo "[skip-cal] results/${out} 已存在"
    elif [ -f "results/${tag}/models/model_0020.p" ]; then
        echo "===== [cal] ${out} ====="
        mkdir -p "results/${out}"
        if python calibrate_suite.py --dataset $ds --backbone $bb \
            --ckpt "results/${tag}/models/model_0020.p" \
            --data_root "$root" --split_path "$sp" \
            --out "results/${out}/calibration_suite.json" \
            2>&1 | tee ~/coformer_calib_logs/${out}.log; then
            echo "[done] ${out}"
        else
            echo "[FAILED-cal] ${out}" | tee -a ${FAILED_LOG}
        fi
    else
        echo "[FAILED] results/${tag} 无 model_0020.p，无法校准" | tee -a ${FAILED_LOG}
    fi
}

# ---- 训练矩阵：grud 13 + talstm 13（grud_p19_split1 由第②步完成，幂等跳过）----
for bb in grud talstm; do
    for i in 1 2 3 4 5; do
        run_one $bb p19 split$i $i 16
    done
    for i in 1 2 3; do
        run_one $bb p12 subset$i $i 4
    done
    for i in 1 2 3 4 5; do
        run_one $bb pam split$i $i 6
    done
done

echo
echo "===== 队列结束 ====="
if [ -s ${FAILED_LOG} ]; then
    echo "以下 run 失败（可修复后直接重跑本脚本，幂等续跑）："
    cat ${FAILED_LOG}
else
    echo "全部 run 成功。"
fi
echo
python log_cal2_summary.py --append
echo
echo "第④步完成。门 B（是否补第 2 个 seed）与门 C（P12/PAM 跨 seed 补训 vs Limitations）的证据已齐，请把汇总发给 Buddy 做判定。"
