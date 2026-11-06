# M5 计算包：患者级配对不确定性（对应审稿意见 2）

> 目的：给论文补上 **P19 主比较的患者级配对 95% 区间**，并顺带产出决策翻转分解（支撑 §5.5 的机制句）。
> 零训练：只重跑前向 + 重拟合校准器。**不改任何既有产物**，所有输出写进新目录。

---

## 1. 为什么必须重算

工程里从来没有保存过逐样本概率——`calibration_suite.json` 只有聚合指标，`alarm_cost.json` 只有 (FPR, FNR, cost, flip_rate)，
全仓库 `*.npy` 概率文件为 **0 个**。配对 bootstrap 需要「同一批患者、同一重抽样索引」下的两条预测，所以必须先重跑前向。

## 2. 前置条件

| 项 | 要求 |
|---|---|
| 位置 | **工程在 WSL**：`~/CoFormer-Calib`（数据 `~/data/{P19,P12}/processed_data/`，产物 `~/CoFormer-Calib/results/`） |
| 脚本位置 | 已安装到 **`~/CoFormer-Calib/m5/`** |
| 环境 | WSL，`conda activate coformer`（torch 2.1.2+cu121、sklearn 1.7.2、scipy 1.15.3，已实测可用） |
| 工作目录 | 运行时**必须 `cd ~/CoFormer-Calib`**（脚本靠 cwd import 既有 `calibrate_suite.py`、`data.dataloader`） |
| GPU | 单卡 4090 即可；显存占用 < 1 GB |
| 产物依赖 | `results/cal2_*/calibration_suite.json`（64 份）与 `results/*/models/model_0020.p` 在位 |

## 3. 怎么跑

```bash
# 一键（P19 + P12 共 44 格）
bash ~/CoFormer-Calib/m5/run_m5.sh

# 只跑 P19 35 格（论文主比较，省 15 分钟）
bash ~/CoFormer-Calib/m5/run_m5.sh p19
```

分步跑（便于中断看结果）：

```bash
cd ~/CoFormer-Calib

# ① 只试 2 格，1-2 分钟，先确认回归核对能过
python m5/m5_01_dump_probs.py --limit 2

# ② 全量（44 格）
python m5/m5_01_dump_probs.py --datasets p19,p12

# ③ 配对 bootstrap
python m5/m5_02_paired_bootstrap.py

# ④ 图
python m5/m5_03_plot_ci.py
```

**实测耗时**（2026-11-06 全量跑）：44 格概率导出 **31.3 分钟**（≈43 s/格；首格含 CUDA 预热约 95 s），
配对 bootstrap ≈1.5 分钟，图 ≈10 秒。重跑一键脚本（已完成的 npz 自动跳过）≈2 分钟。
想再快可加 `--batch_size 32`，但必须复核 `regression_report.md` 仍然零 DIFF。

## 4. 三个脚本各做什么

### `m5_01_dump_probs.py` —— 概率导出 + 回归核对

- **不重写校准实现**：直接 `from calibrate_suite import build_model, collect_logits, fit_binary, binary_metrics`，
  与既有 64 份 JSON 同源同口径（这是能"逐位对上"的前提）。
- 每个格落盘 `probs/probs_<tag>.npz`：`y`、`test_index`（split 文件的 test 索引，便于日后按病人对齐）、
  `p_none / p_temperature / p_vector / p_platt / p_isotonic`（float64，长度 = n_test）、拟合参数。
- **逐字段回归核对**：用新概率复算 AUROC/AUPRC/ECE/Brier/NLL/classwise-ECE/拟合参数/ECE 降幅，
  与既有 `calibration_suite.json` 比对，写 `regression_report.csv|md`。
- 退出码：`0` = 全部一致；`1` = 有 DIFF（此时**不要**进入下一步）。

> 已实测：P19 1 格 + P12 1 格 → **58 个字段全部 OK、0 DIFF**；工程内副本（`~/CoFormer-Calib/m5/`）另跑 1 格 → **29 字段 OK、0 DIFF**。
> 说明重跑的概率与既有产物同源。全量 44 格跑完应看到 `OK≈1300 / DIFF 0`。

### `m5_02_paired_bootstrap.py` —— 三层配对区间 + 翻转分解

估计量与 `alarm_cost.py` 完全一致（`w_FN=10`、`π` = 该格测试集阳性率、`p ≥ τ` 报警）：

```
a_i = w_FN·y_i·1[p_i<τ] + (1−y_i)·1[p_i≥τ]      # 患者 i 的代价分子
D   = w_FN·π + (1−π)                            # 归一化分母（π 固定，不随重抽样变）
Δc_i = (a_i^method − a_i^none)/D ； ΔC = mean_i Δc_i   （负 = 改善）
```

| 层 | 抽样单位 | 回答什么 | 在论文里怎么用 |
|---|---|---|---|
| **L1 逐格** | 格内测试患者（配对） | 这一格的 ΔC 稳不稳 | 「per-cell CI excludes zero in x/35」 |
| **L2a 设计层** | **同一 split 内各格共用一次患者重抽样** | 这套设计下均值符号稳不稳 | §5.5 主句（主区间） |
| **L2b 两级** | 先重抽格、再重抽患者 | 加入格间异质性的敏感性 | 可放脚注或省略 |
| **L3 split 簇** | 以 split 为簇（P19 5 簇 / P12 3 簇） | 换一批切分会不会翻 | §7 的保守界 + 敏感性一句 |

**为什么 L2a 不能让每格独立重抽**：同一 split 内的 3 个 CoFormer seed（以及 GRU-D / ta-LSTM 的 2 个 seed）评估的是**同一批 3,881 名测试患者**，各格 ΔC 正相关。
若每格各抽各的再平均，等于把 35 格当独立样本，区间会被压窄——这正是审稿人第 2 条要拦的口径问题。
脚本里另存了一个 `L2a_naive_independent_cells` 字段，**只用于对照诊断，不进论文**。

同时输出**决策翻转分解**（主阈值 τ=0.2）：阳性 `FN→TP` 与 `TP→FN` 各多少、精确 McNemar p、阴性侧 `TN→FP / FP→TN`。
这直接支撑 §5.5 那句 "raising the miss rate (ΔFNR +0.021)"。

### `m5_03_plot_ci.py` —— Figure 2 加置信带

- `import make_paper2_figures` 复用其配色、数据加载与判定口径，**曲线与现有 fig4 完全一致**，只多画半透明带。
- 只给 `temperature` 与 `vector` 画带（二分类下 Platt 与 vector 数值等价，避免双带重叠）。
- 输出 **新文件** `fig4_alarm_cost_ci.{pdf,png}`，不覆盖 `fig4_alarm_cost.*`，便于对比取舍。
- 图注需补一句：`Shaded bands: 95% patient-level paired bootstrap intervals for the cell-mean ΔC.`

## 5. 产物在哪 / 怎么读

```
~/CoFormer-Calib/results/m5_paired/
├── probs/probs_<tag>.npz      44 个逐样本概率（约 200 KB/个）
├── probs/index.csv            格清单（tag/dataset/backbone/split/n_test/pos_rate）
├── regression_report.md/.csv  与既有 JSON 的逐字段核对（先看这个！）
├── paired_cost_cells.csv      59 行/格 × 4 方法 × 5 阈值 = 880 行逐格明细
├── paired_cost_summary.json   三层区间 + 逐 split 均值 + 翻转计数（机器可读）
└── paper_numbers.md           ★ 可直接抄进论文的数字与英文句子
```

`paper_numbers.md` 里已经**把数值填进句子**，形如：

> At τ=0.2 on P19, the patient-level paired bootstrap (1,000 resamples of test patients, …)
> gives a mean ΔC of −0.0040 [−0.0052, −0.0028] for temperature scaling and +0.0030 [+0.0019, +0.0041]
> for vector scaling …, with per-cell intervals excluding zero in 22/35 and 14/35 cells respectively.

> 上面这组数字是**示例格式**，真实值以你跑出来的 `paper_numbers.md` 为准。

## 6. 数字进论文的落点

| 位置 | 动作 |
|---|---|
| §5.5 末段（搜 `The two views thus invert the ranking`） | 粘贴 `paper_numbers.md` 第 1 句（主区间）+ 第 3 句（机制翻转） |
| §5.5 或图注 | 加 L3 那句（split 簇保守界） |
| §7 Limitations（搜 `mitigated by replication` 附近） | 加「配对区间只量化测试集抽样噪声，无独立队列 → 不外推」 |
| Figure 2 图注 | 加「band = 95% patient-level paired bootstrap」并换用 `fig4_alarm_cost_ci.pdf` |
| 回复信第 2 段 | 把 `<u.lo> <u.hi> <x> <y>` 占位符替换成实算值 |

## 7. 出问题怎么办

| 现象 | 处理 |
|---|---|
| `regression_report.md` 有 DIFF | **停下**。先查 ckpt/split 是否与既有产物同源、库版本是否变过；不要用有偏差的概率往下算 |
| `ModuleNotFoundError: data / calibrate_suite` | 没在 `~/CoFormer-Calib` 下运行（`cd` 后再跑） |
| CUDA OOM | 加 `--batch_size 8` |
| 中断 | 直接重跑 `m5_01`，已完成的 npz 自动跳过（幂等）；`--overwrite` 才重算 |
| L2a 区间跨 0 | 不是 bug，是结果。按 `paper_numbers.md` 的风险提示把 §5.5 主张收窄（"方向一致但区间含 0"），**不要藏** |
| L3 簇区间跨 0 | 预期内的保守界（5 个簇太少），如实披露即可 |

## 8. 边界（这些不做）

1. **不重训、不补 seed、不改 Table 2**——只新增一个测试侧区间，不动任何既有判定。
2. **不把 GAIN 升级成显著性主张**：区间只加在 ΔC（决策代价）上，ECE/Brier 的预指定判定口径不变。
3. **不宣称独立队列**：没有第二家医院数据，L3 只能到 split 层；外部验证不可由本计算替代。
4. **不动既有产物**：`results/cal2_*`、`results/alarm_*`、`master/` 全程只读。
