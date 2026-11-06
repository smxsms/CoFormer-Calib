# Code Release

Code for *"A Multi-Backbone, Multi-Task Empirical Study of Post-hoc Calibration for Irregularly-Sampled Clinical Time Series Prediction"*.

This repository contains the complete pipeline that reproduces every metric in the paper: training the three backbones (CoFormer, GRU-D, time-aware LSTM) on three benchmarks (PhysioNet 2019 sepsis, PhysioNet 2012 length-of-stay, PAMAP2 activity recognition), fitting the four post-hoc calibrators (temperature scaling, vector scaling, Platt scaling, isotonic regression), the pre-specified verdict matrix, the alarm-cost analysis, and the patient-level paired bootstrap intervals.

## Requirements

- Python >= 3.9 (experiments used 3.10, PyTorch 2.1.2 + CUDA 12.1)
- `torch`, `dgl`, `scikit-learn`, `scipy`, `tqdm`, `matplotlib`, `numpy` (see `requirements.txt`)
- A single 24 GB GPU suffices for all experiments

## Data

Preprocessing and split files follow the official [Raindrop](https://github.com/mims-harvard/Raindrop) release.

1. Download the raw data:
   - P19: [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)
   - P12: [PhysioNet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/)
   - PAM: [PAMAP2](http://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring)
2. Run the scripts in `preprocess/` (P19.py, P12.py, PAM.py) to build `PTdict_list.npy`; or download the preprocessed data and split index directly from the links in the Raindrop repository.
3. The exact split files used for every cell, kept under their original names, are: `phy19_split{1..5}_new.npy`, `phy12_split_subset{1..3}.npy`, `PAMAP2_split_{1..5}.npy`.

## Pipeline

The `step*.sh` scripts orchestrate the pipeline end to end; each step can also be run manually.

| Step | Script | Output |
|---|---|---|
| 1. Train a backbone | `train_medical.py --dataset {p19,p12,pam} --split <n> --backbone {coformer,grud,talstm} ...` | checkpoints at epochs 1/10/20 under `results/<run>/models/` |
| 2. Fit calibrators + evaluate | `calibrate_suite.py` | per-cell `calibration_suite.json` (aggregate + classwise metrics, fitted parameters, bootstrap intervals of the fitted temperature) |
| 3. Alarm-cost analysis | `alarm_cost.py` | per-cell `alarm_cost.json` (FPR/FNR/normalized cost at τ ∈ {0.05, 0.10, 0.20, 0.30, 0.50}, flip rates) |
| 4. Master table | `build_master_table.py` | machine-readable table, one row per cell and method |
| 5. Paired uncertainty | `m5/` (see `m5/README.md`) | patient-level paired bootstrap CIs, split-cluster sensitivity bound, McNemar-style flip decomposition |

Configuration for training lives in `cfg/trans_medical.yml` and the argparse defaults of `train_medical.py` (Adam, lr 1e-4 with ×0.5 step decay every 10 epochs, 20 epochs, batch sizes 16/4/6 for P19/P12/PAM). Following the paper's checkpoint protocol, the **final (epoch-20) checkpoint is used for every table and figure**; there is no early stopping and no test-set quantity enters any selection.

## Directory layout

```
├── train_medical.py        # training entry point (three backbones)
├── models/                 # CoFormer encoder + GRU-D / time-aware LSTM backbones
├── data/dataloader.py      # dataset loading
├── preprocess/             # P19 / P12 / PAM preprocessing
├── cfg/                    # training configuration
├── calibrate_suite.py      # the four post-hoc calibrators + evaluation
├── alarm_cost.py           # alarm-cost analysis
├── batch1_recompute.py     # recompute/verify run records
├── build_master_table.py   # machine-readable master table
├── step2–step6             # end-to-end pipeline drivers
├── m5/                     # patient-level paired bootstrap (reviewer-requested addition)
└── requirements.txt
```

Note: run the pipeline from the repository root (scripts import sibling modules via the working directory). `m5/` expects trained checkpoints under `results/` and re-runs the frozen-model forward passes; it does not retrain anything.
