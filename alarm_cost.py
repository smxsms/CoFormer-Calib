"""创新点④：告警代价分析（决策阈值/告警视角，二分类临床任务 P19/P12）。

科学立场（与校准方法保序性自洽）：
  四种校准方法均为保序映射 -> 可达的（假警报率 FPR, 漏报率 FNR）权衡前沿不变
  （以 AUROC 不变性做机器验证）。校准改变的是：
  (a) 固定阈值下的实际决策（临床用固定阈值告警：同一 τ，警报集合改变 -> 代价变化）；
  (b) 阈值迁移性（在 val 上选定的 τ 迁移到 test 时，校准后概率语义一致 -> 漂移更小）。
  两者合起来回答"校准在临床上有什么用、何时没用"。

用法（~/CoFormer-Calib 下）：
  python alarm_cost.py --dataset p19 --backbone grud \
      --ckpt results/grud_s1_p19_split1/models/model_0020.p \
      --data_root ~/data/P19/processed_data/ \
      --split_path ~/data/P19/splits/phy19_split1_new.npy \
      --out results/alarm_grud_s1_p19_split1/alarm_cost.json
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from data.dataloader import medical_dataloader, medicalp12_dataloader, seq_collate_irregular
from calibrate_suite import build_model, collect_logits, fit_binary
from utils.utils import prepare_seed
from sklearn.metrics import roc_auc_score

torch.backends.cudnn.deterministic = True

W_FN = 10.0          # 漏报相对假警报的代价权重（临床默认：漏掉脓毒症远比虚警严重）
TAUS_FIXED = [0.05, 0.1, 0.2, 0.3, 0.5]
N_SWEEP = 200        # 权衡前沿曲线的扫描点数
TARGET_SENS = 0.80   # 阈值迁移实验：val 上目标敏感度


def sweep_metrics(p, y, taus):
    """返回 dict: 每个 tau 的 (FPR, FNR, sens)。p (N,), y (N,) 0/1 numpy。"""
    npos = max(1.0, float((y == 1).sum()))
    nneg = max(1.0, float((y == 0).sum()))
    preds = p[:, None] >= taus[None, :]                       # (N, n_tau)
    tp = (preds & (y == 1)[:, None]).sum(axis=0).astype(float)
    alarms = preds.sum(axis=0).astype(float)
    fp = alarms - tp
    fn = npos - tp
    tn = nneg - fp
    return {'fpr': fp / nneg, 'fnr': fn / npos, 'sens': tp / npos}


def norm_cost(fpr, fnr, pi, w_fn=W_FN):
    """归一化期望代价 C in [0,1]：w_fn·pi·FNR + 1·(1-pi)·FPR，除以最坏情形。"""
    return (w_fn * pi * fnr + (1.0 - pi) * fpr) / (w_fn * pi + (1.0 - pi))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['p19', 'p12'])
    p.add_argument('--backbone', default='coformer', choices=['coformer', 'grud', 'talstm'])
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--split_path', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_layers', type=int, default=8)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--d_ff', type=int, default=256)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--num_neighbors', type=int, default=30)
    args = p.parse_args()

    defaults = {'p19': dict(num_agents=34, static_dim=6, output_dim=2),
                'p12': dict(num_agents=36, static_dim=9, output_dim=2)}[args.dataset]
    for k, v in defaults.items():
        setattr(args, k, v)

    prepare_seed(1)
    device = torch.device('cuda', args.gpu)
    loader_cls = medical_dataloader if args.dataset == 'p19' else medicalp12_dataloader
    val_loader = DataLoader(loader_cls(args.data_root, args.split_path, part=1),
                            batch_size=args.batch_size, num_workers=4,
                            collate_fn=seq_collate_irregular)
    test_loader = DataLoader(loader_cls(args.data_root, args.split_path, part=2),
                             batch_size=args.batch_size, num_workers=4,
                             collate_fn=seq_collate_irregular)

    model = build_model(args)
    ck = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ck['model_dict'] if isinstance(ck, dict) and 'model_dict' in ck else ck)
    model.to(device)

    print('collecting val/test logits ...')
    logits_val, gt_val = collect_logits(model, val_loader, device, multiclass=False)
    logits_test, gt_test = collect_logits(model, test_loader, device, multiclass=False)
    z_val, y_val = logits_val[:, 1], gt_val.view(-1).float()
    z_test, y_test = logits_test[:, 1], gt_test.view(-1).float()
    y = y_test.numpy()
    pi = float(y.mean())

    taus_sweep = np.linspace(0.01, 0.99, N_SWEEP)
    sm_none = sweep_metrics(torch.sigmoid(z_test).numpy(), y, taus_sweep)

    methods = {'none': torch.sigmoid(z_test).numpy()}
    for m in ['temperature', 'vector', 'platt', 'isotonic']:
        apply_f, _ = fit_binary(z_val, y_val, m)
        methods[m] = apply_f(z_test)

    result = {
        'dataset': args.dataset, 'backbone': args.backbone, 'ckpt': args.ckpt,
        'n_test': int(len(y)), 'pos_rate_test': round(pi, 6), 'w_fn': W_FN,
    }

    # ---- (a) 固定阈值下的决策与代价 ----
    fixed = {}
    for tau in TAUS_FIXED:
        row = {}
        dec_none = (methods['none'] >= tau)
        for m, p_arr in methods.items():
            dec = (p_arr >= tau)
            sm = sweep_metrics(p_arr, y, np.array([tau]))
            fpr, fnr = float(sm['fpr'][0]), float(sm['fnr'][0])
            c = float(norm_cost(fpr, fnr, pi))
            row[m] = {'fpr': round(fpr, 4), 'fnr': round(fnr, 4), 'cost': round(c, 5),
                      'flip_rate': round(float((dec != dec_none).mean()), 4)}
        c0 = row['none']['cost']
        for m in row:
            row[m]['dcost'] = round(row[m]['cost'] - c0, 5)
        fixed['tau={:.2f}'.format(tau)] = row
    result['fixed_tau'] = fixed

    # ---- (b) 阈值迁移：val 上选 τ 使 sens>=0.8（取最大满足者），套到 test ----
    transfer = {}
    yv = y_val.numpy()
    methods_val = {'none': torch.sigmoid(z_val).numpy()}
    for m in ['temperature', 'vector', 'platt', 'isotonic']:
        apply_f, _ = fit_binary(z_val, y_val, m)
        methods_val[m] = apply_f(z_val)
    for m in ['none', 'temperature', 'vector', 'platt', 'isotonic']:
        smv = sweep_metrics(methods_val[m], yv, taus_sweep)
        ok = np.where(smv['sens'] >= TARGET_SENS)[0]
        if len(ok) == 0:
            tau_t = float(taus_sweep[0])   # 全阈值都达不到目标 sens，取最激进
        else:
            tau_t = float(taus_sweep[ok[-1]])  # 满足 sens 目标的最大 τ（最保守告警）
        smt = sweep_metrics(methods[m], y, np.array([tau_t]))
        transfer[m] = {'tau': round(tau_t, 3),
                       'val_sens': round(float(sweep_metrics(
                           methods_val[m], yv, np.array([tau_t]))['sens'][0]), 4),
                       'test_sens': round(float(smt['sens'][0]), 4),
                       'test_fpr': round(float(smt['fpr'][0]), 4),
                       'sens_drift': round(float(smt['sens'][0]) - TARGET_SENS, 4)}
    result['transfer_target_sens'] = TARGET_SENS
    result['transfer'] = transfer

    # ---- (c) 保序性机器验证：AUROC 不变 ----
    inv = {}
    a0 = roc_auc_score(y, methods['none'])
    inv['none'] = round(float(a0), 6)
    for m in ['temperature', 'vector', 'platt', 'isotonic']:
        inv[m] = round(float(roc_auc_score(y, methods[m])), 6)
    result['auroc_invariance'] = inv

    # ---- 权衡前沿曲线（作图用，紧凑存储）----
    curves = {'taus': [round(t, 4) for t in taus_sweep[::4]],
              'none': {'fpr': [round(v, 4) for v in sm_none['fpr'][::4]],
                       'fnr': [round(v, 4) for v in sm_none['fnr'][::4]]}}
    result['frontier_curve'] = curves   # 各方法前沿重合（保序），仅存 none

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    brief = {k: v for k, v in result.items() if k != 'frontier_curve'}
    print(json.dumps(brief, indent=2))
    print('saved to', args.out)


if __name__ == '__main__':
    main()
