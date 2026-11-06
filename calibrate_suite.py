"""校准套件 v2（《Post-hoc Calibration for Irregular Clinical Time Series Prediction》）。

统一校准入口（温度/向量/Platt/等渗 + classwise + 预注册判定原料）。增量：
  1. 四种后验校准方法：温度缩放 / 向量缩放 / Platt / 等渗回归（二分类 + 多分类）；
  2. classwise 指标：classwise-ECE、阳性类等频（adaptive）reliability、Brier 分解
     （REL − RES + UNC）、NLL、MCE；
  3. PAM 多分类支持：8 类 logits，温度/向量缩放直接拟合，Platt/等渗按 OvR + 重归一化；
  4. 新骨干支持：--backbone {coformer, grud, talstm}；
  5. 预注册判定原料（§5.7）：温度 T 的 bootstrap 95% CI（B=1000 重采样 val 重拟合）、
     ECE 相对降幅、Brier 是否下降——直接写入 JSON。

协议：所有方法仅在 val（split[1]）上拟合，在 test（split[2]）上报告；
AUROC/AUPRC 为排序指标，校准前后不变（交叉核对用）。

用法（~/CoFormer-Calib 下，conda env coformer）：
  python calibrate_suite.py --dataset p19 --backbone grud \
      --ckpt results/grud_s1_p19_split1/models/model_0020.p \
      --data_root ~/data/P19/processed_data/ \
      --split_path ~/data/P19/splits/phy19_split1_new.npy \
      --out results/cal2_grud_s1_p19_split1/calibration_suite.json
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.getcwd())
from data.dataloader import (medical_dataloader, medicalp12_dataloader,
                             seq_collate_irregular, seq_collate_irregular_wo_static)
from utils.utils import prepare_seed
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression

torch.backends.cudnn.deterministic = True

N_BINS = 15
BOOTSTRAP_B = 1000


# ----------------------------------------------------------------------------
# 数据：PAM 校准专用 loader（原 medicalpam_dataloader 无 part 参数且归一化统计量
# 按所加载子集现算——严格复刻该行为：part 0/1 用 train 统计，part 2 用 test 统计，
# 以保证与训练期评估的 logits 完全同分布）。
# ----------------------------------------------------------------------------
class PAMCalibDataset(Dataset):
    def __init__(self, root, split_path, part):
        split = np.load(split_path, allow_pickle=True)
        index = np.asarray(split[part])
        all_data = np.load(root + 'array.npy')
        norm_src = np.asarray(split[0] if part != 2 else split[2])
        nd = all_data[norm_src, :]
        val = nd[nd < 1e9]
        mean, std = float(np.mean(val)), float(np.std(val))
        self.data = (all_data[index, :] - mean) / std
        self.time = np.load(root + 'time.npy')[index, :]
        self.gt = np.load(root + 'gt.npy')[index, :]
        self.mask = np.load(root + 'mask.npy')[index, :]
        self.data = torch.from_numpy(self.data).type(torch.float)
        self.time = torch.from_numpy(self.time).type(torch.float)
        self.gt = torch.from_numpy(self.gt).type(torch.float)
        self.mask = torch.from_numpy(self.mask).type(torch.int)
        self.batch_len = len(self.data)

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        return [self.data[index], self.time[index], self.mask[index], self.gt[index]]


# ----------------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------------
def _bin_rows(confs, gts, edges):
    rows, ece, mce = [], 0.0, 0.0
    n = len(confs)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        m = int(in_bin.sum())
        if m == 0:
            rows.append({'bin': '{:.3f}-{:.3f}'.format(lo, hi), 'count': 0, 'conf': None, 'acc': None})
            continue
        conf = float(confs[in_bin].mean())
        acc = float(gts[in_bin].mean())
        gap = abs(conf - acc)
        ece += m / n * gap
        mce = max(mce, gap)
        rows.append({'bin': '{:.3f}-{:.3f}'.format(lo, hi), 'count': m,
                     'conf': round(conf, 5), 'acc': round(acc, 5)})
    return ece, mce, rows


def ece_equal_width(p, y, n_bins=N_BINS):
    """总体 ECE/MCE：等宽分箱。p/y: (N,) numpy"""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return _bin_rows(p, y, edges)


def ece_adaptive(p, y, n_bins=N_BINS):
    """等频（adaptive binning）ECE：按 p 排序均分，边界取相邻均值去重。"""
    order = np.argsort(p, kind='stable')
    n = len(p)
    groups = np.array_split(order, min(n_bins, n))
    edges, used = [0.0], []
    for g in groups:
        if len(g) == 0:
            continue
        used.append(g)
    edges = [0.0]
    for k in range(len(used) - 1):
        a, b = p[used[k][-1]], p[used[k + 1][0]]
        edges.append(float((a + b) / 2))
    edges.append(1.0)
    edges = sorted(set(edges))
    return _bin_rows(p, y, np.array(edges))


def classwise_ece_binary(p, y, n_bins=N_BINS):
    """classwise-ECE（二分类）：类 1 用 (p,y)，类 0 用 (1−p,1−y)，各自等频 ECE 后取均值。"""
    e1 = ece_adaptive(p, y, n_bins)[0]
    e0 = ece_adaptive(1.0 - p, 1.0 - y, n_bins)[0]
    return float((e1 + e0) / 2), {'class1': e1, 'class0': e0}


def brier_decomposition(p, y, n_bins=10):
    """Brier 的 Murphy 分解：BS = REL − RES + UNC（等宽 10 箱）。"""
    pi = float(np.mean(y))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rel = res = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        m = int(in_bin.sum())
        if m == 0:
            continue
        conf = float(p[in_bin].mean())
        acc = float(y[in_bin].mean())
        rel += m / n * (conf - acc) ** 2
        res += m / n * (acc - pi) ** 2
    unc = pi * (1 - pi)
    return {'rel': round(rel, 6), 'res': round(res, 6), 'unc': round(unc, 6)}


def binary_metrics(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)  # float64：float32 下 1-1e-12 会退化为 1.0
    ece, mce, rows_ew = ece_equal_width(p, y)
    _, _, rows_ad = ece_adaptive(p, y)
    cw, cw_detail = classwise_ece_binary(p, y)
    return {
        'ece': round(float(ece), 6), 'mce': round(float(mce), 6),
        'brier': round(float(np.mean((p - y) ** 2)), 6),
        'nll': round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 6),
        'classwise_ece': round(cw, 6), 'classwise_ece_detail': {k: round(v, 6) for k, v in cw_detail.items()},
        'brier_decomp': brier_decomposition(p, y),
        'reliability_overall': rows_ew,
        'reliability_adaptive': rows_ad,
    }


def multiclass_metrics(probs, y, K):
    """probs (N,K) softmax 后概率；y (N,) int。ECE=top-1 置信度等宽分箱。"""
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc_vec = (pred == y).astype(float)
    ece, mce, rows = ece_equal_width(conf, acc_vec)
    cw_list = []
    for k in range(K):
        cw_list.append(ece_adaptive(probs[:, k], (y == k).astype(float))[0])
    onehot = np.eye(K)[y]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    p_clip = np.clip(probs, 1e-12, 1.0)
    nll = float(-np.mean(np.log(p_clip[np.arange(len(y)), y])))
    return {
        'ece': round(float(ece), 6), 'mce': round(float(mce), 6),
        'brier': round(brier, 6), 'nll': round(nll, 6),
        'classwise_ece': round(float(np.mean(cw_list)), 6),
        'classwise_ece_per_class': [round(float(c), 6) for c in cw_list],
        'acc': round(float(acc_vec.mean()), 6),
        'reliability_overall': rows,
    }


# ----------------------------------------------------------------------------
# 二分类校准方法（均在 val 上拟合；logits: (N,2)，取 z = logits[:,1]）
# ----------------------------------------------------------------------------
def _lbfgs_fit(params, closure, max_iter=200):
    opt = torch.optim.LBFGS(params, lr=0.05, max_iter=max_iter)

    def wrapped():
        opt.zero_grad()
        loss = closure()
        loss.backward()
        return loss
    opt.step(wrapped)


def fit_binary(z_val, y_val, method):
    """z_val (N,) torch，y_val (N,) torch float。返回 apply(z_test)->p_test 的函数 + 参数 dict。"""
    z = z_val.view(-1, 1).clone()
    y = y_val.view(-1, 1)

    if method == 'temperature':
        logT = nn.Parameter(torch.zeros(1))

        def cl():
            return F.binary_cross_entropy_with_logits(z / torch.exp(logT), y)
        _lbfgs_fit([logT], cl)
        T = float(torch.exp(logT).item())
        return (lambda zt: torch.sigmoid(zt / T).numpy()), {'T': round(T, 6)}

    if method == 'vector':
        w = nn.Parameter(torch.ones(1))
        b = nn.Parameter(torch.zeros(1))

        def cl():
            return F.binary_cross_entropy_with_logits(w * z + b, y)
        _lbfgs_fit([w, b], cl)
        wf, bf = float(w.item()), float(b.item())
        return (lambda zt: torch.sigmoid(wf * zt + bf).numpy()), {'w': round(wf, 6), 'b': round(bf, 6)}

    if method == 'platt':
        # Platt 经典式：p = sigmoid(a*z + b)，a>0 单调性约束（与 vector 的无约束 w 相区分）
        loga = nn.Parameter(torch.zeros(1))
        b = nn.Parameter(torch.zeros(1))

        def cl():
            return F.binary_cross_entropy_with_logits(torch.exp(loga) * z + b, y)
        _lbfgs_fit([loga, b], cl)
        af, bf = float(torch.exp(loga).item()), float(b.item())
        return (lambda zt: torch.sigmoid(af * zt + bf).numpy()), {'a': round(af, 6), 'b': round(bf, 6)}

    if method == 'isotonic':
        p_val = torch.sigmoid(z).numpy()
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(p_val, y_val.numpy())
        return (lambda zt: iso.predict(torch.sigmoid(zt).numpy())), {}

    raise ValueError(method)


def bootstrap_T_ci_binary(z_val, y_val, B=BOOTSTRAP_B):
    """预注册主判定原料：val 上 B 次 bootstrap 重采样重拟合 T，取百分位 CI。"""
    z_np, y_np = z_val.numpy(), y_val.numpy().astype(np.float32)
    n = len(z_np)
    rng = np.random.RandomState(2026)
    Ts = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        zb = torch.from_numpy(z_np[idx]).view(-1, 1)
        yb = torch.from_numpy(y_np[idx]).view(-1, 1)
        logT = nn.Parameter(torch.zeros(1))
        opt = torch.optim.LBFGS([logT], lr=0.05, max_iter=50)

        def cl():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(zb / torch.exp(logT), yb)
            loss.backward()
            return loss
        opt.step(cl)
        Ts.append(float(torch.exp(logT).item()))
    return [round(float(np.percentile(Ts, 2.5)), 6), round(float(np.percentile(Ts, 97.5)), 6)]


# ----------------------------------------------------------------------------
# 多分类校准方法（PAM，K=8；logits: (N,K)）
# ----------------------------------------------------------------------------
def fit_multiclass(logits_val, y_val, K, method):
    z = logits_val.clone()
    y = y_val.view(-1).long()

    if method == 'temperature':
        logT = nn.Parameter(torch.zeros(1))

        def cl():
            return F.cross_entropy(z / torch.exp(logT), y)
        _lbfgs_fit([logT], cl)
        T = float(torch.exp(logT).item())
        return (lambda zt: torch.softmax(zt / T, dim=-1).numpy()), {'T': round(T, 6)}

    if method == 'vector':
        w = nn.Parameter(torch.ones(K))
        b = nn.Parameter(torch.zeros(K))

        def cl():
            return F.cross_entropy(z * w + b, y)
        _lbfgs_fit([w, b], cl)
        wf, bf = w.detach().numpy(), b.detach().numpy()
        return (lambda zt: torch.softmax(zt * torch.from_numpy(wf).float() + torch.from_numpy(bf).float(),
                                         dim=-1).numpy()), {'w': [round(v, 6) for v in wf.tolist()],
                                                            'b': [round(v, 6) for v in bf.tolist()]}

    if method == 'platt':
        # OvR Platt：每类 a_k z_k + b_k 过 sigmoid 再重归一化
        params, appliers = {}, []
        for k in range(K):
            zk = z[:, k]
            yk = (y == k).float()
            loga = nn.Parameter(torch.zeros(1))
            b = nn.Parameter(torch.zeros(1))

            def cl(zk=zk, yk=yk, loga=loga, b=b):
                return F.binary_cross_entropy_with_logits(torch.exp(loga) * zk + b, yk)
            _lbfgs_fit([loga, b], cl)
            params['a{}'.format(k)] = round(float(torch.exp(loga).item()), 6)
            params['b{}'.format(k)] = round(float(b.item()), 6)
            appliers.append((float(torch.exp(loga).item()), float(b.item())))

        def apply_platt(zt):
            s = torch.zeros_like(zt)
            for k, (a, b) in enumerate(appliers):
                s[:, k] = torch.sigmoid(a * zt[:, k] + b)
            return (s / s.sum(dim=-1, keepdim=True)).numpy()
        return apply_platt, params

    if method == 'isotonic':
        p_val = torch.softmax(z, dim=-1).numpy()
        y_np = y.numpy()
        isos = []
        for k in range(K):
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            iso.fit(p_val[:, k], (y_np == k).astype(float))
            isos.append(iso)

        def apply_iso(zt):
            pt = torch.softmax(zt, dim=-1).numpy()
            out = np.stack([isos[k].predict(pt[:, k]) for k in range(K)], axis=1)
            s = out.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            return out / s
        return apply_iso, {'note': 'OvR isotonic + renormalization'}

    raise ValueError(method)


def bootstrap_T_ci_multiclass(logits_val, y_val, K, B=BOOTSTRAP_B):
    z_np = logits_val.numpy()
    y_np = y_val.numpy().astype(np.int64)
    n = len(z_np)
    rng = np.random.RandomState(2026)
    Ts = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        zb = torch.from_numpy(z_np[idx])
        yb = torch.from_numpy(y_np[idx])
        logT = nn.Parameter(torch.zeros(1))
        opt = torch.optim.LBFGS([logT], lr=0.05, max_iter=50)

        def cl():
            opt.zero_grad()
            loss = F.cross_entropy(zb / torch.exp(logT), yb)
            loss.backward()
            return loss
        opt.step(cl)
        Ts.append(float(torch.exp(logT).item()))
    return [round(float(np.percentile(Ts, 2.5)), 6), round(float(np.percentile(Ts, 97.5)), 6)]


# ----------------------------------------------------------------------------
# 模型构建与 logits 收集
# ----------------------------------------------------------------------------
def build_model(args):
    if args.backbone == 'coformer':
        if args.dataset == 'pam':
            from models.model_medical_wo_static import make_model
        else:
            from models.model_medical_attn_aggre import make_model
        return make_model(src_vocab=1, tgt_vocab=args.output_dim, N=args.num_layers,
                          d_model=args.d_model, d_ff=args.d_ff, h=args.heads, dropout=0.1,
                          num_agents=args.num_agents, num_neighbors=args.num_neighbors,
                          agent_encoding_dim=32, static_dim=args.static_dim)
    from models.backbones import make_backbone
    return make_backbone(args.backbone, num_sensors=args.num_agents,
                         static_dim=args.static_dim, out_dim=args.output_dim,
                         use_static=(args.dataset != 'pam'))


def collect_logits(model, loader, device, multiclass):
    model.eval()
    logits_all, gt_all = [], []
    with torch.no_grad():
        for data in loader:
            array = data['data'].to(device)
            time = data['time'].to(device)
            mask = data['mask'].to(device)
            gt = data['gt'].to(device)
            if multiclass:
                pred = model(array, time, mask)
            else:
                static = data['static'].to(device)
                pred = model(array, time, mask, static)
            logits_all.append(pred.cpu())
            gt_all.append(gt.cpu())
    return torch.cat(logits_all, 0), torch.cat(gt_all, 0)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['p19', 'p12', 'pam'])
    p.add_argument('--backbone', default='coformer', choices=['coformer', 'grud', 'talstm'])
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--split_path', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--methods', default='temperature,vector,platt,isotonic')
    # CoFormer 超参（与训练默认一致）
    p.add_argument('--num_layers', type=int, default=8)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--d_ff', type=int, default=256)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--num_neighbors', type=int, default=30)
    args = p.parse_args()

    defaults = {'p19': dict(num_agents=34, static_dim=6, output_dim=2),
                'p12': dict(num_agents=36, static_dim=9, output_dim=2),
                'pam': dict(num_agents=17, static_dim=6, output_dim=8)}[args.dataset]
    for k, v in defaults.items():
        setattr(args, k, v)
    multiclass = (args.dataset == 'pam')
    K = args.output_dim

    prepare_seed(1)
    device = torch.device('cuda', args.gpu)

    if multiclass:
        val_loader = DataLoader(PAMCalibDataset(args.data_root, args.split_path, part=1),
                                batch_size=10, num_workers=4,
                                collate_fn=seq_collate_irregular_wo_static)
        test_loader = DataLoader(PAMCalibDataset(args.data_root, args.split_path, part=2),
                                 batch_size=10, num_workers=4,
                                 collate_fn=seq_collate_irregular_wo_static)
    else:
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

    print('collecting val logits ...')
    logits_val, gt_val = collect_logits(model, val_loader, device, multiclass)
    print('collecting test logits ...')
    logits_test, gt_test = collect_logits(model, test_loader, device, multiclass)

    result = {
        'dataset': args.dataset, 'backbone': args.backbone,
        'split_path': args.split_path, 'ckpt': args.ckpt,
        'n_val': int(len(gt_val)), 'n_test': int(len(gt_test)),
        'methods': {},
    }

    if not multiclass:
        z_val, y_val = logits_val[:, 1], gt_val.view(-1).float()
        z_test, y_test = logits_test[:, 1], gt_test.view(-1).float()
        y_np = y_test.numpy()
        result['pos_rate_test'] = round(float(y_np.mean()), 6)
        p_before = torch.sigmoid(z_test).numpy()
        result['auroc'] = round(float(roc_auc_score(y_np, p_before) * 100), 4)
        result['auprc'] = round(float(average_precision_score(y_np, p_before) * 100), 4)
        base = binary_metrics(p_before, y_np)
        result['methods']['none'] = base
        # 预注册主判定原料：T 的 bootstrap CI
        t_ci = bootstrap_T_ci_binary(z_val, y_val)
        for m in [m.strip() for m in args.methods.split(',') if m.strip()]:
            apply_f, params = fit_binary(z_val, y_val, m)
            p_after = apply_f(z_test)
            met = binary_metrics(p_after, y_np)
            if m == 'temperature':
                params['T_ci95'] = t_ci
                params['T_ci_excludes_1'] = bool(t_ci[0] > 1.0 or t_ci[1] < 1.0)
            met['params'] = params
            met['ece_rel_drop_pct'] = round((base['ece'] - met['ece']) / base['ece'] * 100, 2)
            met['brier_improved'] = bool(met['brier'] < base['brier'])
            result['methods'][m] = met
    else:
        y_val = gt_val.view(-1).long()
        y_test = gt_test.view(-1).long()
        y_np = y_test.numpy()
        result['n_classes'] = K
        p_before = torch.softmax(logits_test, dim=-1).numpy()
        # 宏平均 OvR AUROC/AUPRC（判别能力参考）
        try:
            result['auroc_macro_ovr'] = round(float(roc_auc_score(
                y_np, p_before, multi_class='ovr', average='macro') * 100), 4)
            ap = [average_precision_score((y_np == k).astype(float), p_before[:, k]) for k in range(K)]
            result['auprc_macro_ovr'] = round(float(np.mean(ap) * 100), 4)
        except Exception as e:
            result['metric_note'] = 'auroc/auprc failed: {}'.format(e)
        base = multiclass_metrics(p_before, y_np, K)
        result['methods']['none'] = base
        t_ci = bootstrap_T_ci_multiclass(logits_val, y_val, K)
        for m in [m.strip() for m in args.methods.split(',') if m.strip()]:
            apply_f, params = fit_multiclass(logits_val, y_val, K, m)
            p_after = apply_f(logits_test)
            met = multiclass_metrics(p_after, y_np, K)
            if m == 'temperature':
                params['T_ci95'] = t_ci
                params['T_ci_excludes_1'] = bool(t_ci[0] > 1.0 or t_ci[1] < 1.0)
            met['params'] = params
            met['ece_rel_drop_pct'] = round((base['ece'] - met['ece']) / base['ece'] * 100, 2)
            met['brier_improved'] = bool(met['brier'] < base['brier'])
            result['methods'][m] = met

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    # 控制台摘要（不含 reliability 明细）
    brief = {k: v for k, v in result.items() if k != 'methods'}
    brief['methods'] = {m: {kk: vv for kk, vv in met.items()
                            if kk not in ('reliability_overall', 'reliability_adaptive',
                                          'reliability_adaptive_pos', 'classwise_ece_detail',
                                          'classwise_ece_per_class', 'brier_decomp')}
                        for m, met in result['methods'].items()}
    print(json.dumps(brief, indent=2))
    print('saved to', args.out)


if __name__ == '__main__':
    main()
