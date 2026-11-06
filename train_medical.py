import os
import sys
import argparse
import time
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
from data.dataloader import medical_dataloader,medicalp12_dataloader,medicalpam_dataloader,seq_collate_irregular_wo_static,seq_collate_irregular
from models.model_medical_attn_aggre import make_model

from utils.torch import *
from utils.config import Config
from utils.utils import prepare_seed, print_log, AverageMeter, convert_secs2time, get_timestring
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score, precision_score, recall_score, f1_score,accuracy_score
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.utils.data.distributed import DistributedSampler

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class FocalBCEWithLogitsLoss(torch.nn.Module):
    """创新点3：Focal loss（二分类 BCE 版），用于极不平衡的 P19/P12。
    FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalBCEWithLogitsLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce * ((1 - p_t) ** self.gamma)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (at * loss).mean()


def append_metrics_csv(path, header, row):
    """逐 epoch 指标落盘（CSV，append），保证实验结果完整可追溯"""
    import csv
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def logging(cfg, epoch, total_epoch, iter, total_iter, ep, losses_str, log):
	print_log('{} | Epo: {:02d}/{:02d}, '
		'It: {:04d}/{:04d}, '
		'EP: {:s}, {}'
        .format(cfg, epoch, total_epoch, iter, total_iter, \
		convert_secs2time(ep), losses_str), log)


def test(epoch=-1):
    model.eval()
    all_loss = 0
    all_logits = []
    all_gt = []
    length = 0
    i = 0
    for data in tqdm(test_loader):
        # data = test_loader[22]
        array = data['data'].to(device)
        time = data['time'].to(device)
        static = data['static'].to(device)
        mask = data['mask'].to(device)
        gt = data['gt'].to(device)

        prediction = model(array,time,mask,static)
        # print(i,torch.isnan(prediction).any())
        i += 1
        batch_num = prediction.shape[0]
        logits = nn.functional.softmax(prediction,dim=-1)
        all_logits.append(logits)
        all_gt.append(gt)

        gt = torch.nn.functional.one_hot(gt.to(torch.int64),2).squeeze(1)

        loss = criterion(prediction,gt.float())
        """ optimize """
        all_loss += loss
        length += batch_num

    all_logits = torch.cat(all_logits,dim=0)
    all_gt = torch.cat(all_gt,dim=0)

    auc = roc_auc_score(all_gt.cpu(), all_logits[:, 1].cpu())
    aupr = average_precision_score(all_gt.cpu(), all_logits[:, 1].cpu())
    print('auc',auc,'aupr',aupr,'loss',all_loss/length)
    if epoch >= 0: # 指标落盘
        append_metrics_csv(os.path.join(cfg.result_dir, 'metrics.csv'),
                           ['epoch','auroc','auprc','loss'],
                           [epoch, float(auc)*100, float(aupr)*100, float(all_loss.item()/length)])
    return all_loss/length

def test_multiclass(epoch=-1):
    model.eval()
    all_loss = 0
    all_logits = []
    all_gt = []
    length = 0
    for data in tqdm(test_loader):
        # data = test_loader[22]
        array = data['data'].to(device)
        time = data['time'].to(device)
        # static = data['static'].to(device)
        mask = data['mask'].to(device)
        gt = data['gt'].to(device)

        prediction = model(array,time,mask)
        # print(i,torch.isnan(prediction).any())
        batch_num = prediction.shape[0]
        logits = nn.functional.softmax(prediction,dim=-1)
        all_logits.append(logits)
        all_gt.append(gt)
        gt = gt.squeeze(1)
        # gt = torch.nn.functional.one_hot(gt.to(torch.int64),8).squeeze(1)
        
        loss = criterion(prediction,gt.long())
        """ optimize """
        all_loss += loss
        length += batch_num
        


    all_logits = torch.cat(all_logits,dim=0)
    all_gt = torch.cat(all_gt,dim=0)
    
    all_pred = np.argmax(np.array(all_logits.cpu()),axis=1)

    # acc = np.sum(all_pred.ravel() == all_gt.ravel()) / all_pred.shape[0] 
    acc = accuracy_score(all_gt.cpu(), all_pred )
    precision = precision_score(all_gt.cpu(), all_pred, average='macro', )
    recall = recall_score(all_gt.cpu(), all_pred, average='macro', )
    F1 = f1_score(all_gt.cpu(), all_pred, average='macro', )
    print('acc',acc,'precision',precision,'recall',recall,'F1',F1)
    if epoch >= 0: # 指标落盘
        append_metrics_csv(os.path.join(cfg.result_dir, 'metrics.csv'),
                           ['epoch','acc','precision','recall','f1','loss'],
                           [epoch, float(acc)*100, float(precision)*100, float(recall)*100, float(F1)*100, float(all_loss.item()/length)])
    return all_loss/length



def train(epoch):

    model.train()

    total_iter_num = len(train_loader)
    iter_num = 0
    for data in train_loader:
        array = data['data'].to(device)
        time = data['time'].to(device)
        static = data['static'].to(device)#####
        mask = data['mask'].to(device)
        gt = data['gt'].to(device)
        # gt = gt.squeeze(1)
        prediction = model(array,time,mask,static)
        gt = torch.nn.functional.one_hot(gt.to(torch.int64),2).squeeze(1)#####
        loss = criterion(prediction,gt.float())
        """ optimize (支持梯度累积：等效大 batch，用于 PAM 基线修复) """
        loss = loss / grad_accum
        loss.backward()
        iter_num += 1
        if iter_num % grad_accum == 0 or iter_num == total_iter_num:
            optimizer.step()
            optimizer.zero_grad()

        if iter_num % cfg.print_freq == 0 :

            print('epoch',epoch,'iter',iter_num,'/',total_iter_num,'loss:',loss.item()*grad_accum)

    scheduler.step()

def train_multiclass(epoch):

    model.train()

    total_iter_num = len(train_loader)
    iter_num = 0
    for data in train_loader:
        array = data['data'].to(device)
        time = data['time'].to(device)
        # static = data['static'].to(device)#####
        mask = data['mask'].to(device)
        gt = data['gt'].to(device)
        gt = gt.squeeze(1)
        prediction = model(array,time,mask)
        loss = criterion(prediction,gt.long())
        """ optimize (支持梯度累积：等效大 batch，用于 PAM 基线修复) """
        loss = loss / grad_accum
        loss.backward()
        iter_num += 1
        if iter_num % grad_accum == 0 or iter_num == total_iter_num:
            optimizer.step()
            optimizer.zero_grad()

        if iter_num % cfg.print_freq == 0 :

            print('epoch',epoch,'iter',iter_num,'/',total_iter_num,'loss:',loss.item()*grad_accum)

    scheduler.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='trans_medical')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=2)

    parser.add_argument('--input_dim',type=int,default=1)
    parser.add_argument('--output_dim',type=int,default=2)
    parser.add_argument('--num_layers',type=int,default=8)
    parser.add_argument('--heads',type=int,default=8)
    parser.add_argument('--d_model',type=int,default=256)
    parser.add_argument('--d_ff',type=int,default=256)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--num_agents',type=int,default=34)
    parser.add_argument('--num_neighbors',type=int,default=30)
    parser.add_argument('--agent_encoding_dim',type=int,default=32)
    parser.add_argument('--static_dim',type=int,default=6)

    parser.add_argument('--split_path') # specify the split index
    parser.add_argument('--data_root') # specify the dataset type ('p19','p12','pam')
    parser.add_argument('--dataset') # specify the dataset type ('p19','p12','pam')

    # ===== 创新点实验开关（默认全关 = 与基线行为一致）=====
    parser.add_argument('--dt_bias', action='store_true', default=False,
                        help='创新点1（总开关）：相对时间差注意力偏置（intra MHA + inter GATv2）')
    parser.add_argument('--dt_bias_intra', action='store_true', default=False,
                        help='创新点1a：仅 intra-variate MHA 加 Δt 偏置（诊断用）')
    parser.add_argument('--dt_bias_inter', action='store_true', default=False,
                        help='创新点1b：仅 inter-variate GATv2 加边时间衰减（诊断用）')
    parser.add_argument('--dt_intra_t2v', action='store_true', default=False,
                        help='创新点1第2轮：intra偏置升级为Time2Vec周期参数化（自动启用intra偏置）')
    parser.add_argument('--missing_aware', action='store_true', default=False,
                        help='创新点2：临床缺失感知嵌入（gap+density 残差注入）')
    parser.add_argument('--missing_aware_v2', action='store_true', default=False,
                        help='创新点2特征升级：gap+density+间隔方差+末次观测距窗口末端时长（自动启用missing_aware）')
    parser.add_argument('--focal', action='store_true', default=False,
                        help='创新点3：Focal loss 替换 BCE（仅 p19/p12 生效）')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='梯度累积步数，等效 batch = batch_size*grad_accum（PAM 用 21 模拟 126）')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子覆盖（默认用 cfg 的 seed=1；多种子显著性分析用）')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='覆盖 cfg 中的 batch_size（0=沿用 cfg）')
    parser.add_argument('--exp_name', default='',
                        help='实验标签，写入 args 记录，便于追溯')
    parser.add_argument('--backbone', default='coformer', choices=['coformer', 'grud', 'talstm'],
                        help='骨干选择：默认 coformer（原基线，行为零变化）；grud/talstm 为新增骨干')

    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg, args.tmp)
    if args.seed is not None:
        cfg.seed = args.seed  # 多种子覆盖（默认 None = 用 cfg 的 seed=1，与基线一致）
    prepare_seed(cfg.seed)
    torch.set_default_dtype(torch.float32)

    if args.batch_size > 0:
        cfg.batch_size = args.batch_size

    device=torch.device("cuda",args.gpu)
    print('device',device)


    time_str = get_timestring()
    log = open(os.path.join(cfg.log_dir, 'log.txt'), 'a+')

    print_log("time str: {}".format(time_str), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)
    print_log("exp_name: {} | backbone={} | dt_bias={} dt_bias_intra={} dt_bias_inter={} dt_intra_t2v={} missing_aware={} miss_v2={} focal={} gamma={} alpha={} grad_accum={} batch_size={}".format(
        args.exp_name, args.backbone, args.dt_bias, args.dt_bias_intra, args.dt_bias_inter, args.dt_intra_t2v, args.missing_aware, args.missing_aware_v2, args.focal, args.focal_gamma, args.focal_alpha,
        args.grad_accum, cfg.batch_size), log)
    with open(os.path.join(cfg.log_dir, 'args.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write('{}: {}\n'.format(k, v))
        f.write('batch_size(effective): {}\n'.format(cfg.batch_size))

    print('data prepare')
    """ data """
    if args.dataset == 'p19':
        train_dset = medical_dataloader(
            root = args.data_root,
            split_path = args.split_path,
            training=True)
        
        test_dset = medical_dataloader(
            root = args.data_root,
            split_path = args.split_path,
            training=False)
    
    elif args.dataset == 'p12':
        train_dset = medicalp12_dataloader(
            root = args.data_root,
            split_path = args.split_path,
            training=True)
        
        test_dset = medicalp12_dataloader(
            root = args.data_root,
            split_path = args.split_path,
            training=False)

    elif args.dataset == 'pam':
        train_dset = medicalpam_dataloader(
            root = args.data_root,
            split_path = args.split_path,
            training=True)
        
        test_dset = medicalpam_dataloader(
            root = args.data_root,
            split_path = args.split_path,
            training=False)
    
    
    
    collate_fn_choice = seq_collate_irregular_wo_static if args.dataset == 'pam' else seq_collate_irregular

    train_loader = DataLoader(
        train_dset,
        batch_size=cfg.batch_size,
        num_workers=4,
        collate_fn=collate_fn_choice,
        pin_memory=True)

    test_loader = DataLoader(
        test_dset,
        batch_size=10,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_choice)

    print('model prepare')
    """ model """
    if args.backbone == 'coformer':
        if args.dataset == 'pam':
            from models.model_medical_wo_static import make_model
        model = make_model(src_vocab=args.input_dim,tgt_vocab=args.output_dim,N=args.num_layers, d_model=args.d_model, d_ff=args.d_ff, h=args.heads, dropout=args.dropout,
                           num_agents=args.num_agents,num_neighbors=args.num_neighbors,agent_encoding_dim=args.agent_encoding_dim,static_dim=args.static_dim,
                           dt_bias=args.dt_bias, missing_aware=(args.missing_aware or args.missing_aware_v2),
                           dt_bias_intra=(args.dt_bias_intra or args.dt_intra_t2v), dt_bias_inter=args.dt_bias_inter,
                           dt_mode=('t2v' if args.dt_intra_t2v else 'decay'),
                           miss_v2=args.missing_aware_v2)
    else:
        # 新增骨干（GRU-D / time-aware LSTM）：接口与 CoFormer 完全一致
        from models.backbones import make_backbone
        model = make_backbone(args.backbone, num_sensors=args.num_agents,
                              static_dim=args.static_dim, out_dim=args.output_dim,
                              use_static=(args.dataset != 'pam'))
    if args.dataset == 'pam':
        criterion = torch.nn.CrossEntropyLoss().to(device)
    elif args.focal:
        criterion = FocalBCEWithLogitsLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
        print_log('using FocalBCEWithLogitsLoss alpha={} gamma={}'.format(args.focal_alpha, args.focal_gamma), log)
    else:
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    grad_accum = max(1, args.grad_accum)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler_type = cfg.get('lr_scheduler', 'linear')
    if scheduler_type == 'linear':
        scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.lr_fix_epochs, nepoch=cfg.num_epochs)
    elif scheduler_type == 'step':
        scheduler = get_scheduler(optimizer, policy='step', decay_step=cfg.decay_step, decay_gamma=cfg.decay_gamma)
    else:
        raise ValueError('unknown scheduler type!')

    if args.start_epoch > 0:
        cp_path = cfg.model_path % args.start_epoch
        print_log(f'loading model from checkpoint: {cp_path}', log)
        model_cp = torch.load(cp_path, map_location='cpu')
        model.load_state_dict(model_cp['model_dict'])
        if 'opt_dict' in model_cp:
            optimizer.load_state_dict(model_cp['opt_dict'])
        if 'scheduler_dict' in model_cp:
            scheduler.load_state_dict(model_cp['scheduler_dict'])

    """ start training """
    model.to(device)
    
    model.train()
    best_loss = 100
    for epoch in range(args.start_epoch, cfg.num_epochs):

        with torch.no_grad():
            if args.dataset == 'pam':
                current_loss = test_multiclass(epoch)
            else:
                current_loss = test(epoch)
        if current_loss<best_loss:
            best_loss = current_loss
        print('best loss:',best_loss)
        if args.dataset == 'pam':
            train_multiclass(epoch)
        else:
            train(epoch)
        

        print('start new epoch')
        # """ save model """
        if (cfg.model_save_freq > 0 and (epoch + 1) % cfg.model_save_freq == 0) or epoch == 0:
            cp_path = cfg.model_path % (epoch + 1)
            model_cp = {'model_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(), 'scheduler_dict': scheduler.state_dict(), 'epoch': epoch + 1}
            torch.save(model_cp, cp_path)

    """ 实验收尾：读取 metrics.csv 汇总最佳指标，写 summary.json """
    import json
    try:
        import csv as _csv
        with open(os.path.join(cfg.result_dir, 'metrics.csv')) as f:
            rows = list(_csv.DictReader(f))
        if rows:
            if args.dataset == 'pam':
                best_f1 = max(rows, key=lambda r: float(r['f1']))
                best_acc = max(rows, key=lambda r: float(r['acc']))
                summary = {'dataset': args.dataset, 'split_path': args.split_path, 'seed': cfg.seed,
                           'exp_name': args.exp_name, 'backbone': args.backbone,
                           'dt_bias': args.dt_bias, 'dt_bias_intra': (args.dt_bias_intra or args.dt_intra_t2v), 'dt_bias_inter': args.dt_bias_inter, 'dt_intra_t2v': args.dt_intra_t2v,
                           'missing_aware': (args.missing_aware or args.missing_aware_v2), 'miss_v2': args.missing_aware_v2, 'focal': args.focal,
                           'best_acc': float(best_acc['acc']), 'best_acc_epoch': int(best_acc['epoch']),
                           'best_f1': float(best_f1['f1']), 'best_f1_epoch': int(best_f1['epoch']),
                           'best_precision': float(best_f1['precision']), 'best_recall': float(best_f1['recall'])}
            else:
                best_auc = max(rows, key=lambda r: float(r['auroc']))
                best_aupr = max(rows, key=lambda r: float(r['auprc']))
                summary = {'dataset': args.dataset, 'split_path': args.split_path, 'seed': cfg.seed,
                           'exp_name': args.exp_name, 'backbone': args.backbone,
                           'dt_bias': args.dt_bias, 'dt_bias_intra': (args.dt_bias_intra or args.dt_intra_t2v), 'dt_bias_inter': args.dt_bias_inter, 'dt_intra_t2v': args.dt_intra_t2v,
                           'missing_aware': (args.missing_aware or args.missing_aware_v2), 'miss_v2': args.missing_aware_v2, 'focal': args.focal,
                           'best_auroc': float(best_auc['auroc']), 'best_auroc_epoch': int(best_auc['epoch']),
                           'best_auprc': float(best_aupr['auprc']), 'best_auprc_epoch': int(best_aupr['epoch']),
                           'auroc_at_best_auprc': float(best_aupr['auroc']), 'auprc_at_best_auroc': float(best_auc['auprc'])}
            with open(os.path.join(cfg.result_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            print_log('SUMMARY: {}'.format(json.dumps(summary)), log)
            print('SUMMARY:', json.dumps(summary))
    except Exception as e:
        print_log('summary write failed: {}'.format(e), log)








