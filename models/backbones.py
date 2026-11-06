"""新增骨干：GRU-D 与 time-aware LSTM（T-LSTM 风格）。

数据格式约定（与 CoFormer 基线 / train_medical.py 完全一致）：
  data   (B, S, T)：传感器 × 槽位；传感器 j 的有效槽位为 [0, mask_j)，padding = 1e10
  time   (B, S, T)：每个观测的时间戳；padding = -1e10
  mask   (B, S)   ：每个传感器的观测数（int）
  static (B, D)   ：人口学静态特征（p19/p12；PAM 无，forward 仅传 3 参）

适配协议（写入论文 Method / 附录，两骨干同一条目）：
  以"槽位索引"为统一时间步网格（slot-indexed unified grid）：第 t 步的特征向量
  = 各传感器各自序列的第 t 个观测；某传感器在步 t 是否有观测由 t < mask_j 决定。
  时间衰减量 Δt 全部由 time 张量中各传感器自身时间戳导出（GRU-D 的 per-feature
  δ 与 T-LSTM 的全局步进 Δt），不做等间隔假设；无任何观测的槽位步不更新隐状态。

  值标准化：SensorNorm（per-sensor running mean/var，仅统计有效观测位置，训练期
  更新 buffer、评估期固定）——对应 CoFormer 每传感器 Linear(1,d) 自带仿射能力。
  缺失值：GRU-D 按经典公式用衰减 γ_x 与可学习经验均值 x̄ 插补；T-LSTM 用 x̄ 插补；
  两者的输入均拼接观测指示向量 m（输入维度 2S）。
"""
import torch
import torch.nn as nn


class SensorNorm(nn.Module):
    """per-sensor 运行均值/方差标准化（BatchNorm 语义，作用于 (B,S,T) 的有效位置）。"""

    def __init__(self, num_sensors, momentum=0.1, eps=1e-5):
        super().__init__()
        self.num_sensors = num_sensors
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(num_sensors))
        self.register_buffer('running_var', torch.ones(num_sensors))
        self.register_buffer('num_batches', torch.zeros(1))

    def forward(self, x, obs):
        # x (B,S,T)：无效位已置零；obs (B,S,T) bool
        if self.training and obs.any():
            with torch.no_grad():
                mom = self.momentum if self.num_batches.item() > 0 else 1.0  # 首批直接采用
                for s in range(self.num_sensors):
                    o = obs[:, s, :]
                    if o.any():
                        vals = x[:, s, :][o]
                        m = vals.mean()
                        v = vals.var(unbiased=False) if vals.numel() > 1 else torch.zeros(())
                        self.running_mean[s] = (1 - mom) * self.running_mean[s] + mom * m
                        self.running_var[s] = (1 - mom) * self.running_var[s] + mom * v
                self.num_batches += 1
        xh = (x - self.running_mean.view(1, -1, 1)) / torch.sqrt(
            self.running_var.view(1, -1, 1) + self.eps)
        return xh.clamp(-10.0, 10.0)


def _prepare(data, time, mask, norm_module):
    """公共预处理：观测掩码、标准化、统一时间网格 G、per-feature 末次观测时刻。"""
    B, S, T = data.shape
    device = data.device
    t_arange = torch.arange(T, device=device).view(1, 1, T)
    obs = (t_arange < mask.unsqueeze(-1)) & (data < 1e9) & (time > -1e9)  # (B,S,T)
    x = torch.where(obs, data, torch.zeros_like(data)).clamp(-1e4, 1e4)
    x = norm_module(x, obs)
    tau_neg = torch.where(obs, time, torch.full_like(time, -1e10))         # padding→-1e10
    # 全局单调时间 G_t：每槽位各传感器时间戳的最大值，再沿 T 做 cummax
    step_time = tau_neg.max(dim=1).values                                  # (B,T)
    G = torch.cummax(step_time, dim=1).values.clamp(min=0)                 # (B,T) 单调不减
    # per-feature 含当前步的末次观测时刻（沿 T cummax；从未观测则保持 -1e10）
    tau_cum = torch.cummax(tau_neg, dim=2).values                          # (B,S,T)
    # δ = 当前全局时间 − 该特征末次观测时刻；观测位的 δ = G_t − τ_{t,j} ≥ 0
    delta = (G.unsqueeze(1) - tau_cum).clamp(0.0, 1e3)                     # (B,S,T)
    return obs, x, G, delta


class GRUDBackbone(nn.Module):
    """GRU-D（Che et al., 2018）的槽位网格适配版。

    经典公式：x̃ = m ⊙ x + (1−m) ⊙ (γ_x ⊙ x̄)，h̃ = γ_h ⊙ h_{t−1}，
    γ = exp(−relu(w·δ + b))；γ_x per-feature，γ_h per-hidden（由观测位的 δ 均值驱动）。
    """

    def __init__(self, num_sensors, static_dim, out_dim, hidden=256, use_static=True, dropout=0.1):
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden = hidden
        self.use_static = use_static
        self.norm = SensorNorm(num_sensors)
        self.x_mean = nn.Parameter(torch.zeros(num_sensors))               # 经验均值 x̄（标准化域）
        self.w_delta = nn.Parameter(torch.zeros(num_sensors))              # 初始 γ_x=1（无衰减），训练中学习
        self.b_delta = nn.Parameter(torch.zeros(num_sensors))
        self.w_h = nn.Parameter(torch.zeros(hidden))                       # 初始 γ_h=1
        self.b_h = nn.Parameter(torch.zeros(hidden))
        self.cell = nn.GRUCell(2 * num_sensors, hidden)
        if use_static:
            self.static_mlp = nn.Sequential(
                nn.Linear(static_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + (hidden if use_static else 0), out_dim))
        for m in [self.head, self.static_mlp if use_static else None]:
            if m is not None:
                for p in m.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)

    def forward(self, data, time, mask, static=None):
        B, S, T = data.shape
        obs, x, G, delta = _prepare(data, time, mask, self.norm)
        gamma_x = torch.exp(-torch.relu(
            delta * self.w_delta.view(1, S, 1) + self.b_delta.view(1, S, 1)))   # (B,S,T) ∈ (0,1]
        xbar = self.x_mean.view(1, S, 1).expand_as(x)
        x_tilde = torch.where(obs, x, gamma_x * xbar)                           # 缺失位：衰减插补
        # 观测位的 δ 均值（仅统计观测传感器，驱动隐状态衰减）
        obs_cnt = obs.sum(dim=1).clamp(min=1.0)                                 # (B,T)
        delta_mean = (delta * obs).sum(dim=1) / obs_cnt                         # (B,T)
        gamma_h = torch.exp(-torch.relu(
            delta_mean.unsqueeze(-1) * self.w_h + self.b_h))                    # (B,T,H)
        inp = torch.cat([x_tilde.transpose(1, 2), obs.transpose(1, 2).float()], dim=-1)  # (B,T,2S)
        active = obs.transpose(1, 2).sum(dim=-1) > 0                            # (B,T)
        h = data.new_zeros(B, self.hidden)
        for t in range(T):
            h_dec = gamma_h[:, t] * h
            h_new = self.cell(inp[:, t], h_dec)
            h = torch.where(active[:, t].unsqueeze(-1), h_new, h)               # 空槽位不更新
        feats = h
        if self.use_static:
            feats = torch.cat([h, self.static_mlp(static)], dim=-1)
        return self.head(feats)                                                 # (B,out_dim) logits


class TALSTMBackbone(nn.Module):
    """time-aware LSTM（T-LSTM, Baytas et al., 2017 风格）：Δt 驱动的 h/c 指数衰减。

    全局步进 Δt_t = G_t − G_{t−1}（首步取 G_0）；decay = exp(−relu(w·Δt + b))
    同时作用于 h 与 c，再进入标准 LSTMCell。缺失值用可学习 x̄ 插补并拼接 m。
    """

    def __init__(self, num_sensors, static_dim, out_dim, hidden=256, use_static=True, dropout=0.1):
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden = hidden
        self.use_static = use_static
        self.norm = SensorNorm(num_sensors)
        self.x_mean = nn.Parameter(torch.zeros(num_sensors))
        self.w_dt = nn.Parameter(torch.zeros(hidden))                      # 初始 decay=1
        self.b_dt = nn.Parameter(torch.zeros(hidden))
        self.cell = nn.LSTMCell(2 * num_sensors, hidden)
        if use_static:
            self.static_mlp = nn.Sequential(
                nn.Linear(static_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + (hidden if use_static else 0), out_dim))
        for m in [self.head, self.static_mlp if use_static else None]:
            if m is not None:
                for p in m.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)

    def forward(self, data, time, mask, static=None):
        B, S, T = data.shape
        obs, x, G, _ = _prepare(data, time, mask, self.norm)
        xbar = self.x_mean.view(1, S, 1).expand_as(x)
        x_tilde = torch.where(obs, x, xbar.expand_as(x))                   # 缺失位：经验均值插补
        # 全局步进 Δt
        dG = G.clone()
        dG[:, 1:] = (G[:, 1:] - G[:, :-1]).clamp(min=0)
        decay = torch.exp(-torch.relu(
            dG.unsqueeze(-1) * self.w_dt + self.b_dt))                     # (B,T,H) ∈ (0,1]
        inp = torch.cat([x_tilde.transpose(1, 2), obs.transpose(1, 2).float()], dim=-1)
        active = obs.transpose(1, 2).sum(dim=-1) > 0
        h = data.new_zeros(B, self.hidden)
        c = data.new_zeros(B, self.hidden)
        for t in range(T):
            h_dec = decay[:, t] * h
            c_dec = decay[:, t] * c
            h_new, c_new = self.cell(inp[:, t], (h_dec, c_dec))
            keep = active[:, t].unsqueeze(-1)
            h = torch.where(keep, h_new, h)
            c = torch.where(keep, c_new, c)
        feats = h
        if self.use_static:
            feats = torch.cat([h, self.static_mlp(static)], dim=-1)
        return self.head(feats)


def make_backbone(name, num_sensors, static_dim, out_dim, hidden=256, use_static=True, dropout=0.1):
    """骨干工厂：train_medical.py / calibrate_suite.py 统一入口。"""
    if name == 'grud':
        return GRUDBackbone(num_sensors, static_dim, out_dim, hidden, use_static, dropout)
    if name == 'talstm':
        return TALSTMBackbone(num_sensors, static_dim, out_dim, hidden, use_static, dropout)
    raise ValueError('unknown backbone: {}'.format(name))
