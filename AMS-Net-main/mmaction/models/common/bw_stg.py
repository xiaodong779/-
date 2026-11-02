import math
import torch
import torch.nn as nn


def _safe_logit(p, eps=1e-6):
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def window_partition(x, window_size):
    """
    x: [B, C, T, H, W]

    return:
        windows: [B*T*num_windows, window_size*window_size, C]
        meta: 用于还原窗口
    """
    B, C, T, H, W = x.shape
    ws = window_size

    assert H % ws == 0 and W % ws == 0, \
        f'H,W must be divisible by window_size. Got H={H}, W={W}, ws={ws}. ' \
        f'当前第一版不做 padding，避免边缘 0 padding 干扰。'

    nH = H // ws
    nW = W // ws

    # [B, C, T, H, W] -> [B, T, H, W, C]
    x = x.permute(0, 2, 3, 4, 1).contiguous()

    # [B, T, nH, ws, nW, ws, C]
    x = x.view(B, T, nH, ws, nW, ws, C)

    # [B, T, nH, nW, ws, ws, C]
    x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()

    # [B*T*nH*nW, ws*ws, C]
    windows = x.view(B * T * nH * nW, ws * ws, C)

    meta = dict(
        B=B,
        C=C,
        T=T,
        H=H,
        W=W,
        nH=nH,
        nW=nW,
        window_size=ws,
    )
    return windows, meta


def window_reverse(windows, meta):
    """
    windows: [B*T*num_windows, window_size*window_size, C]

    return:
        x: [B, C, T, H, W]
    """
    B = meta['B']
    C = meta['C']
    T = meta['T']
    H = meta['H']
    W = meta['W']
    nH = meta['nH']
    nW = meta['nW']
    ws = meta['window_size']

    # [B, T, nH, nW, ws, ws, C]
    x = windows.view(B, T, nH, nW, ws, ws, C)

    # [B, T, nH, ws, nW, ws, C]
    x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()

    # [B, T, H, W, C]
    x = x.view(B, T, H, W, C)

    # [B, C, T, H, W]
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x


class WindowCrossAttention(nn.Module):
    """
    局部空间窗口 Cross-Attention。

    q_feat:  [B, Cb, T, H, W]
    kv_feat: [B, Cb, T, H, W]

    每一帧内部按 window_size×window_size 做注意力。
    不跨时间做全局注意力，减少计算量。
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        window_size=7,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()

        assert dim % num_heads == 0, \
            f'dim={dim} must be divisible by num_heads={num_heads}'

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_feat, kv_feat):
        assert q_feat.shape == kv_feat.shape, \
            f'q_feat={q_feat.shape}, kv_feat={kv_feat.shape}'

        q_win, meta = window_partition(q_feat, self.window_size)
        kv_win, _ = window_partition(kv_feat, self.window_size)

        # q_win:  [Nw, N, C]
        # kv_win: [Nw, N, C]
        Nw, N, C = q_win.shape

        q = self.q(q_win)
        kv = self.kv(kv_win)

        # q: [Nw, heads, N, head_dim]
        q = q.view(Nw, N, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3).contiguous()

        # kv: [2, Nw, heads, N, head_dim]
        kv = kv.view(Nw, N, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        # attention: [Nw, heads, N, N]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        # [Nw, heads, N, head_dim] -> [Nw, N, C]
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(Nw, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)

        out = window_reverse(out, meta)
        return out


class TemporalConvPre(nn.Module):
    """
    先对主分支 Y_i 做轻量 Temporal Conv。

    输入:
        x: [B, C, T, H, W]

    输出:
        x_t: [B, C, T, H, W]
    """

    def __init__(self, channels):
        super().__init__()

        self.dw_tconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(channels)
        self.act = nn.GELU()

        self.pw = nn.Conv3d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(channels)

        # 初始化为 0，使 TemporalConvPre 初始接近 identity
        nn.init.zeros_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x):
        out = self.dw_tconv(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.pw(out)
        out = self.bn2(out)
        return x + out


class BWSTGBlock(nn.Module):
    """
    Bilateral Window Spatial-Temporal Guidance Block.

    输入:
        y: 主分支时序特征 [B, C, T, H, W]
        v: Frozen CLIP / Visual 分支对齐后的特征 [B, C, T, H, W]

    输出:
        y_out: [B, C, T, H, W]

    当前第一版不使用 R_clip 可信度。
    """

    def __init__(
        self,
        channels,
        reduction=4,
        num_heads=8,
        window_size=7,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        fusion_init=(0.5, 0.5),
        gate_init_bias=-2.0,
        lambda_init=0.01,
        lambda_max=0.1,
    ):
        super().__init__()

        self.channels = channels
        self.hidden = max(channels // reduction, 64)
        self.window_size = window_size
        self.lambda_max = float(lambda_max)

        assert self.hidden % num_heads == 0, \
            f'hidden={self.hidden} must be divisible by num_heads={num_heads}'

        # 1. 主分支先做 Temporal Conv
        self.temporal_conv = TemporalConvPre(channels)

        # 2. Bottleneck down projection
        self.down_y = nn.Sequential(
            nn.Conv3d(channels, self.hidden, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.hidden),
            nn.GELU(),
        )

        self.down_v = nn.Sequential(
            nn.Conv3d(channels, self.hidden, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.hidden),
            nn.GELU(),
        )

        # 3. V -> Y cross-attention
        # Q 来自主分支，K/V 来自 visual 分支
        self.v_to_y = WindowCrossAttention(
            dim=self.hidden,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # 4. Y -> V cross-attention
        # Q 来自 visual 分支，K/V 来自主分支
        self.y_to_v = WindowCrossAttention(
            dim=self.hidden,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # 5. 双向结果融合权重，默认 0.5 / 0.5
        fusion_init = torch.tensor(fusion_init, dtype=torch.float32)
        fusion_init = fusion_init / fusion_init.sum()
        self.fusion_logits = nn.Parameter(torch.log(fusion_init))

        # 6. Gate Generation
        # 输入 [Y_b, F_i, Y_b * F_i]
        # 输出 G_i: [B, 1, T, H, W]
        self.gate_net = nn.Sequential(
            nn.Conv3d(self.hidden * 3, self.hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.hidden),
            nn.GELU(),
            nn.Conv3d(self.hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # gate 初始偏关闭，避免一开始增强过强
        nn.init.constant_(self.gate_net[-2].bias, gate_init_bias)

        # 7. Spatial Conv
        # 注意：这里处理的是 G_i * Y_b，而不是 G_i * F_i
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(
                self.hidden,
                self.hidden,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                groups=self.hidden,
                bias=False,
            ),
            nn.BatchNorm3d(self.hidden),
            nn.GELU(),
            nn.Conv3d(self.hidden, self.hidden, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.hidden),
            nn.GELU(),
        )

        # 8. Up projection
        self.up = nn.Sequential(
            nn.Conv3d(self.hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
        )

        # 9. lambda 限制到 [0, lambda_max]
        ratio = float(lambda_init) / max(float(lambda_max), 1e-6)
        self.lambda_raw = nn.Parameter(
            torch.tensor(_safe_logit(ratio), dtype=torch.float32)
        )

    def get_fusion_weight(self):
        return torch.softmax(self.fusion_logits, dim=0)

    def get_lambda(self):
        return self.lambda_max * torch.sigmoid(self.lambda_raw)

    def forward(self, y, v, return_aux=False):
        """
        y: [B, C, T, H, W]
        v: [B, C, T, H, W]
        """
        assert y.shape == v.shape, f'y={y.shape}, v={v.shape}'

        # Step 1: 主分支 Temporal Conv
        y_t = self.temporal_conv(y)

        # Step 2: Bottleneck
        y_b = self.down_y(y_t)
        v_b = self.down_v(v)

        # Step 3: 双向窗口 cross-attention
        e_v2y = self.v_to_y(q_feat=y_b, kv_feat=v_b)
        e_y2v = self.y_to_v(q_feat=v_b, kv_feat=y_b)

        # Step 4: Weighted Fusion
        w = self.get_fusion_weight()
        f = w[0] * e_v2y + w[1] * e_y2v

        # Step 5: Gate Generation
        gate = self.gate_net(torch.cat([y_b, f, y_b * f], dim=1))

        # Step 6: gate 乘到 Temporal Conv 后的主分支 bottleneck 特征上
        z = gate * y_b

        # Step 7: Spatial Conv + Up
        r = self.spatial_conv(z)
        r = self.up(r)

        # Step 8: λ 残差注入
        lam = self.get_lambda()
        out = y + lam * r

        if return_aux:
            aux = dict(
                fusion_weight=w.detach(),
                lambda_value=lam.detach(),
                gate_mean=gate.mean().detach(),
                gate_min=gate.min().detach(),
                gate_max=gate.max().detach(),
            )
            return out, aux

        return out
