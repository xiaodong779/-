import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange


class LayerNormProxy(nn.Module):
    """
    LayerNorm for 5D feature:
        [B, C, T, H, W] -> [B, C, T, H, W]
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b t h w c')
        x = self.norm(x)
        x = rearrange(x, 'b t h w c -> b c t h w')
        return x


class ResNet_DeformAttention(nn.Module):
    """
    aDSTA core for ResNet feature.

    Input:
        x: [B, C, T, H, W]

    Output:
        out: [B, C, T, H, W]
    """
    def __init__(self, dim, heads, groups, kernel_size, stride, padding):
        super().__init__()

        assert dim % heads == 0, f'dim={dim} must be divisible by heads={heads}'
        assert dim % groups == 0, f'dim={dim} must be divisible by groups={groups}'
        assert heads % groups == 0, f'heads={heads} must be divisible by groups={groups}'

        self.dim = dim
        self.heads = heads
        self.head_channels = dim // heads
        self.scale = self.head_channels ** -0.5

        self.groups = groups
        self.group_channels = dim // groups
        self.group_heads = heads // groups
        self.factor = 2.0

        self.conv_offset = nn.Sequential(
            nn.Conv3d(
                in_channels=self.group_channels,
                out_channels=self.group_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=self.group_channels
            ),
            LayerNormProxy(self.group_channels),
            nn.GELU(),
            nn.Conv3d(
                in_channels=self.group_channels,
                out_channels=3,
                kernel_size=(1, 1, 1),
                bias=False
            )
        )

        self.proj_q = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1))
        self.proj_k = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1))
        self.proj_v = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1))
        self.proj_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1))

    @torch.no_grad()
    def _get_ref_points(self, T, H, W, B, dtype, device):
        ref_z, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, T - 0.5, T, dtype=dtype, device=device),
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
        )

        ref = torch.stack((ref_z, ref_y, ref_x), -1)  # [T, H, W, 3]

        ref[..., 0].div_(T).mul_(2).sub_(1)
        ref[..., 1].div_(H).mul_(2).sub_(1)
        ref[..., 2].div_(W).mul_(2).sub_(1)

        ref = ref[None, ...].expand(B * self.groups, -1, -1, -1, -1)
        return ref  # [B*G, T, H, W, 3]

    def forward(self, x):
        B, C, T, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)

        q_off = rearrange(
            q,
            'b (g c) t h w -> (b g) c t h w',
            g=self.groups,
            c=self.group_channels
        )

        offset = self.conv_offset(q_off)  # [B*G, 3, Tp, Hp, Wp]

        Tp, Hp, Wp = offset.size(2), offset.size(3), offset.size(4)
        n_sample = Tp * Hp * Wp

        offset_range = torch.tensor(
            [
                min(1.0, self.factor / Tp),
                min(1.0, self.factor / Hp),
                min(1.0, self.factor / Wp),
            ],
            device=device,
            dtype=dtype
        ).reshape(1, 3, 1, 1, 1)

        offset = offset.tanh().mul(offset_range)
        offset = rearrange(offset, 'b p t h w -> b t h w p')  # [B*G, Tp, Hp, Wp, 3]

        reference = self._get_ref_points(Tp, Hp, Wp, B, dtype, device)
        pos = offset + reference

        x_group = x.reshape(B * self.groups, self.group_channels, T, H, W)

        # For 5D input, grid_sample uses trilinear interpolation when mode='bilinear'.
        # grid order should be x, y, z.
        x_sampled = F.grid_sample(
            input=x_group,
            grid=pos[..., (2, 1, 0)],
            mode='bilinear',
            align_corners=True
        )  # [B*G, Cg, Tp, Hp, Wp]

        x_sampled = x_sampled.reshape(B, C, 1, 1, n_sample)

        q = q.reshape(B * self.heads, self.head_channels, T * H * W)
        k = self.proj_k(x_sampled).reshape(B * self.heads, self.head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.heads, self.head_channels, n_sample)

        attn = einsum('b c m, b c n -> b m n', q, k)
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=-1)

        out = einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, T, H, W)
        out = self.proj_out(out)

        return out


class SimpleMambaGate(nn.Module):
    """
    Pure PyTorch Mamba-like temporal state gate.

    这个类不依赖 mamba_ssm，适合 torch 1.11 + cu113 环境。

    Input:
        z: [B, T, C]

    Output:
        gate: [B, T, C]

    Structure:
        Linear projection
        Depth-wise temporal conv
        Forward temporal state scan
        Backward temporal state scan
        Selective gate
        Sigmoid temporal-channel weight
    """
    def __init__(
        self,
        dim,
        expand=2,
        d_conv=3,
        dropout=0.0
    ):
        super().__init__()

        self.dim = dim
        self.inner_dim = dim * expand

        self.in_proj = nn.Linear(dim, self.inner_dim * 2)

        self.dwconv = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.inner_dim
        )

        self.act = nn.SiLU()

        # learnable decay for forward/backward state
        self.f_decay = nn.Parameter(torch.zeros(self.inner_dim))
        self.b_decay = nn.Parameter(torch.zeros(self.inner_dim))

        self.f_in = nn.Linear(self.inner_dim, self.inner_dim)
        self.b_in = nn.Linear(self.inner_dim, self.inner_dim)

        self.out_norm = nn.LayerNorm(self.inner_dim * 2)

        self.out_proj = nn.Sequential(
            nn.Linear(self.inner_dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def _scan_forward(self, u, decay, in_proj):
        """
        u: [B, T, D]
        """
        B, T, D = u.shape

        a = torch.sigmoid(decay).view(1, D)
        h = torch.zeros(B, D, dtype=u.dtype, device=u.device)

        u_in = in_proj(u)

        outs = []
        for t in range(T):
            h = a * h + (1.0 - a) * u_in[:, t, :]
            outs.append(h)

        y = torch.stack(outs, dim=1)
        return y

    def forward(self, z):
        """
        z: [B, T, C]
        """
        u, gate = self.in_proj(z).chunk(2, dim=-1)

        # depth-wise temporal conv
        u = u.transpose(1, 2).contiguous()   # [B, D, T]
        u = self.dwconv(u)
        u = u.transpose(1, 2).contiguous()   # [B, T, D]
        u = self.act(u)

        gate = self.act(gate)

        # forward scan
        y_f = self._scan_forward(u, self.f_decay, self.f_in)

        # backward scan
        u_rev = torch.flip(u, dims=[1])
        y_b = self._scan_forward(u_rev, self.b_decay, self.b_in)
        y_b = torch.flip(y_b, dims=[1])

        y = torch.cat([y_f, y_b], dim=-1)    # [B, T, 2D]
        y = self.out_norm(y)

        # selective gate
        gate_2 = torch.cat([gate, gate], dim=-1)
        y = y * gate_2

        out_gate = self.out_proj(y)          # [B, T, C]
        return out_gate


class ADSTABlock(nn.Module):
    """
    ADSTA Block.

    对齐原作者 Adapter 形式:
        Down-Proj -> DPE -> Branch -> Up-Proj -> Residual

    原版 D2ST:
        aDSTA-S + aDSTA-T

    这里改成:
        aDSTA-S + Mamba-like Temporal Gate

    Input:
        x: [B*T, C, H, W]

    Output:
        y: [B*T, C, H, W]

    Formula:
        X_d = DownProj(X)
        X_p = X_d + DPE(X_d)

        L_d = X_p + aDSTA-S(LN(X_p))
        G_c = SimpleMambaGate(GAP_hw(DWConv_T(X_p)))

        Y = X + scale * UpProj(G_c * L_d)
    """
    def __init__(
        self,
        in_channels,
        num_segments=16,
        reduction=4,
        stage='layer3',
        sample_mode='density_244',
        mamba_expand=2,
        mamba_d_conv=3,
        dropout=0.0,
        heads=None,
        groups=None
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_segments = num_segments
        self.reduction = reduction
        self.stage = stage
        self.sample_mode = sample_mode

        assert in_channels % reduction == 0, \
            f'in_channels={in_channels} must be divisible by reduction={reduction}'

        self.mid_channels = in_channels // reduction
        Cr = self.mid_channels

        # Down-Proj: C -> Cr
        self.down = nn.Conv3d(
            in_channels=in_channels,
            out_channels=Cr,
            kernel_size=(1, 1, 1)
        )

        # DPE: Dynamic Position Embedding
        self.pos_embed = nn.Conv3d(
            in_channels=Cr,
            out_channels=Cr,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            groups=Cr
        )

        self.s_ln = LayerNormProxy(Cr)

        ksize, stride, h, g = self._build_adsta_setting(
            stage=stage,
            num_segments=num_segments,
            sample_mode=sample_mode,
            heads=heads,
            groups=groups
        )

        # Local aDSTA-S branch
        self.s_attn = ResNet_DeformAttention(
            dim=Cr,
            heads=h,
            groups=g,
            kernel_size=ksize,
            stride=stride,
            padding=(0, 0, 0)
        )

        # Temporal local cue before global scan
        self.temporal_dwconv = nn.Conv3d(
            in_channels=Cr,
            out_channels=Cr,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
            groups=Cr,
            bias=False
        )

        self.temporal_norm = nn.LayerNorm(Cr)

        # Mamba-like temporal gate
        self.mamba_gate = SimpleMambaGate(
            dim=Cr,
            expand=mamba_expand,
            d_conv=mamba_d_conv,
            dropout=dropout
        )

        # Up-Proj: Cr -> C
        self.up = nn.Conv3d(
            in_channels=Cr,
            out_channels=in_channels,
            kernel_size=(1, 1, 1)
        )

        self.gelu = nn.GELU()

        # up 初始化为 0，所以 scale 初始化为 1。
        # 初始时模块整体近似 identity，但 up 可以正常收到梯度。
        self.scale = nn.Parameter(torch.ones(1))

        self.init_weights()

    def _build_adsta_setting(self, stage, num_segments, sample_mode, heads, groups):
        """
        sample_mode:
            density_244:
                尽量让每个 stage 的采样密度接近 <2,4,4>。

            d2st:
                更接近原作者的空间 stride 设置。
        """
        t_stride = 8 if num_segments >= 16 else 4

        if sample_mode == 'd2st':
            if stage in ['layer1', 'res2', '56']:
                ksize = (4, 7, 7)
                stride = (t_stride, 7, 7)
                h = 1 if heads is None else heads
                g = 1 if groups is None else groups

            elif stage in ['layer2', 'res3', '28']:
                ksize = (4, 7, 7)
                stride = (t_stride, 7, 7)
                h = 2 if heads is None else heads
                g = 2 if groups is None else groups

            elif stage in ['layer3', 'res4', '14']:
                ksize = (4, 5, 5)
                stride = (t_stride, 3, 3)
                h = 4 if heads is None else heads
                g = 4 if groups is None else groups

            elif stage in ['layer4', 'res5', '7']:
                ksize = (4, 4, 4)
                stride = (t_stride, 3, 3)
                h = 8 if heads is None else heads
                g = 8 if groups is None else groups

            else:
                ksize = (4, 5, 5)
                stride = (t_stride, 3, 3)
                h = 4 if heads is None else heads
                g = 4 if groups is None else groups

        else:
            # density_244: each stage tries to produce about 2 x 4 x 4 reference samples
            if stage in ['layer1', 'res2', '56']:
                ksize = (4, 14, 14)
                stride = (t_stride, 14, 14)
                h = 1 if heads is None else heads
                g = 1 if groups is None else groups

            elif stage in ['layer2', 'res3', '28']:
                ksize = (4, 7, 7)
                stride = (t_stride, 7, 7)
                h = 2 if heads is None else heads
                g = 2 if groups is None else groups

            elif stage in ['layer3', 'res4', '14']:
                ksize = (4, 5, 5)
                stride = (t_stride, 3, 3)
                h = 4 if heads is None else heads
                g = 4 if groups is None else groups

            elif stage in ['layer4', 'res5', '7']:
                ksize = (4, 4, 4)
                stride = (t_stride, 1, 1)
                h = 8 if heads is None else heads
                g = 8 if groups is None else groups

            else:
                ksize = (4, 5, 5)
                stride = (t_stride, 3, 3)
                h = 4 if heads is None else heads
                g = 4 if groups is None else groups

        return ksize, stride, h, g

    def init_weights(self):
        # Zero-initialize Up-Proj so the block starts as identity.
        nn.init.constant_(self.up.weight, 0)
        if self.up.bias is not None:
            nn.init.constant_(self.up.bias, 0)

    def forward(self, x):
        """
        x: [B*T, C, H, W]
        """
        x_in = x

        BT, C, H, W = x.shape
        T = self.num_segments

        assert BT % T == 0, \
            f'BT={BT} must be divisible by num_segments={T}'

        # [B*T, C, H, W] -> [B, C, T, H, W]
        x = rearrange(x, '(b t) c h w -> b c t h w', t=T)

        # Down-Proj + DPE
        x_d = self.down(x)                 # [B, Cr, T, H, W]
        x_p = x_d + self.pos_embed(x_d)    # [B, Cr, T, H, W]

        # Branch 1: local aDSTA-S detail aggregation
        l_d = x_p + self.s_attn(self.s_ln(x_p))
        l_d = self.gelu(l_d)               # [B, Cr, T, H, W]

        # Branch 2: Mamba-like temporal channel gate
        z = self.temporal_dwconv(x_p)       # [B, Cr, T, H, W]
        z = z.mean(dim=(-1, -2))            # [B, Cr, T]
        z = z.permute(0, 2, 1).contiguous() # [B, T, Cr]
        z = self.temporal_norm(z)

        g_c = self.mamba_gate(z)            # [B, T, Cr]
        g_c = g_c.permute(0, 2, 1).contiguous()
        g_c = g_c.unsqueeze(-1).unsqueeze(-1)  # [B, Cr, T, 1, 1]

        # Fusion: first multiply in Cr, then Up-Proj back to C
        out = g_c * l_d                     # [B, Cr, T, H, W]
        out = self.up(out)                  # [B, C, T, H, W]
        out = self.scale * out

        out = rearrange(out, 'b c t h w -> (b t) c h w')

        return x_in + out
