import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out

class GatedConvFFN(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()

        hidden_dim = dim * expansion

        self.project_in = nn.Conv2d(
            dim,
            hidden_dim * 2,
            kernel_size=1
        )

        self.depthwise = nn.Conv2d(
            hidden_dim * 2,
            hidden_dim * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_dim * 2
        )

        self.project_out = nn.Conv2d(
            hidden_dim,
            dim,
            kernel_size=1
        )

    def forward(self, x, height, width):
        # x: [B, H*W, C]
        x = rearrange(
            x,
            "b (h w) c -> b c h w",
            h=height,
            w=width
        )

        x = self.depthwise(
            self.project_in(x)
        )

        x1, x2 = x.chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)

        return rearrange(
            x,
            "b c h w -> b (h w) c"
        )


class DiTStyleBlock(nn.Module):
    """
    Deterministic DiT-inspired transformer block.

    It uses adaptive LayerNorm modulation, but the modulation
    comes from the current bottleneck feature rather than a
    diffusion timestep.
    """

    def __init__(
        self,
        dim,
        num_heads=4,
        mlp_ratio=4.0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            dim,
            elementwise_affine=False
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(
            dim,
            elementwise_affine=False
        )

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = GatedConvFFN(
            dim=dim,
            expansion=2
        )

        # Produces:
        # shift_attn, scale_attn, gate_attn,
        # shift_mlp, scale_mlp, gate_mlp
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )

        # Start close to identity for stable insertion into MST++.
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, condition,height,width):
        """
        x:         [B, N, D]
        condition: [B, D]
        """

        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp
        ) = self.modulation(condition).chunk(
            6,
            dim=-1
        )

        # Attention branch
        h = self.norm1(x)

        h = (
            h
            * (1.0 + scale_attn[:, None, :])
            + shift_attn[:, None, :]
        )

        attention_output, _ = self.attention(
            h,
            h,
            h,
            need_weights=False
        )

        x = (
            x
            + gate_attn[:, None, :]
            * attention_output
        )

        # MLP branch
        h = self.norm2(x)

        h = (
            h
            * (1.0 + scale_mlp[:, None, :])
            + shift_mlp[:, None, :]
        )

        mlp_output = self.mlp(
            h,
            height=height,
            width=width
        )
        
        x = (
            x
            + gate_mlp[:, None, :]
            * mlp_output
        )
        return x

class DiTStyleBottleneck(nn.Module):
    def __init__(
        self,
        in_channels=124,
        hidden_dim=124,
        patch_size=4,
        depth=2,
        num_heads=4
    ):
        super().__init__()

        self.patch_size = patch_size

        # Preserve local information before patchification.
        self.local_in = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=1,
                bias=False
            )
        )

        # 124 channels are projected into hidden_dim patch tokens.
        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.blocks = nn.ModuleList([
            DiTStyleBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=4.0
            )
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        self.unpatchify = nn.ConvTranspose2d(
            hidden_dim,
            in_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.local_out = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=1,
                bias=False
            )
        )

        # Keeps the replacement close to the original feature path
        # at the beginning of training.
        self.residual_scale = nn.Parameter(
            torch.tensor(0.1)
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        output: [B, C, H, W]
        """

        residual = x
        original_h, original_w = x.shape[-2:]

        x = self.local_in(x)

        # Pad when bottleneck dimensions are not divisible
        # by patch_size.
        pad_h = (
            self.patch_size
            - original_h % self.patch_size
        ) % self.patch_size

        pad_w = (
            self.patch_size
            - original_w % self.patch_size
        ) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                (0, pad_w, 0, pad_h),
                mode="reflect"
            )

        x = self.patch_embed(x)

        batch_size, hidden_dim, hp, wp = x.shape

        tokens = rearrange(
            x,
            "b c h w -> b (h w) c"
        )

        # Global bottleneck condition used for adaLN.
        condition = tokens.mean(dim=1)

        for block in self.blocks:
            tokens = block(
                tokens,
                condition,
                height=hp,
                width=wp
            )

        tokens = self.final_norm(tokens)

        x = rearrange(
            tokens,
            "b (h w) c -> b c h w",
            h=hp,
            w=wp
        )

        x = self.unpatchify(x)

        x = x[
            :,
            :,
            :original_h,
            :original_w
        ]

        x = self.local_out(x)

        return residual + self.residual_scale * x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class MST(nn.Module):
    def __init__(self, in_dim=31, out_dim=31, dim=31, stage=2, num_blocks=[2,4,4]):
        super(MST, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = DiTStyleBottleneck(
            in_channels=dim_stage,
            hidden_dim=128,
            patch_size=4,
            depth=2,
            num_heads=4
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
        # LayerNorm with elementwise_affine=False has
        # weight=None and bias=None.
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (MSAB, FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out

class MST_Plus_Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, n_feat=31, stage=3):
        super(MST_Plus_Plus, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        modules_body = [MST(dim=31, stage=2, num_blocks=[1,1,1]) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = self.conv_in(x)
        h = self.body(x)
        h = self.conv_out(h)
        h += x
        return h[:, :, :h_inp, :w_inp]

