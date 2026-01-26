import torch
import torch.nn as nn
import torch.nn.functional as F
from models.linear import Linear
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import numbers
def to_3d(x):
    return rearrange(x, 'b c d h w -> b (d h w) c')


def to_4d(x, d, h, w):
    return rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)

        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        d, h, w = x.shape[-3:]
        return to_4d(self.body(to_3d(x)), d, h, w)

class DepthWiseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias=True):
        super(DepthWiseConv3d, self).__init__()
        self.net = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, stride=stride, bias=bias),
            nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))
    def forward(self, x):
        return self.net(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, bias=False):
        super(Mlp, self).__init__()

        self.project_in = nn.Conv3d(in_features, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv3d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.project_out = nn.Conv3d(hidden_features // 2, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Self_Attention3D(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim , dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v) #.transpose(1, 2)
        x = rearrange(x, 'b h n d -> b (h d) n')
        x = self.pool(x).unsqueeze(-1).unsqueeze(-1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class C_Cross_Attention3D(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=12,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Conv3d(dim, dim , kernel_size=1)
        self.kv = nn.Conv3d(dim, dim * 2, kernel_size=1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, C, D, H, W = x.shape #x_.shape=(B, 64, 1024)
        N = D * H * W
        _, C_, D_, H_, W_ = y.shape
        N_ = D_ * H_ * W_
        q = self.q(y).reshape(B, N_, self.num_heads, C_//self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C_//self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, 1, 1, 1)
        x = self.proj(x)
        #x = self.proj_drop(x)
        return x




class CrossModalAdaptiveFusion(nn.Module):
    """
    Cross-modal Adaptive Fusion Module (CAFM)
    Optimized for input shapes: visual_feat [1, 768, 20, 20, 20], text_feat [1, 768]
    """

    def __init__(self, visual_dim=768, text_dim=768, kernel_size=3, groups=12):
        super(CrossModalAdaptiveFusion, self).__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.kernel_size = kernel_size
        self.groups = groups
        self.group_dim = visual_dim // groups

        # Text processing
        self.text_proj = nn.Linear(text_dim, visual_dim)

        # Global average pooling to get visual context
        self.visual_pool = nn.AdaptiveAvgPool3d(1)

        # Cross attention
        self.cross_attn = nn.MultiheadAttention(visual_dim, num_heads=8, batch_first=True)

        # Kernel generation
        kernel_params = kernel_size * kernel_size * kernel_size * self.group_dim * groups
        self.kernel_net = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(visual_dim * 4, kernel_params)
        )

        # Modulation network
        self.modulation_net = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim),
            nn.Sigmoid()
        )

        # Normalization
        self.norm = nn.GroupNorm(groups, visual_dim)

        # Output projection
        self.output_proj = nn.Conv3d(visual_dim, visual_dim, kernel_size=1)

    def apply_group_conv3d(self, x, kernels):
        """
        Apply group-wise 3D convolution
        x: [B, C, H, W, D]
        kernels: [B, groups, group_dim, kernel_size, kernel_size, kernel_size]
        """
        B, C, H, W, D = x.shape
        group_size = C // self.groups

        outputs = []
        for i in range(self.groups):
            # Get group features
            group_feat = x[:, i * group_size:(i + 1) * group_size]  # [B, group_size, H, W, D]

            # Get group kernel
            group_kernel = kernels[:, i]  # [B, group_dim, k, k, k]

            # Apply convolution for each channel in the group
            group_outputs = []
            for j in range(group_size):
                channel_feat = group_feat[:, j:j + 1]  # [B, 1, H, W, D]
                channel_kernel = group_kernel[:, j:j + 1]  # [B, 1, k, k, k]

                # Apply depth-wise convolution
                padding = self.kernel_size // 2
                conv_output = F.conv3d(channel_feat, channel_kernel,
                                       padding=padding)
                group_outputs.append(conv_output)

            group_output = torch.cat(group_outputs, dim=1)  # [B, group_size, H, W, D]
            outputs.append(group_output)

        return torch.cat(outputs, dim=1)

    def forward(self, visual_feat, text_feat):
        """
        visual_feat: [1, 768, 20, 20, 20] - refined volumetric features
        text_feat: [1, 768] - textual embeddings
        return: fused features [1, 768, 20, 20, 20]
        """
        B, C, H, W, D = visual_feat.shape

        # Get global visual context
        visual_context = self.visual_pool(visual_feat).view(B, C)  # [B, C]

        # Project text feature
        text_projected = self.text_proj(text_feat)  # [B, C]

        # Cross attention between visual context and text
        visual_context_expanded = visual_context.unsqueeze(1)  # [B, 1, C]
        text_expanded = text_projected.unsqueeze(1)  # [B, 1, C]

        # Cross attention
        attn_output, _ = self.cross_attn(
            visual_context_expanded,  # query
            text_expanded,  # key
            text_expanded  # value
        )
        attn_context = attn_output.squeeze(1)  # [B, C]

        # Combine visual and text contexts
        combined_context = torch.cat([visual_context, attn_context], dim=1)  # [B, 2C]

        # Generate dynamic kernels
        kernel_params = self.kernel_net(combined_context)  # [B, kernel_params]
        kernels = kernel_params.view(B, self.groups, self.group_dim,
                                     self.kernel_size, self.kernel_size, self.kernel_size)

        # Generate modulation weights
        mod_weights = self.modulation_net(combined_context)  # [B, C]
        mod_weights = mod_weights.view(B, C, 1, 1, 1)  # [B, C, 1, 1, 1]

        # Apply modulation
        modulated_visual = visual_feat * mod_weights

        # Apply dynamic convolution
        output = self.apply_group_conv3d(modulated_visual, kernels)

        # Normalization and final projection
        output = self.norm(output)
        output = self.output_proj(output)

        return output

class Block3D(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        attn_head_dim=None,
        kernel_size=3,
        groups=12,
        LayerNorm_type='WithBias'
    ):
        super().__init__()
        #self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)

        # self.c_attn = C_Cross_Attention3D(
        # dim,
        # num_heads=num_heads,
        # qk_scale=qk_scale,
        # attn_drop=attn_drop,
        # proj_drop=drop,
        # attn_head_dim=attn_head_dim
        # )

        self.c_attn = CrossModalAdaptiveFusion(visual_dim=dim,text_dim=dim,kernel_size=kernel_size,groups=groups)

        
        self.text_lora = Linear(in_dim=dim, out_dim=dim, hidden_list = [dim])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim
        )

    def forward(self, x, y):
        y = self.text_lora(y.squeeze(1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = x * self.c_attn(x, y)
        x = self.norm2(x)
        x = x + self.drop_path(self.mlp(x))
        return self.norm3(x)

class Basic_block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads
    ):
        super().__init__()
        self.depth = 1
        self.block = nn.ModuleList([Block3D(dim,
        num_heads,
        mlp_ratio=4.0,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        attn_head_dim=None)
    for i in range (self.depth)])

    def forward(self, x, y):
        for blk in self.block:
            x = blk(x, y)
        return x