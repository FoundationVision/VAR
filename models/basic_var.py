import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.helpers import DropPath, drop_path


# this file only provides the 3 blocks used in VAR transformer
__all__ = ['FFN', 'AdaLNSelfAttn', 'AdaLNBeforeHead']


# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError: pass
# automatically import faster attention implementations
try: from xformers.ops import memory_efficient_attention
except ImportError: pass
try: from flash_attn import flash_attn_func              # qkv: BLHc, ret: BLHcq
except ImportError: pass
try: from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
except ImportError:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
        if attn_mask is not None: attn.add_(attn_mask)
        return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
    
    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0,
                heuristic=0, process_group=None,
            ))
        else:
            return self.drop(self.fc2( self.act(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


class SelfAttention(nn.Module):
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12,
        attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads  # =64
        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None
        
        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None
    
    def kv_caching(self, enable: bool): self.caching, self.cached_k, self.cached_v = enable, None, None
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, attn_bias):
        B, L, C = x.shape
        
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype
        # qkv: BL3Hc
        
        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        if using_flash or self.using_xform: q, k, v = qkv.unbind(dim=2); dim_cat = 1   # q or k or v: BLHc
        else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); dim_cat = 2               # q or k or v: BHLc
        
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform: scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        if self.caching:
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat); v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        
        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), attn_bias=None if attn_bias is None else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1), p=dropout_p, scale=self.scale).view(B, L, C)
        else:
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
        
        return self.proj_drop(self.proj(oup))
        # attn = (q @ k.transpose(-2, -1)).add_(attn_bias + self.local_rpb())  # BHLc @ BHcL => BHLL
        # attn = self.attn_drop(attn.softmax(dim=-1))
        # oup = (attn @ v).transpose_(1, 2).reshape(B, L, -1)     # BHLL @ BHLc = BHLc => BLHc => BLC
    
    def extra_repr(self) -> str:
        return f'using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}'


class AdaLNSelfAttn(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=False, fused_if_available=True,
    ):
        super(AdaLNSelfAttn, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SelfAttention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available)
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop, fused_if_available=fused_if_available)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        
        self.fused_add_norm_fn = None
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, cond_BD, attn_bias):   # C: embed_dim, D: cond_dim
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        x = x + self.drop_path(self.attn( self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias ).mul_(gamma1))
        x = x + self.drop_path(self.ffn( self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed when FusedMLP is used
        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2*C))
    
    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
