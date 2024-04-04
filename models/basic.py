import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.helpers import DropPath, drop_path


# this file only defines the 4 blocks used in the transformer
__all__ = ['FFN', 'SelfAttention', 'SABlock', 'AdaLNSABlock',]


# automatically import faster operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError: pass
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
        attn_drop=0., proj_drop=0., tau=4, cos_attn=False, flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads  # =64
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau
        
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
    
    # NOTE: attn_bias is None during inference
    def forward(self, x, attn_bias):
        B, L, C = x.shape
        
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)
        # qkv: BL3Hc
        
        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        if using_flash or self.using_xform: q, k, v = qkv.unbind(dim=2); dim_cat = 1   # q or k or v: BLHc
        else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); dim_cat = 2               # q or k or v: BHLc
        
        if self.cos_attn:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform: scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        if self.caching:
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat); v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        
        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            assert attn_bias is None and qkv.dtype != torch.float32
            oup = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(q, k, v, attn_bias=None if attn_bias is None else attn_bias.to(dtype=q.dtype).expand(B, self.num_heads, -1, -1), p=dropout_p, scale=self.scale).view(B, L, C)
        else:
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
        
        return self.proj_drop(self.proj(oup))
        # attn = (q @ k.transpose(-2, -1)).add_(attn_bias + self.local_rpb())  # BHLc @ BHcL => BHLL
        # attn = self.attn_drop(attn.softmax(dim=-1))
        # oup = (attn @ v).transpose_(1, 2).reshape(B, L, -1)     # BHLL @ BHLc = BHLc => BLHc => BLC
    
    def extra_repr(self) -> str:
        return f'using_flash={self.using_flash}, using_xform={self.using_xform}, tau={self.tau}, cos_attn={self.cos_attn}'


class SABlock(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., tau=4, cos_attn=False,
        layer_scale=-1., flash_if_available=False, fused_if_available=True,
    ):
        super(SABlock, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.drop_prob, self.drop_path = drop_path, (DropPath(drop_path) if drop_path > 0. else nn.Identity())
        self.norm1 = norm_layer(embed_dim)
        self.attn = SelfAttention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, tau=tau, cos_attn=cos_attn, flash_if_available=flash_if_available)
        self.norm2 = norm_layer(embed_dim)
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop, fused_if_available=fused_if_available)
        
        self.fused_add_norm_fn = dropout_add_layer_norm if fused_if_available else None
        self.layer_scale = layer_scale
        if layer_scale >= 0:
            self.gamma1 = 1 if (self.fused_add_norm_fn is not None and block_idx == 0) else nn.Parameter(layer_scale * torch.ones(embed_dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(embed_dim), requires_grad=True)
        else:
            self.gamma1 = self.gamma2 = 1
    
    # NOTE: attn_bias is None during inference
    def forward(self, x, cond_BD, attn_bias):  # todo tky: minGPT and vqgan also uses pre-norm, just like this, while MaskGiT uses post-norm
        if self.fused_add_norm_fn is not None:
            return self.fused_forward_wo_cond(x, attn_bias=attn_bias)
        main_type = x.dtype
        x = x.float() + self.drop_path(self.gamma1 * self.attn(self.norm1(x), attn_bias=attn_bias))     # following flash-attn: using fp32 in residual
        x = x + self.drop_path(self.gamma2 * self.ffn(self.norm2(x.to(dtype=main_type))))               # following flash-attn: using fp32 in residual
        return x.to(dtype=main_type)
    
    # NOTE: attn_bias is None during inference
    def fused_forward_wo_cond(self, x_and_residual, attn_bias):
        x, residual = (x_and_residual, None) if isinstance(x_and_residual, torch.Tensor) else x_and_residual
        rowscale1 = drop_path(x=x.new_ones(x.shape[:-1]), drop_prob=self.last_drop_p, training=True) if self.last_drop_p > 0 and self.training else None
        x, residual = self.fused_add_norm_fn(
            x0=x, residual=residual, weight=self.norm1.weight, bias=self.norm1.bias, dropout_p=0.0,              # todo: no drop
            epsilon=self.norm1.eps, rowscale=rowscale1, layerscale=self.gamma1 if isinstance(self.gamma1, torch.nn.Parameter) else None, prenorm=True, residual_in_fp32=True,
        )
        x = self.attn(x, attn_bias=attn_bias)
        rowscale2 = drop_path(x=x.new_ones(x.shape[:-1]), drop_prob=self.drop_prob, training=True) if self.drop_prob > 0 and self.training else None
        x, residual = self.fused_add_norm_fn(
            x0=x, residual=residual, weight=self.norm2.weight, bias=self.norm2.bias, dropout_p=0.0,              # todo: no drop
            epsilon=self.norm2.eps, rowscale=rowscale2, layerscale=self.gamma2 if isinstance(self.gamma2, torch.nn.Parameter) else None, prenorm=True, residual_in_fp32=True,
        )
        return self.ffn(x), residual
    
    def extra_repr(self) -> str:
        return f'fused_add_norm={self.fused_add_norm_fn is not None}, layer_scale={self.layer_scale:g} (gamma1 {"on" if isinstance(self.gamma1, nn.Parameter) else "off"}) (gamma2 {"on" if isinstance(self.gamma2, nn.Parameter) else "off"})'


class AdaLNSABlock(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., tau=4, cos_attn=False,
        flash_if_available=False, fused_if_available=True,
    ):
        super(AdaLNSABlock, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SelfAttention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, tau=tau, cos_attn=cos_attn, flash_if_available=flash_if_available)
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop, fused_if_available=fused_if_available)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        
        self.fused_add_norm_fn = None
    
    # NOTE: attn_bias is None during inference
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


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = torch.Generator(device=dev)
    # for Li in ([1, 3, 5], [1, 3]):
    rng.manual_seed(0)
    B, H, cq, ckv = 4, 8, 64, 96
    Cq = H*cq
    Ckv = H*ckv
    
    Li = [5, 4, 7, 6]
    Lq = 10
    L = max(Li)
    attn_bias = torch.zeros(B, 1, Lq, L, device=dev)
    for i, x in enumerate(Li):
        attn_bias[i, 0, :, x:] = -torch.inf
    
    q = torch.randn(B, Lq, H, cq, generator=rng, device=dev)
    k = torch.randn(B, L, H, ckv, generator=rng, device=dev)
    v = torch.randn(B, L, H, ckv, generator=rng, device=dev)
    tq, tk, tv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)    # BHLc
    
    seqlen_k = torch.tensor(Li, dtype=torch.int32, device=dev)
    cu_seqlens_k = F.pad(torch.cumsum(seqlen_k, dim=0, dtype=torch.torch.int32), (1, 0))
    kv = torch.stack([k, v], dim=2)
    kv_compact = torch.cat([kv[i, :Li[i]] for i in range(B)], dim=0)
    
    ca = CrossAttention(for_attn_pool=False, embed_dim=Cq, kv_dim=Ckv, num_heads=H)
    CrossAttention.forward
    ca(q, (kv_compact, cu_seqlens_k, max(Li))).mean().backward()


if __name__ == '__main__':
    main()
