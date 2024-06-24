import math
from functools import partial
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn

from huggingface_hub import PyTorchModelHubMixin
from torch.nn import functional as F

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2)) # begins and ends of each level of token pyramid in the whole token sequence
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC) # positional encoding

        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        # class embedding
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        cond_BD_or_gss = self.shared_ada_lin(cond_BD) # shared_ada_lin: SiLU + AdaLin


        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = (
            sos.unsqueeze(1).expand(2 * B, self.first_l, -1) # 2B, l, Cvae; class conditioning
            + self.pos_start.expand(2 * B, self.first_l, -1) # 2B, l, Cvae; positional encoding
            + lvl_pos[:, :self.first_l]  # level encoding
        )

        # so next token map is the first token map, which is the class embedding + positional encoding + level encoding

        # print(f'next_token_map: {next_token_map.shape=}\n {next_token_map.dtype=}\n {next_token_map.device=}\n')
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        # print(f'[autoregressive_infer_cfg] {f_hat.shape=}\n, {next_token_map.shape=}\n')
        
        for b in self.blocks: b.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):   # si: i-th segment, si=0: first segment
            ratio = si / self.num_stages_minus_1 # processed ratio: 0.0 ~ 1.0, how much has been processed
            
            cur_L += pn*pn # cur_L: current length of token sequence
            
            x = next_token_map
            print(f'level {si} x: {x.shape=}\n')

            # refine the token map via self attention and feed forward network
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            
            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio # t: guidance ratio -- rises over time from 0.0 to cfg
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]  # B, l, V
            
            # Sample from the rk logits
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

            # if si == self.num_stages_minus_1 - 1:
            #     idx_Bl = logits_BlV.argmax(dim=-1) # DEBUG 

            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # shape
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            
            # print(f'[autoregressive_infer_cfg]\n {si=}\n {x.shape=}\n {logits_BlV.shape=}\n {idx_Bl.shape=}\n {h_BChw.shape=}\n')
            

            # h_BChw is now the zk before interpolation
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw) # upscale the h_BChw, add to f_hat and downscale the token map


            # print(f'[autoregressive_infer_cfg_2]\n {h_BChw.shape=}\n {f_hat.shape=}\n {next_token_map.shape=}\n')
            

            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)

        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    def get_class_embedding(self, label_B: Optional[Union[int, torch.LongTensor]], double_for_cfg=True):
        batch_size = label_B.shape[0] if isinstance(label_B, torch.Tensor) else label_B

        if isinstance(label_B, int):
            label_B = torch.full((batch_size,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        if double_for_cfg:
            label_B = torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0)

        return self.class_emb(label_B)
    
    def get_class_conditioning(self, label_B: Optional[Union[int, torch.LongTensor]]):
        class_embedding = self.get_class_embedding(label_B)
        return self.shared_ada_lin(class_embedding)

    def get_initial_token_map(self, label_B: torch.LongTensor):
        batch_size = label_B.shape[0]

        class_embedding = self.get_class_embedding(label_B, double_for_cfg=True)
    
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        initial_token_map = (
            class_embedding.unsqueeze(1).expand(2 * batch_size, self.first_l, -1) # 2B, l, Cvae; class conditioning
            + self.pos_start.expand(2 * batch_size, self.first_l, -1) # 2B, l, Cvae; positional encoding
            + lvl_pos[:, :self.first_l]  # level encoding
        )

        return initial_token_map
    
    def get_level_encoding(self, next_level: int):
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC # 1, L, C
        
        if next_level == 0:
            return lvl_pos[:, :self.first_l]

        assert next_level >= 1
        assert next_level < len(self.patch_nums)
        
        current_length = sum([
            patch_size * patch_size
            for patch_size
            in self.patch_nums[:next_level]
        ])

        return lvl_pos[:, current_length:current_length + self.patch_nums[next_level] ** 2]
    
    def prepare_token_map(self, f_hat, map_size_index: int):
        batch_size = f_hat.shape[0]

        token_map_size = self.patch_nums[map_size_index]

        if token_map_size != self.patch_nums[-1]: # not the last level
            downscaled_f_hat = F.interpolate(f_hat, size=token_map_size, mode='area')

        else:
            downscaled_f_hat = f_hat

        pre_token_map = downscaled_f_hat.view(batch_size, self.Cvae, -1).transpose(1, 2)
        token_map = self.word_embed(pre_token_map)

        token_map += self.get_level_encoding(map_size_index)

        token_map_cfg = token_map.repeat(2, 1, 1) # double the batch sizes due to CFG

        return token_map_cfg
    
    def get_causal_mask(self, map_size_index: int):
        current_length = sum([
            patch_size * patch_size
            for patch_size
            in self.patch_nums[:map_size_index + 1]
        ])

        attn_bias = self.attn_bias_for_masking[:, :, :current_length, :current_length]
        return attn_bias

    
    def token_map_to_relevant_logits(self, token_map, class_conditioning, map_size_index: int, masked: bool = True):
        '''
        Predicts the logits for each token map. If masked, then predicts logits only for the last token map.

        Args:
            token_map: the token map to predict logits for
            class_conditioning: the class conditioning for the token map
            map_size_index: the index of the token map in the token pyramid
            masked: whether to mask the logits for the token map
        '''
        attn_bias = self.get_causal_mask(map_size_index) if masked else None

        for b in self.blocks:
            token_map = b(x=token_map, cond_BD=class_conditioning, attn_bias=attn_bias)

        if masked:
            output_length = self.patch_nums[map_size_index] ** 2
            token_map = token_map[:, -output_length:]

        return self.get_logits(token_map, class_conditioning)
    
    def get_full_token_map(self, map_size_index, label_B, f_hat, last_grad_only=True):
        '''
        Returns the full token map for the given level of the token pyramid.

        Args:
            map_size_index: the index of the token map in the token pyramid
            label_B: the class labels
            f_hat: the f_hat tensor
            last_grad_only: whether to require grad for the last level or for all levels
        '''
        full_token_map = self.get_initial_token_map(label_B)
        
        if map_size_index == 0:
            return full_token_map
        
        with torch.no_grad() if last_grad_only else torch.enable_grad():
            for i in range(1, map_size_index): # for each level up to the second to last level
                step_token_map = self.prepare_token_map(f_hat, map_size_index=i)
                full_token_map = torch.cat([full_token_map, step_token_map], dim=1)

        # for the last level we need grad anyway
        last_step_token_map = self.prepare_token_map(f_hat, map_size_index=map_size_index) 
        full_token_map = torch.cat([full_token_map, last_step_token_map], dim=1)

        return full_token_map
    
    def quant_pyramid_to_var_inputs(self, quant_pyramid: List[torch.Tensor], label_B: torch.LongTensor, batch_size:int):
        '''
        Converts the quantized pyramid to a token map and attention bias.
        '''
        previous_map_size_index = len(quant_pyramid) - 1 # last patch index in the pyramid
        map_size_index = previous_map_size_index + 1 # We want to predict the next token map

        current_length = sum([
            patch_size * patch_size
            for patch_size
            in self.patch_nums[:map_size_index + 1]
        ])

        cfg_batch_size = batch_size * 2 # double the batch sizes due to CFG

        class_embedding = self.get_class_embedding(label_B)

        initial_token_map = (
            class_embedding
            .unsqueeze(1)
            .expand(cfg_batch_size, self.first_l, -1)
        )
        
        initial_token_map += self.pos_start.expand(cfg_batch_size, self.first_l, -1) # B, l, Cvae

        raw_token_map = self.vae_quant_proxy[0].limited_quant_pyramid_to_var_input(quant_pyramid)

        # Duplicate batch dim for cfg
        if map_size_index > 0:
            raw_token_map = raw_token_map.repeat(2, 1, 1)
            token_map = torch.cat((initial_token_map, self.word_embed(raw_token_map.float())), dim=1)
        else:
            token_map = initial_token_map

        token_map += self.lvl_embed(self.lvl_1L[:, :current_length].expand(cfg_batch_size, -1)) 
        token_map += self.pos_1LC[:, :current_length] # lvl: BLC;  pos: 1LC

        return token_map
    
    def predict_single_step_from_quant_pyramid(self, quant_pyramid: List[torch.Tensor], label_B: torch.LongTensor, batch_size:int, cfg=4, top_k=0, top_p=0.0):
        '''
        Predicts the next token map from the given quantized pyramid.
        '''
        previous_map_size_index = len(quant_pyramid) - 1 # last patch index in the pyramid
        map_size_index = previous_map_size_index + 1 # We want to predict the next token map

        token_map = self.quant_pyramid_to_var_inputs(quant_pyramid, label_B, batch_size=batch_size)
        class_conditioning = self.get_class_conditioning(label_B)

        logits = self.token_map_to_relevant_logits(token_map, class_conditioning, map_size_index=map_size_index, masked=True)

        indices = self.logits_to_indices(logits, cfg=cfg, level_ratio=map_size_index / (len(self.patch_nums) - 1), batch_size=batch_size, top_k=top_k, top_p=top_p)
        
        level_ratio = map_size_index / (len(self.patch_nums) - 1)
        patch_num = self.patch_nums[map_size_index]
        residual = self.indices_to_h(indices, batch_size=batch_size, level_ratio=level_ratio, patch_num=patch_num)

        return residual
    
    def predict_single_step_residual(self, f_hat, label_B, map_size_index, cfg=4, top_k=0, top_p=0.0):
        batch_size = label_B.shape[0]
        patch_num = self.patch_nums[map_size_index]
        level_ratio = map_size_index / (len(self.patch_nums) - 1)

        full_token_map = self.get_full_token_map(map_size_index, label_B, f_hat, last_grad_only=True)
        class_conditioning = self.get_class_conditioning(label_B)


        logits = self.token_map_to_relevant_logits(full_token_map, class_conditioning, map_size_index=map_size_index, masked=True)

        indices = self.logits_to_indices(logits, cfg=cfg, batch_size=batch_size, level_ratio=level_ratio, top_k=top_k, top_p=top_p)
        residual = self.indices_to_h(indices, batch_size=batch_size, level_ratio=level_ratio, patch_num=patch_num)

        return residual # in the paper it's h (BChw)
    

    def logits_to_indices(self, logits, cfg, level_ratio, batch_size, top_k=0, top_p=0.0):

        guidance_ratio = cfg * level_ratio

        logits_guided = (
            (1+guidance_ratio) * logits[:batch_size] +
            (- guidance_ratio) * logits[batch_size:]
        ) # use CFG to guide the logits

        return sample_with_top_k_top_p_(logits_guided, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

    def indices_to_h(self, indices, level_ratio, patch_num, batch_size):
        zk = (
            self.vae_quant_proxy[0].embedding(indices)
            .transpose_(1, 2)
            .reshape(batch_size, self.Cvae, patch_num, patch_num)
        ) # get back the embeddings and reshape back into B, C, H, W

        if patch_num != self.patch_nums[-1]:
            zk = F.interpolate(zk, size=(self.patch_nums[-1], self.patch_nums[-1]), mode='bicubic')

        h = self.vae_quant_proxy[0].quant_resi[level_ratio](zk) # get the residual (extra conv layers)

        return h
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L) # unused for now, no proressive training

        B = x_BLCv_wo_first_l.shape[0] # B: batch size
        
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) # B, l, Cvae
            
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )
