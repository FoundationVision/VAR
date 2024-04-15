import math
from typing import List, Optional, Tuple, Union

import torch


class NullCtx:
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AmpOptimizer:
    def __init__(
        self,
        mixed_precision: int,
        optimizer: torch.optim.Optimizer, names: List[str], paras: List[torch.nn.Parameter],
        grad_clip: float, n_gradient_accumulation: int = 1,
    ):
        self.enable_amp = mixed_precision > 0
        self.using_fp16_rather_bf16 = mixed_precision == 1
        
        if self.enable_amp:
            self.amp_ctx = torch.autocast('cuda', enabled=True, dtype=torch.float16 if self.using_fp16_rather_bf16 else torch.bfloat16, cache_enabled=True)
            self.scaler = torch.cuda.amp.GradScaler(init_scale=2. ** 11, growth_interval=1000) if self.using_fp16_rather_bf16 else None # only fp16 needs a scaler
        else:
            self.amp_ctx = NullCtx()
            self.scaler = None
        
        self.optimizer, self.names, self.paras = optimizer, names, paras   # paras have been filtered so everyone requires grad
        self.grad_clip = grad_clip
        self.early_clipping = self.grad_clip > 0 and not hasattr(optimizer, 'global_grad_norm')
        self.late_clipping = self.grad_clip > 0 and hasattr(optimizer, 'global_grad_norm')
        
        self.r_accu = 1 / n_gradient_accumulation   # r_accu == 1.0 / n_gradient_accumulation
    
    def backward_clip_step(
        self, stepping: bool, loss: torch.Tensor,
    ) -> Tuple[Optional[Union[torch.Tensor, float]], Optional[float]]:
        # backward
        loss = loss.mul(self.r_accu)   # r_accu == 1.0 / n_gradient_accumulation
        orig_norm = scaler_sc = None
        if self.scaler is not None:
            self.scaler.scale(loss).backward(retain_graph=False, create_graph=False)
        else:
            loss.backward(retain_graph=False, create_graph=False)
        
        if stepping:
            if self.scaler is not None: self.scaler.unscale_(self.optimizer)
            if self.early_clipping:
                orig_norm = torch.nn.utils.clip_grad_norm_(self.paras, self.grad_clip)
            
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                scaler_sc: float = self.scaler.get_scale()
                if scaler_sc > 32768.: # fp16 will overflow when >65536, so multiply 32768 could be dangerous
                    self.scaler.update(new_scale=32768.)
                else:
                    self.scaler.update()
                try:
                    scaler_sc = float(math.log2(scaler_sc))
                except Exception as e:
                    print(f'[scaler_sc = {scaler_sc}]\n' * 15, flush=True)
                    raise e
            else:
                self.optimizer.step()
            
            if self.late_clipping:
                orig_norm = self.optimizer.global_grad_norm
            
            self.optimizer.zero_grad(set_to_none=True)
        
        return orig_norm, scaler_sc
    
    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict()
        } if self.scaler is None else {
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state, strict=True):
        if self.scaler is not None:
            try: self.scaler.load_state_dict(state['scaler'])
            except Exception as e: print(f'[fp16 load_state_dict err] {e}')
        self.optimizer.load_state_dict(state['optimizer'])
