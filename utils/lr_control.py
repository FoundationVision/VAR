import math
from pprint import pformat
from typing import Tuple, List, Dict, Union

import torch.nn

import dist


def lr_wd_annealing(sche_type: str, optimizer, peak_lr, wd, wd_end, cur_it, wp_it, max_it, wp0=0.005, wpe=0.001):
    """Decay the learning rate with half-cycle cosine after warmup"""
    wp_it = round(wp_it)
    
    if cur_it < wp_it:
        cur_lr = wp0 + (1-wp0) * cur_it / wp_it
    else:
        pasd = (cur_it - wp_it) / (max_it-1 - wp_it)   # [0, 1]
        rest = 1 - pasd     # [1, 0]
        if sche_type == 'cos':
            cur_lr = wpe + (1-wpe) * (0.5 + 0.5 * math.cos(math.pi * pasd))
        elif sche_type == 'lin':
            T = 0.15; max_rest = 1-T
            if pasd < T: cur_lr = 1
            else: cur_lr = wpe + (1-wpe) * rest / max_rest  # 1 to wpe
        elif sche_type == 'lin0':
            T = 0.05; max_rest = 1-T
            if pasd < T: cur_lr = 1
            else: cur_lr = wpe + (1-wpe) * rest / max_rest
        elif sche_type == 'lin00':
            cur_lr = wpe + (1-wpe) * rest
        elif sche_type.startswith('lin'):
            T = float(sche_type[3:]); max_rest = 1-T
            wpe_mid = wpe + (1-wpe) * max_rest
            wpe_mid = (1 + wpe_mid) / 2
            if pasd < T: cur_lr = 1 + (wpe_mid-1) * pasd / T
            else: cur_lr = wpe + (wpe_mid-wpe) * rest / max_rest
        elif sche_type == 'exp':
            T = 0.15; max_rest = 1-T
            if pasd < T: cur_lr = 1
            else:
                expo = (pasd-T) / max_rest * math.log(wpe)
                cur_lr = math.exp(expo)
        else:
            raise NotImplementedError(f'unknown sche_type {sche_type}')
    
    cur_lr *= peak_lr
    pasd = cur_it / (max_it-1)
    cur_wd = wd_end + (wd - wd_end) * (0.5 + 0.5 * math.cos(math.pi * pasd))
    
    inf = 1e6
    min_lr, max_lr = inf, -1
    min_wd, max_wd = inf, -1
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)    # 'lr_sc' could be assigned
        max_lr = max(max_lr, param_group['lr'])
        min_lr = min(min_lr, param_group['lr'])
        
        param_group['weight_decay'] = cur_wd * param_group.get('wd_sc', 1)
        max_wd = max(max_wd, param_group['weight_decay'])
        if param_group['weight_decay'] > 0:
            min_wd = min(min_wd, param_group['weight_decay'])

    if min_lr == inf: min_lr = -1
    if min_wd == inf: min_wd = -1
    return min_lr, max_lr, min_wd, max_wd


def filter_params(model, nowd_keys=()) -> Tuple[
    List[str], List[torch.nn.Parameter], List[Dict[str, Union[torch.nn.Parameter, float]]]
]:
    para_groups, para_groups_dbg = {}, {}
    names, paras = [], []
    names_no_grad = []
    count, numel = 0, 0
    for name, para in model.named_parameters():
        name = name.replace('_fsdp_wrapped_module.', '')
        if not para.requires_grad:
            names_no_grad.append(name)
            continue  # frozen weights
        count += 1
        numel += para.numel()
        names.append(name)
        paras.append(para)
        
        if para.ndim == 1 or name.endswith('bias') or any(k in name for k in nowd_keys):
            cur_wd_sc, group_name = 0., 'ND'
        else:
            cur_wd_sc, group_name = 1., 'D'
        cur_lr_sc = 1.
        if group_name not in para_groups:
            para_groups[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': cur_lr_sc}
            para_groups_dbg[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': cur_lr_sc}
        para_groups[group_name]['params'].append(para)
        para_groups_dbg[group_name]['params'].append(name)
    
    for g in para_groups_dbg.values():
        g['params'] = pformat(', '.join(g['params']), width=200)
    
    print(f'[get_param_groups] param_groups = \n{pformat(para_groups_dbg, indent=2, width=240)}\n')
    
    for rk in range(dist.get_world_size()):
        dist.barrier()
        if dist.get_rank() == rk:
            print(f'[get_param_groups][rank{dist.get_rank()}] {type(model).__name__=} {count=}, {numel=}', flush=True, force=True)
    print('')
    
    assert len(names_no_grad) == 0, f'[get_param_groups] names_no_grad = \n{pformat(names_no_grad, indent=2, width=240)}\n'
    return names, paras, list(para_groups.values())
