import datetime
import functools
import os
import sys
from typing import List
from typing import Union

import torch
import torch.distributed as tdist
import torch.multiprocessing as mp

__rank, __local_rank, __world_size, __device = 0, 0, 1, 'cuda' if torch.cuda.is_available() else 'cpu'
__initialized = False


def initialized():
    return __initialized


def initialize(fork=False, backend='nccl', gpu_id_if_not_distibuted=0, timeout=30):
    global __device
    if not torch.cuda.is_available():
        print(f'[dist initialize] cuda is not available, use cpu instead', file=sys.stderr)
        return
    elif 'RANK' not in os.environ:
        torch.cuda.set_device(gpu_id_if_not_distibuted)
        __device = torch.empty(1).cuda().device
        print(f'[dist initialize] env variable "RANK" is not set, use {__device} as the device', file=sys.stderr)
        return
    # then 'RANK' must exist
    global_rank, num_gpus = int(os.environ['RANK']), torch.cuda.device_count()
    local_rank = global_rank % num_gpus
    torch.cuda.set_device(local_rank)
    
    # ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py#L29
    if mp.get_start_method(allow_none=True) is None:
        method = 'fork' if fork else 'spawn'
        print(f'[dist initialize] mp method={method}')
        mp.set_start_method(method)
    tdist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=timeout*60))
    
    global __rank, __local_rank, __world_size, __initialized
    __local_rank = local_rank
    __rank, __world_size = tdist.get_rank(), tdist.get_world_size()
    __device = torch.empty(1).cuda().device
    __initialized = True
    
    assert tdist.is_initialized(), 'torch.distributed is not initialized!'
    print(f'[lrk={get_local_rank()}, rk={get_rank()}]')


def get_rank():
    return __rank


def get_local_rank():
    return __local_rank


def get_world_size():
    return __world_size


def get_device():
    return __device


def set_gpu_id(gpu_id: int):
    if gpu_id is None: return
    global __device
    if isinstance(gpu_id, (str, int)):
        torch.cuda.set_device(int(gpu_id))
        __device = torch.empty(1).cuda().device
    else:
        raise NotImplementedError


def is_master():
    return __rank == 0


def is_local_master():
    return __local_rank == 0


def new_group(ranks: List[int]):
    if __initialized:
        return tdist.new_group(ranks=ranks)
    return None


def barrier():
    if __initialized:
        tdist.barrier()


def allreduce(t: torch.Tensor, async_op=False):
    if __initialized:
        if not t.is_cuda:
            cu = t.detach().cuda()
            ret = tdist.all_reduce(cu, async_op=async_op)
            t.copy_(cu.cpu())
        else:
            ret = tdist.all_reduce(t, async_op=async_op)
        return ret
    return None


def allgather(t: torch.Tensor, cat=True) -> Union[List[torch.Tensor], torch.Tensor]:
    if __initialized:
        if not t.is_cuda:
            t = t.cuda()
        ls = [torch.empty_like(t) for _ in range(__world_size)]
        tdist.all_gather(ls, t)
    else:
        ls = [t]
    if cat:
        ls = torch.cat(ls, dim=0)
    return ls


def allgather_diff_shape(t: torch.Tensor, cat=True) -> Union[List[torch.Tensor], torch.Tensor]:
    if __initialized:
        if not t.is_cuda:
            t = t.cuda()
        
        t_size = torch.tensor(t.size(), device=t.device)
        ls_size = [torch.empty_like(t_size) for _ in range(__world_size)]
        tdist.all_gather(ls_size, t_size)
        
        max_B = max(size[0].item() for size in ls_size)
        pad = max_B - t_size[0].item()
        if pad:
            pad_size = (pad, *t.size()[1:])
            t = torch.cat((t, t.new_empty(pad_size)), dim=0)
        
        ls_padded = [torch.empty_like(t) for _ in range(__world_size)]
        tdist.all_gather(ls_padded, t)
        ls = []
        for t, size in zip(ls_padded, ls_size):
            ls.append(t[:size[0].item()])
    else:
        ls = [t]
    if cat:
        ls = torch.cat(ls, dim=0)
    return ls


def broadcast(t: torch.Tensor, src_rank) -> None:
    if __initialized:
        if not t.is_cuda:
            cu = t.detach().cuda()
            tdist.broadcast(cu, src=src_rank)
            t.copy_(cu.cpu())
        else:
            tdist.broadcast(t, src=src_rank)


def dist_fmt_vals(val: float, fmt: Union[str, None] = '%.2f') -> Union[torch.Tensor, List]:
    if not initialized():
        return torch.tensor([val]) if fmt is None else [fmt % val]
    
    ts = torch.zeros(__world_size)
    ts[__rank] = val
    allreduce(ts)
    if fmt is None:
        return ts
    return [fmt % v for v in ts.cpu().numpy().tolist()]


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force or is_master():
            ret = func(*args, **kwargs)
        else:
            ret = None
        barrier()
        return ret
    return wrapper


def local_master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force or is_local_master():
            ret = func(*args, **kwargs)
        else:
            ret = None
        barrier()
        return ret
    return wrapper


def for_visualize(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_master():
            # with torch.no_grad():
            ret = func(*args, **kwargs)
        else:
            ret = None
        return ret
    return wrapper


def finalize():
    if __initialized:
        tdist.destroy_process_group()
