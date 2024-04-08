# distributed.py
import datetime
import os
import sys
import functools
from typing import List, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

class DistributedManager:
    def __init__(self):
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.initialized = False

    def initialize(self, fork=False, backend='nccl', gpu_id_if_not_distributed=0, timeout=30):
        if not torch.cuda.is_available():
            print('[dist initialize] cuda is not available, using CPU instead', file=sys.stderr)  # Changed: Added comment to indicate CUDA availability
            return

        if 'RANK' not in os.environ:
            torch.cuda.set_device(gpu_id_if_not_distributed)
            self.device = torch.empty(1).cuda().device
            print(f'[dist initialize] "RANK" environment variable not set, using {self.device} as the device', file=sys.stderr)  # Changed: Added comment for device selection
            return

        global_rank, num_gpus = int(os.environ['RANK']), torch.cuda.device_count()
        local_rank = global_rank % num_gpus
        torch.cuda.set_device(local_rank)

        mp_method = 'fork' if fork else 'spawn'
        print(f'[dist initialize] mp method={mp_method}')
        mp.set_start_method(mp_method)

        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=timeout*60))
        
        self.local_rank = local_rank
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.device = torch.empty(1).cuda().device
        self.initialized = True

        assert dist.is_initialized(), 'torch.distributed is not initialized!'
        print(f'[lrk={self.get_local_rank()}, rk={self.get_rank()}]')

    def get_rank(self):
        return self.rank

    def get_local_rank(self):
        return self.local_rank

    def get_world_size(self):
        return self.world_size

    def get_device(self):
        return self.device

    def barrier(self):
        if self.initialized:
            dist.barrier()

    def allreduce(self, tensor: torch.Tensor, async_op=False):
        if self.initialized:
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            dist.all_reduce(tensor, async_op=async_op)
        return tensor

    def allgather(self, tensor: torch.Tensor, cat=True) -> Union[List[torch.Tensor], torch.Tensor]:
        if self.initialized:
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered, tensor)
            if cat:
                gathered = torch.cat(gathered, dim=0)
            return gathered
        else:
            return [tensor]

    def broadcast(self, tensor: torch.Tensor, src_rank) -> None:
        if self.initialized:
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            dist.broadcast(tensor, src=src_rank)
            if not tensor.is_cuda:
                tensor = tensor.cpu()

    def finalize(self):
        if self.initialized:
            dist.destroy_process_group()


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force or args[0].get_rank() == 0:
            ret = func(*args, **kwargs)
        else:
            ret = None
        args[0].barrier()  # Changed: Added comment for barrier synchronization
        return ret
    return wrapper


def local_master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force or args[0].get_local_rank() == 0:
            ret = func(*args, **kwargs)
        else:
            ret = None
        args[0].barrier()  # Changed: Added comment for barrier synchronization
        return ret
    return wrapper


def for_visualize(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args[0].get_rank() == 0:
            # with torch.no_grad():
            ret = func(*args, **kwargs)
        else:
            ret = None
        return ret
    return wrapper
