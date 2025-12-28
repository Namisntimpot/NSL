import os
import numpy as np
import torch
import torch.distributed as dist
import numbers

def setup_distributed_full(backend="nccl", port=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        import subprocess
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size


def setup_ddp(freeport:str, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f"{freeport}"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def print_ddp(*values, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*values, **kwargs)

def get_free_port():
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    return sock.getsockname()[1]


def general_gather(rank, world_size, obj):
    '''
    only support: number, torch.Tensor, np.ndarray, dict of number...  
    if obj is a list, it assumes that it can be converted to tensor by torch.tensor(...) directly  
    Tt won't maintain the original type, numbers/ndarrays/lists will become tensors  

    it won't reduce the results.
    '''
    if obj is None:
        return None
    if isinstance(obj, dict):
        ret = {k: general_gather(rank, world_size, v) for k, v in obj.items()}
        if rank == 0:
            return ret
        else:
            return
    elif isinstance(obj, numbers.Number):
        obj = torch.tensor([obj], dtype=torch.float32, device=f"cuda:{rank}")
    elif isinstance(obj, list):
        obj = torch.tensor(obj, dtype=torch.float32, device=f'cuda:{rank}')
    elif isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj).to(rank)
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(rank)
    gathered_tensors = [torch.zeros_like(obj, device=obj.device) for _ in range(world_size)] if rank == 0 else None
    dist.gather(obj, gathered_tensors, dst=0)
    if rank == 0:
        return gathered_tensors
    
def barrier():
    if dist.is_initialized():
        dist.barrier()

def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0