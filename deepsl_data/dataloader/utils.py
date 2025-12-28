import os
import numpy as np
import torch
from typing import Union

def to_tensor(obj: Union[np.ndarray, list, dict]):
    '''
    将np数组转为torch张量，不处理device
    '''
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    elif isinstance(obj, (list, tuple, set)):
        return [to_tensor(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: to_tensor(v) for k, v in obj.items()}
    else:
        return obj
    
def to_device(obj: Union[torch.Tensor, list, tuple, dict], device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple, set)):
        return [to_device(v, device) for v in obj]
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    else:
        return obj
    
def merge_all(obj:list, merge_fn = lambda x: torch.stack(x, dim=0)):
    '''
    按key合并张量.
    列表中有很多成员，每个成员都是字典，把具有相同key的合并到一起  
    merge_fn: 输入list of tensor. 输出一个结果.  
    只处理一层，list下就是dict或者tensor, 不处理list套list  
    字典可以套字典，但字典的叶子的值不能是list, tuple, set这些.  
    '''
    if isinstance(obj[0], torch.Tensor):
        return merge_fn(obj)
    elif isinstance(obj[0], dict):
        keys = obj[0].keys()
        d = {k: [v[k] for v in obj] for k in keys}
        return {k:merge_all(v, merge_fn) for k, v in d.items()}
    else:
        return obj