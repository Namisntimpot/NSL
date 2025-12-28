import os
import random
import numpy as np
import torch
import torch.distributed as dist
import numbers
from omegaconf import OmegaConf
from easydict import EasyDict
try:
    import imageio.v2 as imageio
except:
    import imageio
from io import BytesIO
from glob import glob

try:
    from ..deepsl_data.dataloader.utils import *
except:
    from deepsl_data.dataloader.utils import *

from .dist import *
from .transforms import load_exr

class FileIOHelper:
    def __init__(self):
        import megfile
        self.open_func = megfile.smart_open
        self.makedirs_func = megfile.smart_makedirs
        self.glob_func = megfile.smart_glob
        self.listdir_func = megfile.smart_listdir
        self.isfile_func = megfile.smart_isfile
        self.exists_func = megfile.smart_exists

    def open(self, path, mode, **kwargs):
        return self.open_func(path, mode, **kwargs)    
    def makedirs(self, path, exist_ok=True):
        return self.makedirs_func(path, exist_ok=exist_ok)    
    def glob(self, path):
        return self.glob_func(path)
    def listdir(self, path):
        return self.listdir_func(path)
    def isfile(self, path):
        return self.isfile_func(path)
    def exists(self, path):
        return self.exists_func(path)
    
    
def load_config(cfg_path):
    if not hasattr(load_config, 'resolver_registered'):
        def add_resolver(a: int, b: int) -> int:
            return a + b
        def sub_resolver(a: int, b: int) -> int:
            return a - b
        OmegaConf.register_new_resolver("add", add_resolver)
        OmegaConf.register_new_resolver("sub", sub_resolver)
        setattr(load_config, 'resolver_registered', True)

    iohelper = FileIOHelper()
    with iohelper.open(cfg_path, 'r') as f:
        cfg = OmegaConf.load(f)
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.to_container(cfg)
    cfg = EasyDict(cfg)
    return cfg


def check_make_dirs(*dirs):
    for d in dirs:
        if not d.startswith('s3:'):
            os.makedirs(d, exist_ok=True)
        else:
            import megfile
            megfile.smart_makedirs(d, exist_ok=True)

def manual_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def copy_file(src:str, dst:str):
    if not src.startswith("s3:") and not dst.startswith("s3:"):
        import shutil
        if os.path.isfile(src):
            shutil.copy(src, dst)
        else:
            shutil.copytree(src, dst)        
    else:
        import megfile
        import subprocess
        cmd = ['megfile', 'cp' if megfile.smart_isfile(src) else 'sync', src, dst]
        while True:
            ret = subprocess.run(cmd, stdout=subprocess.PIPE)
            if ret.returncode == 0:
                break


def get_range_random(min, max):
    return torch.empty(1).uniform_(min, max).item()


def load_images(path:str, **kwargs):
    '''
    value unchanged, dtype unchanged.  
    '''
    iohelper = FileIOHelper()
    with iohelper.open(path, 'rb') as f:
        content = f.read()
    with BytesIO(content) as stream:
        if path.endswith(".exr"):
            img = load_exr(stream, **kwargs)
        else:
            fmt = os.path.splitext(path)[1][1:]
            img = imageio.imread(stream, format=fmt, **kwargs)
    return img