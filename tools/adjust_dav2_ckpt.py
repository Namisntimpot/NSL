import os
import torch
import argparse

from utils.common import FileIOHelper
from tools.fix_ckpt import dav2_to_ours as adjust

# def adjust(ckpt:dict):
#     key_to_adjust = []
#     for k in ckpt:
#         if k.startswith("depth_head.projects") or k.startswith("depth_head.resize_layers"):
#             key_to_adjust.append(k)
#     for k in key_to_adjust:
#         param = ckpt.pop(k)
#         module_path:list = k.split(".")
#         module_path.insert(1, 'reassemble_block')
#         newk = ".".join(module_path)
#         ckpt[newk] = param
#     return ckpt

parser = argparse.ArgumentParser()
parser.add_argument("--dav2-ckpt", type=str)
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

dav2_ckpt_path = args.dav2_ckpt
out_path = args.out
if out_path is None:
    fname = os.path.basename(dav2_ckpt_path)
    out_path = os.path.join(os.path.dirname(dav2_ckpt_path), 
                            os.path.splitext(fname)[0] + "_adjusted" + os.path.splitext(fname)[1])
iohelper = FileIOHelper()

with iohelper.open(dav2_ckpt_path, 'rb') as f:
    dav2_ckpt = torch.load(f, map_location='cpu')
dav2_ckpt = dav2_ckpt['model'] if 'model' in dav2_ckpt else dav2_ckpt
ckpt_adjusted = adjust(dav2_ckpt)

with iohelper.open(out_path, 'wb') as f:
    torch.save(ckpt_adjusted, f)