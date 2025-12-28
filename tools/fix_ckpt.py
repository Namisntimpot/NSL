import torch

def dav2_to_ours(ckpt:dict):
    key_to_adjust = []
    for k in ckpt:
        if k.startswith("depth_head.projects") or k.startswith("depth_head.resize_layers"):
            key_to_adjust.append(k)
    for k in key_to_adjust:
        param = ckpt.pop(k)
        module_path:list = k.split(".")
        module_path.insert(1, 'reassemble_block')
        newk = ".".join(module_path)
        ckpt[newk] = param
    return ckpt


def unwrap_ddp_ckpt(ckpt:dict):
    newckpt = {}
    for k, v in ckpt.items():
        newk = ".".join(k.split(".")[1:]) if k.startswith("module.") else k
        newckpt[newk] = v
    return newckpt