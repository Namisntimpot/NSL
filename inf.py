import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import json
import argparse

from utils.common import load_config, to_device
from utils.visualize import apply_colormap

from zoo.neuralsl.sl_prompt_da import create_model

def load_img(path, pad=False):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)[...,None]
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img)
    img = img.permute(2,0,1).unsqueeze(0)
    if pad:
        h, w = img.shape[-2:]
        new_h = int(np.ceil(h / 14) * 14)
        new_w = int(np.ceil(w / 14) * 14)
        img = F.pad(img, (0, new_w - w, 0, new_h - h))
        return img, h, w
    else:
        return img

def load_param(path:str):
    if path.endswith(".json"):
        with open(path, 'r') as f:
            param = json.load(f)
        param = {k: np.array(v) for k, v in param.items()}
    elif path.endswith(".npy"):
        param = np.load(path, allow_pickle=True).tolist()
    
    param = {k: torch.from_numpy(v.astype(np.float32)).unsqueeze(0) for k, v in param.items()}
    return param

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgs", type=str)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--limg", type=str)
    parser.add_argument("--rimg", type=str, default=None)
    parser.add_argument("--patt", type=str, default=None)
    parser.add_argument("--param", type=str)
    parser.add_argument("--skip_refine", action='store_true')
    args = parser.parse_args()
    return args


@torch.no_grad()
def main(args):
    cfgs = load_config(args.cfgs)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    model_cfgs = cfgs.Model
    val_cfgs = cfgs.Val

    near = val_cfgs.get("near", val_cfgs.get("min_depth", 0.1))
    far  = val_cfgs.get("far",  val_cfgs.get("max_depth", 10.))
    val_fwd_kwargs = val_cfgs.fwd_kwargs

    # load images
    limg = load_img(args.limg, pad=False)
    rimg = load_img(args.rimg, pad=False) if args.rimg is not None else torch.zeros_like(limg)
    patt = load_img(args.patt, pad=False) if args.patt is not None else torch.zeros_like(limg)
    param = load_param(args.param)
    param.update({
        "L_Image": limg, "R_Image": rimg, "Pattern": patt,
        "near": near, "far": far,
        'skip_refine': args.skip_refine,
        **val_fwd_kwargs
    })
    param = to_device(param, device)

    model_cfgs.ckpt_path = args.weight
    model = create_model(return_ckpt=False, **model_cfgs)
    model.to(device)
    model.eval()

    l_depth = model.infer(**param)
    
    l_depth_color = apply_colormap(l_depth.cpu().squeeze().numpy())
    cv2.imwrite("depth.png", l_depth_color)


if __name__ == '__main__':
    args = parse_args()
    main(args)