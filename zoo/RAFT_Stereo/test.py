import os
import sys
# sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import OpenEXR
import Imath
from omegaconf import OmegaConf as oc
from tqdm import tqdm

from core.dataset_deepsl import SimplifiedStereoDatasetWithPattern, depth2disparity, to_tensor, rectify_images_simplified

'''
Conclusion: Matching will only succeed between L_Image and Pattern. Performance significant degrades between Pattern (as images1) and R_Image (as images2).  
Matching will fail between R_Image (as images1) and Pattern (as images2)
'''

DeepSL_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(DeepSL_dir)
from utils.visualize import vis_batch
from utils.transforms import corr_volume
from utils.metrics import zncc
from stereo.naive_matching import NaiveMatching

DEVICE = 'cuda'
def setseed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_dataset(args):
    dataset_cfgs = args.dataset_cfgs
    dataset_cfgs = oc.load(dataset_cfgs)
    oc.resolve(dataset_cfgs)
    dataset_cfgs.DATA_SPLIT = 'train'

    dataset = SimplifiedStereoDatasetWithPattern(dataset_cfgs, None)
    return dataset

def convert_imgs_to_gray(data:dict):
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(3,1,1)
    for k in data:
        if 'Image' in k:
            v = data[k]
            v = torch.sum(v*weights.to(v.device), dim=0, keepdim=True).expand_as(v)
            data[k] = v
    return data

def fetch_sample(file_fetcher, dataset:SimplifiedStereoDatasetWithPattern, idx, key=None, patt=None):
    if key is not None and patt is not None:
        pattern_to_fetch = patt
        flatten_key_to_fetch = key
    else:
        pattern_to_fetch = dataset.list_patterns[idx % dataset.num_patterns]
        flatten_key_to_fetch = file_fetcher.flatten_keys()[idx // dataset.num_patterns]
    data = file_fetcher.fetch(flatten_key_to_fetch, True, pattern_to_fetch, False, False)
    sample = {}
    for k, v in data.items():
        if 'L_Depth' in k:
            sample['L_Depth'] = v
        elif 'R_Depth' in k:
            sample["R_Depth"] = v
        elif 'L_Image' in k:
            sample['L_Image'] = v.transpose(2,0,1) * 255
        elif 'R_Image' in k:
            sample['R_Image'] = v.transpose(2,0,1) * 255  # go back to (0, 255)!
        else:
            sample[k] = v
    sample = to_tensor(sample)
    pat = dataset.patterns_images[pattern_to_fetch].permute(2,0,1)
    pat = rectify_images_simplified(pat.unsqueeze(0), sample['L_intri'], sample['P_intri'], False).squeeze_(0)
    sample['Pattern'] = pat * 255
    sample['P_intri'] = sample['L_intri']
    sample['key'] = [flatten_key_to_fetch]
    sample['pattern_name'] = [pattern_to_fetch]
    return convert_imgs_to_gray(sample)

@torch.no_grad()
def demo(args):
    setseed(args.seed)

    dataset = create_dataset(args)
    file_fetcher = dataset.file_fetcher
    len_ds = len(dataset)

    model = RAFTStereo(args)
    model.load_state_dict(torch.load(args.restore_ckpt))
    model.to(DEVICE)
    model.train()
    model.freeze_bn()
    # model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    padder = InputPadder((720, 1280), divis_by=32)
    for i in range(args.num_test):
        random_id = np.random.randint(0, len_ds)
        sample = fetch_sample(file_fetcher, dataset, random_id, 
                              args.key if args.key != "None" else None,
                              args.patt if args.patt !="None" else None)
        l_image, r_image, pattern = sample['L_Image'].cuda(), sample['R_Image'].cuda(), sample['Pattern'].cuda()
        # images1 = torch.stack((l_image, r_image), dim=0)
        # images2 = torch.stack((pattern, pattern), dim=0)
        # intri1 = torch.stack((sample['L_intri'], sample['R_intri']), dim=0).cuda()
        # extri1 = torch.stack((sample['L_extri'], sample['R_extri']), dim=0).cuda()
        # extri2 = torch.stack((sample['P_extri'], sample['P_extri']), dim=0).cuda()
        images1 = torch.stack((l_image, pattern), dim=0)
        images2 = torch.stack((pattern, r_image), dim=0)
        intri1 = torch.stack((sample['L_intri'], sample['P_intri']), dim=0).cuda()
        extri1 = torch.stack((sample['L_extri'], sample['P_extri']), dim=0).cuda()
        extri2 = torch.stack((sample['P_extri'], sample['R_extri']), dim=0).cuda()
        images1, images2 = padder.pad(images1, images2)
        _, flow_up = model(images1, images2, iters=args.valid_iters, test_mode=True)
        flow_up = padder.unpad(flow_up).squeeze()
        print(flow_up[0].min().item(), flow_up[0].max().item(), flow_up[1].min().item(), flow_up[1].max().item())
        depth = depth2disparity(flow_up, intri1, extri1, extri2).abs()
        l_depth, r_depth = torch.split(depth, 1, 0)

        sample['L_Image'] = sample['L_Image'].unsqueeze_(0) / 255
        sample['R_Image'] = sample['R_Image'].unsqueeze_(0) / 255
        sample['L_Depth'].unsqueeze_(0)
        sample['R_Depth'].unsqueeze_(0)
        vis_batch(sample, l_depth.cpu(), r_depth.cpu(), i, args.output_directory, 10, 0, loss_range=[-0.05, 0.05])

        import matplotlib.pyplot as plt
        images1 = l_image
        images2 = pattern
        intri1 = sample['L_intri'].unsqueeze(0).cuda()
        extri1 = sample['L_extri'].unsqueeze(0).cuda()
        extri2 = sample['P_extri'].unsqueeze(0).cuda()
        if images1.ndim == 3:
            images1.unsqueeze_(0)
        if images2.ndim == 3:
            images2.unsqueeze_(0)
        images1, images2 = padder.pad(images1, images2)

        if args.init_flow == 'yes':
            from kornia.filters import median_blur
            # fmap1, fmap2 = model.fnet([(2 * (images1 / 255.0) - 1.0).contiguous(), (2 * (images2 / 255.0) - 1.0).contiguous()])
            stereo_matcher = NaiveMatching(13)
            flow_init = stereo_matcher(images1, images2, intri1, None, extri1, extri2)
            flow_init = depth2disparity(flow_init.squeeze(1), intri1, extri1, extri2)
            flow_init = median_blur(flow_init.unsqueeze(1), 13).clamp(0, 192)
            # B,C,H,W = images1.shape
            # fmap1 = F.unfold(images1[:,:1], 13, padding=13//2, stride=1).view(-1, 13*13, H, W)
            # fmap2 = F.unfold(images2[:,:1], 13, padding=13//2, stride=1).view(-1, 13*13, H, W)
            # def corr_func(fmap1, fmap2):
            #     D, H, W1 = fmap1.shape[-3:]
            #     W2 = fmap2.shape[-1]
            #     corr = torch.einsum('...ijk,...ijh->...jkh', fmap1, fmap2)
            #     return corr / torch.sqrt(torch.tensor(D).float())
            # flow_init = corr_volume(fmap1, fmap2, 1, True, False, zncc, None, False)
            plt.imshow(flow_init.squeeze().cpu().numpy(), cmap='jet')
            plt.colorbar()
            plt.savefig("flow_init.png")
            plt.close()
            print(flow_init.shape, flow_init.min().item(), flow_init.max().item())
            h, w = flow_init.shape[-2:]
            flow_init = F.interpolate(flow_init, (h // 4, w // 4), mode='bilinear') / 4
            flow_init = torch.concat((-flow_init, torch.zeros_like(flow_init)), dim=1)
        else:
            flow_init = None

        maes = []
        for n_iters in tqdm(range(1, args.valid_iters + 1)):
            _, flow_up = model(images1, images2, iters=n_iters, test_mode = True, flow_init=flow_init)
            flow_up = padder.unpad(flow_up).squeeze().unsqueeze(0)
            depth = depth2disparity(flow_up, intri1, extri1, extri2).abs().squeeze()
            mae = torch.mean((depth - sample['L_Depth'].squeeze().cuda()).abs()).item()
            maes.append(mae)
        print(f"final MAE: {maes[-1]:.3f}")
        plt.plot(maes)  # 绘制折线图
        plt.xlabel("迭代次数-1")  # 设置 X 轴标签
        plt.ylabel("MAE")  # 设置 Y 轴标签
        plt.title(f"init_{args.init_flow}, {sample['key'][0]}")
        plt.savefig(os.path.join(output_directory, f"mae_n_iters_init_{args.init_flow}_{sample['key'][0]}.png".replace("/", "_")))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_cfgs", default="cfgs/train_local_nolr.yaml", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_test", default=1, type=int)
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default=None)
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default=None)
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--init_flow', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--key', type=str, default="None")
    parser.add_argument('--patt', type=str, default="None")

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)
