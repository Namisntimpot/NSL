import os
import sys
# sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import OpenEXR
import Imath

DEVICE = 'cuda'

def load_exr(path, type = "RGB", bit16 = False):
    '''
    type: "RGB", "NORMAL", "Z"
    '''
    exr_file = OpenEXR.InputFile(path)

    # 获取图像的宽度和高度
    dw = exr_file.header()['dataWindow']
    # print(exr_file.header()['channels'])
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT if not bit16 else Imath.PixelType.HALF)
    if type == 'RGB':
        ch = ["R", "G", "B"]
    elif type == 'NORMAL':
        ch = ["X", "Y", "Z"]
    else:
        ch = ["V"]
    channels = exr_file.channels(ch, FLOAT)
    dtype = np.float16 if bit16 else np.float32
    # 将数据转换为NumPy数组
    if type == 'Z':
        d = np.frombuffer(channels[0], dtype=dtype).reshape(height, width).astype(np.float32)
        return d.copy()
    r = np.frombuffer(channels[0], dtype=dtype).reshape(height, width).astype(np.float32)
    g = np.frombuffer(channels[1], dtype=dtype).reshape(height, width).astype(np.float32)
    b = np.frombuffer(channels[2], dtype=dtype).reshape(height, width).astype(np.float32)

    # 如果需要，你可以将RGB值合并为一个图像
    image = np.stack([r, g, b], axis=-1)
    return image.copy()   # 原buffer是只读的, 需要copy一下变成可写的.

def load_exr_to_torch(path, type="RGB"):
    '''
    type: RGB, NORMAL, Z
    '''
    img = load_exr(path, type, bit16=True).astype(np.float32)
    return torch.from_numpy(img)

def load_image(imfile):
    '''float32, range (0, 255)'''
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if img.ndim == 2:
        img = torch.from_numpy(img[None].repeat(3,1,1)).float()
    elif img.shape[-1] == 3:
        img = torch.from_numpy(img).permute(2,0,1).float()
    elif img.shape[-1] == 1:
        img = torch.from_numpy(img[None,:,:,0].repeat(3,1,1)).float()
    return img[None].to(DEVICE)

def load_sample(sample_dir:str, sid:int, vid:int, patname:str):
    from utils.transforms import rectify_images_simplified
    pattern = load_image(os.path.join(sample_dir, f"{patname}.png"))
    l_image = load_image(os.path.join(sample_dir, f"{sid:05d}_{vid:03d}_{patname}_L_Image.png"))
    r_image = load_image(os.path.join(sample_dir, f"{sid:05d}_{vid:03d}_{patname}_R_Image.png"))
    l_depth = load_exr_to_torch(os.path.join(sample_dir, f"{sid:05d}_{vid:03d}_L_Depth.exr"), "Z")
    r_depth = load_exr_to_torch(os.path.join(sample_dir, f"{sid:05d}_{vid:03d}_R_Depth.exr"), "Z")
    params = np.load(os.path.join(sample_dir, f'{sid:05d}_parameters.npz'), allow_pickle=True)['arr_0'].tolist()
    l_intri = torch.from_numpy(params['intrinsic']['L'].astype(np.float32))
    r_intri = torch.from_numpy(params['intrinsic']['R'].astype(np.float32))
    p_intri = torch.from_numpy(params['intrinsic']['Proj'].astype(np.float32))
    l_extri = torch.from_numpy(params['extrinsic'][vid]['L'].astype(np.float32))
    r_extri = torch.from_numpy(params['extrinsic'][vid]['R'].astype(np.float32))
    p_extri = (l_extri + r_extri) / 2
    # rectify_pattern
    pattern = rectify_images_simplified(
        pattern, l_intri, p_intri, False
    )
    p_intri = l_intri
    # batchify.
    sample = {
        'L_Image': l_image.unsqueeze_(0), 'R_Image': r_image.unsqueeze_(0),
        'L_Depth': l_depth.unsqueeze_(0), 'R_Depth': r_depth.unsqueeze_(0),
        'Pattern': pattern.unsqueeze_(0),
        'L_intri': l_intri.unsqueeze_(0), 'R_intri': r_intri.unsqueeze_(0), 'P_intri': p_intri.unsqueeze_(0),
        'L_extri': l_extri.unsqueeze_(0), 'R_extri': l_extri.unsqueeze_(0), 'P_extri': l_extri.unsqueeze_(0), 
    }



def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    if args.left_imgs is None or args.right_imgs is None:
        DeepSL_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        sys.path.append(DeepSL_dir)
        from utils.visualize import vis_batch
        vis_func = vis_batch

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()

            file_stem = imfile1.split('/')[-2]
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
            plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepsl-dir", type=str, default='samples/')
    parser.add_argument("--sid", type=int)
    parser.add_argument("--vid", type=int)
    parser.add_argument("--pad", type=str)
    parser.add_argument("--img-pat", action='store_true')

    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default=None)
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default=None)
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="instance", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)
