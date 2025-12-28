import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DistributedSampler, DataLoader
try:
    import imageio.v2 as imageio
except:
    import imageio
from PIL import Image
from io import BytesIO

from utils.common import to_tensor, FileIOHelper

from utils.transforms import load_exr, resize_image, resize_intrinsic_matrix


def rectify_images(pattern_img_path, camera_intrinsics, projector_intrinsics, baseline_ratio=0.51, normalized_intri: bool = False):
    """
    Rectify images so as to align the projector's intrinsic matrix to the
    camera's intrinsic matrix.

    Args:
        images:          numpy array of shape (H, W, C) – the input image(s).
        align_intri:     numpy array of shape (3, 3) – the 'alignment' intrinsic matrix.
        origin_intri:    numpy array of shape (3, 3) – the original camera/projector intrinsic matrix.
        align_h, align_w: integers – the desired output (height, width).
        normalized_intri: bool – whether the intrinsics are in normalized form
                                 and need to be scaled by image resolution.
    Returns:
        A numpy array of the rectified image(s), shape (align_h, align_w, C).
    """
    if not os.path.exists(pattern_img_path):
        raise ValueError(f"The file at path {pattern_img_path} does not exist.")
    align_h, align_w = 720, 1280  # 设置对齐后的图像的高度和宽度
    pattern_img = cv2.imread(pattern_img_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
    # change pattern image to np.array
    pattern_img = pattern_img.astype(np.float32) / 255.0
    ori_h, ori_w = pattern_img.shape[0],  pattern_img.shape[1] # 获取原图的高度和宽度
    pattern_img = torch.from_numpy(pattern_img).permute(2, 0, 1)  # 将图像转为 (C, H, W) 格式
    align_intri = torch.from_numpy(camera_intrinsics)
    origin_intri = torch.from_numpy(projector_intrinsics)
    if normalized_intri:
        align_scale = torch.tensor([[align_w], [align_h], [1]], dtype=align_intri.dtype)
        origin_scale = torch.tensor([[ori_w], [ori_h], [1]], dtype=origin_intri.dtype)
        align_intri = align_intri * align_scale  # 对投影仪的内参进行缩放
        origin_intri = origin_intri * origin_scale  # 对相机的内参进行缩放
    
    new_pix_y, new_pix_x = torch.meshgrid(
        torch.arange(align_h), 
        torch.arange(align_w),
        indexing='ij'
    )
    new_pix = torch.stack((new_pix_x, new_pix_y), dim=-1)

    homogeneous_coord = torch.concat( 
        (new_pix, torch.ones_like(new_pix[..., :1])), 
        dim=-1
    )  # => (align_h, align_w, 3)
    mat = torch.matmul(origin_intri, torch.linalg.inv(align_intri))  # shape (3, 3)
    proj_coord = torch.matmul(
        mat.view(1, 1, 3, 3).to(torch.float32),
        homogeneous_coord.view(align_h, align_w, 3, 1).to(torch.float32)
    ).squeeze(-1)[..., :2]  # => (align_h, align_w, 2)

    normalized_proj_coord = proj_coord / torch.tensor([ori_w, ori_h], dtype=proj_coord.dtype)
    rectified = F.grid_sample(
        pattern_img.unsqueeze(0).to(torch.float32),     # shape (1, C, H, W)
        2 * normalized_proj_coord.unsqueeze(0) - 1,  # shape (1, align_h, align_w, 2)
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    ).squeeze(0)  # => shape (C, align_h, align_w)

    # 13) Finally, permute back to (align_h, align_w, C) and convert to numpy.
    return rectified.permute(1, 2, 0).squeeze(-1).detach().numpy()


class DredsDataset(Dataset):
    cam_intri = np.array(
        [[892.62539334, 0., 640.],
         [0., 753.15267563, 360.],
         [0., 0.,           1.  ]], dtype=np.float32)
    proj_intri = np.array(
        [[486.59112686, -34.49562263, 406. ],
         [17.48271341,  494.93069154, 291.5],
         [0.,           0.,           1.   ]], dtype=np.float32)
    pattern_path = os.path.join(os.path.dirname(__file__), 'dreds_pattern', 'pattern.png')
    pattern_rectify_path = os.path.join(os.path.dirname(__file__), 'dreds_pattern', 'pattern_rectified.png')
    baseline = 0.055
    projector_baseline_rate = 0.51

    def __init__(self, dataset_path:str, split='train', normal:bool=False, material_type:bool=False):
        super().__init__()
        self.dataset_path = dataset_path
        self.split = split
        self.split_path = os.path.join(dataset_path, split)
        self.material_type = material_type
        self.normal = normal

        iohelper = FileIOHelper()
        assert split in ['train', 'val']  # val_real unsupported.
        self.left_ir_paths = []
        self.right_ir_paths = []
        self.depth_paths = []
        self.rgb_paths = []
        self.material_mask_paths = []
        self.normal_paths = []
        self.keys = []
        ids_dir = sorted(iohelper.listdir(self.split_path))

        for ids in ids_dir:
            id_path = os.path.join(self.split_path, ids)
            subid = 0
            while True:
                rgb_fname = f"{subid:04d}_color.png"
                left_fname= f"{subid:04d}_ir_l.png"
                right_fname=f"{subid:04d}_ir_r.png"
                depth_fname=f"{subid:04d}_depth_120.exr"
                mask_fname =f"{subid:04d}_mask.exr"
                normal_fname=f"{subid:04d}_normal.exr"
                key = f"{ids}-{subid:04d}"
                if not iohelper.exists(os.path.join(id_path, rgb_fname)):
                    break
                self.rgb_paths.append(os.path.join(id_path, rgb_fname))
                self.left_ir_paths.append(os.path.join(id_path, left_fname))
                self.right_ir_paths.append(os.path.join(id_path, right_fname))
                self.depth_paths.append(os.path.join(id_path, depth_fname))
                self.material_mask_paths.append(os.path.join(id_path, mask_fname))
                self.normal_paths.append(os.path.join(id_path, normal_fname))
                self.keys.append(key)
                subid += 1

        self.pattern_image = imageio.imread(self.pattern_path).astype(np.float32) / 255.
        # self.pattern_ractified_image = imageio.imread(self.pattern_rectify_path).astype(np.float32) / 255.
        self.pattern_ractified_image = np.repeat(rectify_images(
            self.pattern_path, self.cam_intri, self.proj_intri
        )[:,:,None], 3, axis=-1)
        self.T_cameras = np.array([self.baseline, 0, 0], dtype=np.float32)
        self.T_cam_proj= np.array([self.baseline * self.projector_baseline_rate, 0, 0], dtype=np.float32)

    def __len__(self):
        return len(self.keys)
    
    @staticmethod
    def load_image(img_path, iohelper:FileIOHelper):
        fmt = os.path.splitext(img_path)[-1][1:].lower()
        with iohelper.open(img_path, 'rb') as f:
            content = f.read()
        with BytesIO(content) as stream:
            img = np.array(Image.open(stream)).astype(np.uint8)
            # img = imageio.imread(stream, fmt)
        if img.ndim == 2:
            img = img.reshape(*img.shape, 1)
        img = np.repeat(img, 3, axis=-1)
        return img.astype(np.float32) / 255.
    
    def load_depth(path, iohelper:FileIOHelper):
        with iohelper.open(path, 'rb') as f:
            content = f.read()
        with BytesIO(content) as stream:
            depth = load_exr(stream)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth
    
    def __getitem__(self, index):
        sample = {}
        sample['key'] = self.keys[index]
        iohelper = FileIOHelper()
        # left, right, ir.
        sample['L_Image'] = DredsDataset.load_image(self.left_ir_paths[index], iohelper)
        sample['R_Image'] = DredsDataset.load_image(self.right_ir_paths[index], iohelper)
        # depth
        sample['L_Depth'] = DredsDataset.load_depth(self.depth_paths[index], iohelper)
        sample['R_Depth'] = np.zeros_like(sample['L_Depth'], dtype=np.float32)
        # pattern
        sample['Pattern'] = self.pattern_ractified_image
        sample['pattern_name'] = 'dreds_d435'
        # unify resolution
        ori_reso = (sample['L_Image'].shape[1], sample['L_Image'].shape[0])
        target_reso = (sample['L_Depth'].shape[1], sample['L_Depth'].shape[0])
        sample['L_Image'] = cv2.resize(sample['L_Image'], target_reso)
        sample['R_Image'] = cv2.resize(sample['R_Image'], target_reso)
        sample['Pattern'] = cv2.resize(sample['Pattern'], target_reso)
        resized_cam_intri = resize_intrinsic_matrix(
            torch.from_numpy(self.cam_intri), target_reso, ori_reso
        )

        # intri, extri.
        sample['L_intri'] = resized_cam_intri
        sample['R_intri'] = resized_cam_intri
        sample['P_intri'] = resized_cam_intri
        sample['L_extri'] = np.zeros((4,4), dtype=np.float32)
        sample['R_extri'] = np.zeros((4,4), dtype=np.float32)
        sample['R_extri'][0, -1] = self.baseline
        sample['P_extri'] = np.zeros((4,4), dtype=np.float32)
        sample['P_extri'][0, -1] = self.baseline * self.projector_baseline_rate     
        
        if self.material_type:
            raise NotImplementedError
        if self.normal:
            raise NotImplementedError
        
        return to_tensor(sample)
    
def create_dreds_dataloader(
        batch_size, num_workers, rank, world_size, 
        split:str, data_root:str, shuffle = True, gray = True, 
        normal=False, material_type=False, ddp:bool = True, **kwargs
    ):
    '''
    data_root: train/test/val所在的文件夹.  
    gray: 在加载时时否把图片转换为灰度图.  
    '''
    dataset = DredsDataset(data_root, split, normal, material_type)
    # print(len(dataset))
    if ddp:
        sampler = DistributedSampler(dataset, world_size, rank, shuffle, drop_last=(split == 'train'))
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size, shuffle if sampler is None else None, sampler, pin_memory=False, 
        drop_last=True if split == 'train' else False,
        num_workers=num_workers, # collate_fn=merge_all
    )  # 提供了sampler的时候不应设置shuffle.
    return dataloader


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--split", type=str, default='train', choices=['train', 'val'])
    args = parser.parse_args()

    dataloader = create_dreds_dataloader(4, 1, 0, 1, args.split, args.dataset_path, False)
    for sample in dataloader:
        for k, v in sample.items():
            print(k, v.shape if hasattr(v, 'shape') else v)
            if 'intri' in k or 'extri' in k:
                print(v.numpy())

        l_image = sample['L_Image']
        r_image = sample['R_Image']
        pat = sample['Pattern']
        l_depth = sample['L_Depth']

        for i in range(l_image.shape[0]):
            imageio.imwrite(f"{i}_l_image.png", np.uint8(l_image[i].numpy()*255))
            imageio.imwrite(f"{i}_r_image.png", np.uint8(r_image[i].numpy()*255))
            imageio.imwrite(f"{i}_pattern.png", np.uint8(pat[i].numpy() * 255))
            plt.imshow(l_depth[i].numpy().squeeze(), cmap='jet', vmin=l_depth[i].min().item(), vmax=l_depth[i].max().item())
            plt.colorbar()
            plt.savefig(f"{i}_l_depth.png")
            plt.close()
        break