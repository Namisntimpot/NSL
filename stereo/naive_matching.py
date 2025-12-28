import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.transforms import corr_volume, d2d_transform
from utils.metrics import zncc

class NaiveMatching(nn.Module):
    def __init__(self, ksize:int = 13):
        super().__init__()
        self.ksize = ksize
        self.pad = ksize // 2

    def forward(
            self, l_image, r_image, l_intri, r_intri, l_extri, r_extri
        ):
        '''
        l_image, r_image: B,C,H,W  
        *_intri: B,3,3, this function assumes that l+intri == r_intri  
        *_extri: B,4,4  
        '''
        H,W = l_image.shape[-2:]
        l_image_unfold = F.unfold(l_image, self.ksize, padding=self.pad).view(1,-1,H,W)  # (1,wnd_size^2,H,W)
        r_image_unfold = F.unfold(r_image, self.ksize, padding=self.pad).view(1,-1,H,W)
        corresp = corr_volume(l_image_unfold, r_image_unfold,reduce=True, corr_func=zncc)  # (B,1,H,W)
        disp = torch.abs(corresp - torch.arange(W, device=corresp.device, dtype=corresp.dtype))
        # disp_2image = corresp
        depth = d2d_transform(disp, l_intri, None, l_extri, r_extri)
        return depth
    

if __name__ == '__main__':
    import imageio.v2 as imageio
    def load_pattern(patname):  # return (1,H,W)
        img = imageio.imread(f"deepsl_data/data/patterns/{patname}.png").astype(np.float32) / 255.
        if img.ndim == 3:
            return img[None,:,:,0]
        else:
            return img[None]
    import os
    import numpy as np
    import megfile
    import io
    from utils.common import load_images
    from utils.visualize import vis_batch
    from utils.transforms import rectify_images_simplified
    # test_dir = "s3://ljh-deepsl-data/test/data/images"
    test_dir = "s3://ljh-deepsl-data/data/images"
    output_dir = "s3://ljh-deepsl-data/test_naive_matching/"
    scene_id = 2562
    view_id = 3
    pattern = 'D415'
    beta = torch.tensor(100.)
    sigmaColor = torch.tensor(0.02)         # color para of bilateral filter
    sigmaSpace = torch.tensor(3.)

    scene_dir = os.path.join(test_dir, f'{scene_id:05d}')
    param_file = os.path.join(scene_dir, 'parameters.npz')

    # stereo matching module
    matching = NaiveMatching(ksize=13)
    
    # load param
    with megfile.smart_open(param_file, 'rb') as f:
        content = f.read()
    with io.BytesIO(content) as stream:
        params = np.load(stream, allow_pickle=True)['arr_0'].tolist()
    l_intri = torch.from_numpy(params['intrinsic']['L'].astype(np.float32))
    r_intri = torch.from_numpy(params['intrinsic']['R'].astype(np.float32))
    p_intri = torch.from_numpy(params['intrinsic']['Proj'].astype(np.float32))
    l_extri = torch.from_numpy(params['extrinsic'][view_id]['L'].astype(np.float32))
    r_extri = torch.from_numpy(params['extrinsic'][view_id]['R'].astype(np.float32))

    baseline_2cam = torch.norm(l_extri[...,:3,3] - r_extri[...,:3,3])
    baseline_cam_proj = baseline_2cam / 2.
    fx = l_intri[...,0,0]

    # load gt_depth
    l_depth = torch.from_numpy(load_images(os.path.join(scene_dir, f"{view_id:03d}_L_Depth.exr"), type='Z', bit16=True)).unsqueeze(0).unsqueeze(0)
    r_depth = torch.from_numpy(load_images(os.path.join(scene_dir, f"{view_id:03d}_R_Depth.exr"), type='Z', bit16=True)).unsqueeze(0).unsqueeze(0)
    
    # load pattern
    pat = torch.from_numpy(load_pattern(pattern)).unsqueeze(0)
    pat = rectify_images_simplified(pat, l_intri, p_intri, normalized_intri=False)

    # load images.
    rgb2gray = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    l_image = load_images(os.path.join(scene_dir, f"{view_id:03d}_{pattern}_L_Image.png")).astype(np.float32) / 255.
    l_image = np.sum(l_image * rgb2gray, axis=-1, keepdims=True)
    l_image = l_image[None,:,:,0] if l_image.ndim == 3 else l_image[None,:,:]
    l_image = torch.from_numpy(l_image).unsqueeze(0)
    r_image = load_images(os.path.join(scene_dir, f"{view_id:03d}_{pattern}_R_Image.png")).astype(np.float32) / 255.
    r_image = np.sum(r_image * rgb2gray, axis=-1, keepdims=True)
    r_image = r_image[None,:,:,0] if r_image.ndim == 3 else r_image[None,:,:]
    r_image = torch.from_numpy(r_image).unsqueeze(0)  # (1,1,H,W)
    H,W = r_image.shape[-2:]

    # match between 2 images.
    dep_2image = matching.forward(
        l_image, r_image, l_intri, r_intri, l_extri, r_extri
    )
    # match between image and pattern.
    dep_img_pat = matching.forward(
        l_image, pat, l_intri, l_intri, l_extri, (l_extri + r_extri) / 2
    )

    batch = {
        'L_Depth': l_depth, 'R_Depth': l_depth,    # both are l_depth!
        'L_Image': l_image, 'R_Image': pat,
        'key': [f"{scene_id:05d}/{view_id:03d}"],
        'pattern_name': [pattern],
    }

    vis_batch(
        batch, dep_2image, dep_img_pat, 0, output_dir, 10, None, share_depth_range=True
    )