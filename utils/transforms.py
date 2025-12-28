from numbers import Number
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Ftrans
from einops import rearrange
try:
    from ..deepsl_data.scripts.utils_transform import *
except:
    from deepsl_data.scripts.utils_transform import *

from .metrics import inner_product

def d2d_transform(
        d:torch.Tensor, self_intri:torch.Tensor, 
        other_intri:torch.Tensor, self_extri:torch.Tensor, other_extri:torch.Tensor,
        eps:float = 1e-8
    ):
    '''
    disparity -> depth or depth -> disparity.  
    d: disparity or depth . (B,(1),H,W)  
    *_intri: Transformation matrix from camera coordinate to pixel coordinate (B,3,3),   
    *_extri: Transformation matrix from world coordinate to camera coordinate (B,3,3), w2c  

    **This function assumes that intrinsics are equal and there's no relative rotation**  

    return: depth map. (B,(1),H,W)  
    '''
    t_self = self_extri[..., :3, 3]  # (B, 3)
    t_other = other_extri[..., :3, 3]
    baseline = torch.norm(t_other - t_self, dim=-1)  # (B,)  

    f = self_intri[..., 0, 0]  # (B,)
    shape = f.shape + (1,) * (d.ndim - 1)
    numerator = (f * baseline).view(shape)
    return numerator / (d + eps)


def depth_to_interp_disp(depth, near, far, intri, extri1, extri2, disp_type:str):
    shape = (depth.shape[0],) + (1,) * (depth.ndim - 1)
    n = 1 / (near + 1e-10)
    f = 1 / (far + 1e-10)
    if isinstance(n, torch.Tensor):
        n = n.view(shape)
    if isinstance(f, torch.Tensor):
        f = f.view(shape)
    if disp_type == 'rel_disp':
        disp = 1 / (depth + 1e-10)
        return 1 - (disp - f) / (n - f)
    else:
        fx = intri[...,0,0]
        baseline = torch.norm(extri1[...,:3,3] - extri2[...,:3,3], dim=-1)
        numerator = (fx * baseline).view(shape)
        n = n*numerator
        f = f*numerator
        disp = numerator / (depth + 1e-10)
        return 1 - (disp - f) / (n - f)
    
def interp_disp_to_depth(disp, near, far, intri, extri1, extri2, disp_type:str):
    shape = (disp.shape[0],) + (1,) * (disp.ndim - 1)
    n = 1 / (near + 1e-10)
    f = 1 / (far + 1e-10)
    if isinstance(n, torch.Tensor):
        n = n.view(shape)
    if isinstance(f, torch.Tensor):
        f = f.view(shape)
    if disp_type == 'rel_disp':
        disp = (1 - disp) * (n - f) + f
        return 1 / (disp + 1e-10)
    else:
        fx = intri[...,0,0]
        baseline = torch.norm(extri1[...,:3,3] - extri2[...,:3,3], dim=-1)
        numerator = (fx * baseline).view(shape)
        n = n*numerator
        f = f*numerator
        disp = (1 - disp) * (n - f) + f
        return numerator / (disp + 1e-10)


def rectify_images_simplified(
        images:torch.Tensor, align_intri:torch.Tensor, origin_intri:torch.Tensor, normalized_intri:bool=True
    ):
    '''
    Rectify images so as to align the projector's intrinsic to the camera's  
    This function only considers a simplifed case where the proj and the cam have the same resolution.  
    images: (B,(C),H,W). Must be batched. If images.ndim==3, it will be considered as missing C dim instead of B dim.  
    *_intri: (B, 3, 3), camera space -> pixel space.  
    '''
    b = images.shape[0]
    ori_pat_dim = images.ndim
    if ori_pat_dim == 3:
        images = images.unsqueeze(1)  # (B,1,H,W)  
    h, w = images.shape[-2:]
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    pix = torch.stack((x, y), dim=-1).to(torch.float32).to(images.device)
    if normalized_intri:
        n = torch.tensor([w, h], dtype=torch.float32, device=images.device)
        pix = pix / n  # range (0,1)
    homogeneous_coord = torch.concat((pix, torch.ones_like(pix[...,:1], dtype=torch.float32, device=pix.device)), dim=-1) # (h,w,3)
    mat = torch.matmul(origin_intri, torch.linalg.inv(align_intri)) # (B, 3, 3)
    proj_coord = torch.matmul(mat.view(b,1,1,3,3), homogeneous_coord.view(h,w,3,1)).squeeze(-1)[..., :2]  # (b, h, w, 2)  
    if not normalized_intri:
        n = torch.tensor([w, h], dtype=torch.float32, device=images.device)
        proj_coord = proj_coord / n
    rectified = F.grid_sample(
        images, 2*proj_coord-1, mode='bilinear', padding_mode='zeros', align_corners=False
    )  # (b,c,h,w)
    return rectified.squeeze(1) if ori_pat_dim == 3 else rectified
    

def transpose_image(img:torch.Tensor, channel_last:bool = True):
    '''
    HWC -> CHW, or vice versa.  
    if channel_last is True, the img is in shape of HWC
    '''
    c_dim = 2 if channel_last else 0
    h_dim = 0 if channel_last else 1
    w_dim = h_dim + 1
    if len(img.shape) == 4:
        b_dim = 0
        c_dim, h_dim, w_dim = c_dim + 1, h_dim + 1, w_dim + 1
    if channel_last:   # (B) HWC -> (B) CHW
        return img.permute(b_dim, c_dim, h_dim, w_dim) if len(img.shape) == 4 else img.permute(h_dim, w_dim, c_dim)
    else:   # (B) CHW -> (B) HWC
        return img.permute(b_dim, h_dim, w_dim, c_dim) if len(img.shape) == 4 else img.permute(h_dim, w_dim, c_dim)
    
def transpose_unsqueeze_image(img:torch.Tensor, channel_last:bool=True):
    '''
    BHWC -> BCHW, or vice versa.  
    the input img must be batched, so the first dim must be B-dim.  
    if img.ndim==3, this function will unsqueeze a C dim.  
    if channel_last is True, the input img is in shape of BHWC
    '''
    if img.ndim == 3:
        img = img.unsqueeze(dim=-1 if channel_last else 1)
    if channel_last:  # BHWC -> BCHW
        return img.permute(0, 3, 1, 2)
    else: # BCHW -> BHWC
        return img.permute(0, 2, 3, 1)

def normalize_image(img:torch.Tensor, mean:list, std:list):
    n_channels = len(mean)
    mean = torch.tensor(mean, dtype=img.dtype, device=img.device)
    std = torch.tensor(std, dtype=img.dtype, device=img.device)
    if img.shape[-1] == n_channels:  # BHWC
        return (img - mean) / (std + 1e-6)
    else:  # BCHw
        mean = mean.view(-1,1,1)
        std = std.view(-1,1,1)
        return (img - mean) / (std + 1e-6)
    
def denormalize_image(img:torch.Tensor, mean:list, std:list):
    n_channels = len(mean)
    mean = torch.tensor(mean, dtype=img.dtype, device=img.device)
    std = torch.tensor(std, dtype=img.dtype, device=img.device)
    if img.shape[-1] != n_channels:  # BCHW
        mean = mean.view(-1,1,1)
        std = std.view(-1,1,1)
    return img * std + mean

def resize_image(img:torch.Tensor, target_reso:tuple, channel_last:bool=False, mode='bilinear', align_corners = False):
    '''
    img: ((B),H,W,C) or ((B),C,H,W), img.ndim must be 3 or 4.  
    target_reso: (w, h)
    '''
    target_w, target_h = target_reso
    batched = True
    if img.ndim == 3:
        batched = False
        img = img.unsqueeze(0)
    if channel_last:
        img = img.permute(0, 3, 1, 2)
    if mode == 'nearest':
        img = F.interpolate(img, size=(target_h, target_w), mode=mode)
    else:
        img = F.interpolate(img, size=(target_h, target_w), mode=mode, align_corners=align_corners)
    if channel_last:
        img = img.permute(0, 2, 3, 1)
    if not batched:
        img = img.squeeze(0)
    return img

def resize_intrinsic_matrix(intri:torch.Tensor, target_reso:tuple, ori_reso:tuple=(1280, 720)):
    '''
    intri: (..., 3, 3)
    '''
    tar_x, tar_y = target_reso
    ori_x, ori_y = ori_reso
    sx = tar_x / ori_x
    sy = tar_y / ori_y
    scale = torch.tensor([
        [sx, 1, sx],
        [1, sy, sy],
        [1,1,1]
    ], dtype=intri.dtype, device=intri.device)
    return intri * scale

    
def center_crop_image(img:torch.Tensor, target_reso:tuple, channel_last:bool = False):
    '''
    target_reso: (target_w, target_h), e.g. (1280, 720)  
    channel_last: if True, img is in shape (BHWC)
    '''
    target_w, target_h = target_reso
    if channel_last:
        *_, h, w, c = img.shape
    else:
        *_, c, h, w = img.shape
     # 计算裁剪区域
    start_x = (w - target_w) // 2
    start_y = (h - target_h) // 2
    end_x = start_x + target_w
    end_y = start_y + target_h
    if channel_last:
        cropped_img = img[..., start_y:end_y, start_x:end_x, :]
    else:
        cropped_img = img[..., start_y:end_y, start_x:end_x]
    return cropped_img

def random_crop_image(imgs:torch.Tensor, target_reso:tuple, channel_last:bool=False, get_crop_region=False, crop_region=None):
    '''
    randomly crop input imgs into target_reso.   
    imgs: (B,..., C, H, W) or (B, ..., H, W, C). Dimension B is a must.  
    this function will apply the same crop region to all the batched imgs.  
    this function assumes that target_reso < origin_reso.  
    target_reso: (target_w, target_h), e.g. (1280, 720)  
    channel_last: if True, img is in shape (BHWC)  

    group_region returned: (y-axis of top-left, x-axis of top-left, height, width). 
    the image's top-left is (0,0)  
    '''
    if channel_last:
        imgs = rearrange(imgs, '... H W C -> ... C H W')
    B, *S, C, H, W = imgs.shape
    target_w, target_h = target_reso
    if crop_region is None:
        start_h = torch.randint(0, H - target_h + 1, size=(1,)).item()
        start_w = torch.randint(0, W - target_w + 1, size=(1,)).item()
        crop_region = (start_h, start_w, target_h, target_w)
    else:
        start_h, start_w = crop_region[:2]
    imgs = Ftrans.crop(imgs, start_h, start_w, target_h, target_w)
    if channel_last:
        imgs = rearrange(imgs, "..., C H W -> ..., H W C")
    if get_crop_region:
        return imgs, crop_region
    else:
        return imgs
    
def normalize_intrinsic_matrix(intri:torch.Tensor, ori_reso:tuple=(1280, 720)):
    '''
    This function convert an intrinsic matrix that transforms camera space to
    image's pixel coordinate (0\~height, 0\~weight)  to an intrinsic matrix that transforms camera
    space to image's normalized coordinate within range (0, 1).  
    shape of input intri: (..., 3, 3)  
    This function assumes that all the intri inputed have the same origin resolution.  
    '''
    scale = torch.tensor([[ori_reso[0]],[ori_reso[1]], [1]], dtype=intri.dtype, device=intri.device)
    return intri / scale

def unnormalize_intrinsic_matrix(intri:torch.Tensor, reso:tuple):
    scale = torch.tensor([[reso[0]],[reso[1]], [1]], dtype=intri.dtype, device=intri.device)
    return intri * scale
    
def crop_intrinsic_matrix(intri:torch.Tensor, crop_region:tuple, ori_reso:tuple=(1280, 720)):
    '''
    intri: **normalized** intrinsic matrix. (..., 3, 3)  
    crop_region: (y-axis of top-left, x-axis of top-left, height, width)  
    '''
    ori_w, ori_h = ori_reso
    new_h, new_w = crop_region[2:]
    topleft_y, topleft_x = crop_region[:2]
    scale = torch.tensor(
        [
            [ori_w / new_w], [ori_h / new_h], [1.]
        ], dtype=intri.dtype, device=intri.device
    )
    offset = -torch.tensor(
        [
            [0,0,topleft_x / new_w], [0,0,topleft_y / new_h], [0.,0.,0.]
        ], dtype=intri.dtype, device=intri.device
    )
    return intri * scale + offset
    

def border_pad_image(imgs:torch.Tensor, target_reso:tuple, channel_last:bool = False):
    batched = True
    if imgs.ndim == 3:
        batched = False
        imgs = imgs.unsqueeze(0)
    if channel_last:
        imgs = imgs.permute(0, 3, 1, 2)
    b, c, h, w = imgs.shape
    target_w, target_h = target_reso
    pad_w = target_w - w
    pad_h = target_h - h
    imgs = F.pad(imgs, [0, pad_w, 0, pad_h], mode='replicate')
    if channel_last:
        imgs = imgs.permute(0, 2, 3, 1)
    if not batched:
        imgs = imgs.squeeze(0)
    return imgs

def gamma_correct_image(imgs:torch.Tensor, gamma:float):
    '''
    the input imgs should be within range (0, 1)  
    '''
    imgs = torch.pow(imgs, gamma)
    imgs = torch.clamp(imgs, 0., 1.)
    return imgs


def corr_volume(img1:torch.Tensor, img2:torch.Tensor, group:int=1, 
                reduce:bool = False, reduce_dual:bool = False,
                corr_func = None, reduce_func = None, get_prob:bool = False):
    '''
    img1, img2: torch.Tensor, (B,C,H,W1), (B,C,H,W2)  
    corr_func((...,C,H,W1),(...,C,H,W2) -> (...,H,W1,W2))  
    reduce_func((...,H,W1,W2), left_first(bool), get_prob) -> (...,H,W1) / tuple((...,H,W1),prob(...,H,W1,W2))
    left_first: torch.tril if left_first else torch.triu. img1 is the left one if left_first is True  
    
    reduce: whether to reduce.  
    reduce both: reduce both (...,H,W1,W2) and (...,H,W2,W1)  
    '''
    def default_reduce_func(volume:torch.Tensor, left_first:bool, get_prob=False, temperature=0.01,):
        '''
        volume: (B,G,H,W1,W2)
        '''
        W1, W2 = volume.shape[-2:]
        mask = torch.tril(torch.ones((W1,W2), dtype=volume.dtype, device=volume.device)) \
               if left_first else \
               torch.triu(torch.ones((W1,W2), dtype=volume.dtype, device=volume.device))
        idx = torch.arange(W2, device=volume.device, dtype=volume.dtype)
        prob = torch.softmax(volume * mask / temperature, dim=-1)
        if get_prob:
            return torch.sum(prob * idx, dim=-1), prob
        return torch.sum(prob * idx, dim=-1)  # ...,H,W1
    
    corr_func = inner_product if corr_func is None else corr_func
    reduce_func = default_reduce_func if reduce_func is None else reduce_func
    
    B,C,H,W1 = img1.shape
    W2 = img2.shape[-1]
    img1 = img1.view(B,group,-1,H,W1)
    img2 = img2.view(B,group,-1,H,W2)  # (B,G,C//G,H,W2)
    volume = corr_func(img1, img2)     # (B,G,H,W1,W2)  
    if reduce:
        reduce1 = reduce_func(volume, True, get_prob)
        if not reduce_dual:
            return reduce1     # (B,G,H,W1)
        else:
            reduce2 = reduce_func(volume.permute(0,1,2,4,3), False, get_prob)
            return *reduce1, *reduce2
    else:
        return volume  # B,G, H, W1, W2


def lcn_image(imgs:torch.Tensor, wnd_size:int=9, channel_last:bool = False):
    '''
    imgs must be gray images, channel dim must be reserved.
    '''
    ndim = imgs.ndim
    if ndim == 3:
        imgs = imgs.unsqueeze(0)
    if channel_last:
        imgs = imgs.permute(0, 3, 1, 2) 
    b, c, h, w = imgs.shape
    unfold_imgs = F.unfold(imgs, wnd_size, padding=wnd_size // 2) # (b, wnd_size^2, h*w)  
    std, mean = torch.std_mean(unfold_imgs, dim=1, keepdim=True)  # (b, 1, h*w)  
    std = std.view(b, 1, h, w)
    mean = mean.view(b, 1, h, w)
    lcn = (imgs - mean) / (std + 1e-6)
    if channel_last:
        lcn = lcn.permute(0, 2, 3, 1)
    if ndim == 3:
        lcn = lcn.squeeze(0)
    return lcn

def depth_metric_to_rel(depth:torch.Tensor, valid_mask:torch.Tensor = None):
    '''
    convert gt_depth to rel_depth  
    warning: the calculation method for relative depth from gt depth may differ between different papers,
    here we implement the one used in DepthAnything.  
    '''
    batched = True
    if depth.ndim == 2:
        batched = False
        depth = depth.unsqueeze(0)
    B, h, w = depth.shape
    if valid_mask is None or torch.all(valid_mask == 1):
        depth_flat = depth.flatten(1, 2)
        depth_median = torch.median(depth_flat, dim=-1).reshape(B,1,1) # (B,1,1)
        n_valid_pix = h*w
    else:
        if not batched:
            valid_mask = valid_mask.unsqueeze(0)
        n_valid_pix = torch.sum(valid_mask, dim=(1,2))  # (B,)
        n_invalid_pix = h * w - n_valid_pix             # (B,)
        median_idx = ((n_valid_pix // 2) + n_invalid_pix).to(torch.int32) # (B,)
        depth_exclude_invalid = depth * valid_mask
        depth_exclude_invalid = depth_exclude_invalid.flatten(1, 2)
        sorted_depth_flat = torch.sort(depth_exclude_invalid, dim=-1)
        depth_median = sorted_depth_flat[torch.arange(B), median_idx].reshape(B, 1, 1)  # B,1,1
    depth_scale = (torch.sum(torch.abs(depth - depth_median), dim=(1,2)) / n_valid_pix).reshape(B,1,1)   # B,1,1
    return (depth - depth_median) / depth_scale

class LCN:
    def __init__(self, wnd_size = 9, channel_last = False):
        self.wnd_size = wnd_size
        self.channel_last = channel_last
    def forward(self, img:torch.Tensor):
        return lcn_image(img, self.wnd_size, self.channel_last)
    def __call__(self, img):
        return self.forward(img)
    

def normalize_depth_use_min_max(*depth_to_normalize, reference_depth, dmin=None, dmax=None, denorm:bool=False):
    '''
    depths: (B,(1),H,W)  
    return: *depth_normalized, min (B,), max (B,).  
    denorm: denormalization or not.  
    dmin, dmax: if provided, reference_depth is not needed, can be None.  

    **warning**: INF will still be INF after normalization!  
    '''
    if dmin is None or dmax is None:
        assert reference_depth is not None, "No tensor assigned to dmin or dmax or both, reference_depth must be provided!"
        B = reference_depth.shape[0]
        ref_for_max = torch.nan_to_num(reference_depth, nan=-torch.inf, posinf=-torch.inf).view(B,-1)
        ref_for_min = torch.nan_to_num(reference_depth, nan=torch.inf, neginf=torch.inf).view(B, -1)
        dmin = torch.min(ref_for_min, dim=1)[0]
        dmax = torch.max(ref_for_max, dim=1)[0]  # (B,)
    shape = dmin.shape + (1,) * (depth_to_normalize[0].ndim - 1)
    dmin = dmin.view(shape)
    dmax = dmax.view(shape)
    if not denorm:
        return [(dep - dmin) / (dmax - dmin) for dep in depth_to_normalize], dmin.squeeze(), dmax.squeeze()
    else:
        return [dep * (dmax - dmin) + dmin for dep in depth_to_normalize], dmin.squeeze(), dmax.squeeze()
    

def depth_linear_scale(*depth_to_normalize, reference_depth, dmin=None, dmax=None):
    '''
    depths: (B,(1),H,W)  
    return: *depth_normalized, min (B,), max (B,).  
    denorm: denormalization or not.  
    dmin, dmax: if provided, reference_depth is not needed, can be None.  

    **WARNING**: INF will always be INF.
    '''
    if dmin is None or dmax is None:
        assert reference_depth is not None, "No tensor assigned to dmin or dmax or both, reference_depth must be provided!"
        B = reference_depth.shape[0]
        ref_for_max = torch.nan_to_num(reference_depth, nan=-torch.inf, posinf=-torch.inf).view(B,-1)
        ref_for_min = torch.nan_to_num(reference_depth, nan=torch.inf, neginf=torch.inf).view(B, -1)
        dmin = torch.min(ref_for_min, dim=1)[0]
        dmax = torch.max(ref_for_max, dim=1)[0]  # (B,)
    dep_for_max = [torch.nan_to_num(dep, nan=-torch.inf, posinf=-torch.inf).view(B,-1) \
                       for dep in depth_to_normalize]
    dep_for_min = [torch.nan_to_num(dep, nan=torch.inf, neginf=torch.inf).view(B, -1) \
                    for dep in depth_to_normalize]
    shape = dmin.shape + (1,) * (depth_to_normalize[0].ndim - 1)
    dmin = dmin.view(shape)
    dmax = dmax.view(shape)
    dep_to_norm_max = [torch.max(dep, dim=1)[0].view(shape) for dep in dep_for_max]
    dep_to_norm_min = [torch.min(dep, dim=1)[0].view(shape) for dep in dep_for_min]

    return [(dep - dep_min) / (dep_max - dep_min) * (dmax - dmin) + dmin  \
            for dep, dep_min, dep_max in \
            zip(depth_to_normalize, dep_to_norm_min, dep_to_norm_max)], \
            dmin.squeeze(), dmax.squeeze()


def depth_conversion(
        d:torch.Tensor, mode:str, reverse:bool=False,
        near=None, far=None, intri=None, extri1=None, extri2=None, 
        ref_depth=None, dmin=None, dmax=None):
    '''
    mode: no, norm_depth, min_max_ref_depth, linear_scale_ref_depth, rel_disp, abs_disp,
    raw_disp  
    raw_disp is the real disparity (fb/depth), rel_disp and abs_disp are interpolated disparity between near and far.  
    no: no change. norm_depth: depth / far. 
    '''
    if mode == 'no':
        return d
    elif mode == 'norm_depth':
        shape = far.shape + (1,) * (d.ndim - far.ndim)
        d = d / far.view(shape) if not reverse else d * far.view(shape)
        return d if reverse else d.clamp(0, 1)
    elif mode == 'min_max_ref_depth':
        d, dmin, dmax = normalize_depth_use_min_max(d, reference_depth=ref_depth, dmin=dmin, dmax=dmax, denorm=reverse)
        return d[0] if reverse else d[0].clamp(0, 1)
    elif mode == 'linear_scale_ref_depth':
        d, dmin, dmax = depth_linear_scale(d, reference_depth=ref_depth, dmin=dmin, dmax=dmax)
        shape = dmin.shape + (1,) * (d.ndim - dmin.ndim)
        return torch.clamp(d[0], dmin.view(shape), dmax.view(shape))
    elif mode == 'rel_disp' or mode == 'abs_dist':
        if not reverse:
            return depth_to_interp_disp(d, near, far, intri, extri1, extri2, mode).clamp(0,1)
        else:
            return interp_disp_to_depth(d, near, far, intri, extri1, extri2, mode)
    elif mode == 'raw_disp':
        return d2d_transform(d, intri, None, extri1, extri2).abs()  # TODO: should the sign of disparity be considered here?
    else:
        raise NotImplementedError(f"unknown depth type: {mode}")
    

def unproject(K:torch.Tensor, dep:torch.Tensor):
    '''K: (B),3,3; dep: (B),h,w'''
    h, w = dep.shape[-2:]
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij') #(h, w)
    grid = torch.stack([grid_x, grid_y], dim=-1).to(dep.device) ## (h,w,2); x,y
    grid_homogenous = torch.concat([grid, torch.ones((h, w, 1), dtype=dep.dtype, device=dep.device)], dim=-1).unsqueeze(-2) # (h,w,1,3)
    Kinv = torch.linalg.inv(K).reshape(K.shape[0], 1, 1, 3, 3)  # B,1,1,3,3
    unproj = torch.sum(Kinv * grid_homogenous, dim=-1)  # (B, h, w, 3)
    unproj = unproj * dep.unsqueeze(-1)
    return unproj  # ((B),h,w,3)


def project(K:torch.Tensor, points:torch.Tensor):
    '''K: ((B), 3, 3); points: (...., 3), 需要x,y,z'''
    proj = torch.sum(K * points.unsqueeze(-2), dim=-1)  (..., 3)
    proj = proj / proj[..., 2:3]
    return proj[..., :2]


def align_depthmap_to_other_view(
        depth:torch.Tensor, K_ori_view:torch.Tensor, K_new_view:torch.Tensor, R:torch.Tensor, T:torch.Tensor, h, w, max_depth=10.
    ):
    from .rasterize import project_pointclouds_to_depthmap
    points = unproject(K_ori_view, depth.squeeze(1))
    zbuf = project_pointclouds_to_depthmap(points, K_new_view, R, T, h, w)
    mask = (zbuf > 0) & (zbuf < max_depth)
    zbuf = zbuf * mask
    zbuf = zbuf[..., 0]
    if depth.ndim == 4:
        return zbuf.unsqueeze(1)
    else:
        return zbuf