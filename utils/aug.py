# Data preprocessing and augmentations.
import torch
import torch.nn as nn
from collections.abc import Iterable
import torchvision.transforms as torchtrans
from torchvision.transforms.functional import gaussian_blur

from .transforms import (random_crop_image, transpose_unsqueeze_image, crop_intrinsic_matrix, 
                         normalize_intrinsic_matrix, gamma_correct_image, rectify_images_simplified,
                         d2d_transform, depth_to_interp_disp, border_pad_image, resize_image, resize_intrinsic_matrix,
                         depth_conversion, unnormalize_intrinsic_matrix)
from .common import get_range_random

class Compose:
    def __init__(self, *augs):
        self.augs = augs
    
    def forward(self, sample:dict):
        # b = len(sample['key'])
        b = sample['L_Image'].shape[0]
        limg, rimg, ldep, rdep, patt = sample['L_Image'], sample['R_Image'], sample['L_Depth'], sample['R_Depth'], sample['Pattern']
        l_extri, r_extri, p_extri = sample['L_extri'], sample['R_extri'], sample['P_extri']
        l_intri, r_intri, p_intri = sample['L_intri'], sample['R_intri'], sample['P_intri']
        imgs = torch.concat((limg, rimg), dim=0).contiguous()
        deps = torch.concat((ldep, rdep), dim=0).contiguous()
        c_extri = torch.concat((l_extri, r_extri), dim=0).contiguous()
        c_intri = torch.concat((l_intri, r_intri), dim=0).contiguous()
        near = sample['near']
        far = sample['far']
        for aug in self.augs:
            imgs, deps, patt, c_extri, p_extri, c_intri, p_intri = aug.forward(imgs, deps, patt, c_extri, p_extri, c_intri, p_intri, b, near, far)
        sample['L_Image'], sample['R_Image'] = imgs[:b], imgs[b:]
        sample['Pattern'] = patt
        sample['L_Depth'], sample['R_Depth'] = deps[:b], deps[b:]
        sample['L_extri'], sample['R_extri'], sample['P_extri'] = c_extri[:b], c_extri[b:], p_extri
        sample['L_intri'], sample['R_intri'], sample['P_intri'] = c_intri[:b], c_intri[b:], p_intri
        return sample
    
    def __call__(self, sample:dict):
        return self.forward(sample)
    
class Transpose:
    def __init__(self, gray_single_channel:bool=True):
        '''
        HWC -> CHW or HW -> 1HW  
        applied to both images and depth.  
        all the input must be batched, so the first dimension must be B-dimension.  
        gray_single_channel: whether only reserve one channel when loading gray images. if true, 
        all images loaded will be regarded as grayscale images, and only one channel will be reserved.
        '''
        self.single_channel = gray_single_channel

    def forward(self, imgs, deps, patt, c_extri, p_extri, c_intri, p_intri, batchsize, near, far):
        imgs = transpose_unsqueeze_image(imgs)
        deps = transpose_unsqueeze_image(deps)
        patt = transpose_unsqueeze_image(patt)
        if self.single_channel:
            imgs, deps, patt = imgs[:,:1], deps[:,:1], patt[:,:1]
        return imgs, deps, patt, c_extri, p_extri, c_intri, p_intri

    def __call__(self, *x):
        return self.forward(*x)
    

class Padpatch:
    def __init__(self, patchsize, ori_w:int=1280, ori_h:int=720,normalized_intri:bool=False, channel_last:bool=False):
        self.normalized_intri = normalized_intri
        self.channel_last = channel_last
        self.osize = (ori_w, ori_h)
        mid_w, mid_h = ori_w // 2, ori_h // 2
        self.tw = (ori_w // patchsize) * patchsize
        self.th = (ori_h // patchsize) * patchsize
        self.tsize = (self.tw ,self.th)
        self.start_w = max(0, mid_w - self.tw // 2)
        self.start_h = max(0, mid_h - self.th // 2)
        self.crop_region = (self.start_h, self.start_w, self.th, self.tw)

    def forward(self, imgs, deps, patt, c_extri, p_extri, c_intri, p_intri, batchsize, near, far):
        imgs_to_crop = torch.concat((imgs, patt), dim=0)
        intris_to_crop = torch.concat((c_intri, p_intri), dim=0)
        imgs_to_crop = random_crop_image(
            imgs_to_crop, self.tsize, self.channel_last, crop_region=self.crop_region
        )
        deps = random_crop_image(deps, self.tsize, self.channel_last, crop_region=self.crop_region)
        if not self.normalized_intri:
            intris_to_crop = normalize_intrinsic_matrix(
                intris_to_crop, ori_reso=self.osize
            )
        intris_to_crop = crop_intrinsic_matrix(
            intris_to_crop, self.crop_region, ori_reso=self.osize
        )
        imgs = imgs_to_crop[:-batchsize].contiguous()
        patt = imgs_to_crop[-batchsize:].contiguous()
        c_intri = intris_to_crop[:-batchsize]
        p_intri = intris_to_crop[-batchsize:]
        return imgs, deps, patt, c_extri, p_extri, c_intri, p_intri
    

class CeilPadpatch:
    def __init__(self, patchsize, ori_w:int=1280, ori_h:int=720, 
                 change_intri:bool=False, normalized_intri:bool=False, channel_last:bool=False,
                 pad_depth:bool=False):
        '''It should keep intrinsics unchanged!'''
        import math
        self.patchsize = patchsize
        self.normalized_intri = normalized_intri
        self.channel_last = channel_last
        self.osize = (ori_w, ori_h)
        self.tw = math.ceil(ori_w / patchsize) * patchsize
        self.th = math.ceil(ori_h / patchsize) * patchsize
        self.tsize = (self.tw ,self.th)
        self.change_intri = change_intri
        self.pad_depth = pad_depth

    def forward(self, imgs, deps, patt, c_extri, p_extri, c_intri, p_intri, batchsize, near, far):
        imgs_to_pad = torch.concat((imgs, patt), dim=0)
        intris_to_pad = torch.concat((c_intri, p_intri), dim=0)
        imgs_to_pad = border_pad_image(imgs_to_pad, self.tsize, self.channel_last)
        if self.change_intri:
            if not self.normalized_intri:
                intris_to_pad = normalize_intrinsic_matrix(intris_to_pad, self.osize)
            intris_to_pad = crop_intrinsic_matrix(intris_to_pad, (0,0,self.th,self.tw), self.osize)
            if not self.normalized_intri:
                intris_to_pad = unnormalize_intrinsic_matrix(intris_to_pad, self.tsize)

        imgs = imgs_to_pad[:-batchsize]
        patt = imgs_to_pad[-batchsize:]
        c_intri = intris_to_pad[:-batchsize]
        p_intri = intris_to_pad[-batchsize:]
        if self.pad_depth:
            deps = border_pad_image(deps, self.tsize, self.channel_last)
        return imgs, deps, patt, c_extri, p_extri, c_intri, p_intri
    

class RandomCrop:
    def __init__(self, target_h:int, target_w:int, ori_w:int=1280, ori_h:int=720,
                 normalized_intri:bool=False, channel_last:bool=False, to_normalize_intri:bool=True):
        '''
        crop image, depth, and then adjust intrinsic matrix accordingly.   
        normalized: whether the input intrinsic matrix is normalized.  
        For now we apply the same crop region to images and patterns.  

        **IMPORTANT**: It will normalize intrinsic matrix if set to_normalize_intri to True.
        '''
        self.th = target_h
        self.tw = target_w
        self.tsize = (target_w, target_h)
        self.osize = (ori_w, ori_h)
        self.normalized_intri = normalized_intri
        self.channel_last = channel_last
        self.to_normalize_intri = to_normalize_intri

    def forward(self, imgs, deps, patt, c_extri, p_extri, c_intri, p_intri, batchsize, near, far):
        imgs_to_crop = torch.concat((imgs, patt), dim=0)
        intris_to_crop = torch.concat((c_intri, p_intri), dim=0)
        imgs_to_crop, crop_region = random_crop_image(
            imgs_to_crop, self.tsize, self.channel_last, get_crop_region=True
        )
        deps = random_crop_image(deps, self.tsize, self.channel_last, get_crop_region=False, crop_region=crop_region)
        if not self.normalized_intri:
            intris_to_crop = normalize_intrinsic_matrix(
                intris_to_crop, ori_reso=self.osize
            )
        intris_to_crop = crop_intrinsic_matrix(
            intris_to_crop, crop_region, ori_reso=self.osize
        )
        if not self.to_normalize_intri and not self.normalized_intri:
            intris_to_crop = unnormalize_intrinsic_matrix(intris_to_crop, self.tsize)
        imgs = imgs_to_crop[:-batchsize].contiguous()
        patt = imgs_to_crop[-batchsize:].contiguous()
        c_intri = intris_to_crop[:-batchsize]
        p_intri = intris_to_crop[-batchsize:]
        return imgs, deps, patt, c_extri, p_extri, c_intri, p_intri
    
    def __call__(self, *x):
        return self.forward(*x)
    

class Resize:
    def __init__(self, target_h:int, target_w:int, resize_depth:bool=True, ori_w:int=1280, ori_h:int=720,
                 normalized_intri:bool=False, channel_last:bool=False, mode:str='bilinear'):
        self.tsize = (target_w, target_h)  # w,h
        self.osize = (ori_w, ori_h)
        self.normalized_intri = normalized_intri
        self.channel_last = channel_last
        self.mode = mode
        self.resize_depth = resize_depth
    
    def forward(self, imgs, deps, patt, c_extri, p_extri, c_intri, p_intri, batchsize, near, far):
        imgs = resize_image(
            imgs, self.tsize, self.channel_last, self.mode, align_corners=False
        )
        patt = resize_image(
            patt, self.tsize, self.channel_last, self.mode, align_corners=False
        )
        if self.resize_depth:
            deps = resize_image(
                deps, self.tsize, self.channel_last, self.mode, align_corners=False
            )
        if not self.normalized_intri:
            c_intri = resize_intrinsic_matrix(c_intri, self.tsize, self.osize)
            p_intri = resize_intrinsic_matrix(p_intri, self.tsize, self.osize)

        return imgs, deps, patt, c_extri, p_extri, c_intri, p_intri

    

class GaussianBlur:
    def __init__(self, ksize:int|list, sigma:float|list):
        self.ksize = ksize
        self.sigma = sigma

    def forward(self, imgs, deps, patt, c_extri, p_extri, c_intri, p_intri, batchsize, near, far):
        ksize = get_range_random(*self.ksize) if isinstance(self.ksize, Iterable) else self.ksize
        ksize = (int(ksize) | int(1))
        sigma = get_range_random(*self.sigma) if isinstance(self.sigma, Iterable) else self.sigma
        imgs = gaussian_blur(imgs, ksize, sigma)
        return imgs, deps, patt, c_extri, p_extri, c_intri, p_intri
    
    def __call__(self, *x):
        return self.forward(*x)
    
class RandomGammaCorrection:
    def __init__(self, gamma:float|list):
        '''
        the input images should be within range(0,1)  
        '''
        self.gamma = gamma

    def forward(self, imgs, deps, patt, c_extri, p_extri, c_intri, p_intri, batchsize, near, far):
        gamma = get_range_random(*self.gamma) if isinstance(self.gamma, Iterable) else self.gamma
        imgs = gamma_correct_image(imgs, gamma)
        return imgs, deps, patt, c_extri, p_extri, c_intri, p_intri
    
    def __call__(self, *x):
        return self.forward(*x)
    
class RectifyPatterns:
    def __init__(self, normalized_intri:bool=True):
        self.normalized_intri = normalized_intri
    def forward(self, imgs, deps, patt, c_extri, p_extri, c_intri, p_intri, batchsize, near, far):
        b = p_intri.shape[0]
        if not torch.all(c_intri[:b] == p_intri):
            patt = rectify_images_simplified(
                patt, c_intri[:b], p_intri, self.normalized_intri
            )  # l_intri, r_intri are the same.
            p_intri.copy_(c_intri[:b])
        return imgs, deps, patt, c_extri, p_extri, c_intri, p_intri

# TODO: 换成更统一的depth_conversion模块.
class D2DTransformation:
    '''
    Depth -> Disparity, or Disparity -> Depth.  
    Note that the disparity is defined in camera-projector system.  
    '''
    def __init__(self, disp_type:str = 'rel_disp'):
        '''
        disp_type: rel_disp or abs_disp. rel_disp: disp = 1/depth; abs: disp = fx*baseline / depth.
        '''
        self.disp_type = disp_type
    def forward(self, imgs, deps, patt, c_extri, p_extri, c_intri, p_intri, batchsize, near, far):
        deps = depth_to_interp_disp(
            deps, near, far, c_intri, c_extri, p_extri, self.disp_type
        )
        return imgs, deps, patt, c_extri, p_extri, c_intri, p_intri
    
class DepthConversion:
    def __init__(self, depth_type:str, match_lr:bool=False):
        '''match_lr: baseline defined as l-r cameras (true) or camera-projector (false).'''
        self.depth_type = depth_type
        self.match_lr = match_lr
    def forward(self, imgs, deps, patt, c_extri, p_extri, c_intri, p_intri, batchsize, near, far):
        if not self.match_lr:
            extri_other = torch.concat((p_extri, p_extri), dim=0)
        else:
            extri_other = torch.concat((c_extri[batchsize:], c_extri[:batchsize]), dim=0)
        deps = depth_conversion(
            deps, self.depth_type, False,
            near, far, c_intri, c_extri, extri_other, deps, None,  None
        )
        return imgs, deps, patt, c_extri, p_extri, c_intri, p_intri
    
class Identity:
    def __init__(self):
        pass
    def forward(self, sample):
        return sample
    def __call__(self, sample):
        return sample        
    

def parse_augmentation(*aug_cfg):
    '''
    aug_cfg should be a list.  
    '''
    if len(aug_cfg) == 0:
        return Identity()
    def parse_single_augmentation(cfg:dict):
        ty, kwargs = list(cfg.items())[0]
        c = eval(ty)
        return c(**kwargs) if kwargs is not None else c()
    augs = [parse_single_augmentation(cfg) for cfg in aug_cfg]
    return Compose(*augs)