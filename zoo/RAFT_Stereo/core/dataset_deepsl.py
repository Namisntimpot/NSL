
import os
import numpy as np
import h5py
from PIL import Image
import OpenEXR
import Imath
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from .dataset_utils.file_fetcher import BaseFileFetcher, LocalFileFetcher, OssFileFetcher
from .utils.augmentor import FlowAugmentor, MultiInputFlowAugmentor

from typing import Union
def to_tensor(obj: Union[np.ndarray, list, dict]):
    '''
    将np数组转为torch张量，不处理device
    '''
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    elif isinstance(obj, (list, tuple, set)):
        return [to_tensor(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: to_tensor(v) for k, v in obj.items()}
    else:
        return obj
    
def depth2disparity(depth, intri, extri, other_extri = None):
    # depth: [B, H, W]
    # intri: [B, 3, 3]
    # extri: [B, 4, 4]
    B, H, W = depth.shape
    assert intri.shape == (B, 3, 3)
    assert extri.shape == (B, 4, 4)
    # get the focal length
    f = intri[..., 0, 0]
    # get the baseline
    if other_extri is None:
        other_extri = torch.zeros_like(extri, dtype=extri.dtype, device=extri.device)
    b = torch.norm(extri[..., :3, 3] - other_extri[..., :3, 3], dim=-1)
    # b = extri[:, 0, 3]
    # get the disparity
    shape = f.shape + (1,)  * (depth.ndim - f.ndim)
    disparity = f.view(shape) * b.view(shape) / depth
    return disparity

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

class DeepslDataset(Dataset):
    def __init__(self, file_fetcher:BaseFileFetcher, gray:bool=True,
                 parameters=True, patternname=None, normal=True, materialtype=True):
        super().__init__()
        self.gray = gray
        self.file_fetcher = file_fetcher
        self.parameters = parameters
        self.patternname= patternname
        self.normal = normal
        self.materialtype = materialtype

    def __len__(self):
        return len(self.file_fetcher.flatten_keys())
    
    def __getitem__(self, index):
        key = self.file_fetcher.flatten_keys()[index]
        data = self.file_fetcher.fetch(key, self.parameters, self.patternname, self.normal, self.materialtype)
        data = to_tensor(data)
        if self.gray:
            self.convert_imgs_to_gray(data)
        data_newk ={}  # remove .xxx from data's keys
        for k, v in data.items():
            newk = k.split(".")[0]
            data_newk[newk] = v
        data_newk['key'] = key
        return data

    def convert_imgs_to_gray(self, data:dict):
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
        for k in data:
            if 'Image' in k:
                v = data[k]
                v = torch.sum(v*weights, dim=-1, keepdim=True).expand_as(v)
                data[k] = v
        return data

    def get_patterns_names(self):
        return self.file_fetcher.fetch_patterns_names()

    def get_pattern(self, pattern_name):
        return self.file_fetcher.fetch_pattern(pattern_name)
    
class SimplifiedStereoDataset(DeepslDataset):
    '''
    no normal, no materialtype, return {'L_Image':,'R_Image':,'L_Depth':, 'R_Depth':, 'L_intri'...}  
    only support 'images' mode, do not support 'decomposition' mode.  

    If the patternname parameter is not specified, images from each viewpoint under multiple patterns 
    will be traversed separately, 
    resulting in num_patterns * num_views sets of stereo data.
    '''
    def __init__(self, file_fetcher, gray=True, parameters=True, patternname=None, normal=False, materialtype=False):
        super().__init__(file_fetcher, gray, parameters, patternname, False, False)
        if self.patternname is None or self.patternname == 'proj':
            self.num_patterns = len(self.file_fetcher.fetch_patterns_names())
            self.list_patterns = self.file_fetcher.fetch_patterns_names()
        elif self.patternname == 'all':
            self.num_patterns = len(self.file_fetcher.fetch_patterns_names()) + 1
            self.list_patterns = self.file_fetcher.fetch_patterns_names() + ['noproj']
        else:
            self.num_patterns = 1
            self.list_patterns = [self.patternname]

    def __len__(self):
        l = super().__len__()
        return l * self.num_patterns

    def __getitem__(self, index):
        # if self.patternname is not None and not self.patternname in ['all', 'proj']:
        #     pattern_to_fetch = self.patternname
        #     flatten_key_to_fetch = self.file_fetcher.flatten_keys()[index]
        #     data = super().__getitem__(index)
        # else:
        pattern_to_fetch = self.list_patterns[index % self.num_patterns]
        flatten_key_to_fetch = self.file_fetcher.flatten_keys()[index // self.num_patterns]
        data = self.file_fetcher.fetch(
            flatten_key_to_fetch, self.parameters, pattern_to_fetch, self.normal, self.materialtype)
        data = to_tensor(data)
        if self.gray:
            data = self.convert_imgs_to_gray(data)
        for k in list(data.keys()):
            newk = k.split(".")[0] # if self.patternname is None else k 去掉文件后缀...
            prefix = newk[:2]
            if prefix != 'L_' and prefix != 'R_' and prefix != 'P_':
                newk = "_".join(newk.split("_")[1:])
            v = data.pop(k)
            data[newk] = v
        data['key'] = flatten_key_to_fetch
        data['pattern'] = pattern_to_fetch
        return data
    
# import objgraph  # debug
# import psutil

class SimplifiedStereoDatasetWithPattern(SimplifiedStereoDataset):
    def __init__(self, args, aug_params):
        split = args.DATA_SPLIT
        data_root = args.DATA_PATH
        decomp = args.DECOMPOSITION
        cleaned = args.CLEANED
        ftype = args.FILE_FETCHER_TYPE
        gray = args.GRAY
        parameters = args.PARAMETERS
        patternname = args.PATTERNNAME
        normal = args.NORMAL
        materialtype = args.MATERIALTYPE

        self.match_lr = args.match_lr

        if ftype == 'local':
            self.file_fetcher = LocalFileFetcher(split, data_root, decomp, cleaned)
        elif ftype == 'oss':
            self.file_fetcher = OssFileFetcher(split, data_root, decomp, cleaned)
        else:
            raise ValueError(f"Unknown file fetcher type: {ftype}")
        
        if patternname == 'noproj':
            assert self.match_lr, "match_lr must be True if you don't want to load images with pattern projected."

        super().__init__(self.file_fetcher, gray, parameters, patternname, normal, materialtype)
        # load patterns.
        self.patterns_images = {
            k: torch.from_numpy(self.file_fetcher.fetch_pattern(k)) for k in self.list_patterns if k!='noproj'
        }
        self.patterns_images['noproj'] = None

        # aug
        self.augmentor = None
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            self.augmentor = FlowAugmentor(**aug_params)

        # debug
        # self.interval = 100
        # self._cnt = 0

    def __getitem__(self, index):
        # debug
        # if self._cnt % self.interval == 0:
        #     memory_percent = psutil.virtual_memory().percent
        #     rss = psutil.Process().memory_info().rss / 1024.**2
        #     with open(f"obj{self._cnt:06d}_{rss:.2f}_{memory_percent:.2f}.txt", 'w') as f:
        #         objgraph.show_growth(limit=1000, file=f)
        # self._cnt += 1

        sample:dict = super().__getitem__(index)
        pattern_name = sample.pop("pattern")
        sample['Pattern'] = self.patterns_images[pattern_name]
        sample['pattern_name'] = pattern_name

        l_image, r_image, patt = sample['L_Image'], sample['R_Image'], sample['Pattern']
        dep = sample['L_Depth']
        l_intri, r_intri, p_intri = sample['L_intri'], sample['R_intri'], sample['P_intri']
        l_extri, r_extri, p_extri = sample['L_extri'], sample['R_extri'], sample['P_extri']
        del sample

        # specify l and r
        if not self.match_lr:
            # rectify.
            patt = rectify_images_simplified(
                patt.unsqueeze(0).permute(0, 3, 1, 2), l_intri, p_intri, False
            ).permute(0, 2, 3, 1).squeeze_(0)  # HWC
            p_intri = l_intri
            r_image = patt
            r_intri = p_intri
            r_extri = p_extri
        # dep2disp
        disp = depth2disparity(dep.unsqueeze(0), l_intri.unsqueeze(0), l_extri.unsqueeze(0), r_extri.unsqueeze(0)).squeeze_(0)
        disp = torch.stack([-disp, torch.zeros_like(disp)], dim=-1)  # (x,y)
        # aug module
        # first go back to numpy.
        l_image, r_image, disp = l_image.numpy(), r_image.numpy(), disp.numpy()
        l_image = (l_image * 255).astype(np.uint8)
        r_image = (r_image * 255).astype(np.uint8)
        if self.augmentor:
            l_image, r_image, disp = self.augmentor(l_image, r_image, disp, self.match_lr)

        l_image = torch.from_numpy(l_image).permute(2, 0, 1).float()
        r_image = torch.from_numpy(r_image).permute(2, 0, 1).float()
        disp = torch.from_numpy(disp).permute(2, 0, 1).float()

        valid = (disp[0].abs() < 512) & (disp[1].abs() < 512)  # HW

        if self.img_pad is not None:
            padH, padW = self.img_pad
            l_image = F.pad(l_image, [padW]*2 + [padH]*2)
            r_image = F.pad(r_image, [padW]*2 + [padH]*2)

        disp = disp[:1]
        
        return torch.ones(1,), l_image, r_image, disp, valid  # 第一个只是占位
    

class TriSimplifiedStereoDatasetWithPattern(SimplifiedStereoDataset):
    def __init__(self, args, aug_params):
        split = args.DATA_SPLIT
        data_root = args.DATA_PATH
        decomp = args.DECOMPOSITION
        cleaned = args.CLEANED
        ftype = args.FILE_FETCHER_TYPE
        gray = args.GRAY
        parameters = args.PARAMETERS
        patternname = args.PATTERNNAME
        normal = args.NORMAL
        materialtype = args.MATERIALTYPE
        assert patternname != 'noproj'

        if ftype == 'local':
            self.file_fetcher = LocalFileFetcher(split, data_root, decomp, cleaned)
        elif ftype == 'oss':
            self.file_fetcher = OssFileFetcher(split, data_root, decomp, cleaned)
        else:
            raise ValueError(f"Unknown file fetcher type: {ftype}")

        super().__init__(self.file_fetcher, gray, parameters, patternname, normal, materialtype)
        # load patterns.
        self.patterns_images = {
            k: torch.from_numpy(self.file_fetcher.fetch_pattern(k)) for k in self.list_patterns if k!='noproj'
        }
        self.patterns_images['noproj'] = None

        # aug
        self.augmentor = None
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            self.augmentor = MultiInputFlowAugmentor(**aug_params)


    def __getitem__(self, index):
        sample:dict = super().__getitem__(index)
        pattern_name = sample.pop("pattern")
        sample['Pattern'] = self.patterns_images[pattern_name]
        sample['pattern_name'] = pattern_name

        l_image, r_image, patt = sample['L_Image'], sample['R_Image'], sample['Pattern']
        dep = sample['L_Depth']
        l_intri, r_intri, p_intri = sample['L_intri'], sample['R_intri'], sample['P_intri']
        l_extri, r_extri, p_extri = sample['L_extri'], sample['R_extri'], sample['P_extri']
        del sample

        patt = rectify_images_simplified(
                    patt.unsqueeze(0).permute(0, 3, 1, 2), l_intri, p_intri, False
                ).permute(0, 2, 3, 1).squeeze_(0)  # HWC
        
        # dep2disp
        disp = depth2disparity(dep.unsqueeze(0), l_intri.unsqueeze(0), l_extri.unsqueeze(0), r_extri.unsqueeze(0)).squeeze_(0)
        disp = torch.stack([-disp, torch.zeros_like(disp)], dim=-1)  # (x,y)
        # aug module
        # first go back to numpy.
        l_image, r_image, patt, disp = l_image.numpy(), r_image.numpy(), patt.numpy(), disp.numpy()
        l_image = (l_image * 255).astype(np.uint8)
        r_image = (r_image * 255).astype(np.uint8)
        patt = (patt * 255).astype(np.uint8)
        if self.augmentor:
            l_image, r_image, patt, disp = self.augmentor(l_image, r_image, patt, disp)

        l_image = torch.from_numpy(l_image).permute(2, 0, 1).float()
        r_image = torch.from_numpy(r_image).permute(2, 0, 1).float()
        patt = torch.from_numpy(patt).permute(2, 0, 1).float()
        disp = torch.from_numpy(disp).permute(2, 0, 1).float()

        valid = (disp[0].abs() < 512) & (disp[1].abs() < 512)  # HW

        if self.img_pad is not None:
            padH, padW = self.img_pad
            l_image = F.pad(l_image, [padW]*2 + [padH]*2)
            r_image = F.pad(r_image, [padW]*2 + [padH]*2)

        disp = disp[:1]
        
        return l_image, r_image, patt, disp, valid