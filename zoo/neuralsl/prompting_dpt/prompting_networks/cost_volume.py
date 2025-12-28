import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from dataclasses import dataclass

from utils.transforms import corr_volume, d2d_transform, depth_to_interp_disp
from utils.metrics import inner_product, zncc

def corr_func_dot(img1:torch.Tensor, img2:torch.Tensor):
    '''
    img: (B,G,C,H,W)
    '''
    B,G,C,H,W1 = img1.shape
    return inner_product(img1, img2) / np.sqrt(C)

def reduce_func_softmax(volume: torch.Tensor, left_first:bool, get_prob, temperature:float=1., geometry_mask:bool=True):
    '''
    volume: B,G(1),H,W1,W2
    '''
    W1, W2 = volume.shape[-2:]
    if geometry_mask:
        mask = torch.tril(torch.ones((W1,W2), dtype=volume.dtype, device=volume.device)) \
                if left_first else \
                torch.triu(torch.ones((W1,W2), dtype=volume.dtype, device=volume.device))
    else:
        mask = 1
    idx = torch.arange(W2, device=volume.device, dtype=volume.dtype)
    prob = torch.softmax(volume / temperature, dim=-1) * mask  # B,G(1),H,W1,W2
    if get_prob:
        return torch.sum(prob * idx, dim=-1), prob
    return torch.sum(prob * idx, dim=-1)  # ...,H,W1

class CostVolume(nn.Module):
    def __init__(self, corr_func_type='dot', temperature=1., dual:bool=False, pack_lr:bool=False, geometry_mask:bool=True):
        '''
        暂不支持限制condidate disparity范围  
        pack_lr: 当dual为True的时候，如果pack_lr为Ture，表示前半是l后半是r  
        '''
        super().__init__()
        self.dual = dual
        self.temperature = temperature
        self.pack_lr = pack_lr
        if corr_func_type == 'dot':
            self.corr_func = corr_func_dot
        elif corr_func_type == 'zncc':
            self.corr_func = zncc
        else:
            raise NotImplementedError(f"Not implemented corr_func: {corr_func_type}")
        self.reduce_func = partial(reduce_func_softmax, temperature=temperature, geometry_mask=geometry_mask)

    def corresp2disp(self, corresp:torch.Tensor, normalize:bool=True):
        B,*_,H,W = corresp.shape
        idx = torch.arange(W, dtype=corresp.dtype, device=corresp.device)
        disp = torch.abs(corresp - idx)
        if normalize:
            disp = disp / W
        return disp
    
    def corresp_prob_to_depth_conf(
            self, corresp, prob, intri1, intri2, extri1, extri2, near, far, normalize
        ):
        confidence = torch.max(prob, dim=-1)[0]  # (B,1,H,W)
        disp = self.corresp2disp(corresp, normalize)  # (B,1,H,W)
        disp = torch.clamp(disp, 1e-1)  # 防止去到0...
        depth = d2d_transform(disp, intri1, intri2, extri1, extri2)
        return depth, confidence

    def forward(
            self, img1:torch.Tensor, img2:torch.Tensor,
            intri1:torch.Tensor, intri2:torch.Tensor, extri1:torch.Tensor, extri2:torch.Tensor,
            near=None, far=None,
            normalized_intri:bool = True # , left_first:list=None
        ):
        '''
        img1, img2: (B,C,H,W)  
        for efficiency, img1 must be the left image.  
        normalized_intri: if True, disparity should be normalized to (0,1) before depth computation.
        '''
        # 如果dual为True, 则认为img1为L而Img2为R，运行dual的情况即可；
        # 如果dual为False且pack_lr为False, 认为img1为L而img2为R，not dual的情况即可；
        # 但如果dual为False且pack_lr为True, 则img1应该展开为(B//2,2,C,H,W), 其中[:,0]是L而[:,1]是R；
        # 相应的img2也应该展开为(B//2, 2, C, H, W), 其中[:,0]为R而[:,1]为L.  
        if not self.dual and not self.pack_lr:
            corresp, prob = corr_volume(
                img1, img2, 1, reduce=True, reduce_dual=False, corr_func=self.corr_func,
                reduce_func=self.reduce_func, get_prob=True
            )  # corresp: (B,1,H,W), prob: (B,1,H,W,W)
            return self.corresp_prob_to_depth_conf(corresp, prob, intri1, intri2, extri1, extri2, near, far, normalized_intri)
        else:
            corresp1, prob1, corresp2, prob2 = corr_volume(
                img1, img2, 1, reduce=True, reduce_dual=True, corr_func=self.corr_func, 
                reduce_func=self.reduce_func, get_prob=True
            )
            dep1, conf1 = self.corresp_prob_to_depth_conf(
                corresp1, prob1, intri1, intri2, extri1, extri2, near, far, normalized_intri
            )
            dep2, conf2 = self.corresp_prob_to_depth_conf(
                corresp2, prob2, intri2, intri1, extri2, extri1, near, far, normalized_intri
            )
            return dep1, conf1, dep2, conf2 

@dataclass
class CostVolumePromptCfg:
    corr_func_type:str = 'dot'
    temperature:float = 1.
    geometry_mask:bool = True
    dual:bool = False
    pack_lr: bool = False   # 只在dual=True的时候有效，l_feat中前半是l, 后半是r.


class CostVolumePrompt(CostVolume):
    def __init__(self, num_context_views, cfg:CostVolumePromptCfg, d_in:int, depth_type:str = 'norm_depth'):
        self.num_context_views = num_context_views
        self.cfg = cfg
        self.pack_lr = cfg.pack_lr
        self.depth_type = depth_type
        self.out_dim = 2
        if self.num_context_views != 2:
            raise NotImplementedError(f"Only support num_context_view==2")
        geometry_mask = cfg.geometry_mask if hasattr(cfg, 'geometry_mask') else True
        super().__init__(cfg.corr_func_type, cfg.temperature, cfg.dual, cfg.pack_lr, geometry_mask)

    def quary_support_bino_imgs(self):
        return True

    def depth_conversion(self, dep:torch.Tensor, intri, extri1, extri2, near, far):
        '''
        dep: (B,1,H,W), near&far: (B,)  
        '''
        if self.depth_type.endswith("disp"):
            return depth_to_interp_disp(dep, near, far, intri, extri1, extri2, self.depth_type)
        elif self.depth_type == 'norm_depth':
            return dep / far[...,None,None,None]
        return dep
    
    def reorder(self, lfeat:torch.Tensor, rfeat:torch.Tensor):
        B, *shape = lfeat.shape
        lfeat = lfeat.view(B//2, 2, *shape)  # [:,0] are left, [:,1] are right
        rfeat = rfeat.view(B//2, 2, *shape)  # [:,0] are right,[:,1] are left  
        # real_left = torch.stack((lfeat[:,0], rfeat[:,0]), dim=1).view(B,*shape)
        # real_right= torch.stack((rfeat[:,1], lfeat[:,1]), dim=1).view(B,*shape)
        real_left = torch.stack((lfeat[:,0], rfeat[:,1]), dim=1).view(B,*shape)
        real_right= torch.stack((rfeat[:,0], lfeat[:,1]), dim=1).view(B,*shape)
        return real_left, real_right
    
    def forward(
            self, feat:torch.Tensor, extri:torch.Tensor, intri:torch.Tensor, 
            near, far, **kwargs
        ):
        '''
        feat: (B,V,C,H,W)  extri: (B,V,4,4)  intri: (B,V,3,3)  near&far: (B,V)  
        '''
        # (B,V)比较复杂，如果是dual，则两边都要比较，就是(B,0)为L而(B,1)为R, 直接设置corr_volume的dual为True即可，得到L,R的匹配结果并返回;  
        # 如果不是dual，则只用匹配L视图，如果pack_lr为False，则认为(B,0)确实是L，直接设置corr_volume的dual为False即可, 得到单个匹配结果并返回；  
        # 如实不是dual且pack_lr为True，则比较特殊(对应和pattern匹配的情况)，此时(B,0)不一定是L,R，而(B,1)根据(B,0)的左右区分;  
        # (B,0)是从 (B//2, 2)展平而来的, (B//2,0,0)是L，(B//2,0,1)是R, 相应的(B//2,1,0)是R而(B//2,1,1)是L.  
        B = feat.shape[0]
        lfeat, rfeat = feat.unbind(1)  # (B,C,H,W)  
        lextri, rextri = extri.unbind(1)
        lintri, rintri = intri.unbind(1)
        lnear, rnear = near.unbind(1)  # near, far are shared between left and right.
        lfar, rfar = far.unbind(1)
        normalized_intri = torch.all(intri[...,0,0] < 50)   # 小于一个数
        # ret = super().forward(
        #     lfeat, rfeat, lintri, rintri, lextri, rextri, near, far, normalized_intri
        # )  # pack_lr时, lfeat是image, rfeat是pattern.  
        if not self.dual and not self.pack_lr:  # 只匹配单边且lfeat全是真的l
            ldepth, lconf = super().forward(lfeat, rfeat, lintri, rintri,
                    lextri, rextri, near, far, normalized_intri)  # (B,1,H,W)
            ldepth_conversed = self.depth_conversion(ldepth, lintri, lextri, rextri, lnear, lfar)
            extra_info = {
                'depth': ldepth, 'conf': lconf
            }
            return torch.concat((ldepth_conversed, lconf), dim=1).unsqueeze(1), extra_info # (B,1,2(C),H,W)  
        elif self.dual:
            ldepth, lconf, rdepth, rconf = super().forward(lfeat, rfeat, lintri, rintri,
                    lextri, rextri, near, far, normalized_intri)  # ldepth: (B,1,H,W), conf: (B,1,H,W)  
            ldepth_conversed = self.depth_conversion(ldepth, lintri, lextri, rextri, lnear, lfar)
            rdepth_conversed = self.depth_conversion(rdepth, rintri, rextri, lextri, rnear, rfar)
            lfeat = torch.concat((ldepth_conversed, lconf), dim=1)  # (B,2,H,W)
            rfeat = torch.concat((rdepth_conversed, rconf), dim=1)  # (B,2,H,W)
            extra_info = {
                'depth': torch.concat((ldepth, rdepth), dim=0),
                'conf': torch.concat((lconf, rconf), dim=0)
            }
            return torch.stack((lfeat, rfeat), dim=1), extra_info     # (B,2,2,H,W)
        else:  # 需要正确处理左右.
            real_lfeat, real_rfeat = self.reorder(lfeat, rfeat)
            real_lintri,real_rintri= self.reorder(lintri, rintri)
            real_lextri,real_rextri= self.reorder(lextri, rextri)
            
            ldepth, lconf, rdepth, rconf = super().forward(
                real_lfeat, real_rfeat, real_lintri, real_rintri,
                real_lextri, real_rextri, near, far, normalized_intri
            )

            ldepth = ldepth.view(B//2, -1, *ldepth.shape[1:])  # 都是(B//2,2,1,H,W), [:,0]是L而[:,1]是R
            rdepth = rdepth.view(B//2, -1, *rdepth.shape[1:])
            lconf = lconf.view(B//2, -1, *lconf.shape[1:])
            rconf = rconf.view(B//2, -1, *rconf.shape[1:])
            depth = torch.stack([ldepth[:,0], rdepth[:,1]], dim=1).view(B, *ldepth.shape[-3:]) #(B,1,H,W)
            conf = torch.stack([lconf[:,0], rconf[:,1]], dim=1).view(B, *ldepth.shape[-3:])

            # depth = torch.concat([ldepth[:B//2], rdepth[B//2:]], dim=0)
            # conf = torch.concat([lconf[:B//2], rconf[B//2:]], dim=0)
            depth_conversed = self.depth_conversion(depth, lintri, lextri, rextri, lnear, lfar)
            extra_info = {
                'depth': depth, 'conf': conf
            }
            return torch.concat((depth_conversed, conf), dim=1).unsqueeze(1), extra_info # (B,1,2(C),H,W)  