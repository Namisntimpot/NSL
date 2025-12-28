from easydict import EasyDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from numbers import Number

from ...raft.raft_stereo import RAFTStereo
from ...raft.dual_corr_raft_stereo import DualCorrRAFTStereo
from ...raft.utils.utils import InputPadder
from utils.transforms import d2d_transform, depth_conversion
from utils.common import FileIOHelper
from utils.dist import print_ddp, barrier, get_rank
from tools.fix_ckpt import unwrap_ddp_ckpt

def print_module_type(module:nn.Module):
    for n, m in module.named_modules():
        print(f"{n}: {type(m)}, isinstance(nn.BatchNorm2d): {isinstance(m, nn.BatchNorm2d)}, training={m.training}")

class FreezedBatchNorm2D(nn.BatchNorm2d):
    @staticmethod
    def replace_batchnorm2d_with_freezed(module:nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_features = child.num_features
                eps = child.eps
                momentum = child.momentum
                affine = child.affine
                track_running_stats = child.track_running_stats
                device = child.weight.device if child.weight is not None else None
                dtype = child.weight.dtype if child.weight is not None else None
                # 创建 FreezedBatchNorm2D 的实例
                freezed_bn = FreezedBatchNorm2D(num_features, eps, momentum, affine, track_running_stats, device, dtype)

                # 复制原始 BatchNorm2d 的状态 (包括 running_mean, running_var, weight, bias)
                freezed_bn.load_state_dict(child.state_dict())

                # 冻结新 BatchNorm 层的参数
                freezed_bn.freeze_parameters()

                # 使用新的 FreezedBatchNorm2D 实例替换原始的 BatchNorm2d
                setattr(module, name, freezed_bn)
            # 递归处理子模块
            elif isinstance(child, nn.Module):
                FreezedBatchNorm2D.replace_batchnorm2d_with_freezed(child)

        return module

    def __init__(self, num_features, eps = 0.00001, momentum = 0.1, affine = True, track_running_stats = True, device=None, dtype=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self.eval()
        self.freeze_parameters()
    def freeze_parameters(self):
        for n, p in self.named_parameters():
            p.requires_grad_(False)
    def train(self, mode = True):
        mode = False
        return super().train(mode)

class RaftDepthPrompt(RAFTStereo):
    '''
    take the depth outputed by raft_stereo as the prompt.  
    It only consider the cases of `left_ir + pattern` and 'left_ir + right_ir'.  
    Must be used in conjunction with `Identity` pattern_encoder  
    '''
    def __init__(self, 
                depth_type='norm_depth',
                iters=32,
                d_in=[128]*3,   # hidden_dims
                pretrained=None,
                context_norm="batch",
                n_downsample=2,
                n_gru_layers=3,
                corr_levels=4,
                corr_radius=4,
                corr_implementation='reg',
                shared_backbone=False,
                mixed_precision=True, 
                slow_fast_gru=False,
                tri_inputs=False,
                freeze_bn=True,
                out_scale_factor=1, # (h,w)
                **kwargs):
        args = EasyDict(
            hidden_dims=d_in, context_norm=context_norm, n_downsample=n_downsample, n_gru_layers=n_gru_layers,
            corr_levels=corr_levels, corr_radius=corr_radius, corr_implementation=corr_implementation,
            shared_backbone=shared_backbone, mixed_precision=mixed_precision, slow_fast_gru=slow_fast_gru
        )
        self.args = args
        self.iters = iters
        self.depth_type = depth_type
        self.tri_inputs=tri_inputs
        self.out_dim = 1

        super().__init__(args)

        self.keep_bn_freeze = freeze_bn

        self.pretrained = pretrained
        if self.pretrained is not None:
            iohelper = FileIOHelper()
            if iohelper.exists(self.pretrained):
                with iohelper.open(pretrained, 'rb') as f:
                    ckpt = torch.load(f, map_location='cpu')
                ckpt = unwrap_ddp_ckpt(ckpt['model'] if 'model' in ckpt else ckpt)
                self.load_state_dict(ckpt)
                print(f"Raft: load pretrained parameters from {pretrained}")
            else:
                import warnings
                warnings.warn(f"Raft pretrained path specified but not exists ({self.pretrained})! If you want to train, you must check this path; If you just want to inference, it doesn't matter")
        self.input_padder = None

        self.train_recent_called = [True, self.training]  # (recently called, mode params when recently called)
        self.out_scale_factor = (out_scale_factor, out_scale_factor) if isinstance(out_scale_factor, Number) else out_scale_factor

    def extra_setup_(self):
        print("freeze all batchnorms' parameters in RAFT!")
        if not self.keep_bn_freeze:
            return
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def train(self, mode = True):
        self.train_recent_called = [True, mode]
        return super().train(mode)
    
    def handle_train_recent_called(self):
        if self.train_recent_called[0]:
            if self.keep_bn_freeze and self.train_recent_called[1]:
                self.freeze_bn()
            self.train_recent_called[0] = False
        
    def quary_support_bino_imgs(self):
        # return self.tri_inputs
        return False
    
    def forward(self, features:torch.Tensor, extrinsics:torch.Tensor, 
            intrinsics:torch.Tensor, near:torch.Tensor, far:torch.Tensor, 
            **kwargs):
        '''
        features: BVCHW, extrinsics: BVCHW  
        extrinsics: BV44, intrinsics: BV33, near/far: BV  
        It only consider the case of `left_ir + pattern`.   
        '''
        self.handle_train_recent_called()

        if self.input_padder is None or (self.input_padder.ht, self.input_padder.wd) != features.shape[-2:]:
            self.input_padder = InputPadder(features.shape[-2:], divis_by=32)
        feat_b = features.shape[0]
        # image-pattern, features[:,0]: images, features[:,1]: pattern.
        left_img_feat = features[:,0]  # BCHW
        right_pat_feat = features[:,1]

        assert left_img_feat.shape[1] == 1 or left_img_feat.shape[1] == 3, "RaftDepthPrompt must be used in conjunction with `Identity` pattern_encoder"
        if left_img_feat.shape[1] == 1:
            left_img_feat = left_img_feat.expand(feat_b, 3, *left_img_feat.shape[2:])
            right_pat_feat= right_pat_feat.expand(feat_b,3, *right_pat_feat.shape[2:])
        left_img_feat, right_pat_feat = self.input_padder.pad(left_img_feat, right_pat_feat)

        _, disp = super().forward(left_img_feat * 255, right_pat_feat * 255, iters=self.iters, test_mode=True) # B1HW
        disp = self.input_padder.unpad(disp)

        if self.depth_type == 'raw_disp':
            disp = self.resize_output(disp)
            return disp.abs().unsqueeze(1), {'ref_depth': disp}
        
        left_img_intri = intrinsics[:,0]
        left_img_extri = extrinsics[:,0]
        right_pat_extri= extrinsics[:,1]
        left_img_near =  near[:,0]
        left_img_far  =  far[:,0]

        depth = d2d_transform(disp, left_img_intri, None, left_img_extri, right_pat_extri).abs()
        shape = left_img_near.shape[:1] + (1,) * (depth.ndim - 1)
        depth = torch.clamp(depth, left_img_near.view(shape), left_img_far.view(shape))
        extra_info = {'ref_depth': depth}
        depth = depth_conversion(
            depth, self.depth_type, False, 
            left_img_near, left_img_far, left_img_intri, left_img_extri,
            right_pat_extri, ref_depth=depth
        )
        depth = self.resize_output(depth)
        return depth.unsqueeze(1), extra_info
    

    def resize_output(self, out):
        if self.out_scale_factor == (1,1):
            return out
        return F.interpolate(out, scale_factor=self.out_scale_factor, mode='bilinear', align_corners=True)
    

class TriRaftDepthPrompt(nn.Module):
    def __init__(self, 
                 depth_type='norm_depth', 
                 iters=32, 
                 d_in=[128] * 3, 
                 pretrained=None, 
                 context_norm="batch", 
                 n_downsample=2, 
                 n_gru_layers=3, corr_levels=4, 
                 corr_radius=4, corr_implementation='reg', 
                 shared_backbone=False, mixed_precision=True, 
                 slow_fast_gru=False, tri_inputs=False, 
                 freeze_bn=True, 
                 out_scale_factor=1, 
                 left_right_pretrained=None, **kwargs):
        super().__init__()
        self.out_dim = 2
        self.depth_type = depth_type
        self.tri_inputs = True

        self.left_pattern_raft = RaftDepthPrompt(
            depth_type, iters, d_in, pretrained, context_norm, n_downsample,
            n_gru_layers, corr_levels, corr_radius, corr_implementation,
            shared_backbone, mixed_precision, slow_fast_gru, False,
            freeze_bn, out_scale_factor, **kwargs
        )
        self.left_right_raft = RaftDepthPrompt(
            depth_type, iters, d_in, left_right_pretrained, context_norm, n_downsample,
            n_gru_layers, corr_levels, corr_radius, corr_implementation,
            shared_backbone, mixed_precision, slow_fast_gru, False,
            freeze_bn, out_scale_factor, **kwargs
        )

    def extra_setup_(self):
        self.left_pattern_raft.extra_setup_()
        self.left_right_raft.extra_setup_()
    
    def quary_support_bino_imgs(self):
        # return self.tri_inputs
        return False
    
    def forward(self, features:torch.Tensor, extrinsics:torch.Tensor, 
            intrinsics:torch.Tensor, near:torch.Tensor, far:torch.Tensor, 
            **kwargs):
        b,v,c,h,w = features.shape
        b = b//2
        features = features.view(b, 2, *features.shape[1:]) # b//2,2,2,c,h,w
        extrinsics=extrinsics.view(b,2,*extrinsics.shape[1:])
        intrinsics=intrinsics.view(b,2,*intrinsics.shape[1:])
        near = near.view(b,2,*near.shape[1:])
        far = far.view(b,2,*far.shape[1:])
        # [:,0,0]: left image, [:,1,0]: right image, [:,:,1]: pattern.
        lp_features = torch.stack([features[:,0,0], features[:,0,1]], dim=1)
        lp_extrinsics=torch.stack([extrinsics[:,0,0],extrinsics[:,0,1]],dim=1)
        lp_intrinsics=torch.stack([intrinsics[:,0,0],intrinsics[:,0,1]],dim=1)
        lp_near = torch.stack([near[:,0,0], near[:,0,1]],dim=1)
        lp_far = torch.stack([far[:,0,0], far[:,0,1]], dim=1)

        lr_features = torch.stack([features[:,0,0], features[:,1,0]], dim=1)
        lr_extrinsics=torch.stack([extrinsics[:,0,0], extrinsics[:,1,0]], dim=1)
        lr_intrinsics=torch.stack([intrinsics[:,0,0], intrinsics[:,1,0]], dim=1)
        lr_near = torch.stack([near[:,0,0], near[:,1,0]], dim=1)
        lr_far  = torch.stack([far[:,0,0], far[:,1,0]], dim=1)

        depth_lp, extra_lp = self.left_pattern_raft.forward(
            lp_features, lp_extrinsics, lp_intrinsics, lp_near, lp_far, **kwargs
        )
        depth_lr, extra_lr = self.left_right_raft.forward(
            lr_features, lr_extrinsics, lr_intrinsics, lr_near, lr_far, **kwargs
        )

        depth = torch.concat((depth_lp, depth_lr), dim=2)
        return depth, extra_lr
    

class DualCorrRaftDepthPrompt(DualCorrRAFTStereo):
    def __init__(self,
                depth_type='norm_depth',
                iters=32,
                d_in=[128]*3,   # hidden_dims
                pretrained=None,
                context_norm="batch",
                n_downsample=2,
                n_gru_layers=3,
                corr_levels=4,
                corr_radius=4,
                corr_implementation='reg',
                corr_multiplier=2.,
                corr_middle_rate=0.5,
                shared_fnet=False,
                shared_backbone=False,
                mixed_precision=True, 
                slow_fast_gru=False,
                tri_inputs=True,
                freeze_bn=True,
                out_scale_factor=1, # (h,w)
                buggy_tri_corr=False,
                **kwargs):
        args = EasyDict(
            hidden_dims=d_in, context_norm=context_norm, n_downsample=n_downsample, n_gru_layers=n_gru_layers,
            corr_levels=corr_levels, corr_radius=corr_radius, corr_implementation=corr_implementation,
            corr_multiplier=corr_multiplier, corr_middle_rate=corr_middle_rate, shared_fnet=shared_fnet,
            shared_backbone=shared_backbone, mixed_precision=mixed_precision, slow_fast_gru=slow_fast_gru,
            buggy_tri_corr=buggy_tri_corr
        )
        super().__init__(args)
        self.depth_type = depth_type
        self.pretrained = pretrained
        self.iters = iters
        self.keep_bn_freeze = freeze_bn
        self.out_dim = 1
        assert tri_inputs

        if self.pretrained is not None:
            iohelper = FileIOHelper()
            with iohelper.open(self.pretrained, 'rb') as f:
                ckpt = torch.load(f, map_location='cpu')
            ckpt = unwrap_ddp_ckpt(ckpt['model'] if 'model' in ckpt else ckpt)
            self.load_state_dict(ckpt)
            print(f"DualCorrRaftStereo: load pretrianed parameters from {self.pretrained}")
        self.input_padder = None

        self.train_recent_called = [True, self.training]
        self.out_scale_factor = (out_scale_factor, out_scale_factor) if isinstance(out_scale_factor, Number) else out_scale_factor

    def extra_setup_(self):
        print("freeze all batchnorms' parameters in DualCorrRaftStereo!")
        if not self.keep_bn_freeze:
            return
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def train(self, mode = True):
        self.train_recent_called = [True, mode]
        return super().train(mode)
    
    def handle_train_recent_called(self):
        if self.train_recent_called[0]:
            if self.keep_bn_freeze and self.train_recent_called[1]:
                self.freeze_bn()
            self.train_recent_called[0] = False
        
    def quary_support_bino_imgs(self):
        # return self.tri_inputs
        return False
    
    def forward(self, features:torch.Tensor, extrinsics:torch.Tensor, 
            intrinsics:torch.Tensor, near:torch.Tensor, far:torch.Tensor, 
            **kwargs):
        '''
        features: BVCHW, extrinsics: BVCHW  
        extrinsics: BV44, intrinsics: BV33, near/far: BV  
        It only consider the case of `left_ir + pattern`.   
        '''
        self.handle_train_recent_called()

        if self.input_padder is None or (self.input_padder.ht, self.input_padder.wd) != features.shape[-2:]:
            self.input_padder = InputPadder(features.shape[-2:], divis_by=32)

        feat_b = features.shape[0]
        n_pairs = feat_b // 2
        features = features.view(n_pairs, 2, *features.shape[1:])
        extrinsics=extrinsics.view(n_pairs,2,*extrinsics.shape[1:])
        intrinsics=intrinsics.view(n_pairs,2,*intrinsics.shape[1:])
        near = near.view(n_pairs, 2, *near.shape[1:])
        far = far.view(n_pairs, 2, *far.shape[1:])

        left_img_feat = features[:,0,0]  # BCHW
        right_img_feat= features[:,1,0]
        middle_pat_feat = features[:,0,1]

        left_extri = extrinsics[:,0,0]
        right_extri= extrinsics[:,1,0]
        left_intri = intrinsics[:,0,0]
        left_near = near[:,0,0]
        left_far = far[:,0,0]

        # TODO: resolve corr_middle_rate better...

        assert left_img_feat.shape[1] == 1 or left_img_feat.shape[1] == 3, "RaftDepthPrompt must be used in conjunction with `Identity` pattern_encoder"
        if left_img_feat.shape[1] == 1:
            left_img_feat = left_img_feat.expand(n_pairs, 3, *left_img_feat.shape[2:])
            right_img_feat= right_img_feat.expand(n_pairs ,3, *right_img_feat.shape[2:])
            middle_pat_feat = middle_pat_feat.expand(n_pairs, 3, *middle_pat_feat.shape[2:])
        left_img_feat, right_img_feat, middle_pat_feat = self.input_padder.pad(left_img_feat, right_img_feat, middle_pat_feat)
        _, disp = super().forward(
            left_img_feat*255, right_img_feat*255, middle_pat_feat*255, iters=self.iters, test_mode=True,
            corr_middle_rate=kwargs.get("corr_middle_rate", None)
        )
        disp = self.input_padder.unpad(disp)

        if self.depth_type == 'raw_disp':
            disp = self.resize_output(disp)
            return disp.abs().unsqueeze(1), {'ref_depth': disp}

        depth = d2d_transform(disp, left_intri, None, left_extri, right_extri).abs()
        shape = left_near.shape[:1] + (1,) * (depth.ndim - 1)
        depth = torch.clamp(depth, left_near.view(shape), left_far.view(shape))
        extra_info = {'ref_depth': depth}
        depth = depth_conversion(
            depth, self.depth_type, False, 
            left_near, left_far, left_intri, left_extri,
            right_extri, ref_depth=depth
        )
        depth = self.resize_output(depth)
        return depth.unsqueeze(1), extra_info
    

    def resize_output(self, out):
        if self.out_scale_factor == (1,1):
            return out
        return F.interpolate(out, scale_factor=self.out_scale_factor, mode='bilinear', align_corners=True)