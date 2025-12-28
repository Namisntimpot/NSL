import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from easydict import EasyDict
import logging
from numbers import Number

from . import register_inference_model
from .data_info import MEAN_STD_INFO, RESO_INFO
from utils.metrics import compute_metrics, compute_material_classfied_metrics, count_flops, count_parameters
from utils.transforms import (d2d_transform, transpose_image, border_pad_image, interp_disp_to_depth, denormalize_image,
                              normalize_image, resize_image, resize_intrinsic_matrix, normalize_intrinsic_matrix,
                              normalize_depth_use_min_max, depth_linear_scale, rectify_images_simplified)
from utils.common import to_device, load_config

class BaseInferenceModel:
    def __init__(self, dsname:str, inp_size:tuple, pretrained_model:str, cfgs_path:set, ddp:bool, match_lr:bool, metrics_names:list):
        '''
        match_lr: for models that predict disparity, match_lr specifies whether the matching 
        is done between left/right images or between images and patterns.
        metrics_names can be any of the following:  \\
        d1, bad_1, bad_2, bad_3, thresh_1, thresh_2, thresh_3, epe, rmse, mae, absrel, sqrel  \\
        这里的inp_size是全程需要的inp_size, 预期的输出也是inp_size, 算metric也是这个inp_size.  
        '''
        self.dsname = dsname
        self.mean = MEAN_STD_INFO[self.dsname]['mean']
        self.std = MEAN_STD_INFO[self.dsname]['std']
        self.origin_reso = RESO_INFO[self.dsname]
        self.inp_size = inp_size
        self.ddp = ddp
        self.pretrained_model = pretrained_model
        self.metrics_names = metrics_names
        self.model = None
        self.match_lr = match_lr
        self.rank = 0
        self._init_logger()

    def _init_logger(self):
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")

    def _do_setup_ddp(self, rank, world_size, **ddp_extra_params):
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        self.rank = rank
        self.world_size = world_size
        device = torch.device(f"cuda:{rank}")
        if self.model is None:
            return
        self.model = self.model.to(device)
        if self.model.device != device:
            self.model.device = device  # fake device attribute
        self.model = DDP(
            self.model, device_ids=[rank],
            **ddp_extra_params
        )

    def update_inp_size(self, new_ori_reso, new_inp_reso):
        '''newsize: (w, h)'''
        self.origin_reso = (int(new_ori_reso[0]), int(new_ori_reso[1]))
        self.inp_size = (int(new_inp_reso[0]), int(new_inp_reso[1]))

    def setup_ddp(self, rank, world_size):
        self._do_setup_ddp(rank, world_size)

    def preprocess(self, data:dict):
        '''
        in base class, we: resize, transpose, move to appropriate device
        '''
        # resize or crop?
        ret = {}
        for k, v in data.items():
            if 'Image' in k:
                # v = center_crop_image(v, self.inp_size, channel_last=True)
                v = transpose_image(v, channel_last=True)
                v = resize_image(v, self.inp_size, channel_last=False)
            elif 'Depth' in k:
                # v = v.unsqueeze(-1)
                # v = center_crop_image(v, self.inp_size, channel_last=True).squeeze(-1)
                v = v.unsqueeze(1)
                v = resize_image(v, self.inp_size, channel_last=False, mode='nearest').squeeze(1)
            elif 'Mask' in k:
                v = v.unsqueeze(1)
                v = resize_image(v.to(torch.float32), self.inp_size, channel_last=False, mode='nearest').squeeze(1) > 0
            # if resize, the intrinsic matrix should also be scaled...
            elif 'intri' in k and not 'P_' in k:  # avoid resizing P_intri
                v = resize_intrinsic_matrix(v, self.inp_size, self.origin_reso)
            ret[k] = v
        device = self.model.device # if not self.ddp else self.model.module.device
        return to_device(ret, device)

    def forward(self, data:dict):
        raise NotImplementedError

    def postprocess(self, data:dict, pred:dict):
        return pred

    def evaluate(self, data:dict, pred:dict, *metrics_names, **kwargs):
        '''
        **warning: for now, it only supports 'images' mode**  
        valid mask hasn't been considered
        '''
        metrics = {}
        if kwargs.get("per_mat_metrics", False):
            per_mat_metrics = {}
        for k in pred:
            pred_depth = pred[k]
            if pred_depth is None:
                metrics[k] = None
                continue
            lr = 'L' if 'L_' in k else 'R'
            gt_depth = data[f"{lr}_Depth"]
            metrics[k] = compute_metrics(pred_depth, gt_depth, data.get(f"{lr}_Mask", None), True, *metrics_names)
            if kwargs.get("per_mat_metrics", False):
                per_mat_metrics[k] = compute_material_classfied_metrics(
                    pred_depth, gt_depth, data[f"{lr}_MaterialType"], data.get(f"{lr}_Mask", None), True, 1, *metrics_names
                )
        if kwargs.get("per_mat_metrics", False):
            return metrics, per_mat_metrics
        return metrics
        

    def inference(self, data:dict, **kwargs):
        '''
        return pred depth, per-image metrics.
        '''
        # if self.ddp:
        #     self.model.module.eval()
        # else:
        #     self.model.eval()
        if isinstance(self.model, (nn.Module, DDP)):
            self.model.eval()

        with torch.no_grad():
            data = self.preprocess(data)
            pred = self.forward(data)
            pred = self.postprocess(data, pred)
            metrics = self.evaluate(data, pred, *self.metrics_names, **kwargs)
        return pred, metrics, data


class DisparityInferenceModel(BaseInferenceModel):
    def __init__(self, dsname, inp_size, pretrained_model, cfgs_path, ddp, match_lr, metrics_names):
        super().__init__(dsname, inp_size, pretrained_model, cfgs_path, ddp, match_lr, metrics_names)
        self.eval_mode = 'depth'  # depth or disp

    def postprocess(self, data:dict, pred:dict):
        '''
        disparity -> depth...
        '''
        if self.eval_mode == 'disp':
            return pred

        self_intri = data['L_intri']
        other_intri = data['R_intri']
        self_extri = data['L_extri']
        other_extri = data['R_extri']

        ret = {}
        for k, v in pred.items():
            if v is None or not 'Image' in k:
                ret[k] = v
                continue
            if 'ref_disp' in data:
                v = v.abs() + data['ref_disp']
            ret[k] = d2d_transform(
                v, self_intri, other_intri, self_extri, other_extri)
        return super().postprocess(data, ret)
    
    def preprocess(self, data):
        data = super().preprocess(data)
        # convert gt_depth to gt_disp
        if self.eval_mode == 'disp':
            for k in data:
                if not 'Depth' in k:
                    continue
                self_lr = k[0]
                other_lr = 'R' if self_lr == 'L' else 'L'
                self_intri = data[f'{self_lr}_intri']
                other_intri = data[f'{other_lr}_intri']
                self_extri = data[f'{self_lr}_extri']
                other_extri = data[f'{other_lr}_extri']
                dep = data[k]
                disp = d2d_transform(
                    dep, self_intri, other_intri, self_extri, other_extri
                )
                data[k] = disp

        # if matching l_image and pattern.
        if not self.match_lr:  # replace R_Image with Pattern
            pat = data['Pattern']
            pat = transpose_image(pat, channel_last=True)
            l_intri = data['L_intri']
            pat = resize_image(pat, self.inp_size, channel_last=False)
            p_intri = data['P_intri']
            p_intri = resize_intrinsic_matrix(p_intri, self.inp_size, self.origin_reso)
            pat = rectify_images_simplified(pat, l_intri, p_intri, False)
            p_intri = l_intri
            data['R_Image'] = pat
            data['P_intri'] = data['R_intri'] = p_intri
            data['R_extri'] = data['P_extri']

        return data


@register_inference_model("SlPromptDA")
class SlPromptDAV2InferenceModel(BaseInferenceModel):
    def __init__(self, 
        dsname, inp_size, pretrained_model, cfgs_path, ddp, match_lr, metrics_names
    ):
        super().__init__(dsname, inp_size, pretrained_model, cfgs_path, ddp, match_lr, metrics_names)
        self.cfgs = load_config(cfgs_path)
        self.model_cfgs = self.cfgs.Model
        self.train_cfgs = self.cfgs.Train
        self.val_cfgs = self.cfgs.Val
        if 'override_model' in self.val_cfgs:
            self.model_cfgs = self.val_cfgs.pop('override_model')
        self.near = self.val_cfgs.get("near", self.val_cfgs.get("min_depth", 0.1))
        self.far = self.val_cfgs.get("far", self.val_cfgs.get("max_depth", 10.))
        self.depth_type = self.val_cfgs.get("depth_type", "norm_depth")
        # create model
        from .neuralsl.sl_prompt_da import create_model
        try:
            from ..utils.aug import parse_augmentation
        except:
            from utils.aug import parse_augmentation
        self.model_cfgs.ckpt_path = pretrained_model
        self.model = create_model(return_ckpt=False, **self.model_cfgs)
        if not self.ddp:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.model = self.model.to(device)
            self.model.device = device
        else:
            self.model.device = torch.device("cpu")
        self.preprocess_module = parse_augmentation(*self.val_cfgs.preprocess)

    def preprocess(self, data):
        data['near'] = self.near
        data['far'] = self.far
        data = self.preprocess_module(to_device(data, self.model.device))
        return data

    def forward(self, data):
        override_inp_size = self.val_cfgs.get("inp_size", None)  # (w,h)
        if override_inp_size is not None:
            override_inp_size = (
                int(np.ceil(override_inp_size[0] / 14) * 14),
                int(np.ceil(override_inp_size[1] / 14) * 14),
            )
        ori_size = data['L_Image'].shape[-2:]
        ori_size = (ori_size[1], ori_size[0])
        if override_inp_size is not None and override_inp_size != ori_size:
            bak_l_image = data['L_Image']
            bak_r_image = data['R_Image']
            bak_patt = data['Pattern']
            bak_l_intri = data['L_intri']
            bak_r_intri = data['R_intri']
            bak_p_intri = data['P_intri']
            data['L_intri'] = resize_intrinsic_matrix(bak_l_intri, override_inp_size, ori_size)
            data['R_intri'] = resize_intrinsic_matrix(bak_r_intri, override_inp_size, ori_size)
            data['P_intri'] = resize_intrinsic_matrix(bak_p_intri, override_inp_size, ori_size)
            data['L_Image'] = F.interpolate(bak_l_image, (override_inp_size[1], override_inp_size[0]), mode='bilinear', align_corners=False)
            data['R_Image'] = F.interpolate(bak_r_image, (override_inp_size[1], override_inp_size[0]), mode='bilinear', align_corners=False)
            data['Pattern'] = F.interpolate(bak_patt, (override_inp_size[1], override_inp_size[0]), mode='bilinear', align_corners=False)

        kwargs = self.val_cfgs.get("fwd_kwargs", {})
        kwargs['get_extra'] = False
        l_depth, r_depth = self.model(**{**data, **kwargs})
        # l_depth, r_depth = self.model(depth_type = self.depth_type, **data)

        if override_inp_size is not None and override_inp_size != ori_size:
            data['L_Image'] = bak_l_image
            data['R_Image'] = bak_r_image
            data['Pattern'] = bak_patt
            data['L_intri'] = bak_l_intri
            data['R_intri'] = bak_r_intri
            data['P_intri'] = bak_p_intri
            l_depth = F.interpolate(l_depth, (ori_size[1], ori_size[0]), mode='bilinear', align_corners=True)
            if r_depth is not None:
                r_depth = F.interpolate(r_depth, (ori_size[1], ori_size[0]), mode='bilinear', align_corners=True)

        return {
            "L_Image": l_depth, "R_Image": r_depth
        }
    
    def postprocess(self, data, pred):
        pred['L_Image'] = pred['L_Image'][..., :self.inp_size[1], :self.inp_size[0]]
        pred['R_Image'] = pred['R_Image'][..., :self.inp_size[1], :self.inp_size[0]] if pred['R_Image'] is not None else None
        data['L_Image'] = data['L_Image'][..., :self.inp_size[1], :self.inp_size[0]]
        data['R_Image'] = data['R_Image'][..., :self.inp_size[1], :self.inp_size[0]]
        data['Pattern'] = data['Pattern'][..., :self.inp_size[1], :self.inp_size[0]]
        data['L_Depth'] = data['L_Depth'][..., :self.inp_size[1], :self.inp_size[0]]
        data['R_Depth'] = data['R_Depth'][..., :self.inp_size[1], :self.inp_size[0]]
        return pred
    

@register_inference_model("RAFTStereo")
class RAFTStereoInferenceModel(DisparityInferenceModel):
    def __init__(self, 
        dsname, inp_size, pretrained_model, cfgs_path, ddp, match_lr, metrics_names
    ):
        super().__init__(dsname, inp_size, pretrained_model, cfgs_path, ddp, match_lr, metrics_names)
        self.cfgs = load_config(cfgs_path)
        self.valid_iters = self.cfgs.valid_iters
        self.eval_mode = self.cfgs.eval_mode
        raft_class_name = self.cfgs.get("raft_class", "RAFTStereo")
        raft_module_name = "zoo.RAFT_Stereo.core.raft_stereo"

        import sys
        import importlib
        sys.path.append("zoo/RAFT_Stereo/")
        from core.utils.utils import InputPadder
        raft_module = importlib.import_module(raft_module_name)
        raft_class  = getattr(raft_module, raft_class_name)

        self.model = raft_class(self.cfgs) # note that raft_stereo expects images within range (0,255).
        pretrained_parameters = torch.load(pretrained_model, map_location='cpu')
        if 'model' in pretrained_parameters:
            pretrained_parameters = pretrained_parameters['model']
        pretrained_parameters = {
            ".".join(k.split('.')[1:]) if k.startswith('module.') else k: v for k, v in pretrained_parameters.items()
        }
        self.model.load_state_dict(pretrained_parameters)
        if not self.ddp:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.model = self.model.to(device)
            self.model.device = device
        else:
            self.model.device = torch.device("cpu")
        self.model.freeze_bn()

        self.input_padder = InputPadder((self.inp_size[1], self.inp_size[0]), divis_by=32)
        self.pad_h = int(np.ceil(self.inp_size[1] / 32) * 32)   # divis by 32
        self.pad_w = int(np.ceil(self.inp_size[0] / 32) * 32)

    def preprocess(self, data):
        data = super().preprocess(data)  # resize, transpose, device
        for k in data:
            if 'Image' in k:
                v = data[k]
                if hasattr(self, 'input_padder'):
                    v = self.input_padder.pad(v)[0] * 255
                else:
                    v = border_pad_image(v, (self.pad_w, self.pad_h), False) * 255.  # go back to (0,255)
                data[k] = v   
        return data

    def forward(self, data):
        if isinstance(self.model, DDP):
            self.model.module.freeze_bn()
        else:
            self.model.freeze_bn()
        _, l_depth = self.model(data["L_Image"], data["R_Image"], iters=self.valid_iters, test_mode=True)
        # l_depth, r_depth = self.model(depth_type = self.depth_type, **data)
        return {
            "L_Image": l_depth.abs_(), "R_Image": None,
        }
    
    def postprocess(self, data, pred):
        if hasattr(self, 'input_padder'):
            data['L_Image'] = self.input_padder.unpad(data['L_Image']) / 255.
            data['R_Image'] = self.input_padder.unpad(data['R_Image']) / 255.
            pred['L_Image'] = self.input_padder.unpad(pred['L_Image'])
        else:
            data['L_Image'] = data['L_Image'][..., :self.inp_size[1], :self.inp_size[0]] / 255.
            data['R_Image'] = data['R_Image'][..., :self.inp_size[1], :self.inp_size[0]] / 255.
            pred['L_Image'] = pred['L_Image'][..., :self.inp_size[1], :self.inp_size[0]]
        return super().postprocess(data, pred)
    

@register_inference_model("DualCorrRaftStereo")
class DualCorrRaftStereo(DisparityInferenceModel):
    def __init__(self, dsname, inp_size, pretrained_model, cfgs_path, ddp, match_lr, metrics_names):
        super().__init__(dsname, inp_size, pretrained_model, cfgs_path, ddp, match_lr, metrics_names)
        assert self.match_lr
        self.cfgs = load_config(cfgs_path)
        self.valid_iters = self.cfgs.valid_iters
        self.eval_mode = self.cfgs.eval_mode
        raft_class_name = self.cfgs.get("raft_class", "TriRAFTStereo")
        raft_module_name= "zoo.RAFT_Stereo.core.tri_raft_stereo"

        import sys
        import importlib
        sys.path.append("zoo/RAFT_Stereo/")
        raft_module = importlib.import_module(raft_module_name)
        raft_class = getattr(raft_module, raft_class_name)

        self.model = raft_class(self.cfgs)
        pretrained_parameters = torch.load(pretrained_model, map_location='cpu')
        pretrained_parameters = pretrained_parameters['model'] if 'model' in pretrained_parameters else pretrained_parameters
        self.model.load_state_dict(pretrained_parameters)
        if not self.ddp:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.model = self.model.to(device)
            self.model.device = device
        else:
            self.model.device = torch.device('cpu')

        self.pad_h = int(np.ceil(self.inp_size[1] / 32) * 32)   # divis by 32
        self.pad_w = int(np.ceil(self.inp_size[0] / 32) * 32)

    def preprocess(self, data):
        data = super().preprocess(data)  # resize, transpose, device
        for k in data:
            if 'Image' in k:
                v = data[k]
                v = border_pad_image(v, (self.pad_w, self.pad_h), False) * 255.  # go back to (0,255)
                data[k] = v
        
        pat = data['Pattern']
        pat = transpose_image(pat, channel_last=True)
        l_intri = data['L_intri']
        pat = resize_image(pat, self.inp_size, channel_last=False)
        p_intri = data['P_intri']
        p_intri = resize_intrinsic_matrix(p_intri, self.inp_size, self.origin_reso)
        pat = rectify_images_simplified(pat, l_intri, p_intri, False)
        pat = border_pad_image(pat, (self.pad_w, self.pad_h), False) * 255
        data['Pattern'] = pat
        data['P_intri'] = l_intri

        return data

    def forward(self, data):
        if isinstance(self.model, DDP):
            self.model.module.freeze_bn()
        else:
            self.model.freeze_bn()
        _, l_depth = self.model(data["L_Image"], data["R_Image"], data["Pattern"], iters=self.valid_iters, 
                                test_mode=True, corr_middle_rate=data.get("corr_middle_rate", None))
        # l_depth, r_depth = self.model(depth_type = self.depth_type, **data)
        return {
            "L_Image": l_depth.abs_(), "R_Image": None,
        }
    
    def postprocess(self, data, pred):
        data['L_Image'] = data['L_Image'][..., :self.inp_size[1], :self.inp_size[0]] / 255.
        data['R_Image'] = data['R_Image'][..., :self.inp_size[1], :self.inp_size[0]] / 255.
        data['Pattern'] = data['Pattern'][..., :self.inp_size[1], :self.inp_size[0]] / 255.
        pred['L_Image'] = pred['L_Image'][..., :self.inp_size[1], :self.inp_size[0]]
        return super().postprocess(data, pred)