import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from numbers import Number

from .vit import DinoV2
from .prompting_dpt.pattern_prompting import PatternPrompt
from .prompting_dpt.dpt_sl import DPTHead
from .model_config import vit_configs

from utils.transforms import LCN, interp_disp_to_depth, resize_image, resize_intrinsic_matrix, depth_conversion, transpose_unsqueeze_image
from utils.dist import print_ddp


class SlPromptDA(nn.Module):
    # fixed.
    patch_size = 14  # patch size of the pretrained dinov2 model
    use_bn = False
    use_clstoken = False
    output_act = 'sigmoid'

    def __init__(self, 
        vit: Literal['vits', 'vitb', 'vitl', 'vitg'],
        # module configs.
        pat_epi_trans_cfg,
        pat_encoder_dim:int,
        prompt_feat_dim:int, 
        # extra config for DPTHead
        prompt_feat_proj_type:str = 'cnn', use_zero_module:bool=True, do_reassemble:bool=True,
        # extra config for PatternPrompt
        prompting_net_type:str = 'EpiTrans',
        pat_prompt_shared_transformer:bool=True, pat_enc_type:str='raft', pat_enc_extra_cfg:dict=None,
        lcn_wnd_size:int=9,
        max_depth:float = 10,
        rgb:bool = False,
        prompt_input_modal:str = 'left_right_pattern',
    ):
        super().__init__()

        self.num_context_view = 1
        self.vit_type = vit
        self.vit_config = vit_configs[vit]
        # vit.
        self.pretrained, vit_dim = DinoV2(vit)

        self.depth_head = DPTHead(nclass=1,
            in_channels=vit_dim,
            features_dim=self.vit_config['features'],
            out_channels=self.vit_config['out_channels'],
            use_bn=self.use_bn,
            use_clstoken=self.use_clstoken,
            output_act=self.output_act,
            prompt_feat_dim=prompt_feat_dim,
            prompt_feat_proj_type=prompt_feat_proj_type,
            use_zero_module=use_zero_module,
            do_reassemble=do_reassemble
        )

        self.prompting_net_type = prompting_net_type
        self.pattern_prompt = PatternPrompt(
            pat_encoder_dim, prompt_feat_dim, prompting_net_type, pat_prompt_shared_transformer,
            pat_epi_trans_cfg, pat_enc_type, pat_enc_extra_cfg, rgb, prompt_input_modal
        )
        self.lcn = LCN(lcn_wnd_size, False)
        self.support_bino_imgs = self.pattern_prompt.quary_support_bino_imgs()
        self.prompt_input_modal= self.pattern_prompt.quary_input_modal()

        self.register_buffer('_mean', torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.max_depth = max_depth

    def extra_setup_(self):
        for m in self.children():
            if hasattr(m, 'extra_setup_'):  # double check if batchnorm is frozen...
                m.extra_setup_()

    def pack_images(self,
            L_Image:torch.Tensor, R_Image:torch.Tensor, Pattern:torch.Tensor,
            L_extri:torch.Tensor, R_extri:torch.Tensor, P_extri:torch.Tensor,
            L_intri:torch.Tensor, R_intri:torch.Tensor, P_intri:torch.Tensor,
            near:float|torch.Tensor = None, far:float|torch.Tensor = None,
            **kwargs
        ):
        b,c,h,w = L_Image.shape
        P_Image = Pattern
        local_var = locals()
        # There is a special case where only one side of the depth can be inferred, but pattern_prompt requires the use of left, Right, pattern as input
        if self.support_bino_imgs or (not self.support_bino_imgs and self.prompt_input_modal == 'left_right_pattern'):
            if self.prompt_input_modal == 'left_right_pattern' or self.prompt_input_modal == 'left_pattern':
                inp1_prefix, inp2_prefix = ("L", "R"), ("P", "P")
            elif self.prompt_input_modal == 'left_right':
                inp1_prefix, inp2_prefix = ("L", "R"), ("R", "L")
            else:
                raise NotImplementedError
            imgs = torch.stack((local_var[f"{inp1_prefix[0]}_Image"], local_var[f"{inp1_prefix[1]}_Image"]), dim=1).reshape(2*b,-1,h,w)
            pats = torch.stack((local_var[f"{inp2_prefix[0]}_Image"], local_var[f"{inp2_prefix[1]}_Image"]), dim=1).reshape(2*b,-1,h,w)
            cam_extri = torch.stack((local_var[f"{inp1_prefix[0]}_extri"], local_var[f"{inp1_prefix[1]}_extri"]), dim=1).reshape(2*b,4,4)
            proj_extri = torch.stack((local_var[f"{inp2_prefix[0]}_extri"], local_var[f"{inp2_prefix[1]}_extri"]), dim=1).reshape(2*b,4,4)
            cam_intri = torch.stack((local_var[f"{inp1_prefix[0]}_intri"], local_var[f"{inp1_prefix[1]}_intri"]), dim=1).reshape(2*b,3,3)
            proj_intri = torch.stack((local_var[f"{inp2_prefix[0]}_intri"], local_var[f"{inp2_prefix[1]}_intri"]), dim=1).reshape(2*b,3,3)
            if isinstance(near, Number):
                near = torch.tensor(near, dtype=imgs.dtype, device=imgs.device).expand(2*b,)
            if isinstance(far, Number):
                far = torch.tensor(far, dtype=imgs.dtype, device=imgs.device).expand(2*b,)
            
            depth_type = kwargs.get("depth_type", "norm_depth")
            if 'ref_depth' in depth_type:
                L_Depth, R_Depth = kwargs['L_Depth'], kwargs['R_Depth']
                ref_depth = torch.stack((L_Depth, R_Depth), dim=1).reshape(2*b,-1,h,w)
        else:
            if self.prompt_input_modal == 'left_pattern' or self.prompt_input_modal == 'left_right_pattern':
                inp1_prefix, inp2_prefix = 'L', 'P'
            elif self.prompt_input_modal == 'left_right':
                inp1_prefix, inp2_prefix = 'L', 'R'
            else:
                raise NotImplementedError
            imgs = local_var[f"{inp1_prefix}_Image"]
            pats = local_var[f"{inp2_prefix}_Image"]
            cam_extri = local_var[f"{inp1_prefix}_extri"]
            proj_extri= local_var[f"{inp2_prefix}_extri"]
            cam_intri = local_var[f"{inp1_prefix}_intri"]
            proj_intri= local_var[f"{inp2_prefix}_intri"]
            if isinstance(near, Number):
                near = torch.tensor(near, dtype=imgs.dtype, device=imgs.device).expand(b,)
            if isinstance(far, Number):
                far = torch.tensor(far, dtype=imgs.dtype, device=imgs.device).expand(b,)
            depth_type = kwargs.get("depth_type", "norm_depth")
            if 'ref_depth' in depth_type:
                ref_depth = kwargs['L_Depth']

        ret = (imgs, pats, cam_extri, proj_extri, cam_intri, proj_intri, near, far)
        if 'ref_depth' in depth_type:
            ret += (ref_depth,)
        return ret
    
    def unpack_images(self, 
            imgs, pats, cam_extri, proj_extri, cam_intri, proj_intri, near, far, ref_depth,
            **kwargs):
        if not self.support_bino_imgs and self.prompt_input_modal == 'left_right_pattern':
            b, c, h, w = imgs.shape
            b = b // 2
            imgs = imgs.reshape(b,2,*imgs.shape[1:])[:,0]
            pats = pats.reshape(b,2,*pats.shape[1:])[:,0]
            cam_extri = cam_extri.reshape(b,2,*cam_extri.shape[1:])[:,0]
            proj_extri= proj_extri.reshape(b,2,*proj_extri.shape[1:])[:,0]
            near = near.reshape(b,2,*near.shape[1:])[:,0]
            far  = far.reshape(b,2,*far.shape[1:])[:,0]
            if ref_depth is not None:
                ref_depth = ref_depth.reshape(b,2,*ref_depth.shape[1:])[:,0]
            return imgs, pats, cam_extri, proj_extri, cam_intri, proj_intri, near, far, ref_depth
        else:
            return imgs, pats, cam_extri, proj_extri, cam_intri, proj_intri, near, far, ref_depth
        

    def infer(self,
            L_Image:torch.Tensor, R_Image:torch.Tensor, Pattern:torch.Tensor,
            L_extri:torch.Tensor, R_extri:torch.Tensor, P_extri:torch.Tensor,
            L_intri:torch.Tensor, R_intri:torch.Tensor, P_intri:torch.Tensor,
            near:float|torch.Tensor = None, far:float|torch.Tensor = None,
            **kwargs
        ):
        '''L_Image: b,c,h,w in range(0,1), c==1'''
        ori_size = L_Image.shape[-2:]
        ori_size_wh = ori_size[::-1]
        new_size_wh = [924, 518]
        new_size = new_size_wh[::-1]
        L_intri = resize_intrinsic_matrix(L_intri, new_size_wh, ori_size_wh)
        R_intri = resize_intrinsic_matrix(R_intri, new_size_wh, ori_size_wh)
        P_intri = resize_intrinsic_matrix(P_intri, new_size_wh, ori_size_wh)
        L_Image = F.interpolate(L_Image, new_size, mode='bilinear', align_corners=False)
        R_Image = F.interpolate(R_Image, new_size, mode='bilinear', align_corners=False)
        Pattern = F.interpolate(Pattern, new_size, mode='bilinear', align_corners=False)

        depth = self.__call__(L_Image, R_Image, Pattern, L_extri, R_extri, P_extri, L_intri, R_intri, P_intri, near, far, **kwargs)[0]

        depth = F.interpolate(depth, ori_size, mode='bilinear', align_corners=False)

        return depth

    def forward(self,
            L_Image:torch.Tensor, R_Image:torch.Tensor, Pattern:torch.Tensor,
            L_extri:torch.Tensor, R_extri:torch.Tensor, P_extri:torch.Tensor,
            L_intri:torch.Tensor, R_intri:torch.Tensor, P_intri:torch.Tensor,
            near:float|torch.Tensor = None, far:float|torch.Tensor = None,
            **kwargs
        ):
        '''
        inp_size: (h,w) or int, optional.  
        '''
        b,c,h,w = L_Image.shape
        imgs, pats, cam_extri, proj_extri, cam_intri, proj_intri, near, far, *ref_depth = self.pack_images(
            L_Image, R_Image, Pattern, L_extri, R_extri, P_extri, L_intri, 
            R_intri, P_intri, near, far, **kwargs
        )
        ref_depth = None if len(ref_depth) == 0 else ref_depth[0]
        depth_type = kwargs.get("depth_type", "norm_depth")
        
        # resize if specified in kwargs...
        if 'inp_size' in kwargs:
            ori_h, ori_w = imgs.shape[-2:]
            inp_size = kwargs['inp_size']  # should be (h,w)
            inp_size = (inp_size, inp_size) if isinstance(inp_size, Number) else inp_size
            imgs = F.interpolate(imgs, inp_size, mode='bilinear', align_corners=False)
            pats = F.interpolate(pats, inp_size, mode='bilinear', align_corners=False)
            if torch.all(cam_intri[...,0,0] > 50):  # be most likely not normalized
                cam_intri = resize_intrinsic_matrix(
                    cam_intri, (inp_size[1], inp_size[0]), (ori_w, ori_h)
                )
                proj_intri = resize_intrinsic_matrix(
                    proj_intri, (inp_size[1], inp_size[0]), (ori_w, ori_h)
                )

        depth = self.predict(
            imgs, pats, cam_extri, proj_extri, cam_intri, proj_intri, near, far, **kwargs
        )
        depth, extra_info = depth

        imgs, pats, cam_extri, proj_extri, cam_intri, proj_intri, near, far, ref_depth = self.unpack_images(
            imgs, pats, cam_extri, proj_extri, cam_intri, proj_intri, near, far, ref_depth
        )

        if isinstance(extra_info, dict) and 'ref_depth' in extra_info:
            ref_depth = extra_info['ref_depth']

        if kwargs.get("restore_depth", True):
            depth = depth_conversion(
                depth, depth_type, True, near, far,
                cam_intri, cam_extri, proj_extri, ref_depth if 'ref_depth' in depth_type else None, None, None
            )

        if 'inp_size' in kwargs:
            depth = F.interpolate(depth, (ori_h, ori_w), mode='bilinear', align_corners=True)

        if kwargs.get("get_extra", False):
            if self.support_bino_imgs:
                return *torch.unbind(depth.view(b, 2, *depth.shape[1:]), dim=1), extra_info
            else:
                return depth, None, extra_info
        else:
            if self.support_bino_imgs:
                return torch.unbind(depth.view(b, 2, *depth.shape[1:]), dim=1)
            else:
                return depth, None
            
    def align_to_rgb(self, prompt_depth, cam_intri, **kwargs):
        from utils.transforms import align_depthmap_to_other_view
        new_depth_map = align_depthmap_to_other_view(
            prompt_depth, cam_intri, kwargs['RGB_intri'], kwargs['R_d2rgb'], kwargs['T_d2rgb'],
            kwargs['RGB_Image'].shape[-3], kwargs['RGB_Image'].shape[-2]  , self.max_depth
        )  # kwargs['RGB_Image'] is still (B,H,W,3) here.
        return new_depth_map

    def predict(
            self, img:torch.Tensor, pats:torch.Tensor,
            cam_extri:torch.Tensor, proj_extri:torch.Tensor,
            cam_intri:torch.Tensor, proj_intri:torch.Tensor,
            near:float|torch.Tensor, far:float|torch.Tensor,
            **kwargs,
        ):
        '''
        img, pats: (b, c, h, w) (gray, c=1)  
        *_extri: (b, 4, 4);   
        *_intri: (b, 3, 3);  
        near, far: float or (b,)  
        img haven't been normalized  
        '''
        b, c, h, w = img.shape
        n_patch_h = h // self.patch_size
        n_patch_w = w // self.patch_size
        # pattern-lcn encoder for prompt
        img_lcn = self.lcn.forward(img) if kwargs.get("lcn", True) else img
        prompt_features, pat_prompt_extra = self.pattern_prompt.forward(
            pats, img_lcn, cam_extri, proj_extri, cam_intri, proj_intri,
            near, far, **kwargs
        )

        # skip refinement..
        if kwargs.get("skip_refine", False):
            return prompt_features, pat_prompt_extra

        # vit.
        if not self.support_bino_imgs and 'L_Depth' in kwargs and b != kwargs['L_Depth'].shape[0]:
            assert prompt_features.shape[0] == kwargs['L_Depth'].shape[0]
            b = b // 2
            img = img.reshape(b, 2, *img.shape[1:])[:,0]
            cam_intri = cam_intri.reshape(b, 2, *cam_intri.shape[1:])[:,0]

        if kwargs.get("align_rgb", False) and "RGB_Image" in kwargs:
            assert self.prompting_net_type == 'RaftDepth', "Raft only support initial depth as prompt."
            assert self.pattern_prompt.transformer.depth_type == 'norm_depth', "for now, we only consider converting prompt to normed depth."
            prompt_depth = prompt_features * self.max_depth
            prompt_depth = self.align_to_rgb(prompt_depth, cam_intri, **kwargs)
            prompt_features = prompt_depth / self.max_depth
            vit_img = transpose_unsqueeze_image(kwargs['RGB_Image'], channel_last=True)
            
            h_align, w_align = prompt_features.shape[-2:]
            h_ori, w_ori = img.shape[-2:]
            if (h_align, w_align) != (h_ori, w_ori):
                vit_img = F.interpolate(vit_img, (h_ori, w_ori), align_corners=False, mode='bilinear')
                prompt_features = F.interpolate(prompt_features, (h_ori, w_ori), align_corners=False, mode='bilinear')
        else:
            vit_img = img

        if not self.training and kwargs.get("scale_prompt", False):  # prompt depth can be scale to a appropriate range if needed.
            scale_min, scale_max = 0.05, 0.3
            target_range_min = torch.tensor([scale_min]*b, dtype=torch.float32, device=prompt_features.device).reshape(b,1,1,1)
            target_range_max = torch.tensor([scale_max]*b, dtype=torch.float32, device=prompt_features.device).reshape(b,1,1,1)
            tmp_prompt = prompt_features
            invalid_mask = (tmp_prompt <= 0) | (tmp_prompt > 1)
            ori_min = torch.tensor([tmp_prompt[i][tmp_prompt[i] > 0].min().item() for i in range(b)], dtype=torch.float32, device=prompt_features.device).reshape(1,1,1)
            ori_max = torch.tensor([tmp_prompt[i][tmp_prompt[i] > 0].max().item() for i in range(b)], dtype=torch.float32, device=prompt_features.device).reshape(1,1,1)
            tmp_prompt = (tmp_prompt - ori_min) * (target_range_max - target_range_min) / (ori_max - ori_min) + target_range_min
            tmp_prompt = torch.clip(tmp_prompt, scale_min, scale_max)
            prompt_features = tmp_prompt
        
        norm_img = (vit_img.expand(b, 3, h, w) - self._mean) / self._std
        vit_features = self.pretrained.get_intermediate_layers(
            norm_img, self.vit_config['layer_idxs'],
            return_class_token=True)
        # dpt
        depth = self.depth_head.forward(
            vit_features, n_patch_h, n_patch_w, prompt_feat=prompt_features
        )

        if not self.training and kwargs.get("scale_prompt", False):
            depth = (depth - target_range_min) * (ori_max - ori_min) / (target_range_max - target_range_min) + ori_min

        return depth, pat_prompt_extra
    

def create_model(pretrained_path:str=None, ckpt_path:str=None,
                 override_vit_pretrained:str=None, return_ckpt:bool=False, **kwargs):
    '''
    pretrained_path: pretrained dav2.  
    ckpt_path: ours. ckpt_path has higher priority.  
    '''
    from tools.fix_ckpt import unwrap_ddp_ckpt
    from utils.common import FileIOHelper
    iohelper = FileIOHelper()
    if 'special_model' in kwargs:
        model_module = kwargs.pop("special_model")
        model = model_module(**kwargs)
    else:
        model =  SlPromptDA(**kwargs)
            
    if ckpt_path is not None:
        with iohelper.open(ckpt_path, 'rb') as f:
            pretrained = torch.load(f, map_location='cpu')
            if 'model' in pretrained:
                pretrained['model'] = unwrap_ddp_ckpt(pretrained['model'])
            else:
                pretrained = unwrap_ddp_ckpt(pretrained)
        strict = True
        print_ddp(f"load ckpt from {ckpt_path}")
    elif pretrained_path is not None:
        with iohelper.open(pretrained_path, 'rb') as f:
            pretrained = torch.load(f, map_location='cpu')
        pretrained = pretrained['model'] if 'model' in pretrained else pretrained  # only need model parameters.
        strict = False
        print_ddp(f"load pretrained dav2 from {pretrained_path}")
    else:
        pretrained = None
    if pretrained is not None:
        model.load_state_dict(pretrained['model'] if 'model' in pretrained else pretrained, strict=strict)
    if ckpt_path is None and override_vit_pretrained is not None:
        with iohelper.open(override_vit_pretrained, 'rb') as f:
            override = torch.load(f, map_location='cpu')
        model.load_state_dict(override, strict=False)
        print_ddp(f"override the vit pretrained parameters using {override_vit_pretrained}")
        for k, v in override:
            if 'model' in pretrained:
                if k in pretrained['model']:
                    pretrained['model'][k] = v
            else:
                if k in pretrained:
                    pretrained[k] = v
    if not return_ckpt:
        return model
    else:
        return model, pretrained