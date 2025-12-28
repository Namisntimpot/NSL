import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import _make_prompting_network

class PatternPrompt(nn.Module):
    def __init__(self, encoder_feat_dim:int, prompt_feat_dim:int, prompting_type:str,
            shared_transformer:bool, transformer_cfg:dict, 
            encoder_type:str, encoder_cfg:dict, rgb:bool=False, modal:str='left_right_pattern'):
        '''
        encoder_type: none, raft, backbone, shallowcnn  
        modal: left_right_pattern (default), left_pattern, left_right
        '''
        super().__init__()
        # create encoder.
        self.encoder_type = encoder_type
        self.encoder_feat_dim = encoder_feat_dim
        self.prompt_feat_dim = prompt_feat_dim
        self.num_context_views = transformer_cfg['num_context_views'] = 2
        self.rgb = rgb
        assert modal in ['left_right_pattern', 'left_pattern', 'left_right']
        self.modal = modal
        img_channels = 3 if rgb else 1

        kwargs = {'rgb': rgb}
        if encoder_type == 'none':
            from .pattern_encoders import IdentityEncoder
            self.encoder = IdentityEncoder()
            self.encoder_out_dim = img_channels
            self.encoder_out_single = True
        elif encoder_type == 'raft':
            from .pattern_encoders import RaftEncoder
            self.encoder = RaftEncoder(encoder_feat_dim, **{**encoder_cfg, **kwargs})
            self.encoder_out_dim = encoder_feat_dim
            self.encoder_out_single = True
        elif encoder_type == 'raw_raft':
            from .pattern_encoders import RawRaftEncoder
            self.encoder = RawRaftEncoder(encoder_feat_dim, **{**encoder_cfg, **kwargs})
            self.encoder_out_dim = encoder_feat_dim
            self.encoder_out_single = True
        elif encoder_type == 'backbone':
            from .pattern_encoders import BackboneEncoder
            self.encoder = BackboneEncoder(enc_feat_dim=encoder_feat_dim, **{**encoder_cfg, **kwargs})
            self.encoder_out_dim = encoder_feat_dim
            self.encoder_out_single = True
        elif encoder_type == 'backbone_multi_mid_layers':
            from .pattern_encoders import BackboneMultiMidLayersEncoder
            self.encoder = BackboneMultiMidLayersEncoder(**{**encoder_cfg, **kwargs})
            self.encoder_out_dim = self.encoder.get_intermediate_feat_dims()
            self.encoder_out_single = False
        elif encoder_type == 'shallowcnn':
            from .pattern_encoders import ShallowCnnEncoder
            self.encoder = ShallowCnnEncoder(encoder_feat_dim, **{**encoder_cfg, **kwargs})
            self.encoder_out_dim = encoder_feat_dim
            self.encoder_out_single = True
        elif encoder_type == 'slide_window':
            from .pattern_encoders import SlideWindowEncoder
            self.encoder = SlideWindowEncoder(**{**encoder_cfg, **kwargs})
            assert encoder_feat_dim == self.encoder.ksize**2 * img_channels
            self.encoder_out_dim = encoder_feat_dim
            self.encoder_out_single = True
        else:
            raise ValueError(f"unknown pattern encoder {encoder_type}")

        self.shared_transformer = shared_transformer
        if shared_transformer:
            self.transformer, prompting_net_in_dim, prompting_net_out_dim = _make_prompting_network(
                prompting_type, transformer_cfg, encoder_feat_dim, prompt_feat_dim
            )
            self.support_bino_imgs = self.transformer.quary_support_bino_imgs()
            assert prompting_net_out_dim == prompt_feat_dim

            if self.encoder_out_single:
                if self.encoder_out_dim != prompting_net_in_dim:
                    self.pre_transformer_projector = nn.Conv2d(
                        self.encoder_out_dim, prompting_net_in_dim, 1
                    )
                else:
                    self.pre_transformer_projector = nn.Identity()
            else:
                self.pre_transformer_projector = nn.ModuleList(
                    [nn.Conv2d(self.encoder_out_dim[f'out{i}'], prompting_net_in_dim, 1)
                     for i in range(4)]
                )
        else:  # 4 prompting networks.
            enc_out_dims = self.encoder_out_dim if not self.encoder_out_single else \
                           {f'out{i}': self.encoder_out_dim for i in range(4)}
            nets, prompting_net_in_dims, prompting_net_out_dims = [],[],[]
            for i in range(4):
                net, prompting_net_in_dim, prompting_net_out_dim = _make_prompting_network(
                    prompting_type, transformer_cfg, encoder_feat_dim, prompt_feat_dim
                )
                self.support_bino_imgs = net.quary_support_bino_imgs()
                assert prompting_net_out_dim == prompt_feat_dim
                nets.append(net)
                prompting_net_in_dims.append(prompting_net_in_dim)
                prompting_net_out_dims.append(prompting_net_out_dim)
            self.transformer = nn.ModuleList(nets)
            self.pre_transformer_projector = nn.ModuleList(
                [nn.Conv2d(enc_out_dims[f'out{i}'], prompting_net_in_dims[i], 1) \
                 if enc_out_dims[f'out{i}'] != prompting_net_in_dims[i] else nn.Identity() for i in range(4)]
            )
            
    def quary_support_bino_imgs(self):
        return self.support_bino_imgs
    
    def quary_input_modal(self):
        '''
        left_right_pattern (default), left_pattern, left_right
        '''
        return self.modal
    
    def extra_setup_(self):
        for m in self.children():
            if hasattr(m, 'extra_setup_'):
                m.extra_setup_()

    def forward(self, pattern:torch.Tensor, image:torch.Tensor, 
                cam_extri:torch.Tensor, proj_extri:torch.Tensor, 
                cam_intri:torch.Tensor, proj_intri:torch.Tensor,
                near:torch.Tensor, far:torch.Tensor, **kwargs):
        '''
        pattern, img: (b, c, h, w)  
        {cam/proj}_extri: (b, 4, 4)  
        {cam/proj}intri: (b, 4, 4)  
        near, far: (b, )  
        '''
        b = pattern.shape[0]
        if self.rgb:
            pattern = pattern.expand(b, 3, *pattern.shape[2:])
            image = image.expand(b, 3, *image.shape[2:])
        
        pat_enc, img_enc = self.encoder.forward(pattern, image)
        extri_2view = torch.stack((cam_extri, proj_extri), dim=1)
        intri_2view = torch.stack((cam_intri, proj_intri), dim=1)
        near_2view = torch.stack((near, near), dim=1)
        far_2view = torch.stack((far, far), dim=1)
        if self.encoder_out_single:
            feat = torch.stack((img_enc, pat_enc), dim=1)
        else:
            feat = [torch.stack((img_enc[i], pat_enc[i]), dim=1) for i in range(4)]

        if self.shared_transformer:
            if self.encoder_out_single:
                feat = self.pre_transformer_projector(feat.reshape(b*2, *feat.shape[2:]))
                feat = feat.reshape(b, 2, *feat.shape[1:])
                fused, extra = self.transformer.forward(
                    feat, extri_2view, intri_2view, near_2view, far_2view, **kwargs
                )  # b,v,c,h,w

                # DEBUG.
                # import matplotlib.pyplot as plt
                # from utils.dist import get_rank
                # import numpy as np
                # import cv2
                # def draw_imshow(toshow, name, vmin=None, vmax=None):
                #     plt.imshow(toshow, cmap='jet', vmin=toshow.min() if vmin is None else vmin, 
                #                vmax=toshow.max() if vmax is None else vmax)
                #     plt.colorbar()
                #     plt.savefig(name)
                #     plt.close()
                # print("feat", feat.shape, feat.min().item(), feat.max().item())
                # print("fused", fused.shape, fused.min().item(), fused.max().item())
                # fused_to_show = fused.view(-1, *fused.shape[2:])
                # for i in range(fused_to_show.shape[0]):
                #     toshow = fused_to_show[i].squeeze().detach().cpu().numpy()
                #     print(f"{get_rank()}_batch{i}, prompt_depth: {toshow.min() * 10}, {toshow.max() * 10}, "
                #            f"gt_depth: {kwargs['L_Depth'][i].squeeze().detach().cpu().numpy().min()}, {kwargs['L_Depth'][i].squeeze().detach().cpu().numpy().max()}")
                #     if toshow.shape[0] == 2:
                #         for j in range(2):
                #             draw_imshow(toshow[j], f"rank{get_rank()}_RaftDepth_prompt_{i}_{j}.png")
                #     else:
                #         draw_imshow(toshow, f"rank{get_rank()}_RaftDepth_prompt_{i}.png", vmax=0.1)
                #     # toshow = kwargs['L_Depth'][i].squeeze().detach().cpu().numpy() / 10
                #     # draw_imshow(toshow, f"rank{get_rank()}_GtDepth_prompt_{i}.png")
                #     # img_to_show = (feat[i,0].squeeze().detach().cpu().numpy()*255).astype(np.uint8)
                #     # pat_to_show = (feat[i,1].squeeze().detach().cpu().numpy()*255).astype(np.uint8)
                #     # if img_to_show.ndim == 3:
                #     #     img_to_show = img_to_show.transpose(1,2,0)
                #     #     pat_to_show = pat_to_show.transpose(1,2,0)
                #     # cv2.imwrite(f"rank{get_rank()}_RaftDepth_feat_img_{i}_{kwargs['key'][i].replace('/','_')}.png", img_to_show)
                #     # cv2.imwrite(f"rank{get_rank()}_RaftDepth_feat_pat_{i}_{kwargs['key'][i].replace('/','_')}.png", pat_to_show)
                # # mae = torch.mean(torch.abs(fused_to_show.squeeze() * 10  - kwargs['L_Depth'].squeeze())).item()
                # # print(f'rank{get_rank()} mae', mae)
                # # DEBUG ENDS

            else:
                fused = []
                extra = []
                for i in range(4):
                    feat_i = feat[i].reshape(b*2, *feat[i].shape[2:])
                    feat_i = self.pre_transformer_projector[i](feat_i)
                    feat_i = feat_i.reshape(b, 2, *feat_i.shape[1:])
                    fused_i, extra_i = self.transformer.forward(
                        feat_i, extri_2view, intri_2view, near_2view, far_2view, **kwargs
                    )  # b,v,c,h,w
                    fused.append(fused_i)
                    extra.append(extra_i)
        else:
            fused = []
            extra = []
            for i in range(4):
                feat_i = feat[i].reshape(b*2, *feat[i].shape[2:]) if not self.encoder_out_single else feat.reshape(b*2, *feat.shape[2:])
                feat_i = self.pre_transformer_projector[i](feat_i).reshape(b,2,*feat_i.shape[1:])
                fused_i, extra_i = self.transformer[i].forward(
                    feat_i, extri_2view, intri_2view, near_2view, far_2view, **kwargs
                )
                fused.append(fused_i)
                extra.append(extra_i)
        if isinstance(fused, torch.Tensor):
            return fused[:,0], extra  # only camera.
        else:
            return [f[:,0] for f in fused], extra