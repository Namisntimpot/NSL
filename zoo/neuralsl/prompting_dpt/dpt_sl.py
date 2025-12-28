# This script is modified from
# https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import _make_scratch, _make_prompt_fusion_block


class DPTHead(nn.Module):
    def __init__(self,
                 nclass,
                 in_channels,
                 prompt_feat_dim,
                 features_dim=256,
                 out_channels=[256, 512, 1024, 1024],
                 use_bn=False,
                 use_clstoken=False,
                 output_act='sigmoid',
                 prompt_feat_proj_type = 'cnn',
                 use_zero_module = True,
                 do_reassemble = True,
                 **kwargs
                ):
        '''
        in_channel: dim of vit encoded output featueres.  
        out_channels: output dim of each reassemble sub module.  
        features_dim: feature dim within DPT's fusion blocks  
        (patch_reso, in_channel) -reassemble-> (multi-scale reso, out_channels[i]) -resnet-> (multi-scale reso, features_dim)  
        '''
        super(DPTHead, self).__init__()

        self.nclass = nclass
        self.use_clstoken = use_clstoken
        self.out_channels = out_channels
        self.feat_dim = features_dim
        self.prompt_feat_dim = prompt_feat_dim
        self.prompt_feat_proj_type = prompt_feat_proj_type
        self.use_zero_module = use_zero_module
        self.do_reassemble = do_reassemble

        if self.do_reassemble:
            from .reassemble import ReassembleBlock
            self.reassemble_block = ReassembleBlock(
                in_channels, self.out_channels, self.use_clstoken,
            )

        self.scratch = _make_scratch(
            self.out_channels,
            features_dim,
            groups=1,
            expand=False,
        )   # the resnet after reassembling.

        self.scratch.stem_transpose = None

        self.scratch = self._make_fusion_blocks(
            self.scratch, features_dim, use_bn, self.prompt_feat_dim,
            self.prompt_feat_proj_type, self.use_zero_module, **kwargs
        )

        head_features_1 = features_dim
        head_features_2 = 32

        if output_act == 'sigmoid':
            act_func = nn.Sigmoid()
        elif output_act == 'relu':
            act_func = nn.ReLU(False)
        else:
            act_func = nn.Identity()

        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(False),
                nn.Conv2d(head_features_1, nclass,
                          kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(False),
                nn.Conv2d(head_features_2, 1, kernel_size=1,
                          stride=1, padding=0),   # 在这个.weights，会报警告Grad strides do not match bucket view strides.未知原因.
                act_func,
            )

    def _make_fusion_blocks(self, 
            scratch:nn.Module, features_dim, use_bn, prompt_feat_dim, prompt_feat_project_type,
            use_zero_module, **kwargs):
        scratch.refinenet1 = _make_prompt_fusion_block(
            features_dim, use_bn, prompt_feat_dim=prompt_feat_dim,
            prompt_feat_project_type=prompt_feat_project_type, use_zero_module=use_zero_module)
        scratch.refinenet2 = _make_prompt_fusion_block(
            features_dim, use_bn, prompt_feat_dim=prompt_feat_dim,
            prompt_feat_project_type=prompt_feat_project_type, use_zero_module=use_zero_module)
        scratch.refinenet3 = _make_prompt_fusion_block(
            features_dim, use_bn, prompt_feat_dim=prompt_feat_dim,
            prompt_feat_project_type=prompt_feat_project_type, use_zero_module=use_zero_module)
        scratch.refinenet4 = _make_prompt_fusion_block(
            features_dim, use_bn, prompt_feat_dim=prompt_feat_dim,
            prompt_feat_project_type=prompt_feat_project_type, use_zero_module=use_zero_module)
        return scratch
        

    def forward(self, encoder_features, n_patch_h, n_patch_w, prompt_feat=None):
        '''
        encoder_features: list of features of certain layers in vit.  
        if self.do_reassemble = true, out features' shape is (b, n_patches, n_feat). 
        otherwise the shape is (b, c, n_patch_h, n_patch_w) (each has a different c)  
        '''
        # reassemble
        out = self.reassemble_block.forward(encoder_features, n_patch_h, n_patch_w) \
                if self.do_reassemble else encoder_features
        # end reassemble

        layer_1, layer_2, layer_3, layer_4 = out
        if isinstance(prompt_feat, torch.Tensor):
            prompt_feat_1 = prompt_feat_2 = prompt_feat_3 = prompt_feat_4 = prompt_feat
        else:
            prompt_feat_1, prompt_feat_2, prompt_feat_3, prompt_feat_4 = prompt_feat

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(
            layer_4_rn, size=layer_3_rn.shape[2:], prompt_feat=prompt_feat_4)
        path_3 = self.scratch.refinenet3(
            path_4, layer_3_rn, size=layer_2_rn.shape[2:], prompt_feat=prompt_feat_3)
        path_2 = self.scratch.refinenet2(
            path_3, layer_2_rn, size=layer_1_rn.shape[2:], prompt_feat=prompt_feat_2)
        path_1 = self.scratch.refinenet1(
            path_2, layer_1_rn, prompt_feat=prompt_feat_1)
        out = self.scratch.output_conv1(path_1)
        out_feat = F.interpolate(
            out, (int(n_patch_h * 14), int(n_patch_w * 14)),
            mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out_feat)
        return out