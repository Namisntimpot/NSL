# modified from: 
# https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/depth_anything_v2/util/blocks.py
# https://github.com/DepthAnything/PromptDA/blob/main/promptda/model/blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath


def _make_prompting_network(ty:str, cfg:dict, encoder_feat_dim:int, prompt_feat_dim:int):
    '''
    cfg: {'num_context_views': int, 'cfg': {...}, 'depth_type': str}  
    return: module, dim_input, dim_output.  
    '''
    if ty == 'CostVolume':
        from .prompting_networks.cost_volume import CostVolumePrompt
        return CostVolumePrompt(d_in=encoder_feat_dim, **cfg), encoder_feat_dim, 2
    elif ty == 'GtDepth':
        from .prompting_networks.gt_depth_prompt import GtDepthPrompt
        prompting_net = GtDepthPrompt(d_in=encoder_feat_dim, **cfg)
        return prompting_net, encoder_feat_dim, prompting_net.out_dim
    elif ty == 'RaftDepth':
        from .prompting_networks.raft import RaftDepthPrompt
        prompting_net = RaftDepthPrompt(**cfg)
        return prompting_net, encoder_feat_dim, prompting_net.out_dim
    elif ty == 'TriRaftDepth':
        from .prompting_networks.raft import TriRaftDepthPrompt
        prompting_net = TriRaftDepthPrompt(**cfg)
        return prompting_net, encoder_feat_dim, prompting_net.out_dim
    elif ty == 'DualCorrRaftDepth':
        from .prompting_networks.raft import DualCorrRaftDepthPrompt
        prompting_net = DualCorrRaftDepthPrompt(**cfg)
        return prompting_net, encoder_feat_dim, prompting_net.out_dim
    else:
        raise ValueError(f"Unknown prompting network type: {ty}")


def _make_base_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


def _make_prompt_fusion_block(
        features, use_bn, size=None, 
        prompt_feat_dim=1, prompt_feat_project_type='cnn', use_zero_module=True,
    ):
    return FeatureFusionPromptBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
        prompt_feat_dim=prompt_feat_dim,
        prompt_feat_project_type = prompt_feat_project_type,
        use_zero_module=use_zero_module
    )


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)
    

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 
            kernel_size, stride=stride, padding=padding, dilation=dilation, 
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None
    ):
        """Init.
        
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        
        output = self.out_conv(output)

        return output


class FeatureFusionControlBlock(FeatureFusionBlock):
    """Feature fusion block.
    这个类应该是没有实现.  
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super.__init__(features, activation, deconv,
                       bn, expand, align_corners, size)
        self.copy_block = FeatureFusionBlock(
            features, activation, deconv, bn, expand, align_corners, size)

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class FeatureFusionPromptBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
            self, feat_dim, activation, deconv=False, bn=False, expand=False, 
            align_corners=True, size=None, prompt_feat_dim=1, prompt_feat_project_type='cnn',
            use_zero_module=True,):
        """
        **feat_dim**: length of input feature.  
        **activation**: activation module such as nn.ReLu, nn.Gelu...  
        **decomv**: it does not seem to be used.  
        **bn**: whether to perform batch normalization in ResBlocks  
        **expand**: whether to expand feat_dims within fusion blocks. it halves the output dimension. it is always set to false. seems always to be false.
        **align_corners**: used when doubling the resolution with F.interpolate(...) at the end.  
        **size**: size of the output feature map
        **prompt_feat_dim**: dim of prompting feature inputs.
        **prompt_feat_project_type**: how to aligh the dimesions between prompt feat and internal feat. 'cnn', 'mlp', 'none'  
        **use_zero_module**: whether to use zero module at the end of prompt feature projection.   
        """
        super(FeatureFusionPromptBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = feat_dim
        if self.expand == True:
            out_features = feat_dim//2

        self.out_conv = nn.Conv2d(
            feat_dim, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(feat_dim, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(feat_dim, activation, bn)

        # about prompt feature
        self.prompt_feat_dim = prompt_feat_dim
        self.prompt_feat_project_type = prompt_feat_project_type
        if prompt_feat_project_type == 'cnn':
            self.prompt_feat_projector = nn.Sequential(
                nn.Conv2d(self.prompt_feat_dim, feat_dim, kernel_size=3, stride=1,
                        padding=1, bias=True, groups=1),
                activation,
                nn.Conv2d(feat_dim, feat_dim, kernel_size=3,
                        stride=1, padding=1, bias=True, groups=1),
                activation,
            )
        elif prompt_feat_project_type == 'mlp':
            self.prompt_feat_projector = nn.Sequential(
                nn.Linear(self.prompt_feat_dim, feat_dim), activation, 
                nn.Linear(feat_dim, feat_dim), activation,
            )
        elif prompt_feat_project_type == 'none':
            assert self.prompt_feat_dim == feat_dim, "dimensions of input feature and prompt feature must match!"
            self.prompt_feat_projector == nn.Sequential(
                nn.Identity()
            )
        else:
            raise ValueError(f"Unkown prompt feature projection type: {prompt_feat_project_type}.")
        # self.prompt_injection_type = prompt_injection_type 先不考虑这个。
        # if prompt_injection_type == 'add':
        #     pass  # nothing to do here.
        # elif prompt_injection_type == 'concat':
        #     self.injection_projector = nn.Conv2d(2*feat_dim, feat_dim, 1)
        
        
        if use_zero_module:
            self.prompt_feat_projector.append(
                zero_module(nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0, bias=True, groups=1)) \
                if prompt_feat_project_type == 'cnn' else \
                zero_module(nn.Linear(feat_dim, feat_dim))
            )
        
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, prompt_feat:torch.Tensor=None, size=None):
        """
        **xs**: for the first fusion block, xs is the reassembled feature with lowest resolution from vit.
        for other fusion blocks, xs[0] is the output from the previous fusion block, and xs[1] is the reassebled 
        feature with the same resolution from vit. the shapes are (b,c,h,w)  
        **prompt_feat**: prompting feature map. if prompt feature map has different resolution, it will be interpolated to 
        the target resolution. (b, c_prompt, h, w)  
        **size**: target output resolution. by default, fusion block will doubling resolution.  
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        b, c, h, w = output.shape
        if prompt_feat is not None:
            b, pdim, ph, pw = prompt_feat.shape
            if h!=ph or w != pw:
                prompt_feat = F.interpolate(
                    prompt_feat, output.shape[2:], mode='bilinear', align_corners=False
                )
            if self.prompt_feat_project_type == 'cnn':
                res = self.prompt_feat_projector(prompt_feat)
            else:
                prompt_feat = prompt_feat.permute(0, 2, 3, 1)
                res = self.prompt_feat_projector(prompt_feat)
                res = res.permute(0, 3, 1, 2)
            output = self.skip_add.add(output, res)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output