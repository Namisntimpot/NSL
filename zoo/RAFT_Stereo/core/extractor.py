import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)



class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, downsample=3):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x, dual_inp=False):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = x.split(split_size=batch_dim, dim=0)

        return x

class MultiBasicEncoder(nn.Module):
    def __init__(self, output_dim=[128], norm_fn='batch', dropout=0.0, downsample=3):
        super(MultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs32 = nn.ModuleList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False, num_layers=3):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if dual_inp:
            v = x
            x = x[:(x.shape[0]//2)]

        outputs08 = [f(x) for f in self.outputs08]
        if num_layers == 1:
            return (outputs08, v) if dual_inp else (outputs08,)

        y = self.layer4(x)
        outputs16 = [f(y) for f in self.outputs16]

        if num_layers == 2:
            return (outputs08, outputs16, v) if dual_inp else (outputs08, outputs16)

        z = self.layer5(y)
        outputs32 = [f(z) for f in self.outputs32]

        return (outputs08, outputs16, outputs32, v) if dual_inp else (outputs08, outputs16, outputs32)


class Dav2ContextEncoder(nn.Module):
    PATCH_SIZE = 14
    STEM_DIM_32X = 384
    STEM_DIM_16X = 192
    STEM_DIM_8X  = 96
    STEM_DIM_4X  = 48
    STEM_DIM_2X  = 32
    def __init__(self, args, output_dim=[128], norm_fn='batch', dropout=0.0, downsample=3):
        assert downsample == 2, "only support downsample=2 now."
        super().__init__()
        from depth_anything_v2.dpt import DINOv2, DPTHead_decoder
        from .blocks import BasicConv_IN
        self.vit_type = args.vit_type
        self.pretrained_path = args.pretrained_path
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        mono_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.pretrained = DINOv2(self.vit_type)
        pretrained_params = torch.load(self.pretrained_path, map_location='cpu')
        pretrained_params = pretrained_params['model'] if 'model' in pretrained_params else pretrained_params
        try:
            self.pretrained.load_state_dict(pretrained_params)
        except:  # ckpt of dav2
            vit_params = {
                ".".join(k.split(".")[1:]): v for k, v in pretrained_params.items() \
                if k.startswith("pretrained.")
            }
            self.pretrained.load_state_dict(vit_params)
        
        self.vit_features_dim = mono_model_configs[self.vit_type]['features']
        self.out_channels = mono_model_configs[self.vit_type]['out_channels']

        self.depth_head = DPTHead_decoder(self.pretrained.embed_dim, self.vit_features_dim,
                                          args.use_bn, self.out_channels, args.use_clstoken)
        dpt_params = {
            ".".join(k.split(".")[1:]): v for k, v in pretrained_params.items() \
            if k.startswith("depth_head.")
        }
        self.depth_head.load_state_dict(dpt_params, strict=False)

        # stems.
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, self.STEM_DIM_2X, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.STEM_DIM_2X, self.STEM_DIM_2X, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(self.STEM_DIM_2X), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(self.STEM_DIM_2X, self.STEM_DIM_4X, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.STEM_DIM_4X, self.STEM_DIM_4X, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(self.STEM_DIM_4X), nn.ReLU()
            )

        self.stem_8 = nn.Sequential(
            BasicConv_IN(self.STEM_DIM_4X, self.STEM_DIM_8X, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.STEM_DIM_8X, self.STEM_DIM_8X, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(self.STEM_DIM_8X), nn.ReLU()
            )

        self.stem_16 = nn.Sequential(
            BasicConv_IN(self.STEM_DIM_8X, self.STEM_DIM_16X, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.STEM_DIM_16X, self.STEM_DIM_16X, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(self.STEM_DIM_16X), nn.ReLU()
            )
        
        # self.stem_32 = nn.Sequential(
        #     BasicConv_IN(self.STEM_DIM_16X, self.STEM_DIM_32X, kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(self.STEM_DIM_32X, self.STEM_DIM_32X, 3, 1, 1, bias=False),
        #     nn.InstanceNorm2d(self.STEM_DIM_32X), nn.ReLU()
        #     )
        
        # output projectors
        ## 1/4
        output_list = []
        # inp_dim = self.out_channels[3] + self.STEM_DIM_4X
        inp_dim = self.vit_features_dim + self.STEM_DIM_4X
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(inp_dim, inp_dim, norm_fn, stride=1),
                nn.Conv2d(inp_dim, dim[2], 3, padding=1))
            output_list.append(conv_out)
        self.outputs4 = nn.ModuleList(output_list)

        ## 1/8
        output_list = []
        # inp_dim = self.out_channels[1] + self.STEM_DIM_8X
        inp_dim = self.vit_features_dim + self.STEM_DIM_8X
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(inp_dim, inp_dim, norm_fn, stride=1),
                nn.Conv2d(inp_dim, dim[1], 3, padding=1))
            output_list.append(conv_out)
        self.outputs08 = nn.ModuleList(output_list)

        ## 1/16
        output_list = []
        # inp_dim = self.out_channels[2] + self.STEM_DIM_16X
        inp_dim = self.vit_features_dim + self.STEM_DIM_16X
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(inp_dim, inp_dim, norm_fn, stride=1),
                nn.Conv2d(inp_dim, dim[0], 3, padding=1))
            output_list.append(conv_out)
        self.outputs16 = nn.ModuleList(output_list)


        if args.freeze_vit:
            print("freeze vit encoder!")
            self.freeze_vit()

        # padder for vit
        from core.utils.utils import InputPadder
        self.padder:InputPadder = None

    def freeze_vit(self):
        for p in self.pretrained.parameters():
            p.requires_grad_(False)
        
    def forward(self, x, dual_inp=False, num_layers=3):
        assert num_layers == 3 and not dual_inp, 'now only support num_layers=3 and dual_inp=False'
        if self.padder is None or x.shape[-2:] != (self.padder.ht, self.padder.wd):
            from core.utils.utils import InputPadder
            self.padder = InputPadder(x.shape[-2:], divis_by=self.PATCH_SIZE)        
        
        x_pad = self.padder.pad(x)[0]
        patch_h, patch_w = x_pad.shape[-2] // self.PATCH_SIZE, x_pad.shape[-1] // self.PATCH_SIZE
        vit_feats = self.pretrained.get_intermediate_layers(x_pad, self.intermediate_layer_idx[self.vit_type], return_class_token=True)
        l_feat_4x, l_feat_8x, l_feat_16x, l_feat_32x = self.depth_head.forward(vit_feats, patch_h, patch_w)

        # stems
        stem_2x = self.stem_2(x)
        stem_4x = self.stem_4(stem_2x)
        stem_8x = self.stem_8(stem_4x)
        stem_16x= self.stem_16(stem_8x)

        if l_feat_4x.shape[-2:] != stem_4x.shape[-2:]:
            l_feat_4x = F.interpolate(l_feat_4x, stem_4x.shape[-2:], align_corners=True, mode='bilinear')
        conj_feat_4= torch.concat((l_feat_4x, stem_4x), dim=1)
        outputs4x= [f(conj_feat_4) for f in self.outputs4]

        if l_feat_8x.shape[-2:] != stem_8x.shape[-2:]:
            l_feat_8x = F.interpolate(l_feat_8x, stem_8x.shape[-2:], align_corners=True, mode='bilinear')
        conj_feat_8 = torch.concat((l_feat_8x, stem_8x), dim=1)
        outputs8x = [f(conj_feat_8) for f in self.outputs08]

        if l_feat_16x.shape[-2:] != stem_16x.shape[-2:]:
            l_feat_16x = F.interpolate(l_feat_16x, stem_16x.shape[-2:], align_corners=True, mode='bilinear')
        conj_feat_16= torch.concat((l_feat_16x, stem_16x), dim=1)
        outputs16x= [f(conj_feat_16) for f in self.outputs16]

        return outputs4x, outputs8x, outputs16x