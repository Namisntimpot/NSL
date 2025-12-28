import torch
import torch.nn as nn
import torch.nn.functional as F
from ..raft.extractor import BasicEncoder as RawRaftBasicEncoder

class IdentityEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pattern, lcn):
        return pattern, lcn
    
class SlideWindowEncoder(nn.Module):
    def __init__(self, ksize:int = 13, downscale=1, **kwargs):
        super().__init__()
        self.ksize = ksize
        self.downscale = downscale
    def forward(self, pattern:torch.Tensor, lcn:torch.Tensor):
        # 都是(B,C,H,W)
        B,C,H,W = pattern.shape
        if self.downscale != 1:
            H = H // self.downscale
            W = W // self.downscale
            pattern = F.interpolate(pattern, (H, W), mode='bilinear', align_corners=False)
            lcn = F.interpolate(lcn, (H, W), mode='bilinear', align_corners=False)

        pad = self.ksize // 2
        pattern = F.unfold(pattern, self.ksize, padding=pad).view(B,-1,H,W)
        lcn = F.unfold(lcn, self.ksize, padding=pad).view(B,-1,H,W)
        return pattern, lcn
    
class RaftResidualBlock(nn.Module):
    '''
    this class is modified from https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/extractor.py
    '''
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(RaftResidualBlock, self).__init__()
  
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
    
class RaftEncoder(nn.Module):
    def __init__(self, out_channels:int, downscale:int, resblock_type:str, norm_type:str, dropout:float=0.0, **kwargs):
        '''
        resblock_type: raft or da  
        downscale: 1, 2, 4, 8 ...  
        norm_type: group, batch, instance, none  
        '''
        super().__init__()
        num_downscales = int(torch.floor(torch.log2(torch.tensor(downscale, dtype=torch.float32))))
        self.num_layers = max(0, num_downscales)
        self.resblock_type = resblock_type
        self.norm_type = norm_type
        self.norm_type == 'batch' if self.resblock_type=='da' and self.norm_type!='none' else self.norm_type
        rgb = kwargs.get("rgb", False)
        self.conv = nn.Conv2d(1 if not rgb else 3, out_channels, kernel_size=7, padding=3, stride=1)
        if self.norm_type == 'group':
            self.norm_fn = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        elif self.norm_type == 'instance':
            self.norm_fn = nn.InstanceNorm2d(out_channels)
        elif self.norm_type == 'batch':
            self.norm_fn = nn.BatchNorm2d(out_channels)
        elif self.norm_type == 'none':
            self.norm_fn = nn.Identity()
        else:
            raise ValueError(f"Unkown norm type: {self.norm_type}")
        for i in range(self.num_layers):
            layer = self._make_layer(out_channels, out_channels)
            setattr(self, f"layer{i}", layer)
        self.dropout = nn.Dropout2d(dropout)

    def _make_layer(self, in_channels, out_channels):
        '''
        2个resblocks, 会将分辨率减半.
        '''
        if self.resblock_type == 'raft':
            layer1 = RaftResidualBlock(in_channels, out_channels, self.norm_type, stride=2)
            layer2 = RaftResidualBlock(out_channels, out_channels, self.norm_type, stride=1)
            layers = [layer1, layer2]
        elif self.resblock_type == 'da':
            assert in_channels == out_channels
            from .blocks import ResidualConvUnit
            layer0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            layer1 = ResidualConvUnit(in_channels, nn.ReLU(False), self.norm_type!='none')
            layer2 = ResidualConvUnit(in_channels, nn.ReLU(False), self.norm_type!='none')
            layers = [layer0, layer1, layer2]
        else:
            raise ValueError(f"Unknown resblock_type: {self.resblock_type}")
        return nn.Sequential(*layers)
    
    def forward(self, pattern:torch.Tensor, lcn:torch.Tensor):
        '''
        pattern & lcn: b, 1, h, w  
        '''
        b = pattern.shape[0]
        imgs = torch.concat((pattern, lcn), dim=0)
        if self.conv.in_channels == 3 and imgs.shape[1] == 1:
            imgs = imgs.expand(2*b, 3, *imgs.shape[2:])
        x = F.relu(self.norm_fn(self.conv(imgs)))
        for i in range(self.num_layers):
            x = getattr(self, f"layer{i}")(x)
        x = self.dropout(x)
        p, l = torch.split(x, b)
        return p, l
    

class RawRaftEncoder(RawRaftBasicEncoder):
    '''Ensure that the model structure and parameter keys are completely consistent'''
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0, downsample=3, **kwargs):
        # input image must have 3 channels
        super().__init__(output_dim, norm_fn, dropout, downsample)

    def forward(self, pattern:torch.Tensor, lcn:torch.Tensor):
        b = pattern.shape[0]
        if pattern.shape[1] != 3:
            pattern = pattern.expand(b, 3, *pattern.shape[2:])
            lcn = lcn.expand(b, 3, **lcn.shape[2:])
        return super().forward([pattern, lcn])


class ShallowCnnEncoder(nn.Module):
    def __init__(self, enc_feat_dim:int, downscale:bool, activation:str='Sigmoid', **kwargs):
        '''
        downscale: only true (1/4) or false.  
        activation: activation module name contained in nn. default: sigmoid (for sparsity)  
        '''
        super().__init__()
        self.prompt_feat_dim = enc_feat_dim
        self.downscale = downscale
        activation = getattr(nn, activation)()
        self.rgb = kwargs.get("rgb", False)
        self.layers = nn.Sequential(
            nn.Conv2d(1 if not self.rgb else 3, enc_feat_dim, kernel_size=3, padding=1, stride=2 if downscale else 1, bias=True, groups=1),
            activation,
            nn.Conv2d(enc_feat_dim, enc_feat_dim, kernel_size=3, padding=1, stride=2 if downscale else 1, bias=True, groups=1),
            activation
        )
    def forward(self, pattern:torch.Tensor, lcn:torch.Tensor):
        b = pattern.shape[0]
        if self.rgb and pattern.shape[1] == 1:
            pattern = pattern.expand(b, 3, *pattern.shape[2:])
            lcn = lcn.expand(b, 3, *lcn.shape[2:])
        imgs = torch.concat((pattern, lcn), dim=0)
        imgs = self.layers(imgs)
        p, l = torch.split(imgs, b, dim=0)
        return p, l


class BackboneEncoder(nn.Module):
    def __init__(self, name:str, enc_feat_dim:int, pretrained:bool=True, **cfgs):
        raise NotImplementedError("backbones大多输出非稠密特征，暂未处理.")
        super().__init__()
        import timm
        self.backbone:nn.Module = timm.create_model(name, pretrained=pretrained)
        self.encoder_feat_dim = enc_feat_dim

    def forward(self, pattern:torch.Tensor, lcn:torch.Tensor):
        b, c, h, w = lcn.shape
        if c == 1:
            pattern = pattern.repeat(1,3,1,1)
            lcn = lcn.repeat(1,3,1,1)
        inp = torch.concat((pattern, lcn), dim=0)
        out = self.backbone(inp)
        p, l = torch.split(out, b, dim=0)
        return p, l    

class BackboneMultiMidLayersEncoder(nn.Module):
    def __init__(self, name:str, pretrained:bool=True, **cfgs):
        super().__init__()
        import timm
        from utils.models import remove_module_after_layer
        self.backbone:nn.Module = timm.create_model(name, pretrained=pretrained)
        self.intermediate_layers = {}
        self.intermediate_feat_dims = {}
        
        def hook_fn(module, input, output, layer_name):
            self.intermediate_layers[layer_name] = output
        
        for i in range(4):  # 注意0是最浅的.
            k = f'out{i}'
            cfg = cfgs[k]
            self.intermediate_feat_dims[k] = cfg['feat_dim']
            module = self.backbone.get_submodule(cfg['layername'])
            module.register_forward_hook(
                lambda module, input, output, name=k: hook_fn(module, input, output, name)
            )
        last_reserve_layer = cfgs['out3']['layername']
        self.backbone = remove_module_after_layer(self.backbone, last_reserve_layer)

    def get_intermediate_feat_dims(self):
        return self.intermediate_feat_dims
    
    def forward(self, pattern:torch.Tensor, lcn:torch.Tensor):
        '''
        both pattern and lcn is (b, c, h, w)
        '''
        b, c, h, w = lcn.shape
        if c == 1:
            c=3
            pattern = pattern.expand(b, 3, h, w)
            lcn = lcn.expand(b,3,h,w)
        b = pattern.shape[0]
        x = torch.concat((pattern, lcn), dim=0)
        self.backbone.forward(x)
        pats_enc, lcns_enc = [], []
        for i in range(4):
            k = f"out{i}"
            feat = self.intermediate_layers[k]
            p, l = torch.split(feat, b)
            pats_enc.append(p)
            lcns_enc.append(l)
        return pats_enc, lcns_enc