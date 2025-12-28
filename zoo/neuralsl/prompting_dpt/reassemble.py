# this script is modified from
# https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py
import torch
import torch.nn as nn

class ReassembleBlock(nn.Module):
    def __init__(self, in_feat_dim:int, out_channels:list[int], use_clstoken:bool=False):
        '''
        do reassembling.  
        handle cls tokens, resize, project.
        '''
        super().__init__()
        self.use_clstoken = use_clstoken
        self.in_feat = in_feat_dim
        self.out_feats = out_channels

        # mlp for cls tokens
        if self.use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * self.in_feat, self.in_feat),
                        nn.GELU()))
        # resize layers.
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=self.out_feats[0],
                out_channels=self.out_feats[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=self.out_feats[1],
                out_channels=self.out_feats[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=self.out_feats[3],
                out_channels=self.out_feats[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        # project layers
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.in_feat,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in self.out_feats
        ])

    def forward(self, vit_features:list[torch.Tensor], n_patch_h:int, n_patch_w:int):
        '''
        out_features: list of features of certain layers in vit.  
        n_patch_h, n_patch_w, number of pathces in vertical/horizontal direction
        '''
        out = []
        for idx, x in enumerate(vit_features):
            # handle cls token.
            if self.use_clstoken:
                x, cls_token = x[0], x[1]  # x: (b, npatch, n_feat), cls_token: (b, n_feat)
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[idx](torch.cat((x, readout), -1))
            else:
                x = x[0]
            ## x: b, npatch, nfeat.
            # permute to bchw
            b, npatch, nfeat = x.shape
            x = x.permute(0, 2, 1).reshape(b, nfeat, n_patch_h, n_patch_w)
            # project
            x = self.projects[idx](x)
            # resize
            x = self.resize_layers[idx](x)
            out.append(x)
        return out