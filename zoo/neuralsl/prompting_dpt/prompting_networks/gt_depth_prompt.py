import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.transforms import depth_conversion

class GtDepthPrompt(nn.Module):
    def __init__(self, d_in, num_context_views, out_reso, depth_type, mode='dense_gt', converse_depth:bool=True):
        '''
        待扩展：加噪声、变稀疏、噪声+伪置信度...  
        out_reso: (h, w)  
        depth_type: [no, norm_depth, min_max_ref_depth, linear_scale_ref_depth, abs_disp, rel_disp, 
                     raw_abs_disp, raw_rel_disp]  
        '''
        super().__init__()
        self.d_in = d_in  # useless.
        self.out_reso = out_reso

        self.mode = mode
        self.out_dim = 1  # extensible
        self.depth_type = depth_type
        self.converse_depth = converse_depth

        self.num_context_views = num_context_views

    def quary_support_bino_imgs(self):
        return True
    
    def forward(self, features:torch.Tensor, extrinsics:torch.Tensor, 
                intrinsics:torch.Tensor, near:torch.Tensor, far:torch.Tensor, 
                **kwargs):
        '''
        features: BVCHW, extrinsics: BVCHW  
        extrinsics: BV44, intrinsics: BV33, near/far: BV  
        there must be L_Depth, R_Depth in kwargs. each of them should be of (B,(1),H,W)  
        '''
        # 2 possibilities: features[:,0] is image and featuers[:,1] is pattern; or features[:,0] is left and features[:,1] is right.  
        if 'L_Depth' in kwargs and 'R_Depth' in kwargs:
            L_Depth, R_Depth = kwargs['L_Depth'], kwargs['R_Depth']
        else:
            raise ValueError("In kwargs, there must be either both L_Depth and R_Depth, or Depth")
        
        if L_Depth.ndim == 3:
            L_Depth = L_Depth.unsqueeze(1)
        if R_Depth.ndim == 3:
            R_Depth = R_Depth.unsqueeze(1)  # B1HW
        # resize
        b_feat = features.shape[0]
        b_dep, _, h, w = L_Depth.shape
        Depth = torch.stack((L_Depth, R_Depth), dim=1).reshape(2*b_dep, 1, h, w)
        if (h,w) != self.out_reso:
            Depth = F.interpolate(Depth, self.out_reso, mode='nearest')
        # depth conversion
        if b_dep == b_feat:  # binocular, left right ir images.
            near_dep_converse = near.view(-1)
            far_dep_converse = far.view(-1)
            intri_dep_converse = intrinsics.view(b_dep*2, 3, 3)
            extri_dep_converse = extrinsics.view(b_dep*2, 4, 4)
        else:                # ir_pattern
            near_dep_converse = near[:,0]
            far_dep_converse = far[:,0]
            intri_dep_converse= intrinsics[:,0]
            extri_dep_converse= extrinsics[:,0]

        if self.converse_depth:
            Depth = depth_conversion(
                Depth, self.depth_type, reverse=False,
                near=near_dep_converse, far=far_dep_converse,
                intri=intri_dep_converse, extri1=extri_dep_converse, 
                ref_depth=Depth
            )  # 2*b_dep, 1, h, 2  # WARNING: 暂时不考虑提供额外的reference_depth!

        if b_dep == b_feat: # binocular, left right ir images.
            return Depth.view(b_dep, 2, 1, h, w), None
        else:               # ir_pattern for pattern prompting.
            return Depth.unsqueeze(1), None