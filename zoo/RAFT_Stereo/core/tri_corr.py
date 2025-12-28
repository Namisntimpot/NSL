import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler

try:
    import corr_sampler
except:
    pass

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass

class TriCorrBlock1D:
    def __init__(self, fmap1_lr, fmap2_lr, fmap1_lm, fmap2_lm, num_levels=4, radius=4, middle_rate=0.5):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid_lm = []
        self.corr_pyramid_lr = []
        self.middle_rate = middle_rate

        # all pairs correlation
        corr_lr = TriCorrBlock1D.corr(fmap1_lr, fmap2_lr)
        corr_lm = TriCorrBlock1D.corr(fmap1_lm, fmap2_lm)
        
        b_lr, h1_lr, w1_lr, d_lr, w2_lr = corr_lr.shape
        corr_lr = corr_lr.reshape(b_lr * h1_lr * w1_lr, 1, 1, w2_lr)

        b_lm, h1_lm, w1_lm, d_lm, w2_lm = corr_lm.shape
        corr_lm = corr_lm.reshape(b_lm * h1_lm * w1_lm, 1, 1, w2_lm)

        self.corr_pyramid_lm.append(corr_lm)
        for i in range(self.num_levels):
            corr_lm = F.avg_pool2d(corr_lm, [1,2], stride=[1,2])
            self.corr_pyramid_lm.append(corr_lm)

        self.corr_pyramid_lr.append(corr_lr)
        for i in range(self.num_levels):
            corr_lr = F.avg_pool2d(corr_lr, [1,1], stride=[1,2])
            self.corr_pyramid_lr.append(corr_lr)

    def __call__(self, coords_lr, coords0):
        r = self.radius
        coords_lr = coords_lr[:, :1].permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords_lr.shape

        coords0 = coords0[:, :1].permute(0, 2, 3, 1)
        flow_lr = coords_lr - coords0
        flow_lm = flow_lr * self.middle_rate
        coords_lm = coords0 + flow_lm

        out_pyramid = []
        for i in range(self.num_levels):
            corr_lr = self.corr_pyramid_lr[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(2*r+1, 1).to(coords_lr.device)
            x0 = dx + coords_lr.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0,y0], dim=-1)
            corr_lr = bilinear_sampler(corr_lr, coords_lvl)
            corr_lr = corr_lr.view(batch, h1, w1, -1)
            out_pyramid.append(corr_lr)

            corr_lm = self.corr_pyramid_lm[i]
            # dx = dx * self.middle_rate
            # x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            x0 = dx * self.middle_rate + coords_lm.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)
            coords_lvl = torch.cat([x0,y0], dim=-1)
            corr_lm = bilinear_sampler(corr_lm, coords_lvl)
            corr_lm = corr_lm.view(batch, h1, w1, -1)
            out_pyramid.append(corr_lm)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())
    

class TriCorrBlock1D_Buggy:  # 为了能进行测试.
    def __init__(self, fmap1_lr, fmap2_lr, fmap1_lm, fmap2_lm, num_levels=4, radius=4, middle_rate=0.5):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid_lm = []
        self.corr_pyramid_lr = []
        self.middle_rate = middle_rate

        # all pairs correlation
        corr_lr = TriCorrBlock1D.corr(fmap1_lr, fmap2_lr)
        corr_lm = TriCorrBlock1D.corr(fmap1_lm, fmap2_lm)
        
        b_lr, h1_lr, w1_lr, d_lr, w2_lr = corr_lr.shape
        corr_lr = corr_lr.reshape(b_lr * h1_lr * w1_lr, 1, 1, w2_lr)

        b_lm, h1_lm, w1_lm, d_lm, w2_lm = corr_lm.shape
        corr_lm = corr_lm.reshape(b_lm * h1_lm * w1_lm, 1, 1, w2_lm)

        self.corr_pyramid_lm.append(corr_lm)
        for i in range(self.num_levels):
            corr_lm = F.avg_pool2d(corr_lm, [1,2], stride=[1,2])
            self.corr_pyramid_lm.append(corr_lm)

        self.corr_pyramid_lr.append(corr_lr)
        for i in range(self.num_levels):
            corr_lr = F.avg_pool2d(corr_lr, [1,1], stride=[1,2])
            self.corr_pyramid_lr.append(corr_lr)

    def __call__(self, coords_lr, coords0=None):
        r = self.radius
        coords_lr = coords_lr[:, :1].permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords_lr.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr_lr = self.corr_pyramid_lr[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(2*r+1, 1).to(coords_lr.device)
            x0 = dx + coords_lr.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0,y0], dim=-1)
            corr_lr = bilinear_sampler(corr_lr, coords_lvl)
            corr_lr = corr_lr.view(batch, h1, w1, -1)
            out_pyramid.append(corr_lr)

            corr_lm = self.corr_pyramid_lm[i]
            x0 = dx * self.middle_rate + coords_lr.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)
            coords_lvl = torch.cat([x0,y0], dim=-1)
            corr_lm = bilinear_sampler(corr_lm, coords_lvl)
            corr_lm = corr_lm.view(batch, h1, w1, -1)
            out_pyramid.append(corr_lm)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())