import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock, BasicMultiUpdateBlockWithConf
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.utils.utils import coords_grid, upflow8
from core.tri_corr import TriCorrBlock1D, TriCorrBlock1D_Buggy

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class TriRAFTStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert not self.args.shared_backbone
        
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        print(f"shared fnet: {self.args.shared_fnet}")
        print(f"TriRAFTStereo: will use buggy tri-corr class: {hasattr(self.args, 'buggy_tri_corr') and self.args.buggy_tri_corr}")
        if not self.args.shared_fnet:
            self.fnet_lr = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)
            self.fnet_lm = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)


    def forward(self, image_l, image_r, image_m, iters=12, flow_init=None, test_mode=False, corr_middle_rate=None):
        """ Estimate optical flow between pair of frames """

        image_l = (2 * (image_l / 255.0) - 1.0).contiguous()
        image_r = (2 * (image_r / 255.0) - 1.0).contiguous()
        image_m = (2 * (image_m / 255.0) - 1.0).contiguous()

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet_list = self.cnet(image_l, num_layers=self.args.n_gru_layers)
            if not self.args.shared_fnet:
                fmap1_lr, fmap2_lr = self.fnet_lr([image_l, image_r])
                fmap1_lm, fmap2_lm = self.fnet_lm([image_l, image_m])
            else:
                fmap1_lr, fmap2_lr, fmap2_lm = self.fnet([image_l, image_r, image_m])
                fmap1_lm = fmap1_lr
            fmap1_lr, fmap2_lr, fmap1_lm, fmap2_lm = fmap1_lr.float(), fmap2_lr.float(), fmap1_lm.float(), fmap2_lm.float()
            
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning 
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        if hasattr(self.args, 'buggy_tri_corr') and self.args.buggy_tri_corr:
            corr_cls = TriCorrBlock1D_Buggy
        else:
            corr_cls = TriCorrBlock1D
        corr_fn = corr_cls(
            fmap1_lr, fmap2_lr, fmap1_lm, fmap2_lm, radius=self.args.corr_radius,
            num_levels=self.args.corr_levels, middle_rate=self.args.corr_middle_rate if corr_middle_rate is None else corr_middle_rate
        )

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1, coords0) # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions


class TriRAFTStereoWithDAV2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert not self.args.shared_backbone
        
        context_dims = args.hidden_dims
        from .extractor import Dav2ContextEncoder
        self.cnet = Dav2ContextEncoder(args, output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        if not self.args.shared_fnet:
            self.fnet_lr = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)
            self.fnet_lm = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)

        print(f"TriRAFTStereoWithDAV2: will use buggy tri-corr class: {hasattr(self.args, 'buggy_tri_corr') and self.args.buggy_tri_corr}")

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)


    def forward(self, image_l, image_r, image_m, iters=12, flow_init=None, test_mode=False, corr_middle_rate=None):
        """ Estimate optical flow between pair of frames """

        image_l = (2 * (image_l / 255.0) - 1.0).contiguous()
        image_r = (2 * (image_r / 255.0) - 1.0).contiguous()
        image_m = (2 * (image_m / 255.0) - 1.0).contiguous()

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet_list = self.cnet(image_l, num_layers=self.args.n_gru_layers)
            if not self.args.shared_fnet:
                fmap1_lr, fmap2_lr = self.fnet_lr([image_l, image_r])
                fmap1_lm, fmap2_lm = self.fnet_lm([image_l, image_m])
            else:
                fmap1_lr, fmap2_lr, fmap2_lm = self.fnet([image_l, image_r, image_m])
                fmap1_lm = fmap1_lr
            fmap1_lr, fmap2_lr, fmap1_lm, fmap2_lm = fmap1_lr.float(), fmap2_lr.float(), fmap1_lm.float(), fmap2_lm.float()
            
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning 
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        if hasattr(self.args, 'buggy_tri_corr') and self.args.buggy_tri_corr:
            corr_cls = TriCorrBlock1D_Buggy
        else:
            corr_cls = TriCorrBlock1D
        corr_fn = corr_cls(
            fmap1_lr, fmap2_lr, fmap1_lm, fmap2_lm, radius=self.args.corr_radius,
            num_levels=self.args.corr_levels, middle_rate=self.args.corr_middle_rate if corr_middle_rate is None else corr_middle_rate
        )

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1, coords0) # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions