import torch
import torch.nn as nn
# DEBUG
from utils.dist import print_ddp

_min_depth = 0.1
_max_depth = 10

class MaskedDepthLoss(nn.Module):
    def __init__(self, conf=True):
        super().__init__()
        self.conf = conf

    def forward(self, pred_depth:torch.Tensor, gt_depth:torch.Tensor, mask:torch.Tensor=None, confidence:torch.Tensor=None):
        '''
        pred_depth: (B, 1, H, W); gt_depth: (B, 1, H, W); mask: (B, 1, H, W)
        '''
        hp, wp = pred_depth.shape[-2:]
        hg, wg = gt_depth.shape[-2:]
        if (hp, wp) != (hg, wg):
            gt_depth = nn.functional.interpolate(gt_depth, (hp, wp), mode='nearest')
            if mask is not None:
                mask = nn.functional.interpolate(mask, (hp, wp), mode='nearest')
        if mask is not None:
            mask_bool = (mask != 0) & ((gt_depth >= _min_depth) & (gt_depth <= _max_depth)) & (~gt_depth.isnan())
        else:
            mask_bool = (gt_depth >= _min_depth) & (gt_depth <= _max_depth) & (~gt_depth.isnan())  # Exclude regions that are infinitely far away
        if confidence is None or not self.conf:
            confidence = 1
        gt_depth = gt_depth.clamp(_min_depth, _max_depth)  # remove inf
        loss, mask_bool = self.compute_loss(pred_depth, gt_depth, mask_bool, confidence)
        mask = mask_bool.to(torch.float32)
        num_valid_pixels = torch.sum(mask)
        loss = torch.nan_to_num(loss, 1e-6)
        return torch.sum(loss * mask) / num_valid_pixels

        # if masked_loss.numel() > 0:
        #     return torch.mean(masked_loss)
        # else:
        #     return torch.tensor(0., requires_grad=True, device=pred_depth.device, dtype=pred_depth.dtype)

    def compute_loss(self, pred:torch.Tensor, gt:torch.Tensor, mask:torch.Tensor, confidence:torch.Tensor):
        raise NotImplementedError
    
class DepthL1Loss(MaskedDepthLoss):  # also MAE.
    def __init__(self, conf:bool=True):
        super().__init__(conf)
    def compute_loss(self, pred, gt, mask, confidence = 1):
        return torch.abs(pred - gt) * confidence, mask
    
class DepthL2Loss(MaskedDepthLoss):
    def __init__(self, conf=True):
        super().__init__(conf)
    def compute_loss(self, pred, gt, mask, confidence = 1):
        return torch.pow(pred-gt, 2) * confidence, mask

class DepthSmoothL1Loss(MaskedDepthLoss):
    def __init__(self, beta:float=0.1, conf=True):
        super().__init__(conf)
        self.beta = beta
    def compute_loss(self, pred, gt, mask, confidence = 1):
        abs_diff = torch.abs(pred - gt)
        return confidence * torch.where(abs_diff < self.beta, 0.5 * abs_diff**2 / self.beta, abs_diff - 0.5 * self.beta),\
               mask
    

class DepthGradientLoss(MaskedDepthLoss):
    def __init__(self, gradient_operator:str = 'naive', conf=True):
        super().__init__(conf)
        self.gradient_operator = gradient_operator
    def compute_loss(self, pred, gt, mask, confidence = 1):
        diff = pred - gt
        if self.gradient_operator.startswith('naive'):
            grad_x = torch.abs(diff[..., :-1] - diff[..., 1:]) # B,C,H,W-1
            grad_y = torch.abs(diff[..., :-1, :] - diff[..., 1:, :]) # B,C,H-1,W
            # Both operands should have meaning
            mask_x = mask[..., :-1] & mask[..., 1:]  # B,C,H,W-1
            mask_y = mask[..., :-1, :] & mask[..., 1:, :]  # B,C,H-1,W
            mask = mask_x[..., :-1, :] & mask_y[..., :, :-1]
            if isinstance(confidence, torch.Tensor) and confidence.ndim > 0:
                confidence = confidence[...,:-1,:-1]
            if self.gradient_operator.endswith('diag'):
                grad = torch.sqrt(grad_x[...,:-1,:]**2 + grad_y[...,:,:-1]**2)
            else:
                grad = grad_x[...,:-1,:] + grad_y[...,:,:-1]
            return grad * confidence, mask
        else:
            raise NotImplementedError(f"Unimplemented gradient operator: {self.gradient_operator}")

class ConfRegLogLoss(MaskedDepthLoss):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps
    def compute_loss(self, pred, gt, mask, confidence = 1):
        return -torch.log(confidence + self.eps), mask

class ConfRegSquareLoss(MaskedDepthLoss):
    def __init__(self):
        super().__init__()
    def compute_loss(self, pred, gt, mask, confidence = 1):
        return (1-confidence)**2, mask

class ConfRegLinearLoss(MaskedDepthLoss):
    def __init__(self):
        super().__init__()
    def compute_loss(self, pred, gt, mask, confidence = 1):
        return torch.abs(1-confidence), mask


class DepthSiLogLoss(MaskedDepthLoss):
    def __init__(self, lambd = 0.5, conf=True):
        super().__init__(conf)
        self.lambd = lambd
    def compute_loss(self, pred, gt, mask, confidence = 1):
        diff_log = torch.log(gt) - torch.log(pred)
        diff_log = diff_log.nan_to_num(1e-6)
        loss = torch.pow(diff_log, 2) - self.lambd * torch.pow(diff_log.mean(), 2)
        return loss * confidence, mask
    def forward(self, pred_depth, gt_depth, mask = None, confidence=None):
        return torch.sqrt(super().forward(pred_depth, gt_depth, mask, confidence))

# Metrics...
class DepthRmseMetric(DepthL2Loss):   # RMSE
    def __init__(self):
        super().__init__()
    def forward(self, pred_depth, gt_depth, mask, confidence=1):
        confidence = 1  # metric, no confidence
        loss = super().forward(pred_depth, gt_depth, mask, confidence)
        return torch.sqrt(loss)
    
class DepthRelMetric(MaskedDepthLoss):  # REL, or AbsRel
    def __init__(self):
        super().__init__()
    def compute_loss(self, pred, gt, mask, confidence=1):
        diff = torch.abs(gt - pred)
        return diff / (gt + 1e-8), mask
    
class DepthDeltaThresholdMetric(MaskedDepthLoss):  # delta_1.25, delta_1.1, delta_1.05
    def __init__(self, threshold:float):
        super().__init__()
        self.thresh = threshold
        # self.sqrt_thresh = torch.sqrt(torch.tensor(threshold, dtype=torch.float32)).item()
    def compute_loss(self, pred, gt, mask, confidence=1):
        pred_div_gt = pred / (gt + 1e-8)
        gt_div_pred = gt / (pred + 1e-8)
        val = torch.where(pred_div_gt > gt_div_pred, pred_div_gt, gt_div_pred)
        return (val < self.thresh).to(torch.float32), mask


class ComposedLoss(nn.Module):
    def __init__(self, **loss_modules):
        '''
        loss_modules: {
            'loss_name': (loss_module:nn.Module, lambda:float)
        }
        '''
        super().__init__()
        self.losses = {k: v[0] for k, v in loss_modules.items()}
        self.lambdas = {k: v[1] for k, v in loss_modules.items()}
    
    def forward(self, pred:torch.Tensor, gt:torch.Tensor, mask:torch.Tensor=None, confidence=None):
        ret = {k: v(pred, gt, mask, confidence) for k, v in self.losses.items()}
        total = 0
        for k, v in self.lambdas.items():
            total = total + v * ret[k]
        ret['total_loss'] = total
        return ret
    

def parse_losses(**loss_cfgs):
    '''
    loss_cfgs: {
        'loss_name': {
            'lambda': float, 'type': str, 'kwargs':str | None
        }
    }
    '''
    assert len(loss_cfgs) > 0, "At least one loss function is needed."
    loss_modules = {}
    for loss_name, cfg in loss_cfgs.items():
        if loss_name == 'min_depth':
            import numbers
            assert isinstance(cfg, numbers.Number)
            global _min_depth
            _min_depth = cfg
            continue
        if loss_name == 'max_depth':
            import numbers
            assert isinstance(cfg, numbers.Number)
            global _max_depth
            _max_depth = cfg
            continue

        coef = cfg['lambda']
        ty = cfg['type']
        kwargs = cfg.get('kwargs', {})
        kwargs = {} if kwargs is None else kwargs
        loss_modules[loss_name] = (eval(ty)(**kwargs), coef)
    return ComposedLoss(**loss_modules)