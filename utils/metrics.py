import torch
from functools import partial
from torch.nn import Module as torch_module

from deepsl_data.dataloader.material_mask import material_masks

def get_valid_mask(gt_depth, valid_mask):
    basic_mask = (~gt_depth.isnan()) & (gt_depth > 1e-1) & (gt_depth < 1000)
    if valid_mask is None:
        mask = basic_mask  # 存在一些距离为无限的意外像素.
    else:
        mask = (valid_mask>0) & basic_mask
    return mask

def validate_input(pred:torch.Tensor, gt:torch.Tensor, valid_mask):
    valid_mask = get_valid_mask(gt, valid_mask)
    gt = gt.clamp_(1e-2, 10000)  # remove inf.
    gt.nan_to_num_(1e-6)
    pred.nan_to_num_(1e-6)
    return pred, gt, valid_mask

###########################
# modified from https://github.com/XiandaGuo/OpenStereo/blob/v2/stereo/evaluation/metric_per_image.py
def d1_metric(disp_pred, disp_gt, valid_mask):
    '''
    disp_pred, disp_gt, mask: (B,(1),H,W)  
    threshold is 0.05
    '''
    disp_pred, disp_gt, valid_mask = validate_input(disp_pred, disp_gt, valid_mask)
    E = torch.abs(disp_gt - disp_pred)
    err_mask = (E > 3) & (E / torch.abs(disp_gt) > 0.05)  # B,(1),H,W

    err_mask = err_mask & valid_mask
    num_errors = err_mask.sum(dim=[-2, -1]).squeeze()  # (B,) 
    num_valid_pixels = valid_mask.sum(dim=[-2, -1]).squeeze()  # (B,)

    d1_per_image = num_errors.float() / num_valid_pixels.float() * 100  # (B,)
    d1_per_image = torch.where(num_valid_pixels > 0, d1_per_image, torch.zeros_like(d1_per_image))

    return d1_per_image


def threshold_metric(disp_pred, disp_gt, valid_mask, threshold):
    '''
    disp_pred, disp_gt, mask: (B,(1),H,W)
    '''
    disp_pred, disp_gt, valid_mask = validate_input(disp_pred, disp_gt, valid_mask)
    err_mask = torch.abs(disp_gt - disp_pred) > threshold

    err_mask = err_mask & valid_mask  # 是valid才计入统计.
    num_errors = err_mask.sum(dim=[-2, -1]).squeeze()  # (B,)
    num_valid_pixels = valid_mask.sum(dim=[-2, -1]).squeeze()

    bad_per_image = num_errors.float() / num_valid_pixels.float() * 100 #(B, )
    bad_per_image = torch.where(num_valid_pixels > 0, bad_per_image, torch.zeros_like(bad_per_image))

    return bad_per_image


def epe_metric(disp_pred, disp_gt, valid_mask):
    '''
    disp_pred, disp_gt, mask: (B,(1),H,W)
    '''
    disp_pred, disp_gt, valid_mask = validate_input(disp_pred, disp_gt, valid_mask)
    E = torch.abs(disp_gt - disp_pred)
    E_masked = torch.where(valid_mask, E, torch.zeros_like(E))

    E_sum = E_masked.sum(dim=[-2, -1]).squeeze()
    num_valid_pixels = valid_mask.sum(dim=[-2, -1]).squeeze()  # B,
    epe_per_image = E_sum / num_valid_pixels
    epe_per_image = torch.where(num_valid_pixels > 0, epe_per_image, torch.zeros_like(epe_per_image))

    return epe_per_image

#####################################

def rmse_metric(d_pred, d_gt, valid_mask):
    '''
    Root Mean Squared Error  
    d_pred, d_gt, mask: (B,(1),H,W)
    '''
    d_pred, d_gt, valid_mask = validate_input(d_pred, d_gt, valid_mask)
    diff = d_pred - d_gt
    diff = diff * valid_mask  # 将无效像素的误差置为0
    rmse = torch.sqrt(torch.sum(diff ** 2, dim=(-2, -1)).squeeze() / valid_mask.sum(dim=(-2, -1)).squeeze())
    return rmse

def mae_metric(d_pred, d_gt, valid_mask):
    '''
    Mean Absolute Error  
    d_pred, d_gt, mask: (B,H,W)
    '''
    d_pred, d_gt, valid_mask = validate_input(d_pred, d_gt, valid_mask)
    diff = torch.abs(d_pred - d_gt)
    diff = diff * valid_mask
    mae = torch.sum(diff, dim=(-2, -1)).squeeze() / valid_mask.sum(dim=(-2, -1)).squeeze()
    return mae

def absrel_metric(d_pred, d_gt, valid_mask):
    '''
    Absolute Relative Error  
    d_pred, d_gt, mask: (B,H,W)
    '''
    d_pred, d_gt, valid_mask = validate_input(d_pred, d_gt, valid_mask)
    diff = torch.abs(d_pred - d_gt)
    diff = diff * valid_mask
    absrel = torch.sum(diff / d_gt, dim=(-2, -1)).squeeze() / valid_mask.sum(dim=(-2, -1)).squeeze()
    return absrel

def sqrel_metric(d_pred, d_gt, valid_mask):
    '''
    Squared Relative Error  
    d_pred, d_gt, mask: (B,H,W)
    '''
    d_pred, d_gt, valid_mask = validate_input(d_pred, d_gt, valid_mask)
    diff = d_pred - d_gt
    diff = diff * valid_mask
    sqrel = torch.sum((diff ** 2) / d_gt, dim=(-2, -1)).squeeze() / valid_mask.sum(dim=(-2, -1)).squeeze()
    return sqrel

def delta_threshold(d_pred, d_gt, valid_mask, thresh = 1.25):
    '''
    percentage of max(d_pred / d_gt, d_gt / d_pred) < thresh  
    when thresh = 1.25, it is delta_1
    '''
    d_pred, d_gt, valid_mask = validate_input(d_pred, d_gt, valid_mask)
    pred_div_gt = d_pred / (d_gt + 1e-6)
    gt_div_pred = d_gt / (d_pred + 1e-6)
    val = torch.where(pred_div_gt > gt_div_pred, pred_div_gt, gt_div_pred)
    delta = (val < thresh) * valid_mask #.to(torch.float32)
    percentage = torch.sum(delta, dim=(-2,-1)).squeeze() / valid_mask.sum(dim=(-2,-1)).squeeze()
    return percentage


metric_funcs = {
    'd1': d1_metric,
    'bad_1': partial(threshold_metric, threshold=1.),
    'bad_2': partial(threshold_metric, threshold=2.),
    'bad_3': partial(threshold_metric, threshold=3.),
    'thresh_1': partial(threshold_metric, threshold=1.),
    'thresh_2': partial(threshold_metric, threshold=2.),
    'thresh_3': partial(threshold_metric, threshold=3.),
    'epe': epe_metric,
    'rmse': rmse_metric,
    'mae': mae_metric,
    'absrel': absrel_metric,
    'sqrel': sqrel_metric,
    'delta_1': partial(delta_threshold, thresh = 1.25),
    'delta_05': partial(delta_threshold, thresh = 1.10),
    'delta_125': partial(delta_threshold, thresh=1.25),
    'delta_110': partial(delta_threshold, thresh=1.10),
    'delta_105': partial(delta_threshold, thresh=1.05),
}

def compute_metrics(d_pred, d_gt, valid_mask, per_image = False, *metrics):
    '''
    d_pred, d_gt, valid_mask: (B,(1),H,W)  
    per_image: compute metric for each instance in the batch, otherwise average all results  
    *metrics: name of metrics to be computed, can be any of the following:  
    d1, bad_1, bad_2, bad_3, thresh_1, thresh_2, thresh_3, epe, rmse, mae, absrel, sqrel, delta_1  
    '''
    d_pred = d_pred.squeeze(1)
    d_gt = d_gt.squeeze(1)
    if valid_mask is not None:
        valid_mask = valid_mask.squeeze(1)
    ret = {k: metric_funcs[k](d_pred, d_gt, valid_mask) for k in metrics}
    ret = {k: v.unsqueeze(0) if v.ndim==0 else v for k, v in ret.items()}
    if not per_image:
        ret = {k: torch.mean(v) for k, v in ret.items()}
    return ret


def compute_material_classfied_metrics(d_pred, d_gt, material_type, valid_mask, per_image = False, level=1, *metrics):
    d_pred = d_pred.squeeze(1)
    d_gt = d_gt.squeeze(1)
    material_type = material_type.squeeze(1)
    if valid_mask is not None:
        valid_mask = valid_mask.squeeze(1)
    matmsks = material_masks(material_type, level=level)
    mat_metrics = {}
    for material_name, msk in matmsks.items():
        if valid_mask is not None:
            msk = msk & valid_mask
        mat_metrics[material_name] = compute_metrics(d_pred, d_gt, msk, per_image, *metrics)
    return mat_metrics


def count_parameters(model:torch_module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model:torch_module, inputs:dict):
    from thop import profile
    inp = tuple(inputs.values())
    flops, params = profile(model, inputs)
    return flops


def zncc(a:torch.Tensor, b:torch.Tensor):
    '''
    在图片的同一行计算ZNCC.  
    a, b: (..., C, H, WA), (..., C, H, WB)  
    return: (..., H, WA, WB); out[..., h, wa, wb] = zncc(a[..., h, wa, :], b[..., h, wb, :])
    '''
    za = a - torch.mean(a, dim=-3, keepdim=True) + 1e-6  # (B,1,H,WA)
    zb = b - torch.mean(b, dim=-3, keepdim=True) + 1e-6  # (B,1,H,WB)
    l2_za = torch.sqrt(torch.sum(za ** 2, dim=-3, keepdim=True))
    l2_zb = torch.sqrt(torch.sum(zb ** 2, dim=-3, keepdim=True))
    return torch.einsum("...cha, ...chb -> ...hab", za / l2_za, zb / l2_zb)
    # return torch.einsum("...pk, ...qk -> ...pq", za / l2_za, zb / l2_zb)


def inner_product(a:torch.Tensor, b:torch.Tensor):
    '''
    在图片的同一行计算内积.  
    a, b: (..., C, H, WA), (..., C, H, WB)  
    return: (..., H, WA, WB); out[..., h, wa, wb] = dot(a[..., :, h, wa], b[..., :, h, wb])
    '''
    return torch.einsum("...cha, ...chb -> ...hab", a, b)