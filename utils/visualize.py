import os
import re
import shutil
import numpy as np
import cv2
try:
    import imageio.v2 as imageio
except:
    import imageio
from matplotlib import colormaps
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import json
import torch
import gc

import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from io import BytesIO


from .common import copy_file, FileIOHelper
from .transforms import coord_pixel2camera
from .ply import export_ply

########################################################
# This class is modified from 
# https://github.com/bennyguo/instant-nsr-pl/blob/main/utils/mixins.py
class SaveHanlder():
    def __init__(self, save_dir):
        self.io_helper = FileIOHelper()
        self.save_dir = save_dir

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir

    def do_save_image(self, fullpath, img):
        pathsplit = os.path.splitext(fullpath)
        if pathsplit[1] == '':
            fmt = 'jpg'
            fullpath = fullpath + ".jpg"
        else:
            fmt = pathsplit[1]
        with self.io_helper.open(fullpath, 'wb') as f:
            imageio.imwrite(f, img, fmt)
    
    def convert_data(self, data):
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, list):
            return [self.convert_data(d) for d in data]
        elif isinstance(data, dict):
            return {k: self.convert_data(v) for k, v in data.items()}
        else:
            raise TypeError('Data must be in type numpy.ndarray, torch.Tensor, list or dict, getting', type(data))
    
    def get_save_path(self, filename):
        save_path = os.path.join(self.save_dir, filename)
        self.io_helper.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path
    
    DEFAULT_RGB_KWARGS = {'data_format': 'CHW', 'data_range': (0, 1)}
    DEFAULT_UV_KWARGS = {'data_format': 'CHW', 'data_range': (0, 1), 'cmap': 'checkerboard'}
    DEFAULT_GRAYSCALE_KWARGS = {'data_range': None, 'cmap': 'jet'}

    def get_rgb_image_(self, img, data_format, data_range):
        img = self.convert_data(img)
        assert data_format in ['CHW', 'HWC']
        if data_format == 'CHW':
            img = img.transpose(1, 2, 0)
        img = img.clip(min=data_range[0], max=data_range[1])
        img = (((img - data_range[0]) / (data_range[1] - data_range[0])).clip(0,1) * 255.).astype(np.uint8)
        imgs = [img[...,start:start+3] for start in range(0, img.shape[-1], 3)]
        for i in range(len(imgs)):
            img_ = imgs[i]
            if img_.shape[-1] == 1:
                imgs[i] = np.repeat(img_, 3, axis=-1)
            elif img_.shape[-1] == 2:
                imgs[i] = np.concatenate([img_, np.zeros((img_.shape[0], img_.shape[1], 3 - img_.shape[2]), dtype=img_.dtype)], axis=-1)
        # imgs = [img_ if img_.shape[-1] == 3 else np.concatenate([img_, np.zeros((img_.shape[0], img_.shape[1], 3 - img_.shape[2]), dtype=img_.dtype)], axis=-1) for img_ in imgs]
        img = np.concatenate(imgs, axis=1)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    def save_rgb_image(self, filename, img, data_format=DEFAULT_RGB_KWARGS['data_format'], data_range=DEFAULT_RGB_KWARGS['data_range']):
        img = self.get_rgb_image_(img, data_format, data_range)
        self.do_save_image(self.get_save_path(filename), img)
    
    def get_uv_image_(self, img, data_format, data_range, cmap):
        img = self.convert_data(img)
        assert data_format in ['CHW', 'HWC']
        if data_format == 'CHW':
            img = img.transpose(1, 2, 0)
        img = img.clip(min=data_range[0], max=data_range[1])
        img = (img - data_range[0]) / (data_range[1] - data_range[0])
        assert cmap in ['checkerboard', 'color']
        if cmap == 'checkerboard':
            n_grid = 64
            mask = (img * n_grid).astype(int)
            mask = (mask[...,0] + mask[...,1]) % 2 == 0
            img = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            img[mask] = np.array([255, 0, 255], dtype=np.uint8)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif cmap == 'color':
            img_ = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img_[..., 0] = (img[..., 0] * 255).astype(np.uint8)
            img_[..., 1] = (img[..., 1] * 255).astype(np.uint8)
            # img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
            img = img_
        return img
    
    def save_uv_image(self, filename, img, data_format=DEFAULT_UV_KWARGS['data_format'], data_range=DEFAULT_UV_KWARGS['data_range'], cmap=DEFAULT_UV_KWARGS['cmap']):
        img = self.get_uv_image_(img, data_format, data_range, cmap)
        self.do_save_image(self.get_save_path(filename), img)

    def get_grayscale_image_(self, img, data_range, cmap):
        img = self.convert_data(img).squeeze()
        img = np.nan_to_num(img)
        if data_range is None:
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = img.clip(data_range[0], data_range[1])
            img = (img - data_range[0]) / (data_range[1] - data_range[0])
        assert cmap in [None, 'jet', 'magma']
        if cmap == None:
            img = (img * 255.).astype(np.uint8)
            img = np.repeat(img[...,None], 3, axis=2)
        elif cmap == 'jet':
            img = (img * 255.).astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        elif cmap == 'magma':
            img = 1. - img
            base = cm.get_cmap('magma')
            num_bins = 256
            colormap = LinearSegmentedColormap.from_list(
                f"{base.name}{num_bins}",
                base(np.linspace(0, 1, num_bins)),
                num_bins
            )(np.linspace(0, 1, num_bins))[:,:3]
            a = np.floor(img * 255.)
            b = (a + 1).clip(max=255.)
            f = img * 255. - a
            a = a.astype(np.uint16).clip(0, 255)
            b = b.astype(np.uint16).clip(0, 255)
            img = colormap[a] + (colormap[b] - colormap[a]) * f[...,None]
            img = (img * 255.).astype(np.uint8)
        return img

    def save_grayscale_image(self, filename, img, data_range=DEFAULT_GRAYSCALE_KWARGS['data_range'], cmap=DEFAULT_GRAYSCALE_KWARGS['cmap']):
        img = self.get_grayscale_image_(img, data_range, cmap)
        self.do_save_image(self.get_save_path(filename), img)

    def get_image_grid_(self, imgs):
        if isinstance(imgs[0], list):
            return np.concatenate([self.get_image_grid_(row) for row in imgs], axis=0)
        cols = []
        for col in imgs:
            assert col['type'] in ['rgb', 'uv', 'grayscale']
            if col['type'] == 'rgb':
                rgb_kwargs = self.DEFAULT_RGB_KWARGS.copy()
                rgb_kwargs.update(col.get('kwargs', {}))
                cols.append(self.get_rgb_image_(col['img'], **rgb_kwargs))
            elif col['type'] == 'uv':
                uv_kwargs = self.DEFAULT_UV_KWARGS.copy()
                uv_kwargs.update(col.get('kwargs', {}))
                cols.append(self.get_uv_image_(col['img'], **uv_kwargs))
            elif col['type'] == 'grayscale':
                grayscale_kwargs = self.DEFAULT_GRAYSCALE_KWARGS.copy()
                grayscale_kwargs.update(col.get('kwargs', {}))
                cols.append(self.get_grayscale_image_(col['img'], **grayscale_kwargs))
        return np.concatenate(cols, axis=1)
    
    def save_image_grid(self, filename, imgs):
        img = self.get_image_grid_(imgs)
        self.do_save_image(self.get_save_path(filename), img)
    
    def save_image(self, filename, img):
        img = self.convert_data(img)
        assert img.dtype == np.uint8
        # if img.shape[-1] == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # elif img.shape[-1] == 4:
        #     img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        self.do_save_image(self.get_save_path(filename), img)
    
    def save_cubemap(self, filename, img, data_range=(0, 1)):
        img = self.convert_data(img)
        assert img.ndim == 4 and img.shape[0] == 6 and img.shape[1] == img.shape[2]

        imgs_full = []
        for start in range(0, img.shape[-1], 3):
            img_ = img[...,start:start+3]
            img_ = np.stack([self.get_rgb_image_(img_[i], 'HWC', data_range) for i in range(img_.shape[0])], axis=0)
            size = img_.shape[1]
            placeholder = np.zeros((size, size, 3), dtype=np.float32)
            img_full = np.concatenate([
                np.concatenate([placeholder, img_[2], placeholder, placeholder], axis=1),
                np.concatenate([img_[1], img_[4], img_[0], img_[5]], axis=1),
                np.concatenate([placeholder, img_[3], placeholder, placeholder], axis=1)
            ], axis=0)
            # img_full = cv2.cvtColor(img_full, cv2.COLOR_RGB2BGR)
            imgs_full.append(img_full)
        
        imgs_full = np.concatenate(imgs_full, axis=1)
        self.do_save_image(self.get_save_path(filename), imgs_full)

    def save_data(self, filename, data):
        data = self.convert_data(data)
        if isinstance(data, dict):
            if not filename.endswith('.npz'):
                filename += '.npz'
            np.savez(self.get_save_path(filename), **data)
        else:
            if not filename.endswith('.npy'):
                filename += '.npy'
            np.save(self.get_save_path(filename), data)
        
    def save_state_dict(self, filename, data):
        torch.save(data, self.get_save_path(filename))
    
    def save_img_sequence(self, filename, img_dir, matcher, save_format='gif', fps=30):
        assert save_format in ['gif', 'mp4']
        if not filename.endswith(save_format):
            filename += f".{save_format}"
        matcher = re.compile(matcher)
        img_dir = os.path.join(self.save_dir, img_dir)
        imgs = []
        for f in os.listdir(img_dir):
            if matcher.search(f):
                imgs.append(f)
        imgs = sorted(imgs, key=lambda f: int(matcher.search(f).groups()[0]))
        imgs = [cv2.imread(os.path.join(img_dir, f)) for f in imgs]
        
        if save_format == 'gif':
            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
            imageio.mimsave(self.get_save_path(filename), imgs, fps=fps, palettesize=256)
        elif save_format == 'mp4':
            H, W, _ = imgs[0].shape
            writer = cv2.VideoWriter(self.get_save_path(filename), cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H), True)
            for img in imgs:
                writer.write(img)
            writer.release()
    
    def save_file(self, filename, src_path):
        copy_file(src_path, self.get_save_path(filename))
    
    def save_json(self, filename, payload):
        with open(self.get_save_path(filename), 'w') as f:
            f.write(json.dumps(payload))
###############################################################

def save_images(*batched_imgs:torch.Tensor, save_path:str):
    '''
    batched_imgs: images to dump. Different elements in the sequence will be exported in parallel  
    each element in batched_imgs must be in shape of (B, C, H, W) or (B, H, W, C)  
    The pixel value should be within range (0, 1)  
    '''
    iohelper = FileIOHelper()

    def save_unbatched_imgs(*unbatched_imgs_np:np.ndarray, save_path:str):
        # each element in unbatched_imgs_np should be in shape (H, W, C) with dtype=uint8
        h, w, c = unbatched_imgs_np[0].shape
        num = len(unbatched_imgs_np)
        concat_imgs = np.concatenate(unbatched_imgs_np, axis=1)
        if concat_imgs.shape[-1] == 1:
            concat_imgs = np.repeat(concat_imgs, 3, axis=-1)
        if os.path.splitext(save_path)[1] == "":
            fmt = "png"
            save_path += ".png"
        else:
            fmt = os.path.splitext(save_path)[1][1:]  # remove "."
        with iohelper.open(save_path, 'wb') as f:
            imageio.imwrite(f, concat_imgs, fmt)

    if batched_imgs[0].shape[1] == 3 or batched_imgs[0].shape[1] == 1:  # B,C,H,W  
        channel_last = False
        b,c,h,w = batched_imgs[0].shape
    else:
        channel_last = True
        b,h,w,c = batched_imgs[0].shape
    batched_imgs_np = []
    for img in batched_imgs:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        assert isinstance(img, np.ndarray)
        if not channel_last:
            img = img.transpose(0,2,3,1)
        assert img.shape[-3:-1] == (h,w)
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        batched_imgs_np.append(img)  # all uint8
    
    if b == 1:
        unbatched_imgs_np = [img.squeeze(0) for img in batched_imgs_np]
        save_unbatched_imgs(*unbatched_imgs_np, save_path=save_path)
    else:
        for i in range(b):
            unbatched_imgs_np = [img[i] for img in batched_imgs_np]
            path_split = os.path.splitext(save_path)
            path = path_split[0] + f"_{i}" + path_split[1]
            save_unbatched_imgs(*unbatched_imgs_np, save_path=path)


def material_type_to_rgb(material_type:torch.Tensor):
    '''r: Diffuse, g: Specular, b: Transmission'''
    from deepsl_data.dataloader.material_mask import material_masks
    matmsks = material_masks(material_type.squeeze(), 1)  # (..., H, W)
    stacked_masks = torch.stack(
        (matmsks['Diffuse'], matmsks['Specular'], matmsks['Transmission']), dim=-3
    ).to(torch.float32) # (...,3,H,W)
    return stacked_masks


def vis_batch(sample:dict, l_pred_depth, r_pred_depth, step, expdir:str, max_depth:int, rank:int=None, loss_range = [-0.1, 0.1], 
              fname_list:str=None, share_depth_range:bool=False, r_pd_depth_special=False, r_special_range=None):
    '''
    r_pred_depth可以是其他东西，此时指定r_pd_depth_special为True
    '''
    handler = SaveHanlder(os.path.join(expdir, 'samples'))
    b = l_pred_depth.shape[0]

    if 'L_MaterialType' in sample:
        l_material_mask_rgb = material_type_to_rgb(sample['L_MaterialType'])
    if 'R_MaterialType' in sample:
        r_material_mask_rgb = material_type_to_rgb(sample['R_MaterialType'])

    for i in range(b):
        l_gt_depth = sample['L_Depth'][i].squeeze()
        if 'R_Depth' in sample:
            r_gt_depth = sample['R_Depth'][i].squeeze()
        else:
            r_gt_depth = None
            assert r_pred_depth is None, "no right gt depth, right predicted depth must be None."
        l_pd_depth = l_pred_depth[i].squeeze()
        l_mask = (l_gt_depth > 0) & (l_gt_depth < max_depth)
        l_masked_gt_depth = l_gt_depth[l_mask]
        l_gt_min, l_gt_max = l_masked_gt_depth.min(), l_masked_gt_depth.max()
        gt_depth_range = [l_gt_min.item(), l_gt_max.item()]
        l_masked_pred_depth = l_pd_depth[l_mask]
        l_pd_min, l_pd_max = l_masked_pred_depth.min(), l_masked_pred_depth.max()
        pd_depth_range = [l_pd_min.item(), l_pd_max.item()] if not share_depth_range else gt_depth_range
        
        if r_pred_depth is not None:
            r_pd_depth = r_pred_depth[i].squeeze()
            r_mask = (r_gt_depth > 0) & (r_gt_depth < max_depth) if not r_pd_depth_special else torch.ones_like(r_pd_depth)

        grids = [
            {'type': 'rgb', 'img': sample['L_Image'][i]},
            {'type': 'grayscale', 'img': l_gt_depth * l_mask, 'kwargs':{'cmap':'jet','data_range':gt_depth_range}},
            {'type': 'grayscale', 'img': l_pd_depth * l_mask, 'kwargs':{'cmap':'jet','data_range':pd_depth_range}},
            {'type': 'grayscale', 'img': (l_pd_depth - l_gt_depth) * l_mask, 'kwargs':{'cmap':'jet','data_range':loss_range}},
        ]
        if 'R_Image' in sample:
            grids.insert(1, {'type': 'rgb', 'img': sample['R_Image'][i]})
        if r_pred_depth is not None:  # There must be 'R_Image' in sample.
            grids.insert(3, {'type': 'grayscale', 'img': r_gt_depth * r_mask, 'kwargs':{'cmap':'jet','data_range':gt_depth_range}})
            if not r_pd_depth_special:
                r_pd_range = pd_depth_range
            else:
                if r_special_range is not None:
                    r_pd_range = r_special_range
                else:
                    r_pd_range = [r_pd_depth.min().item(), r_pd_depth.max().item()]
            grids.insert(5, {'type': 'grayscale', 'img': r_pd_depth * r_mask, 'kwargs':{'cmap':'jet','data_range':r_pd_range}})
            
            if not r_pd_depth_special:
                grids.append({'type': 'grayscale', 'img': (r_pd_depth - r_gt_depth) * r_mask, 'kwargs':{'cmap':'jet','data_range':loss_range}})

        if 'L_MaterialType' in sample:
            grids.append({'type':'rgb','img':l_material_mask_rgb[i]})
        if 'R_MaterialType' in sample and r_pred_depth is not None:
            grids.append({'type':'rgb','img':r_material_mask_rgb[i]})

        if fname_list is None:
            if rank is None:
                fname = f"{step:07d}_{i}_{sample['key'][i]}_{sample['pattern_name'][i]}_gt-{gt_depth_range[0]:.2f}~{gt_depth_range[1]:.2f}_pd-{pd_depth_range[0]:.2f}~{pd_depth_range[1]:.2f}.png"
            else:
                fname = f"{step:07d}_{i}_rank{rank}_{sample['key'][i]}_{sample['pattern_name'][i]}_gt-{gt_depth_range[0]:.2f}~{gt_depth_range[1]:.2f}_pd-{pd_depth_range[0]:.2f}~{pd_depth_range[1]:.2f}.png"
        else:
            fname = fname_list[i]
        fname = fname.replace("/", "_")
        handler.save_image_grid(
            fname, grids
        )
    del handler
    gc.collect()


# from D3RoMa, https://github.com/songlin/d3roma/blob/main/utils/utils.py
def viz_cropped_pointcloud(K, rgb, depth, show=False, fname=None, max_depth=10, remove_edge = False, rtol=0.04):
    """ visualize point cloud which is random cropped from the original image
        K: 3x3
        rgb: HxWx3, range: (0, 255)  
        depth: HXW (meter)  
        only support local storage.  
    """
    import open3d as o3d
    K = K.detach().cpu().numpy() if isinstance(K, torch.Tensor) else K
    rgb = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else rgb
    depth = depth.detach().cpu().numpy() if isinstance(depth, torch.Tensor) else depth
    mask = None
    if remove_edge:
        import utils3d
        mask = ~utils3d.numpy.depth_edge(depth, rtol=rtol)
        depth = depth * mask.astype(np.float32)
    

    assert type(rgb) == np.ndarray and type(depth) == np.ndarray, "rgb and depth must be numpy array"
    assert rgb.shape[:2] == depth.shape, f"rgb ({rgb.shape}) & depth ({depth.shape}) do not match"

    H, W = rgb.shape[:2]
    depth_raw = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
    color_raw = o3d.geometry.Image(np.ascontiguousarray(rgb).astype(np.uint8))
    rgbd_raw = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 
        depth_scale=1., depth_trunc=max_depth, convert_rgb_to_intensity=False)
    K_vec = [K[0,0], K[1,1], K[0,2], K[1,2]]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, *K_vec)
    pcd_rgbd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_raw, intrinsic)
    if show:
        o3d.visualization.draw_geometries([pcd_rgbd])
    if fname is not None:
        o3d.io.write_point_cloud(fname, pcd_rgbd)
        if mask is not None:
            mask_img = np.uint8(mask) * 255
            imgname = fname + ".mask.png"
            cv2.imwrite(imgname, mask_img)
    return pcd_rgbd


def preview_depth_pointcloud(pointcloud, outpath):
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pointcloud)
    opt = vis.get_render_option()
    opt.point_size = 5.0

    view_control = vis.get_view_control()

    camera_position = [0, 0.2, 0.2]
    points = np.asarray(pointcloud.points).reshape(-1, 3)
    center = np.mean(points, axis=0)
    look_at_point = center.tolist()
    up_direction = [0, 0, 1]

    param = o3d.camera.PinholeCameraParameters()
    param.lookat = look_at_point
    param.front = np.array(look_at_point) - np.array(camera_position)
    param.up = up_direction
    param.zoom = 0.5

    view_control.convert_from_pinhole_camera_parameters(param)

    # 4. 渲染并保存图片
    vis.capture_screen_image(outpath)
    vis.destroy_window()

@torch.no_grad()
def vis_pointcloud(K, rgb, depth, fname:str=None, max_depth=10):
    '''K: 3x3  
    rgb: HxWx3, range: (0, 255), can be None  
    depth: HXW (meter)      
    '''
    pointcloud = coord_pixel2camera(
        torch.from_numpy(depth).clamp_(0, max_depth) if isinstance(depth, np.ndarray) else depth.clamp_(0, max_depth), 
        torch.from_numpy(K) if isinstance(K, np.ndarray) else K
    ).detach().cpu().numpy()
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    export_ply(fname, pointcloud, rgb)

@torch.no_grad()
def vis_comparable_pointcloud(K, gt_depth, pd_depth, fname:str=None, rgb = None, max_depth=10):
    '''
    *_depth: (H,W) (meter)  
    rgb: (H,W,3) within range (0, 255.)  
    '''
    K = torch.from_numpy(K) if isinstance(K, np.ndarray) else K
    K = K.squeeze_()
    gt_depth = torch.from_numpy(gt_depth) if isinstance(gt_depth, np.ndarray) else gt_depth
    gt_depth = gt_depth.squeeze_()
    gt_depth = gt_depth.clamp_(0, max_depth)
    pd_depth = torch.from_numpy(pd_depth) if isinstance(pd_depth, np.ndarray) else pd_depth
    pd_depth = pd_depth.squeeze_()
    pd_depth = pd_depth.clamp_(0, max_depth)
    gt_point_cloud = coord_pixel2camera(gt_depth, K).detach().cpu().numpy()
    pd_point_cloud = coord_pixel2camera(pd_depth, K).detach().cpu().numpy()
    K = K.detach().cpu().numpy()
    if rgb is None:
        gt_color = np.tile(np.array([0, 0, 255], dtype=np.uint8)[None,None,:], (*gt_point_cloud.shape[:2], 1))  # b
    else:
        gt_color = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else rgb
        gt_color = gt_color.reshape(*gt_point_cloud.shape)
    pd_color = np.tile(np.array([255, 0, 0], dtype=np.uint8)[None,None,:], (*pd_point_cloud.shape[:2], 1))  # r
    point_cloud = np.concatenate([gt_point_cloud, pd_point_cloud], axis=0)
    colors = np.concatenate([gt_color, pd_color], axis=0)
    export_ply(fname, point_cloud, colors)

def apply_colormap(arr, vmin=None, vmax=None, errormap=False, to_bgr=True):
    if not hasattr(apply_colormap, 'depth_map'):
        setattr(apply_colormap, 'depth_map', colormaps.get("Spectral_r"))
        setattr(apply_colormap, 'error_map', colormaps.get("bwr"))
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if vmin is None:
        vmin = arr.min()
    if vmax is None:
        vmax = arr.max()
    arr = (arr - vmin) / (vmax - vmin)
    cmap = apply_colormap.error_map if errormap else apply_colormap.depth_map
    color = (cmap(arr)[:,:,:3] * 255).astype(np.uint8)
    if to_bgr:
        color = color[:,:,::-1]
    return color


def generate_colorbar(vmin, vmax, cmap_name, output_path=None, size=None):
    """生成独立色条并保存为图片
    
    参数：
        vmin (float): 最小值
        vmax (float): 最大值
        cmap_name (str): matplotlib 支持的色谱名称（如 'viridis', 'jet' 等）
        orientation (str): 方向（'vertical' 或 'horizontal'）
        output_path (str): 输出图片路径
    """
    # 配置图形参数
    dpi = 100
    orientation = 'vertical'
    figsize = (1, 9)  # 宽，高（英寸）
    ax_rect = [0.3, 0.05, 0.2, 0.9]  # 左，底，宽，高

    # 创建图形和坐标轴
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.rcParams.update({'font.size': 12})  # 统一字体大小
    ax = fig.add_axes(ax_rect)

    # 创建色条
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation=orientation, ticks=[vmin, vmax])

    # 设置刻度格式（自动适应整数/浮点数）
    if all(isinstance(x, int) for x in [vmin, vmax]):
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    else:
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # 使用内存缓冲区保存图像
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    # 转换为 OpenCV 格式
    buf.seek(0)
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

    if size is not None:
        size = (min(size), max(size))
        img = cv2.resize(img, size)
    # 保存最终结果
    if output_path is not None:
        cv2.imwrite(output_path, img)
    return img


if __name__ == '__main__':
    from deepsl_data.dataloader.file_fetcher import LocalFileFetcher
    data_root = 'data/data'

    sid = 2562
    vid = 3
    pattern = 'D415'

    file_fetcher = LocalFileFetcher('train', data_root, False, True)
    key = f"{sid:05d}/{vid}"
    sample = file_fetcher.fetch(key, True, pattern, False, False)
    color = None
    depth = None
    K = None
    for k, v in sample.items():
        if 'Depth' in k and 'L_' in k:
            depth = v
        if 'Image' in k and 'L_' in k:
            color = v
        if 'intri' in k and 'L_' in k:
            K = v
    if color.ndim == 2:
        color = color[..., None]
    if color.shape[-1] == 1:
        color = color.repeat(3, axis=-1) * 255
    else:
        color = color * 255
    # viz_cropped_pointcloud(K, color, depth.squeeze(), False, f"{sid:05d}_{vid:03d}_L_Depth.ply")
    vis_pointcloud(K, color, depth, f"{sid:05d}_{vid:03d}_L_Depth.ply")