import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
import OpenEXR
import Imath

from math import pi
from scipy.spatial.transform import Rotation

from .utils_color import ColorConversionFunc


def compute_ideal_intrinsic(focal_length, reso_x, reso_y, sensor_x, sensor_y, reverse_y:bool = False):
    '''
    Compute an ideal intrinsic matrix  (camera coordinate -> pixel coordinate)
    use mm as the unit
    '''
    pixel_rho_x = reso_x / sensor_x
    pixel_rho_y = reso_y / sensor_y
    cx, cy = sensor_x / 2, sensor_y / 2
    return torch.tensor(
        [
            [pixel_rho_x * focal_length, 0, pixel_rho_x * cx],
            [0, pixel_rho_y * focal_length * (-1 if reverse_y else 1), pixel_rho_y * cy],
            [0, 0, 1]
        ], dtype=torch.float32
    )

def compute_blender_projector_intrinsic(reso_x, reso_y, scale_x, scale_y, reverse_y:bool = False):
    '''
    reso: the resolutionos the pattern used.
    scale: the scale value of of the second Mapping node of the spot light in blender

    当reverse_y == False, 所用的相机坐标系是一般的，x左y下、相机拍摄z正方向, 右手系.
    '''
    return torch.tensor(
        [
            [reso_x / scale_x, 0, 0.5 * reso_x],
            [0, reso_y / scale_y * (-1 if reverse_y else 1), 0.5 * reso_y],  # ??是否要加负号？
            [0, 0, 1]
        ], dtype=torch.float32
    )

def euler_to_rotation_matrix(rot_x, rot_y, rot_z, order = 'xyz', eps=1e-6, degree = True):
    '''
    compute rotation matrix from euler engles. rot_x, rot_y, rot_z are rotation angles in degree.  
    for blender's camera, the order should be 'xyz'
    '''
    # 定义欧拉角（单位是弧度）
    if degree:
        angles = np.array([rot_x * pi / 180, rot_y * pi / 180, rot_z * pi / 180], dtype=np.float32)  # [X轴旋转, Y轴旋转, Z轴旋转]
    else:
        angles = np.array([rot_x, rot_y, rot_z], dtype=np.float32)
    od = {'x': 0, 'y': 1, 'z': 2}
    idx = [od[k] for k in order]
    angles = angles[idx]
    # 指定旋转顺序，例如 'xyz'
    r = Rotation.from_euler(order, angles)

    # 将欧拉角转换为旋转矩阵
    rotation_matrix = r.as_matrix()
    rotation_matrix[np.abs(rotation_matrix) < eps] = 0
    return torch.from_numpy(rotation_matrix.astype(np.float32))


def rotation_matrix_to_euler(rotmatrix, order = 'xyz', degree = True, eps=1e-6):
    '''
    旋转矩阵转欧拉角  
    '''
    r = Rotation.from_matrix(rotmatrix)
    euler = r.as_euler(order)
    if degree:
        euler = euler / np.pi * 180
    return torch.from_numpy(euler.astype(np.float32))

def euler_to_quaternion(rot, order = 'xyz', degree = True, wxyz = False):
    if degree:
        rot = rot * pi / 180
    r = Rotation.from_euler(order, rot)
    quat = r.as_quat(scalar_first=wxyz)
    return torch.from_numpy(quat.astype(np.float32))

def quaternion_to_euler(quat, order = 'xyz', degree = True):
    r = Rotation.from_quat(quat)
    euler = r.as_euler(seq=order)
    if degree:
        euler = euler / np.pi * 180
    return torch.from_numpy(euler.astype(np.float32))

def decomposite_instrisic(intri:torch.Tensor):
    '''
    return fx, cx, fy, cy in pixel unit
    '''
    return intri[0, 0], intri[0, 2], abs(intri[1, 1]), intri[1, 2]


def compute_coresponding_from_depth(depth_map:torch.Tensor, cam_intri:torch.Tensor, proj_intri:torch.Tensor, R:torch.Tensor, T:torch.Tensor):
    '''
    compute camera pixels' corresponding point in projector's pixels.
    with the real depth map, camera and projector's intrisic matrix known, compute the real coresponding map.  
    R:  the rotation matrix of the camera relative to the projector, (3, 3)
    T:  the translation vector of the camera relative to the projector, (3)
    it's a re-projection process.  
    the depth here is defined as the projected distance along the z-axis, rather than the distance from the camera's optical center to the object point.
    '''
    # h, w = depth_map.shape
    # y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij'), # x, y (h, w)
    # y = y.to(depth_map.device)
    # x = x.to(depth_map.device)
    # pixel_coord_cam = torch.stack([x, y, torch.ones_like(x, device=x.device)], dim=-1)
    # cam_intri_inv = torch.linalg.inv(cam_intri)
    # space_coord_cam = torch.einsum("mk, hwk -> hwm", cam_intri_inv, pixel_coord_cam * depth_map[:,:,None])
    space_coord_cam = coord_pixel2camera(depth_map, cam_intri)
    space_coord_cam_ho = torch.concat([space_coord_cam, torch.ones_like(space_coord_cam[...,:1], device=space_coord_cam.device)], dim=-1)
    trans = RT2TransformMatrix(R, T, want='3x4')
    space_coord_proj = torch.einsum("...jk, ...hwk -> ...hwj", trans, space_coord_cam_ho)  # (h, w, 3)
    pixel_coord_proj = torch.einsum("...jk, ...hwk -> ...hwj", proj_intri, space_coord_proj) / (space_coord_proj[...,2:3] + 1e-6)  # (h, w, 3)
    # 这里的坐标是(x, y), 而不是(y, x)
    return pixel_coord_proj[..., :2]

    # y, x = np.mgrid[:h, :w]
    # # camera's pixel coordinates
    # pixel_coord_cam = np.stack([x, y, np.ones_like(x)], axis=-1)  # (h, w, 3)
    # cam_intri_inv = np.linalg.inv(cam_intri)  # (3, 3)
    # space_coord_cam = np.einsum("mk, hwk -> hwm", cam_intri_inv, pixel_coord_cam * depth_map[:,:,np.newaxis])
    # space_coord_cam_ho = np.concatenate([space_coord_cam, np.ones_like(space_coord_cam[:,:,:1])] , axis=-1)  # (h, w, 4)
    # trans = RT2TransformMatrix(R, T, want='3x4')
    # space_coord_proj = np.einsum("jk, hwk -> hwj", trans, space_coord_cam_ho)  # (h, w, 3)
    # pixel_coord_proj = np.einsum("jk, hwk -> hwj", proj_intri, space_coord_proj) / (space_coord_proj[...,2:3] + 1e-6)  # (h, w, 3)
    # # y坐标是匹配到的结果.
    # return pixel_coord_proj[...,0]

def RT2TransformMatrix(R:torch.Tensor, T:torch.Tensor, want:str = '3x4'):
    '''
    R: (..., 3, 3), T:(..., 3)  
    return [R, T // 0, 1]  
    the return value is np.ndarray only when both R, T are np.ndarray
    '''
    isnumpy = 0
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)
        isnumpy += 1
    if isinstance(T, np.ndarray):
        T = torch.from_numpy(T)
        isnumpy += 1
    isnumpy = isnumpy == 2
    m3x4 = torch.concat([R, T.unsqueeze(-1)], axis=-1)
    if want == '3x4':
        return m3x4 if not isnumpy else m3x4.numpy()
    else:
        m4x4 = torch.concat([m3x4, torch.tensor([0, 0, 0, 1], device=m3x4.device).expand_as(m3x4)[...,0:1,:]], axis=-2)
        return m4x4 if not isnumpy else m4x4.numpy()
    

def reflect(a:torch.Tensor, n:torch.Tensor):
    '''
    a: (..., 3), b: (..., 3), all normalized.  
    a: from scene point to camera.
    '''
    return 2 * n * torch.sum(a*n, dim=-1, keepdim=True) - a


def coord_pixel2camera(depth:torch.Tensor, intrinsic:torch.Tensor):
    '''
    depth: (B, H, W), intrinsic: (B, 3, 3)  
    return: (B, H, W, 3)
    '''
    h, w = depth.shape[-2:]
    y,x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij') # x, y (h, w)
    y = y.to(depth.device)
    x = x.to(depth.device)
    pixel_coord_cam = torch.stack([x, y, torch.ones_like(x, device=x.device)], dim=-1)
    cam_intri_inv = torch.linalg.inv(intrinsic)
    space_coord_cam = torch.einsum("...mk, ...hwk -> ...hwm", cam_intri_inv, pixel_coord_cam * depth[...,None])
    return space_coord_cam


def normalize_pixel_coord(c:torch.Tensor, h, w):
    '''
    c: (B, H, W, 2)
    '''
    t = torch.tensor([w, h], dtype=torch.float32, device=c.device)
    return c / t * 2 - 1


def normalize_image(img:np.ndarray, bit_depth: int = 8):
    max_val = 2 ** bit_depth - 1
    return img.astype(np.float32) / max_val


def load_cv2_for_exr():
    import sys, os
    import importlib
    if "cv2" in sys.modules:
        if not 'OPENCV_IO_ENABLE_OPENEXR' in os.environ:
            os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
            importlib.reload(sys.modules['cv2'])
        else:
            return
    else:
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
        importlib.import_module("cv2")

def load_exr(path, type = "RGB", bit16 = False):
    '''
    type: "RGB", "NORMAL", "Z"
    '''
    exr_file = OpenEXR.InputFile(path)

    # 获取图像的宽度和高度
    dw = exr_file.header()['dataWindow']
    # print(exr_file.header()['channels'])
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT if not bit16 else Imath.PixelType.HALF)
    if type == 'RGB':
        ch = ["R", "G", "B"]
    elif type == 'NORMAL':
        ch = ["X", "Y", "Z"]
    else:
        ch = ["V"]
    channels = exr_file.channels(ch, FLOAT)
    dtype = np.float16 if bit16 else np.float32
    # 将数据转换为NumPy数组
    if type == 'Z':
        d = np.frombuffer(channels[0], dtype=dtype).reshape(height, width).astype(np.float32)
        return d.copy()
    r = np.frombuffer(channels[0], dtype=dtype).reshape(height, width).astype(np.float32)
    g = np.frombuffer(channels[1], dtype=dtype).reshape(height, width).astype(np.float32)
    b = np.frombuffer(channels[2], dtype=dtype).reshape(height, width).astype(np.float32)

    # 如果需要，你可以将RGB值合并为一个图像
    image = np.stack([r, g, b], axis=-1)
    exr_file.close()
    return image.copy()   # 原buffer是只读的, 需要copy一下变成可写的.

def save_exr_to_png(exr:np.ndarray, path):
    '''
    linear转换为srgb，然后保存为png...  
    默认为RGB而非BGR
    '''
    if isinstance(exr, torch.Tensor):
        exr = exr.detach().cpu().numpy()
    srgb = ColorConversionFunc.get_conversion_func("LINEAR", "SRGB")(exr.squeeze())
    srgb = np.clip(srgb * 255, 0, 255).astype(np.uint8)
    srgb = cv2.cvtColor(srgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, srgb)

def exr_to_srgb_png(path, bit16 = False):
    '''
    直接把path转化为同名的.png srgb图片.
    '''
    outpath = os.path.join(os.path.dirname(path), f"{Path(path).stem}.png")
    exr = load_exr(path, bit16=bit16)
    save_exr_to_png(exr, outpath)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # f = "tmp/test_random/decomp_white/MaterialType0001.exr"
    f = "data/images/00000/000_L_MaterialType.exr"
    depth = load_exr(f, "Z", bit16=True)
    print(depth.shape)
    print(depth.min(), depth.max())
    plt.imshow(depth, cmap='gray', vmin=depth.min(), vmax=depth.max())
    plt.colorbar()
    plt.show()