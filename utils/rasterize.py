import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer
from pytorch3d.renderer.points import rasterize_points

def project_pointclouds_to_depthmap(pointscloud:torch.Tensor, K:torch.Tensor, R:torch.Tensor, T:torch.Tensor, h, w):
    '''
    points: (N,...,3)  
    K: (N, 3, 3)
    R: (N, 3, 3)
    T: (N, 3,)
    '''
    b = pointscloud.shape[0]
    pointscloud = pointscloud.view(b, -1, 3)
    f = torch.stack([K[:,0,0], K[:,1,1]], dim=-1) # (N, 2)
    c = torch.stack([K[:,0,2], K[:,1,2]], dim=-1) # (N, 2)
    image_size = torch.tensor([[h, w]], dtype=pointscloud.dtype, device=pointscloud.device).expand(b, 2)
    
    camera = PerspectiveCameras(f, c, R, T, None, pointscloud.device, False, image_size)
    raster_settings = PointsRasterizationSettings(
        image_size=(h, w),
        radius=0.01,            # 点半径
        points_per_pixel=1      # 每像素最多考虑的点数
    )
    rasterizer = PointsRasterizer(camera, raster_settings)
    
    point_cloud = Pointclouds(points=pointscloud)
    fragments = rasterizer.forward(point_cloud)

    return fragments.zbuf