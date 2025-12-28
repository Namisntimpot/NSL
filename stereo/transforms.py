import numpy as np
import cv2
from scipy.ndimage import generic_filter

def bino_rectify(l_images, r_images, l_intri, r_intri, l_dist, r_dist, R, T, alpha,
                 R1=None, P1=None, R2=None, P2=None, Q=None):
    '''要求图像前两个维度是h, w, l_images, r_images可以是list.'''
    h, w = l_images.shape[:2] if not isinstance(l_images, list) else l_images[0].shape[:2]
    if R1 is None or P1 is None or R2 is None or P2 is None or Q is None:
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            l_intri, l_dist, r_intri, r_dist, (w, h), R, T, alpha=alpha
        )
    # 计算左右图像的映射
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        l_intri, l_dist, R1, P1, (w, h), cv2.CV_32FC2
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        r_intri, r_dist, R2, P2, (w, h), cv2.CV_32FC2
    )

    # 应用映射进行校正
    if isinstance(l_images, list):
        rectified_left = [cv2.remap(l_image, left_map1, left_map2, cv2.INTER_LINEAR) for l_image in l_images]
        rectified_right= [cv2.remap(r_image, right_map1, right_map2, cv2.INTER_LINEAR) for r_image in r_images]
    else:
        rectified_left = cv2.remap(l_images, left_map1, left_map2, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(r_images, right_map1, right_map2, cv2.INTER_LINEAR)

    return rectified_left, rectified_right, R1, R2, P1, P2, Q

def __trinocular_rectify(l_intri, r_intri, m_intri, l_dist, r_dist, m_dist, Rlr, Tlr, Rlm, Tlm, alpha, w, h):
    '''简化三目矫正，假设三者共线.'''
    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(
        l_intri, l_dist, r_intri, r_dist, (w, h), Rlr, Tlr, alpha=alpha, flags=cv2.CALIB_ZERO_DISPARITY
    )
    Knew = PL[:,:3]
    Rlm_inv = Rlm.T
    Tlm_inv = -Rlm_inv @ Tlm
    RM = RL @ Rlm_inv
    TM = RL @ Tlm_inv
    EM = np.hstack((RM, TM))
    PM = Knew @ EM
    return RL, RR, RM, PL, PR, PM, TM
    

def trino_rectify(l_images, r_images, m_images, l_intri, r_intri, m_intri, l_dist, r_dist, m_dist, Rlr, Tlr, Rlm, Tlm, alpha,
                  RL=None, PL=None, RR=None, PR=None, RP=None, PP=None, Q=None, TP=None):
    '''要求图像前两个维度是h, w'''
    h, w = l_images.shape[:2] if not isinstance(l_images, list) else l_images[0].shape[:2]
    if RL is None or PL is None or RR is None or PR is None or RP is None or PP is None or TP is None:
        RL, RR, RP, PL, PR, PP, TP = __trinocular_rectify(
            l_intri, r_intri, m_intri, l_dist, r_dist, m_dist, Rlr, Tlr, Rlm, Tlm, alpha, w, h
        )
    # 计算左右图像的映射
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        l_intri, l_dist, RL, PL, (w, h), cv2.CV_32FC2
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        r_intri, r_dist, RR, PR, (w, h), cv2.CV_32FC2
    )
    # middle_map1, middle_map2 = cv2.initUndistortRectifyMap(
    #     m_intri, m_dist, RP, PP, (w, h), cv2.CV_32FC2
    # )
    t = np.array([[0], [np.abs(TP[1,0])], [0]])
    middle_map1, middle_map2 = initUndistortRectifyMapWithT(
        m_intri, m_dist, RP, t, PL[:,:3], (w, h)
    )

    # 应用映射进行校正
    if isinstance(l_images, list):
        rectified_left = [cv2.remap(l_image, left_map1, left_map2, cv2.INTER_LINEAR) for l_image in l_images]
        rectified_right = [cv2.remap(r_image, right_map1, right_map2, cv2.INTER_LINEAR) for r_image in r_images]
        rectified_middle = [cv2.remap(m_image, middle_map1, middle_map2, cv2.INTER_LINEAR) for m_image in m_images]
    else:
        rectified_left = cv2.remap(l_images, left_map1, left_map2, cv2.INTER_LINEAR)
        rectified_right= cv2.remap(r_images, right_map1, right_map2, cv2.INTER_LINEAR)
        rectified_middle=cv2.remap(m_images, middle_map1, middle_map2, cv2.INTER_LINEAR)

    return rectified_left, rectified_right, rectified_middle, RL, RR, RP, PL, PR, PP, Q



def trino_rectify_v2(KL, DL, KR, DR, KM, DM, Rlr, Tlr, Rlm, Tlm, alpha, w, h, wm=None, hm=None, z = 1000.):
    '''
    KM需要已经提前unfied了..  
    直接返回rectified images.  
    z: in mm  
    '''
    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(
        KL, DL, KR, DR, (w, h), Rlr, Tlr, alpha=alpha, flags=cv2.CALIB_ZERO_DISPARITY
    )
    b_lr = 1 / Q[3, 2]
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        KL, DL, RL, PL, (w, h), cv2.CV_32FC2
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        KR, DR, RR, PR, (w, h), cv2.CV_32FC2
    )

    if wm is None:
        wm = w
    if hm is None:
        hm = w
    Kp_undistort, roi = cv2.getOptimalNewCameraMatrix(KM, DM, (wm, hm), -1, (wm, hm))
    m_undistort_map1, m_undistort_map2 = cv2.initUndistortRectifyMap(KM, DM, None, Kp_undistort, (wm, hm), cv2.CV_32FC2)

    K_new = PL[:, :3]
    RM = Rlm @ RL.T
    TM = RL @ Tlm

    tx, ty, tz = 0, TM[1,0], TM[2,0]
    t = np.array([[tx], [ty], [tz]], dtype=TM.dtype)
    # print(t)
    TM = np.array([[TM[0,0]], [0], [0]], dtype=TM.dtype)
    b_lm = -TM[0,0]

    new_grid_y, new_grid_x = np.mgrid[:h, :w]
    new_grid = np.stack([new_grid_x, new_grid_y, np.ones_like(new_grid_x)], axis=-1) # h,w,3
    new_grid = np.float32(new_grid[...,None]) # h,w,3,1
    uvw = (Kp_undistort @ RM @ np.linalg.inv(K_new) @ new_grid)[...,0]  # h,w,3
    u0_w0 = uvw[...,0] / (uvw[..., -1] + 1e-8)
    v0_w0 = uvw[...,1] / (uvw[..., -1] + 1e-8)
    w0 = uvw[..., -1]
    t_off = K_new @ t
    a, b, tz = t_off[0,0], t_off[1,0], t_off[2,0]

    map_x = u0_w0 + (a - u0_w0 * tz) / (w0 * z + tz)
    map_y = v0_w0 + (b - v0_w0 * tz) / (w0 * z + tz)

    middle_map1 = np.float32(np.stack((map_x, map_y), axis=-1))

    return left_map1, right_map1, middle_map1, m_undistort_map1, K_new, b_lr, b_lm



def unify_L_intri(Klr, Klp, Kp, Rp, Tp):
    H = np.linalg.inv(Klr) @ Klp
    decomp = H @ Rp @ np.linalg.inv(Kp)
    retval, Kp_new, Rp_new, *_ = cv2.RQDecomp3x3(np.linalg.inv(decomp))
    Rp_new = Rp_new.T
    return Kp_new, Rp_new, Tp



def initUndistortRectifyMapWithT(cameraMatrix, distCoeffs, R, T, newCameraMatrix, size):
    """
    手动实现 cv2.initUndistortRectifyMap，考虑前5个畸变参数，并额外输入平移向量 T。
    使用 NumPy 数组广播计算，避免逐个像素遍历。
    假设 R 和 T 描述从原始相机到矫正后 3D 空间的变换，
    newCameraMatrix 是矫正后 3D 空间到图像平面的投影。

    Args:
        cameraMatrix (np.ndarray): 原始相机内参矩阵 (3x3).
        distCoeffs (np.ndarray): 畸变系数 (k1, k2, p1, p2, k3). 形状应为 (5,) 或 (1, 5) 等。
        R (np.ndarray): 矫正旋转矩阵 (3x3).
        T (np.ndarray): 平移向量 (3,).
        newCameraMatrix (np.ndarray): 新的相机内参矩阵 (3x3).
        size (tuple): 输出图像的尺寸 (width, height).

    Returns:
        tuple: map1, map2 (np.ndarray), 用于 cv2.remap 的映射图。
    """
    width, height = size

    # 确保畸变系数是包含 k1, k2, p1, p2, k3 的 numpy 数组
    if distCoeffs is None:
        distCoeffs = np.zeros(5, dtype=np.float64)
    elif distCoeffs.size < 5:
        temp_dist = np.zeros(5, dtype=np.float64)
        temp_dist[:distCoeffs.size] = distCoeffs.flatten()
        distCoeffs = temp_dist
    else:
        distCoeffs = distCoeffs.flatten()[:5]

    k1, k2, p1, p2, k3 = distCoeffs

    # 计算矫正旋转矩阵的逆 (对于旋转矩阵，逆等于转置)
    R_inv = R.T

    # 将平移向量 T 转换为列向量
    T_col = T.reshape(3, 1)

    # 原始相机矩阵参数
    fx_orig = cameraMatrix[0, 0]
    fy_orig = cameraMatrix[1, 1]
    cx_orig = cameraMatrix[0, 2]
    cy_orig = cameraMatrix[1, 2]

    # 新相机矩阵参数 (用于将矫正后像素坐标转换为归一化坐标)
    fx_new = newCameraMatrix[0, 0]
    fy_new = newCameraMatrix[1, 1]
    cx_new = newCameraMatrix[0, 2]
    cy_new = newCameraMatrix[1, 2]

    T_col[0, 0] = T_col[0, 0] / fx_new
    T_col[1, 0] = T_col[1, 0] / fy_new
    T_col[2, 0] = 0

    # 1. 创建输出图像的像素坐标网格
    u_prime_grid, v_prime_grid = np.meshgrid(np.arange(width), np.arange(height))

    # 2. 输出图像像素坐标到新相机坐标系下的归一化坐标 (在 Z=1 平面上)
    x_double_prime = (u_prime_grid.astype(np.float64) - cx_new) / fx_new
    y_double_prime = (v_prime_grid.astype(np.float64) - cy_new) / fy_new

    # 将归一化坐标和 Z=1 坐标堆叠起来形成 3D 点在矫正后空间 Z=1 平面上的表示
    # 形状为 (3, height * width)
    P_double_prime_norm = np.vstack((x_double_prime.flatten(), y_double_prime.flatten(), np.ones(width * height)))

    # 3. 矫正后 3D 空间坐标系的点到原始相机坐标系 (应用逆变换)
    # P_orig = R_inv * (P_rectified - T)
    # 这里 P_rectified 我们使用 P_double_prime_norm 作为在矫正后空间 Z=1 平面上的点
    # T_col 的形状是 (3, 1)，P_double_prime_norm 的形状是 (3, height * width)
    # NumPy 的广播机制会自动将 T_col 广播到 (3, height * width) 进行减法
    P_orig_3d = R_inv @ (P_double_prime_norm - T_col)

    # 4. 原始相机 3D 空间点到未畸变归一化坐标
    # P_orig_3d 的形状是 (3, height * width)
    x_u = P_orig_3d[0, :] / P_orig_3d[2, :]
    y_u = P_orig_3d[1, :] / P_orig_3d[2, :]

    # 形状现在是 (height * width,) 的一维数组

    # 5. 未畸变归一化坐标到畸变归一化坐标 (应用畸变模型)
    r_u_squared = x_u**2 + y_u**2

    # 径向畸变
    radial_distortion_factor = 1 + k1 * r_u_squared + k2 * r_u_squared**2 + k3 * r_u_squared**3

    x_d_radial = x_u * radial_distortion_factor
    y_d_radial = y_u * radial_distortion_factor

    # 切向畸变
    x_d_tangential = 2 * p1 * x_u * y_u + p2 * (r_u_squared + 2 * x_u**2)
    y_d_tangential = p1 * (r_u_squared + 2 * y_u**2) + 2 * p2 * x_u * y_u

    # 总畸变归一化坐标
    x_d = x_d_radial + x_d_tangential
    y_d = y_d_radial + y_d_tangential

    # 6. 畸变归一化坐标到原始图像像素坐标
    u = fx_orig * x_d + cx_orig
    v = fy_orig * y_d + cy_orig

    # 7. 将结果重塑回图像尺寸
    map1 = u.reshape((height, width)).astype(np.float32)
    map2 = v.reshape((height, width)).astype(np.float32)

    return np.stack((map1, map2), axis=-1), None


def __projected_coord_2_depthmap(pixel_coords, z_coords, image_h, image_w):
    # 确定深度图的尺寸
    max_u = image_w - 1
    max_v = image_h - 1
    # 创建一个空的深度图，初始化为 0 或其他表示无效深度的值
    depth_map = np.zeros((max_v + 1, max_u + 1), dtype=np.float32)

    # 填充深度图
    for i in range(len(pixel_coords)):
        coord = map(np.round, pixel_coords[i])  # 将像素坐标转换为整数
        u, v = map(int, coord)
        z = z_coords[i][0]
        if 0 <= v <= max_v and 0 <= u <= max_u:
            depth_map[v, u] = z if depth_map[v, u] == 0 else min(z, depth_map[v, u])
    return depth_map


def point_cloud_2_depth_map_unrectified(
        point_cloud: np.ndarray, K:np.ndarray, D:np.ndarray, image_h, image_w
    ):
    '''point_cloud: (N, 3)'''
    # 将点云数据转换为 OpenCV 期望的格式 (N x 1 x 3)
    object_points = point_cloud.reshape(-1, 1, 3).astype(np.float32)
    # 使用 cv2.projectPoints 进行投影
    # rvec 和 tvec 分别是旋转向量和平移向量，因为点已经在相机坐标系下，所以都为零
    projected_points, jacobian = cv2.projectPoints(object_points,
                                                           np.array([0.0, 0.0, 0.0]),
                                                           np.array([0.0, 0.0, 0.0]),
                                                           K, D)
    # projected_points 是一个 N x 1 x 2 的数组，需要 Reshape 成 N x 2
    pixel_coords = projected_points.reshape(-1, 2)
    # Z 坐标直接从输入的点云中获取
    z_coords = point_cloud[:, 2].reshape(-1, 1)

    return __projected_coord_2_depthmap(pixel_coords, z_coords, image_h, image_w)


def point_cloud_2_depth_map_rectified(
        point_cloud:np.ndarray, R:np.ndarray, P:np.ndarray, image_h, image_w
    ):
    '''point_cloud: (N, 3)'''
    # 1. 将点云变换到矫正后的相机坐标系
    rotated_points = point_cloud @ R.T  # 使用矩阵乘法，注意 R1 需要转置

    # 2. 添加齐次坐标
    homogeneous_rotated_points = np.hstack((rotated_points, np.ones((rotated_points.shape[0], 1))))

    # 3. 使用投影矩阵 P1 进行投影
    projected_points_homogeneous = homogeneous_rotated_points @ P.T  # 使用矩阵乘法，注意 P1 需要转置

    # 4. 从齐次坐标获取像素坐标
    u = projected_points_homogeneous[:, 0] / projected_points_homogeneous[:, 2]
    v = projected_points_homogeneous[:, 1] / projected_points_homogeneous[:, 2]
    rectified_pixel_coords = np.vstack((u, v)).T

    # 5. 矫正后的 Z 坐标
    rectified_z_coords = rotated_points[:, 2].reshape(-1, 1)

    return __projected_coord_2_depthmap(rectified_pixel_coords, rectified_z_coords, image_h, image_w)


def noisy_mask(arr, wnd_size, percentage):
    '''arr must be np.ndarray. percentage: (0,100)'''
    local_std = generic_filter(arr, np.std, size=wnd_size)
    threshold = np.percentile(local_std, percentage)
    mask = local_std < threshold
    return mask