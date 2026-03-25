# Copyright (c) Teacher-Student Distillation Pipeline
"""
教师线：生成伪 3D 框（与 cubercnn/generate_label/process_indoor.py 的 estimate_bbox 逻辑严格一致）

与原版一致的处理流程：
  1. create_uv_depth + project_image_to_cam -> 3D 点云
  2. adaptive_erode_mask（分离轴腐蚀，与原版完全一致）
  3. PCA 估计 yaw + min/max 求尺寸 / proposal 优化
  4. 无物理补偿、无先验软约束（与原版一致）

与原版不同的可选增强（通过参数开启）：
  - L-Shape 分支（use_lshape=True）：方差准则求 yaw，用于对比实验
  - SOR 滤波（sor_k > 0）：3D 离群点去除
"""
import numpy as np
import cv2
import torch
from sklearn.decomposition import PCA

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from cubercnn.generate_label.util import (
    project_image_to_cam,
    rotation_matrix_from_vectors,
    rotate_y,
    point_to_plane_distance,
    convert_box_vertices,
    normalize,
    find_min_max_indices,
    generate_possible_bboxs,
)
from cubercnn.generate_label.raytrace import calc_dis_ray_tracing, calc_inside_ratio


# =============================================================================
# 深度 -> 点云（与原版完全一致，无 depth_scale）
# =============================================================================

def create_uv_depth(depth, mask=None):
    """与 cubercnn/generate_label/util.py create_uv_depth 完全一致。"""
    if mask is not None:
        depth = depth * mask
    x, y = np.meshgrid(
        np.linspace(0, depth.shape[1] - 1, depth.shape[1]),
        np.linspace(0, depth.shape[0] - 1, depth.shape[0]),
    )
    uv_depth = np.stack((x, y, depth), axis=-1)
    uv_depth = uv_depth.reshape(-1, 3)
    return uv_depth[uv_depth[:, 2] != 0]


def reproject_to_point_cloud(depth_np, mask_np, K):
    """
    与 cubercnn/process_indoor.py process_instances 逻辑一致：
    depth_np 必须是米（由调用方通过 depth_scale 统一缩放后再传入），
    此函数不再做任何缩放。

    depth_np: (H, W)，单位：米
    mask_np : (H, W)
    K        : (3, 3)
    Returns  : (N, 3) 相机坐标系点云，单位：米
    """
    uv_depth = create_uv_depth(depth_np, mask_np)
    if uv_depth.shape[0] < 10:
        return np.zeros((0, 3))
    return project_image_to_cam(uv_depth, np.array(K))


# =============================================================================
# Mask 腐蚀（与原版完全一致）
# =============================================================================

def erode_mask(mask, k_vertical, k_horizontal):
    """
    与 cubercnn/generate_label/util.py erode_mask 完全一致。
    mask: (H, W)
    """
    mask_uint8 = (mask.squeeze().astype(np.float32) > 0.5).astype(np.uint8)
    kernel_vertical = np.ones((3, 1), np.uint8)
    kernel_horizontal = np.ones((1, 3), np.uint8)
    eroded_v = cv2.erode(mask_uint8, kernel_vertical, iterations=k_vertical)
    eroded_h = cv2.erode(mask_uint8, kernel_horizontal, iterations=k_horizontal)
    return np.logical_and(eroded_v, eroded_h).astype(np.float32)


def adaptive_erode_mask(mask, k_vertical, k_vertical_min, k_horizontal, k_horizontal_min):
    """
    与 cubercnn/generate_label/util.py adaptive_erode_mask 完全一致。
    mask: (N, 1, H, W) 或 (H, W)
    返回: 腐蚀后的 mask
    """
    if mask.ndim == 2:
        mask = mask[np.newaxis, np.newaxis]
    new_mask = np.zeros_like(mask)
    kernel_vertical = np.ones((3, 1), np.uint8)
    kernel_horizontal = np.ones((1, 3), np.uint8)

    for i in range(mask.shape[0]):
        mask_i = mask[i, 0]
        try:
            min_row, min_col, max_row, max_col = find_min_max_indices(mask_i)
        except (ValueError, IndexError):
            # Empty mask (all zeros) - skip erosion
            continue
        k_v = k_vertical if max_row - min_row >= 10 else k_vertical_min
        k_h = k_horizontal if max_col - min_col >= 10 else k_horizontal_min
        eroded_v = cv2.erode(mask_i, kernel_vertical, iterations=k_v)
        eroded_h = cv2.erode(mask_i, kernel_horizontal, iterations=k_h)
        new_mask[i, 0] = np.logical_and(eroded_v, eroded_h).astype(np.uint8)
    return new_mask


def adaptive_erode_mask_single(mask_np, k_vertical=12, k_vertical_min=2,
                                k_horizontal=6, k_horizontal_min=2):
    """
    对单张 mask (H, W) 做自适应腐蚀。
    与 cubercnn adaptive_erode_mask 逻辑一致（单通道版本）。
    """
    mask_bin = (mask_np.squeeze().astype(np.float32) > 0.5).astype(np.uint8)
    if mask_bin.sum() < 10:
        return mask_bin.astype(np.float32)
    try:
        min_row, min_col, max_row, max_col = find_min_max_indices(mask_bin)
    except (ValueError, IndexError):
        return mask_bin.astype(np.float32)
    k_v = k_vertical if max_row - min_row >= 10 else k_vertical_min
    k_h = k_horizontal if max_col - min_col >= 10 else k_horizontal_min
    kernel_vertical = np.ones((3, 1), np.uint8)
    kernel_horizontal = np.ones((1, 3), np.uint8)
    eroded_v = cv2.erode(mask_bin, kernel_vertical, iterations=k_v)
    eroded_h = cv2.erode(mask_bin, kernel_horizontal, iterations=k_h)
    return np.logical_and(eroded_v, eroded_h).astype(np.float32)


def sor_filter_3d(points, k=20, std_ratio=2.0):
    """
    3D 统计离群点去除 (SOR)。
    与原版一致（默认不开启，仅作可选增强）。
    """
    if points.shape[0] < k + 1:
        return points
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(points)
    distances, _ = nbrs.kneighbors(points)
    mean_d = np.mean(distances[:, 1:], axis=1)
    thresh = np.mean(mean_d) + std_ratio * np.std(mean_d)
    inlier = mean_d < thresh
    return points[inlier]


# =============================================================================
# L-Shape 方差准则（可选增强，不在原版中）
# =============================================================================

def lshape_fit_yaw_and_dims(rotated_pc_xz):
    """
    L-Shape 方差准则：求 yaw 角和 x-z 平面内的跨度。
    返回值语义与原版 PCA 分支一致：
      yaw : 偏航角
      dx  : 投影到 yaw 方向后的跨度（= 原版中的 dx）
      dz  : 投影到垂直 yaw 方向后的跨度（= 原版中的 dz）
    """
    if rotated_pc_xz.shape[0] < 10:
        return 0.0, 0.1, 0.1

    best_yaw, best_var = 0.0, -1.0
    # Use seeded RNG for reproducibility
    rng = np.random.default_rng(42)
    sample = rotated_pc_xz if rotated_pc_xz.shape[0] <= 500 else rotated_pc_xz[rng.choice(rotated_pc_xz.shape[0], size=500, replace=False)]
    for angle in np.linspace(0, np.pi, 36):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        proj = sample @ R.T
        var_x = np.var(proj[:, 0])
        if var_x > best_var:
            best_var = var_x
            best_yaw = angle

    c, s = np.cos(best_yaw), np.sin(best_yaw)
    R = np.array([[c, -s], [s, c]])
    proj = sample @ R.T
    dx = max(float(np.ptp(proj[:, 0])), 0.1)
    dz = max(float(np.ptp(proj[:, 1])), 0.1)
    return best_yaw, dx, dz


# =============================================================================
# 核心：点云 -> 3D 框（与 process_indoor.py estimate_bbox 严格一致）
# =============================================================================

def teacher_point_cloud_to_bbox(
    in_pc,
    category_name,
    prior_dict,
    ground_equ=None,
    use_lshape=False,
    erosion_k=2,
    sor_k=20,
    sor_std=2.0,
    use_ray_tracing=True,
    ray_tracing_weight=1.0,
    inside_ratio_weight=5.0,
    min_points_for_bbox=10,  # 新增参数：用于 bbox 估计的最小点数
    debug=False,
):
    """
    从点云生成伪 3D 框——与 cubercnn/generate_label/process_indoor.py
    estimate_bbox 逻辑严格一致。

    两种分支（通过 use_lshape 切换）：
      - use_lshape=True：  L-Shape 方差准则求 yaw（仅 yaw 估计方式不同，无 proposal 优化）
      - use_lshape=False： PCA 求 yaw + proposal 优化（原版默认，完全一致）

    返回格式（两分支统一）：
      center_cam : (3,) [X, Y, Z]，相机坐标系
      dimensions : (3,) [W, H, L] = [dz, dy, dx]，与 cubercnn convert_box_vertices 约定一致
      R_cam      : (3,3) 旋转矩阵
      success    : bool

    与原版一致（不含自定义增强）：
      - 无物理补偿
      - 无先验软约束
      - 仅高度有先验：dy < prior_h*0.5 时替换
      - SOR 跳过（与原版一致）
    """
    if in_pc.shape[0] < min_points_for_bbox:
        return np.zeros(3), np.array([0.5, 0.5, 0.5]), np.eye(3), False

    prior = np.array(prior_dict.get(category_name.strip().lower(), [0.5, 0.5, 0.5]))
    w, h, l = prior  # prior = [w, h, l]

    # ---- 地面对齐（与原版完全一致）----
    if ground_equ is not None:
        dot_product = np.dot([0, -1, 0], ground_equ[:3])
        if dot_product <= 0:
            ground_equ = -ground_equ
        new_ground_equ = np.array([0, -1, 0, point_to_plane_distance(ground_equ, 0, 0, 0)])
        rotation_matrix = rotation_matrix_from_vectors([0, -1, 0], ground_equ[:3])
    else:
        rotation_matrix = np.eye(3)

    rotated_pc = in_pc @ rotation_matrix.T

    # -------------------------------------------------------------------------
    # L-Shape 分支（可选增强，非原版）
    #   坐标处理流程与原版一致：地面旋转 -> 求 dy/cx/cy/cz -> convert_box_vertices
    #   仅 yaw 估计用方差准则替代 PCA，无 proposal 优化
    # -------------------------------------------------------------------------
    if use_lshape:
        xz = rotated_pc[:, [0, 2]]
        rng = np.random.default_rng(42)
        xz_sample = xz if xz.shape[0] <= 500 else xz[rng.choice(xz.shape[0], size=500, replace=False)]
        yaw, dx, dz = lshape_fit_yaw_and_dims(xz_sample)
        dy = float(np.ptp(rotated_pc[:, 1]))
        cx = float(np.mean(rotated_pc[:, 0]))
        cy = float(np.mean(rotated_pc[:, 1]))
        cz = float(np.mean(rotated_pc[:, 2]))

        # 高度先验（与原版一致）
        if dy < h * 0.5:
            dy = h
            if ground_equ is not None:
                cdis = point_to_plane_distance(new_ground_equ, cx, cy, cz)
                if cdis - dy / 2 < 0.5:
                    cy += cdis - dy / 2

        # 顶点（用于求 center_cam）
        verts = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0)
        verts = np.dot(rotate_y(-yaw), verts.T).T
        verts = np.dot(verts, rotation_matrix.T)
        center_cam = verts.mean(axis=0)
        R_cam = rotation_matrix @ rotate_y(-yaw)

        # dimensions = [W, H, L] = [dz, dy, dx]，与原版一致
        dimensions = np.array([dz, dy, dx])
        if debug:
            print(f"    [L-Shape] yaw={yaw:.3f}, dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")
            print(f"    [L-Shape] dims=[W={dz:.3f}, H={dy:.3f}, L={dx:.3f}]")
        return center_cam, dimensions, R_cam, True

    # -------------------------------------------------------------------------
    # PCA 分支：与 cubercnn process_indoor.py estimate_bbox 完全一致
    # -------------------------------------------------------------------------
    # 点云采样（500 点上限）- 使用局部 RNG 确保可复现性
    rng = np.random.default_rng(42)
    pc_for_pca = rotated_pc.copy()
    if pc_for_pca.shape[0] > 500:
        rand_ind = rng.choice(pc_for_pca.shape[0], size=500, replace=False)
        pc_for_pca = pc_for_pca[rand_ind]

    # PCA 求 yaw
    pca = PCA(2)
    pca.fit(pc_for_pca[:, [0, 2]])
    yaw_vec = pca.components_[0, :]
    yaw = np.arctan2(yaw_vec[1], yaw_vec[0])

    # 旋转到物体坐标系（yaw）
    rotated_pc_2 = rotate_y(yaw) @ rotated_pc.T

    x_min, x_max = rotated_pc_2[0, :].min(), rotated_pc_2[0, :].max()
    y_min, y_max = rotated_pc_2[1, :].min(), rotated_pc_2[1, :].max()
    z_min, z_max = rotated_pc_2[2, :].min(), rotated_pc_2[2, :].max()
    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2

    # 高度先验（与原版一致）
    if dy < h * 0.5:
        dy = h
        if ground_equ is not None:
            cdis = point_to_plane_distance(new_ground_equ, cx, cy, cz)
            if cdis - dy / 2 < 0.5:
                cy += cdis - dy / 2

    # 判断尺寸是否合理：dx/dz 至少覆盖先验的一半
    use_proposal = (l * 0.5 <= dx and w * 0.5 <= dz) or (l * 0.5 <= dz and w * 0.5 <= dx)

    if use_proposal:
        # 直接使用测量尺寸
        if debug:
            print(f"    [PCA] 直接使用: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")
        verts = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0)
        verts = np.dot(rotate_y(-yaw), verts.T).T
        verts = np.dot(verts, rotation_matrix.T)
        center_cam = verts.mean(axis=0)
        R_cam = rotation_matrix @ rotate_y(-yaw)
        dimensions = np.array([dz, dy, dx])
    else:
        # Proposal 优化（与原版完全一致）
        if debug:
            print(f"    [PCA] 触发 proposal！dx={dx:.3f}, dz={dz:.3f}")
        possible_bboxs = generate_possible_bboxs(cx, cz, dx, dz, w, l)
        min_loss = float('inf')
        best_verts = None
        best_dx_p, best_dz_p = dx, dz

        for possible_bbox in possible_bboxs:
            x_min_p, x_max_p, z_min_p, z_max_p = possible_bbox
            dx_p, dz_p = x_max_p - x_min_p, z_max_p - z_min_p
            cx_p, cz_p = (x_min_p + x_max_p) / 2, (z_min_p + z_max_p) / 2
            inside_ratio = calc_inside_ratio(rotated_pc_2, x_min_p, x_max_p, z_min_p, z_max_p)
            verts_p = convert_box_vertices(cx_p, cy, cz_p, dx_p, dy, dz_p, 0)
            verts_p = np.dot(rotate_y(-yaw), verts_p.T).T
            new_cx_p, new_cz_p = verts_p[:, 0].mean(), verts_p[:, 2].mean()

            pc_tensor = torch.from_numpy(rotated_pc).float()
            loss_ray = calc_dis_ray_tracing(
                torch.Tensor([dz_p, dx_p]), torch.Tensor([yaw]),
                pc_tensor[:, [0, 2]], torch.Tensor([new_cx_p, new_cz_p]),
            )
            loss_inside = 1 - inside_ratio
            loss = loss_ray + 5 * loss_inside

            if loss < min_loss:
                min_loss = loss
                best_verts = verts_p
                best_dx_p, best_dz_p = dx_p, dz_p

        # Fallback if no valid proposal found
        if best_verts is None:
            verts = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0)
            verts = np.dot(rotate_y(-yaw), verts.T).T
            verts = np.dot(verts, rotation_matrix.T)
            center_cam = verts.mean(axis=0)
            R_cam = rotation_matrix @ rotate_y(-yaw)
            dimensions = np.array([dz, dy, dx])
        else:
            best_verts = np.dot(best_verts, rotation_matrix.T)
            center_cam = best_verts.mean(axis=0)
            R_cam = rotation_matrix @ rotate_y(-yaw)
            # dimensions = [W, H, L] = [dz, dy, dx]，与原版一致
            dimensions = np.array([best_dz_p, dy, best_dx_p])

    if debug:
        print(f"    [PCA] yaw={yaw:.3f}, dims=[W={dimensions[0]:.3f}, H={dimensions[1]:.3f}, L={dimensions[2]:.3f}]")

    return center_cam, dimensions, R_cam, True


def validate_and_clamp_3d_box_in_image(
    center_cam, dimensions, R_cam, K, image_width, image_height, margin=10, _recursion_count=0
):
    """
    验证 3D 框反投影到 2D 图像时是否越界，如果越界则进行裁剪。
    dimensions 格式为 [W, H, L] = [dz, dy, dx]。
    
    _recursion_count: 内部递归计数器，防止无限递归。
    """
    MAX_RECURSION = 5  # 最大递归深度，防止栈溢出
    W, H, L = dimensions
    x, y, z = center_cam[0], center_cam[1], center_cam[2]

    # 8 corners relative to center
    # convert_box_vertices 约定: l=L(x), w=W(z), h=H(y)
    corners = np.array([
        [-L/2, -W/2, -H/2], [ L/2, -W/2, -H/2],
        [-L/2,  W/2, -H/2], [ L/2,  W/2, -H/2],
        [-L/2, -W/2,  H/2], [ L/2, -W/2,  H/2],
        [-L/2,  W/2,  H/2], [ L/2,  W/2,  H/2],
    ])
    corners_cam = (corners @ R_cam.T) + center_cam

    # Project to 2D
    valid_corners = []
    for corner in corners_cam:
        if corner[2] > 1e-6:
            u = K[0, 0] * corner[0] / corner[2] + K[0, 2]
            v = K[1, 1] * corner[1] / corner[2] + K[1, 2]
            valid_corners.append((u, v))

    if not valid_corners:
        return center_cam, dimensions, R_cam

    us = [u for u, v in valid_corners]
    vs = [v for u, v in valid_corners]
    u_min, u_max = min(us), max(us)
    v_min, v_max = min(vs), max(vs)

    out_of_bounds = (
        u_min < -margin or u_max > image_width + margin or
        v_min < -margin or v_max > image_height + margin
    )

    if out_of_bounds:
        scale_u = image_width / (u_max - u_min + 1e-6)
        scale_v = image_height / (v_max - v_min + 1e-6)
        scale = min(scale_u, scale_v, 1.0)
        dimensions = dimensions * scale
        if scale < 0.95 and _recursion_count < MAX_RECURSION:
            return validate_and_clamp_3d_box_in_image(
                center_cam, dimensions, R_cam, K, image_width, image_height, margin,
                _recursion_count=_recursion_count + 1
            )

    return center_cam, dimensions, R_cam


def run_teacher_pipeline_per_instance(
    depth_np,
    mask_np,
    K,
    category_name,
    prior_dict,
    ground_equ=None,
    erosion_k=2,
    sor_k=20,
    sor_std=2.0,
    use_lshape=False,
    depth_scale=1.0,
    use_adaptive_erode=True,
    adaptive_erode_params=(12, 2, 6, 2),
    image_width=None,
    image_height=None,
    debug=False,
):
    """
    单实例教师流水线——与 cubercnn/process_indoor.py 流程完全一致。

    处理步骤：
      1. 自适应腐蚀 mask
      2. depth_np * depth_scale（转米） -> reproject_to_point_cloud -> 3D 点云
      3. teacher_point_cloud_to_bbox（PCA 或 L-Shape） -> 伪 3D 框

    与原版的区别：
      - depth_scale 参数：调用方在此处统一做缩放，确保传入 teacher_point_cloud_to_bbox
        时 depth 已经是米。原版处理的是原始 SUN RGB-D uint16 深度（需除以 8000），
        所以调用方应先对 depth_np 做 scale 再传进来。

    默认 use_lshape=False，与原版一致使用 PCA 分支。
    """
    if debug:
        print(f"  [DEBUG] category={category_name}, mask_sum={mask_np.sum():.0f}, K_fx={K[0,0]:.1f}")

    # ---- 1. 自适应腐蚀 mask ----
    if use_adaptive_erode:
        mask_eroded = adaptive_erode_mask_single(
            mask_np,
            k_vertical=adaptive_erode_params[0],
            k_vertical_min=adaptive_erode_params[1],
            k_horizontal=adaptive_erode_params[2],
            k_horizontal_min=adaptive_erode_params[3],
        )
        if debug:
            print(f"  [DEBUG] adaptive erosion applied")
    else:
        mask_eroded = erode_mask(mask_np, erosion_k, erosion_k)
        if debug:
            print(f"  [DEBUG] fixed erosion k={erosion_k}")

    # ---- 2. 深度缩放 -> 点云反投影 ----
    # 调用方负责确保 depth_np 已按正确 scale 转米后传入
    depth_m = np.asarray(depth_np, dtype=np.float64)
    if abs(depth_scale - 1.0) > 1e-6:
        depth_m = depth_m * depth_scale

    mask_orig = (mask_np.squeeze() > 0.5).astype(np.float32)

    pc = reproject_to_point_cloud(depth_m, mask_eroded, K)
    if pc.shape[0] < 10:
        if debug:
            print(f"  [DEBUG] erosion gave too few points, trying original mask...")
        pc = reproject_to_point_cloud(depth_m, mask_orig, K)

    if pc.shape[0] < 10:
        return np.zeros(3), np.array([0.5, 0.5, 0.5]), np.eye(3), False

    if debug:
        print(f"  [DEBUG] reprojected pc.shape={pc.shape[0]}")

    # ---- 3. teacher_point_cloud_to_bbox（与原版 estimate_bbox 完全一致）----
    result = teacher_point_cloud_to_bbox(
        pc,
        category_name,
        prior_dict,
        ground_equ=ground_equ,
        use_lshape=use_lshape,
        debug=debug,
    )

    # ---- 4. 可选：3D 框越界检查 ----
    if result[3] and image_width is not None and image_height is not None:
        center_cam, dimensions, R_cam, ok = result
        center_cam, dimensions, R_cam = validate_and_clamp_3d_box_in_image(
            center_cam, dimensions, R_cam, K, image_width, image_height, margin=20
        )
        result = (center_cam, dimensions, R_cam, ok)

    return result
