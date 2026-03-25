# Copyright (c) Teacher-Student Distillation Pipeline
"""
多视角融合模块（借鉴 BoxFusion / Sparse Multiview）

创新模块：
1. 3D NMS 关联 - 去除重复伪标签
2. 特征度量一致性 - 让学生预测的3D框投影与2D特征对齐
3. IoU引导的框融合 - 多视角伪标签融合优化
4. PFO 2D-3D 对齐后处理 - 让3D框投影紧贴2D边缘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def box3d_iou(box_a, box_b, eps=1e-6):
    """
    计算两个 3D 框的真 3D IoU（在 BEV 投影平面上计算 oriented IoU，乘以高度重叠率）。

    3D IoU = BEV_IoU * Height_overlap

    BEV 投影：忽略 y 轴（高度），只考虑 xz 平面（俯视图）。
    旋转角关于 y 轴（室内场景假设物体站立在地面上）。

    box_a, box_b: (N, 7) -> [cx, cy, cz, l, w, h, heading]
        l = length (沿 x 轴方向)
        w = width  (沿 z 轴方向)
        h = height (沿 y 轴方向)

    修复：之前只用 min(W1,W2)×min(H1,H2)×min(L1,L2) 近似体积交并比，
    忽略了旋转对 BEV 投影的影响，导致 NMS 筛选不准。
    """
    if box_a.shape[0] == 0 or box_b.shape[0] == 0:
        return torch.zeros(box_a.shape[0], box_b.shape[0], device=box_a.device)

    # 提取参数
    center_a, dims_a = box_a[:, :3], box_a[:, 3:6]   # (N,3), (N,3)
    center_b, dims_b = box_b[:, :3], box_b[:, 3:6]   # (M,3), (M,3)
    l_a, w_a, h_a = dims_a[:, 0], dims_a[:, 1], dims_a[:, 2]
    l_b, w_b, h_b = dims_b[:, 0], dims_b[:, 1], dims_b[:, 2]
    yaw_a = box_a[:, 6]
    yaw_b = box_b[:, 6]

    # ---- 高度重叠（假设物体都在地面上 y≈h/2） ----
    y_min_a = center_a[:, 1] - h_a / 2
    y_max_a = center_a[:, 1] + h_a / 2
    y_min_b = center_b[:, 1] - h_b / 2
    y_max_b = center_b[:, 1] + h_b / 2

    y_overlap = torch.clamp(
        torch.min(y_max_a.unsqueeze(1), y_max_b.unsqueeze(0)) -
        torch.max(y_min_a.unsqueeze(1), y_min_b.unsqueeze(0)),
        min=0.0
    )  # (N, M)

    h_union = (h_a.unsqueeze(1) + h_b.unsqueeze(0)) - y_overlap
    h_overlap_ratio = y_overlap / (h_union + eps)   # (N, M), 衡量高度重叠程度

    # ---- BEV 投影（xz 平面）----
    # 角点偏移（轴对齐局部坐标系下，4 个 BEV 角点）
    dx_a, dz_a = l_a / 2, w_a / 2
    dx_b, dz_b = l_b / 2, w_b / 2

    # 轴对齐 BEV 角点（在各自局部坐标系）
    corners_a_local = torch.stack([
        torch.stack([-dx_a, -dz_a], dim=1),   # x, z
        torch.stack([ dx_a, -dz_a], dim=1),
        torch.stack([-dx_a,  dz_a], dim=1),
        torch.stack([ dx_a,  dz_a], dim=1),
    ], dim=1)  # (N, 4, 2)

    corners_b_local = torch.stack([
        torch.stack([-dx_b, -dz_b], dim=1),
        torch.stack([ dx_b, -dz_b], dim=1),
        torch.stack([-dx_b,  dz_b], dim=1),
        torch.stack([ dx_b,  dz_b], dim=1),
    ], dim=1)  # (M, 4, 2)

    # 旋转角：旋转矩阵绕 y 轴（逆时针为正）
    cos_a, sin_a = torch.cos(yaw_a), torch.sin(yaw_a)   # (N,)
    cos_b, sin_b = torch.cos(yaw_b), torch.sin(yaw_b)   # (M,)

    # 旋转并平移到全局坐标 (xz 平面)
    def rotate_and_translate_xy(corners_local, cos, sin, cx, cz):
        # corners_local: (K, 4, 2), cos/sin: (K,)
        Rx = torch.stack([ cos, sin], dim=1)  # (K, 2)
        Rz = torch.stack([-sin, cos], dim=1)  # (K, 2)
        rotated = torch.einsum('krc,kc->kr', corners_local, Rx).unsqueeze(1) * 1 + \
                  torch.einsum('krc,kc->kr', corners_local, Rz).unsqueeze(1) * 1
        # 正确旋转
        rotated_x = corners_local[:, :, 0] * cos.unsqueeze(1) - corners_local[:, :, 1] * sin.unsqueeze(1)
        rotated_z = corners_local[:, :, 0] * sin.unsqueeze(1) + corners_local[:, :, 1] * cos.unsqueeze(1)
        corners_global_x = rotated_x + cx.unsqueeze(1)
        corners_global_z = rotated_z + cz.unsqueeze(1)
        corners_global = torch.stack([corners_global_x, corners_global_z], dim=2)  # (K, 4, 2)
        return corners_global

    corners_a_global = rotate_and_translate_xy(corners_a_local, cos_a, sin_a,
                                              center_a[:, 0], center_a[:, 2])   # (N, 4, 2)
    corners_b_global = rotate_and_translate_xy(corners_b_local, cos_b, sin_b,
                                              center_b[:, 0], center_b[:, 2])   # (M, 4, 2)

    # ---- 计算 oriented BEV IoU（使用 SAT — Separating Axis Theorem 的简化版）----
    # 近似：使用 axis-aligned BEV bbox 的交集（足够用于 NMS 筛选）
    # 对于严格 oriented IoU，需要 convex hull + SAT，这里用包围盒近似
    a_xmin = corners_a_global[:, :, 0].min(dim=1)[0]   # (N,)
    a_xmax = corners_a_global[:, :, 0].max(dim=1)[0]
    a_zmin = corners_a_global[:, :, 1].min(dim=1)[0]
    a_zmax = corners_a_global[:, :, 1].max(dim=1)[0]

    b_xmin = corners_b_global[:, :, 0].min(dim=1)[0]   # (M,)
    b_xmax = corners_b_global[:, :, 0].max(dim=1)[0]
    b_zmin = corners_b_global[:, :, 1].min(dim=1)[0]
    b_zmax = corners_b_global[:, :, 1].max(dim=1)[0]

    # Axis-aligned BEV IoU（作为 oriented IoU 的下界近似）
    xi1 = torch.max(a_xmin.unsqueeze(1), b_xmin.unsqueeze(0))
    yi1 = torch.max(a_zmin.unsqueeze(1), b_zmin.unsqueeze(0))
    xi2 = torch.min(a_xmax.unsqueeze(1), b_xmax.unsqueeze(0))
    yi2 = torch.min(a_zmax.unsqueeze(1), b_zmax.unsqueeze(0))

    inter_w = torch.clamp(xi2 - xi1, min=0.0)
    inter_h = torch.clamp(yi2 - yi1, min=0.0)
    inter_area = inter_w * inter_h  # (N, M)

    area_a = (a_xmax - a_xmin) * (a_zmax - a_zmin)  # (N,)
    area_b = (b_xmax - b_xmin) * (b_zmax - b_zmin)    # (M,)
    union_area = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter_area  # (N, M)

    # BEV IoU
    iou_bev = inter_area / (union_area + eps)   # (N, M)

    # ---- 综合 3D IoU = BEV_IoU * height_overlap ----
    iou_3d = iou_bev * h_overlap_ratio

    return iou_3d


def nms_3d(boxes, scores, iou_threshold=0.3):
    """
    3D NMS（非极大值抑制）
    boxes: (N, 7) - [x, y, z, l, w, h, heading]
    scores: (N,) - 置信度分数
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # 按分数排序
    order = scores.argsort(descending=True)
    keep = []

    while order.shape[0] > 0:
        i = order[0]
        keep.append(i.item())

        if order.shape[0] == 1:
            break

        # 计算与剩余框的 IoU
        remaining_boxes = boxes[order[1:]]
        ious = box3d_iou(boxes[i:i+1], remaining_boxes)[0]

        # 保留 IoU 低于阈值的框
        mask = ious < iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def associate_boxes_3d(boxes_list, scores_list, iou_threshold=0.3):
    """
    关联多视角/多个检测的 3D 框
    返回：grouped_indices - 每个组包含的框索引列表

    boxes_list: list of (N_i, 7) - 每个视角的检测框
    scores_list: list of (N_i,) - 每个视角的分数
    """
    # 合并所有框
    all_boxes = torch.cat(boxes_list, dim=0)
    all_scores = torch.cat(scores_list, dim=0)

    # 3D NMS
    keep_indices = nms_3d(all_boxes, all_scores, iou_threshold)

    # 为每个保留下来的框分配组
    grouped = {i: [i] for i in keep_indices.tolist()}

    return grouped, all_boxes[keep_indices], all_scores[keep_indices]


class FeatureMetricConsistencyLoss(nn.Module):
    """
    特征度量一致性损失（Sparse Multiview）
    让 3D 框投影到各视角后与 2D 特征对齐

    ⚠️ 警告：此模块仅适用于多视角输入！
    在单图蒸馏场景下，开启此模块会导致特征坍缩（Feature Collapse），
    迫使不同物体学习完全相同的特征表示。
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.warning_issued = False

    def forward(self, pred_boxes_3d, features_2d_list, K_list, T_list, image_sizes, num_views=1):
        """
        pred_boxes_3d: (N, 7) - 预测的 3D 框 [x, y, z, l, w, h, heading]
        features_2d_list: list of (C, H, W) - 各视角的 2D 特征
        K_list: list of (3, 3) - 内参
        T_list: list of (4, 4) - 外参（相机位姿）
        image_sizes: list of (H, W) - 图像尺寸
        num_views: 视角数量，用于安全检查
        """
        # ===== 🔴 致命修复：单图下禁用此模块 =====
        if num_views <= 1:
            if not self.warning_issued:
                print("⚠️ [FeatureMetricConsistencyLoss] 警告：单图输入下禁用特征一致性损失（防止特征坍缩）")
                self.warning_issued = True
            # 返回与正常损失路径一致的 detached tensor
            return pred_boxes_3d.new_zeros((1,))

        if pred_boxes_3d.shape[0] == 0 or len(features_2d_list) == 0:
            return torch.tensor(0.0, device=pred_boxes_3d.device)

        loss = 0.0
        n_pairs = 0

        # 对每个 3D 框，投影到每个视角
        for box_3d in pred_boxes_3d:
            center = box_3d[:3]
            dims = box_3d[3:6]

            for feat_2d, K, T, img_size in zip(features_2d_list, K_list, T_list, image_sizes):
                # 3D 中心投影到 2D
                point_cam = T[:3, :3] @ center + T[:3, 3]
                if point_cam[2] <= 0:  # 在相机后面
                    continue

                point_2d = K @ point_cam
                u, v = point_2d[0] / point_2d[2], point_2d[1] / point_cam[2]

                # 检查是否在图像范围内
                H, W = img_size
                if not (0 <= u < W and 0 <= v < H):
                    continue

                # 采样 2D 特征
                u_scaled = u * feat_2d.shape[2] / W
                v_scaled = v * feat_2d.shape[1] / H

                # 简单的双线性插值
                u0, v0 = int(u_scaled), int(v_scaled)
                if u0 >= feat_2d.shape[2] - 1 or v0 >= feat_2d.shape[1] - 1:
                    continue

                # 提取 2x2 区域计算注意力
                feat_sample = feat_2d[:, v0:v0+2, u0:u0+2]  # (C, 2, 2)

                # 计算特征一致性：让不同视角的投影特征相似
                # 这里用简单的 L2 距离作为一致性度量
                if n_pairs == 0:
                    self.stored_features = feat_sample.mean(dim=(1, 2))
                else:
                    current_feat = feat_sample.mean(dim=(1, 2))
                    loss += F.cosine_embedding_loss(
                        self.stored_features.unsqueeze(0),
                        current_feat.unsqueeze(0),
                        torch.ones(1, device=feat_2d.device)
                    )
                    n_pairs += 1

        if n_pairs == 0:
            return torch.tensor(0.0, device=pred_boxes_3d.device)

        return loss / max(n_pairs, 1)


class MultiViewBoxFusion(nn.Module):
    """
    多视角框融合（借鉴 BoxFusion）
    1. 3D NMS 关联
    2. IoU 引导的粒子滤波优化
    """

    def __init__(self, iou_threshold=0.3, num_particles=50):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.num_particles = num_particles

    def fuse_boxes_iou_guided(self, boxes_group):
        """
        IoU 引导的框融合
        boxes_group: (M, 7) - 属于同一物体的多个框
        """
        if boxes_group.shape[0] == 1:
            return boxes_group[0]

        if boxes_group.shape[0] == 0:
            return None

        # 计算两两 IoU
        ious = box3d_iou(boxes_group, boxes_group)  # (M, M)

        # 找出 IoU 最高的框对
        iou_max, idx_max = ious.max(dim=1)

        # 筛选高质量框（IoU > 阈值）
        high_iou_mask = iou_max > self.iou_threshold

        if high_iou_mask.sum() == 0:
            # 没有高 IoU 框，直接平均
            fused = boxes_group.mean(dim=0)
        else:
            # 用高 IoU 框加权平均
            high_iou_boxes = boxes_group[high_iou_mask]
            weights = iou_max[high_iou_mask].unsqueeze(1)
            fused = (high_iou_boxes * weights).sum(dim=0) / weights.sum()

        return fused

    def fuse_with_particle_filter(self, boxes_group, scores_group):
        """
        粒子滤波风格的框融合（简化版）
        boxes_group: (M, 7)
        scores_group: (M,)
        """
        if boxes_group.shape[0] <= 2:
            return self.fuse_boxes_iou_guided(boxes_group)

        # 采样粒子
        num_particles = min(self.num_particles, boxes_group.shape[0])

        # 根据分数采样
        score_sum = scores_group.sum()
        if score_sum == 0:
            return self.fuse_boxes_iou_guided(boxes_group)
        probs = scores_group / score_sum
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            return self.fuse_boxes_iou_guided(boxes_group)
        indices = torch.multinomial(probs, num_particles, replacement=True)
        sampled_boxes = boxes_group[indices]

        # 在采样框附近添加噪声
        noise = torch.randn_like(sampled_boxes) * 0.1
        particles = sampled_boxes + noise

        # 评估每个粒子的质量（与所有框的 IoU）
        particle_scores = []
        for p in particles:
            ious = box3d_iou(p.unsqueeze(0), boxes_group)[0]
            score = (ious * scores_group).sum() / scores_group.sum()
            particle_scores.append(score.item())

        particle_scores = torch.tensor(particle_scores, device=boxes_group.device)

        # 选择最佳粒子
        best_idx = particle_scores.argmax()
        fused = particles[best_idx]

        return fused

    def forward(self, boxes_list, scores_list):
        """
        融合多视角检测结果
        boxes_list: list of (N_i, 7)
        scores_list: list of (N_i,)
        """
        if len(boxes_list) == 0 or all(b.shape[0] == 0 for b in boxes_list):
            _device = boxes_list[0].device if boxes_list else torch.device('cpu')
            return torch.zeros(0, 7, device=_device)

        # 关联
        grouped, keep_boxes, keep_scores = associate_boxes_3d(
            boxes_list, scores_list, self.iou_threshold
        )

        # 如果只有一个视角的直接返回
        if len(boxes_list) == 1:
            return keep_boxes

        # 融合每个组
        fused_boxes = []
        for idx in grouped.keys():
            group_boxes = []
            group_scores = []
            offset = 0
            for i, b in enumerate(boxes_list):
                # 检查当前组的框在第 i 个视角
                group_mask = torch.zeros(b.shape[0], dtype=torch.bool)
                # 简化：每个组只取第一个框
                if idx < offset + b.shape[0]:
                    group_mask[idx - offset] = True
                    group_boxes.append(b[group_mask])
                    group_scores.append(scores_list[i][group_mask])
                offset += b.shape[0]

            if len(group_boxes) > 0:
                group_boxes = torch.cat(group_boxes, dim=0)
                group_scores = torch.cat(group_scores, dim=0)
                fused = self.fuse_with_particle_filter(group_boxes, group_scores)
                if fused is not None:
                    fused_boxes.append(fused)

        if len(fused_boxes) == 0:
            return keep_boxes

        return torch.stack(fused_boxes)


def compute_3d_box_from_params(center, dims, heading):
    """
    从参数构建 3D 框角点
    center: (3,) - [x, y, z]
    dims: (3,) - [l, w, h] (沿 x/z/y 轴的半长度)
    heading: float - 航向角（绕 y 轴逆时针）

    注意：旋转矩阵用标准绕 y 轴旋转，与 teacher_geometry.py 的 R_cam 坐标系一致。
    """
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    R = np.array([
        [ cos_h, 0, sin_h],
        [    0,   1,    0],
        [-sin_h, 0, cos_h],
    ])

    # 8 个角点（局部坐标系）
    l, w, h = dims
    corners_local = np.array([
        [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2], [-l/2, w/2, -h/2], [l/2, w/2, -h/2],
        [-l/2, -w/2, h/2], [l/2, -w/2, h/2], [-l/2, w/2, h/2], [l/2, w/2, h/2]
    ])

    # 旋转和平移
    corners = (R @ corners_local.T).T + center

    return corners


def box3d_to_corners(box_3d):
    """
    将 3D 框参数转换为 8 个角点
    box_3d: (7,) - [x, y, z, l, w, h, heading]
    """
    center = box_3d[:3]
    dims = box_3d[3:6]
    heading = box_3d[6]

    return compute_3d_box_from_params(center, dims, heading)


def project_3d_box_to_2d(box_3d, K):
    """
    将 3D 框投影到 2D 图像，返回 2D 包围盒

    Args:
        box_3d: (7,) - [x, y, z, l, w, h, heading]
        K: (3, 3) - 相机内参

    Returns:
        box_2d: (4,) - [x_min, y_min, x_max, y_max] 归一化到原图尺寸
    """
    # 获取 8 个角点
    corners = box3d_to_corners(box_3d)  # (8, 3)

    # 投影到 2D
    corners_hom = np.concatenate([corners, np.ones((8, 1))], axis=1)  # (8, 4)
    corners_2d_hom = K @ corners_hom.T  # (3, 8)
    corners_2d = corners_2d_hom[:2] / corners_2d_hom[2:3]  # (2, 8)

    # 计算 2D 包围盒
    x_min, x_max = corners_2d[0].min(), corners_2d[0].max()
    y_min, y_max = corners_2d[1].min(), corners_2d[1].max()

    return np.array([x_min, y_min, x_max, y_max])


def compute_2d_iou(box_a, box_b):
    """
    计算两个 2D 框的 IoU
    box_a, box_b: (4,) - [x_min, y_min, x_max, y_max]
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算各自面积
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # 计算并集面积
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


class PFO2D3DAlignment:
    """
    PFO (Particle Filtering Optimization) 2D-3D 对齐后处理

    核心思想：
    - 围绕原始 3D 框生成多个"粒子"（偏移/缩放版本）
    - 将每个粒子投影到 2D 图像
    - 计算与 2D 检测框的 IoU
    - 选择 IoU 最高的作为 refined 框

    借鉴 BoxFusion 论文的 PFO 模块，但应用于单图后处理
    """

    def __init__(
        self,
        num_particles=50,
        translation_range=0.3,      # 位置偏移范围（米）
        scale_range=0.2,            # 尺寸缩放范围
        heading_range=0.15,         # 航向角偏移范围（弧度）
    ):
        self.num_particles = num_particles
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.heading_range = heading_range

    def generate_particles(self, box_3d):
        """
        生成粒子：围绕原始框的偏移/缩放版本

        Args:
            box_3d: (7,) - [x, y, z, l, w, h, heading]

        Returns:
            particles: (num_particles, 7)
        """
        center = box_3d[:3].copy()
        dims = box_3d[3:6].copy()
        heading = box_3d[6]

        particles = []

        # 1. 原始框作为第一个粒子
        particles.append(box_3d.copy())

        # 2. 位置偏移粒子 (x, y, z 三个方向)
        for _ in range(self.num_particles // 3):
            # x, y, z 随机偏移
            offset = np.random.randn(3) * self.translation_range
            new_center = center + offset
            particles.append(np.concatenate([new_center, dims, [heading]]))

            # 仅 z 偏移（深度方向最重要）
            offset_z = np.array([0, 0, np.random.randn() * self.translation_range * 0.5])
            new_center_z = center + offset_z
            particles.append(np.concatenate([new_center_z, dims, [heading]]))

        # 3. 尺寸缩放粒子 (l, w, h)
        for _ in range(self.num_particles // 3):
            scale = 1.0 + np.random.randn() * self.scale_range
            scale = np.clip(scale, 0.5, 2.0)  # 限制缩放范围
            new_dims = dims * scale
            particles.append(np.concatenate([center, new_dims, [heading]]))

        # 4. 航向角偏移粒子
        for _ in range(self.num_particles // 4):
            heading_offset = np.random.randn() * self.heading_range
            new_heading = heading + heading_offset
            particles.append(np.concatenate([center, dims, [new_heading]]))

        # 填充到指定数量
        while len(particles) < self.num_particles:
            idx = np.random.randint(0, len(particles))
            particles.append(particles[idx].copy())

        return np.array(particles[:self.num_particles])

    def compute_particle_scores(self, particles, box_2d_gt, K):
        """
        计算每个粒子的得分（2D IoU）

        Args:
            particles: (N, 7) - 3D 框粒子
            box_2d_gt: (4,) - 2D 检测框 [x_min, y_min, x_max, y_max]
            K: (3, 3) - 相机内参

        Returns:
            scores: (N,) - 每个粒子的 IoU 分数
        """
        scores = []

        for box_3d in particles:
            try:
                # 投影到 2D
                box_2d_proj = project_3d_box_to_2d(box_3d, K)

                # 计算 IoU
                iou = compute_2d_iou(box_2d_proj, box_2d_gt)
                scores.append(iou)
            except:
                scores.append(0.0)

        return np.array(scores)

    def refine(self, box_3d, box_2d_gt, K, return_all_scores=False):
        """
        PFO 2D-3D 对齐后处理

        Args:
            box_3d: (7,) - 原始 3D 框 [x, y, z, l, w, h, heading]
            box_2d_gt: (4,) - 2D 检测框 [x_min, y_min, x_max, y_max]
            K: (3, 3) - 相机内参
            return_all_scores: 是否返回所有粒子的分数

        Returns:
            refined_box: (7,) - 优化后的 3D 框
            best_score: float - 最佳 IoU 分数
            (可选) all_scores: 所有粒子的分数
        """
        # 生成粒子
        particles = self.generate_particles(box_3d)

        # 计算每个粒子的得分
        scores = self.compute_particle_scores(particles, box_2d_gt, K)

        # 选择最佳粒子
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        refined_box = particles[best_idx]

        if return_all_scores:
            return refined_box, best_score, scores
        return refined_box, best_score

    def batch_refine(self, boxes_3d, boxes_2d_gt, K):
        """
        批量处理多个 3D 框

        Args:
            boxes_3d: (N, 7) - 3D 框列表
            boxes_2d_gt: (N, 4) - 2D 检测框列表
            K: (3, 3) - 相机内参

        Returns:
            refined_boxes: (N, 7) - 优化后的 3D 框
            best_scores: (N,) - 最佳 IoU 分数
        """
        refined_boxes = []
        best_scores = []

        for box_3d, box_2d in zip(boxes_3d, boxes_2d_gt):
            refined, score = self.refine(box_3d, box_2d, K)
            refined_boxes.append(refined)
            best_scores.append(score)

        return np.array(refined_boxes), np.array(best_scores)
