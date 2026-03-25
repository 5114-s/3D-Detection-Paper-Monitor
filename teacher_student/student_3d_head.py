# Copyright (c) Teacher-Student Distillation Pipeline
"""
学生：类别无关 3D 头，对齐 OVMono3D/CubeRCNN 的 LIFT 架构

网络结构（对齐 OVMono3D）：
1. RoI 特征 -> (Δu, Δv) 2D 中心偏移
2. RoI 特征 -> z 深度
3. RoI 特征 -> (W, H, L) 尺寸
4. RoI 特征 -> pose 6D 姿态
5. LIFT 反投影: cube_z * (cube_xy - cx) / fx -> 3D 中心

损失函数（对齐 OVMono3D）：
1. 2D 偏移损失 (L1 on Δu, Δv)
2. Disentangled 损失 - 逐组件角点损失 (z, xy, dims, pose)
3. Joint 损失 - LIFT 反投影后的联合角点损失
4. Chamfer 损失 - 角点级姿态监督
5. 深度逆加权 - 近距离物体权重更高
6. Uncertainty 加权 - 预测不确定性并加权损失
7. Log 变换 - 深度和尺寸归一化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import roi_align
from pytorch3d.transforms.so3 import so3_relative_angle


def rotation_6d_to_matrix(d6):
    """6D 旋转表示 -> 3x3 矩阵 (batch, 6) -> (batch, 3, 3)"""
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def compute_corners_from_box(center, dims, R_cam):
    """
    从 center, dims, R_cam 计算 8 个角点 (N, 8, 3)
    center: (N, 3) - 在相机坐标系下的中心 [x, y, z]
    dims: (N, 3) - 尺寸 [l, w, h]
    R_cam: (N, 3, 3) - 旋转矩阵
    """
    N = center.shape[0]
    if N == 0:
        return torch.zeros(0, 8, 3, device=center.device)

    l = dims[:, 0:1]
    w = dims[:, 1:2]
    h = dims[:, 2:3]
    half = torch.cat([l, h, w], dim=1) / 2.0

    offsets = torch.tensor([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
    ], dtype=torch.float32, device=center.device)

    corners_local = offsets.unsqueeze(0) * half.unsqueeze(1)
    corners = torch.einsum('nij,nkj->nki', R_cam, corners_local) + center.unsqueeze(1)

    return corners


def lift_project_to_3d(cube_x, cube_y, cube_z, K):
    """
    LIFT 风格反投影：2D 偏移 + 深度 + K -> 3D 相机坐标
    对齐 OVMono3D roi_heads.py 第 802-804 行

    Args:
        cube_x: (N,) 预测的 2D x 坐标（像素，已应用 Δu 偏移）
        cube_y: (N,) 预测的 2D y 坐标（像素，已应用 Δv 偏移）
        cube_z: (N, 1) 预测的深度（米）
        K: (3, 3) 或 (N, 3, 3) 相机内参

    Returns:
        center_3d: (N, 3) 相机坐标系下的 3D 中心 [x, y, z]
    """
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(cube_z.shape[0], -1, -1)

    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]

    x3d = cube_z * (cube_x - cx) / fx
    y3d = cube_z * (cube_y - cy) / fy

    center_3d = torch.stack([x3d, y3d, cube_z.squeeze(-1)], dim=-1)

    return center_3d


def get_cuboid_verts_faces(box3d, R):
    """
    根据 3D 框参数计算 8 个角点（来自 cubercnn.util.math_util）
    box3d: (N, 6) [cx, cy, cz, W, H, L] 或 (N, 7) [cx, cy, cz, W, H, L, heading]
    R: (N, 3, 3) 旋转矩阵
    Returns: (N, 8, 3) 角点坐标
    """
    N = box3d.shape[0]
    if N == 0:
        return torch.zeros(0, 8, 3, device=box3d.device)

    cx = box3d[:, 0]
    cy = box3d[:, 1]
    cz = box3d[:, 2]
    W = box3d[:, 3]
    H = box3d[:, 4]
    L = box3d[:, 5]

    x_corners = [-L / 2, L / 2, L / 2, -L / 2, -L / 2, L / 2, L / 2, -L / 2]
    y_corners = [-H / 2, -H / 2, H / 2, H / 2, -H / 2, -H / 2, H / 2, H / 2]
    z_corners = [-W / 2, -W / 2, -W / 2, -W / 2, W / 2, W / 2, W / 2, W / 2]

    corners = torch.stack([
        torch.tensor(x_corners, device=box3d.device),
        torch.tensor(y_corners, device=box3d.device),
        torch.tensor(z_corners, device=box3d.device),
    ], dim=1)

    corners_rot = torch.einsum('nij,jk->nik', corners.unsqueeze(0).expand(N, -1, -1), R.permute(0, 2, 1))

    corners_rot = corners_rot.permute(0, 2, 1)

    center = torch.stack([cx, cy, cz], dim=1).unsqueeze(1).expand(-1, 8, -1)

    corners_final = corners_rot + center

    return corners_final


class Student3DHead(nn.Module):
    """
    LIFT 风格 3D 检测头（对齐 OVMono3D/CubeRCNN）

    网络输出（不直接预测 3D 中心）：
    - bbox_2d_deltas: (N, 2) 2D 框中心的偏移量 (Δu, Δv)
    - bbox_z: (N, 1) 深度
    - bbox_dims: (N, 3) 尺寸 (W, H, L)
    - bbox_pose: (N, 6) 姿态 6D

    LIFT 反投影得到 3D 中心：
    - cube_x = src_ctr_x + src_widths * Δu
    - cube_y = src_ctr_y + src_heights * Δv
    - x3d = cube_z * (cube_x - cx) / fx
    - y3d = cube_z * (cube_y - cy) / fy
    - center_3d = (x3d, y3d, cube_z)

    支持完全解耦分支（shared_fc=False，与 OVMono3D 一致）
    """

    def __init__(self, in_channels=256, roi_size=7, hidden_dim=256, num_fc=2,
                 use_uncertainty=True, use_disentangled=True, use_chamfer=True,
                 use_log_dims=True, use_joint_loss=True, use_inverse_z_weight=True,
                 z_type='log', shared_fc=True):
        super().__init__()
        self.in_channels = in_channels
        self.roi_size = roi_size
        self.feat_dim = in_channels * roi_size * roi_size

        self.use_uncertainty = use_uncertainty
        self.use_disentangled = use_disentangled
        self.use_chamfer = use_chamfer
        self.use_log_dims = use_log_dims
        self.use_joint_loss = use_joint_loss
        self.use_inverse_z_weight = use_inverse_z_weight
        self.z_type = z_type
        self.shared_fc = shared_fc

        if shared_fc:
            layers = []
            d = self.feat_dim
            for _ in range(num_fc):
                layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
                d = hidden_dim
            self.feature_generator = nn.Sequential(*layers)
            self._out_dim = d
        else:
            self.feature_generator_XY = self._build_fc_branch()
            self.feature_generator_Z = self._build_fc_branch()
            self.feature_generator_dims = self._build_fc_branch()
            self.feature_generator_pose = self._build_fc_branch()
            if use_uncertainty:
                self.feature_generator_conf = self._build_fc_branch()

        self._output_size = hidden_dim

        self.bbox_2d_deltas = nn.Linear(self._out_dim, 2)
        self.bbox_dims = nn.Linear(self._out_dim, 3)
        self.bbox_pose = nn.Linear(self._out_dim, 6)
        self.bbox_z = nn.Linear(self._out_dim, 1)

        nn.init.normal_(self.bbox_2d_deltas.weight, std=0.001)
        nn.init.normal_(self.bbox_dims.weight, std=0.001)
        nn.init.normal_(self.bbox_pose.weight, std=0.001)
        nn.init.normal_(self.bbox_z.weight, std=0.001)
        nn.init.constant_(self.bbox_2d_deltas.bias, 0)
        nn.init.constant_(self.bbox_dims.bias, 0)
        nn.init.constant_(self.bbox_pose.bias, 0)
        nn.init.constant_(self.bbox_z.bias, 0)

        if self.use_uncertainty:
            self.bbox_uncertainty = nn.Linear(self._out_dim, 1)
            nn.init.normal_(self.bbox_uncertainty.weight, std=0.001)
            nn.init.constant_(self.bbox_uncertainty.bias, 5)

        with torch.no_grad():
            self.bbox_pose.weight.fill_(0.0)
            self.bbox_pose.weight[0, :] = 0.1
            self.bbox_pose.weight[4, :] = 0.1

    def _build_fc_branch(self):
        layers = []
        d = self.feat_dim
        for _ in range(2):
            layers += [nn.Linear(d, self._output_size), nn.ReLU(inplace=True)]
            d = self._output_size
        return nn.Sequential(*layers)

    def forward(self, features, box_xyxy_norm, image_size_hw, K=None):
        """
        Args:
            features: (B, C, H, W) 特征图
            box_xyxy_norm: (N, 4) 归一化 [0,1] 的 xyxy 坐标
            image_size_hw: (H, W) 特征图对应原图尺寸
            K: (3, 3) 相机内参（用于 LIFT 反投影，但反投影在损失函数中进行）

        Returns:
            如果 shared_fc=True:
                deltas: (N, 2), z: (N, 1), dims: (N, 3), pose_6d: (N, 6), uncertainty: (N, 1)
            如果 shared_fc=False:
                同上，但特征来自独立分支
        """
        N = box_xyxy_norm.shape[0]
        if N == 0:
            return (
                torch.zeros(0, 2, device=features.device),
                torch.zeros(0, 1, device=features.device),
                torch.zeros(0, 3, device=features.device),
                torch.zeros(0, 6, device=features.device),
                torch.zeros(0, 1, device=features.device) if self.use_uncertainty else None,
            )

        B, C, H, W = features.shape
        scale_x = W / image_size_hw[1]
        scale_y = H / image_size_hw[0]

        boxes_xyxy_feat = box_xyxy_norm.clone()
        boxes_xyxy_feat[:, [0, 2]] *= scale_x
        boxes_xyxy_feat[:, [1, 3]] *= scale_y
        boxes_for_roi = boxes_xyxy_feat.float()

        rois = roi_align(features, [boxes_for_roi], output_size=(self.roi_size, self.roi_size),
                         spatial_scale=1.0, aligned=True)
        rois = rois.flatten(1)

        if self.shared_fc:
            feat = self.feature_generator(rois)
        else:
            feat_xy = self.feature_generator_XY(rois)
            feat_z = self.feature_generator_Z(rois)
            feat_dims = self.feature_generator_dims(rois)
            feat_pose = self.feature_generator_pose(rois)
            feat = feat_z

        deltas = self.bbox_2d_deltas(feat if self.shared_fc else feat_xy)
        z = self.bbox_z(feat if self.shared_fc else feat_z)
        dims = self.bbox_dims(feat if self.shared_fc else feat_dims)
        pose_6d = self.bbox_pose(feat if self.shared_fc else feat_pose)

        dims = F.softplus(dims) + 0.1

        if self.z_type == 'log':
            z = torch.exp(z.clamp(max=5))
        elif self.z_type == 'sigmoid':
            z = torch.sigmoid(z) * 100
        elif self.z_type == 'direct':
            z = F.softplus(z) + 0.01

        pose_R = rotation_6d_to_matrix(pose_6d)

        uncertainty = None
        if self.use_uncertainty:
            if self.shared_fc:
                uncertainty = self.bbox_uncertainty(feat).clamp(0.01)
            else:
                uncertainty = self.bbox_uncertainty(self.feature_generator_conf(rois)).clamp(0.01)

        if not self.shared_fc:
            deltas = self.bbox_2d_deltas(feat_xy)
            pose_6d = self.bbox_pose(feat_pose)
            pose_6d_raw = self.bbox_pose(feat_pose)
            pose_R = rotation_6d_to_matrix(pose_6d_raw)

        return deltas, z, dims, pose_6d, uncertainty


def compute_2d_box_info(box_xyxy_norm, image_size_hw):
    """
    从归一化 2D 框计算中心、宽高
    对齐 OVMono3D roi_heads.py 第 415-419 行

    Args:
        box_xyxy_norm: (N, 4) 归一化的 [x1, y1, x2, y2]
        image_size_hw: (H, W)

    Returns:
        src_ctr_x, src_ctr_y, src_widths, src_heights: (N,) 各维度都是像素坐标
    """
    w_img, h_img = image_size_hw[1], image_size_hw[0]

    src_widths = (box_xyxy_norm[:, 2] - box_xyxy_norm[:, 0]) * w_img
    src_heights = (box_xyxy_norm[:, 3] - box_xyxy_norm[:, 1]) * h_img
    src_ctr_x = box_xyxy_norm[:, 0] * w_img + 0.5 * src_widths
    src_ctr_y = box_xyxy_norm[:, 1] * h_img + 0.5 * src_heights

    return src_ctr_x, src_ctr_y, src_widths, src_heights


def apply_2d_deltas(deltas, src_ctr_x, src_ctr_y, src_widths, src_heights):
    """
    应用 2D 偏移预测得到 2D 投影中心
    对齐 OVMono3D roi_heads.py 第 480-481 行

    Args:
        deltas: (N, 2) 预测的 (Δu, Δv)
        src_ctr_x, src_ctr_y: (N,) 原始 2D 框中心
        src_widths, src_heights: (N,) 原始 2D 框尺寸

    Returns:
        cube_x, cube_y: (N,) 应用偏移后的 2D 坐标（像素）
    """
    cube_x = src_ctr_x + src_widths * deltas[:, 0]
    cube_y = src_ctr_y + src_heights * deltas[:, 1]
    return cube_x, cube_y


def compute_student_loss(deltas, z, dims, pose_6d, uncertainty,
                         gt_center, gt_dims, gt_R_cam,
                         box_xyxy_norm, image_size_hw, K,
                         loss_weights=None,
                         use_chamfer=True, use_joint_loss=True,
                         use_inverse_z_weight=True, use_log_dims=True,
                         z_type='log', use_disentangled_loss=True,
                         shared_fc=True):
    """
    学生损失（对齐 OVMono3D/CubeRCNN）

    关键改进（LIFT 架构）：
    - 2D 偏移损失：gt_deltas = (gt_center_xy - src_ctr_xy) / src_widths
    - Disentangled 损失：分别用 z、xy 反投影后的角点计算损失
    - Joint 损失：LIFT 反投影后的完整 3D 中心

    Args:
        deltas: (N, 2) 预测的 2D 偏移 (Δu, Δv)
        z: (N, 1) 预测的深度
        dims: (N, 3) 预测的尺寸
        pose_6d: (N, 6) 预测的 6D 姿态
        uncertainty: (N, 1) 预测的不确定性
        gt_center: (N, 3) GT 3D 中心
        gt_dims: (N, 3) GT 尺寸
        gt_R_cam: (N, 3, 3) GT 旋转矩阵
        box_xyxy_norm: (N, 4) 归一化的 2D 框
        image_size_hw: (H, W) 图像尺寸
        K: (3, 3) 相机内参
    """
    if loss_weights is None:
        loss_weights = {
            "delta_xy": 1.0, "dims": 1.0, "pose": 1.0,
            "uncertainty": 0.1, "disentangled": 1.0,
            "chamfer": 1.0, "joint": 1.0,
            "z": 1.0,
        }

    loss_dict = {}

    N = deltas.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=gt_center.device), loss_dict

    device = gt_center.device

    # Ensure K is on the same device as other tensors
    if K.dim() == 2:
        K = K.to(device).unsqueeze(0).expand(N, -1, -1)
    elif K.dim() == 3 and K.shape[0] == 1:
        K = K.to(device).expand(N, -1, -1)
    elif K.device != device:
        K = K.to(device)

    src_ctr_x, src_ctr_y, src_widths, src_heights = compute_2d_box_info(box_xyxy_norm, image_size_hw)
    cube_x, cube_y = apply_2d_deltas(deltas, src_ctr_x, src_ctr_y, src_widths, src_heights)

    z_vals = z.squeeze(-1)
    gt_z = gt_center[:, 2]
    gt_xy = gt_center[:, :2]

    E_CONSTANT = 2.71828
    if use_inverse_z_weight and gt_z.numel() > 0:
        inverse_z_w = 1.0 / torch.log(gt_z.clip(E_CONSTANT))
    else:
        inverse_z_w = torch.ones(N, device=device)

    cube_z = z_vals.clamp(min=0.1)

    center_3d = lift_project_to_3d(cube_x, cube_y, z, K)

    if use_log_dims:
        dims_log = torch.log(dims)
        gt_dims_log = torch.log(gt_dims)
        loss_dims = F.l1_loss(dims_log, gt_dims_log, reduction="none")
    else:
        loss_dims = F.l1_loss(dims, gt_dims, reduction="none")
    loss_dims = (loss_dims.mean(dim=1) * inverse_z_w).mean()

    R_pred = rotation_6d_to_matrix(pose_6d)
    loss_pose = 1.0 - so3_relative_angle(R_pred, gt_R_cam, eps=0.1, cos_angle=True)
    loss_pose = (loss_pose * inverse_z_w).mean()

    loss_dict["loss_dims"] = loss_dims.item()
    loss_dict["loss_pose"] = loss_pose.item()

    loss = loss_weights["dims"] * loss_dims + loss_weights["pose"] * loss_pose

    gt_deltas_x = (gt_center[:, 0] - src_ctr_x) / (src_widths + 1e-6)
    gt_deltas_y = (gt_center[:, 1] - src_ctr_y) / (src_heights + 1e-6)
    gt_deltas = torch.stack([gt_deltas_x, gt_deltas_y], dim=-1)

    loss_delta_xy = F.l1_loss(deltas, gt_deltas, reduction="none")
    loss_delta_xy = (loss_delta_xy.mean(dim=1) * inverse_z_w).mean()
    loss += loss_weights["delta_xy"] * loss_delta_xy
    loss_dict["loss_delta_xy"] = loss_delta_xy.item()

    if z_type == 'log':
        z_norm = torch.log(cube_z)
        gt_z_norm = torch.log(gt_z.clamp(min=0.01))
        loss_z = F.l1_loss(z_norm, gt_z_norm, reduction="none")
    elif z_type == 'sigmoid':
        z_norm = torch.sigmoid(z * 0.01)
        gt_z_norm = torch.sigmoid(gt_z.unsqueeze(-1) * 0.01)
        loss_z = F.l1_loss(z_norm, gt_z_norm, reduction="none")
    else:
        loss_z = F.l1_loss(z, gt_z.unsqueeze(-1), reduction="none")
    loss_z = (loss_z.squeeze(-1) * inverse_z_w).mean()
    loss += loss_weights["z"] * loss_z
    loss_dict["loss_z"] = loss_z.item()

    if use_joint_loss:
        pred_corners_joint = compute_corners_from_box(center_3d, dims, R_pred)
        gt_corners = compute_corners_from_box(gt_center, gt_dims, gt_R_cam)

        if pred_corners_joint.shape[0] > 0:
            loss_joint = F.l1_loss(pred_corners_joint, gt_corners, reduction="none")
            loss_joint = loss_joint.contiguous().view(N, -1).mean(dim=1)
            loss_joint = (loss_joint * inverse_z_w).mean()

            loss += loss_weights["joint"] * loss_joint
            loss_dict["loss_joint"] = loss_joint.item()

    if use_chamfer:
        pred_corners = compute_corners_from_box(center_3d, dims, R_pred)
        gt_corners = compute_corners_from_box(gt_center, gt_dims, gt_R_cam)

        if pred_corners.shape[0] > 0:
            pred_corners_exp = pred_corners.unsqueeze(2)
            gt_corners_exp = gt_corners.unsqueeze(1)

            dist_pred_to_gt = (pred_corners_exp - gt_corners_exp).pow(2).sum(-1)
            dist_pred_to_gt_min, _ = dist_pred_to_gt.min(dim=2)

            dist_gt_to_pred = (gt_corners_exp - pred_corners_exp).pow(2).sum(-1)
            dist_gt_to_pred_min, _ = dist_gt_to_pred.min(dim=1)

            chamfer_dist = (dist_pred_to_gt_min.mean(dim=1) + dist_gt_to_pred_min.mean(dim=1)).mean()
            loss += loss_weights["chamfer"] * chamfer_dist
            loss_dict["loss_chamfer"] = chamfer_dist.item()

    if use_disentangled_loss:
        eps_fx = 1e-6
        eps_fy = 1e-6
        fx_safe = torch.where(K[:, 0, 0].abs() > eps_fx, K[:, 0, 0], torch.ones_like(K[:, 0, 0]) * eps_fx)
        fy_safe = torch.where(K[:, 1, 1].abs() > eps_fy, K[:, 1, 1], torch.ones_like(K[:, 1, 1]) * eps_fy)
        gt_x3d = gt_z * (gt_center[:, 0] - K[:, 0, 2]) / fx_safe
        gt_y3d = gt_z * (gt_center[:, 1] - K[:, 1, 2]) / fy_safe

        cube_dis_x3d_from_z = cube_z * (cube_x - K[:, 0, 2]) / fx_safe
        cube_dis_y3d_from_z = cube_z * (cube_y - K[:, 1, 2]) / fy_safe
        cube_dis_z = torch.stack([cube_dis_x3d_from_z, cube_dis_y3d_from_z, cube_z], dim=-1)

        box3d_dis_z = torch.cat([cube_dis_z, dims], dim=-1)
        dis_z_corners = get_cuboid_verts_faces(box3d_dis_z, R_pred)
        loss_z_dis = F.l1_loss(dis_z_corners, gt_corners, reduction="none")
        loss_z_dis = loss_z_dis.contiguous().view(N, -1).mean(dim=1)
        loss_z_dis = (loss_z_dis * inverse_z_w).mean()

        cube_dis_x3d = gt_z * (cube_x - K[:, 0, 2]) / fx_safe
        cube_dis_y3d = gt_z * (cube_y - K[:, 1, 2]) / fy_safe
        cube_dis_xy = torch.stack([cube_dis_x3d, cube_dis_y3d, gt_z], dim=-1)

        box3d_dis_xy = torch.cat([cube_dis_xy, gt_dims], dim=-1)
        dis_xy_corners = get_cuboid_verts_faces(box3d_dis_xy, gt_R_cam)
        loss_xy_dis = F.l1_loss(dis_xy_corners, gt_corners, reduction="none")
        loss_xy_dis = loss_xy_dis.contiguous().view(N, -1).mean(dim=1)
        loss_xy_dis = (loss_xy_dis * inverse_z_w).mean()

        box3d_dis_dims = torch.cat([gt_center, dims], dim=-1)
        dis_dims_corners = get_cuboid_verts_faces(box3d_dis_dims, gt_R_cam)
        loss_dims_dis = F.l1_loss(dis_dims_corners, gt_corners, reduction="none")
        loss_dims_dis = loss_dims_dis.contiguous().view(N, -1).mean(dim=1)
        loss_dims_dis = (loss_dims_dis * inverse_z_w).mean()

        box3d_dis_pose = torch.cat([gt_center, gt_dims], dim=-1)
        dis_pose_corners = get_cuboid_verts_faces(box3d_dis_pose, R_pred)
        loss_pose_dis = F.l1_loss(dis_pose_corners, gt_corners, reduction="none")
        loss_pose_dis = loss_pose_dis.contiguous().view(N, -1).mean(dim=1)
        loss_pose_dis = (loss_pose_dis * inverse_z_w).mean()

        loss_disentangled = loss_z_dis + loss_xy_dis + loss_dims_dis + loss_pose_dis
        loss += loss_weights["disentangled"] * loss_disentangled

        loss_dict["loss_z_dis"] = loss_z_dis.item()
        loss_dict["loss_xy_dis"] = loss_xy_dis.item()
        loss_dict["loss_dims_dis"] = loss_dims_dis.item()
        loss_dict["loss_pose_dis"] = loss_pose_dis.item()

    if uncertainty is not None and uncertainty.numel() > 0:
        uncert_sf = np.sqrt(2) * torch.exp(-uncertainty.squeeze(-1))

        loss_center_unc = F.l1_loss(center_3d, gt_center, reduction="none")
        loss_center_unc = ((loss_center_unc.mean(dim=1) * inverse_z_w) * uncert_sf).mean()

        loss_dims_unc = F.l1_loss(torch.log(dims), torch.log(gt_dims), reduction="none")
        loss_dims_unc = ((loss_dims_unc.mean(dim=1) * inverse_z_w) * uncert_sf).mean()

        loss_pose_unc = 1.0 - so3_relative_angle(R_pred, gt_R_cam, eps=0.1, cos_angle=True)
        loss_pose_unc = ((loss_pose_unc * inverse_z_w) * uncert_sf).mean()

        loss_uncertainty_reg = torch.log(uncertainty.squeeze(-1) + 0.01).mean()

        loss_uncertainty = loss_center_unc + loss_dims_unc + loss_pose_unc + loss_uncertainty_reg
        loss += loss_weights["uncertainty"] * loss_uncertainty
        loss_dict["loss_uncertainty"] = loss_uncertainty.item()

    return loss, loss_dict


def gt_R_to_pose_6d(R_cam_batch):
    """(N, 3, 3) -> (N, 6) 取前两列作为 6D"""
    if R_cam_batch.shape[0] == 0:
        return R_cam_batch.new_zeros((0, 6))
    return R_cam_batch[:, :, :2].reshape(R_cam_batch.shape[0], -1)


def lift_project_inference(deltas, z, dims, pose_6d, box_xyxy_norm, image_size_hw, K):
    """
    LIFT 推理：将网络输出反投影为 3D 框
    对齐 OVMono3D roi_heads.py 第 801-804 行

    Args:
        deltas: (N, 2) 2D 偏移
        z: (N, 1) 深度
        dims: (N, 3) 尺寸
        pose_6d: (N, 6) 6D 姿态
        box_xyxy_norm: (N, 4) 归一化 2D 框
        image_size_hw: (H, W)
        K: (3, 3) 相机内参

    Returns:
        center_3d: (N, 3) 3D 中心
        dims: (N, 3) 尺寸
        R_cam: (N, 3, 3) 旋转矩阵
    """
    src_ctr_x, src_ctr_y, src_widths, src_heights = compute_2d_box_info(box_xyxy_norm, image_size_hw)
    cube_x, cube_y = apply_2d_deltas(deltas, src_ctr_x, src_ctr_y, src_widths, src_heights)

    center_3d = lift_project_to_3d(cube_x, cube_y, z, K)

    R_cam = rotation_6d_to_matrix(pose_6d)

    return center_3d, dims, R_cam
