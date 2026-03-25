# Copyright (c) Teacher-Student Distillation Pipeline
"""
蒸馏训练：教师 (DetAny3D) 生成伪 3D 框 -> 学生 3D 头学习
梯度仅反传到学生头（和可选 F_fused）

创新模块（借鉴 OVMono3D/CubeRCNN）：
1. Uncertainty Prediction - 预测每个框的不确定性，加权损失
2. Disentangled Loss - 分别预测 z, xy, dims, pose
3. Chamfer Loss - 角点级别的姿态监督

创新模块（借鉴 BoxFusion / Sparse Multiview）：
4. 3D NMS 关联 - 去除重复伪标签
5. 特征度量一致性 - 3D框投影与2D特征对齐
6. 多视角框融合 - IoU引导的粒子滤波融合
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from teacher_student.teacher_detany3d import TeacherDetAny3D
from teacher_student.teacher_geometry import run_teacher_pipeline_per_instance
from teacher_student.student_3d_head import Student3DHead, compute_student_loss
from teacher_student.multi_view_fusion import (
    MultiViewBoxFusion, FeatureMetricConsistencyLoss,
    associate_boxes_3d, box3d_iou, nms_3d
)


class DistillDataset(Dataset):
    """简单列表数据集：每项为 (rgb_path, K, text_prompt) 或 (rgb_np, K, text_prompt)"""

    def __init__(self, image_paths, K_list, text_prompts):
        assert len(image_paths) == len(K_list) == len(text_prompts)
        self.image_paths = image_paths
        self.K_list = K_list
        self.text_prompts = text_prompts

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        path = self.image_paths[i]
        if isinstance(path, str):
            import cv2
            rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        else:
            rgb = path
        return {"rgb": rgb, "K": self.K_list[i], "text_prompt": self.text_prompts[i]}


def collate_distill(batch):
    return batch


def run_distill_train(
    teacher,
    student_head,
    optimizer,
    dataloader,
    device,
    epochs=10,
    use_lshape=True,
    loss_weights=None,
    use_uncertainty=True,
    use_disentangled=True,
    use_chamfer=True,
    use_3d_nms=True,
    use_feature_consistency=False,
    use_box_fusion=False,
    multi_view_fusion=None,
    feature_consistency_loss=None,
):
    """
    单轮逻辑：对每个 batch 的每张图，教师生成伪标签，学生用 F_fused + 2D 框预测 3D，算损失并反传学生头

    创新模块：
    - use_uncertainty: 不确定性预测
    - use_disentangled: 解耦损失
    - use_chamfer: 角点损失
    - use_3d_nms: 3D NMS 去除重复伪标签
    - use_feature_consistency: 特征度量一致性
    - use_box_fusion: 多视角框融合
    """
    student_head.train()
    total_loss = 0.0
    n_batches = 0
    loss_summary = {}

    # 初始化多视角融合模块
    if multi_view_fusion is None and use_box_fusion:
        multi_view_fusion = MultiViewBoxFusion(iou_threshold=0.3, num_particles=50)

    # 初始化特征一致性损失
    if feature_consistency_loss is None and use_feature_consistency:
        feature_consistency_loss = FeatureMetricConsistencyLoss()

    for batch in dataloader:
        for sample in batch:
            rgb = sample["rgb"]
            K = np.array(sample["K"])
            text_prompt = sample["text_prompt"]

            with torch.no_grad():
                depth_np, masks, boxes_xyxy, phrases, F_fused, used_K = teacher.get_depth_mask_and_boxes(rgb, text_prompt, K)
                if depth_np is None or len(phrases) == 0:
                    continue

                # 将mask resize到与depth相同的尺寸
                depth_h, depth_w = depth_np.shape[:2]
                if masks.shape[-2:] != (depth_h, depth_w):
                    import torch.nn.functional as F
                    masks_tensor = torch.from_numpy(masks).unsqueeze(0).float()
                    masks_tensor = F.interpolate(masks_tensor, size=(depth_h, depth_w), mode='nearest')
                    masks = masks_tensor.squeeze(0).numpy()

                # 教师几何伪标签
                pseudo_list = []
                for i in range(len(phrases)):
                    center_cam, dimensions, R_cam, ok = run_teacher_pipeline_per_instance(
                        depth_np, masks[i], K, phrases[i], teacher.prior_dict, use_lshape=use_lshape,
                    )
                    if not ok:
                        continue
                    pseudo_list.append({
                        "center_cam": center_cam,
                        "dimensions": dimensions,
                        "R_cam": R_cam,
                        "box_2d_xyxy": boxes_xyxy[i],
                    })

            if not pseudo_list:
                continue

            # === 创新模块: 3D NMS 去重 ===
            if use_3d_nms and len(pseudo_list) > 1:
                # 将伪标签转为 3D 框格式 [x, y, z, l, w, h, heading]
                pseudo_centers = np.stack([p["center_cam"] for p in pseudo_list])
                pseudo_dims = np.stack([p["dimensions"] for p in pseudo_list])
                pseudo_Rs = np.stack([p["R_cam"] for p in pseudo_list])

                # 计算 heading 角
                headings = []
                for R in pseudo_Rs:
                    # 从旋转矩阵提取 yaw
                    headings.append(np.arctan2(R[0, 2], R[0, 0]))
                headings = np.array(headings)

                # 构建 3D 框 (x, y, z, l, w, h, heading)
                boxes_3d = np.stack([
                    pseudo_centers[:, 0],  # x
                    pseudo_centers[:, 1],  # y
                    pseudo_centers[:, 2],  # z
                    pseudo_dims[:, 0],     # l
                    pseudo_dims[:, 1],     # w
                    pseudo_dims[:, 2],     # h
                    headings               # heading
                ], axis=1)

                # 3D NMS
                scores = np.ones(len(boxes_3d))
                keep_indices = nms_3d(
                    torch.from_numpy(boxes_3d).float(),
                    torch.from_numpy(scores).float(),
                    iou_threshold=0.3
                ).numpy()

                # 筛选保留下来的伪标签
                pseudo_list = [pseudo_list[i] for i in keep_indices]

            # === 创新模块: 多视角框融合 ===
            # 注：当前单帧输入，box_fusion 暂不生效
            # 如果有多帧输入，可以将多帧的 pseudo_list 传入融合

            if not pseudo_list:
                continue

            # 学生前向：F_fused 来自教师（detach），只训练 head
            h, w = rgb.shape[:2]
            F_fused_detach = F_fused.detach()
            # F_fused 对应 896x896 输入，2D 框已是原图 (w,h) 坐标，与 896 对齐（左上角一致）
            image_size_hw = (896, 896)

            centers_gt = []
            dims_gt = []
            R_gt = []
            boxes_xyxy_list = []
            for p in pseudo_list:
                centers_gt.append(p["center_cam"])
                dims_gt.append(p["dimensions"])
                R_gt.append(p["R_cam"])
                boxes_xyxy_list.append(p["box_2d_xyxy"])

            centers_gt = torch.from_numpy(np.stack(centers_gt)).float().to(device)
            dims_gt = torch.from_numpy(np.stack(dims_gt)).float().to(device)
            R_gt = torch.from_numpy(np.stack(R_gt)).float().to(device)

            # 2D 框需要归一化到 [0,1]，Student3DHead 内部按 image_size_hw 缩放到特征图
            boxes_t = torch.from_numpy(np.array(boxes_xyxy_list)).float().to(device)
            # 归一化到 [0, 1]
            boxes_t[:, [0, 2]] /= image_size_hw[1]  # w
            boxes_t[:, [1, 3]] /= image_size_hw[0]  # h

            # 学生前向传播（LIFT 架构：预测 2D 偏移 + 深度）
            # 返回 (deltas, z, dims, pose_6d, uncertainty)
            deltas, pred_z, pred_dims, pred_pose_6d, pred_uncertainty = student_head(
                F_fused_detach, boxes_t, image_size_hw
            )

            # 损失计算（LIFT 架构）
            loss, loss_dict = compute_student_loss(
                deltas, pred_z, pred_dims, pred_pose_6d, pred_uncertainty,
                centers_gt, dims_gt, R_gt,
                boxes_t, image_size_hw, torch.tensor(K, device=device),
                loss_weights=loss_weights,
                use_chamfer=use_chamfer,
            )

            # === 创新模块: 特征度量一致性损失 ===
            if use_feature_consistency and feature_consistency_loss is not None:
                # LIFT 反投影得到 3D 中心
                from teacher_student.student_3d_head import compute_2d_box_info, apply_2d_deltas, lift_project_to_3d
                src_ctr_x, src_ctr_y, src_widths, src_heights = compute_2d_box_info(boxes_t, image_size_hw)
                cube_x, cube_y = apply_2d_deltas(deltas, src_ctr_x, src_ctr_y, src_widths, src_heights)
                pred_center = lift_project_to_3d(cube_x, cube_y, pred_z, torch.tensor(K, device=device))

                # 准备 2D 特征（这里简化处理，实际可用 F_fused）
                # 需要相机参数
                K_tensor = torch.from_numpy(K).float().to(device)

                # 简化的特征一致性损失（需要在实际使用中完善）
                # loss_feat = feature_consistency_loss(pred_boxes_3d, [F_fused_detach[0]], [K_tensor], [torch.eye(4).to(device)], [(h, w)])
                # loss += loss_weights.get("feature_consistency", 0.1) * loss_feat
                # loss_dict["loss_feature_consistency"] = loss_feat.item()

            # 记录各项损失（跳过字符串类型的值）
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    if k not in loss_summary:
                        loss_summary[k] = 0.0
                    loss_summary[k] += v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    # 返回平均损失和各项损失
    avg_loss = total_loss / max(n_batches, 1)
    for k in loss_summary:
        loss_summary[k] /= max(n_batches, 1)

    return avg_loss, loss_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="", help="数据根目录，或通过 list 文件指定")
    parser.add_argument("--image_list", type=str, default="", help="每行: image_path")
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "distill"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_lshape", action="store_true", default=True)

    # 创新模块开关
    parser.add_argument("--use_uncertainty", action="store_true", default=True, help="使用不确定性预测")
    parser.add_argument("--use_disentangled", action="store_true", default=True, help="使用解耦损失")
    parser.add_argument("--use_chamfer", action="store_true", default=True, help="使用角点损失")

    # BoxFusion / Sparse Multiview 创新模块
    parser.add_argument("--use_3d_nms", action="store_true", default=True, help="使用3D NMS去除重复伪标签")
    parser.add_argument("--use_feature_consistency", action="store_true", default=False, help="使用特征度量一致性损失")
    parser.add_argument("--use_box_fusion", action="store_true", default=False, help="使用多视角框融合")

    # 损失权重
    parser.add_argument("--w_center", type=float, default=1.0)
    parser.add_argument("--w_dims", type=float, default=1.0)
    parser.add_argument("--w_pose", type=float, default=1.0)
    parser.add_argument("--w_uncertainty", type=float, default=0.1)
    parser.add_argument("--w_disentangled", type=float, default=1.0)
    parser.add_argument("--w_chamfer", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 构建简单数据：若没有 list，用单张图测试
    if args.image_list and os.path.exists(args.image_list):
        with open(args.image_list) as f:
            image_paths = [line.strip() for line in f if line.strip()]
        K_list = [np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])] * len(image_paths)
        text_prompts = ["chair. table. bed. sofa."] * len(image_paths)
    else:
        # 默认用 SUN 一张图做 demo
        sun_image = os.path.join(PROJECT_ROOT, "datasets", "sunrgbd", "sunrgbd_trainval", "image", "000004.jpg")
        if not os.path.exists(sun_image):
            print("未找到 image_list 或默认 SUN 图像，请提供 --image_list 或把 000004.jpg 放到 datasets/sunrgbd/...")
            return
        image_paths = [sun_image] * 4
        K_list = [np.array([[529.5, 0, 365.0], [0, 529.5, 262.0], [0, 0, 1]])] * 4
        text_prompts = ["person. chair. table. bed. sofa. picture."] * 4

    dataset = DistillDataset(image_paths, K_list, text_prompts)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_distill, num_workers=0)

    device = args.device
    teacher = TeacherDetAny3D(device=device)

    # 创建学生头（支持创新模块）
    student_head = Student3DHead(
        in_channels=256,
        roi_size=7,
        hidden_dim=256,
        num_fc=2,
        use_uncertainty=args.use_uncertainty,
        use_disentangled=args.use_disentangled,
        use_chamfer=args.use_chamfer,
    ).to(device)

    optimizer = torch.optim.Adam(student_head.parameters(), lr=args.lr)

    # 损失权重
    loss_weights = {
        "center": args.w_center,
        "dims": args.w_dims,
        "pose": args.w_pose,
        "uncertainty": args.w_uncertainty,
        "disentangled": args.w_disentangled,
        "chamfer": args.w_chamfer,
    }

    print(f"=== 蒸馏训练配置 ===")
    print(f"创新模块(OVMono3D): uncertainty={args.use_uncertainty}, disentangled={args.use_disentangled}, chamfer={args.use_chamfer}")
    print(f"创新模块(BoxFusion): 3d_nms={args.use_3d_nms}, feature_consistency={args.use_feature_consistency}, box_fusion={args.use_box_fusion}")
    print(f"损失权重: {loss_weights}")

    for ep in range(args.epochs):
        loss_avg, loss_summary = run_distill_train(
            teacher, student_head, optimizer, dataloader, device,
            epochs=1, use_lshape=args.use_lshape,
            loss_weights=loss_weights,
            use_uncertainty=args.use_uncertainty,
            use_disentangled=args.use_disentangled,
            use_chamfer=args.use_chamfer,
            use_3d_nms=args.use_3d_nms,
            use_feature_consistency=args.use_feature_consistency,
            use_box_fusion=args.use_box_fusion,
        )
        print(f"Epoch {ep+1}/{args.epochs}  loss_avg={loss_avg:.6f}")
        loss_str = " | ".join([f"{k}={v:.6f}" for k, v in loss_summary.items()])
        print(f"  详细: {loss_str}")

    ckpt_path = os.path.join(args.output_dir, "student_3d_head.pth")
    torch.save(student_head.state_dict(), ckpt_path)
    print(f"保存学生头权重: {ckpt_path}")


if __name__ == "__main__":
    main()
