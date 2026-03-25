#!/usr/bin/env python3
"""
快速测试脚本 - 用小批量数据验证蒸馏训练流程
测试内容：
1. 数据加载
2. 教师伪标签生成
3. 学生前向传播
4. 损失计算
5. 梯度反传

使用方法:
  python tools/test_distill_pipeline.py --num_images 50
"""
import os
import sys
import cv2
import numpy as np
import torch
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from teacher_student.teacher_detany3d import TeacherDetAny3D
from teacher_student.teacher_geometry import run_teacher_pipeline_per_instance
from teacher_student.student_3d_head import Student3DHead, compute_student_loss


def load_sunrgbd_sample(dataset_root, num_images=50, start_idx=0):
    """
    从 SUNRGBD 加载少量样本用于快速测试
    """
    import json

    json_path = os.path.join(dataset_root, "Omni3D", "SUNRGBD_train.json")
    if not os.path.exists(json_path):
        print(f"找不到 JSON: {json_path}")
        return []

    with open(json_path) as f:
        data = json.load(f)

    samples = []
    for i, img_info in enumerate(data["images"][start_idx:start_idx + num_images]):
        file_path = os.path.join(dataset_root, img_info["file_path"])
        if not os.path.exists(file_path):
            continue

        samples.append({
            "path": file_path,
            "K": np.array(img_info["K"], dtype=np.float32),
            "image_id": img_info["id"],
            "width": img_info["width"],
            "height": img_info["height"],
        })

        if len(samples) >= num_images:
            break

    return samples


def test_data_loading(num_images=10):
    """测试1：数据加载"""
    print("\n" + "=" * 60)
    print("测试1: 数据加载")
    print("=" * 60)

    dataset_root = os.path.join(PROJECT_ROOT, "datasets")
    samples = load_sunrgbd_sample(dataset_root, num_images)

    print(f"  加载样本数: {len(samples)}")

    for i, s in enumerate(samples[:3]):
        print(f"  [{i}] {os.path.basename(s['path'])}")
        print(f"      K = {s['K'][0, 0]:.1f}, {s['K'][1, 1]:.1f}, {s['K'][0, 2]:.1f}, {s['K'][1, 2]:.1f}")

    return len(samples) > 0


def test_teacher_pseudo_labels(num_images=5, device="cuda"):
    """测试2：教师伪标签生成"""
    print("\n" + "=" * 60)
    print("测试2: 教师伪标签生成")
    print("=" * 60)

    dataset_root = os.path.join(PROJECT_ROOT, "datasets")
    samples = load_sunrgbd_sample(dataset_root, num_images)

    if not samples:
        print("  失败: 无法加载数据")
        return False

    # 初始化教师模型
    print("  初始化教师模型...")
    teacher = TeacherDetAny3D(device=device, use_sam2_mask=True, use_ram_gpt=False)

    success_count = 0
    total_objects = 0

    for i, s in enumerate(samples[:num_images]):
        print(f"\n  [{i+1}/{num_images}] {os.path.basename(s['path'])}")

        # 读取图像
        rgb = cv2.imread(s["path"])
        if rgb is None:
            print(f"    失败: 无法读取图像")
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # 固定 prompt（避免 RAM+GPT 每次调用）
        text_prompt = "chair. table. bed. sofa. cabinet. desk."

        # 教师前向
        depth_np, masks, boxes_xyxy, phrases, F_fused, used_K = \
            teacher.get_depth_mask_and_boxes(rgb, text_prompt, s["K"])

        if depth_np is None or len(phrases) == 0:
            print(f"    警告: 未检测到物体")
            continue

        print(f"    检测到 {len(phrases)} 个物体: {phrases[:3]}...")
        print(f"    深度范围: [{depth_np.min():.2f}, {depth_np.max():.2f}] m")
        print(f"    F_fused shape: {F_fused.shape}")

        # 生成伪标签
        pseudo_count = 0
        for j in range(len(phrases)):
            center_cam, dimensions, R_cam, ok = run_teacher_pipeline_per_instance(
                depth_np, masks[j], used_K, phrases[j], teacher.prior_dict, use_lshape=True,
            )
            if ok:
                pseudo_count += 1
                total_objects += 1

        print(f"    生成 {pseudo_count} 个伪标签")
        success_count += 1

    print(f"\n  总结: {success_count}/{num_images} 张图成功, 共 {total_objects} 个物体")
    return success_count > 0


def test_student_forward(num_images=3, device="cuda"):
    """测试3：学生前向传播"""
    print("\n" + "=" * 60)
    print("测试3: 学生前向传播")
    print("=" * 60)

    dataset_root = os.path.join(PROJECT_ROOT, "datasets")
    samples = load_sunrgbd_sample(dataset_root, num_images)

    if not samples:
        print("  失败: 无法加载数据")
        return False

    # 初始化模型
    print("  初始化教师模型...")
    teacher = TeacherDetAny3D(device=device, use_sam2_mask=True, use_ram_gpt=False)

    print("  初始化学生头...")
    student_head = Student3DHead(
        in_channels=256,
        roi_size=7,
        hidden_dim=256,
        num_fc=2,
        use_uncertainty=True,
        use_disentangled=True,
        use_chamfer=True,
    ).to(device)
    student_head.eval()

    text_prompt = "chair. table. bed. sofa. cabinet. desk."
    success_count = 0

    for i, s in enumerate(samples[:num_images]):
        print(f"\n  [{i+1}/{num_images}] {os.path.basename(s['path'])}")

        rgb = cv2.imread(s["path"])
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # 教师前向
        depth_np, masks, boxes_xyxy, phrases, F_fused, used_K = \
            teacher.get_depth_mask_and_boxes(rgb, text_prompt, s["K"])

        if depth_np is None or len(phrases) == 0:
            continue

        # 生成伪标签
        pseudo_list = []
        for j in range(len(phrases)):
            center_cam, dimensions, R_cam, ok = run_teacher_pipeline_per_instance(
                depth_np, masks[j], used_K, phrases[j], teacher.prior_dict, use_lshape=True,
            )
            if ok:
                pseudo_list.append({
                    "center_cam": center_cam,
                    "dimensions": dimensions,
                    "R_cam": R_cam,
                    "box_2d_xyxy": boxes_xyxy[j],
                })

        if not pseudo_list:
            continue

        # 学生前向 (LIFT 架构)
        image_size_hw = (896, 896)
        boxes_t = torch.from_numpy(np.stack([p["box_2d_xyxy"] for p in pseudo_list])).float().to(device)
        boxes_t[:, [0, 2]] /= image_size_hw[1]
        boxes_t[:, [1, 3]] /= image_size_hw[0]

        with torch.no_grad():
            deltas, pred_z, pred_dims, pred_pose_6d, pred_uncertainty = student_head(
                F_fused.detach(), boxes_t, torch.tensor(image_size_hw, device=device)
            )

        print(f"    学生输出: deltas={deltas.shape}, z={pred_z.shape}, dims={pred_dims.shape}, pose={pred_pose_6d.shape}")
        success_count += 1

    print(f"\n  总结: {success_count}/{num_images} 张图成功")
    return success_count > 0


def test_loss_and_backward(num_images=2, device="cuda"):
    """测试4：损失计算和梯度反传"""
    print("\n" + "=" * 60)
    print("测试4: 损失计算和梯度反传")
    print("=" * 60)

    dataset_root = os.path.join(PROJECT_ROOT, "datasets")
    samples = load_sunrgbd_sample(dataset_root, num_images)

    if not samples:
        print("  失败: 无法加载数据")
        return False

    # 初始化
    print("  初始化模型...")
    teacher = TeacherDetAny3D(device=device, use_sam2_mask=True, use_ram_gpt=False)

    student_head = Student3DHead(
        in_channels=256,
        roi_size=7,
        hidden_dim=256,
        num_fc=2,
        use_uncertainty=True,
        use_disentangled=True,
        use_chamfer=True,
    ).to(device)
    student_head.train()

    optimizer = torch.optim.AdamW(student_head.parameters(), lr=1e-4)

    loss_weights = {
        "delta_xy": 1.0,
        "dims": 1.0,
        "pose": 1.0,
        "z": 1.0,
        "uncertainty": 0.1,
        "disentangled": 1.0,
        "chamfer": 1.0,
        "joint": 1.0,
    }

    text_prompt = "chair. table. bed. sofa. cabinet. desk."
    image_size_hw = (896, 896)

    for iter_idx in range(3):
        print(f"\n  迭代 {iter_idx + 1}/3")

        for i, s in enumerate(samples[:num_images]):
            rgb = cv2.imread(s["path"])
            if rgb is None:
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # 教师前向（无梯度）
            with torch.no_grad():
                depth_np, masks, boxes_xyxy, phrases, F_fused, used_K = \
                    teacher.get_depth_mask_and_boxes(rgb, text_prompt, s["K"])

            if depth_np is None or len(phrases) == 0:
                continue

            # 生成伪标签
            pseudo_list = []
            for j in range(len(phrases)):
                center_cam, dimensions, R_cam, ok = run_teacher_pipeline_per_instance(
                    depth_np, masks[j], used_K, phrases[j], teacher.prior_dict, use_lshape=True,
                )
                if ok:
                    pseudo_list.append({
                        "center_cam": center_cam,
                        "dimensions": dimensions,
                        "R_cam": R_cam,
                        "box_2d_xyxy": boxes_xyxy[j],
                    })

            if not pseudo_list:
                continue

            # 准备 GT
            centers_gt = torch.from_numpy(np.stack([p["center_cam"] for p in pseudo_list])).float().to(device)
            dims_gt = torch.from_numpy(np.stack([p["dimensions"] for p in pseudo_list])).float().to(device)
            R_gt = torch.from_numpy(np.stack([p["R_cam"] for p in pseudo_list])).float().to(device)

            boxes_t = torch.from_numpy(np.stack([p["box_2d_xyxy"] for p in pseudo_list])).float().to(device)
            boxes_t[:, [0, 2]] /= image_size_hw[1]
            boxes_t[:, [1, 3]] /= image_size_hw[0]

            # 学生前向 (LIFT 架构)
            deltas, pred_z, pred_dims, pred_pose_6d, pred_uncertainty = student_head(
                F_fused.detach(), boxes_t, torch.tensor(image_size_hw, device=device)
            )

            # 损失计算 (LIFT 架构)
            loss, loss_dict = compute_student_loss(
                deltas, pred_z, pred_dims, pred_pose_6d, pred_uncertainty,
                centers_gt, dims_gt, R_gt,
                boxes_t, image_size_hw, torch.tensor(s["K"], device=device),
                loss_weights=loss_weights,
                use_chamfer=True,
            )

            # 梯度反传
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_str = " | ".join([f"{k}:{v:.4f}" for k, v in loss_dict.items() if isinstance(v, float)])
            print(f"    图像 {i}: loss={loss.item():.4f} ({loss_str})")

    print("\n  梯度反传测试通过!")
    return True


def test_full_pipeline(num_images=10, device="cuda"):
    """完整流程测试"""
    print("\n" + "=" * 60)
    print("完整流程测试 (完整训练循环)")
    print("=" * 60)

    dataset_root = os.path.join(PROJECT_ROOT, "datasets")
    samples = load_sunrgbd_sample(dataset_root, num_images)

    if len(samples) < 5:
        print("  失败: 样本数不足")
        return False

    # 初始化
    print("  初始化模型...")
    teacher = TeacherDetAny3D(device=device, use_sam2_mask=True, use_ram_gpt=False)

    student_head = Student3DHead(
        in_channels=256,
        roi_size=7,
        hidden_dim=256,
        num_fc=2,
        use_uncertainty=True,
        use_disentangled=True,
        use_chamfer=True,
    ).to(device)
    student_head.train()

    optimizer = torch.optim.AdamW(student_head.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    loss_weights = {
        "delta_xy": 1.0,
        "dims": 1.0,
        "pose": 1.0,
        "z": 1.0,
        "uncertainty": 0.1,
        "disentangled": 1.0,
        "chamfer": 1.0,
        "joint": 1.0,
    }

    text_prompt = "chair. table. bed. sofa. cabinet. desk."
    image_size_hw = (896, 896)
    num_epochs = 3

    for epoch in range(num_epochs):
        print(f"\n  Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        n_batches = 0

        for i, s in enumerate(samples):
            rgb = cv2.imread(s["path"])
            if rgb is None:
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # 教师前向
            with torch.no_grad():
                depth_np, masks, boxes_xyxy, phrases, F_fused, used_K = \
                    teacher.get_depth_mask_and_boxes(rgb, text_prompt, s["K"])

            if depth_np is None or len(phrases) == 0:
                continue

            # 生成伪标签
            pseudo_list = []
            for j in range(len(phrases)):
                center_cam, dimensions, R_cam, ok = run_teacher_pipeline_per_instance(
                    depth_np, masks[j], used_K, phrases[j], teacher.prior_dict, use_lshape=True,
                )
                if ok:
                    pseudo_list.append({
                        "center_cam": center_cam,
                        "dimensions": dimensions,
                        "R_cam": R_cam,
                        "box_2d_xyxy": boxes_xyxy[j],
                    })

            if not pseudo_list:
                continue

            # 学生前向
            centers_gt = torch.from_numpy(np.stack([p["center_cam"] for p in pseudo_list])).float().to(device)
            dims_gt = torch.from_numpy(np.stack([p["dimensions"] for p in pseudo_list])).float().to(device)
            R_gt = torch.from_numpy(np.stack([p["R_cam"] for p in pseudo_list])).float().to(device)

            boxes_t = torch.from_numpy(np.stack([p["box_2d_xyxy"] for p in pseudo_list])).float().to(device)
            boxes_t[:, [0, 2]] /= image_size_hw[1]
            boxes_t[:, [1, 3]] /= image_size_hw[0]

            # 学生前向 (LIFT 架构)
            deltas, pred_z, pred_dims, pred_pose_6d, pred_uncertainty = student_head(
                F_fused.detach(), boxes_t, torch.tensor(image_size_hw, device=device)
            )

            # 损失计算 (LIFT 架构)
            loss, _ = compute_student_loss(
                deltas, pred_z, pred_dims, pred_pose_6d, pred_uncertainty,
                centers_gt, dims_gt, R_gt,
                boxes_t, image_size_hw, torch.tensor(s["K"], device=device),
                loss_weights=loss_weights,
                use_chamfer=True,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_head.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"    平均损失: {avg_loss:.4f}, 有效批次: {n_batches}")

    # 保存模型
    output_dir = os.path.join(PROJECT_ROOT, "output", "distill_test")
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "student_3d_head_test.pth")
    torch.save(student_head.state_dict(), ckpt_path)
    print(f"\n  模型已保存: {ckpt_path}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=10, help="测试图像数量")
    parser.add_argument("--test_data", action="store_true", help="只测试数据加载")
    parser.add_argument("--test_teacher", action="store_true", help="只测试教师")
    parser.add_argument("--test_student", action="store_true", help="只测试学生")
    parser.add_argument("--test_loss", action="store_true", help="只测试损失")
    parser.add_argument("--test_full", action="store_true", help="完整流程测试")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("  蒸馏训练流程快速测试")
    print("=" * 60)
    print(f"  设备: {args.device}")
    print(f"  测试图像数: {args.num_images}")

    all_passed = True

    # 如果没有指定测试项，运行所有测试
    if not any([args.test_data, args.test_teacher, args.test_student, args.test_loss, args.test_full]):
        args.test_full = True

    if args.test_data or args.test_full:
        all_passed &= test_data_loading(args.num_images)

    if args.test_teacher:
        all_passed &= test_teacher_pseudo_labels(args.num_images, args.device)

    if args.test_student:
        all_passed &= test_student_forward(args.num_images, args.device)

    if args.test_loss:
        all_passed &= test_loss_and_backward(2, args.device)

    if args.test_full:
        all_passed &= test_full_pipeline(args.num_images, args.device)

    print("\n" + "=" * 60)
    if all_passed:
        print("  所有测试通过!")
    else:
        print("  部分测试失败，请检查错误信息")
    print("=" * 60)


if __name__ == "__main__":
    main()
