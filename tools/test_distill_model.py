# Copyright (c) Teacher-Student Distillation Pipeline
"""
学生模型评测脚本 - SUNRGBD 测试集
完整对齐原版 OVM3D-Det 评测流程 (tools/train_net.py)

评测内容：
1. 加载训练好的 Student3DHead 检查点
2. 使用 GT 2D 框（或教师 GDINO 检测）做评测
3. 使用 Omni3D 官方评测工具计算 2D/3D AP
4. 支持分布式评测和可视化

使用方法:
  # 使用 GT 2D 框（公平对比教师 vs 学生）
  python tools/test_distill_model.py \
    --config-file configs/Base_Omni3D_SUNRGBD.yaml \
    --student-checkpoint output/distill_sunrgbd/student_3d_head_final.pth \
    --box-source gt

  # 使用 GDINO 2D 检测（端到端学生评测）
  python tools/test_distill_model.py \
    --config-file configs/Base_Omni3D_SUNRGBD.yaml \
    --student-checkpoint output/distill_sunrgbd/student_3d_head_final.pth \
    --box-source gdino

  # 与教师模型对比
  python tools/test_distill_model.py \
    --config-file configs/Base_Omni3D_SUNRGBD.yaml \
    --box-source teacher
"""
import os
import sys
import copy
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.dont_write_bytecode = True

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.structures import Boxes, Instances

from cubercnn.data import (
    Omni3D,
    simple_register,
    build_detection_test_loader,
)
from cubercnn.data import get_filter_settings_from_cfg as cube_get_filter_settings
from cubercnn import data as cube_data
from cubercnn.util import CubeRCNNHandler
from cubercnn.evaluation.omni3d_evaluation import (
    Omni3DEvaluationHelper,
    Omni3DEvaluator,
    instances_to_coco_json,
)

from detany3d_frontend.image_encoder import ImageEncoderViT
from teacher_student.student_3d_head import Student3DHead, lift_project_inference
from teacher_student.teacher_detany3d import TeacherDetAny3D
from cubercnn.generate_label.util import llm_generated_prior

logger_initialized = False


def rotation_6d_to_matrix(d6):
    """6D 旋转 -> 3x3 矩阵"""
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def setup(args):
    """配置加载，与 train_distill_sunrgbd.py 完全一致"""
    cfg = get_cfg()
    cfg.set_new_allowed(True)

    cfg_file = os.path.join(PROJECT_ROOT, "cubercnn", "config", "config.py")
    if os.path.exists(cfg_file):
        from cubercnn.config import get_cfg_defaults
        get_cfg_defaults(cfg)

    config_file = args.config_file
    if config_file.startswith(CubeRCNNHandler.PREFIX):
        config_file = CubeRCNNHandler._get_local_path(CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    import logging
    global logger_initialized
    if not logger_initialized:
        logging.basicConfig(level=logging.INFO)
        logger_initialized = True

    return cfg


def build_student_model(cfg, checkpoint_path, device="cuda"):
    """
    构建学生模型并加载检查点
    与 train_distill_sunrgbd.py 中 Student3DHead 初始化参数完全一致
    """
    student_head = Student3DHead(
        in_channels=256,
        roi_size=7,
        hidden_dim=256,
        num_fc=2,
        use_uncertainty=True,
        use_disentangled=True,
        use_chamfer=True,
        use_log_dims=True,
        use_joint_loss=cfg.MODEL.ROI_CUBE_HEAD.get("LOSS_W_JOINT", 0.0) > 0,
        use_inverse_z_weight=cfg.MODEL.ROI_CUBE_HEAD.get("INVERSE_Z_WEIGHT", True),
        z_type=cfg.MODEL.ROI_CUBE_HEAD.get("Z_TYPE", "log"),
        shared_fc=True,
    ).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        student_head.load_state_dict(state, strict=False)
        print(f"加载学生检查点: {checkpoint_path}")
    else:
        print(f"警告: 未找到学生检查点 {checkpoint_path}，使用随机初始化权重")

    student_head.eval()
    return student_head


def build_dino_encoder(device="cuda"):
    """
    构建 DINOv2 图像编码器（与 teacher_detany3d.py 中相同）
    返回: image_encoder, cfg
    """
    cfg_model = type("Config", (), {
        "model": type("M", (), {
            "pad": 896,
            "additional_adapter": True,
            "multi_level_box_output": 1,
            "image_encoder": type("IE", (), {
                "patch_size": 16,
                "global_attn_indexes": [7, 15, 23, 31],
            })(),
        })(),
        "contain_edge_obj": False,
        "output_rotation_matrix": False,
        "dino_path": os.path.join(PROJECT_ROOT, "weights", "dinov2_vitl14_pretrain.pth"),
        "sam_path": os.path.join(PROJECT_ROOT, "weights", "sam_vit_h_4b8939.pth"),
        "unidepth_path": os.path.join(PROJECT_ROOT, "weights", "model.pth"),
    })()

    image_encoder = ImageEncoderViT(
        img_size=cfg_model.model.pad,
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        cfg=cfg_model,
        device=device,
    ).to(device)
    image_encoder.eval()
    return image_encoder, cfg_model


def extract_dino_features(image_encoder, rgb_np, K, device="cuda"):
    """
    使用 DINOv2 编码器提取图像特征
    与 teacher_detany3d.py get_depth_mask_and_boxes 中的编码器使用逻辑完全一致
    返回: F_fused (1, 256, H, W)
    """
    h, w = rgb_np.shape[:2]

    pad_img = cv2.copyMakeBorder(
        rgb_np,
        0, max(0, 896 - h),
        0, max(0, 896 - w),
        cv2.BORDER_CONSTANT,
    )[:896, :896]

    pad_tensor = torch.from_numpy(pad_img).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_t_dino = (
        pad_tensor / 255.0
        - torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    ) / torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    gt_intrinsic = torch.eye(4).float().unsqueeze(0).to(device)
    gt_intrinsic[0, :3, :3] = torch.tensor(K, dtype=torch.float32, device=device)

    input_dict = {
        "images": pad_tensor,
        "image_for_dino": img_t_dino,
        "vit_pad_size": torch.tensor([[h // 16, w // 16]], dtype=torch.long, device=device),
        "gt_intrinsic": gt_intrinsic,
    }

    with torch.no_grad():
        output_dict = image_encoder(input_dict)
    F_fused = output_dict["image_embeddings"]
    return F_fused, h, w


def run_student_inference_per_image(
    student_head,
    image_encoder,
    rgb_np,
    K,
    gt_instances,
    device="cuda",
    use_gt_boxes=True,
    score_threshold=0.1,
    image_size_hw=(896, 896),
):
    """
    对单张图像运行学生模型推理

    Args:
        student_head: Student3DHead 模型
        image_encoder: DINOv2 编码器
        rgb_np: (H, W, 3) numpy RGB 图像
        K: (3, 3) 相机内参
        gt_instances: Omni3D GT instances dict (from dataset)
        use_gt_boxes: True=使用 GT 2D 框, False=使用教师 GDINO 检测
        score_threshold: 过滤低置信度预测

    Returns:
        instances: detectron2 Instances 对象（含 pred_boxes, scores, pred_classes,
                   pred_center_cam, pred_dimensions, pred_pose, pred_bbox3D）
    """
    h, w = rgb_np.shape[:2]

    if gt_instances is None or len(gt_instances) == 0:
        return None

    inst = gt_instances
    n = len(inst.get("gt_boxes", []))

    if n == 0:
        return None

    if "gt_boxes" in inst:
        boxes_xyxy = inst["gt_boxes"].tensor.numpy()
    else:
        boxes_xyxy = np.array(inst.get("bbox", []))

    if boxes_xyxy.shape[0] == 0:
        return None

    box_xyxy_norm = boxes_xyxy.copy()
    box_xyxy_norm[:, [0, 2]] /= float(w)
    box_xyxy_norm[:, [1, 3]] /= float(h)
    box_xyxy_norm = np.clip(box_xyxy_norm, 0, 1)

    box_xyxy_norm_t = torch.from_numpy(box_xyxy_norm).float().to(device)
    F_fused, feat_h, feat_w = extract_dino_features(image_encoder, rgb_np, K, device)

    B, C, H_feat, W_feat = F_fused.shape
    scale_x = W_feat / float(w)
    scale_y = H_feat / float(h)
    boxes_feat = box_xyxy_norm_t.clone()
    boxes_feat[:, [0, 2]] *= scale_x
    boxes_feat[:, [1, 3]] *= scale_y
    boxes_feat[:, [0, 2]] *= (896.0 / w)
    boxes_feat[:, [1, 3]] *= (896.0 / h)

    from torchvision.ops import roi_align
    boxes_for_roi = boxes_feat.float()
    if boxes_for_roi.numel() > 0:
        boxes_for_roi = boxes_for_roi.unsqueeze(0) if boxes_for_roi.dim() == 2 else boxes_for_roi
        rois = roi_align(
            F_fused, [boxes_for_roi[0]],
            output_size=(7, 7),
            spatial_scale=1.0,
            aligned=True,
        )
        rois = rois.flatten(1)
    else:
        return None

    if student_head.shared_fc:
        feat = student_head.feature_generator(rois)
    else:
        feat = student_head.feature_generator_Z(rois)

    deltas = student_head.bbox_2d_deltas(feat if student_head.shared_fc
                                          else student_head.feature_generator_XY(rois))
    z = student_head.bbox_z(feat if student_head.shared_fc
                             else student_head.feature_generator_Z(rois))
    dims = student_head.bbox_dims(feat if student_head.shared_fc
                                   else student_head.feature_generator_dims(rois))
    pose_6d = student_head.bbox_pose(feat if student_head.shared_fc
                                      else student_head.feature_generator_pose(rois))

    dims = torch.nn.functional.softplus(dims) + 0.1

    if student_head.z_type == 'log':
        z = torch.exp(z.clamp(max=5))
    elif student_head.z_type == 'sigmoid':
        z = torch.sigmoid(z) * 100
    elif student_head.z_type == 'direct':
        z = torch.nn.functional.softplus(z) + 0.01

    K_t = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)

    center_3d, pred_dims, R_cam = lift_project_inference(
        deltas, z, dims, pose_6d, box_xyxy_norm_t, image_size_hw, K_t
    )

    uncertainty = None
    if student_head.use_uncertainty:
        uncertainty = student_head.bbox_uncertainty(
            feat if student_head.shared_fc
            else student_head.feature_generator_conf(rois)
        ).clamp(0.01)

    scores = 1.0 / (uncertainty.squeeze(-1) + 0.1) if uncertainty is not None else torch.ones(n, device=device)
    scores = scores.cpu().numpy()

    if "category_id" in inst:
        pred_classes = np.array(inst["category_id"])
    elif "category_ids" in inst:
        pred_classes = np.array(inst["category_ids"])
    else:
        pred_classes = np.zeros(n, dtype=np.int64)

    keep = scores > score_threshold
    if not keep.any():
        return None

    center_3d = center_3d.cpu().numpy()[keep]
    pred_dims = pred_dims.cpu().numpy()[keep]
    R_cam = R_cam.cpu().numpy()[keep]
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    pred_classes = pred_classes[keep]
    n_keep = len(keep_idx := np.where(keep)[0])

    if n_keep == 0:
        return None

    from cubercnn.util.math_util import get_cuboid_verts_faces
    bbox3d_list = []
    for i in range(n_keep):
        box3d_6 = np.array([center_3d[i, 0], center_3d[i, 1], center_3d[i, 2],
                            pred_dims[i, 0], pred_dims[i, 1], pred_dims[i, 2]])
        R_i = R_cam[i]
        verts_t, _ = get_cuboid_verts_faces(
            torch.from_numpy(box3d_6).float().unsqueeze(0),
            torch.from_numpy(R_i).float().unsqueeze(0),
        )
        bbox3d_list.append(verts_t.squeeze(0).numpy())

    instances = Instances((h, w))
    instances.pred_boxes = Boxes(boxes_xyxy)
    instances.scores = torch.from_numpy(scores).float()
    instances.pred_classes = torch.from_numpy(pred_classes).long()
    instances.pred_center_cam = torch.from_numpy(center_3d).float()
    instances.pred_dimensions = torch.from_numpy(pred_dims).float()
    instances.pred_pose = torch.from_numpy(R_cam).float()
    instances.pred_bbox3D = torch.from_numpy(np.array(bbox3d_list)).float()

    return instances


def run_teacher_inference_per_image(
    teacher,
    rgb_np,
    K,
    text_prompt=None,
    score_threshold=0.1,
    device="cuda",
):
    """
    对单张图像运行教师模型推理（用于对比评测）
    """
    with torch.no_grad():
        depth_np, masks, boxes_xyxy, phrases, F_fused, used_K = \
            teacher.get_depth_mask_and_boxes(rgb_np, text_prompt, K)

    if depth_np is None or phrases is None or len(phrases) == 0:
        return None, None, None

    from teacher_student.teacher_geometry import run_teacher_pipeline_per_instance
    pseudo_list = []
    for i in range(len(phrases)):
        center_cam, dimensions, R_cam, ok = run_teacher_pipeline_per_instance(
            depth_np, masks[i], used_K, phrases[i], teacher.prior_dict,
            use_lshape=False,
        )
        if ok:
            pseudo_list.append({
                "center_cam": center_cam,
                "dimensions": dimensions,
                "R_cam": R_cam,
                "box_2d_xyxy": boxes_xyxy[i],
                "phrase": phrases[i],
            })

    if not pseudo_list:
        return None, None, None

    n = len(pseudo_list)
    h, w = rgb_np.shape[:2]

    boxes_xyxy = np.stack([p["box_2d_xyxy"] for p in pseudo_list])
    center_3d = np.stack([p["center_cam"] for p in pseudo_list])
    pred_dims = np.stack([p["dimensions"] for p in pseudo_list])
    R_cam = np.stack([p["R_cam"] for p in pseudo_list])
    scores = np.ones(n)
    pred_classes = np.zeros(n, dtype=np.int64)

    from teacher_student.ram_gpt_labeler import category_to_id
    class_ids = [category_to_id.get(p["phrase"].lower(), 0) for p in pseudo_list]
    pred_classes = np.array(class_ids, dtype=np.int64)

    from cubercnn.util.math_util import get_cuboid_verts_faces
    bbox3d_list = []
    for i in range(n):
        box3d_6 = np.array([center_3d[i, 0], center_3d[i, 1], center_3d[i, 2],
                            pred_dims[i, 0], pred_dims[i, 1], pred_dims[i, 2]])
        verts_t, _ = get_cuboid_verts_faces(
            torch.from_numpy(box3d_6).float().unsqueeze(0),
            torch.from_numpy(R_cam[i]).float().unsqueeze(0),
        )
        bbox3d_list.append(verts_t.squeeze(0).numpy())

    instances = Instances((h, w))
    instances.pred_boxes = Boxes(boxes_xyxy)
    instances.scores = torch.from_numpy(scores).float()
    instances.pred_classes = torch.from_numpy(pred_classes).long()
    instances.pred_center_cam = torch.from_numpy(center_3d).float()
    instances.pred_dimensions = torch.from_numpy(pred_dims).float()
    instances.pred_pose = torch.from_numpy(R_cam).float()
    instances.pred_bbox3D = torch.from_numpy(np.array(bbox3d_list)).float()

    return instances, phrases, F_fused


class DistillDatasetMapper:
    """
    评测数据映射器 - 返回原始图像和 GT instances
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, dataset_dict):
        d = copy.deepcopy(dataset_dict)
        rgb = cv2.imread(d["file_name"])
        if rgb is None:
            raise IOError(f"无法读取图像: {d['file_name']}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        d["image"] = rgb
        return d


def run_evaluation(
    cfg,
    student_head,
    image_encoder,
    teacher,
    box_source,
    device="cuda",
    max_images=None,
):
    """
    在 SUNRGBD 测试集上运行评测

    Args:
        cfg: detectron2 配置
        student_head: 学生模型（或教师评测时为 None）
        image_encoder: DINOv2 编码器（学生评测时使用）
        teacher: 教师模型（教师评测时使用）
        box_source: "gt" | "gdino" | "teacher"
        device: CUDA 设备
        max_images: 最大评测图像数（None=全部）
    """
    filter_settings = cube_get_filter_settings(cfg)

    for dataset_name in cfg.DATASETS.TEST:
        simple_register(dataset_name, filter_settings, folder_name=cfg.DATASETS.FOLDER_NAME, filter_empty=False)

    test_dataset_name = cfg.DATASETS.TEST[0]
    print(f"评测数据集: {test_dataset_name}")

    dataset_paths = [os.path.join("datasets", cfg.DATASETS.FOLDER_NAME, test_dataset_name + ".json")]
    test_dataset = Omni3D(dataset_paths, filter_settings=filter_settings)

    image_infos = test_dataset.dataset["images"]
    if max_images is not None:
        image_infos = image_infos[:max_images]
    print(f"评测图像数: {len(image_infos)}")

    MetadataCatalog.get("omni3d_model").thing_classes = MetadataCatalog.get(test_dataset_name).thing_classes
    MetadataCatalog.get("omni3d_model").thing_dataset_id_to_contiguous_id = \
        MetadataCatalog.get(test_dataset_name).thing_dataset_id_to_contiguous_id

    eval_helper = Omni3DEvaluationHelper(
        dataset_names=[test_dataset_name],
        filter_settings=filter_settings,
        output_folder=cfg.OUTPUT_DIR,
        iter_label=os.path.basename(cfg.OUTPUT_DIR),
        only_2d=False,
    )

    ann_dict = {img["id"]: img for img in test_dataset.dataset["images"]}
    if "annotations" in test_dataset.dataset:
        for ann in test_dataset.dataset["annotations"]:
            img_id = ann["image_id"]
            if img_id not in ann_dict:
                continue

    eval_count = 0
    text_prompt = "chair. table. bed. sofa. cabinet. desk. bookcase. counter. door. pillow. " \
                  "refrigerator. shower curtain. sink. stool. toilet. tv."

    for img_info in image_infos:
        img_id = img_info["id"]
        rgb_path = os.path.join("datasets", img_info["file_path"])

        if not os.path.exists(rgb_path):
            print(f"跳过缺失图像: {rgb_path}")
            continue

        rgb = cv2.imread(rgb_path)
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        K = np.array(img_info["K"], dtype=np.float32)

        img_anns = [a for a in test_dataset.dataset.get("annotations", [])
                     if a["image_id"] == img_id]

        gt_instances_dict = None
        if len(img_anns) > 0:
            boxes = np.array([a["bbox"] for a in img_anns])
            cat_ids = [a["category_id"] for a in img_anns]
            if len(boxes) > 0 and boxes.shape[1] == 4:
                x, y, w_box, h_box = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                boxes_xyxy = np.stack([x, y, x + w_box, y + h_box], axis=1)
                gt_instances_dict = {
                    "gt_boxes": Boxes(torch.from_numpy(boxes_xyxy).float()),
                    "category_id": np.array(cat_ids),
                }

        if box_source == "gt":
            instances = run_student_inference_per_image(
                student_head, image_encoder, rgb, K, gt_instances_dict,
                device=device, use_gt_boxes=True,
            )
        elif box_source == "gdino":
            instances = run_student_inference_per_image(
                student_head, image_encoder, rgb, K, gt_instances_dict,
                device=device, use_gt_boxes=False,
            )
        elif box_source == "teacher":
            instances, phrases, F_fused = run_teacher_inference_per_image(
                teacher, rgb, K, text_prompt=text_prompt, device=device,
            )
        else:
            raise ValueError(f"未知的 box_source: {box_source}")

        prediction = {
            "image_id": img_id,
            "K": K,
            "width": rgb.shape[1],
            "height": rgb.shape[0],
        }

        if instances is not None:
            instances_cpu = Instances((rgb.shape[0], rgb.shape[1]))
            instances_cpu.pred_boxes = Boxes(instances.pred_boxes.tensor.numpy())
            instances_cpu.scores = instances.scores
            instances_cpu.pred_classes = instances.pred_classes
            instances_cpu.pred_center_cam = instances.pred_center_cam
            instances_cpu.pred_dimensions = instances.pred_dimensions
            instances_cpu.pred_pose = instances.pred_pose
            instances_cpu.pred_bbox3D = instances.pred_bbox3D
            prediction["instances"] = instances_to_coco_json(instances_cpu, img_id)
        else:
            prediction["instances"] = []

        eval_helper.add_predictions(test_dataset_name, [prediction])
        eval_count += 1

        if eval_count % 100 == 0:
            print(f"已评测 {eval_count}/{len(image_infos)} 张图像...")

    print(f"评测完成，共 {eval_count} 张图像。正在计算指标...")

    eval_helper.evaluate(test_dataset_name)
    eval_helper.save_predictions(test_dataset_name)
    eval_helper.summarize_all()

    results = eval_helper.results_analysis
    print("\n" + "=" * 60)
    print("评测汇总:")
    for k, v in results.items():
        print(f"  {k}: AP2D={v['AP2D']:.2f}, AP3D={v['AP3D']:.2f}")
    print("=" * 60)

    return results


def main(args):
    cfg = setup(args)
    device = f"cuda:{comm.get_local_rank()}" if comm.get_world_size() > 1 else "cuda"

    box_source = args.box_source.lower()
    print(f"\n评测配置:")
    print(f"  模型来源: {box_source}")
    print(f"  学生检查点: {args.student_checkpoint}")
    print(f"  输出目录: {cfg.OUTPUT_DIR}")

    if box_source == "teacher":
        print("\n初始化教师模型...")
        teacher = TeacherDetAny3D(
            device=device,
            use_sam2_mask=True,
            use_ram_gpt=False,
        )
        student_head = None
        image_encoder = None
    else:
        print("\n初始化 DINOv2 编码器...")
        image_encoder, _ = build_dino_encoder(device)

        print("初始化学生模型...")
        student_head = build_student_model(cfg, args.student_checkpoint, device)

        teacher = None

    results = run_evaluation(
        cfg,
        student_head,
        image_encoder,
        teacher,
        box_source=box_source,
        device=device,
        max_images=args.max_images,
    )

    print("\n评测完成！结果保存在:", cfg.OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--student-checkpoint",
        type=str,
        default="output/distill_sunrgbd/student_3d_head_final.pth",
        help="学生模型检查点路径",
    )
    parser.add_argument(
        "--box-source",
        type=str,
        default="gt",
        choices=["gt", "gdino", "teacher"],
        help=(
            "gt: 使用 GT 2D 框（公平对比）\n"
            "gdino: 使用 GDINO 2D 检测（端到端学生评测）\n"
            "teacher: 教师模型评测（基准对比）"
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="最大评测图像数（None=全部）",
    )
    args = parser.parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
