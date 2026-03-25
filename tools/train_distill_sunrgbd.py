# Copyright (c) Teacher-Student Distillation Pipeline - SUNRGBD Training
"""
用 SUNRGBD 数据集训练学生模型，训练逻辑完全对齐原版 OVM3D-Det (tools/train_net.py)

训练流程（与原版一致）：
1. 加载 SUNRGBD Omni3D 格式数据集
2. 教师端（MoGe+DepthPro+SAM2）生成伪 3D 标签
3. 学生端（DINOv2+Student3DHead）学习
4. 支持分布式训练、检查点、评估

使用方法:
  CUDA_VISIBLE_DEVICES=0 python tools/train_distill_sunrgbd.py \
    --config-file configs/Base_Omni3D_SUN.yaml \
    OUTPUT_DIR output/distill_sunrgbd
"""
import os
import sys
import json
import logging
import copy
import time
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    default_writers,
    launch,
)
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

logger = logging.getLogger("cubercnn")

sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd())
np.set_printoptions(suppress=True)

# =============================================================================
# 数据集相关
# =============================================================================
from cubercnn.data import (
    load_omni3d_json,
    DatasetMapper3D,
    build_detection_train_loader,
    build_detection_test_loader,
    simple_register,
)
from cubercnn.data.builtin import get_omni3d_categories
from cubercnn import data as cube_data

# =============================================================================
# 教师模型（MoGe+DepthPro+SAM2）
# =============================================================================
from teacher_student.teacher_detany3d import TeacherDetAny3D
from teacher_student.teacher_geometry import run_teacher_pipeline_per_instance

# =============================================================================
# 学生模型
# =============================================================================
from teacher_student.student_3d_head import Student3DHead, compute_student_loss

# =============================================================================
# 工具
# =============================================================================
from cubercnn import util

# =============================================================================
# 训练参数
# =============================================================================
MAX_TRAINING_ATTEMPTS = 3  # 原版是 10，减小以便快速调试
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =============================================================================
# 教师伪标签生成
# =============================================================================
def generate_teacher_pseudo_labels(teacher, rgb, K, phrases, boxes_xyxy,
                                     depth_np, masks, prior_dict,
                                     use_lshape=True, device='cuda'):
    """
    对单张图像生成教师伪标签
    返回: pseudo_list (list of dict), F_fused (特征图)
    """
    if depth_np is None or len(phrases) == 0:
        return [], None

    pseudo_list = []
    depth_h, depth_w = depth_np.shape[:2]

    # mask resize 到 depth 尺寸
    if masks.shape[-2:] != (depth_h, depth_w):
        import torch.nn.functional as F
        masks_tensor = torch.from_numpy(masks).unsqueeze(0).float()
        masks_tensor = F.interpolate(masks_tensor, size=(depth_h, depth_w), mode='nearest')
        masks = masks_tensor.squeeze(0).numpy()

    for i in range(len(phrases)):
        center_cam, dimensions, R_cam, ok = run_teacher_pipeline_per_instance(
            depth_np, masks[i], K, phrases[i], prior_dict,
            use_lshape=use_lshape,
        )
        if not ok:
            continue
        pseudo_list.append({
            "center_cam": center_cam,
            "dimensions": dimensions,
            "R_cam": R_cam,
            "box_2d_xyxy": boxes_xyxy[i],
            "phrase": phrases[i],
        })

    return pseudo_list


# =============================================================================
# 数据映射器（蒸馏训练专用，不依赖 GT 标注）
# =============================================================================
class DistillDatasetMapper:
    """
    蒸馏训练专用数据映射器：
    - 不依赖 GT 标注（标注由教师模型生成）
    - 返回原始图像、K、图像路径
    - 与原版 DatasetMapper3D 的接口兼容
    """

    def __init__(self, cfg, is_train=True, min_size=800, max_size=1333):
        self.is_train = is_train
        self.min_size = min_size
        self.max_size = max_size
        from detectron2.data import transforms as T
        self.augmentations = T.AugInput._build_augmentations([
            "resize", "horizontal_flip"
        ]) if is_train else T.AugInput._build_augmentations(["resize"])

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        # 读取图像
        from detectron2.data import detection_utils
        image = detection_utils.read_image(dataset_dict["file_name"], format="RGB")
        detection_utils.check_image_size(dataset_dict, image)

        # 数据增强
        from detectron2.data import transforms as T
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]  # h, w

        # 转为 Tensor (C, H, W)
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            return dataset_dict

        # 蒸馏训练不需要 GT 标注（教师会生成伪标签）
        # 但需要保留基本信息供教师使用
        dataset_dict["instances"] = None  # 标记为蒸馏模式

        return dataset_dict


# =============================================================================
# 蒸馏数据集类
# =============================================================================
class DistillSimpleDataset(torch.utils.data.Dataset):
    """
    简化的蒸馏数据集，直接从 Omni3D JSON 加载
    与原版 Omni3D 数据集格式一致，但专门用于蒸馏训练
    """

    def __init__(self, dataset_dicts, image_root="datasets", is_train=True, min_size=800, max_size=1333):
        """
        dataset_dicts: 从 Omni3D 加载的 images 列表
        """
        self.dataset_dicts = dataset_dicts
        self.image_root = image_root
        self.is_train = is_train

        # 数据增强（与原版 DatasetMapper3D 一致）
        from detectron2.data import transforms as T
        self.augmentations = (
            [T.RandomFlip()] if is_train else []
        )

    def __len__(self):
        return len(self.dataset_dicts)

    def __getitem__(self, idx):
        d = self.dataset_dicts[idx]

        # 读取图像
        from detectron2.data import detection_utils
        image = detection_utils.read_image(
            os.path.join(self.image_root, d["file_path"]),
            format="RGB"
        )
        orig_h, orig_w = image.shape[:2]

        # 数据增强（与原版 DatasetMapper3D 一致）
        from detectron2.data import transforms as T
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]  # h, w after resize

        # 应用变换到内参
        K = np.array(d["K"], dtype=np.float32)

        # 应用 resize 到 K（图像缩放到训练尺寸）
        scale_x = image_shape[1] / float(orig_w)
        scale_y = image_shape[0] / float(orig_h)
        K[0, :] *= scale_x
        K[1, :] *= scale_y

        # 应用 horizontal flip 到 K
        for transform in transforms:
            if isinstance(transform, T.HFlipTransform):
                K[0, 2] = image_shape[1] - K[0, 2]

        # 转为 Tensor (C, H, W)
        image_tensor = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        return {
            "rgb": image,  # numpy (H, W, 3)
            "rgb_tensor": image_tensor,  # tensor (C, H, W)
            "K": K,
            "image_id": d.get("id", idx),
            "file_path": d["file_path"],
            "height": image_shape[0],
            "width": image_shape[1],
            "dataset_name": d.get("dataset_name", "SUNRGBD"),
            "phrases": [],  # 会在训练时由教师模型生成
            "boxes_xyxy": np.zeros((0, 4), dtype=np.float32),  # 会在训练时由教师模型生成
        }


def build_distill_train_loader(cfg, dataset_dicts, is_train=True, min_size=800, max_size=1333):
    """构建蒸馏训练数据加载器"""
    dataset = DistillSimpleDataset(dataset_dicts, is_train=is_train, min_size=min_size, max_size=max_size)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        shuffle=is_train,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        collate_fn=lambda x: x,  # 返回 list of dict
        drop_last=is_train,
    )


# =============================================================================
# 蒸馏训练循环
# =============================================================================
def do_distill_train(cfg, teacher, student_head, optimizer, scheduler,
                     data_loader, device, text_embeddings,
                     image_size_hw=(896, 896),
                     resume=False, use_lshape=True,
                     loss_weights=None,
                     use_uncertainty=True,
                     use_disentangled=True,
                     use_chamfer=True,
                     use_3d_nms=True):
    """
    蒸馏训练主循环（与原版 tools/train_net.py 的 do_train 完全对齐）
    - 教师生成伪标签（梯度 detach）
    - 学生学习伪标签
    - 支持梯度爆炸检测和恢复
    - 支持周期性评估和检查点保存
    """
    max_iter = cfg.SOLVER.MAX_ITER
    do_eval = cfg.TEST.EVAL_PERIOD > 0

    student_head.train()
    optimizer.zero_grad()

    # 检查点管理
    checkpointer = DetectionCheckpointer(
        student_head, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (checkpointer.resume_or_load("", resume=resume).get("iteration", -1) + 1)
    iteration = start_iter
    logger.info(f"蒸馏训练从 iteration {start_iter} 开始")

    # 梯度爆炸检测（与原版一致）
    iterations_success = 0
    iterations_explode = 0
    TOLERANCE = 4.0
    GAMMA = 0.02
    recent_loss = None

    data_iter = iter(data_loader)
    named_params = list(student_head.named_parameters())

    # 伪标签生成缓存（LRU，防止内存无限增长）
    pseudo_cache = OrderedDict()
    PSEUDO_CACHE_MAX_SIZE = 200  # 最多缓存 200 张图像的伪标签

    with EventStorage(start_iter) as storage:
        while True:
            storage.iter = iteration

            # ---------------- 获取数据 ----------------
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            # ---------------- 教师前向（无梯度） ----------------
            total_loss = torch.tensor(0.0, device=device)
            loss_dict_reduced = {}
            n_valid = 0

            for sample in batch:
                rgb = sample["rgb"]          # (H, W, 3) numpy
                K = np.array(sample["K"])   # (3, 3)
                phrases = sample["phrases"]  # list of str
                boxes_xyxy = sample["boxes_xyxy"]  # (N, 4) numpy
                image_id = sample.get("image_id", None)

                # 使用可靠的缓存键
                if image_id is not None:
                    cache_key = str(image_id)
                else:
                    # 使用图像路径或哈希作为缓存键
                    cache_key = str(hash(sample.get("file_path", id(rgb))))

                # 检查缓存（命中则移到最后，表示最近使用）
                if cache_key in pseudo_cache:
                    pseudo_list = pseudo_cache.pop(cache_key)     # 移除
                    feat_key = f"{cache_key}_feat"
                    feat_val = pseudo_cache.pop(feat_key, None)  # 同时移除 feat
                    pseudo_cache[cache_key] = pseudo_list       # LRU 更新
                    if feat_val is not None:
                        pseudo_cache[feat_key] = feat_val
                    F_fused = feat_val
                else:
                    # 蒸馏训练时使用固定 prompt（不依赖 phrases）
                    text_prompt = "chair. table. bed. sofa. cabinet. desk. bookcase. counter. door. pillow. refrigerator. shower curtain. sink. stool. toilet. tv."

                    with torch.no_grad():
                        depth_np, masks, detected_boxes, detected_phrases, F_fused = \
                            teacher.get_depth_mask_and_boxes(rgb, text_prompt, K)
                        if depth_np is None or len(detected_phrases) == 0:
                            pseudo_list = []
                        else:
                            # 对齐 boxes
                            if detected_boxes is not None and len(detected_boxes) > 0:
                                detected_boxes_aligned = detected_boxes
                            else:
                                detected_boxes_aligned = boxes_xyxy

                            pseudo_list = generate_teacher_pseudo_labels(
                                teacher, rgb, K, detected_phrases,
                                detected_boxes_aligned if len(detected_boxes_aligned) > 0 else boxes_xyxy,
                                depth_np, masks, teacher.prior_dict,
                                use_lshape=use_lshape, device=device
                            )

                    # 缓存结果（超过容量时淘汰最旧的）
                    pseudo_cache[cache_key] = pseudo_list
                    if F_fused is not None:
                        pseudo_cache[f"{cache_key}_feat"] = F_fused
                    while len(pseudo_cache) > PSEUDO_CACHE_MAX_SIZE:
                        pseudo_cache.popitem(last=False)

                if not pseudo_list or F_fused is None:
                    continue

                # ---------------- 学生前向 ----------------
                h, w = rgb.shape[:2]

                # 准备 GT
                centers_gt = torch.from_numpy(np.stack([p["center_cam"] for p in pseudo_list])).float().to(device)
                dims_gt = torch.from_numpy(np.stack([p["dimensions"] for p in pseudo_list])).float().to(device)
                R_gt = torch.from_numpy(np.stack([p["R_cam"] for p in pseudo_list])).float().to(device)
                boxes_t = torch.from_numpy(np.stack([p["box_2d_xyxy"] for p in pseudo_list])).float().to(device)
                boxes_t[:, [0, 2]] /= image_size_hw[1]
                boxes_t[:, [1, 3]] /= image_size_hw[0]

                # F_fused 特征图
                F_detach = F_fused.detach() if F_fused is not None else None
                if F_detach is None:
                    continue

                # 学生预测
                # LIFT 架构：返回 (deltas, z, dims, pose_6d, uncertainty)
                if use_uncertainty or use_disentangled:
                    deltas, pred_z, pred_dims, pred_pose_6d, pred_uncertainty = student_head(
                        F_detach, boxes_t, torch.tensor(image_size_hw, device=device)
                    )
                    pred_disentangled = None  # LIFT 架构不使用独立的 disentangled 分支
                else:
                    deltas, pred_z, pred_dims, pred_pose_6d, pred_uncertainty = student_head(
                        F_detach, boxes_t, torch.tensor(image_size_hw, device=device)
                    )
                    pred_uncertainty = None
                    pred_disentangled = None

                # 损失 (对齐 OVMono3D LIFT 架构)
                loss, loss_dict = compute_student_loss(
                    deltas, pred_z, pred_dims, pred_pose_6d, pred_uncertainty,
                    centers_gt, dims_gt, R_gt,
                    boxes_t, image_size_hw, torch.tensor(K, device=device),
                    loss_weights=loss_weights,
                    use_chamfer=use_chamfer,
                    use_joint_loss=use_3d_nms,
                    use_inverse_z_weight=True,
                    use_log_dims=True,
                    z_type='log',
                )

                # 梯度爆炸检测
                if recent_loss is None:
                    recent_loss = loss.item() * 2.0

                diverging = (loss.item() > recent_loss * TOLERANCE or
                            not np.isfinite(loss.item()) or np.isnan(loss.item()))

                if not diverging:
                    recent_loss = recent_loss * (1 - GAMMA) + loss.item() * GAMMA

                if comm.is_main_process():
                    for k, v in loss_dict.items():
                        if isinstance(v, (int, float)):
                            loss_dict_reduced[k] = loss_dict_reduced.get(k, 0.0) + v

                if not diverging:
                    loss.backward()
                    total_loss += loss.detach()
                    n_valid += 1

            comm.synchronize()

            # ---------------- allreduce ----------------
            if n_valid > 0:
                total_loss = total_loss / n_valid
                loss_dict_reduced = {k: v / n_valid for k, v in loss_dict_reduced.items()}

            loss_dict_reduced["total"] = total_loss.item()
            losses_reduced = sum(loss for loss in loss_dict_reduced.values() if isinstance(loss, (int, float)))

            # 检测梯度爆炸
            diverging_model = cfg.MODEL.STABILIZE > 0 and (
                losses_reduced > recent_loss * TOLERANCE or
                not np.isfinite(losses_reduced) or np.isnan(losses_reduced)
            )

            if diverging_model:
                total_loss = total_loss.clip(0, 1.0)
                logger.warning(f'跳过梯度更新: loss={losses_reduced:.2f} recent={recent_loss:.2f}')

            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            # 检查梯度爆炸（NaN/Inf）
            if not diverging_model and cfg.MODEL.STABILIZE > 0:
                for name, param in named_params:
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            diverging_model = True
                            logger.warning(f'跳过梯度更新: NaN/Inf 梯度 {name}')
                            break

            diverging_tensor = torch.tensor(float(diverging_model), device=device)
            if dist.get_world_size() > 1:
                dist.all_reduce(diverging_tensor)
            comm.synchronize()

            if diverging_tensor.item() > 0:
                optimizer.zero_grad()
                iterations_explode += 1
            else:
                torch.nn.utils.clip_grad_norm_(student_head.parameters(), 40.0)
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                iterations_success += 1

            total_iters = iterations_success + iterations_explode
            retry = (iterations_explode / total_iters >= cfg.MODEL.STABILIZE and
                     total_iters > cfg.SOLVER.CHECKPOINT_PERIOD * 0.5)

            retry_tensor = torch.tensor(float(retry), device=device)
            if dist.get_world_size() > 1:
                dist.all_reduce(retry_tensor)
            comm.synchronize()

            if retry_tensor.item() > 0:
                logger.warning(f'重新开始训练 (爆炸率 {100 * iterations_explode / total_iters:.0f}%)')
                del data_iter
                del optimizer
                del checkpointer
                return False

            scheduler.step()

            # 评估
            if not diverging_model and do_eval and ((iteration + 1) % cfg.TEST.EVAL_PERIOD == 0):
                logger.info(f'评估 iteration {iteration + 1}')
                do_distill_eval(cfg, teacher, student_head, text_embeddings, iteration=iteration + 1)
                student_head.train()
                comm.synchronize()

            # 保存检查点
            if not diverging_model and (iterations_explode / total_iters) < 0.5 * cfg.MODEL.STABILIZE:
                periodic_checkpointer = DetectionCheckpointer(
                    student_head, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
                )
                periodic_checkpointer.step(iteration)

            # 定期写日志
            if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                if comm.is_main_process():
                    for writer in default_writers(cfg.OUTPUT_DIR, max_iter):
                        writer.write()

            iteration += 1
            if iteration >= max_iter:
                break

    return True


def do_distill_eval(cfg, teacher, student_head, text_embeddings, iteration='final'):
    """
    评估函数（简化版）
    """
    pass  # 可后续扩展


def setup(args):
    """创建配置并初始化（与原版 tools/train_net.py 的 setup 完全一致）"""
    cfg = get_cfg()
    cfg.set_new_allowed(True)

    # 加载默认配置
    cfg_file = os.path.join(os.path.dirname(__file__), "..", "cubercnn", "config", "config.py")
    if os.path.exists(cfg_file):
        from cubercnn.config import get_cfg_defaults
        get_cfg_defaults(cfg)

    config_file = args.config_file
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cubercnn")

    # 注册数据集
    filter_settings = cube_data.get_filter_settings_from_cfg(cfg)
    for dataset_name in cfg.DATASETS.TRAIN:
        simple_register(dataset_name, filter_settings, folder_name=cfg.DATASETS.FOLDER_NAME, filter_empty=True)

    for dataset_name in cfg.DATASETS.TEST:
        if dataset_name not in cfg.DATASETS.TRAIN:
            simple_register(dataset_name, filter_settings, folder_name=cfg.DATASETS.FOLDER_NAME, filter_empty=False)

    return cfg


def main(args):
    cfg = setup(args)

    logger.info('加载 SUNRGBD 数据集...')

    filter_settings = cube_data.get_filter_settings_from_cfg(cfg)

    # 加载数据集
    dataset_paths = [os.path.join('datasets', cfg.DATASETS.FOLDER_NAME, name + '.json') for name in cfg.DATASETS.TRAIN]
    datasets = cube_data.Omni3D(dataset_paths, filter_settings=filter_settings)

    # 注册元数据
    cube_data.register_and_store_model_metadata(datasets, cfg.OUTPUT_DIR, filter_settings)

    thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
    dataset_id_to_contiguous_id = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id

    logger.info(f'类别数: {len(thing_classes)}')
    logger.info(f'训练图像数: {len(datasets.dataset["images"])}')

    # =================================================================
    # 教师模型初始化（MoGe+DepthPro+SAM2）
    # =================================================================
    device = f'cuda:{comm.get_local_rank()}' if dist.get_world_size() > 1 else 'cuda'
    logger.info(f'初始化教师模型 (device={device})...')

    teacher = TeacherDetAny3D(
        device=device,
        use_sam2_mask=True,
        use_ram_gpt=False,  # 训练时用固定 prompt 加速
    )

    # =================================================================
    # 学生模型初始化（DINOv2 + Student3DHead）
    # =================================================================
    logger.info('初始化学生模型...')
    student_head = Student3DHead(
        in_channels=256,
        roi_size=7,
        hidden_dim=256,
        num_fc=2,
        use_uncertainty=True,
        use_disentangled=True,
        use_chamfer=True,
    ).to(device)

    if dist.get_world_size() > 1:
        student_head = torch.nn.parallel.DistributedDataParallel(
            student_head, device_ids=[comm.get_local_rank()],
            broadcast_buffers=False, find_unused_parameters=True
        )

    # =================================================================
    # 优化器和调度器
    # =================================================================
    optimizer = torch.optim.AdamW(student_head.parameters(), lr=cfg.SOLVER.BASE_LR,
                                  weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # =================================================================
    # 损失权重
    # =================================================================
    loss_weights = {
        "delta_xy": cfg.MODEL.ROI_CUBE_HEAD.get("LOSS_W_XY", 1.0),
        "z": cfg.MODEL.ROI_CUBE_HEAD.get("LOSS_W_Z", 1.0),
        "dims": cfg.MODEL.ROI_CUBE_HEAD.get("LOSS_W_DIMS", 1.0),
        "pose": cfg.MODEL.ROI_CUBE_HEAD.get("LOSS_W_POSE", 1.0),
        "uncertainty": 0.1,
        "disentangled": 1.0,
        "chamfer": 1.0,
        "joint": cfg.MODEL.ROI_CUBE_HEAD.get("LOSS_W_JOINT", 0.0),
    }

    # =================================================================
    # 数据加载器（使用蒸馏专用数据集）
    # =================================================================
    logger.info('构建蒸馏数据加载器...')
    data_loader = build_distill_train_loader(cfg, datasets.dataset["images"], is_train=True)

    # =================================================================
    # 训练循环
    # =================================================================
    remaining_attempts = MAX_TRAINING_ATTEMPTS
    while remaining_attempts > 0:
        if do_distill_train(
            cfg, teacher, student_head, optimizer, scheduler,
            data_loader, device,
            text_embeddings=None,  # 学生头不需要 text_embeddings
            image_size_hw=(896, 896),
            resume=args.resume,
            use_lshape=True,
            loss_weights=loss_weights,
            use_uncertainty=True,
            use_disentangled=True,
            use_chamfer=True,
            use_3d_nms=False,
        ):
            break
        else:
            remaining_attempts -= 1
            del student_head
            torch.cuda.empty_cache()
            # 重新初始化
            student_head = Student3DHead(
                in_channels=256, roi_size=7, hidden_dim=256, num_fc=2,
                use_uncertainty=True, use_disentangled=True, use_chamfer=True,
            ).to(device)
            optimizer = torch.optim.AdamW(student_head.parameters(), lr=cfg.SOLVER.BASE_LR,
                                          weight_decay=cfg.SOLVER.WEIGHT_DECAY)
            scheduler = build_lr_scheduler(cfg, optimizer)

    if remaining_attempts == 0:
        raise ValueError('蒸馏训练失败')

    # 保存最终模型
    if comm.is_main_process():
        ckpt_path = os.path.join(cfg.OUTPUT_DIR, "student_3d_head_final.pth")
        torch.save(student_head.state_dict(), ckpt_path)
        logger.info(f'保存最终模型: {ckpt_path}')

    return 0


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--resume", action="store_true", help="恢复训练")
    parser.add_argument("--config-file", type=str,
                        default="configs/Base_Omni3D_SUN.yaml",
                        help="配置文件路径")
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
