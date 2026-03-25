# Copyright (c) Teacher-Student Distillation Pipeline
"""
教师：DetAny3D 前端 (固定) -> Mask + Depth -> 几何模块 -> 伪 3D 框
"""
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from cubercnn.util import math_util as util_math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUNDED_SAM_DIR = os.path.join(PROJECT_ROOT, "Grounded-SAM-2")
GDINO_PKG_DIR = os.path.join(GROUNDED_SAM_DIR, "grounding_dino")
for _p in (GDINO_PKG_DIR, GROUNDED_SAM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from groundingdino.util.inference import load_model as load_gdino, predict as gdino_predict
import groundingdino.datasets.transforms as T

from detany3d_frontend.image_encoder import ImageEncoderViT
from detany3d_frontend.prompt_encoder import PromptEncoder
from detany3d_frontend.mask_decoder import MaskDecoder
from types import SimpleNamespace
import torch.nn as nn

from .teacher_geometry import run_teacher_pipeline_per_instance

# 复用室内先验
from cubercnn.generate_label.priors import llm_generated_prior
from cubercnn.generate_label.util import extract_ground

# 改进1和3: RAM+GPT标签提取
from .ram_gpt_labeler import RAMGPTLabeler, create_ram_gpt_labeler

# MoGe+DepthPro 深度融合
from detany3d_frontend.depth_predictor.moge_depthpro_fusion import MoGeLoader, DepthProLoader, align_depth_ransac


class OVM3DCompatibleTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src, pos_src, tokens, control_q=None, control_k=None):
        return tokens.contiguous(), src.contiguous()


class TeacherDetAny3D:
    """教师模型：DetAny3D 前端 + 几何伪标签。Mask 可选 SAM2（纯 2D）或 DetAny3D decoder（3D 向）。"""

    def __init__(self, device="cuda", detany3d_ckpt=None, use_sam2_mask=True,
                 use_ram_gpt=True, ram_model_path=None, gemini_api_key=None,
                 restrict_masks_to_boxes=False, debug=False):
        """
        use_sam2_mask: True 时用 SAM2 根据 2D 框生成纯 2D 前景 mask；
                       False 时用 DetAny3D 的 mask_decoder（与 depth 联合训练，易出全图/噪点）。
        use_ram_gpt: True 时使用RAM+Gemini自动生成text prompt, False时使用手动text_prompt
        ram_model_path: RAM模型权重路径
        gemini_api_key: Google Gemini API密钥(可选,也可通过环境变量 GEMINI_API_KEY 设置)
        restrict_masks_to_boxes: False 时允许 mask 溢出 2D 框（推荐），True 时裁剪到框内（旧行为）
        """
        self.device = device
        self.use_sam2_mask = use_sam2_mask
        self.use_ram_gpt = use_ram_gpt
        self.restrict_masks_to_boxes = restrict_masks_to_boxes
        self.debug = debug  # 控制是否保存 debug 可视化
        self.cfg = self._build_config()
        
        # 初始化RAM+GPT标签器(改进1和3)
        self.ram_gpt_labeler = None
        if use_ram_gpt:
            try:
                self.ram_gpt_labeler = create_ram_gpt_labeler(
                    device=device,
                    ram_model_path=ram_model_path,
                    gemini_api_key=gemini_api_key,
                    use_gemini=True
                )
                print(">> Teacher: RAM+Gemini标签器已初始化")
            except Exception as e:
                print(f">> Teacher: RAM+Gemini标签器初始化失败: {e}, 将使用手动text_prompt")
                self.use_ram_gpt = False
        detany3d_ckpt = detany3d_ckpt or os.path.join(
            GROUNDED_SAM_DIR, "checkpoints", "zero_shot_category_ckpt-002.pth"
        )

        # Grounding DINO
        print(">> Teacher: 加载 Grounding DINO...")
        self.gdino_model = load_gdino(
            os.path.join(GROUNDED_SAM_DIR, "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
            os.path.join(GROUNDED_SAM_DIR, "checkpoints/groundingdino_swint_ogc.pth"),
        )
        self.gdino_transform = T.Compose([
            T.RandomResize([800], max_size=1333), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # DetAny3D 前端
        print(">> Teacher: 加载 ImageEncoderViT + PromptEncoder + MaskDecoder...")
        self.image_encoder = ImageEncoderViT(
            img_size=self.cfg.model.pad,
            patch_size=16,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            cfg=self.cfg,
            device=device,
        ).to(device)
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(896 // 16, 896 // 16),
            input_image_size=(896, 896),
            mask_in_chans=16,
        ).to(device)
        self.mask_decoder = MaskDecoder(
            cfg=self.cfg,
            transformer_dim=256,
            transformer=OVM3DCompatibleTransformer(),
            num_multimask_outputs=3,
        ).to(device)
        if not hasattr(self.mask_decoder, "transformer2"):
            self.mask_decoder.transformer2 = OVM3DCompatibleTransformer().to(device)

        self._load_detany3d_weights(detany3d_ckpt)
        self.image_encoder.eval()
        self.prompt_encoder.eval()
        self.mask_decoder.eval()

        self._sam2_predictor = None  # 懒加载，仅当 use_sam2_mask 时用
        self.prior_dict = llm_generated_prior.get("SUNRGBD", {})

        # MoGe + DepthPro 深度融合 (与 demo_teacher_lightweight.py 一致)
        self.moge_loader = None
        self.depthpro_loader = None
        self._init_moge_depthpro()

        print(">> Teacher 就绪 (仅前向，不反传)." + (" [Mask=SAM2]" if use_sam2_mask else " [Mask=DetAny3D]"))

    def _build_config(self):
        c = SimpleNamespace(
            dino_path=os.path.join(PROJECT_ROOT, "weights", "dinov2_vitl14_pretrain.pth"),
            sam_path=os.path.join(PROJECT_ROOT, "weights", "sam_vit_h_4b8939.pth"),
            unidepth_path=os.path.join(PROJECT_ROOT, "weights", "model.pth"),
            model=SimpleNamespace(
                pad=896,
                additional_adapter=True,
                multi_level_box_output=1,
                original_sam=False,
            ),
            contain_edge_obj=False,
            output_rotation_matrix=False,
        )
        c.model.image_encoder = SimpleNamespace(patch_size=16, global_attn_indexes=[7, 15, 23, 31])
        # 启用 MoGe + DepthPro 融合深度（与 demo_teacher_lightweight.py 一致）
        c.use_moge_depthpro = True
        return c

    def _load_detany3d_weights(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"   ⚠️ 未找到 DetAny3D 权重: {ckpt_path}")
            return
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get("model", ckpt)
        for name, module in [("image_encoder", self.image_encoder), ("prompt_encoder", self.prompt_encoder), ("mask_decoder", self.mask_decoder)]:
            prefix = name + "."
            sd = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if sd:
                module.load_state_dict(sd, strict=False)
                print(f"   ✅ {name}: {len(sd)} 个参数")

    def _init_moge_depthpro(self):
        """初始化 MoGe + DepthPro 深度融合模型"""
        try:
            print(">> Teacher: 加载 MoGe + DepthPro...")
            
            # 添加MoGe路径
            moge_root = os.path.join(PROJECT_ROOT, "external", "MoGe")
            if os.path.exists(moge_root):
                sys.path.insert(0, moge_root)
                sys.path.insert(0, os.path.join(moge_root, "moge"))
            
            # 添加ml-depth-pro路径
            depthpro_root = os.path.join(PROJECT_ROOT, "external", "ml-depth-pro", "src")
            if os.path.exists(depthpro_root):
                sys.path.insert(0, depthpro_root)
            
            self.moge_loader = MoGeLoader(device=self.device)
            self.depthpro_loader = DepthProLoader(device=self.device)
            self.moge_loader.load_model()
            self.depthpro_loader.load_model()
            print(">> Teacher: MoGe + DepthPro 加载完成")
        except Exception as e:
            print(f">> Teacher: MoGe+DepthPro加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.moge_loader = None
            self.depthpro_loader = None

    def _get_moge_depthpro_fused(self, rgb_np, K):
        """
        使用 MoGe + DepthPro 融合深度，与 demo_teacher_lightweight.py 一致
        返回: (depth_fused, moge_K)
        """
        if self.moge_loader is None or self.depthpro_loader is None:
            # 回退到 DetAny3D 深度
            return None, K
        
        h, w = rgb_np.shape[:2]
        
        try:
            # MoGe 推理
            moge_result = self.moge_loader.infer(rgb_np)
            if moge_result is None:
                raise Exception("MoGe inference returned None")
            depth_moge = moge_result['depth']
            moge_K = moge_result['intrinsics']
            moge_mask = moge_result.get('mask', None)
            
            # Depth Pro 推理
            focal_length_px = moge_K[0, 0]
            depthpro_result = self.depthpro_loader.infer(rgb_np, focal_length_px=focal_length_px)
            if depthpro_result is None:
                raise Exception("DepthPro inference returned None")
            depth_depthpro = depthpro_result['depth']
            
            # 调整到原图大小
            if depth_moge.shape != (h, w):
                depth_moge = cv2.resize(depth_moge, (w, h), interpolation=cv2.INTER_LINEAR)
                if moge_mask is not None:
                    moge_mask = cv2.resize(moge_mask.astype(np.float32), (w, h)) > 0.5
            if depth_depthpro.shape != (h, w):
                depth_depthpro = cv2.resize(depth_depthpro, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # RANSAC 对齐
            aligned_depth, diag = align_depth_ransac(
                depth_moge, depth_depthpro, mask=moge_mask
            )
            print(f">> Teacher: 对齐诊断 {diag['status']} | scale={diag['scale']:.4f} "
                  f"| 内点率={diag['inlier_ratio']:.1%} | P95误差={diag['p95_error']:.3f}m")
            
            print(f">> Teacher: MoGe+DepthPro 融合深度范围 [{aligned_depth.min():.3f}, {aligned_depth.max():.3f}] m")
            return aligned_depth, moge_K
        except Exception as e:
            print(f">> Teacher: MoGe+DepthPro 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None, K

    def _get_masks_sam2(self, rgb_np, xyxy_np):
        """用 SAM2 根据 2D 框在原图上生成纯 2D 前景 mask，返回 (N, H, W) float32。"""
        if self._sam2_predictor is None:
            try:
                sam2_path = os.path.join(GROUNDED_SAM_DIR, "sam2")
                if sam2_path not in sys.path:
                    sys.path.insert(0, sam2_path)
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                sam2_ckpt = os.path.join(GROUNDED_SAM_DIR, "checkpoints", "sam2.1_hiera_large.pt")
                if not os.path.isfile(sam2_ckpt):
                    sam2_ckpt = os.path.join(PROJECT_ROOT, "weights", "sam2.1_hiera_large.pt")
                if not os.path.isfile(sam2_ckpt):
                    raise FileNotFoundError(f"SAM2.1 Large 权重未找到: {sam2_ckpt}")
                # 使用 facebook/sam2.1-hiera-large 会自动找到正确的配置
                sam2_model = build_sam2("facebook/sam2.1-hiera-large", sam2_ckpt, device=self.device)
                self._sam2_predictor = SAM2ImagePredictor(sam2_model)
                print(">> Teacher: 已加载 SAM2.1 Large (用于 2D Mask).")
            except Exception as e:
                import traceback
                print(f">> Teacher: SAM2 加载失败 ({e})，将回退到 DetAny3D mask_decoder")
                traceback.print_exc()
                self.use_sam2_mask = False
                return None
        xyxy = np.asarray(xyxy_np, dtype=np.float64)
        if xyxy.ndim == 1:
            xyxy = xyxy[np.newaxis, :]
        self._sam2_predictor.set_image(rgb_np)
        # 修复：multimask_output=True + 按面积选最大，避免 SAM2 丢弃完整物体掩码
        masks_all, scores, _ = self._sam2_predictor.predict(
            point_coords=None, point_labels=None, box=xyxy, multimask_output=True,
            normalize_coords=False,  # 使用像素坐标（不是归一化坐标）
        )
        # SAM2 返回 (N, 3, H, W) - 每框 3 个候选，按面积选最大的
        n_boxes = xyxy.shape[0]
        expected_masks = n_boxes * 3
        if masks_all.shape[0] != expected_masks:
            print(f"⚠️ SAM2 返回 {masks_all.shape[0]} 个 mask，期望 {expected_masks}，使用面积最大的")
            # Fallback: take first n_boxes masks (or select best per box)
            if masks_all.shape[0] >= n_boxes:
                final_masks = []
                for i in range(n_boxes):
                    # Try to get 3 candidates if available, otherwise use first available
                    start_idx = i * 3
                    end_idx = min(start_idx + 3, masks_all.shape[0])
                    candidates = masks_all[start_idx:end_idx]
                    if len(candidates) > 1:
                        areas = [int(np.sum(m > 0.5)) for m in candidates]
                        best_idx = int(np.argmax(areas))
                        final_masks.append(candidates[best_idx])
                    else:
                        final_masks.append(candidates[0])
            else:
                final_masks = [masks_all[i] for i in range(masks_all.shape[0])]
        else:
            final_masks = []
            for i in range(n_boxes):
                candidates = masks_all[i * 3:(i + 1) * 3]
                areas = [int(np.sum(m > 0.5)) for m in candidates]
                best_idx = int(np.argmax(areas))
                final_masks.append(candidates[best_idx])
        masks = np.stack(final_masks, axis=0).astype(np.float32)
        return masks

    @staticmethod
    def _restrict_masks_to_boxes(masks, boxes_xyxy, image_shape):
        """
        将 mask 限制在对应 2D 检测框内，避免 SAM2 前景泄漏到墙面/地面。
        masks: (N, H, W)
        boxes_xyxy: (N, 4) in original image coordinates
        """
        if masks is None:
            return None

        masks = np.asarray(masks, dtype=np.float32)
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
        h, w = int(image_shape[0]), int(image_shape[1])
        clipped_masks = []

        for mask_i, box_i in zip(masks, boxes_xyxy):
            x1, y1, x2, y2 = np.round(box_i).astype(np.int32)
            x1 = int(np.clip(x1, 0, max(w - 1, 0)))
            y1 = int(np.clip(y1, 0, max(h - 1, 0)))
            x2 = int(np.clip(x2, 0, max(w - 1, 0)))
            y2 = int(np.clip(y2, 0, max(h - 1, 0)))
            # Ensure x2 >= x1 and y2 >= y1 for valid slicing
            x2 = max(x1 + 1, x2)
            y2 = max(y1 + 1, y2)

            box_mask = np.zeros_like(mask_i, dtype=np.float32)
            box_mask[y1:y2 + 1, x1:x2 + 1] = 1.0
            clipped_masks.append(mask_i * box_mask)

        return np.stack(clipped_masks, axis=0) if clipped_masks else masks

    @torch.no_grad()
    def get_depth_mask_and_boxes(self, rgb_np, text_prompt=None, K=None, box_threshold=0.3, text_threshold=0.25,
                                  boxes_xyxy=None, phrases=None):
        """
        输入: rgb_np (H,W,3), text_prompt(可选,如果use_ram_gpt=True会自动生成), K (3,3)
        可选: boxes_xyxy (N,4), phrases (list) — 若提供则不再跑 DINO，与调用方 2D 检测一致
        输出: depth_np (H,W), masks (N,H,W), boxes_xyxy (N,4), phrases (list), F_fused (tensor), used_K (3,3)
        
        改进1和3: 如果use_ram_gpt=True,会自动使用RAM+GPT生成text_prompt
        """
        h, w = rgb_np.shape[:2]
        
        # 改进1和3: 自动生成text_prompt
        if self.use_ram_gpt and self.ram_gpt_labeler is not None and text_prompt is None:
            print(">> 使用RAM+GPT自动生成text prompt...")
            text_prompt = self.ram_gpt_labeler.generate_text_prompt(rgb_np)
            if text_prompt is None:
                print(">> RAM+GPT生成失败,需要手动提供text_prompt")
                return None, None, None, None, None, None
        
        if text_prompt is None:
            print(">> 错误: 未提供text_prompt且RAM+GPT不可用")
            return None, None, None, None, None, None

        if boxes_xyxy is not None and phrases is not None and len(boxes_xyxy) > 0:
            # 使用调用方传入的 2D 框与类别，只做深度 + SAM mask
            xyxy = torch.from_numpy(np.asarray(boxes_xyxy, dtype=np.float32)).to(self.device)
            if xyxy.dim() == 1:
                xyxy = xyxy.unsqueeze(0)
            print(f">>> 使用调用方 2D 检测: {len(phrases)} 个物体")
        else:
            image_pil = Image.fromarray(rgb_np)
            image_tensor, _ = self.gdino_transform(image_pil, None)
            boxes, scores, phrases = gdino_predict(
                self.gdino_model, image_tensor, text_prompt,
                box_threshold=box_threshold, text_threshold=text_threshold,
            )
            if len(boxes) == 0:
                return None, None, None, None, None, None
            boxes_abs = boxes * torch.Tensor([w, h, w, h]).to(boxes.device)
            xyxy = boxes_abs.clone()
            xyxy[:, 0] = boxes_abs[:, 0] - boxes_abs[:, 2] / 2
            xyxy[:, 1] = boxes_abs[:, 1] - boxes_abs[:, 3] / 2
            xyxy[:, 2] = boxes_abs[:, 0] + boxes_abs[:, 2] / 2
            xyxy[:, 3] = boxes_abs[:, 1] + boxes_abs[:, 3] / 2
            if not self.use_sam2_mask:
                xyxy = xyxy[0:1]
                phrases = phrases[0:1]
            print(f">>> 检测到 {len(phrases)} 个物体" + (f"，使用第1个框: {xyxy[0].cpu().numpy()}" if len(xyxy) == 1 else ""))

        # 使用 MoGe + DepthPro 融合深度（与 demo_teacher_lightweight.py 一致）
        fused_depth, fused_K = self._get_moge_depthpro_fused(rgb_np, K)
        used_K = K if K is not None else fused_K

        # 深度分支
        if fused_depth is not None:
            # 使用 MoGe+DepthPro 融合深度
            depth_np = fused_depth
            used_K = fused_K
            print(f">> 使用 MoGe+DepthPro 融合深度")
        else:
            # 回退到 DetAny3D 深度
            depth_np = None

        # =============================================================
        # 始终运行 DINOv2 编码器获取 F_fused（学生头的输入特征）
        # 与深度选择独立
        # =============================================================
        pad_img = cv2.copyMakeBorder(
            rgb_np,
            0,
            max(0, 896 - h),
            0,
            max(0, 896 - w),
            cv2.BORDER_CONSTANT,
        )[:896, :896]
        pad_tensor = torch.from_numpy(pad_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        img_t_dino = (
            pad_tensor / 255.0 - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        ) / torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        gt_intrinsic = torch.eye(4).float().unsqueeze(0).to(self.device)
        gt_intrinsic[0, :3, :3] = torch.tensor(used_K, dtype=torch.float32, device=self.device)
        input_dict = {
            "images": pad_tensor,  # SAM 用（会做归一化）
            "image_for_dino": img_t_dino,
            "vit_pad_size": torch.tensor([[h // 16, w // 16]], dtype=torch.long, device=self.device),
            "gt_intrinsic": gt_intrinsic,
        }
        output_dict = self.image_encoder(input_dict)
        F_fused = output_dict["image_embeddings"]

        # 如果深度还是 None（MoGe+DepthPro 也失败了），用 DetAny3D 的深度
        if depth_np is None:
            depth_map_raw = output_dict["depth_maps"]
            if depth_map_raw.dim() == 5:
                depth_map_raw = depth_map_raw[0, 0]  # (H, W)
            elif depth_map_raw.dim() == 4:
                depth_map_raw = depth_map_raw[0, 0]
            dh, dw = depth_map_raw.shape[-2], depth_map_raw.shape[-1]
            # Safe slicing: ensure we don't exceed available depth map size
            h_slice = min(h, dh)
            w_slice = min(w, dw)
            depth_np = np.ascontiguousarray(depth_map_raw[:h_slice, :w_slice].cpu().numpy())
            # If depth map is smaller than expected, warn and pad if needed
            if dh < h or dw < w:
                print(f"⚠️ DetAny3D 深度图 ({dh}x{dw}) 小于原图 ({h}x{w})，已裁剪")
            print(f">> 使用 DetAny3D 回退深度 (shape={depth_np.shape}, 原图={h}x{w})")
        else:
            print(f">> Teacher: 融合深度范围 [{depth_np.min():.3f}, {depth_np.max():.3f}] m")

        # Mask：优先用 SAM2（纯 2D 前景），但仍限制在原始 2D 检测框内，避免几何拟合吃进背景。
        if self.use_sam2_mask:
            masks = self._get_masks_sam2(rgb_np, xyxy.cpu().numpy())
            masks = self._restrict_masks_to_boxes(masks, xyxy.cpu().numpy(), (h, w))
        else:
            masks = None
        if masks is None:
            boxes_tensor = xyxy.float().to(self.device).unsqueeze(0)
            sparse_emb, dense_emb = self.prompt_encoder(points=None, boxes=boxes_tensor, masks=None)
            for key in ["metric_features", "camera_features"]:
                if output_dict.get(key) is not None:
                    output_dict[key] = output_dict[key].reshape(output_dict[key].shape[0], -1, 1, 1)
            if output_dict.get("depth_features") is not None:
                df = output_dict["depth_features"]
                df = df.permute(0, 3, 1, 2) if df.dim() == 4 else df
                output_dict["depth_features"] = torch.nn.functional.interpolate(
                    df, size=(F_fused.shape[2], F_fused.shape[3]), mode="bilinear", align_corners=False
                )
            decoder_out = self.mask_decoder(
                input_dict=output_dict,
                image_embeddings=F_fused,
                image_pe=self.prompt_encoder.get_dense_pe(),
                metric_feature=output_dict.get("metric_features"),
                camera_feature=output_dict.get("camera_features"),
                depth_feature=output_dict.get("depth_features"),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            masks_tensor = decoder_out["masks"]
            if masks_tensor.dim() == 5:
                mask_logits = masks_tensor[0, :, 0, :h, :w]
            elif masks_tensor.dim() == 4:
                mask_logits = masks_tensor[0, :, :h, :w]
            else:
                raise ValueError(f"Unexpected mask tensor shape: {tuple(masks_tensor.shape)}")
            masks = (mask_logits > 0.0).float().cpu().numpy()

        # 返回深度、掩码、2D框、类别、特征图、以及使用的K
        return depth_np, masks, xyxy.cpu().numpy(), phrases, F_fused, used_K

    def generate_pseudo_3d_boxes(self, rgb_np, text_prompt=None, K=None, ground_equ=None, use_lshape=False,
                                  boxes_xyxy=None, phrases=None, depth_scale=1.0):
        """
        完整教师线：图像 -> 深度+Mask+2D框 -> 逐实例几何 -> 伪 3D 框列表
        若传入 boxes_xyxy 与 phrases，则不再跑 DINO，与调用方 2D 检测一致。
        depth_scale: 深度转米的缩放，若 Unidepth 整体偏小可设为 2.0 或 3.0。
        
        重要: use_lshape=False 使用 PCA 分支 + proposal 优化，与原模型一致
              use_lshape=True 使用 L-shape 分支（无 proposal 优化）
        
        改进1和3: text_prompt可选,如果use_ram_gpt=True会自动生成
        
        Returns:
            pseudo_list: 与 phrases 同长，成功项为 dict(center_cam, dimensions, R_cam, ...)，失败为 None
            F_fused: (1, C, H, W) 特征图，供学生头使用
        """
        depth_np, masks, boxes_xyxy, phrases, F_fused, used_K = self.get_depth_mask_and_boxes(
            rgb_np, text_prompt, K,
            boxes_xyxy=boxes_xyxy,
            phrases=phrases,
        )
        if depth_np is None or phrases is None:
            return [], None
        rgb_h, rgb_w = rgb_np.shape[:2]
        depth_h, depth_w = depth_np.shape[:2] if depth_np is not None else (0, 0)

        # === Debug 可视化：2D 框 + Mask（原图分辨率）===
        # 只有开启 debug 时才保存，避免磁盘爆满
        if self.debug:
            debug_img = rgb_np.copy()
            for i in range(len(phrases)):
                box = boxes_xyxy[i].astype(int)
                cv2.rectangle(debug_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                mask_i = masks[i]
                if mask_i is not None and mask_i.sum() > 0 and depth_np is not None:
                    mask_bool = mask_i > 0.5
                    debug_img[mask_bool] = (debug_img[mask_bool] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
                cv2.putText(debug_img, phrases[i], (box[0], max(0, box[1]-5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            import time
            debug_path = f"debug_2d_dino_sam_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
            print(f"Debug 2D 图已保存到: {debug_path}")
        # ======================================
        
        # 与 phrases/boxes 一一对齐，失败的位置为 None，便于可视化按索引匹配
        pseudo_list = [None] * len(phrases)

        for i in range(len(phrases)):
            mask_i = masks[i].copy()
            if mask_i.sum() < 10:
                continue
            center_cam, dimensions, R_cam, ok = run_teacher_pipeline_per_instance(
                depth_np,
                mask_i,
                used_K,  # 使用融合后的K
                phrases[i],
                self.prior_dict,
                ground_equ=ground_equ,
                use_lshape=use_lshape,
                depth_scale=depth_scale,
                image_width=rgb_w,
                image_height=rgb_h,
                debug=True,
            )
            if not ok:
                continue
            # ===== 调试：投影 3D 框到图像，验证是否合理 =====
            # teacher_geometry 返回 dimensions = [W, H, L] = [dz, dy, dx]
            dims = np.array(dimensions)
            W, H, L = dims[0], dims[1], dims[2]
            cx_c, cy_c, cz_c = center_cam[0], center_cam[1], center_cam[2]
            box3d_arr = np.array([[cx_c, cy_c, cz_c, W, H, L]], dtype=np.float32)
            box3d_t = torch.from_numpy(box3d_arr).float()
            R_t = torch.from_numpy(R_cam).float()
            verts_t, _ = util_math.get_cuboid_verts_faces(box3d_t, R_t)
            verts_np = verts_t.squeeze(0).cpu().numpy()
            K_vis = np.array(used_K, dtype=np.float64)
            z_vals = verts_np[:, 2]
            if i == 0:
                print(f"    [DEBUG verts] phrase={phrases[i]}, verts Z: [{z_vals.min():.3f}, {z_vals.max():.3f}]")
                print(f"    [DEBUG verts] sample: {verts_np[:2]}")
            proj = (K_vis @ verts_np.T).T
            proj[:, 0] /= proj[:, 2]
            proj[:, 1] /= proj[:, 2]
            proj_2d = proj[:, :2]
            x_min, x_max = proj_2d[:, 0].min(), proj_2d[:, 0].max()
            y_min, y_max = proj_2d[:, 1].min(), proj_2d[:, 1].max()
            x1_p, y1_p, x2_p, y2_p = x_min, y_min, x_max, y_max
            x1_g, y1_g, x2_g, y2_g = boxes_xyxy[i]
            # IoU 在原图空间
            xi1, yi1, xi2, yi2 = max(x1_p, x1_g), max(y1_p, y1_g), min(x2_p, x2_g), min(y2_p, y2_g)
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area_p = (x2_p - x1_p) * (y2_p - y1_p)
            area_g = (x2_g - x1_g) * (y2_g - y1_g)
            iou = inter / max(area_p + area_g - inter, 1e-6)
            # 中心对齐度
            cx_p, cy_p = (x1_p + x2_p) / 2, (y1_p + y2_p) / 2
            cx_g, cy_g = (x1_g + x2_g) / 2, (y1_g + y2_g) / 2
            align = max(0, 1 - np.sqrt((cx_p - cx_g)**2 + (cy_p - cy_g)**2) / max(depth_w, depth_h))
            status = "✅ 对齐良好" if iou > 0.3 else ("⚠️ 部分对齐" if iou > 0.1 else "❌ 严重偏差")
            print(f"  [TEACHER GEO CHECK] phrase={phrases[i]}")
            print(f"    center_cam = [{cx_c:.3f}, {cy_c:.3f}, {cz_c:.3f}]  ← camera coords (X=right, Y=up, Z=depth)")
            print(f"    dimensions = [{dims[0]:.3f}, {dims[1]:.3f}, {dims[2]:.3f}]  ← [W=dz, H=dy, L=dx]")
            print(f"    3D→2D proj (orig) = [{x1_p:.0f}, {y1_p:.0f}, {x2_p:.0f}, {y2_p:.0f}]  image=[{rgb_w},{rgb_h}]")
            print(f"    2D检测框 (orig)   = [{x1_g:.0f}, {y1_g:.0f}, {x2_g:.0f}, {y2_g:.0f}]")
            print(f"    iou={iou:.3f}, align={align:.3f}  → {status}")
            # =================================================
            pseudo_list[i] = {
                "center_cam": center_cam,
                "dimensions": dimensions,
                "R_cam": R_cam,
                "phrase": phrases[i],
                "box_2d_xyxy": boxes_xyxy[i],
                "K": used_K,  # 保存融合后的K用于可视化
            }
        return pseudo_list, F_fused
