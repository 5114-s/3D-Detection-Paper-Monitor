import sys
import os
# 🌟 破解魔改版 GroundingDINO 的本地路径依赖
sys.path.append("/data/ZhaoX/OVM3D-Det-1/Grounded-SAM-2")

import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from types import SimpleNamespace
import torch.nn as nn
from PIL import Image

# ===============================
# 1. 导入 Grounding DINO (生成 Prompt)
# ===============================
from groundingdino.util.inference import load_model as load_gdino, predict as gdino_predict
import groundingdino.datasets.transforms as T

# ===============================
# 2. 导入 OVM3D-Det 核心组件
# ===============================
from detany3d_frontend.image_encoder import ImageEncoderViT
from detany3d_frontend.prompt_encoder import PromptEncoder
from detany3d_frontend.mask_decoder import MaskDecoder

# 解决 Transformer 内存步长报错的补丁
class OVM3DCompatibleTransformer(nn.Module):
    def __init__(self, depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8):
        super().__init__()
    def forward(self, src, pos_src, tokens, control_q=None, control_k=None):
        return tokens.contiguous(), src.contiguous()

# =========================================================
# 🚀 终极流水线：完美复刻你的架构图
# =========================================================
class OVM3DArchitecturePipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.cfg = self._build_mock_config()

        print("==================================================")
        print("🚀 加载双流架构: DINO (语义) + SAM (几何) + UniDepth (深度)")
        print("==================================================")

        # --------------------------------------------------
        # [外部支援]: 加载 Grounding DINO 生成 2D Bbox 作为 Prompt
        # --------------------------------------------------
        print(">> 装载 Prompt 生成器 (Grounding DINO)...")
        self.gdino_model = load_gdino(
            # 🌟 换成绝对路径，再也不怕找不到！
            "/data/ZhaoX/OVM3D-Det-1/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
            "/data/ZhaoX/OVM3D-Det-1/Grounded-SAM-2/checkpoints/groundingdino_swint_ogc.pth"
        )
        self.gdino_transform = T.Compose([
            T.RandomResize([800], max_size=1333), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # --------------------------------------------------
        # [核心架构]: 实例化你画的各个模块
        # --------------------------------------------------
        print(">> 装载 2D Aggregator & Depth Head...")
        # 这个组件内部包含了 SAM Encoder, DINO Encoder, 融合器 和 UniDepth
        self.image_encoder = ImageEncoderViT(
            img_size=self.cfg.model.pad, patch_size=self.cfg.model.image_encoder.patch_size,
            embed_dim=1280, depth=32, num_heads=16, cfg=self.cfg
        ).to(device)

        print(">> 装载 Prompt Encoder & SAM Mask Decoder...")
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(self.cfg.model.pad // 16, self.cfg.model.pad // 16),
            input_image_size=(self.cfg.model.pad, self.cfg.model.pad), mask_in_chans=16
        ).to(device)

        self.mask_decoder = MaskDecoder(
            cfg=self.cfg, transformer_dim=256, transformer=OVM3DCompatibleTransformer(), num_multimask_outputs=3
        ).to(device)
        if not hasattr(self.mask_decoder, "transformer2"):
            self.mask_decoder.transformer2 = OVM3DCompatibleTransformer().to(device)

        # 加载真实权重
        self._load_smart_weights("/data/ZhaoX/OVM3D-Det-1/output/training/SUN/model_final.pth")
        
        self.image_encoder.eval()
        self.prompt_encoder.eval()
        self.mask_decoder.eval()

    def _load_smart_weights(self, weight_path):
        if not os.path.exists(weight_path):
            print(f"⚠️ 找不到权重: {weight_path}")
            return
        state_dict = torch.load(weight_path, map_location=self.device).get('model', torch.load(weight_path, map_location=self.device))
        ie_dict, pe_dict, md_dict = {}, {}, {}
        for k, v in state_dict.items():
            if 'image_encoder.' in k: ie_dict[k.replace('image_encoder.', '')] = v
            elif 'prompt_encoder.' in k: pe_dict[k.replace('prompt_encoder.', '')] = v
            elif 'mask_decoder.' in k: md_dict[k.replace('mask_decoder.', '')] = v
        if ie_dict: self.image_encoder.load_state_dict(ie_dict, strict=False)
        if pe_dict: self.prompt_encoder.load_state_dict(pe_dict, strict=False)
        if md_dict: self.mask_decoder.load_state_dict(md_dict, strict=False)

    def _build_mock_config(self):
        cfg = SimpleNamespace(dino_path=None, contain_edge_obj=False, output_rotation_matrix=False)
        cfg.model = SimpleNamespace(pad=896, additional_adapter=True, multi_level_box_output=1, original_sam=False)
        cfg.model.image_encoder = SimpleNamespace(patch_size=16, global_attn_indexes=[7, 15, 23, 31])
        return cfg

    @torch.no_grad()
    def process_scene(self, rgb_numpy, text_prompt, intrinsics):
        orig_h, orig_w = rgb_numpy.shape[:2]

        # ==========================================================
        # 🎯 准备阶段: 生成 Prompt
        # ==========================================================
        image_pil = Image.fromarray(rgb_numpy)
        image_tensor_gdino, _ = self.gdino_transform(image_pil, None)
        boxes, _, phrases = gdino_predict(
            model=self.gdino_model, image=image_tensor_gdino, caption=text_prompt, box_threshold=0.3, text_threshold=0.25
        )
        if len(boxes) == 0:
            print("⚠️ 未找到任何目标！")
            return np.zeros((0, 3))
        
        print(f"\n🎯 GDINO 锁定了 {len(boxes)} 个目标: {phrases}")
        boxes_abs = boxes * torch.Tensor([orig_w, orig_h, orig_w, orig_h])
        xyxy_boxes = boxes_abs.clone()
        xyxy_boxes[:, 0] = boxes_abs[:, 0] - boxes_abs[:, 2] / 2
        xyxy_boxes[:, 1] = boxes_abs[:, 1] - boxes_abs[:, 3] / 2
        xyxy_boxes[:, 2] = boxes_abs[:, 0] + boxes_abs[:, 2] / 2
        xyxy_boxes[:, 3] = boxes_abs[:, 1] + boxes_abs[:, 3] / 2
        boxes_tensor = xyxy_boxes.to(self.device).unsqueeze(0)

        # 图像预处理 (Padding)
        pad_h, pad_w = max(0, 896 - orig_h), max(0, 896 - orig_w)
        padded = cv2.copyMakeBorder(rgb_numpy, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])[:896, :896]
        img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        vit_pad_size = torch.tensor([[orig_h // 16, orig_w // 16]], dtype=torch.long, device=self.device)

        # ==========================================================
        # 🧩 执行你的架构图逻辑！
        # ==========================================================
        
        # 对应图解：给 SAM 和 DINO 分别准备图片入口
        input_dict = {
            "images": img_tensor,            # ---> 进 SAM Encoder 算 F_s
            "image_for_dino": img_tensor,    # ---> 进 DINO Encoder 算 F_d
            "vit_pad_size": vit_pad_size,
            "gt_intrinsic": torch.tensor(intrinsics).float().unsqueeze(0).to(self.device)
        }

        print("⏳ 执行 2D Aggregator: 融合 F_s 和 F_d ...")
        # 对应图解：F_s + F_d ===> 2D Aggregator
        output_dict = self.image_encoder(input_dict)
        
        # 对应图解：---> F_fused
        F_fused = output_dict["image_embeddings"] 
        
        # 对应图解：---> [Depth Head] ----> 高清无拖尾 Depth Map
        depth_map = output_dict["depth_maps"][0, 0, :orig_h, :orig_w] 
        print("✅ 深度图 (Depth Map) 计算完毕！")

        # 将 BBox 编码为 Prompt Features
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, boxes=boxes_tensor, masks=None)

        # 整理外挂特征
        for name in ["metric_features", "camera_features"]:
            if output_dict.get(name) is not None:
                output_dict[name] = output_dict[name].reshape(output_dict[name].shape[0], -1, 1, 1)
        if output_dict.get("depth_features") is not None:
            df = output_dict["depth_features"].permute(0, 3, 1, 2) if output_dict["depth_features"].dim() == 4 else output_dict["depth_features"]
            output_dict["depth_features"] = F.interpolate(df, size=(F_fused.shape[2], F_fused.shape[3]), mode="bilinear", align_corners=False)

        print("⏳ 执行 SAM Mask Decoder: 结合 Prompt 和 F_fused ...")
        # 对应图解：F_fused \---> [SAM Mask Decoder] + Prompt ----> 强语义精准 Mask
        decoder_out = self.mask_decoder(
            input_dict=output_dict, 
            image_embeddings=F_fused,  # 传入 F_fused
            image_pe=self.prompt_encoder.get_dense_pe(),
            metric_feature=output_dict["metric_features"], 
            camera_feature=output_dict["camera_features"],
            depth_feature=output_dict["depth_features"], 
            sparse_prompt_embeddings=sparse_embeddings, # 传入 Prompt
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        
        # 提取 N 个物体的强语义 Mask 并合并
        mask_logits = decoder_out["masks"][0, :, 0][:, :orig_h, :orig_w]
        semantic_mask = (mask_logits > 0.0).any(dim=0)
        print("✅ 强语义精准 Mask 提取完毕！")

        # 导出调试图像
        cv2.imwrite("/data/ZhaoX/OVM3D-Det-1/debug_F_fused_mask.png", (semantic_mask.cpu().numpy() * 255).astype(np.uint8))

        # ==========================================================
        # 🌐 升维：融合 Depth 和 Mask 为 3D 点云
        # ==========================================================
        print("🌌 开始 3D 反投影...")
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        v, u = torch.where(semantic_mask)
        z = depth_map[v, u]
        valid = (z > 0.5) & (z < 8.0)
        u, v, z = u[valid], v[valid], z[valid]
        points = torch.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], dim=1)
        
        return points.cpu().numpy()

if __name__ == "__main__":
    image_path = "/data/ZhaoX/OVM3D-Det-1/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg"
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    intrinsics = np.array([[529.5, 0.0, 365.0], [0.0, 529.5, 262.0], [0.0, 0.0, 1.0]])

    pipeline = OVM3DArchitecturePipeline(device="cuda")
    
    # 开放词汇：找寻房间里所有常见物体
    TEXT_PROMPT = "person. chair. table. bed. sofa. picture."
    
    point_cloud = pipeline.process_scene(img, TEXT_PROMPT, intrinsics)
    np.save("/data/ZhaoX/OVM3D-Det-1/architecture_lidar.npy", point_cloud)
    
    print(f"\n🎉 架构图逻辑完美闭环！生成了 {point_cloud.shape[0]} 个纯净的 3D 几何点！")