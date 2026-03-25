import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from types import SimpleNamespace
from tqdm import tqdm

# ==========================================
# 🌟 0. 环境与依赖路径配置
# ==========================================
GROUNDED_SAM_DIR = "/data/ZhaoX/OVM3D-Det-1/Grounded-SAM-2"
sys.path.append(GROUNDED_SAM_DIR)

from groundingdino.util.inference import load_model as load_gdino, predict as gdino_predict
import groundingdino.datasets.transforms as T

from detany3d_frontend.image_encoder import ImageEncoderViT
from detany3d_frontend.prompt_encoder import PromptEncoder
from detany3d_frontend.mask_decoder import MaskDecoder
from detany3d_frontend.transformer import TwoWayTransformer

# ==========================================
# ⚙️ 1. 模型权重与全局配置
# ==========================================
DETANY3D_CKPT = "/data/ZhaoX/OVM3D-Det-1/Grounded-SAM-2/checkpoints/zero_shot_category_ckpt-002.pth"
GDINO_CONFIG = os.path.join(GROUNDED_SAM_DIR, "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GDINO_CKPT = os.path.join(GROUNDED_SAM_DIR, "checkpoints/groundingdino_swint_ogc.pth")

TEXT_PROMPT = "bicycle. books. bottle. chair. cup. laptop. shoes. towel. blinds. window. lamp. shelves. mirror. sink. cabinet. bathtub. door. toilet. desk. box. bookcase. picture. table. counter. bed. night stand. pillow. sofa. television. floor mat. curtain. clothes. stationery. refrigerator. bin. stove. oven. machine."
INTRINSICS = np.array([[529.5, 0.0, 365.0], [0.0, 529.5, 262.0], [0.0, 0.0, 1.0]])

# ==========================================
# 🚀 2. 核心架构类: 纯正 DetAny3D 级流水线
# ==========================================
class Step1_PseudoPointCloudGenerator:
    def __init__(self, device="cuda"):
        self.device = device
        self.cfg = self._build_cfg()
        
        print(">> 🧱 按照官方架构图，构建 2D Aggregator 与 Mask Decoder...")
        
        self.image_encoder = ImageEncoderViT(img_size=896, patch_size=16, embed_dim=1280, depth=32, num_heads=16, cfg=self.cfg).to(device)
        self.prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(56, 56), input_image_size=(896, 896), mask_in_chans=16).to(device)
        
        self.mask_decoder = MaskDecoder(
            cfg=self.cfg, 
            transformer_dim=256, 
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8,
                inject_layer=0  
            ), 
            num_multimask_outputs=3
        ).to(device)
        self.mask_decoder.initzeroconv()

        print(f">> 💉 加载融合特征与深度估计权重 (DetAny3D Checkpoint)...")
        ckpt = torch.load(DETANY3D_CKPT, map_location=device)
        state_dict = ckpt.get('model', ckpt)
        self.image_encoder.load_state_dict(state_dict, strict=False)
        self.prompt_encoder.load_state_dict(state_dict, strict=False)
        self.mask_decoder.load_state_dict(state_dict, strict=False)
        
        print(">> 🦅 加载 Grounding DINO (2D BBox 引擎)...")
        self.gdino = load_gdino(GDINO_CONFIG, GDINO_CKPT)
        self.gdino_t = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def _build_cfg(self):
        c = SimpleNamespace()
        c.model = SimpleNamespace(pad=896, additional_adapter=True, multi_level_box_output=1, original_sam=False)
        c.model.image_encoder = SimpleNamespace(patch_size=16, global_attn_indexes=[7, 15, 23, 31])
        c.contain_edge_obj = False
        c.output_rotation_matrix = False
        c.sam_path = "/data/ZhaoX/OVM3D-Det-1/weights/sam_vit_h_4b8939.pth"
        c.dino_path = "/data/ZhaoX/OVM3D-Det-1/weights/dinov2_vitl14_pretrain.pth"
        c.unidepth_path = "/data/ZhaoX/OVM3D-Det-1/weights/model.pth"
        return c

    @torch.no_grad()
    def process_image(self, img_path, target_labels=None):
        raw_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = raw_img.shape[:2]
        
        current_prompt = ". ".join(list(set(target_labels))) + "." if target_labels else TEXT_PROMPT

        # ---------------------------------------------------------
        # 步骤 A: Grounding DINO 提取 2D 框
        # ---------------------------------------------------------
        image_pil = Image.fromarray(raw_img)
        it, _ = self.gdino_t(image_pil, None)
        
        predict_results = gdino_predict(model=self.gdino, image=it, caption=current_prompt, box_threshold=0.3, text_threshold=0.25)
        boxes, phrases = predict_results[0], predict_results[2]
        
        if len(boxes) == 0: return []
        print(f"✅ 成功锁定 {len(boxes)} 个目标: {phrases}")

        boxes_abs = boxes * torch.Tensor([w, h, w, h])
        xyxy = boxes_abs.clone()
        xyxy[:, 0:2] -= boxes_abs[:, 2:4] / 2
        xyxy[:, 2:4] = xyxy[:, 0:2] + boxes_abs[:, 2:4]

        # ---------------------------------------------------------
        # 步骤 B: 2D Aggregator 提取终极融合特征与高清深度
        # ---------------------------------------------------------
        print("🧬 [2D Aggregator] 提取全图终极融合特征与深度...")
        pad_img = cv2.copyMakeBorder(raw_img, 0, max(0, 896-h), 0, max(0, 896-w), cv2.BORDER_CONSTANT)[:896, :896]
        img_t = torch.from_numpy(pad_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        
        out = self.image_encoder({
            "images": img_t, "image_for_dino": img_t, 
            "vit_pad_size": torch.tensor([[h//16, w//16]], device=self.device), 
            "gt_intrinsic": torch.tensor(INTRINSICS).float().unsqueeze(0).to(self.device)
        })
        
        F_fused = out["image_embeddings"]
        depth_map = out["depth_maps"][0, 0, :h, :w]

        for name in ["metric_features", "camera_features"]:
            if out.get(name) is not None:
                out[name] = out[name].reshape(out[name].shape[0], -1, 1, 1)
        if out.get("depth_features") is not None:
            df = out["depth_features"].permute(0, 3, 1, 2) if out["depth_features"].dim() == 4 else out["depth_features"]
            out["depth_features"] = F.interpolate(df, size=(F_fused.shape[2], F_fused.shape[3]), mode="bilinear", align_corners=False)

        # ---------------------------------------------------------
        # 步骤 C & D: SAM Mask Decoder -> 物理反投影
        # ---------------------------------------------------------
        print("🎭 [Mask Decoder] 结合空间 Prompt，解码纯净 3D 点云...")
        fx, fy, cx, cy = INTRINSICS[0,0], INTRINSICS[1,1], INTRINSICS[0,2], INTRINSICS[1,2]
        instances_data = [] 
        
        for idx in range(len(xyxy)):
            box = xyxy[idx].unsqueeze(0).unsqueeze(0).to(self.device) 
            label = phrases[idx]
            
            sparse_emb, dense_emb = self.prompt_encoder(points=None, boxes=box, masks=None)
            
            decoder_out = self.mask_decoder(
                input_dict=out, 
                image_embeddings=F_fused, 
                image_pe=self.prompt_encoder.get_dense_pe(),
                metric_feature=out["metric_features"], camera_feature=out["camera_features"], depth_feature=out["depth_features"], 
                sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb, multimask_output=False
            )
            
            # 🌟 破案核心修复：将 1/4 微缩蒙版放大回 896x896 的原尺寸！
            upscaled_masks = F.interpolate(decoder_out["masks"], size=(896, 896), mode="bilinear", align_corners=False)
            mask_logits = upscaled_masks[0, 0, :h, :w]
            
            obj_mask = mask_logits > 0.0
            v, u = torch.where(obj_mask)
            z = depth_map[v, u]
            
            valid = (z > 0.5) & (z < 8.0)
            u_valid, v_valid, z_valid = u[valid], v[valid], z[valid]
            
            box_np = xyxy[idx].cpu().numpy()
            xmin, ymin, xmax, ymax = box_np[0], box_np[1], box_np[2], box_np[3]
            in_box = (u_valid >= xmin) & (u_valid <= xmax) & (v_valid >= ymin) & (v_valid <= ymax)
            u_final, v_final, z_final = u_valid[in_box], v_valid[in_box], z_valid[in_box]
            
            if len(z_final) < 30: 
                print(f"   ⚠️ 目标 [{label}] 有效点太少({len(z_final)})，已丢弃。")
                continue
                
            points = torch.stack([(u_final - cx) * z_final / fx, (v_final - cy) * z_final / fy, z_final], dim=1).cpu().numpy()
            
            instances_data.append({"label": label.replace(" ", "_"), "box_2d": box_np, "points": points})
            
        return instances_data

# ==========================================
# 🧪 3. 测试与保存入口
# ==========================================
if __name__ == "__main__":
    TEST_IMG = "/data/ZhaoX/OVM3D-Det-1/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg"
    OUTPUT_DIR = "/data/ZhaoX/OVM3D-Det-1/detany3d_pts"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"========== 启动 Step 1: 原生 DetAny3D 级点云提炼工厂 ==========")
    generator = Step1_PseudoPointCloudGenerator(device="cuda")
    
    mock_gt_labels_for_this_image = ["bed", "chair", "picture", "pillow"] 
    instances = generator.process_image(TEST_IMG, target_labels=mock_gt_labels_for_this_image)
    
    if instances and len(instances) > 0:
        print(f"\n🎉 提炼完成！共分离出 {len(instances)} 个带语义的绝对物理 3D 目标！")
        for i, inst in enumerate(instances):
            save_name = f"obj_{i}_{inst['label']}.npy"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            np.save(save_path, inst["points"])
            print(f"  ✅ 目标 {i} [{inst['label']}]: 提取了 {inst['points'].shape[0]} 个 3D 点 -> {save_name}")
            
    torch.cuda.empty_cache()