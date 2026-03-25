import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from types import SimpleNamespace
import traceback

# 🌟 修复本地路径依赖
sys.path.append("/data/ZhaoX/OVM3D-Det-1/Grounded-SAM-2")

from groundingdino.util.inference import load_model as load_gdino, predict as gdino_predict
import groundingdino.datasets.transforms as T
from detany3d_frontend.image_encoder import ImageEncoderViT
from detany3d_frontend.prompt_encoder import PromptEncoder
from detany3d_frontend.mask_decoder import MaskDecoder

# ==========================================
# ⚙️ 配置中心
# ==========================================
IMG_DIR = "/data/ZhaoX/OVM3D-Det-1/datasets/sunrgbd/sunrgbd_trainval/image"
OUTPUT_DIR = "/data/ZhaoX/OVM3D-Det-1/datasets/sunrgbd/pseudo_lidar_instances"
DETANY3D_CKPT = "/data/ZhaoX/OVM3D-Det-1/detany3d_private/checkpoints/detany3d_ckpts/zero_shot_category_ckpt-002.pth"

# 兜底用的全局 Prompt
DEFAULT_PROMPT = "person. chair. table. bed. sofa. picture. desk. cabinet."
INTRINSICS = np.array([[529.5, 0.0, 365.0], [0.0, 529.5, 262.0], [0.0, 0.0, 1.0]])

os.makedirs(OUTPUT_DIR, exist_ok=True)

class OVM3DCompatibleTransformer(torch.nn.Module):
    def __init__(self, depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8):
        super().__init__()
    def forward(self, src, pos_src, tokens, control_q=None, control_k=None):
        return tokens.contiguous(), src.contiguous()

class DetAny3DGenerator:
    def __init__(self, device="cuda"):
        self.device = device
        cfg = self._build_cfg()
        
        self.image_encoder = ImageEncoderViT(img_size=896, patch_size=16, embed_dim=1280, depth=32, num_heads=16, cfg=cfg).to(device)
        self.prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(56, 56), input_image_size=(896, 896), mask_in_chans=16).to(device)
        self.mask_decoder = MaskDecoder(cfg=cfg, transformer_dim=256, transformer=OVM3DCompatibleTransformer(), num_multimask_outputs=3).to(device)
        if not hasattr(self.mask_decoder, "transformer2"):
            self.mask_decoder.transformer2 = OVM3DCompatibleTransformer().to(device)

        ckpt = torch.load(DETANY3D_CKPT, map_location=device)
        state_dict = ckpt.get('model', ckpt)
        self.image_encoder.load_state_dict(state_dict, strict=False)
        self.prompt_encoder.load_state_dict(state_dict, strict=False)
        self.mask_decoder.load_state_dict(state_dict, strict=False)
        
        self.gdino = load_gdino("/data/ZhaoX/OVM3D-Det-1/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                                "/data/ZhaoX/OVM3D-Det-1/Grounded-SAM-2/checkpoints/groundingdino_swint_ogc.pth")
        self.gdino_t = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def _build_cfg(self):
        c = SimpleNamespace()
        c.model = SimpleNamespace(pad=896, additional_adapter=True, multi_level_box_output=1, original_sam=False)
        c.model.image_encoder = SimpleNamespace(patch_size=16, global_attn_indexes=[7, 15, 23, 31])
        c.contain_edge_obj = False
        c.output_rotation_matrix = False
        return c

    @torch.no_grad()
    def process_one(self, img_path, target_labels=None):
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            raise ValueError(f"图片损坏或无法读取: {img_path}")
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        h, w = raw_img.shape[:2]
        
        # 1. 动态生成 Prompt (Target-Aware)
        current_prompt = ". ".join(list(set(target_labels))) + "." if target_labels else DEFAULT_PROMPT

        image_pil = Image.fromarray(raw_img)
        it, _ = self.gdino_t(image_pil, None)
        boxes, _, phrases = gdino_predict(model=self.gdino, image=it, caption=current_prompt, box_threshold=0.3, text_threshold=0.25)
        
        if len(boxes) == 0: return []

        boxes_abs = boxes * torch.Tensor([w, h, w, h])
        xyxy = boxes_abs.clone()
        xyxy[:, 0:2] -= boxes_abs[:, 2:4] / 2
        xyxy[:, 2:4] = xyxy[:, 0:2] + boxes_abs[:, 2:4]
        
        sparse_emb, dense_emb = self.prompt_encoder(points=None, boxes=xyxy.to(self.device).unsqueeze(0), masks=None)

        # 2. 聚合推理
        pad_img = cv2.copyMakeBorder(raw_img, 0, max(0, 896-h), 0, max(0, 896-w), cv2.BORDER_CONSTANT)[:896, :896]
        img_t = torch.from_numpy(pad_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        
        out = self.image_encoder({"images": img_t, "image_for_dino": img_t, "vit_pad_size": torch.tensor([[h//16, w//16]], device=self.device), 
                                  "gt_intrinsic": torch.tensor(INTRINSICS).float().unsqueeze(0).to(self.device)})
        
        F_fused = out["image_embeddings"]
        depth_map = out["depth_maps"][0, 0, :h, :w]

        for name in ["metric_features", "camera_features"]:
            if out.get(name) is not None: out[name] = out[name].reshape(out[name].shape[0], -1, 1, 1)
        if out.get("depth_features") is not None:
            df = out["depth_features"].permute(0, 3, 1, 2) if out["depth_features"].dim() == 4 else out["depth_features"]
            out["depth_features"] = F.interpolate(df, size=(F_fused.shape[2], F_fused.shape[3]), mode="bilinear", align_corners=False)

        # 3. 恢复了关键的 Mask Decoder！
        decoder_out = self.mask_decoder(
            input_dict=out, image_embeddings=F_fused, image_pe=self.prompt_encoder.get_dense_pe(),
            metric_feature=out["metric_features"], camera_feature=out["camera_features"], depth_feature=out["depth_features"], 
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb, multimask_output=False
        )
        
        mask_logits = decoder_out["masks"][0, :, 0][:, :h, :w]
        
        # 4. 实例级物理反投影
        fx, fy, cx, cy = INTRINSICS[0,0], INTRINSICS[1,1], INTRINSICS[0,2], INTRINSICS[1,2]
        instances_data = []
        
        for idx in range(mask_logits.shape[0]):
            obj_mask = mask_logits[idx] > 0.0
            v, u = torch.where(obj_mask)
            z = depth_map[v, u]
            valid = (z > 0.5) & (z < 8.0)
            u, v, z = u[valid], v[valid], z[valid]
            
            if len(z) < 30: continue
                
            points = torch.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], dim=1).cpu().numpy()
            instances_data.append({"label": phrases[idx].replace(" ", "_"), "points": points})
            
        return instances_data

# ==========================================
# 🚀 工业级批量执行
# ==========================================
if __name__ == "__main__":
    gen = DetAny3DGenerator()
    imgs = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    
    # 获取真实标签的模拟字典 (实际中你需要写一个解析 SUNRGB-D 标注文件的函数)
    # mock_annotations = {"000004.jpg": ["bed", "chair"], "000005.jpg": ["table"]}
    mock_annotations = {} 
    
    success_count = 0
    
    # 使用 tqdm 进度条
    pbar = tqdm(imgs, desc="批量产出极品点云")
    for img_name in pbar:
        img_path = os.path.join(IMG_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        
        try:
            # 读取当前图片的真实标签 (如果没有就传 None，自动降级为盲搜)
            gt_labels = mock_annotations.get(img_name, None)
            
            # 执行核心推理
            instances = gen.process_one(img_path, target_labels=gt_labels)
            
            # 分离保存每个实例
            if instances and len(instances) > 0:
                for i, inst in enumerate(instances):
                    save_name = f"{base_name}_obj{i}_{inst['label']}.npy"
                    save_path = os.path.join(OUTPUT_DIR, save_name)
                    # 只有当文件不存在时才保存，方便断点续传
                    if not os.path.exists(save_path):
                        np.save(save_path, inst["points"])
                success_count += 1
                
        except Exception as e:
            # 记录错误图，但不中断程序
            print(f"\n❌ 处理 {img_name} 时发生错误: {str(e)}")
            traceback.print_exc()
            continue
            
        finally:
            # 🌟 极其关键：释放 GPU 显存残留，防止 OOM
            torch.cuda.empty_cache()
            
        pbar.set_postfix({"成功图数": success_count})

    print(f"\n🎉 搞定！高精度实例伪点云已存入 {OUTPUT_DIR}，随时准备接入 SOR 和 L-Shape！")