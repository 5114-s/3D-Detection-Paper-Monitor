import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from types import SimpleNamespace

# ==========================================
# 🌟 环境配置
# ==========================================
GROUNDED_SAM_DIR = "/data/ZhaoX/OVM3D-Det-1/Grounded-SAM-2"
sys.path.append(GROUNDED_SAM_DIR)

from groundingdino.util.inference import load_model as load_gdino, predict as gdino_predict
import groundingdino.datasets.transforms as T
from detany3d_frontend.image_encoder import ImageEncoderViT
from cubercnn.generate_label.util import project_image_to_cam

def create_uv_depth(depth, mask=None):
    """OVM3D 原版点云生成逻辑"""
    if mask is not None:
        depth = depth * mask
    x, y = np.meshgrid(np.linspace(0, depth.shape[1] - 1, depth.shape[1]), np.linspace(0, depth.shape[0] - 1, depth.shape[0]))
    uv_depth = np.stack((x, y, depth), axis=-1).reshape(-1, 3)
    return uv_depth[uv_depth[:, 2] != 0]

if __name__ == "__main__":
    DEVICE = "cuda"
    IMG_PATH = "/data/ZhaoX/OVM3D-Det-1/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg"
    GDINO_CONFIG = os.path.join(GROUNDED_SAM_DIR, "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    GDINO_CKPT = os.path.join(GROUNDED_SAM_DIR, "checkpoints/groundingdino_swint_ogc.pth")
    DETANY3D_CKPT = "/data/ZhaoX/OVM3D-Det-1/Grounded-SAM-2/checkpoints/zero_shot_category_ckpt-002.pth"
    K = np.array([[529.5, 0.0, 365.0], [0.0, 529.5, 262.0], [0.0, 0.0, 1.0]])

    print(">> 🚀 启动现场对比实验 (无需离线文件)...")
    
    # 1. 加载 DINO
    gdino = load_gdino(GDINO_CONFIG, GDINO_CKPT)
    gdino_t = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 2. 加载 DetAny3D 只用于获取深度图
    c = SimpleNamespace(model=SimpleNamespace(pad=896, additional_adapter=True, multi_level_box_output=1, original_sam=False))
    c.model.image_encoder = SimpleNamespace(patch_size=16, global_attn_indexes=[7, 15, 23, 31])
    depth_engine = ImageEncoderViT(img_size=896, patch_size=16, embed_dim=1280, depth=32, num_heads=16, cfg=c).to(DEVICE)
    depth_engine.load_state_dict(torch.load(DETANY3D_CKPT, map_location=DEVICE), strict=False)

    # 3. 执行 2D 检测
    raw_img = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)
    h, w = raw_img.shape[:2]
    image_pil = Image.fromarray(raw_img)
    it, _ = gdino_t(image_pil, None)
    results = gdino_predict(model=gdino, image=it, caption="bed. chair. picture. pillow.", box_threshold=0.3, text_threshold=0.25)
    boxes, phrases = results[0], results[2]
    print(f"✅ 成功锁定 {len(phrases)} 个目标")

    # 4. 获取深度图
    pad_img = cv2.copyMakeBorder(raw_img, 0, max(0, 896-h), 0, max(0, 896-w), cv2.BORDER_CONSTANT)[:896, :896]
    img_t = torch.from_numpy(pad_img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    depth_out = depth_engine({"images": img_t, "image_for_dino": img_t, "vit_pad_size": torch.tensor([[h//16, w//16]], device=DEVICE), "gt_intrinsic": torch.eye(4)[:3, :3].unsqueeze(0).to(DEVICE)})
    depth = depth_out["depth_maps"][0, 0, :h, :w].cpu().numpy()

    # 5. 模拟原版无腐蚀逻辑
    print("\n📊 --- 原版流程 (2D 矩形框直投) 结果 ---")
    for i, box in enumerate(boxes):
        box_abs = (box * torch.Tensor([w, h, w, h])).numpy()
        x_c, y_c, bw, bh = box_abs
        xmin, ymin, xmax, ymax = int(x_c - bw/2), int(y_c - bh/2), int(x_c + bw/2), int(y_c + bh/2)
        
        mask = np.zeros((h, w))
        mask[max(0, ymin):min(h, ymax), max(0, xmin):min(w, xmax)] = 1
        
        uv_depth = create_uv_depth(depth * mask)
        pts = project_image_to_cam(uv_depth, K)
        print(f"  ✅ 目标 {i} [{phrases[i]}]: {len(pts)} 个点")

    print("\n💡 提示：你可以拿这些数字和你刚才用 DetAny3D 跑出来的数字对比。")
    print("你会发现原版这种方法下，'picture' 的点数会多得不正常，因为它把框里的墙壁也算进去了。")