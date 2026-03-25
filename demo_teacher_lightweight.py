"""
教师端3D检测 - 轻量版（按你的流程设计）
不使用DetAny3D，只用：Grounding DINO + MoGe+DepthPro + SAM2 + L-Shape

流程：
1. RAM+GPT 或 手动text_prompt -> Grounding DINO 2D检测
2. MoGe+DepthPro 融合深度估计
3. SAM2 Box-Prompted 掩码生成
4. L-Shape/PCA 3D框拟合
"""
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = '/data/ZhaoX/OVM3D-Det-1'

# 导入地面估计工具
from cubercnn.generate_label.util import extract_ground, project_image_to_cam
from teacher_student.teacher_geometry import create_uv_depth


def estimate_ground(depth_np, K, depth_scale=1.0, min_points=50):
    """
    从深度图估计地面方程 [A, B, C, D]，使得 Ax+By+Cz+D=0。
    使用图像下半部分深度点做 RANSAC 平面拟合。
    返回 None 如果估计失败。
    """
    h, w = depth_np.shape[:2]
    # 使用图像下半部分（地面区域假设在画面下半部）
    mask = np.zeros((h, w), dtype=np.float32)
    mask[h//2:, :] = 1.0

    depth_for_ground = np.asarray(depth_np, dtype=np.float64)
    if abs(depth_scale - 1.0) > 1e-6:
        depth_for_ground = depth_for_ground * depth_scale
    depth_for_ground = np.clip(depth_for_ground, 0.3, 8.0)

    uv_depth = create_uv_depth(depth_for_ground, mask)
    if uv_depth.shape[0] < min_points:
        return None

    pc = project_image_to_cam(uv_depth, np.array(K))
    ground_equ = extract_ground(pc)
    return ground_equ

def draw_3d_box(img, K, center_cam, dims, R_cam, color=(0, 255, 0), thickness=2):
    """绘制3D边界框

    dims 来自 teacher_geometry: [l, w, h]
    get_cuboid_verts_faces 约定 box3d=[cx, cy, cz, W, H, L]:
      W = dims[1] (z方向宽度)
      H = dims[2] (y方向高度)
      L = dims[0] (x方向长度)
    """
    L, W, H = float(dims[0]), float(dims[1]), float(dims[2])
    cx, cy, cz = float(center_cam[0]), float(center_cam[1]), float(center_cam[2])

    box3d = np.array([[cx, cy, cz, W, H, L]], dtype=np.float32)
    box3d_t = torch.from_numpy(box3d).float()
    R_t = torch.from_numpy(R_cam).float()
    
    from cubercnn.util import math_util as util_math
    verts_t, _ = util_math.get_cuboid_verts_faces(box3d_t, R_t)
    verts = verts_t.squeeze(0).numpy()
    
    proj = (K @ verts.T).T
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]
    
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        pt1 = (int(proj[i,0]), int(proj[i,1]))
        pt2 = (int(proj[j,0]), int(proj[j,1]))
        if 0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0]:
            if 0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]:
                cv2.line(img, pt1, pt2, color, thickness)
    return img


class LightweightTeacher3D:
    """轻量级教师端 - 按你的流程设计"""
    
    def __init__(self, device='cuda', use_sam2=True):
        self.device = device
        self.use_sam2 = use_sam2
        self._load_models()
    
    def _load_models(self):
        # ====== 1. Grounding DINO ======
        print(">> 加载 Grounding DINO...")
        GROUNDED_SAM_DIR = os.path.join(PROJECT_ROOT, "Grounded-SAM-2")
        
        # 重要：彻底清理所有 groundingdino 相关模块
        for mod in list(sys.modules.keys()):
            if 'grounding' in mod.lower():
                del sys.modules[mod]
        
        # 确保正确的路径在最前面
        for p in [os.path.join(GROUNDED_SAM_DIR, "grounding_dino"), GROUNDED_SAM_DIR]:
            if p in sys.path:
                sys.path.remove(p)
            sys.path.insert(0, p)
        
        from groundingdino.util.inference import load_model as load_gdino, predict as gdino_predict
        import groundingdino.datasets.transforms as T
        
        self.gdino_model = load_gdino(
            os.path.join(GROUNDED_SAM_DIR, "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
            os.path.join(GROUNDED_SAM_DIR, "checkpoints/groundingdino_swint_ogc.pth"),
        ).to(self.device)
        self.gdino_transform = T.Compose([
            T.RandomResize([800], max_size=1333), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.gdino_predict = gdino_predict
        print(f"  DEBUG: predict函数来源: {gdino_predict.__code__.co_filename}")
        print(">> Grounding DINO 加载完成")
        
        # ====== 2. MoGe + Depth Pro ======
        print(">> 加载 MoGe + Depth Pro...")
        
        # 添加MoGe路径 - 必须从MoGe根目录作为包导入
        moge_root = "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/MoGe"
        sys.path.insert(0, moge_root)  # 让 moge 成为可导入的包
        sys.path.insert(0, os.path.join(moge_root, "moge"))  # 让 moge.model 可导入
        
        # 添加ml-depth-pro路径
        depthpro_root = "/data/ZhaoX/LabelAny3D-main/LabelAny3D-main/external/ml-depth-pro/src"
        sys.path.insert(0, depthpro_root)
        
        from detany3d_frontend.depth_predictor.moge_depthpro_fusion import MoGeLoader, DepthProLoader, align_depth_ransac
        
        self.moge_loader = MoGeLoader()
        self.depthpro_loader = DepthProLoader()
        self.align_depth_ransac = align_depth_ransac
        
        self.moge_loader.load_model()
        self.depthpro_loader.load_model()
        print(">> MoGe + Depth Pro 加载完成")
        
        # ====== 3. SAM2 ======
        self.sam2_predictor = None
        if self.use_sam2:
            print(">> 加载 SAM2...")
            try:
                import sam2
                sam2_root = os.path.dirname(sam2.__file__)

                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                sam2_cfg = os.path.join(GROUNDED_SAM_DIR, "sam2", "configs", "sam2", "sam2_hiera_s.yaml")
                sam2_ckpt = os.path.join(GROUNDED_SAM_DIR, "checkpoints", "sam2_hiera_small.pt")
                if not os.path.isfile(sam2_ckpt):
                    sam2_ckpt = os.path.join(PROJECT_ROOT, "weights", "sam2_hiera_small.pt")
                if not os.path.isfile(sam2_cfg) or not os.path.isfile(sam2_ckpt):
                    raise FileNotFoundError(f"SAM2 配置或权重未找到: cfg={sam2_cfg}, ckpt={sam2_ckpt}")

                # build_sam2 需要在 Grounded-SAM-2 目录下才能正确解析 config
                cwd = os.getcwd()
                os.chdir(GROUNDED_SAM_DIR)
                try:
                    sam2_model = build_sam2("sam2_hiera_s.yaml", sam2_ckpt, device=self.device)
                finally:
                    os.chdir(cwd)
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                print(">> SAM2 加载完成")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f">> SAM2 加载失败: {e}")
                self.use_sam2 = False
        
        # ====== 4. 几何工具 ======
        print(">> 加载几何工具...")
        from cubercnn.generate_label.priors import llm_generated_prior
        self.prior_dict = llm_generated_prior.get("SUNRGBD", {})
        from teacher_student.teacher_geometry import run_teacher_pipeline_per_instance
        self.run_geometry = run_teacher_pipeline_per_instance
        print(">> 几何工具加载完成")
        
        print(">> 教师端加载完成!")
    
    @torch.no_grad()
    def detect(self, rgb_np, text_prompt, K, box_threshold=0.3, text_threshold=0.25):
        """完整检测流程"""
        h, w = rgb_np.shape[:2]
        
        # ====== Step 1: Grounding DINO 2D检测 ======
        print("\n[Step 1] Grounding DINO 2D检测...")
        image_pil = Image.fromarray(rgb_np)
        image_tensor, _ = self.gdino_transform(image_pil, None)
        image_tensor = image_tensor.to(self.device)
        
        boxes, scores, phrases = self.gdino_predict(
            self.gdino_model, image_tensor, text_prompt,
            box_threshold=box_threshold, text_threshold=text_threshold,
        )
        
        if len(boxes) == 0:
            print("未检测到物体")
            return [], None
        
        # 转换到原图坐标
        boxes_abs = boxes * torch.Tensor([w, h, w, h]).to(boxes.device)
        xyxy = boxes_abs.clone()
        xyxy[:, 0] = boxes_abs[:, 0] - boxes_abs[:, 2] / 2
        xyxy[:, 1] = boxes_abs[:, 1] - boxes_abs[:, 3] / 2
        xyxy[:, 2] = boxes_abs[:, 0] + boxes_abs[:, 2] / 2
        xyxy[:, 3] = boxes_abs[:, 1] + boxes_abs[:, 3] / 2
        
        print(f"  检测到 {len(phrases)} 个物体: {phrases}")
        xyxy_np = xyxy.cpu().numpy()
        
        # ====== Step 2: MoGe+DepthPro 深度融合 ======
        print("\n[Step 2] MoGe+DepthPro 深度融合...")
        
        # MoGe 推理
        moge_result = self.moge_loader.infer(rgb_np)
        depth_moge = moge_result['depth']
        moge_K = moge_result['intrinsics']
        print(f"  MoGe 深度范围: [{depth_moge.min():.3f}, {depth_moge.max():.3f}]")
        
        # Depth Pro 推理 - 使用 MoGe 推断的相机内参（与 LabelAny3D 一致）
        focal_length_px = moge_K[0, 0]  # 使用 MoGe 从图像推断的焦距
        depthpro_result = self.depthpro_loader.infer(rgb_np, focal_length_px=focal_length_px)
        depth_depthpro = depthpro_result['depth']
        print(f"  Depth Pro 深度范围: [{depth_depthpro.min():.3f}, {depth_depthpro.max():.3f}] m")
        
        # 深度对齐 - 使用 MoGe 的 mask 约束 RANSAC 对齐（与 LabelAny3D 一致）
        # MoGe mask 标识了有效深度区域，避免远处墙面等干扰
        moge_mask = moge_result.get('mask', None)
        if moge_mask is not None and moge_mask.shape != (h, w):
            moge_mask = cv2.resize(moge_mask.astype(np.float32), (w, h)) > 0.5
        aligned_depth, diag = self.align_depth_ransac(depth_moge, depth_depthpro, mask=moge_mask)
        print(f"  [对齐] {diag['status']} | scale={diag['scale']:.4f} | 内点率={diag['inlier_ratio']:.1%} | P95误差={diag['p95_error']:.3f}m")
        print(f"  对齐后深度范围: [{aligned_depth.min():.3f}, {aligned_depth.max():.3f}] m")
        
        # 调整到原图大小
        if aligned_depth.shape != (h, w):
            aligned_depth = cv2.resize(aligned_depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # 估计地面 - 使用 MoGe 推断的 K
        print("\n  估计地面...")
        ground_equ = estimate_ground(aligned_depth, moge_K, depth_scale=1.0)
        if ground_equ is not None:
            print(f"  地面方程: [{ground_equ[0]:.3f}, {ground_equ[1]:.3f}, {ground_equ[2]:.3f}, {ground_equ[3]:.3f}]")
            print(f"  地面法向量: [{ground_equ[0]:.3f}, {ground_equ[1]:.3f}, {ground_equ[2]:.3f}]")
        else:
            print("  地面估计失败，使用默认参数")

        # ====== Step 3: SAM2 掩码生成 ======
        print("\n[Step 3] SAM2 掩码生成...")
        if self.use_sam2 and self.sam2_predictor is not None:
            self.sam2_predictor.set_image(rgb_np)
            masks_list = []
            for i in range(len(xyxy_np)):
                box = xyxy_np[i:i+1].astype(np.float64)
                mask, _, _ = self.sam2_predictor.predict(
                    point_coords=None, point_labels=None, box=box, multimask_output=False
                )
                masks_list.append(mask[0].astype(np.float32))
            masks = np.stack(masks_list)
            print(f"  生成 {len(masks)} 个掩码")
        else:
            # SAM2不可用时，使用2D框生成粗掩码
            masks = []
            for i in range(len(xyxy_np)):
                box = xyxy_np[i]
                mask = np.zeros((h, w), dtype=np.float32)
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask[y1:y2, x1:x2] = 1.0
                masks.append(mask)
            masks = np.stack(masks)
            print("  SAM2不可用，使用2D框作为掩码")
        
        # ====== Step 4: L-Shape 3D拟合 ======
        print("\n[Step 4] L-Shape 3D拟合...")
        pseudo_list = []
        for i in range(len(phrases)):
            mask_i = masks[i] if masks is not None else None
            center_cam, dimensions, R_cam, ok = self.run_geometry(
                aligned_depth,
                mask_i,
                moge_K,  # 使用 MoGe 推断的 K
                phrases[i],
                self.prior_dict,
                ground_equ=ground_equ,
                use_lshape=True,
                depth_scale=1.0,
                debug=True,
            )
            if ok:
                pseudo_list.append({
                    'center_cam': center_cam,
                    'dimensions': dimensions,
                    'R_cam': R_cam,
                    'phrase': phrases[i],
                    'box_2d_xyxy': xyxy_np[i],
                    'K': moge_K,  # 保存 K 用于可视化
                })
                print(f"  [{phrases[i]}] center=({center_cam[0]:.2f}, {center_cam[1]:.2f}, {center_cam[2]:.2f}), dims=({dimensions[0]:.2f}, {dimensions[1]:.2f}, {dimensions[2]:.2f})")
        
        return pseudo_list, aligned_depth


def main():
    print("="*60)
    print("轻量级教师端3D检测（MoGe+DepthPro流程）")
    print("="*60)
    
    # 设置使用 GPU 1
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # 1. 加载图像
    image_path = '/data/ZhaoX/OVM3D-Det-1/datasets/sunrgbd/sunrgbd_trainval/image/000004.jpg'
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"\n图像: {img_rgb.shape}")
    
    # 2. 相机内参 (SUNRGBD标准参数)
    # 原始SUNRGBD使用 640x480, fx=577.87
    # 缩放到 730x530: fx=577.87*730/640≈659
    K = np.array([
        [659.1, 0, 365.0],
        [0, 638.1, 265.0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 3. Text prompt
    text_prompt = "bed. table. chair."
    
    # 4. 加载模型并检测
    teacher = LightweightTeacher3D(device='cuda:1', use_sam2=True)
    
    pseudo_list, depth = teacher.detect(
        img_rgb,
        text_prompt=text_prompt,
        K=K,
    )
    
    # 5. 可视化
    print("\n" + "="*60)
    print("可视化结果")
    print("="*60)
    
    vis_img = img.copy()
    
    for pseudo in pseudo_list:
        center = pseudo['center_cam']
        dims = pseudo['dimensions']
        R = pseudo['R_cam']
        K_vis = pseudo.get('K', K)
        
        draw_3d_box(vis_img, K_vis, center, dims, R, color=(0, 255, 0), thickness=3)
        
        box2d = pseudo['box_2d_xyxy']
        cv2.rectangle(vis_img, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), (255, 0, 0), 2)
        
        label = f"{pseudo['phrase']}: z={center[2]:.1f}m"
        cv2.putText(vis_img, label, (int(box2d[0]), int(box2d[1])-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    output_path = '/data/ZhaoX/OVM3D-Det-1/demo_3d_lightweight.jpg'
    cv2.imwrite(output_path, vis_img)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
