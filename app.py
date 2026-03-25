import gradio as gr
import cv2
import numpy as np
import torch
import clip

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.layers import nms
from cubercnn.config import get_cfg_defaults
from cubercnn.vis.vis import draw_3d_box_from_verts, draw_text

# 注册模块
import cubercnn.modeling.meta_arch.rcnn3d_text
import cubercnn.modeling.roi_heads
import cubercnn.modeling.backbone.dla


class OVM3D_Predictor:
    def __init__(self, config_path, weights_path):
        print("🚀 加载模型中...")

        self.cfg = get_cfg()
        get_cfg_defaults(self.cfg)

        self.cfg.set_new_allowed(True)
        self.cfg.merge_from_file(config_path)
        self.cfg.set_new_allowed(False)

        self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.predictor = DefaultPredictor(self.cfg)
        self.device = self.predictor.model.device

        print("🚀 加载 CLIP...")
        self.clip_model, _ = clip.load("ViT-L/14", device=self.device)

        # 🎯 核心修复 1：不要自己猜，直接从模型底层的元数据中提取它严格对齐的类别和顺序！
        if hasattr(self.predictor.metadata, "thing_classes"):
            self.model_classes = self.predictor.metadata.thing_classes
        else:
            # 万一没读到，使用备用的典型字母顺序
            self.model_classes =['bathtub', 'bed', 'bin', 'blinds', 'books', 'bookshelf', 'box', 'cabinet', 'ceiling', 'chair', 'clothes', 'computer', 'counter', 'curtain', 'desk', 'door', 'dresser', 'floor_mat', 'fridge', 'lamp', 'mirror', 'monitor', 'night_stand', 'paper', 'person', 'picture', 'pillow', 'plant', 'shelves', 'shower_curtain', 'sink', 'sofa', 'table', 'toilet', 'towel', 'trash_can', 'tv', 'whiteboard', 'window'][:38]

        print(f"✅ 初始化完成！模型底层包含 {len(self.model_classes)} 个基准类别。")

        # 大物体类别：检测框面积过小时视为误检（如 bed 不应比柜子/凳子还小）
        self.large_object_classes = {
            "bed", "sofa", "table", "desk", "bathtub", "cabinet", "bookshelf",
            "fridge", "toilet", "counter", "door", "window"
        }
        # 大物体至少占图像面积的比例，过小则过滤（减少柜子/凳子被标成 bed）
        self.min_area_ratio_for_large = 0.02  # 2%

        # 提前计算文本特征（保持严格顺序，防止 3D 框参数错乱）
        self.cached_text_features = self.encode_all_classes()

    @staticmethod
    def estimate_camera_matrix(width, height, hfov_deg=55.0):
        # 没有真实内参时，只能给一个近似相机模型用于 3D 投影显示。
        focal = 0.5 * width / np.tan(np.deg2rad(hfov_deg) / 2.0)
        cx = width / 2.0
        cy = height / 2.0
        return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float32)

    def encode_all_classes(self):
        # 严格遵守模型预设的 38 个类别顺序，最后补一个 background
        text_inputs =[f"a {c}" for c in self.model_classes] + ["background"]
        tokens = clip.tokenize(text_inputs).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens).float()
        return text_features

    def predict(self, image_bgr, K, prompt_text, conf_thresh=0.7, topk=10, min_area_ratio=None):
        # 🎯 核心修复 2：正确使用 Detectron2 的图像缩放逻辑
        original_image = image_bgr.copy()
        height_orig, width_orig = original_image.shape[:2]
        used_estimated_k = K is None
        
        # 让模型按训练时的逻辑自动缩放图片
        transform = self.predictor.aug.get_transform(original_image)
        image_resized = transform.apply_image(original_image)
        height_new, width_new = image_resized.shape[:2]

        # 🎯 核心修复 3：根据缩放比例，同步修正虚拟相机矩阵 K，解决框的畸变
        if K is None:
            K = self.estimate_camera_matrix(width_orig, height_orig)
        else:
            K = np.array(K, dtype=np.float32)

        K_resized = K.copy()
        K_resized[0, :] *= (width_new / width_orig)
        K_resized[1, :] *= (height_new / height_orig)

        inputs = {
            "image": torch.as_tensor(image_resized.astype("float32").transpose(2, 0, 1)),
            "height": height_orig, # 告诉模型原图多大，它会自动把框还原
            "width": width_orig,
            "K": K_resized.tolist()
        }

        with torch.no_grad():
            outputs = self.predictor.model([inputs], self.cached_text_features)[0]

        instances = outputs["instances"].to("cpu")

        if len(instances) == 0:
            return image_bgr, "❌ 没有检测到目标"

        # NMS 去重（适当放宽 IoU 阈值，避免把不重叠的真床和误检压成一块）
        boxes = instances.pred_boxes.tensor
        scores = instances.scores
        keep = nms(boxes, scores, 0.5)
        instances = instances[keep]

        # 分数过滤
        keep = instances.scores > conf_thresh
        instances = instances[keep]

        # 🎯 核心修复 4：只从检测结果中，摘出用户真正想搜的物体
        user_prompts =[p.strip().lower() for p in prompt_text.split(",") if p.strip()]
        target_ids =[]
        for i, class_name in enumerate(self.model_classes):
            c_lower = class_name.lower()
            for up in user_prompts:
                # 模糊匹配，比如 "bed" 能匹配到底层的 "bed"
                if up in c_lower or c_lower in up:
                    target_ids.append(i)
                    break
                    
        if len(target_ids) == 0:
            avail_classes = ", ".join(self.model_classes[:10]) + "..."
            return image_bgr, f"⚠️ 模型底层库未匹配到 '{prompt_text}'。试试这些: {avail_classes}"

        # 过滤掉不需要的类别
        pred_classes = instances.pred_classes.numpy()
        keep_class = [idx for idx, cls_id in enumerate(pred_classes) if cls_id in target_ids]
        instances = instances[keep_class]

        if len(instances) == 0:
            return image_bgr, "⚠️ 在画面中没有检测到该物体，或者置信度偏低（可尝试调低阈值）。"

        # 🎯 核心修复 5：大物体（如 bed）面积过小则视为误检，过滤掉
        min_ratio = self.min_area_ratio_for_large if min_area_ratio is None else float(min_area_ratio)
        img_area = float(height_orig * width_orig)
        pred_boxes_np = instances.pred_boxes.tensor.numpy()
        keep_area = []
        for idx in range(len(instances)):
            cls_id = int(instances.pred_classes[idx])
            class_name = self.model_classes[cls_id] if cls_id < len(self.model_classes) else ""
            x1, y1, x2, y2 = pred_boxes_np[idx]
            box_area = max(0, (x2 - x1) * (y2 - y1))
            ratio = box_area / img_area
            if class_name.lower() in self.large_object_classes:
                if ratio >= min_ratio:
                    keep_area.append(idx)
            else:
                keep_area.append(idx)
        instances = instances[keep_area]

        if len(instances) == 0:
            return image_bgr, "⚠️ 过滤后无目标（大物体类别需占画面一定比例，可调低置信度或检查类别）。"

        # Top-K
        scores = instances.scores.numpy()
        idx = np.argsort(-scores)[:topk]
        instances = instances[idx]

        # 使用模型原生输出的 3D 顶点进行绘制；无真实 K 时可缩放 3D 框以贴合 2D 框
        result = self.draw(
            original_image, instances, K, self.model_classes,
            scale_3d_to_2d=used_estimated_k
        )

        msg = f"✅ 成功检测到 {len(instances)} 个目标"
        if used_estimated_k:
            msg += "；当前未提供真实相机内参，3D 框为近似投影"
        return result, msg

    def _scale_verts_to_match_2d(self, verts3d, K, box_2d, min_depth=0.1):
        """无真实内参时，将 3D 框缩放使投影与 2D 框大致一致，缓解框过小/漂移。"""
        if verts3d.shape != (8, 3) or np.any(verts3d[:, 2] < min_depth):
            return verts3d
        proj = (K @ verts3d.T).T
        proj = proj[:, :2] / np.clip(proj[:, 2:3], min_depth, None)
        x1_p, y1_p = proj[:, 0].min(), proj[:, 1].min()
        x2_p, y2_p = proj[:, 0].max(), proj[:, 1].max()
        w_p, h_p = max(1e-4, x2_p - x1_p), max(1e-4, y2_p - y1_p)
        x1_b, y1_b, x2_b, y2_b = box_2d
        w_b, h_b = max(1e-4, x2_b - x1_b), max(1e-4, y2_b - y1_b)
        center_3d = verts3d.mean(axis=0)
        scale_w, scale_h = w_b / w_p, h_b / h_p
        scale = np.sqrt(scale_w * scale_h)  # 统一缩放
        verts3d = center_3d + (verts3d - center_3d) * scale
        return verts3d.astype(np.float32)

    def draw(self, img, instances, K, model_classes, scale_3d_to_2d=True):
        verts3d_all = instances.pred_bbox3D.numpy() if instances.has("pred_bbox3D") else None
        scores = instances.scores.numpy()
        pred_classes = instances.pred_classes.numpy()
        pred_boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None

        for i in range(len(instances)):
            class_id = int(pred_classes[i])
            class_name = model_classes[class_id] if 0 <= class_id < len(model_classes) else "Unknown"
            label = f"{class_name}: {scores[i]:.2f}"

            if verts3d_all is not None:
                verts3d = np.asarray(verts3d_all[i], dtype=np.float32)
                if verts3d.shape == (8, 3) and np.any(verts3d[:, 2] > 0.1):
                    if scale_3d_to_2d and pred_boxes is not None:
                        verts3d = self._scale_verts_to_match_2d(
                            verts3d, K, pred_boxes[i], min_depth=0.1
                        )
                    draw_3d_box_from_verts(img, K, verts3d, color=(0, 255, 0), thickness=2)

            if pred_boxes is not None:
                x1, y1, x2, y2 = pred_boxes[i]
                x1 = int(np.clip(x1, 0, img.shape[1] - 1))
                y1 = int(np.clip(y1, 18, img.shape[0] - 1))
                draw_text(img, label, [x1, y1], scale=0.7, bg_color=(0, 255, 255))

        return img


# ======================
CONFIG = "/data/ZhaoX/OVM3D-Det/configs/Base_Omni3D_SUN.yaml"
WEIGHTS = "/data/ZhaoX/OVM3D-Det/output/training/SUN/model_final.pth"

model = OVM3D_Predictor(CONFIG, WEIGHTS)


def process(image, prompt, conf, min_area):
    if image is None:
        return None, "⚠️ 请上传图片"

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    result, msg = model.predict(
        img_bgr,
        K=None,
        prompt_text=prompt,
        conf_thresh=conf,
        topk=10,
        min_area_ratio=min_area,
    )

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result, msg


with gr.Blocks() as demo:
    gr.Markdown("# 🚀 OVM3D 高质量版本 Demo")

    with gr.Row():
        with gr.Column():
            img = gr.Image()
            prompt = gr.Textbox(value="bed", label="检测类别（逗号分隔）")
            conf = gr.Slider(0.3, 0.9, value=0.7, step=0.05, label="置信度阈值")
            min_area = gr.Slider(0.005, 0.1, value=0.02, step=0.005,
                label="大物体最小面积比例（过滤小框误检，如柜子被标成床）")
            btn = gr.Button("开始")

        with gr.Column():
            out = gr.Image()
            log = gr.Textbox()

    btn.click(process, inputs=[img, prompt, conf, min_area], outputs=[out, log])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
