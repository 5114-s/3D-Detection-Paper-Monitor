# Copyright (c) Teacher-Student Distillation Pipeline
"""
改进2: MoGe高频细节 + Depth Pro物理尺度融合
替换单一UniDepth模型,提升几何拟合上限

参考 LabelAny3D 的实现: https://github.com/xxx/LabelAny3D
- MoGe: 提供高频细节(边缘、纹理),尺度不变深度
- Depth Pro: 提供物理尺度(绝对深度值),度量深度
- RANSAC对齐: 将MoGe尺度对齐到Depth Pro的物理尺度

融合策略:
1. ransac_align: RANSAC线性回归对齐(推荐,LabelAny3D原生方法)
2. weighted_adaptive: 自适应加权融合
3. frequency_split: 频域分离融合
4. weighted: 简单加权融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Union
from PIL import Image
import sys
import os


class MoGeLoader:
    """
    MoGe 模型加载器
    
    参考 LabelAny3D 的加载方式:
    - 模型来源: HuggingFace Hub "Ruicheng/moge-vitl"
    - 输出: points, depth, mask, intrinsics
    """
    
    _instance = None
    _model = None
    
    def __init__(self, device=None):
        self.model = None
        # 尊重传入的 device 参数，如果为 None 则自动检测
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
    
    @classmethod
    def get_instance(cls, device=None):
        if cls._instance is None or (device is not None and str(cls._instance.device) != str(device)):
            cls._instance = cls(device=device)
        return cls._instance
    
    def load_model(self, model_path: str = None):
        """加载 MoGe 模型"""
        if self.model is not None:
            return self.model
        
        try:
            # 获取项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # 添加MoGe相关路径 - 必须先切换工作目录
            moge_base_path = os.path.join(project_root, "external", "MoGe")
            
            if os.path.exists(moge_base_path):
                # 切换到MoGe目录（与test_depthpro_fixed2.py一致的处理方式）
                original_cwd = os.getcwd()
                os.chdir(moge_base_path)
                sys.path.insert(0, moge_base_path)  # 让moge成为顶层包
                sys.path.insert(0, os.path.join(moge_base_path, "moge"))  # 让moge.model可导入
            
            from moge.model.moge_model import MoGeModel
            
            self.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl")
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f">> MoGe模型加载成功 (device={self.device})")
            return self.model
            
        except Exception as e:
            print(f"警告: MoGe模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            print(">> 请确保已安装: pip install huggingface_hub")
            return None
    
    def infer(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, np.ndarray]:
        """
        MoGe 推理
        
        Args:
            image: RGB图像 (H, W, 3) numpy数组或 PIL Image
            
        Returns:
            dict包含:
            - points: (H, W, 3) 点云
            - depth: (H, W) 尺度不变深度
            - mask: (H, W) 有效区域掩码
            - intrinsics: (3, 3) 相机内参
        """
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            return None
        
        # 预处理图像
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        H, W = image.shape[:2]
        
        # MoGe 模型期望: B, C, H, W 格式的归一化图像
        image_tensor = torch.tensor(
            image / 255.0,  # 归一化到 [0, 1]
            dtype=torch.float32,
            device=self.device
        ).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # 推理
        with torch.no_grad():
            output = self.model.infer(image_tensor)
        
        # 提取结果
        depth = output['depth']  # (1, 1, H, W) 或 (1, H, W)
        mask = output['mask'] if 'mask' in output else None
        intrinsics = output['intrinsics']  # (1, 3, 3)
        points = output['points'] if 'points' in output else None
        
        # 处理深度维度
        if depth.dim() == 4:
            depth = depth.squeeze(0)  # (1, H, W)
        if depth.dim() == 3:
            depth = depth.squeeze(0)  # (H, W)
        
        result = {
            'depth': depth.squeeze().cpu().numpy(),  # (H, W)
            'intrinsics': intrinsics.squeeze().cpu().numpy(),  # (3, 3)
        }
        
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.squeeze(0)
            if mask.dim() == 3:
                mask = mask.squeeze(0)
            result['mask'] = mask.squeeze().cpu().numpy()
        
        if points is not None:
            if points.dim() == 4:
                points = points.squeeze(0)
            result['points'] = points.squeeze().cpu().numpy()
        
        # 与 LabelAny3D 完全一致：内参从归一化坐标转换到像素坐标（无条件）
        # LabelAny3D: intrinsics * np.array([[W, 1, W], [1, H, H], [1, 1, 1]])
        W, H = image.shape[1], image.shape[0]
        result['intrinsics'] = result['intrinsics'] * np.array([
            [W, 1, W],
            [1, H, H],
            [1, 1, 1]
        ])
        
        return result


class DepthProLoader:
    """
    Depth Pro 模型加载器
    
    参考 LabelAny3D 的加载方式:
    - 使用 depth_pro 库
    - 输出: 深度(米), 焦距
    """
    
    _instance = None
    _model = None
    _transform = None
    
    def __init__(self, device=None):
        self.model = None
        self.transform = None
        # 尊重传入的 device 参数，如果为 None 则自动检测
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
    
    @classmethod
    def get_instance(cls, device=None):
        if cls._instance is None or (device is not None and str(cls._instance.device) != str(device)):
            cls._instance = cls(device=device)
        return cls._instance
    
    def load_model(self, checkpoint_path: str = None, precision: torch.dtype = None):
        """加载 Depth Pro 模型，与 test_depthpro_fixed2.py 一致"""
        if self.model is not None:
            return self.model, self.transform
        
        # CPU 不支持 Half 精度的 interpolate，自动降级
        if precision is None:
            precision = torch.float16 if self.device.type == "cuda" else torch.float32
        
        try:
            # 添加ml-depth-pro路径并切换工作目录（与test_depthpro_fixed2.py一致）
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ml_depth_pro_path = os.path.join(project_root, "external", "ml-depth-pro", "src")
            
            if os.path.exists(ml_depth_pro_path):
                sys.path.insert(0, ml_depth_pro_path)
                # 切换到ml-depth-pro目录，确保模块导入正确
                os.chdir(ml_depth_pro_path)
            
            import depth_pro
            from depth_pro.depth_pro import DepthProConfig
            
            # 检查权重路径
            default_ckpt = os.path.join(project_root, "external", "checkpoints", "depth_pro.pt")
            if checkpoint_path is None:
                checkpoint_path = default_ckpt
            
            # 创建配置（与test_depthpro_fixed2.py一致）
            config = DepthProConfig(
                patch_encoder_preset="dinov2l16_384",
                image_encoder_preset="dinov2l16_384",
                decoder_features=256,
                checkpoint_uri=None,  # 先不加载，让模型创建完成
                use_fov_head=True,
                fov_encoder_preset="dinov2l16_384",  # 必须设置！
            )

            # 创建模型
            model, transform = depth_pro.create_model_and_transforms(
                config=config,
                device=self.device,
                precision=precision,
            )
            
            # 手动加载权重（与test_depthpro_fixed2.py一致）
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"  缺少的权重键: {missing[:5]}...")
                if unexpected:
                    print(f"  多余的权重键: {unexpected[:5]}...")
                if not missing and not unexpected:
                    print(f">> DepthPro所有权重加载成功")
            else:
                print(f">> 警告: 权重文件不存在: {checkpoint_path}")
            
            self.model = model
            self.transform = transform
            print(f">> DepthPro模型加载成功 (device={self.device}, precision={precision}, use_fov_head=True)")
            return model, transform
            
        except ImportError as e:
            print(f"警告: depth_pro库导入失败: {e}")
            print(">> 请安装: cd external/ml-depth-pro && pip install -e .")
            return None, None
        except Exception as e:
            print(f"警告: DepthPro模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def infer(self, image: Union[np.ndarray, Image.Image], focal_length_px: float = None) -> Dict[str, np.ndarray]:
        """
        Depth Pro 推理
        
        Args:
            image: RGB图像 (H, W, 3) numpy数组或 PIL Image
            focal_length_px: 焦距(像素),可选
            
        Returns:
            dict包含:
            - depth: (H, W) 度量深度(米)
            - focallength_px: 焦距(像素)
        """
        if self.model is None or self.transform is None:
            self.load_model()
        
        if self.model is None:
            return None
        
        # 预处理图像
        if isinstance(image, Image.Image):
            image_pil = image
        else:
            image_pil = Image.fromarray(image)
        
        # 获取图像尺寸
        W, H = image_pil.size

        # 转换图像为 tensor
        image_tensor = self.transform(image_pil)

        # 如果没有提供焦距，让模型自己估计（启用 fov 头后模型会自己预测）
        # 否则传入焦距 tensor（官方源码要求 torch.Tensor 类型）
        f_px_kwarg = None
        if focal_length_px is not None:
            f_px_kwarg = torch.tensor(focal_length_px, dtype=image_tensor.dtype, device=self.device)

        # 推理
        with torch.no_grad():
            prediction = self.model.infer(image_tensor, f_px=f_px_kwarg)

        # 转换为 numpy
        # 官方源码中 f_px 可能是 0.0（模型估计）或传入的值
        result_f_px = prediction.get('focallength_px', focal_length_px)
        if torch.is_tensor(result_f_px):
            result_f_px = result_f_px.item()

        result = {
            'depth': prediction['depth'].cpu().numpy(),
            'focallength_px': result_f_px,
        }

        return result


def align_depth_ransac(
    relative_depth: np.ndarray,
    metric_depth: np.ndarray,
    mask: np.ndarray = None,
    min_samples: float = 0.2,
    max_valid_depth: float = 400.0,
    residual_threshold: float = 0.5,
    verbose: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    使用 RANSAC 线性回归将 MoGe 的尺度无关深度对齐到 Depth Pro 的度量深度。

    参考 LabelAny3D 实现 (batch_scripts/depth.py)，完全对齐其行为。

    物理意义:
    - MoGe 输出: 尺度不变的相对深度（经 recover_focal_shift 处理）
    - Depth Pro 输出: 度量深度(米)
    - 线性变换: metric = scale * relative（无截距）

    诊断返回值（用于判定对齐是否成功）:
    - scale: 拟合的 scale 因子
    - inlier_ratio: RANSAC 内点比例（>0.5 为佳）
    - median_error: 对齐后误差中位数(米)
    - p95_error: 误差 95 分位数(米)
    - method: "ransac" | "median" | "mean"
    - status: "success" | "warning" | "failed"
    """
    from sklearn.linear_model import RANSACRegressor, LinearRegression

    diagnostics = {
        "scale": None,
        "inlier_ratio": None,
        "median_error": None,
        "p95_error": None,
        "num_valid_points": None,
        "method": None,
        "status": None,
    }

    rel_flat = relative_depth.flatten().astype(np.float64)
    met_flat = metric_depth.flatten().astype(np.float64)

    # ---- 与 LabelAny3D 对齐：有效点筛选 ----
    # LabelAny3D: valid = (~inf(relative)) & (metric < 400)
    # 不要求 metric_depth > 0（DepthPro 可能输出零/负深度，但只要 < 400 就参与）
    # 但保留 rel > 0 筛选，因为 MoGe 深度为 0/负没有物理意义
    valid = (
        np.isfinite(rel_flat)
        & (rel_flat > 0)
        & np.isfinite(met_flat)
        & (met_flat < max_valid_depth)
    )
    if mask is not None:
        valid &= mask.flatten().astype(bool)

    rel_valid = rel_flat[valid].reshape(-1, 1)
    met_valid = met_flat[valid].reshape(-1, 1)
    diagnostics["num_valid_points"] = int(valid.sum())

    if len(rel_valid) < 100:
        if verbose:
            print(f"警告: 有效点 {len(rel_valid)} < 100，使用简单均值对齐")
        diagnostics["method"] = "mean"
        diagnostics["scale"] = met_valid.mean() / (rel_valid.mean() + 1e-6)
        aligned = relative_depth * diagnostics["scale"]
        diagnostics["status"] = "warning"
        if verbose:
            _print_diagnostics(diagnostics, {})
        return aligned, diagnostics

    # ---- RANSAC 线性回归（与 LabelAny3D 完全一致）----
    # fit_intercept=False: 无截距，只求 scale
    ransac = RANSACRegressor(
        estimator=LinearRegression(fit_intercept=False),
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        random_state=42,
    )

    try:
        ransac.fit(rel_valid, met_valid)
        scale = ransac.estimator_.coef_[0, 0]
        inlier_mask = ransac.inlier_mask_
    except Exception:
        if verbose:
            print(f"警告: RANSAC拟合异常，使用中值对齐")
        diagnostics["method"] = "median"
        diagnostics["scale"] = np.median(met_valid) / (np.median(rel_valid) + 1e-6)
        aligned = relative_depth * diagnostics["scale"]
        diagnostics["status"] = "warning"
        if verbose:
            _print_diagnostics(diagnostics, {})
        return aligned, diagnostics

    # ---- 安全检查 ----
    if not np.isfinite(scale) or scale <= 0:
        if verbose:
            print(f"警告: scale={scale} 无效，使用中值对齐")
        diagnostics["method"] = "median"
        diagnostics["scale"] = np.median(met_valid) / (np.median(rel_valid) + 1e-6)
        aligned = relative_depth * diagnostics["scale"]
        diagnostics["status"] = "warning"
        if verbose:
            _print_diagnostics(diagnostics, {})
        return aligned, diagnostics

    # ---- 计算诊断指标 ----
    diagnostics["scale"] = float(scale)
    inlier_ratio = float(inlier_mask.sum() / len(rel_valid))
    diagnostics["inlier_ratio"] = inlier_ratio

    # 计算对齐后误差（仅在内点上）
    rel_inlier = rel_valid[inlier_mask]
    met_inlier = met_valid[inlier_mask]
    pred_inlier = rel_inlier * scale
    errors = np.abs(pred_inlier.flatten() - met_inlier.flatten())
    diagnostics["median_error"] = float(np.median(errors))
    diagnostics["p95_error"] = float(np.percentile(errors, 95))

    # ---- 内点比例判断（与 LabelAny3D 对齐）----
    if inlier_ratio < 0.5:
        if verbose:
            print(f"警告: RANSAC内点比例 {inlier_ratio:.1%} < 50%，使用中值对齐")
        diagnostics["method"] = "median"
        diagnostics["scale"] = np.median(met_valid) / (np.median(rel_valid) + 1e-6)
        diagnostics["status"] = "warning"
        aligned = relative_depth * diagnostics["scale"]
        if verbose:
            _print_diagnostics(diagnostics, {"inlier_ratio": inlier_ratio})
        return aligned, diagnostics

    # ---- 对齐 ----
    aligned = relative_depth * scale
    diagnostics["method"] = "ransac"

    # ---- 状态判定 ----
    if inlier_ratio >= 0.7 and diagnostics["p95_error"] < 1.0:
        diagnostics["status"] = "success"
    elif inlier_ratio >= 0.5 and diagnostics["p95_error"] < 2.0:
        diagnostics["status"] = "success"
    else:
        diagnostics["status"] = "warning"

    if verbose:
        _print_diagnostics(diagnostics, {"inlier_ratio": inlier_ratio})

    return aligned, diagnostics


def _print_diagnostics(diagnostics: dict, extra: dict):
    """打印对齐诊断信息（格式化，便于快速定位问题）"""
    status_icon = {
        "success": "✅",
        "warning": "⚠️",
        "failed": "❌",
    }.get(diagnostics["status"], "?")

    print(f"{status_icon} [深度对齐] "
          f"方法={diagnostics['method']} "
          f"scale={diagnostics['scale']:.4f} "
          f"内点率={diagnostics['inlier_ratio']:.1%} "
          f"中位误差={diagnostics['median_error']:.3f}m "
          f"P95误差={diagnostics['p95_error']:.3f}m "
          f"有效点数={diagnostics['num_valid_points']} "
          f"状态={diagnostics['status']}")


def depth_to_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    将深度图转换为点云
    
    参考 LabelAny3D 的实现
    
    Args:
        depth: 深度图 (H, W)
        K: 相机内参 (3, 3)
        
    Returns:
        points: 点云 (H, W, 3), 相机坐标系
    """
    H, W = depth.shape
    
    # 生成像素坐标
    y, x = np.meshgrid(
        np.arange(H), 
        np.arange(W), 
        indexing='ij'
    )
    
    # 归一化坐标
    x_norm = (x - K[0, 2]) / K[0, 0]
    y_norm = (y - K[1, 2]) / K[1, 1]
    
    # 反投影到相机坐标系
    points = np.stack([
        x_norm * depth,
        y_norm * depth,
        depth,
    ], axis=-1)
    
    return points


class MoGeDepthProFusion(nn.Module):
    """
    MoGe和Depth Pro融合深度模型
    
    参考 LabelAny3D 的融合方法:
    - MoGe: 提供高频细节(边缘、纹理),输出尺度不变深度
    - Depth Pro: 提供物理尺度(绝对深度值),输出度量深度
    - RANSAC对齐: 将MoGe深度对齐到Depth Pro的物理尺度
    
    融合策略:
    1. ransac_align: RANSAC线性回归对齐(推荐)
    2. weighted_adaptive: 自适应加权融合
    3. frequency_split: 频域分离融合
    4. weighted: 简单加权融合
    """
    
    def __init__(
        self,
        moge_model=None,
        depthpro_model=None,
        fusion_method='ransac_align',
        moge_weight=0.3,
        depthpro_weight=0.7,
        use_confidence=True,
    ):
        """
        Args:
            moge_model: MoGe深度模型实例
            depthpro_model: Depth Pro深度模型实例
            fusion_method: 融合方法
                - 'ransac_align': RANSAC线性对齐(推荐,参考LabelAny3D)
                - 'weighted_adaptive': 自适应加权融合
                - 'frequency_split': 频域分离融合
                - 'weighted': 简单加权融合
            moge_weight: MoGe权重
            depthpro_weight: Depth Pro权重
            use_confidence: 是否使用置信度加权
        """
        super().__init__()
        self.moge_model = moge_model
        self.depthpro_model = depthpro_model
        self.fusion_method = fusion_method
        self.moge_weight = moge_weight
        self.depthpro_weight = depthpro_weight
        self.use_confidence = use_confidence
        
        # 自适应融合参数
        if fusion_method == 'weighted_adaptive':
            self.depth_threshold = nn.Parameter(torch.tensor(3.0))
            self.near_weight_moge = 0.4
            self.far_weight_depthpro = 0.8
        
        # 频域融合参数
        if fusion_method == 'frequency_split':
            self.register_buffer('lowpass_kernel', self._create_lowpass_kernel())
    
    def _create_lowpass_kernel(self, size=5, sigma=1.0):
        """创建低通滤波核"""
        kernel = torch.zeros(1, 1, size, size)
        center = size // 2
        for i in range(size):
            for j in range(size):
                dist = ((i - center)**2 + (j - center)**2)**0.5
                kernel[0, 0, i, j] = torch.exp(-dist**2 / (2 * sigma**2))
        return kernel / kernel.sum()
    
    def forward_moge(self, input_dict):
        """运行 MoGe 模型"""
        if self.moge_model is None:
            return None, None
        
        try:
            if hasattr(self.moge_model, 'forward'):
                output = self.moge_model(input_dict)
            else:
                output = self.moge_model(**input_dict)
            
            if isinstance(output, dict):
                depth_moge = output.get('depth_maps', output.get('depth', None))
                confidence_moge = output.get('confidence', output.get('mask', None))
            elif isinstance(output, torch.Tensor):
                depth_moge = output
                confidence_moge = None
            else:
                depth_moge, confidence_moge = output[0], output[1] if len(output) > 1 else None
            
            if depth_moge is not None:
                if depth_moge.dim() == 2:
                    depth_moge = depth_moge.unsqueeze(0).unsqueeze(0)
                elif depth_moge.dim() == 3:
                    depth_moge = depth_moge.unsqueeze(0)
            
            return depth_moge, confidence_moge
        except Exception as e:
            print(f"警告: MoGe推理失败: {e}")
            return None, None
    
    def fuse_ransac_align(self, depth_moge, depth_depthpro, mask_moge=None):
        """
        RANSAC线性对齐融合 (参考 LabelAny3D)
        
        步骤:
        1. MoGe 输出尺度不变深度
        2. Depth Pro 输出度量深度
        3. 使用 RANSAC 线性回归: metric = scale * relative + offset
        4. 将 MoGe 深度对齐到 Depth Pro 尺度
        """
        if depth_moge is None and depth_depthpro is None:
            return None
        if depth_moge is None:
            return depth_depthpro
        if depth_depthpro is None:
            return depth_moge
        
        # 转换到 numpy (RANSAC 在 CPU 上运行更快)
        moge_np = depth_moge.squeeze().cpu().numpy()
        depthpro_np = depth_depthpro.squeeze().cpu().numpy()
        
        # MoGe mask (如果可用)
        mask_np = mask_moge.squeeze().cpu().numpy() if mask_moge is not None else None
        
        # RANSAC 对齐（返回对齐深度 + 诊断信息）
        aligned_depth, diag = align_depth_ransac(moge_np, depthpro_np, mask=mask_np)
        self._last_depth_diag = diag  # 供外部查询诊断信息
        
        # 转换回 tensor
        aligned_tensor = torch.from_numpy(aligned_depth).to(depth_moge.device)
        if aligned_tensor.dim() == 2:
            aligned_tensor = aligned_tensor.unsqueeze(0).unsqueeze(0)
        
        return aligned_tensor
    
    def fuse_weighted_adaptive(self, depth_moge, depth_depthpro, confidence_moge=None, confidence_depthpro=None):
        """自适应加权融合"""
        if depth_moge is None and depth_depthpro is None:
            return None
        if depth_moge is None:
            return depth_depthpro
        if depth_depthpro is None:
            return depth_moge
        
        depth_moge_norm = self._normalize_depth(depth_moge)
        depth_depthpro_norm = self._normalize_depth(depth_depthpro)
        
        depth_scale = depth_depthpro / (depth_depthpro.mean() + 1e-6)
        weight_moge = torch.sigmoid(self.depth_threshold - depth_scale) * self.near_weight_moge
        weight_depthpro = (1 - torch.sigmoid(self.depth_threshold - depth_scale)) * self.far_weight_depthpro
        
        total_weight = weight_moge + weight_depthpro + 1e-6
        weight_moge = weight_moge / total_weight
        weight_depthpro = weight_depthpro / total_weight
        
        if self.use_confidence and confidence_moge is not None and confidence_depthpro is not None:
            weight_moge = weight_moge * confidence_moge
            weight_depthpro = weight_depthpro * confidence_depthpro
            total_weight = weight_moge + weight_depthpro + 1e-6
            depth_fused = (weight_moge * depth_moge_norm + weight_depthpro * depth_depthpro_norm) / total_weight
        else:
            depth_fused = weight_moge * depth_moge_norm + weight_depthpro * depth_depthpro_norm
        
        depth_fused = self._denormalize_depth(depth_fused, depth_depthpro)
        
        return depth_fused
    
    def fuse_frequency_split(self, depth_moge, depth_depthpro, confidence_moge=None, confidence_depthpro=None):
        """频域分离融合"""
        if depth_moge is None and depth_depthpro is None:
            return None
        if depth_moge is None:
            return depth_depthpro
        if depth_depthpro is None:
            return depth_moge
        
        depth_depthpro_low = F.conv2d(
            depth_depthpro,
            self.lowpass_kernel.expand(depth_depthpro.shape[1], -1, -1, -1),
            padding=self.lowpass_kernel.shape[-1] // 2,
            groups=depth_depthpro.shape[1]
        )
        
        depth_moge_high = depth_moge - F.conv2d(
            depth_moge,
            self.lowpass_kernel.expand(depth_moge.shape[1], -1, -1, -1),
            padding=self.lowpass_kernel.shape[-1] // 2,
            groups=depth_moge.shape[1]
        )
        
        depth_fused = depth_depthpro_low + depth_moge_high * 0.5
        
        return depth_fused
    
    def _normalize_depth(self, depth):
        """归一化深度到[0,1]"""
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min < 1e-6:
            return depth
        return (depth - depth_min) / (depth_max - depth_min + 1e-6)
    
    def _denormalize_depth(self, depth_norm, reference_depth):
        """反归一化到参考深度尺度"""
        ref_min = reference_depth.min()
        ref_max = reference_depth.max()
        return depth_norm * (ref_max - ref_min + 1e-6) + ref_min
    
    def forward(self, input_dict):
        """前向传播: 融合 MoGe 和 Depth Pro"""
        depth_moge, confidence_moge = self.forward_moge(input_dict)
        
        depthpro_full_output = None
        if self.depthpro_model is not None:
            try:
                if hasattr(self.depthpro_model, 'forward'):
                    depthpro_full_output = self.depthpro_model(input_dict)
                else:
                    depthpro_full_output = self.depthpro_model(**input_dict)
            except Exception as e:
                print(f"警告: Depth Pro完整推理失败: {e}")
        
        if isinstance(depthpro_full_output, dict):
            depth_depthpro = depthpro_full_output.get('depth_maps', depthpro_full_output.get('depth', None))
            confidence_depthpro = depthpro_full_output.get('confidence', None)
        elif isinstance(depthpro_full_output, torch.Tensor):
            depth_depthpro = depthpro_full_output
            confidence_depthpro = None
        elif depthpro_full_output is not None:
            depth_depthpro, confidence_depthpro = depthpro_full_output[0], depthpro_full_output[1] if len(depthpro_full_output) > 1 else None
        else:
            depth_depthpro, confidence_depthpro = None, None
        
        if depth_depthpro is not None:
            if depth_depthpro.dim() == 2:
                depth_depthpro = depth_depthpro.unsqueeze(0).unsqueeze(0)
            elif depth_depthpro.dim() == 3:
                depth_depthpro = depth_depthpro.unsqueeze(0)
        
        # 融合
        if self.fusion_method == 'ransac_align':
            depth_fused = self.fuse_ransac_align(depth_moge, depth_depthpro, confidence_moge)
        elif self.fusion_method == 'weighted':
            depth_fused = self.fuse_weighted(depth_moge, depth_depthpro, confidence_moge, confidence_depthpro)
        elif self.fusion_method == 'weighted_adaptive':
            depth_fused = self.fuse_weighted_adaptive(depth_moge, depth_depthpro, confidence_moge, confidence_depthpro)
        elif self.fusion_method == 'frequency_split':
            depth_fused = self.fuse_frequency_split(depth_moge, depth_depthpro, confidence_moge, confidence_depthpro)
        else:
            raise ValueError(f"未知的融合方法: {self.fusion_method}")
        
        if depth_fused is None:
            if depth_depthpro is not None:
                depth_fused = depth_depthpro
            elif depth_moge is not None:
                depth_fused = depth_moge
            else:
                raise RuntimeError("MoGe和Depth Pro都不可用")
        
        if confidence_moge is not None and confidence_depthpro is not None:
            confidence_fused = (confidence_moge + confidence_depthpro) / 2.0
        elif confidence_depthpro is not None:
            confidence_fused = confidence_depthpro
        elif confidence_moge is not None:
            confidence_fused = confidence_moge
        else:
            confidence_fused = None
        
        output_dict = {
            'depth_maps': depth_fused,
            'confidence': confidence_fused,
        }
        
        if isinstance(depthpro_full_output, dict):
            for key in ['metric_features', 'camera_features', 'depth_features', 'scale', 'shift', 'pred_K', 'rays']:
                if key in depthpro_full_output:
                    output_dict[key] = depthpro_full_output[key]
        
        return output_dict
    
    def fuse_weighted(self, depth_moge, depth_depthpro, confidence_moge=None, confidence_depthpro=None):
        """简单加权融合"""
        if depth_moge is None and depth_depthpro is None:
            return None
        if depth_moge is None:
            return depth_depthpro
        if depth_depthpro is None:
            return depth_moge
        
        depth_moge_norm = self._normalize_depth(depth_moge)
        depth_depthpro_norm = self._normalize_depth(depth_depthpro)
        
        if self.use_confidence and confidence_moge is not None and confidence_depthpro is not None:
            w_moge = confidence_moge * self.moge_weight
            w_depthpro = confidence_depthpro * self.depthpro_weight
            total_weight = w_moge + w_depthpro + 1e-6
            depth_fused = (w_moge * depth_moge_norm + w_depthpro * depth_depthpro_norm) / total_weight
        else:
            depth_fused = self.moge_weight * depth_moge_norm + self.depthpro_weight * depth_depthpro_norm
        
        depth_fused = self._denormalize_depth(depth_fused, depth_depthpro)
        
        return depth_fused


def create_moge_depthpro_fusion(
    moge_model=None, 
    depthpro_model=None, 
    fusion_method='ransac_align',
    **kwargs
):
    """工厂函数: 创建 MoGe+Depth Pro 融合模型"""
    return MoGeDepthProFusion(
        moge_model=moge_model,
        depthpro_model=depthpro_model,
        fusion_method=fusion_method,
        **kwargs
    )
