# 改进说明文档

本文档说明了对3D检测系统的三个主要改进。

## 改进1和3: RAM + Gemini 自动标签提取

### 功能说明
- **RAM模型**: 自动扫描图像,提取所有可能的标签(如"room, bed, floor, desk")
- **Gemini/LLM过滤**: 使用Gemini或规则过滤掉背景词(如room, floor, wall),保留核心实体(如bed, desk)
- **Grounding DINO**: 使用过滤后的标签进行2D检测

### 使用方法

#### 1. 安装依赖
```bash
# RAM模型(Recognize Anything Model)
pip install /tmp/recognize-anything

# Google Gemini API(可选,用于LLM过滤)
pip install google-genai
export GEMINI_API_KEY="your-api-key"
```

#### 2. 下载RAM模型权重
将RAM模型权重放到以下路径之一:
- `weights/ram_plus_swin_large_14m.pth`
- `Grounded-SAM-2/checkpoints/ram_plus_swin_large_14m.pth`

#### 3. 代码使用
```python
from teacher_student.teacher_detany3d import TeacherDetAny3D

# 启用RAM+GPT自动标签提取
teacher = TeacherDetAny3D(
    device="cuda",
    use_sam2_mask=True,
    use_ram_gpt=True,  # 启用RAM+Gemini
    ram_model_path=None,  # 自动查找默认路径
    gemini_api_key=None,  # 从环境变量GEMINI_API_KEY读取
)

# 现在可以不用提供text_prompt,系统会自动生成
rgb_np = ...  # RGB图像
K = ...  # 相机内参
pseudo_list, F_fused = teacher.generate_pseudo_3d_boxes(
    rgb_np, 
    text_prompt=None,  # 可选,如果use_ram_gpt=True会自动生成
    K=K
)
```

#### 4. 回退机制
- 如果RAM模型不可用,会自动回退到手动text_prompt
- 如果Gemini API不可用,会使用规则过滤(基于关键词列表)

---

## 改进2: MoGe + Depth Pro 深度融合

### 功能说明
- **MoGe**: 提供高频细节(边缘、纹理),适合近处物体
- **Depth Pro**: 提供物理尺度(绝对深度值),保证深度准确性
- **融合策略**: 
  - `weighted_adaptive`: 自适应加权(近处更依赖MoGe,远处更依赖Depth Pro)
  - `weighted`: 简单加权融合
  - `frequency_split`: 频域分离融合(MoGe提供高频,Depth Pro提供低频)

### 使用方法

#### 1. 准备模型
需要准备MoGe和Depth Pro模型实例。由于模型加载方式可能不同,需要在代码中手动加载:

```python
# 示例: 加载MoGe模型
moge_model = load_moge_model("path/to/moge/weights.pth")

# 示例: 加载Depth Pro模型  
depthpro_model = load_depthpro_model("path/to/depthpro/weights.pth")
```

#### 2. 配置ImageEncoderViT
```python
from types import SimpleNamespace
from detany3d_frontend.image_encoder import ImageEncoderViT

cfg = SimpleNamespace(
    # ... 其他配置 ...
    use_moge_depthpro=True,  # 启用融合
    moge_model=moge_model,  # MoGe模型实例
    depthpro_model=depthpro_model,  # Depth Pro模型实例
    fusion_method='weighted_adaptive',  # 融合方法
    moge_weight=0.3,  # MoGe权重
    depthpro_weight=0.7,  # Depth Pro权重
)

image_encoder = ImageEncoderViT(
    img_size=896,
    patch_size=16,
    embed_dim=1280,
    depth=32,
    num_heads=16,
    cfg=cfg,
).to(device)
```

#### 3. 当前实现状态
- ✅ 融合框架已实现
- ⚠️ MoGe和Depth Pro模型需要根据实际模型接口手动加载
- ⚠️ 如果模型不可用,会自动回退到单一UniDepth

---

## 改进优势

### 改进1和3的优势
1. **全自动**: 无需手动输入text prompt
2. **更准确**: RAM能识别更多物体,GPT能过滤背景词
3. **可扩展**: 支持新物体类别,无需更新代码

### 改进2的优势
1. **细节+尺度**: MoGe提供高频细节,Depth Pro保证物理尺度
2. **自适应融合**: 根据深度范围自动调整权重
3. **提升上限**: 解决2D Aggregator归一化导致的尺度问题(如0.3m沙发)

---

## 注意事项

1. **RAM模型**: 需要下载权重文件,如果不可用会自动回退
2. **GPT API**: 需要OpenAI API密钥,如果不可用会使用规则过滤
3. **MoGe/Depth Pro**: 需要根据实际模型接口实现加载函数
4. **兼容性**: 所有改进都有回退机制,不影响现有代码

---

## 文件结构

```
teacher_student/
├── ram_gpt_labeler.py          # RAM+GPT标签提取模块
└── teacher_detany3d.py        # 修改后的教师模型(集成RAM+GPT)

detany3d_frontend/depth_predictor/
├── moge_depthpro_fusion.py    # MoGe+Depth Pro融合模块
└── unidepth.py                # 原始UniDepth(作为后备)

detany3d_frontend/
└── image_encoder.py            # 修改后的图像编码器(支持融合)
```

---

## 后续工作

1. **MoGe/Depth Pro模型加载**: 需要根据实际模型实现加载函数
2. **性能优化**: 可以缓存RAM/GPT结果,避免重复计算
3. **参数调优**: 融合权重和阈值可以根据数据集调整
