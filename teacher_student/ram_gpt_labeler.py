# Copyright (c) Teacher-Student Distillation Pipeline
"""
RAM模型自动标签提取 + Gemini/LLM过滤 + Grounding DINO检测
替换手动text prompt输入

- RAM Plus: 自动识别图像中的所有标签
- Gemini: 过滤背景词、颜色词、场景词，保留核心实体
- 输出: Grounding DINO 可用的 text prompt
"""
import os
import sys
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUNDED_SAM_DIR = os.path.join(PROJECT_ROOT, "Grounded-SAM-2")

try:
    from ram.models import ram_plus
    from ram import inference_ram
    from ram import get_transform
    RAM_AVAILABLE = True
except ImportError:
    RAM_AVAILABLE = False
    print("警告: RAM模型未安装,请运行: pip install ram-tag")

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("警告: Google Gemini API未安装,请运行: pip install google-genai")


LABEL_FILTER_PROMPT = """I want you to act as an English expert in words understanding.\
Please understand every word in the word lists:
{tags}
Select appropriate words to form a new word list as following rules:
    -the word should refer to a specific object such as (chair, table, television, etc.), delete ambiguous things such as (appliance, furniture, etc.);
    -the word should not represent colors, like (brown, white, black, gray, etc.);
    -the word should not refer to room types or some scene, like (bathroom, living room, kitchen, bedroom, office, street, cinema, indoor, outdoor, etc.);
    -if there are some words with similar meanings, please keep the most general term and delete others, for example, retaining 'table' and delete 'kitchen table', 'glass table' in (table, kitchen table, glass table);
    -do not output any explanation or dialog, only output the final selected word list."""


class RAMGPTLabeler:
    """
    RAM模型 + Gemini/LLM过滤 + Grounding DINO检测流程

    1. 使用 RAM Plus 识别图像中的所有可能标签
    2. 使用 Gemini 过滤背景词、颜色词、场景词
    3. 生成 Grounding DINO 可用的 text prompt

    输出格式:
    - 单图像: "bed. desk. chair. sofa."
    - 多图像: "bed. desk. chair. sofa." (合并所有唯一标签)
    """

    def __init__(self, device="cuda", ram_model_path=None, gemini_api_key=None,
                 use_gemini=True, image_size=384, gemini_model="gemini-2.0-flash"):
        """
        Args:
            device: 计算设备
            ram_model_path: RAM模型权重路径
            gemini_api_key: Google Gemini API密钥(可选,也可通过环境变量 GEMINI_API_KEY 设置)
            use_gemini: 是否使用Gemini过滤, False时使用规则过滤
            image_size: RAM模型输入图像大小, 默认384
            gemini_model: Gemini模型名称, 默认 gemini-2.0-flash
        """
        self.device = device
        self.use_gemini = use_gemini and GEMINI_AVAILABLE
        self.image_size = image_size
        self.gemini_model = gemini_model
        self.gemini_client = None

        # 初始化RAM模型
        self.ram_model = None
        self.ram_transform = None
        if RAM_AVAILABLE:
            try:
                if ram_model_path is None:
                    ram_model_path = os.path.join(PROJECT_ROOT, "weights", "ram_plus_swin_large_14m.pth")
                    if not os.path.exists(ram_model_path):
                        ram_model_path = os.path.join(GROUNDED_SAM_DIR, "checkpoints", "ram_plus_swin_large_14m.pth")

                if os.path.exists(ram_model_path):
                    self.ram_model = ram_plus(
                        pretrained=ram_model_path,
                        image_size=image_size,
                        vit='swin_l'
                    )
                    self.ram_model.eval()
                    self.ram_model = self.ram_model.to(device)

                    self.ram_transform = get_transform(image_size=image_size)

                    print(f">> RAM模型加载成功: {ram_model_path}")
                else:
                    print(f"警告: RAM模型权重未找到: {ram_model_path}")
                    print(">> 请下载: https://github.com/xinyu1205/recognize-anything")
            except Exception as e:
                print(f"警告: RAM模型加载失败: {e}")

        # 初始化Gemini API
        if self.use_gemini:
            api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    self.gemini_client = genai.Client(api_key=api_key)
                    print(f">> Gemini 客户端初始化成功 (model: {self.gemini_model})")
                except Exception as e:
                    print(f"警告: Gemini 客户端初始化失败: {e}, 将使用规则过滤")
                    self.use_gemini = False
                    self.gemini_client = None
            else:
                print("警告: 未设置Gemini API密钥(设置环境变量 GEMINI_API_KEY), 将使用规则过滤")
                self.use_gemini = False

        # 背景词过滤规则
        self.background_keywords = {
            # 颜色
            'color', 'brown', 'white', 'black', 'gray', 'grey', 'red', 'blue',
            'green', 'yellow', 'orange', 'purple', 'pink', 'beige', 'tan', 'gold',
            'silver', 'metallic',
            # 场景/房间
            'room', 'floor', 'wall', 'ceiling', 'corner', 'background',
            'indoor', 'outdoor', 'space', 'area', 'surface', 'texture',
            'bathroom', 'living room', 'kitchen', 'bedroom', 'office', 'hallway',
            'street', 'cinema', 'outdoor', 'balcony',
            # 其他背景
            'light', 'shadow', 'reflection', 'window', 'door', 'frame',
            'furniture', 'appliance', 'equipment', 'object', 'item',
            'left', 'right', 'top', 'bottom', 'front', 'back', 'side',
        }

        # 模糊类别
        self.ambiguous_categories = {
            'furniture', 'appliance', 'equipment', 'object', 'item',
            'thing', 'stuff', 'property', 'asset', 'fixture',
            'structure', 'component', 'element', 'material',
        }

    def extract_tags_ram(self, image_rgb):
        """
        使用RAM模型提取图像中的所有可能标签

        Args:
            image_rgb: RGB图像 (H, W, 3) numpy数组或 PIL Image

        Returns:
            tags: 标签列表
        """
        if self.ram_model is None or self.ram_transform is None:
            return []

        try:
            if isinstance(image_rgb, np.ndarray):
                image_pil = Image.fromarray(image_rgb)
            else:
                image_pil = image_rgb

            image_tensor = self.ram_transform(image_pil).unsqueeze(0).to(self.device)

            res = inference_ram(image_tensor, self.ram_model)

            tags_str = res[0] if hasattr(res, '__getitem__') else res

            if isinstance(tags_str, str):
                tags = [t.strip() for t in tags_str.split(" | ") if t.strip()]
            elif isinstance(tags_str, (list, tuple)):
                tags = [str(t).strip() for t in tags_str if t]
            else:
                tags = []

            return tags

        except Exception as e:
            print(f"警告: RAM标签提取失败: {e}")
            return []

    def extract_tags_batch(self, images_rgb):
        """
        批量提取多张图像的标签

        Args:
            images_rgb: RGB图像列表

        Returns:
            all_tags: 所有图像的合并标签列表 (去重)
            tags_with_count: (标签, 出现次数) 元组列表
        """
        if self.ram_model is None:
            return [], []

        all_tags = []

        for image_rgb in images_rgb:
            tags = self.extract_tags_ram(image_rgb)
            all_tags.extend(tags)

        if len(all_tags) == 0:
            return [], []

        unique_tags, counts = np.unique(all_tags, return_counts=True)

        sorted_indices = np.argsort(-counts)
        tags_with_count = list(zip(unique_tags[sorted_indices], counts[sorted_indices]))

        return list(unique_tags), tags_with_count

    def filter_tags_llm(self, tags):
        """
        使用 Gemini 过滤背景词

        Args:
            tags: RAM提取的原始标签列表

        Returns:
            filtered_tags: 过滤后的核心实体标签列表
        """
        if not self.use_gemini or len(tags) == 0:
            return self.filter_tags_rule(tags)

        try:
            tags_str = "\n".join(tags[:100])

            prompt = LABEL_FILTER_PROMPT.format(tags=tags_str)

            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config={"temperature": 0.2}  # 降低温度确保输出格式稳定
            )

            filtered_text = response.text.strip()

            filtered_tags = self._parse_llm_output(filtered_text)

            print(f">> Gemini 过滤后保留 {len(filtered_tags)} 个核心实体")
            if filtered_tags:
                print(f">>    过滤后标签: {filtered_tags[:10]}...")

            return filtered_tags

        except Exception as e:
            print(f"警告: Gemini 过滤失败: {e}, 回退到规则过滤")
            return self.filter_tags_rule(tags)

    def _parse_llm_output(self, llm_output):
        """
        解析 Gemini 的输出格式

        Args:
            llm_output: Gemini 返回的原始文本

        Returns:
            tags: 解析后的标签列表
        """
        if not llm_output:
            return []

        for sep in ['\n', ',', '|', ';']:
            if sep in llm_output:
                tags = [t.strip() for t in llm_output.split(sep) if t.strip()]
                if tags:
                    return tags

        return [llm_output.strip()] if llm_output.strip() else []

    def filter_tags_rule(self, tags):
        """
        使用规则过滤背景词 (当 Gemini 不可用时的后备方案)

        1. 删除颜色词
        2. 删除房间/场景词
        3. 删除模糊类别
        4. 保留具体物体

        Args:
            tags: RAM提取的原始标签列表

        Returns:
            filtered_tags: 过滤后的核心实体标签列表
        """
        if len(tags) == 0:
            return []

        filtered = []
        for tag in tags:
            tag_lower = tag.lower().strip()

            if len(tag_lower) < 2:
                continue

            if tag_lower in self.background_keywords:
                continue

            if tag_lower in self.ambiguous_categories:
                continue

            is_background = False
            for bg in self.background_keywords:
                if bg in tag_lower.split() or tag_lower in bg.split():
                    is_background = True
                    break

            if not is_background:
                filtered.append(tag)

        filtered = self._merge_similar_tags(filtered)

        print(f">> 规则过滤后保留 {len(filtered)} 个核心实体")
        if filtered:
            print(f">>    过滤后标签: {filtered[:10]}...")

        return filtered

    def _merge_similar_tags(self, tags):
        """
        合并相似标签,保留最通用的词

        例如: ["table", "kitchen table", "glass table"] -> ["table"]
        """
        if len(tags) <= 1:
            return tags

        sorted_tags = sorted(tags, key=len)
        merged = []

        for tag in sorted_tags:
            tag_lower = tag.lower()

            is_substring = False
            for kept in merged:
                kept_lower = kept.lower()
                if tag_lower in kept_lower and tag_lower != kept_lower:
                    is_substring = True
                    break
                if kept_lower in tag_lower and kept_lower != tag_lower:
                    merged = [tag if t.lower() == kept_lower else t for t in merged]
                    is_substring = True
                    break

            if not is_substring:
                merged.append(tag)

        return merged

    def generate_text_prompt(self, image_rgb):
        """
        单图像: RAM提取 -> Gemini/规则过滤 -> 生成 Grounding DINO 可用的 text prompt

        Args:
            image_rgb: RGB图像 (H, W, 3) numpy数组

        Returns:
            text_prompt: Grounding DINO可用的文本提示,如 "bed. desk. chair."
        """
        raw_tags = self.extract_tags_ram(image_rgb)

        if len(raw_tags) == 0:
            print("警告: RAM未提取到任何标签")
            return None

        print(f">> RAM提取到 {len(raw_tags)} 个标签")

        filtered_tags = self.filter_tags_llm(raw_tags)

        if len(filtered_tags) == 0:
            print("警告: 过滤后无有效标签")
            return None

        text_prompt = ". ".join(filtered_tags) + "."

        print(f">> 最终 text prompt: {text_prompt}")
        return text_prompt

    def generate_text_prompt_batch(self, images_rgb):
        """
        多图像: RAM批量提取 -> Gemini/规则过滤 -> 生成 Grounding DINO 可用的 text prompt

        1. 对每张图像运行 RAM
        2. 合并所有唯一标签
        3. 通过 Gemini 过滤
        4. 生成统一的 text prompt

        Args:
            images_rgb: RGB图像列表

        Returns:
            text_prompt: Grounding DINO可用的文本提示
        """
        if len(images_rgb) == 0:
            return None

        unique_tags, tags_count = self.extract_tags_batch(images_rgb)

        if len(unique_tags) == 0:
            print("警告: RAM未提取到任何标签")
            return None

        print(f">> RAM从 {len(images_rgb)} 张图像提取到 {len(unique_tags)} 个唯一标签")

        filtered_tags = self.filter_tags_llm(unique_tags)

        if len(filtered_tags) == 0:
            print("警告: 过滤后无有效标签")
            return None

        text_prompt = ". ".join(filtered_tags) + "."

        print(f">> 最终 text prompt: {text_prompt}")
        return text_prompt


def create_ram_gpt_labeler(device="cuda", ram_model_path=None, gemini_api_key=None,
                           use_gemini=True, image_size=384, gemini_model="gemini-2.0-flash"):
    """
    工厂函数: 创建 RAMGPTLabeler 实例

    Args:
        device: 计算设备
        ram_model_path: RAM模型权重路径
        gemini_api_key: Google Gemini API密钥
        use_gemini: 是否使用Gemini过滤
        image_size: RAM模型输入大小
        gemini_model: Gemini模型名称

    Returns:
        RAMGPTLabeler 实例
    """
    return RAMGPTLabeler(
        device=device,
        ram_model_path=ram_model_path,
        gemini_api_key=gemini_api_key,
        use_gemini=use_gemini,
        image_size=image_size,
        gemini_model=gemini_model
    )


# ========== 使用示例 ==========
if __name__ == "__main__":
    import cv2

    labeler = create_ram_gpt_labeler(
        device="cuda",
        use_gemini=True,
        image_size=384
    )

    image = cv2.imread("test.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    text_prompt = labeler.generate_text_prompt(image_rgb)
    print(f"生成的 prompt: {text_prompt}")
