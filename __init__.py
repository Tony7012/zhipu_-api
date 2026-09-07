# -*- coding: utf-8 -*-
"""ComfyUI-Zhipu-API:在 ComfyUI / RunningHub 中调用智谱 AI 开放平台。

节点:
- 智谱图像分析 (ZhipuImageAnalysis): 图像 + 分析要求 → 视觉大模型 → 文字
- 智谱文字对话 (ZhipuTextChat):      提示词扩写 / 改写 / 问答
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
