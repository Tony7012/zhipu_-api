# -*- coding: utf-8 -*-
"""ComfyUI / RunningHub 智谱 AI 节点定义。

提供两个节点:
- ZhipuImageAnalysis  智谱图像分析:图像 + 分析要求 → 视觉大模型 → 文字结果
- ZhipuTextChat       智谱文字对话:提示词扩写 / 改写 / 问答
"""

try:
    from .zhipu_client import chat, tensor_to_data_urls
except ImportError:  # 以脚本方式直接加载时(本地测试)
    from zhipu_client import chat, tensor_to_data_urls

# 视觉模型(下拉菜单;free=免费)。custom_model 非空时优先生效。
VISION_MODELS = [
    "glm-5.3-flash",          # 原生多模态,效果最强(推荐)
    "glm-4.6v",
    "glm-4.6v-flash",         # 免费
    "glm-4.1v-thinking-flash",  # 免费
    "glm-4v-flash",           # 免费(仅 1K 输出,轻量打标够用)
]

# 文本模型
TEXT_MODELS = [
    "glm-5.3",                # 旗舰
    "glm-5.3-flash",          # 高性价比多模态
    "glm-4.7",
    "glm-4.7-flash",          # 免费
    "glm-4.6",
    "glm-4.5-flash",          # 免费
    "glm-4-flash-250414",     # 免费
]

FREE_MODELS = {
    "glm-4.6v-flash", "glm-4.1v-thinking-flash", "glm-4v-flash",
    "glm-4.7-flash", "glm-4.5-flash", "glm-4-flash-250414",
}


def _resolve_model(model, custom_model):
    model_id = (custom_model or "").strip() or model
    tag = "免费" if model_id in FREE_MODELS else "付费"
    print("[ComfyUI-Zhipu-API] 使用模型: %s (%s)" % (model_id, tag))
    return model_id


class ZhipuImageAnalysis:
    """智谱图像分析:输入图像 + 分析要求 → 视觉大模型 → 文字结果。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "待分析的图像(批量图像默认取第一帧)"}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "请详细描述这张图片的内容,包括主体、场景、风格等细节。",
                    "tooltip": "分析要求:告诉模型你想从图片中得到什么,例如反推提示词、识别文字、描述画面等",
                }),
                "model": (VISION_MODELS, {
                    "default": "glm-5.3-flash",
                    "tooltip": "视觉模型;glm-4.6v-flash / glm-4.1v-thinking-flash / glm-4v-flash 免费",
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "可选:填写任意模型ID(优先于下拉菜单),用于使用最新模型",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "智谱API Key(https://bigmodel.cn/usercenter/apikeys);留空则读取环境变量 ZHIPU_API_KEY",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "可选:系统提示词,用于设定模型角色与输出格式",
                }),
                "send_batch": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "开启后把整批图像作为多图一起发送(多图对比/连贯画面分析),上限16张",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "采样温度,越高越发散",
                }),
                "max_tokens": ("INT", {
                    "default": 4096, "min": 256, "max": 32768, "step": 64,
                    "tooltip": "最大输出 token 数",
                }),
                "timeout": ("INT", {
                    "default": 300, "min": 30, "max": 600, "step": 10,
                    "tooltip": "API 超时时间(秒)",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("analysis_text",)
    FUNCTION = "analyze"
    CATEGORY = "ZhipuAI"
    OUTPUT_NODE = True

    def analyze(self, image, prompt, model, custom_model, api_key, system_prompt,
                send_batch, temperature, max_tokens, timeout):
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("请填写分析要求(prompt)")

        model_id = _resolve_model(model, custom_model)
        data_urls = tensor_to_data_urls(image, send_batch=send_batch)
        print("[ComfyUI-Zhipu-API] 发送 %d 张图像进行分析" % len(data_urls))

        content = [{"type": "image_url", "image_url": {"url": url}} for url in data_urls]
        content.append({"type": "text", "text": prompt})

        messages = []
        if (system_prompt or "").strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": content})

        text = chat(
            api_key=api_key,
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        print("[ComfyUI-Zhipu-API] 图像分析完成,输出 %d 字符" % len(text))
        return {"ui": {"text": [text]}, "result": (text,)}


class ZhipuTextChat:
    """智谱文字对话:提示词扩写 / 改写 / 问答。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "把下面这句话扩写成一段适合文生图的中文提示词:",
                    "tooltip": "你的问题或指令;可右键转换为输入,接入上游节点的文本",
                }),
                "model": (TEXT_MODELS, {
                    "default": "glm-5.3-flash",
                    "tooltip": "文本模型;glm-4.7-flash / glm-4.5-flash / glm-4-flash-250414 免费",
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "可选:填写任意模型ID(优先于下拉菜单)",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "智谱API Key;留空则读取环境变量 ZHIPU_API_KEY",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "可选:系统提示词,例如'你是专业的AI绘画提示词专家'",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "采样温度,越高越发散",
                }),
                "max_tokens": ("INT", {
                    "default": 4096, "min": 256, "max": 32768, "step": 64,
                    "tooltip": "最大输出 token 数",
                }),
                "timeout": ("INT", {
                    "default": 300, "min": 30, "max": 600, "step": 10,
                    "tooltip": "API 超时时间(秒)",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "chat_run"
    CATEGORY = "ZhipuAI"
    OUTPUT_NODE = True

    def chat_run(self, prompt, model, custom_model, api_key, system_prompt,
                 temperature, max_tokens, timeout):
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("请填写对话内容(prompt)")

        model_id = _resolve_model(model, custom_model)

        messages = []
        if (system_prompt or "").strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})

        text = chat(
            api_key=api_key,
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        print("[ComfyUI-Zhipu-API] 对话完成,输出 %d 字符" % len(text))
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "ZhipuImageAnalysis": ZhipuImageAnalysis,
    "ZhipuTextChat": ZhipuTextChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZhipuImageAnalysis": "智谱图像分析 (Zhipu Image Analysis)",
    "ZhipuTextChat": "智谱文字对话 (Zhipu Text Chat)",
}
