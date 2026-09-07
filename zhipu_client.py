# -*- coding: utf-8 -*-
"""智谱 AI (BigModel) 开放平台 API 客户端,供 ComfyUI / RunningHub 节点调用。

接口文档: https://docs.bigmodel.cn/api-reference/模型-api/对话补全
调用地址: POST https://open.bigmodel.cn/api/paas/v4/chat/completions
"""

import base64
import io
import os
import re
import time

import numpy as np
import requests
from PIL import Image

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

API_KEY_URL = "https://bigmodel.cn/usercenter/apikeys"

KEY_HELP = (
    "未找到智谱 API Key:请在节点的 api_key 输入框中填写,"
    "或在环境变量中设置 ZHIPU_API_KEY / ZHIPUAI_API_KEY。"
    "API Key 获取地址: {url}"
).format(url=API_KEY_URL)

# 单次请求最多携带的图片数量(超出部分截断)
MAX_IMAGES_PER_REQUEST = 16


def resolve_api_key(api_key):
    """按 节点输入 → 环境变量 ZHIPU_API_KEY → ZHIPUAI_API_KEY 的顺序取 Key。"""
    candidates = [
        api_key or "",
        os.environ.get("ZHIPU_API_KEY", ""),
        os.environ.get("ZHIPUAI_API_KEY", ""),
    ]
    for key in candidates:
        key = key.strip()
        if key:
            return key
    raise ValueError(KEY_HELP)


def tensor_to_data_urls(image, send_batch=False, max_side=4096, jpeg_quality=92):
    """把 ComfyUI 的 IMAGE 张量 [B,H,W,C](float 0..1)转成 Base64 Data URL 列表。

    send_batch=False 时只取批次第一帧;True 时整批作为多图发送(上限 16 张)。
    RGB 帧编码为 JPEG(体积小),灰度帧编码为 PNG;最长边超过 max_side 会等比缩小。
    """
    arr = np.asarray(image.detach().cpu().numpy())
    if arr.ndim == 4:
        frames = arr if send_batch else [arr[0]]
    elif arr.ndim == 3:
        frames = [arr]
    else:
        raise ValueError("不支持的图像张量维度: %d(期望 [B,H,W,C])" % arr.ndim)

    urls = []
    for frame in frames:
        frame = np.clip(frame, 0.0, 1.0)
        img = Image.fromarray((frame * 255.0).astype(np.uint8))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        if img.mode == "RGB":
            img.save(buf, format="JPEG", quality=jpeg_quality)
            mime = "image/jpeg"
        else:
            img.save(buf, format="PNG")
            mime = "image/png"
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        urls.append("data:%s;base64,%s" % (mime, b64))
        if len(urls) >= MAX_IMAGES_PER_REQUEST:
            break

    if send_batch and len(frames) > MAX_IMAGES_PER_REQUEST:
        print(
            "[ComfyUI-Zhipu-API] 输入图像 %d 张,超过单次上限 %d,已截断"
            % (len(frames), MAX_IMAGES_PER_REQUEST)
        )
    return urls


def _extract_error(resp):
    """从错误响应里提取尽量可读的信息。"""
    try:
        err = resp.json().get("error", {}) or {}
        code = err.get("code", "")
        msg = err.get("message", "")
        if msg:
            return "HTTP %s (code=%s): %s" % (resp.status_code, code or "?", msg)
    except Exception:
        pass
    return "HTTP %s: %s" % (resp.status_code, resp.text[:300])


def _extract_content(data):
    """从 200 响应里取出最终回答文本(兼容内容块列表、剥离 <think> 思考段)。"""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("API 返回结构异常: %s" % str(data)[:500])

    content = message.get("content")
    if isinstance(content, list):  # 部分模型按内容块返回
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    text = (content or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        raise RuntimeError(
            "模型未返回内容(可能被 max_tokens 截断或内容审核拦截): %s" % str(data)[:500]
        )
    return text


def chat(api_key, model, messages, temperature=0.7, max_tokens=4096,
         timeout=300, max_retries=3):
    """调用智谱对话补全接口,返回模型回答文本。

    429 / 5xx / 网络错误自动重试(带退避),其余错误立即抛出中文提示。
    """
    key = resolve_api_key(api_key)
    headers = {
        "Authorization": "Bearer %s" % key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    last_err = None
    attempt = 0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = "网络错误: %s: %s" % (type(e).__name__, e)
        else:
            if resp.status_code == 200:
                return _extract_content(resp.json())
            last_err = _extract_error(resp)
            if resp.status_code != 429 and resp.status_code < 500:
                break  # 鉴权/参数类错误,重试没有意义
        if attempt < max_retries:
            time.sleep(2 * attempt)

    raise RuntimeError(
        "智谱 API 调用失败(模型: %s,第 %d 次尝试): %s" % (model, attempt, last_err)
    )
