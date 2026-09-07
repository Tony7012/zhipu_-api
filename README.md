# ComfyUI-Zhipu-API

在 ComfyUI / RunningHub 中调用智谱 AI(BigModel)开放平台的 API,根据输入的图像和提示词分析生成内容。

## 节点说明

菜单分类:`ZhipuAI`

| 节点 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| 智谱图像分析 (ZhipuImageAnalysis) | 图像 + 分析要求 → 视觉大模型 → 文字 | `image` 图像、`prompt` 分析要求 | `analysis_text` 文字 |
| 智谱文字对话 (ZhipuTextChat) | 提示词扩写 / 改写 / 问答 | `prompt` 指令文本 | `text` 文字 |

两个节点执行完成后,结果会直接显示在节点下方的只读文本框中,同时可连接到下游节点(如 CLIP Text Encode)。

## 快速开始

1. 获取 API Key:打开 https://bigmodel.cn/usercenter/apikeys 注册并创建 Key(新用户通常有免费额度,免费模型不消耗额度)。
2. 把 API Key 粘贴到节点的 `api_key` 输入框(留空则读取服务器环境变量 `ZHIPU_API_KEY`)。
3. 连接图像、填写分析要求,运行即可。

## 参数说明

| 参数 | 说明 |
| --- | --- |
| model | 下拉选择模型 |
| custom_model | 填写任意模型 ID,非空时**优先于**下拉菜单(官方出了新模型时用这里填) |
| api_key | 智谱 API Key;留空读取环境变量 `ZHIPU_API_KEY` / `ZHIPUAI_API_KEY` |
| system_prompt | 系统提示词,设定角色与输出格式(可留空) |
| send_batch | 仅图像分析节点:开启后把整批图像作为多图一起发送(上限 16 张),关闭只取第一帧 |
| temperature | 采样温度,越高越发散,一般 0.6~1.0 |
| max_tokens | 最大输出 token 数 |
| timeout | API 超时(秒),长文分析可调大 |

### 可选模型(2026-09)

视觉模型(图像分析节点):

| 模型 ID | 说明 |
| --- | --- |
| `glm-5.3-flash` | 原生多模态,综合能力最强(默认,推荐) |
| `glm-4.6v` | 付费,前端代码复刻见长 |
| `glm-4.6v-flash` | **免费**,支持视觉推理 |
| `glm-4.1v-thinking-flash` | **免费**,视觉推理,64K 上下文 |
| `glm-4v-flash` | **免费**,轻量图像理解,输出上限 1K |

文本模型(文字对话节点):

| 模型 ID | 说明 |
| --- | --- |
| `glm-5.3` | 旗舰 |
| `glm-5.3-flash` | 高性价比(默认) |
| `glm-4.7` / `glm-4.6` | 付费 |
| `glm-4.7-flash` / `glm-4.5-flash` / `glm-4-flash-250414` | **免费** |

模型 ID 可能随官方更新变动,最新列表见 https://docs.bigmodel.cn/cn/guide/start/model-overview ;新模型直接填到 `custom_model` 即可使用,无需改代码。

## 安装

### 本地 ComfyUI

把本目录放入 `ComfyUI/custom_nodes/` 后重启:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Tony7012/zhipu_-api.git ComfyUI-Zhipu-API
# 或直接把 ComfyUI-Zhipu-API 文件夹拷进来
```

依赖只有 `requests` / `Pillow`(ComfyUI 自带),无需额外安装。

### RunningHub

方式一(自定义节点):
1. 把 `ComfyUI-Zhipu-API` 整个文件夹打包为 zip(压缩包内保留文件夹这一层)。
2. 在 RunningHub「云空间 → 自定义节点 / 节点管理」中上传该 zip 并安装。

方式二(GitHub):把代码推到 GitHub 仓库,在 RunningHub 节点管理中通过仓库地址安装。

## 常见问题

- **提示未找到 API Key**:检查节点 `api_key` 是否填写(注意前后空格),或环境变量 `ZHIPU_API_KEY` 是否在 ComfyUI 进程启动前设置。
- **HTTP 401 / 1002**:API Key 错误或已失效,到 https://bigmodel.cn/usercenter/apikeys 重新生成。
- **HTTP 429**:触发限流,节点会自动重试 3 次;仍失败请稍后再试或换用 flash 系列模型。
- **想零成本测试**:选 `glm-4.6v-flash` / `glm-4.7-flash` 等免费模型。
- **图像较大上传慢**:节点会自动把最长边超过 4096px 的图等比缩小并以 JPEG 压缩后再发送。

## 安全提示

`api_key` 会明文保存在工作流 JSON 中。不要把含 Key 的工作流分享给他人或公开发布;分享前先清空该字段。 RunningHub 用户建议仅在自用工作流中填写。
