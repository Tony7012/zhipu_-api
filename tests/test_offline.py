# -*- coding: utf-8 -*-
"""离线单元测试:mock 掉网络请求,不消耗 API 额度。

运行: python tests/test_offline.py
"""

import base64
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zhipu_client as zc  # noqa: E402
import nodes  # noqa: E402


class FakeTensor:
    """模拟 torch.Tensor:提供 detach().cpu().numpy() 链路。"""

    def __init__(self, arr):
        self.arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


def make_image(batch=1, h=32, w=32, gray=False):
    shape = (batch, h, w) if gray else (batch, h, w, 3)
    return FakeTensor(np.random.rand(*shape).astype(np.float32))


def ok_response(content="你好,分析结果在此"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": content}}]
    }
    return resp


class TestApiKey(unittest.TestCase):
    def test_missing_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                zc.resolve_api_key("  ")

    def test_node_input_wins(self):
        with patch.dict(os.environ, {"ZHIPU_API_KEY": "env-key"}):
            self.assertEqual(zc.resolve_api_key(" node-key "), "node-key")
            self.assertEqual(zc.resolve_api_key(""), "env-key")


class TestImageConversion(unittest.TestCase):
    def test_single_frame_from_batch(self):
        urls = zc.tensor_to_data_urls(make_image(batch=3))
        self.assertEqual(len(urls), 1)

    def test_send_batch_all_frames(self):
        urls = zc.tensor_to_data_urls(make_image(batch=3), send_batch=True)
        self.assertEqual(len(urls), 3)

    def test_jpeg_payload_valid(self):
        urls = zc.tensor_to_data_urls(make_image())
        self.assertTrue(urls[0].startswith("data:image/jpeg;base64,"))
        raw = base64.b64decode(urls[0].split(",", 1)[1])
        self.assertEqual(raw[:2], b"\xff\xd8")  # JPEG 魔数

    def test_downscale(self):
        urls = zc.tensor_to_data_urls(make_image(h=5000, w=200), max_side=1024)
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(base64.b64decode(urls[0].split(",", 1)[1])))
        self.assertLessEqual(max(img.size), 1024)

    def test_bad_dims_raise(self):
        with self.assertRaises(ValueError):
            zc.tensor_to_data_urls(FakeTensor(np.zeros((5, 5))))


class TestChat(unittest.TestCase):
    def _args(self, **kw):
        base = dict(api_key="k", model="glm-5.3-flash",
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=0.7, max_tokens=100, timeout=10, max_retries=2)
        base.update(kw)
        return base

    @patch("zhipu_client.requests.post")
    def test_success_payload(self, post):
        post.return_value = ok_response()
        text = zc.chat(**self._args())
        self.assertEqual(text, "你好,分析结果在此")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "glm-5.3-flash")
        self.assertEqual(payload["messages"][0]["content"], "hi")
        self.assertEqual(payload["stream"], False)
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer k")

    @patch("zhipu_client.requests.post")
    def test_retry_on_500_then_ok(self, post):
        err = MagicMock()
        err.status_code = 502
        err.json.return_value = {}
        err.text = "bad gateway"
        post.side_effect = [err, ok_response("第二次成功")]
        text = zc.chat(**self._args())
        self.assertEqual(text, "第二次成功")
        self.assertEqual(post.call_count, 2)

    @patch("zhipu_client.requests.post")
    def test_no_retry_on_401(self, post):
        err = MagicMock()
        err.status_code = 401
        err.json.return_value = {"error": {"code": "1002", "message": "令牌无效"}}
        err.text = "{}"
        post.return_value = err
        with self.assertRaises(RuntimeError) as ctx:
            zc.chat(**self._args())
        self.assertIn("令牌无效", str(ctx.exception))
        self.assertEqual(post.call_count, 1)

    @patch("zhipu_client.requests.post")
    def test_think_tags_stripped(self, post):
        post.return_value = ok_response("<think>推理过程</think>最终答案")
        text = zc.chat(**self._args())
        self.assertEqual(text, "最终答案")

    @patch("zhipu_client.requests.post")
    def test_empty_content_raises(self, post):
        post.return_value = ok_response("")
        with self.assertRaises(RuntimeError):
            zc.chat(**self._args())


class TestNodes(unittest.TestCase):
    @patch("nodes.chat")
    def test_image_analysis_node(self, chat_mock):
        chat_mock.return_value = "这是一张风景照"
        node = nodes.ZhipuImageAnalysis()
        out = node.analyze(
            image=make_image(batch=2),
            prompt="描述图片",
            model="glm-5.3-flash",
            custom_model="",
            api_key="k",
            system_prompt="",
            send_batch=False,
            temperature=0.7,
            max_tokens=4096,
            timeout=60,
        )
        self.assertEqual(out["result"], ("这是一张风景照",))
        self.assertEqual(out["ui"]["text"], ["这是一张风景照"])
        # 只发了一帧图像
        sent = chat_mock.call_args.kwargs["messages"][-1]["content"]
        self.assertEqual(len([b for b in sent if b["type"] == "image_url"]), 1)

    @patch("nodes.chat")
    def test_image_analysis_custom_model_and_system(self, chat_mock):
        chat_mock.return_value = "ok"
        node = nodes.ZhipuImageAnalysis()
        node.analyze(
            image=make_image(),
            prompt="描述图片",
            model="glm-4.6v",
            custom_model="glm-未来新型号",
            api_key="k",
            system_prompt="你是打标器",
            send_batch=False,
            temperature=0.7,
            max_tokens=4096,
            timeout=60,
        )
        kwargs = chat_mock.call_args.kwargs
        self.assertEqual(kwargs["model"], "glm-未来新型号")
        self.assertEqual(kwargs["messages"][0]["role"], "system")

    @patch("nodes.chat")
    def test_text_chat_node(self, chat_mock):
        chat_mock.return_value = "扩写结果"
        node = nodes.ZhipuTextChat()
        out = node.chat_run(
            prompt="扩写:一只猫",
            model="glm-5.3-flash",
            custom_model="",
            api_key="k",
            system_prompt="你是提示词专家",
            temperature=0.7,
            max_tokens=4096,
            timeout=60,
        )
        self.assertEqual(out["result"], ("扩写结果",))
        kwargs = chat_mock.call_args.kwargs
        self.assertEqual(kwargs["messages"][-1]["content"], "扩写:一只猫")

    def test_empty_prompt_raises(self):
        with self.assertRaises(ValueError):
            nodes.ZhipuTextChat().chat_run(
                prompt="  ", model="glm-5.3-flash", custom_model="", api_key="k",
                system_prompt="", temperature=0.7, max_tokens=4096, timeout=60,
            )

    def test_mappings(self):
        self.assertIn("ZhipuImageAnalysis", nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("ZhipuTextChat", nodes.NODE_CLASS_MAPPINGS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
