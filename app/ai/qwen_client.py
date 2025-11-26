# file: qwen_client.py
import os
from typing import Iterable, Optional, Dict, Any, List, Literal

from openai import OpenAI, AsyncOpenAI
from api_key import DASHSCOPE_API_KEY

Role = Literal["system", "user", "assistant"]

class QwenClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "qwen-plus",
        default_params: Optional[Dict[str, Any]] = None,
    ):
        """
        :param api_key: 默认从配置文件 DASHSCOPE_API_KEY 读取
        :param base_url: 默认北京地域的 OpenAI 兼容地址
        :param model: 默认用 qwen-plus，可以按需换 qwen-max / qwen-turbo 等
        :param default_params: 默认传给 completions.create 的参数（temperature 等）
        """
        self.api_key = DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 未设置，请先配置环境变量。")

        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model = model
        self.default_params = default_params or {}

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._aclient = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    # ---------- 同步非流式 ----------
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        :param messages: [{"role": "user", "content": "..."}, ...]
        :return: 模型返回的文本（只取第一个 choice）
        """
        params = {**self.default_params, **kwargs}
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            **params,
        )
        return resp.choices[0].message.content

    # ---------- 同步流式 ----------
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        include_usage: bool = False,
        **kwargs,
    ) -> Iterable[str]:
        """
        生成器：逐块（chunk）返回文本片段
        """
        params = {**self.default_params, **kwargs}
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": include_usage},
            **params,
        )
        for chunk in completion:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    # ---------- 异步 ----------
    async def achat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        params = {**self.default_params, **kwargs}
        resp = await self._aclient.chat.completions.create(
            model=self.model,
            messages=messages,
            **params,
        )
        return resp.choices[0].message.content
