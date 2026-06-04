"""模型工厂 —— 通过 ``MODEL_PROVIDER`` 环境变量切换 LLM Provider。

支持:
- anthropic  → Claude 3.5 Sonnet (默认, 代码能力强)
- openai     → GPT-4o-mini (低成本)
- deepseek   → DeepSeek-Coder (走 OpenAI 兼容协议)
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

# 让 .env 始终优先于 shell 环境变量 —— 避免老 shell 里残留的空/过期 key
# 屏蔽掉 .env 实际填的新值, 是学习场景下最隐蔽的坑之一。
load_dotenv(override=True)


@lru_cache(maxsize=4)
def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    provider = os.getenv("MODEL_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model_name="claude-3-5-sonnet-latest",
            temperature=temperature,
            timeout=60,
            stop=None,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    if provider == "deepseek":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        # 注: DeepSeek 已将 deepseek-coder 并入 deepseek-chat,
        # 真要更强代码推理可换 "deepseek-reasoner"。
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown MODEL_PROVIDER: {provider!r}. "
        "Expected one of: anthropic, openai, deepseek."
    )
