"""pytest 配置 —— 加载 .env, 让需要 API key 的测试能跑。"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv(override=True)


def _has_key_for_current_provider() -> bool:
    provider = os.getenv("MODEL_PROVIDER", "anthropic").lower()
    keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    key_name = keys.get(provider)
    return bool(key_name and os.getenv(key_name))


@pytest.fixture(scope="session")
def has_llm_credentials() -> bool:
    return _has_key_for_current_provider()


def pytest_collection_modifyitems(config, items):
    """没有 API key 的环境下自动跳过 e2e 测试。"""
    if _has_key_for_current_provider():
        return
    skip_e2e = pytest.mark.skip(
        reason="No API key for current MODEL_PROVIDER; skipping live-LLM e2e test."
    )
    for item in items:
        if "e2e" in item.keywords or item.name.startswith("test_linear_flow"):
            item.add_marker(skip_e2e)
