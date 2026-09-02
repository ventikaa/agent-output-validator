from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

load_dotenv()


@lru_cache
def get_claude() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=4096,
    )


@lru_cache
def get_gpt4o() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=4096,
    )


def get_llm(provider: str = "anthropic") -> ChatAnthropic | ChatOpenAI:
    if provider == "openai":
        return get_gpt4o()
    return get_claude()
