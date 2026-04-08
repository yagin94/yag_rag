from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class LLMClient(ABC):
    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict:
        """
        Expected return format:
        {
            "text": str,
            "model": str,
            "usage": dict | None,
            "finish_reason": str | None,
        }
        """
        raise NotImplementedError

    @abstractmethod
    async def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> AsyncIterator[dict]:
        """
        Expected yielded chunk format:
        {
            "text": str,
            "model": str,
            "done": bool,
            "finish_reason": str | None,
            "usage": dict | None,
        }
        """
        raise NotImplementedError