from abc import ABC, abstractmethod
from typing import Any


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