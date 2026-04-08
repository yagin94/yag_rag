from packages.core.config import settings
from apps.llm.ollama_client import OllamaClient


def get_llm_client():
    provider = settings.llm_provider

    if provider == "ollama":
        return OllamaClient(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout_sec,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER={provider}")