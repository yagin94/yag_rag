from packages.core.config import settings
from apps.llm.ollama_client import OllamaClient

LLM_PROVIDERS = {
    "ollama": OllamaClient,
}

_llm_client_instance = None


def get_llm_client():
    global _llm_client_instance

    if _llm_client_instance is not None:
        return _llm_client_instance

    provider = settings.llm_provider

    if provider not in LLM_PROVIDERS:
        raise ValueError(f"Unsupported LLM_PROVIDER={provider}")

    client_cls = LLM_PROVIDERS[provider]

    _llm_client_instance = client_cls(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=settings.llm_timeout_sec,
    )

    return _llm_client_instance