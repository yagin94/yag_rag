from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "yag-rag"
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "documents"
    vector_size: int = 768

    llm_provider: str = "ollama"
    llm_base_url: str = "http://host.docker.internal:11434"
    llm_model: str = "llama3"
    llm_timeout_sec: int = 60

    log_level: str = "INFO"
    log_json: bool = False
    log_queries: bool = True

    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_k: int = 5
    reranker_candidate_k: int = 12
    
    hybrid_enabled: bool = False
    dense_top_k: int = 5
    hybrid_candidate_k: int = 10
    hybrid_dense_weight: float = 1.0
    hybrid_lexical_weight: float = 1.0
    
    

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()