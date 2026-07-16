from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "production"
    log_level: str = "INFO"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None

    postgres_dsn: str = "postgresql+asyncpg://medcode:medcode@localhost:5432/medcode_rag"

    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    embedding_input_type: str = "passage"

    llm_model: str = "z-ai/glm-5.2"
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 3

    api_keys: str = ""
    rate_limit_per_minute: int = 60

    top_k_codes: int = 8
    top_k_similar_cases: int = 3
    top_k_error_patterns: int = 5
    low_confidence_threshold: float = 0.5

    target_rules_path: str = "data/target_rules.json"

    @property
    def valid_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


settings = Settings()
