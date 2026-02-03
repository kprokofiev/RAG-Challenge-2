import os
from typing import Optional
from pydantic import validator, Field
from pydantic_settings import BaseSettings


class WorkerSettings(BaseSettings):
    """Settings for DD Kit RAG worker with validation."""

    # Required settings
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    # Storage settings (required)
    storage_endpoint_url: str = Field("", env="STORAGE_ENDPOINT_URL")
    storage_endpoint: Optional[str] = Field(None, env="STORAGE_ENDPOINT")
    storage_access_key: str = Field(..., env="STORAGE_ACCESS_KEY")
    storage_secret_key: str = Field(..., env="STORAGE_SECRET_KEY")
    storage_bucket_name: str = Field("ddkit-documents", env="STORAGE_BUCKET_NAME")
    storage_use_ssl: bool = Field(True, env="STORAGE_USE_SSL")

    # Redis settings (required)
    redis_url: str = Field(..., env="REDIS_URL")

    # Queue settings (optional with defaults)
    queue_doc_parse_index: str = Field("ddkit:doc_parse_index", env="QUEUE_DOC_PARSE_INDEX")
    queue_report_generate: str = Field("ddkit:report_generate", env="QUEUE_REPORT_GENERATE")

    # Optional settings
    log_level: str = Field("INFO", env="LOG_LEVEL")
    job_callback_url: Optional[str] = Field(None, env="JOB_CALLBACK_URL")
    job_callback_token: Optional[str] = Field(None, env="DDKIT_CALLBACK_TOKEN")

    # Job processing settings
    max_job_attempts: int = Field(3, env="MAX_JOB_ATTEMPTS")
    job_timeout_seconds: int = Field(3600, env="JOB_TIMEOUT_SECONDS")  # 1 hour default
    worker_mode: str = Field("all", env="WORKER_MODE")  # all|doc_parse_index|report_generate
    worker_concurrency: int = Field(1, env="WORKER_CONCURRENCY")
    worker_poll_timeout: int = Field(2, env="WORKER_POLL_TIMEOUT")

    # Embeddings settings
    embeddings_model: str = Field("text-embedding-3-large", env="EMBEDDINGS_MODEL")
    embeddings_batch_size: int = Field(128, env="EMBEDDINGS_BATCH_SIZE")
    embeddings_max_concurrency: int = Field(4, env="EMBEDDINGS_MAX_CONCURRENCY")
    embeddings_retry_max: int = Field(3, env="EMBEDDINGS_RETRY_MAX")
    embeddings_backoff_seconds: int = Field(5, env="EMBEDDINGS_BACKOFF_SECONDS")

    # Chunking settings
    chunk_size_tokens: int = Field(800, env="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(100, env="CHUNK_OVERLAP_TOKENS")
    chunk_dedup: bool = Field(True, env="CHUNK_DEDUP")

    # Docling tuning
    docling_do_ocr: bool = Field(False, env="DOCLING_DO_OCR")
    docling_do_tables: bool = Field(False, env="DOCLING_DO_TABLES")
    docling_do_pictures: bool = Field(True, env="DOCLING_DO_PICTURES")

    # PDF validation
    min_pdf_bytes: int = Field(5000, env="MIN_PDF_BYTES")

    @validator('storage_endpoint_url', pre=True)
    def resolve_storage_endpoint(cls, v):
        if v:
            return v
        alt = os.getenv("STORAGE_ENDPOINT")
        if alt:
            return alt
        raise ValueError("STORAGE_ENDPOINT_URL or STORAGE_ENDPOINT is required")

    @validator('storage_use_ssl', pre=True)
    def parse_storage_use_ssl(cls, v):
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return bool(v)

    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    @validator('worker_mode')
    def validate_worker_mode(cls, v):
        allowed = {"all", "doc_parse_index", "report_generate"}
        if v not in allowed:
            raise ValueError(f"WORKER_MODE must be one of {sorted(allowed)}")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = WorkerSettings()
