"""Runtime settings, all overridable via environment variables."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    engine: str = os.getenv("STT_ENGINE", "whisper")          # "whisper" | "mock"
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")   # tiny/base/small/medium/large-v3
    whisper_device: str = os.getenv("WHISPER_DEVICE", "auto")  # cpu | cuda | auto
    whisper_compute: str = os.getenv("WHISPER_COMPUTE", "int8")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "500"))
    max_duration_s: float = float(os.getenv("MAX_DURATION_S", str(4 * 3600)))
    chunk_target_s: float = float(os.getenv("CHUNK_TARGET_S", "600"))   # aim for ~10 min chunks
    chunk_search_s: float = float(os.getenv("CHUNK_SEARCH_S", "30"))    # look +/-30s for silence
    hard_cut_overlap_s: float = float(os.getenv("HARD_CUT_OVERLAP_S", "1.0"))
    async_threshold_s: float = float(os.getenv("ASYNC_THRESHOLD_S", "120"))  # >2 min -> background job
    workers: int = int(os.getenv("CHUNK_WORKERS", "1"))


settings = Settings()
