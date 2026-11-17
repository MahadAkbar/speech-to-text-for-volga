"""STT engines behind one interface, so the model/vendor is swappable (Whisper, Deepgram, AWS, Azure...)."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from .config import settings


@dataclass
class RawSegment:
    start: float  # relative to the chunk
    end: float
    text: str
    confidence: float | None = None


@dataclass
class EngineResult:
    segments: list[RawSegment]
    language: str | None


class STTEngine(Protocol):
    name: str
    def transcribe(self, wav: Path, language: str | None = None) -> EngineResult: ...


class WhisperEngine:
    """faster-whisper (CTranslate2): ~4x faster than openai-whisper, int8 runs fine on CPU."""
    name = "faster-whisper"

    def __init__(self, model: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel  # lazy: tests don't need the model
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.name = f"faster-whisper:{model}"

    def transcribe(self, wav: Path, language: str | None = None) -> EngineResult:
        segments, info = self.model.transcribe(
            str(wav), language=language, beam_size=5,
            vad_filter=True,  # skip silence; cuts hallucinated text on quiet stretches
            condition_on_previous_text=False,  # limits repetition loops on long audio
        )
        out = [RawSegment(s.start, s.end, s.text.strip(),
                          confidence=round(min(1.0, 2.718281828 ** s.avg_logprob), 3))
               for s in segments]
        return EngineResult(out, info.language)


class MockEngine:
    """Deterministic engine: one fake segment per 5s of audio. Used in tests and when no model is available."""
    name = "mock"

    def transcribe(self, wav: Path, language: str | None = None) -> EngineResult:
        import wave
        with wave.open(str(wav)) as w:
            dur = w.getnframes() / w.getframerate()
        segs, t, i = [], 0.0, 0
        while t < dur:
            end = min(t + 5.0, dur)
            segs.append(RawSegment(t, end, f"  mock   segment {i}  ", 0.9))
            t, i = end, i + 1
        return EngineResult(segs, language or "en")


@lru_cache(maxsize=1)
def get_engine() -> STTEngine:
    """Load the model once per process; model load is the most expensive step."""
    if settings.engine == "mock":
        return MockEngine()
    return WhisperEngine(settings.whisper_model, settings.whisper_device, settings.whisper_compute)
