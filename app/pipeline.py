"""Orchestration: validate -> normalize -> chunk -> transcribe -> merge -> post-process."""
import logging
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import audio
from .config import settings
from .engines import STTEngine, get_engine

log = logging.getLogger("pipeline")


@dataclass
class Segment:
    id: int
    start: float  # absolute seconds in the original file
    end: float
    text: str
    confidence: float | None = None


@dataclass
class Transcript:
    language: str | None
    duration: float
    engine: str
    source: dict
    segments: list[Segment]
    text: str = ""
    word_count: int = 0
    processing_s: float = 0.0
    chunks: int = 1
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- post-processing for downstream consumers ----------
_WS = re.compile(r"\s+")


def clean_text(t: str) -> str:
    return _WS.sub(" ", t).strip()


def merge_chunk_segments(per_chunk: list[tuple[audio.Chunk, list]]) -> list[Segment]:
    """Shift chunk-relative times to absolute, and drop duplicates from overlapped hard cuts."""
    merged: list[Segment] = []
    covered_until = 0.0
    for chunk, raws in per_chunk:
        for r in raws:
            start, end = round(chunk.start + r.start, 2), round(chunk.start + r.end, 2)
            if merged and start < covered_until - 0.25:  # already transcribed in previous chunk's tail
                continue
            text = clean_text(r.text)
            if not text:
                continue
            merged.append(Segment(len(merged), start, end, text, r.confidence))
            covered_until = max(covered_until, end)
    return merged


def to_srt(t: Transcript) -> str:
    def ts(s: float) -> str:
        ms = int(round(s * 1000))
        return f"{ms // 3_600_000:02}:{ms // 60_000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
    return "\n".join(f"{s.id + 1}\n{ts(s.start)} --> {ts(s.end)}\n{s.text}\n" for s in t.segments)


# ---------- main entry ----------
def transcribe_file(path: Path, language: str | None = None, engine: STTEngine | None = None) -> Transcript:
    t0 = time.perf_counter()
    engine = engine or get_engine()
    info = audio.probe(path)  # raises AudioError on bad input
    if info.duration > settings.max_duration_s:
        raise audio.AudioError(f"Audio is {info.duration:.0f}s; limit is {settings.max_duration_s:.0f}s")

    with tempfile.TemporaryDirectory(prefix="stt_") as tmp:  # always cleaned up, even on failure
        tmpdir = Path(tmp)
        wav = audio.normalize(path, tmpdir / "normalized.wav")
        chunks = audio.split(wav, info.duration, tmpdir, settings.chunk_target_s,
                             settings.chunk_search_s, settings.hard_cut_overlap_s)
        log.info("file=%s duration=%.1fs codec=%s chunks=%d", path.name, info.duration, info.codec, len(chunks))

        # Lock language after first chunk so chunks don't flip-flop on auto-detection.
        first = engine.transcribe(chunks[0].path, language)
        lang = language or first.language
        results = [first]
        if len(chunks) > 1:
            with ThreadPoolExecutor(max_workers=settings.workers) as pool:
                results += list(pool.map(lambda c: engine.transcribe(c.path, lang), chunks[1:]))

    segments = merge_chunk_segments([(c, r.segments) for c, r in zip(chunks, results)])
    warnings = []
    if not segments:
        warnings.append("No speech detected")
    low = [s.id for s in segments if s.confidence is not None and s.confidence < 0.4]
    if low:
        warnings.append(f"{len(low)} low-confidence segments: ids {low[:20]}")

    text = " ".join(s.text for s in segments)
    return Transcript(
        language=lang, duration=round(info.duration, 2), engine=engine.name,
        source={"codec": info.codec, "container": info.container,
                "sample_rate": info.sample_rate, "channels": info.channels},
        segments=segments, text=text, word_count=len(text.split()),
        processing_s=round(time.perf_counter() - t0, 2), chunks=len(chunks), warnings=warnings,
    )
