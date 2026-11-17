"""Format handling and chunking. ffmpeg is the single decoding path for every input format."""
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

TARGET_SR = 16_000  # what Whisper-family models expect


class AudioError(ValueError):
    """Input is not usable audio (bad format, no audio stream, too long, corrupt)."""


@dataclass
class AudioInfo:
    duration: float
    codec: str
    sample_rate: int
    channels: int
    container: str


@dataclass
class Chunk:
    path: Path
    start: float  # offset in the original file, seconds
    end: float


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    # No shell=True: file paths never get interpreted by a shell.
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def probe(path: Path) -> AudioInfo:
    """Detect the real format from file contents, not the extension or client MIME type."""
    res = _run(["ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path)])
    if res.returncode != 0:
        raise AudioError(f"Unreadable media file: {res.stderr.strip()[:200]}")
    meta = json.loads(res.stdout)
    audio = next((s for s in meta.get("streams", []) if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise AudioError("File contains no audio stream")
    duration = float(meta.get("format", {}).get("duration") or audio.get("duration") or 0)
    if duration <= 0:
        raise AudioError("Could not determine audio duration (empty or corrupt file)")
    return AudioInfo(duration=duration, codec=audio.get("codec_name", "?"),
                     sample_rate=int(audio.get("sample_rate", 0)),
                     channels=int(audio.get("channels", 0)),
                     container=meta["format"].get("format_name", "?"))


def normalize(src: Path, dst: Path) -> Path:
    """Any input (mp3/m4a/ogg/flac/webm/video...) -> 16 kHz mono 16-bit PCM WAV.
    One canonical format downstream means the STT engine and chunker never care about codecs."""
    res = _run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src),
                "-vn", "-ac", "1", "-ar", str(TARGET_SR), "-c:a", "pcm_s16le", str(dst)])
    if res.returncode != 0:
        raise AudioError(f"Decode failed: {res.stderr.strip()[:200]}")
    return dst


def detect_silences(path: Path, noise_db: int = -35, min_len: float = 0.4) -> list[tuple[float, float]]:
    res = _run(["ffmpeg", "-nostdin", "-v", "info", "-i", str(path),
                "-af", f"silencedetect=noise={noise_db}dB:d={min_len}", "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", res.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", res.stderr)]
    return list(zip(starts, ends))


def plan_cuts(duration: float, silences: list[tuple[float, float]],
              target: float, search: float) -> list[tuple[float, bool]]:
    """Pick cut points near every `target` seconds, preferring the middle of a silence
    so we never split a word. Returns [(cut_time, is_clean_cut), ...]."""
    cuts, pos = [], 0.0
    while duration - pos > target + search:
        ideal = pos + target
        window = [((s + e) / 2) for s, e in silences if ideal - search <= (s + e) / 2 <= ideal + search]
        if window:
            cut = min(window, key=lambda m: abs(m - ideal))
            cuts.append((cut, True))
        else:
            cut = ideal
            cuts.append((cut, False))  # no silence nearby: hard cut, handled with overlap
        pos = cut
    return cuts


def split(wav: Path, info_duration: float, out_dir: Path, target: float,
          search: float, overlap: float) -> list[Chunk]:
    """Split a normalized WAV into chunks. Short files return a single chunk (no copy)."""
    if info_duration <= target + search:
        return [Chunk(wav, 0.0, info_duration)]
    cuts = plan_cuts(info_duration, detect_silences(wav), target, search)
    bounds, start = [], 0.0
    for cut, clean in cuts:
        bounds.append((start, cut))
        start = cut if clean else max(0.0, cut - overlap)  # overlap only around hard cuts
    bounds.append((start, info_duration))

    chunks = []
    for i, (s, e) in enumerate(bounds):
        p = out_dir / f"chunk_{i:04d}.wav"
        res = _run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                    "-i", str(wav), "-c", "copy", str(p)])
        if res.returncode != 0:
            raise AudioError(f"Chunking failed at {s:.1f}s: {res.stderr.strip()[:200]}")
        chunks.append(Chunk(p, s, e))
    return chunks
