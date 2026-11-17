"""HTTP service. Short audio -> sync 200. Long audio -> 202 + job id, poll for result.

Run: uvicorn app.api:app --port 8000
"""
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse

from . import audio
from .config import settings
from .pipeline import to_srt, transcribe_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = FastAPI(title="Transcription Pipeline")

# In-memory job store: fine for a demo/single process. Production: Redis/DynamoDB + SQS/Service Bus workers.
JOBS: dict[str, dict] = {}
UPLOAD_DIR = Path(tempfile.gettempdir()) / "stt_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


async def _save_upload(f: UploadFile) -> Path:
    """Stream to disk with a size cap; never trust the client filename for the path."""
    limit = settings.max_upload_mb * 1024 * 1024
    dst = UPLOAD_DIR / f"{uuid.uuid4().hex}{Path(f.filename or '').suffix[:8]}"
    size = 0
    with dst.open("wb") as out:
        while chunk := await f.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close(); dst.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
            out.write(chunk)
    if size == 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, "Empty file")
    return dst


def _run_job(job_id: str, path: Path, language: str | None):
    JOBS[job_id]["status"] = "processing"
    try:
        JOBS[job_id].update(status="completed", result=transcribe_file(path, language).to_dict())
    except audio.AudioError as e:
        JOBS[job_id].update(status="failed", error=str(e))
    except Exception:  # noqa: BLE001 - surface a generic error, log the detail
        logging.exception("job %s failed", job_id)
        JOBS[job_id].update(status="failed", error="Internal transcription error")
    finally:
        path.unlink(missing_ok=True)


@app.post("/transcriptions")
async def create(background: BackgroundTasks, file: UploadFile = File(...),
                 language: str | None = Query(None, description="ISO code, e.g. 'en'. Omit to auto-detect.")):
    path = await _save_upload(file)
    try:
        info = audio.probe(path)
    except audio.AudioError as e:
        path.unlink(missing_ok=True)
        raise HTTPException(415, str(e))

    if info.duration > settings.async_threshold_s:
        job_id = uuid.uuid4().hex
        JOBS[job_id] = {"id": job_id, "status": "queued", "duration": info.duration}
        background.add_task(_run_job, job_id, path, language)
        return {"id": job_id, "status": "queued", "poll": f"/transcriptions/{job_id}"}

    try:
        return transcribe_file(path, language).to_dict()
    except audio.AudioError as e:
        raise HTTPException(422, str(e))
    finally:
        path.unlink(missing_ok=True)


@app.get("/transcriptions/{job_id}")
def get_job(job_id: str, format: str = Query("json", pattern="^(json|srt|txt)$")):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    if job["status"] != "completed" or format == "json":
        return job
    from .pipeline import Segment, Transcript
    r = job["result"]
    t = Transcript(**{**r, "segments": [Segment(**s) for s in r["segments"]]})
    return PlainTextResponse(to_srt(t) if format == "srt" else t.text)


@app.get("/health")
def health():
    return {"ok": shutil.which("ffmpeg") is not None, "engine": settings.engine}
