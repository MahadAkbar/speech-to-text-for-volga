# Transcription Pipeline

Audio file in, timestamped transcript out (JSON / SRT / plain text). Python, FastAPI, ffmpeg, faster-whisper.

```
upload -> validate (ffprobe) -> normalize (ffmpeg -> 16kHz mono WAV) -> chunk on silence
       -> transcribe (pluggable engine) -> merge + re-offset timestamps -> post-process -> JSON/SRT/TXT
```

## Run

```bash
pip install -r requirements.txt          # needs ffmpeg on PATH
uvicorn app.api:app --port 8000          # service
python -m app.cli talk.mp3 --format srt  # script
STT_ENGINE=mock pytest -q                # tests (no model download needed)
```

API:
- `POST /transcriptions` (multipart `file`, optional `?language=en`)
  - audio <= 2 min: `200` with transcript
  - longer: `202 {id, poll}`; then `GET /transcriptions/{id}?format=json|srt|txt`
- `GET /health`

Response (trimmed):
```json
{ "language": "en", "duration": 70.03, "engine": "faster-whisper:small",
  "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "...", "confidence": 0.91}],
  "text": "...", "word_count": 51, "chunks": 4, "warnings": [] }
```

## Answers

**1. Accept an audio file** - `app/api.py` (HTTP) and `app/cli.py` (script). Upload streams to disk in 1 MB blocks with a size cap (no full file in memory), server-generated filename (no path traversal), then `ffprobe` validates it really is audio before any work is done. Bad input: 400/413/415/422, not a 500.

**2. Transcribe** - `app/engines.py`. Default: **faster-whisper** (open-source Whisper on CTranslate2, ~4x faster than reference Whisper, int8 runs on CPU). Behind an `STTEngine` interface so swapping to Deepgram / AWS Transcribe / Azure Speech is one class. VAD filter on (reduces hallucinated text in silence), model loaded once per process.

**3. Timestamps per segment** - every segment has absolute `start`/`end` seconds, text, and a confidence score; low-confidence segments are flagged in `warnings` for human review. Post-processing: whitespace cleanup, full text, word count, SRT export.

**4. Different audio formats** - Don't support formats individually. **ffmpeg is the single decode path**: everything (mp3, wav, m4a, flac, ogg, webm, even video) is converted to one canonical format: 16 kHz mono PCM WAV, which is what Whisper expects. Format is detected from file contents via `ffprobe`, not the extension or client MIME type (tested: an MP3 renamed `.wav` is correctly identified). Files with no audio stream or corrupt data are rejected up front.

**5. Long audio files**
- **Chunking on silence**: split into ~10 min chunks, cutting at the nearest silence within +/-30s so words are never cut in half. If there's no silence (music, constant noise), hard cut with 1s overlap and drop duplicated segments at merge time.
- **Timestamp re-offset**: chunk-relative times are shifted by chunk start so the output is one continuous timeline.
- **Language lock**: detected on chunk 1, forced on the rest (prevents chunks flipping language).
- **Async jobs**: > 2 min returns `202` + job id instead of holding an HTTP connection open (avoids gateway timeouts).
- **Bounded resources**: max duration/size limits, temp files always deleted, chunks can run in parallel (`CHUNK_WORKERS`).

## Trade-offs and what I'd change for production

| Decision here | Why | Production version |
|---|---|---|
| In-memory job store + FastAPI BackgroundTasks | Zero infra for a demo | Upload to S3/Blob, queue (SQS / Service Bus), GPU workers, job state in DynamoDB/Postgres. Survives restarts, scales independently |
| faster-whisper `small` on CPU | Free, local, no data leaves the box | `large-v3` on GPU for accuracy, or a managed API if no ML ops capacity. Cost vs accuracy vs data-residency call |
| Fixed silence threshold (-35 dB) | Simple, works for speech | Adaptive threshold or use Silero VAD for cut points |
| Segment-level timestamps | What was asked | `word_timestamps=True` if downstream needs word-level alignment |
| No diarization | Out of scope | pyannote for "who said what" |

Not done, would be next: auth on the API, retries per chunk (one failed chunk shouldn't fail the job), metrics (real-time factor, error rate), WER evaluation on a small labeled set, PII redaction on transcripts before storage.

---

# Part 2: System Design

This part is design only. The code above is the single-box version; this is how I'd run it for real traffic.

```
Client ──► API Gateway (auth, rate limit) ──► API service ──► Postgres (jobs, transcripts)
   │                                              │
   └── presigned PUT ──► Object storage (audio)    └──► Queue ──► Workers (GPU, faster-whisper)
                                ▲                                   │
                                └──────── read audio / write JSON ◄─┘
                                                                    └──► Dead-letter queue + alerts
```

## How would you store audio and transcripts?

Audio and transcripts are very different kinds of data, so they go in different places.

**Audio goes to object storage (S3 or Azure Blob).** Files are large, written once and almost never read again after transcription. A database is the wrong tool for that.
- The client uploads straight to the bucket with a presigned URL. 500 MB files never pass through the API servers.
- Key format: `audio/{tenant_id}/{job_id}/original.{ext}`. The job ID ties everything together.
- Private bucket, encrypted at rest (KMS), no public access.
- Lifecycle rule: move to a cold tier after 7 days, delete after 30 (or whatever the client contract says). Raw audio is the most sensitive and most expensive thing we keep, so we shouldn't keep it longer than needed.

**Transcripts go to Postgres.** They're small, structured, and we need to query them (by status, by tenant, by date, by text).
- `jobs` table: id, tenant_id, status, attempts, last_error, audio_key, duration, language, engine, **model_version**, created_at, updated_at.
- `segments` table: job_id, idx, start, end, text, confidence. One row per segment so we can search and link to a timestamp.
- Full-text search with a Postgres `tsvector` index. No separate search engine until we actually need one.
- The full JSON result is also written to object storage next to the audio. That's the permanent record and makes re-exports (SRT, VTT) cheap.

Storing `model_version` matters. When we upgrade Whisper, we know which transcripts came from the old model and can re-run them if needed.

Privacy: transcripts can contain names, phone numbers, card numbers. There's an optional PII redaction step before saving, and `DELETE` removes audio, JSON and DB rows together.

## How do you retry or recover failed transcriptions?

First, split failures into two types, because retrying the wrong kind just wastes GPU time.

| Type | Examples | Action |
|---|---|---|
| Permanent | not audio, corrupt file, too long, unsupported language | Fail immediately, clear error to client, no retry |
| Transient | worker crashed, out of GPU memory, network blip, 429/5xx from an external API | Retry automatically |

**How retries work:**
1. **Queue does the heavy lifting.** Workers pull from SQS / Service Bus. If a worker dies mid-job, it never acknowledges the message, the visibility timeout runs out, and another worker picks it up. Nothing is lost.
2. **Exponential backoff with jitter.** Retry after ~30s, 2 min, 8 min. Jitter stops every failed job from retrying at the same moment.
3. **Max 3 attempts, then dead-letter queue.** The job is marked `failed`, an alert fires, and someone can look at it. Poison files don't loop forever.
4. **Chunk-level checkpoints.** Each chunk's result is saved as soon as it finishes, keyed by `(job_id, chunk_index)`. If chunk 7 of 12 fails, the retry only redoes chunk 7. On a 3-hour file that saves a lot.
5. **Idempotent work.** Writing the same chunk twice just overwrites it. That's why it's safe for the queue to deliver a message more than once.
6. **Stuck-job sweeper.** Workers send a heartbeat. A scheduled job finds anything in `processing` with no heartbeat for 10+ min and puts it back on the queue.
7. **Fallback.** If the GPU model keeps failing on a file, try a smaller model or a managed API (AWS Transcribe / Azure Speech) before giving up.
8. **Manual retry.** `POST /v1/transcriptions/{id}/retry` for ops or the client after fixing something.

What we track: failure rate, retries per job, DLQ depth, average time in queue. An alert on DLQ depth is the one that actually wakes someone up.

## How would you expose this as an API?

REST, **async by default**. Transcription takes minutes, and HTTP requests shouldn't stay open that long.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/v1/uploads` | Returns a presigned URL to upload large audio directly to storage |
| POST | `/v1/transcriptions` | Create job from `upload_id` or `audio_url`. Options: `language`, `word_timestamps`, `redact_pii`, `webhook_url`. Returns `202 {id, status}` |
| GET | `/v1/transcriptions/{id}` | Status + result. `?format=json\|srt\|vtt\|txt` |
| GET | `/v1/transcriptions` | List with filters (`status`, date) and cursor pagination |
| POST | `/v1/transcriptions/{id}/retry` | Re-run a failed job |
| DELETE | `/v1/transcriptions/{id}` | Delete audio + transcript (data deletion requests) |

Design choices:
- **Webhooks instead of polling.** Client passes a `webhook_url`; we POST when the job finishes, signed with HMAC so they can verify it came from us. Polling still works for simple clients.
- **`Idempotency-Key` header** on POST. If the client's network drops and they resend, they get the same job back, not a duplicate (and not a double charge).
- **Auth and limits at the gateway** (AWS API Gateway / Azure APIM): API keys or OAuth2, per-tenant rate limits and quotas. Every query is scoped by `tenant_id` so one client can never see another's data.
- **Versioned** under `/v1` so we can change the response shape later without breaking anyone.
- **Consistent errors:** `{ "error": { "code": "unsupported_audio", "message": "..." } }` with proper 4xx/5xx codes.
- **OpenAPI docs** come free from FastAPI at `/docs`.
- **Real-time** (live captions) would be a separate WebSocket endpoint with a streaming model. Different latency needs, so I wouldn't force it into the batch API.

**Implemented in this repo vs designed only:** the code has sync/async upload, job polling and JSON/SRT/TXT output. Object storage, Postgres, the queue, webhooks, auth, retries and DELETE are designed above but not built.

---

## Validation status

- 13 tests pass using a mock engine and ffmpeg-generated audio (5 formats, disguised extension, garbage input, silence cuts, hard-cut overlap dedup, sync + async API, SRT).
- **Not verified:** real faster-whisper accuracy. Model download was blocked in the build environment; run once locally with `STT_ENGINE=whisper` on a real speech sample before demoing.
