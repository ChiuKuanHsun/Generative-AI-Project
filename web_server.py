"""
FastAPI web server for the Financial Short Video Generator.

Local:  uvicorn web_server:app --reload --port 8000
GCP:    uvicorn web_server:app --host 0.0.0.0 --port ${PORT:-8080}
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import threading
import traceback
import uuid
import moviepy
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv("api-key.env")

print(f"Python: {sys.executable}")

# ── Config (all overridable via env for GCP) ──────────────────────────────────

VOICE_CONFIG_PATH     = os.getenv("VOICE_CONFIG_PATH", "voice_config.json")
CHARACTER_CONFIG_PATH = os.getenv("CHARACTER_CONFIG_PATH", "character_config.json")
CHARACTER_LIBRARY_DIR = "characters"
PORT = int(os.getenv("PORT", "8080"))

# ── Per-thread stdout → SSE bridge ────────────────────────────────────────────

_thread_local = threading.local()
_real_stdout = sys.stdout


class _TeeWriter:
    """Forwards print() output to terminal AND the active run's SSE queue."""

    def write(self, text: str) -> int:
        try:
            _real_stdout.write(text)
        except Exception:
            pass
        info = getattr(_thread_local, "run_info", None)
        if info is not None:
            queue, loop, buf = info
            buf[0] += text
            while "\n" in buf[0]:
                line, buf[0] = buf[0].split("\n", 1)
                stripped = line.strip()
                if stripped:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            queue.put({"type": "log", "text": stripped}), loop
                        )
                    except Exception:
                        pass
        return len(text)

    def flush(self):
        try:
            _real_stdout.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False


sys.stdout = _TeeWriter()

# ── In-memory run registry ────────────────────────────────────────────────────

_runs: dict[str, dict] = {}


# ── Pydantic models ───────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    topic: str
    test: bool = False


class VoiceConfig(BaseModel):
    gugugaga: str
    meowchan: str


class YoutubeUploadRequest(BaseModel):
    privacy: str = "public"


class CharSelectRequest(BaseModel):
    filename: str


# ── Pipeline helpers ──────────────────────────────────────────────────────────


def _emit_sync(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, msg: dict | None):
    try:
        asyncio.run_coroutine_threadsafe(queue.put(msg), loop)
    except Exception:
        pass


def _interleave_visuals(images: list[str], charts: list[str]) -> list[str]:
    out: list[str] = []
    i = c = 0
    while i < len(images) or c < len(charts):
        if i < len(images):
            out.append(images[i]); i += 1
        if c < len(charts):
            out.append(charts[c]); c += 1
    return out


def _file_url(run_id: str, path: str) -> str:
    return f"/api/files/{run_id}/{Path(path).name}"


# ── Character config helpers ──────────────────────────────────────────────────

def _load_char_config() -> dict:
    try:
        if os.path.exists(CHARACTER_CONFIG_PATH):
            with open(CHARACTER_CONFIG_PATH, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_char_config(cfg: dict):
    with open(CHARACTER_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def _apply_char_config(cfg: dict):
    from video_composer import CHARACTER_IMAGES, _char_cache
    for role, path in cfg.items():
        if role in CHARACTER_IMAGES:
            CHARACTER_IMAGES[role] = path
            _char_cache.pop(role, None)


# ── Test-mode helpers ─────────────────────────────────────────────────────────


def _find_latest_output_dir() -> Path | None:
    """Return the most recent output_* dir that has dialogue.json + audio/."""
    candidates = sorted(
        [p for p in Path(".").glob("output_*")
         if p.is_dir() and (p / "dialogue.json").exists() and (p / "audio").is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ── Pipeline thread ───────────────────────────────────────────────────────────


def _run_pipeline(run_id: str, topic: str, loop: asyncio.AbstractEventLoop, test_mode: bool = False):
    """Blocking pipeline — executed in a thread-pool worker."""
    run = _runs[run_id]
    queue: asyncio.Queue = run["queue"]
    buf = [""]
    _thread_local.run_info = (queue, loop, buf)

    def emit(msg: dict):
        _emit_sync(queue, loop, msg)

    def step_start(n: int, label: str):
        run["steps"][n - 1] = "running"
        emit({"type": "step_start", "step": n, "label": label})

    def step_done(n: int, **extra):
        run["steps"][n - 1] = "done"
        emit({"type": "step_done", "step": n, **extra})

    try:
        from video_composer import compose_video
        out_dir = run["dir"]

        if test_mode:
            # ── TEST MODE: load assets from latest output_* dir ───────────────
            from tts_generator import get_mp3_duration

            src_dir = _find_latest_output_dir()
            if src_dir is None:
                raise RuntimeError("No previous output_* directory with cached assets found.")
            print(f"Test mode: using assets from {src_dir.name}")

            # Step 1 ── cached dialogue
            step_start(1, f"Loading cached dialogue ({src_dir.name})")
            with open(src_dir / "dialogue.json", encoding="utf-8") as f:
                dialogue_data = json.load(f)
            shutil.copy(str(src_dir / "dialogue.json"), os.path.join(out_dir, "dialogue.json"))
            step_done(1, dialogue=dialogue_data)

            # Step 2 ── cached audio (paths point to source dir, not copied)
            step_start(2, "Loading cached audio")
            audio_data = []
            audio_src = src_dir / "audio"
            for i, item in enumerate(dialogue_data["dialogue"]):
                for ext in (".mp3", ".wav"):
                    p = audio_src / f"line_{i:03d}{ext}"
                    if p.exists():
                        audio_data.append({
                            "role":       item["role"],
                            "line":       item["line"],
                            "emotion":    item.get("emotion", ""),
                            "audio_path": str(p),
                            "duration":   get_mp3_duration(str(p)),
                        })
                        break
            run["audio_data"] = audio_data
            step_done(2)

            # Step 3 ── cached charts (copy to new run dir for file serving)
            step_start(3, "Loading cached charts")
            chart_paths = []
            for p in sorted(src_dir.glob("chart_*.png")):
                dst = os.path.join(out_dir, p.name)
                shutil.copy(str(p), dst)
                chart_paths.append(dst)
            run["chart_paths"] = chart_paths
            step_done(3, charts=[_file_url(run_id, p) for p in chart_paths])

            # Step 4 ── cached images (copy to new run dir for file serving)
            step_start(4, "Loading cached images")
            image_paths = []
            for p in sorted(src_dir.glob("news_image_*.png")):
                dst = os.path.join(out_dir, p.name)
                shutil.copy(str(p), dst)
                image_paths.append(dst)
            step_done(4, images=[_file_url(run_id, p) for p in image_paths])

        else:
            # ── NORMAL MODE ───────────────────────────────────────────────────
            from news_fetcher import fetch_news_and_generate_dialogue
            from tts_generator import generate_audio_files
            from chart_generator import generate_chart_set
            from image_agent import generate_news_images

            # Step 1 ── news + dialogue
            step_start(1, "Fetching news & generating dialogue")
            dialogue_data = fetch_news_and_generate_dialogue(topic)
            dialogue_path = os.path.join(out_dir, "dialogue.json")
            with open(dialogue_path, "w", encoding="utf-8") as f:
                json.dump(dialogue_data, f, ensure_ascii=False, indent=2)
            step_done(1, dialogue=dialogue_data)

            # Step 2 ── TTS
            step_start(2, "Generating voices (TTS @ 1.3×)")
            audio_dir = os.path.join(out_dir, "audio")
            audio_data = generate_audio_files(dialogue_data["dialogue"], output_dir=audio_dir)
            run["audio_data"] = audio_data
            step_done(2)

            # Step 3 ── charts
            step_start(3, "Generating charts")
            try:
                chart_paths = generate_chart_set(topic, output_dir=out_dir)
            except Exception as e:
                print(f"Chart generation failed: {e}")
                chart_paths = []
            run["chart_paths"] = chart_paths
            step_done(3, charts=[_file_url(run_id, p) for p in chart_paths])

            # Step 4 ── images
            image_count = max(0, len(audio_data) - len(chart_paths))
            step_start(4, f"Searching news images ({image_count} needed)")
            if image_count > 0:
                try:
                    image_paths = generate_news_images(
                        dialogue_data.get("topic", topic),
                        output_dir=out_dir,
                        count=image_count,
                    )
                except Exception as e:
                    print(f"Image search failed: {e}")
                    image_paths = []
            else:
                image_paths = []
            step_done(4, images=[_file_url(run_id, p) for p in image_paths])

        # ── STEP 5: video composition (shared by both modes) ─────────────────
        step_start(5, "Composing video (MoviePy)")
        visuals = _interleave_visuals(image_paths, chart_paths)
        video_path = os.path.join(out_dir, "output.mp4")
        compose_video(
            audio_data=audio_data,
            topic=dialogue_data.get("topic", topic),
            chart_path=chart_paths[0] if chart_paths else None,
            image_paths=visuals,
            output_path=video_path,
        )
        video_url = _file_url(run_id, video_path)
        run["video_url"] = video_url
        step_done(5, video=video_url)

        run["status"] = "done"
        emit({"type": "done"})

    except Exception as exc:
        run["status"] = "error"
        run["error"] = str(exc)
        emit({"type": "error", "error": str(exc), "detail": traceback.format_exc()})

    finally:
        _thread_local.run_info = None
        _emit_sync(queue, loop, None)  # sentinel → SSE generator exits


# ── App ───────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path("static").mkdir(exist_ok=True)
    if os.path.exists(VOICE_CONFIG_PATH):
        try:
            with open(VOICE_CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
            from tts_generator import set_vits_voices
            set_vits_voices(cfg)
        except Exception:
            pass
    char_cfg = _load_char_config()
    if char_cfg:
        try:
            _apply_char_config(char_cfg)
        except Exception:
            pass
    yield


app = FastAPI(title="FinVid Generator", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/index.html")


@app.get("/api/topics")
async def get_topics():
    from news_fetcher import TOPICS
    return {"topics": [{"key": k, "label": v} for k, v in TOPICS.items()]}


@app.post("/api/run")
async def start_run(req: RunRequest):
    active = [r for r in _runs.values() if r["status"] == "running"]
    if active:
        raise HTTPException(409, "A pipeline run is already in progress")

    topic = req.topic.strip()
    if not req.test and not topic:
        raise HTTPException(400, "Topic cannot be empty")

    run_id = str(uuid.uuid4())[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"output_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)

    run: dict = {
        "id": run_id,
        "topic": topic,
        "status": "running",
        "steps": [None, None, None, None, None],
        "queue": asyncio.Queue(),
        "dir": str(run_dir),
        "audio_data": [],
        "chart_paths": [],
        "video_url": None,
        "error": None,
    }
    _runs[run_id] = run

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, lambda: _run_pipeline(run_id, topic, loop, req.test))
    return {"run_id": run_id}


@app.get("/api/run/{run_id}/stream")
async def stream_run(run_id: str):
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    queue: asyncio.Queue = _runs[run_id]["queue"]

    async def generate():
        while True:
            msg = await queue.get()
            if msg is None:
                return
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/run/{run_id}/status")
async def get_run_status(run_id: str):
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    run = _runs[run_id]
    return {
        "status": run["status"],
        "steps": run["steps"],
        "video_url": run.get("video_url"),
        "error": run.get("error"),
    }


@app.get("/api/has_test_assets")
async def has_test_assets():
    src = _find_latest_output_dir()
    if src is None:
        return {"available": False}
    return {"available": True, "dir": src.name}


@app.get("/api/files/{run_id}/{filename:path}")
async def get_file(run_id: str, filename: str):
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    run_dir = Path(_runs[run_id]["dir"])
    file_path = (run_dir / filename).resolve()
    base = run_dir.resolve()
    if not str(file_path).startswith(str(base)):
        raise HTTPException(403, "Forbidden")
    if not file_path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(str(file_path))


@app.post("/api/run/{run_id}/upload_youtube")
async def upload_to_youtube(run_id: str, req: YoutubeUploadRequest):
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    run = _runs[run_id]
    if run["status"] != "done":
        raise HTTPException(400, "Run is not complete yet")
    if req.privacy not in ("public", "unlisted", "private"):
        raise HTTPException(400, "privacy must be public, unlisted, or private")

    video_path = os.path.join(run["dir"], "output.mp4")
    if not os.path.isfile(video_path):
        raise HTTPException(404, "Video file not found")

    dialogue_path = os.path.join(run["dir"], "dialogue.json")
    dialogue_data: dict = {}
    if os.path.exists(dialogue_path):
        with open(dialogue_path, encoding="utf-8") as f:
            dialogue_data = json.load(f)

    topic   = dialogue_data.get("topic", run["topic"])
    summary = dialogue_data.get("summary", "")
    title   = f"{topic} #Shorts"
    description = (
        f"{summary}\n\n"
        "Auto-generated financial news short. Data for reference only — invest responsibly.\n\n"
        "#Shorts #Finance #News #Crypto #Stocks #AI"
    )
    tags = ["finance", "news", "crypto", "stocks", "AI", "shorts", "market"]

    loop = asyncio.get_running_loop()
    try:
        from youtube_uploader import upload_video
        url = await loop.run_in_executor(
            None,
            lambda: upload_video(video_path, title, description, tags=tags, privacy=req.privacy),
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))

    return {"url": url}


@app.post("/api/characters/upload")
async def upload_to_library(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    Path(CHARACTER_LIBRARY_DIR).mkdir(exist_ok=True)

    import io
    from PIL import Image as PILImage
    content = await file.read()
    img = PILImage.open(io.BytesIO(content)).convert("RGBA")
    uid = str(uuid.uuid4())[:8]
    filename = f"upload_{uid}.png"
    img.save(str(Path(CHARACTER_LIBRARY_DIR) / filename), "PNG")
    return {"filename": filename}


@app.get("/api/characters/library")
async def get_library():
    from video_composer import CHARACTER_IMAGES
    lib_dir = Path(CHARACTER_LIBRARY_DIR)
    images = []
    if lib_dir.is_dir():
        for p in sorted(lib_dir.glob("*.png"), key=lambda x: x.stat().st_mtime):
            images.append({"filename": p.name, "url": f"/api/characters/images/{p.name}"})
    return {"images": images, "current": CHARACTER_IMAGES.copy()}


@app.get("/api/characters/images/{filename}")
async def get_library_image(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(403, "Forbidden")
    path = Path(CHARACTER_LIBRARY_DIR) / filename
    if not path.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(str(path))


@app.post("/api/characters/{role}/select")
async def select_character(role: str, req: CharSelectRequest):
    from video_composer import CHARACTER_IMAGES
    if role not in CHARACTER_IMAGES:
        raise HTTPException(400, f"Invalid role '{role}'")
    if "/" in req.filename or "\\" in req.filename or ".." in req.filename:
        raise HTTPException(403, "Forbidden")
    path = Path(CHARACTER_LIBRARY_DIR) / req.filename
    if not path.is_file():
        raise HTTPException(404, "Image not found")
    cfg = _load_char_config()
    for r in CHARACTER_IMAGES:
        if r not in cfg:
            cfg[r] = CHARACTER_IMAGES[r]
    cfg[role] = str(path)
    _save_char_config(cfg)
    _apply_char_config(cfg)
    return {"ok": True}


@app.get("/api/characters/{role}/preview")
async def get_character_preview(role: str):
    from video_composer import CHARACTER_IMAGES
    if role not in CHARACTER_IMAGES:
        raise HTTPException(400, f"Invalid role '{role}'")
    path = Path(CHARACTER_IMAGES[role])
    if not path.is_file():
        raise HTTPException(404, "Character image not found")
    return FileResponse(str(path))


@app.get("/api/voices")
async def get_voices():
    from tts_generator import get_vits_speakers
    speakers = get_vits_speakers()
    current: dict = {}
    if os.path.exists(VOICE_CONFIG_PATH):
        try:
            with open(VOICE_CONFIG_PATH, encoding="utf-8") as f:
                current = json.load(f)
        except Exception:
            pass
    return {"speakers": speakers, "current": current}


@app.post("/api/voices")
async def save_voices(config: VoiceConfig):
    cfg = {"gugugaga": config.gugugaga, "meowchan": config.meowchan}
    with open(VOICE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    from tts_generator import set_vits_voices
    set_vits_voices(cfg)
    return {"ok": True, "config": cfg}


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_server:app", host="0.0.0.0", port=PORT, reload=True)
