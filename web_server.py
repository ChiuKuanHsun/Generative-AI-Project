"""
FastAPI web server for the Financial Short Video Generator.

Local:  uvicorn web_server:app --reload --port 8000
GCP:    uvicorn web_server:app --host 0.0.0.0 --port ${PORT:-8080}
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re as _re
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
RATINGS_PATH          = os.getenv("RATINGS_PATH", "ratings.jsonl")
CHARACTER_LIBRARY_DIR = "characters"
SPEAKER_ICON_DIR      = "speaker_icons"
SPEAKER_ICON_MANIFEST = os.path.join(SPEAKER_ICON_DIR, "manifest.json")
PORT = int(os.getenv("PORT", "8080"))

# ── Speaker icon cache ────────────────────────────────────────────────────────

_icon_manifest: dict[str, str] = {}
_icon_lock = threading.Lock()

_GENSHIN_SLUG_OVERRIDES: dict[str, str] = {
    "tartalia":     "tartaglia",   # misspelling in VITS speaker list
    "fishl":        "fischl",
    "player male":  "aether",
    "player female":"lumine",
}
_GENSHIN_SKIP = {"oz"}            # Oz is Fischl's summon, not a playable character


def _load_icon_manifest() -> None:
    global _icon_manifest
    try:
        if os.path.exists(SPEAKER_ICON_MANIFEST):
            with open(SPEAKER_ICON_MANIFEST, encoding="utf-8") as f:
                _icon_manifest = json.load(f)
    except Exception:
        _icon_manifest = {}


def _save_icon_manifest() -> None:
    Path(SPEAKER_ICON_DIR).mkdir(exist_ok=True)
    with open(SPEAKER_ICON_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(_icon_manifest, f, ensure_ascii=False, indent=2)


def _speaker_cache_key(speaker: str) -> str:
    return hashlib.sha1(speaker.encode()).hexdigest()[:16]


def _parse_speaker(full_name: str) -> tuple[str, str]:
    """Return (english_name, game) from e.g. '七七 Qiqi (Genshin Impact)'."""
    m = _re.search(r'\((.+?)\)\s*$', full_name)
    game = m.group(1).strip() if m else ""
    base = full_name[:m.start()].strip() if m else full_name.strip()
    eng  = " ".join(t for t in base.split() if t and all(ord(c) < 128 for c in t))
    return eng.strip(), game


def _img_get(req_lib, url: str, hdrs: dict, timeout: int = 10) -> bytes | None:
    """Fetch a URL and return content only if it's a real image (> 2 KB)."""
    try:
        r = req_lib.get(url, headers=hdrs, allow_redirects=True, timeout=timeout)
        if (r.status_code == 200
                and r.headers.get("content-type", "").startswith("image")
                and len(r.content) > 2048):
            return r.content
    except Exception:
        pass
    return None


def _wiki_pageimages(req_lib, base: str, title: str, hdrs: dict, size: int = 300) -> bytes | None:
    """Return the lead thumbnail of a MediaWiki page via the pageimages API."""
    try:
        r = req_lib.get(f"{base}/api.php", headers=hdrs, timeout=12, params={
            "action": "query", "titles": title, "prop": "pageimages",
            "pithumbsize": size, "format": "json", "origin": "*",
        })
        if r.status_code == 200:
            for page in r.json().get("query", {}).get("pages", {}).values():
                src = page.get("thumbnail", {}).get("source", "")
                if src:
                    return _img_get(req_lib, src, hdrs)
    except Exception:
        pass
    return None


def _wiki_page_scan(req_lib, base: str, title: str, hdrs: dict) -> bytes | None:
    """Scan all images on a wiki page and return the best-looking portrait."""
    name_key = title.lower().replace(" ", "")
    try:
        r = req_lib.get(f"{base}/api.php", headers=hdrs, timeout=12, params={
            "action": "query", "titles": title, "prop": "images",
            "imlimit": "20", "format": "json", "origin": "*",
        })
        if r.status_code != 200:
            return None
        images: list = []
        for page in r.json().get("query", {}).get("pages", {}).values():
            images = page.get("images", [])

        def _score(img_info: dict) -> int:
            t = img_info["title"].lower().replace("_", "").replace(" ", "")
            if name_key in t:                                              return 0
            if any(k in t for k in ("icon", "card", "portrait", "chara", "face")): return 1
            if any(k in t for k in ("logo", "banner", "bg", "background", "item", "skill")): return 9
            return 5

        for img_info in sorted(images, key=_score)[:6]:
            try:
                r2 = req_lib.get(f"{base}/api.php", headers=hdrs, timeout=10, params={
                    "action": "query", "titles": img_info["title"],
                    "prop": "imageinfo", "iiprop": "url", "iiurlwidth": 300,
                    "format": "json", "origin": "*",
                })
                if r2.status_code == 200:
                    for p in r2.json().get("query", {}).get("pages", {}).values():
                        ii = (p.get("imageinfo") or [{}])[0]
                        url = ii.get("thumburl") or ii.get("url", "")
                        if url:
                            data = _img_get(req_lib, url, hdrs)
                            if data:
                                return data
            except Exception:
                continue
    except Exception:
        pass
    return None


def _fetch_icon_sync(english: str, game: str) -> bytes | None:
    """Blocking fetch — run via asyncio.to_thread."""
    import requests as _req
    hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    if game == "Genshin Impact":
        eng_lower = english.lower()
        if eng_lower in _GENSHIN_SKIP:
            return None
        slug = _GENSHIN_SLUG_OVERRIDES.get(
            eng_lower,
            eng_lower.replace(" ", "-").replace("'", "").replace(".", ""),
        )
        data = _img_get(_req, f"https://genshin.jmp.blue/characters/{slug}/icon", hdrs, 15)
        if data:
            return data
        # Source 2: Enka Network CDN
        enka_name = _GENSHIN_SLUG_OVERRIDES.get(eng_lower, english).replace(" ", "_")
        return _img_get(_req, f"https://enka.network/ui/UI_AvatarIcon_{enka_name}.png", hdrs, 15)

    elif game == "Umamusume Pretty Derby":
        wiki_name = english.replace(" ", "_")
        base = "https://umamusume.fandom.com"
        # ① pageimages — lead thumbnail of the character wiki page
        data = _wiki_pageimages(_req, base, english, hdrs)
        if data:
            return data
        # ② scan all images listed on the page, pick the best portrait
        data = _wiki_page_scan(_req, base, english, hdrs)
        if data:
            return data
        # ③ Special:FilePath common naming patterns
        for suffix in (f"{wiki_name}_Icon.png", f"{wiki_name}_icon.png",
                       f"{wiki_name}_Portrait.png", f"{wiki_name}_Card.png"):
            data = _img_get(_req, f"{base}/wiki/Special:FilePath/{suffix}", hdrs, 12)
            if data:
                return data

    elif game == "Sanoba Witch":
        # ① sanobawitch fandom (sparse, may fail)
        data = _wiki_pageimages(_req, "https://sanobawitch.fandom.com", english, hdrs)
        if data:
            return data
        # ② VNDB character API
        try:
            r = _req.post("https://api.vndb.org/kana/character",
                          headers={**hdrs, "Content-Type": "application/json"},
                          json={"filters": ["search", "=", english],
                                "fields": "image.url", "results": 1},
                          timeout=12)
            if r.status_code == 200:
                items = r.json().get("results", [])
                if items:
                    url = (items[0].get("image") or {}).get("url", "")
                    if url:
                        data = _img_get(_req, url, hdrs)
                        if data:
                            return data
        except Exception:
            pass

    return None

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


class SpeakerToCharRequest(BaseModel):
    speaker: str


class RatingRequest(BaseModel):
    score: int        # 1–5
    feedback: str = ""


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
    _load_icon_manifest()
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


@app.post("/api/characters/{role}/from_speaker")
async def char_from_speaker(role: str, req: SpeakerToCharRequest):
    """Copy a cached speaker portrait into the characters library and assign it."""
    from video_composer import CHARACTER_IMAGES
    if role not in CHARACTER_IMAGES:
        raise HTTPException(400, f"Invalid role '{role}'")
    key = _speaker_cache_key(req.speaker)
    with _icon_lock:
        cached = _icon_manifest.get(key)
    if not cached or cached == "not_found":
        raise HTTPException(404, "Speaker icon not cached yet")
    src = Path(SPEAKER_ICON_DIR) / cached
    if not src.is_file():
        raise HTTPException(404, "Cached icon file missing")
    dest_name = f"_speaker_{key[:16]}.png"
    dest = Path(CHARACTER_LIBRARY_DIR) / dest_name
    import shutil
    shutil.copy2(str(src), str(dest))
    cfg = _load_char_config()
    for r in CHARACTER_IMAGES:
        if r not in cfg:
            cfg[r] = CHARACTER_IMAGES[r]
    cfg[role] = str(dest)
    _save_char_config(cfg)
    _apply_char_config(cfg)
    return {"ok": True, "filename": dest_name}


@app.get("/api/characters/{role}/preview")
async def get_character_preview(role: str):
    from video_composer import CHARACTER_IMAGES
    if role not in CHARACTER_IMAGES:
        raise HTTPException(400, f"Invalid role '{role}'")
    path = Path(CHARACTER_IMAGES[role])
    if not path.is_file():
        raise HTTPException(404, "Character image not found")
    return FileResponse(str(path))


@app.get("/api/speaker_icon")
async def get_speaker_icon(speaker: str, retry: bool = False):
    """Return a cached portrait for the given speaker name.

    On first request the server fetches from external sources (Genshin API /
    Umamusume fandom wiki) and caches the result locally.  Pass ?retry=true to
    bypass a cached "not_found" result and try again.
    """
    key = _speaker_cache_key(speaker)

    with _icon_lock:
        cached = _icon_manifest.get(key)

    if cached and not retry:
        if cached == "not_found":
            raise HTTPException(404, "Icon not available")
        icon_path = Path(SPEAKER_ICON_DIR) / cached
        if icon_path.is_file():
            return FileResponse(str(icon_path), media_type="image/png")
        # File missing from disk → re-fetch below

    english, game = _parse_speaker(speaker)

    try:
        data = await asyncio.to_thread(_fetch_icon_sync, english, game)
    except Exception:
        data = None

    if data is None:
        with _icon_lock:
            _icon_manifest[key] = "not_found"
            _save_icon_manifest()
        raise HTTPException(404, "Icon not available")

    fname    = f"{key}.png"
    out_path = Path(SPEAKER_ICON_DIR) / fname
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_bytes(data)

    with _icon_lock:
        _icon_manifest[key] = fname
        _save_icon_manifest()

    return FileResponse(str(out_path), media_type="image/png")


@app.post("/api/speaker_icon/retry_all")
async def retry_all_not_found():
    """Clear every 'not_found' entry from the manifest so they'll be re-fetched."""
    with _icon_lock:
        keys = [k for k, v in _icon_manifest.items() if v == "not_found"]
        for k in keys:
            del _icon_manifest[k]
        _save_icon_manifest()
    return {"cleared": len(keys)}


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


# ── Evaluation / rating endpoints ────────────────────────────────────────────

@app.post("/api/run/{run_id}/rate")
async def rate_run(run_id: str, req: RatingRequest):
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    if not 1 <= req.score <= 5:
        raise HTTPException(400, "Score must be 1–5")
    entry = {
        "run_id":   run_id,
        "topic":    _runs[run_id].get("topic", ""),
        "score":    req.score,
        "feedback": req.feedback.strip(),
        "ts":       datetime.now().isoformat(),
    }
    with open(RATINGS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _runs[run_id]["rating"] = req.score
    return {"ok": True}


@app.get("/api/stats")
async def get_stats():
    entries: list[dict] = []
    if os.path.exists(RATINGS_PATH):
        with open(RATINGS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
    if not entries:
        return {"count": 0, "avg_score": None, "distribution": {}}
    scores = [e["score"] for e in entries]
    dist = {str(i): scores.count(i) for i in range(1, 6)}
    return {
        "count":        len(entries),
        "avg_score":    round(sum(scores) / len(scores), 2),
        "distribution": dist,
        "recent":       entries[-5:],
    }


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_server:app", host="0.0.0.0", port=PORT, reload=True)
