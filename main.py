"""
Auto-generated financial short video pipeline
Full flow: news fetch -> dialogue -> TTS -> chart -> video composition

Usage:
    pip install -r requirements.txt
    python main.py
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("api-key.env")

from news_fetcher import fetch_news_and_generate_dialogue, TOPICS
from tts_generator import generate_audio_files, get_mp3_duration, set_vits_voices, get_vits_speakers
from chart_generator import generate_chart, generate_chart_set
from image_agent import generate_news_images
from video_composer import compose_video
from youtube_uploader import upload_video

TEST_ASSETS_DIR  = "experiment-assets"
VOICE_CONFIG_PATH = "voice_config.json"


# ── Voice config persistence ──────────────────────────────────────────────────

def load_voice_config() -> dict:
    if os.path.exists(VOICE_CONFIG_PATH):
        with open(VOICE_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_voice_config(cfg: dict):
    with open(VOICE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ── Voice selection UI ────────────────────────────────────────────────────────

def _pick_speaker(speakers: list[str], character: str, current: str) -> str:
    """Interactive speaker picker. Shows the list, accepts a number or a search filter."""
    print(f"\n  Picking voice for: {character}")
    if current:
        print(f"  Current: {current}  (press Enter to keep)")

    matches = speakers
    while True:
        shown = matches[:30]
        print()
        for i, name in enumerate(shown, 1):
            print(f"    [{i:2d}] {name}")
        if len(matches) > 30:
            print(f"    ... and {len(matches) - 30} more. Type text to filter.")

        prompt = f"\n  Number 1-{len(shown)} to pick, text to filter"
        if current:
            prompt += ", Enter to keep current"
        prompt += ": "
        choice = input(prompt).strip()

        if not choice:
            if current:
                return current
            print("  No current voice — pick a number or type a search term.")
            continue

        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= len(shown):
                return shown[n - 1]
            print(f"  Out of range. Pick 1-{len(shown)}.")
            continue

        q = choice.lower()
        filtered = [s for s in speakers if q in s.lower()]
        if not filtered:
            print(f"  No speakers contain {choice!r}. Try another term.")
            continue
        matches = filtered
        print(f"  {len(matches)} match(es) for {choice!r}:")


def configure_voices():
    """Interactive UI to assign VITS Umamusume speakers to each character."""
    print("\n╔══════════════════════════════════════════╗")
    print("║       Voice Configuration (VITS)        ║")
    print("╚══════════════════════════════════════════╝")

    current = load_voice_config()

    print("\nFetching speaker list from HuggingFace Space...")
    speakers = get_vits_speakers()

    if not speakers:
        print("\n  Could not load speaker list.")
        print("  The HF Space may be sleeping/loading — try again in a moment.")
        print("  Or run:  python tts_generator.py --diagnose")
        return

    print(f"  Found {len(speakers)} speakers.\n")

    cfg = {}
    for role in ("gugugaga", "meowchan"):
        cfg[role] = _pick_speaker(speakers, role, current.get(role, ""))

    print(f"\n  gugugaga → {cfg['gugugaga']}")
    print(f"  meowchan → {cfg['meowchan']}")
    confirm = input("\n  Save these voices? (Y/n) ").strip().lower()
    if confirm != "n":
        save_voice_config(cfg)
        set_vits_voices(cfg)
        print("  Saved to voice_config.json")
    else:
        print("  Cancelled — keeping previous config.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prompt_youtube_upload(video_path: str, dialogue_data: dict) -> None:
    """Ask whether to upload to YouTube as a Short, then run the upload."""
    if not os.path.exists(video_path):
        return

    confirm = input("\nUpload to YouTube as a Short? (Y/n) ").strip().lower()
    if confirm == "n":
        return

    topic   = dialogue_data.get("topic", "Financial News")
    summary = dialogue_data.get("summary", "")

    title = f"{topic} #Shorts"
    description = (
        f"{summary}\n\n"
        "Auto-generated financial news short. Data for reference only — invest responsibly.\n\n"
        "#Shorts #Finance #News #Crypto #Stocks #AI"
    )
    tags = ["finance", "news", "crypto", "stocks", "AI", "shorts", "market"]

    privacy_in = input("Privacy [public / unlisted / private] (default: public): ").strip().lower()
    privacy = privacy_in if privacy_in in ("public", "unlisted", "private") else "public"

    try:
        url = upload_video(video_path, title, description, tags=tags, privacy=privacy)
        print(f"\n  YouTube: {url}")
    except Exception as e:
        print(f"\nUpload failed: {e}")


def make_output_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"output_{ts}"
    Path(out_dir).mkdir(exist_ok=True)
    return out_dir


def _interleave_visuals(images: list[str], charts: list[str]) -> list[str]:
    """Round-robin: image, chart, image, chart, ... so charts appear throughout."""
    out: list[str] = []
    i, c = 0, 0
    while i < len(images) or c < len(charts):
        if i < len(images):
            out.append(images[i]); i += 1
        if c < len(charts):
            out.append(charts[c]); c += 1
    return out


def _find_cached_audio(audio_dir: str, i: int) -> str | None:
    for ext in (".mp3", ".wav"):
        p = os.path.join(audio_dir, f"line_{i:03d}{ext}")
        if os.path.exists(p):
            return p
    return None


def _audio_is_stale(audio_dir: str, dialogue_path: str, n_lines: int) -> bool:
    """Stale if any file is missing, empty/tiny, or older than the dialogue JSON."""
    if not os.path.isdir(audio_dir):
        return True
    try:
        dlg_mtime = os.path.getmtime(dialogue_path)
    except OSError:
        return True
    for i in range(n_lines):
        p = _find_cached_audio(audio_dir, i)
        if p is None or os.path.getmtime(p) < dlg_mtime:
            return True
        if os.path.getsize(p) < 1000:  # corrupt/empty file
            return True
    return False


def load_test_assets(force_regen_audio: bool = False) -> tuple[dict, list, list[str], list[str]]:
    """Load cached assets. Regenerates audio via local TTS (free) when stale or forced."""
    dialogue_path = os.path.join(TEST_ASSETS_DIR, "dialogue.json")
    with open(dialogue_path, encoding="utf-8") as f:
        dialogue_data = json.load(f)

    audio_dir = os.path.join(TEST_ASSETS_DIR, "audio")
    Path(audio_dir).mkdir(exist_ok=True)

    n = len(dialogue_data["dialogue"])
    stale = _audio_is_stale(audio_dir, dialogue_path, n)

    if force_regen_audio or stale:
        reason = "forced" if force_regen_audio else "cached audio is stale or missing"
        print(f"  Regenerating audio ({reason}) — uses local TTS, no Haiku cost...")
        audio_data = generate_audio_files(dialogue_data["dialogue"], output_dir=audio_dir)
    else:
        print(f"  Using cached audio from {audio_dir}/")
        audio_data = []
        for i, item in enumerate(dialogue_data["dialogue"]):
            path = _find_cached_audio(audio_dir, i)
            duration = get_mp3_duration(path)
            audio_data.append({
                "role":       item["role"],
                "line":       item["line"],
                "emotion":    item.get("emotion", ""),
                "audio_path": path,
                "duration":   duration,
            })

    # Charts: prefer the new chart_<type>.png set, fall back to legacy chart.png
    chart_paths = sorted(str(p) for p in Path(TEST_ASSETS_DIR).glob("chart_*.png"))
    legacy_chart = os.path.join(TEST_ASSETS_DIR, "chart.png")
    if not chart_paths and os.path.exists(legacy_chart):
        chart_paths = [legacy_chart]

    # Pick up cached news images: news_image_00.png, news_image_01.png, ...
    # plus the legacy single news_image.png.
    image_paths = sorted(str(p) for p in Path(TEST_ASSETS_DIR).glob("news_image_*.png"))
    legacy = os.path.join(TEST_ASSETS_DIR, "news_image.png")
    if os.path.exists(legacy) and legacy not in image_paths:
        image_paths.insert(0, legacy)

    return dialogue_data, audio_data, chart_paths, image_paths


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Apply saved voice config on every startup
    saved_voices = load_voice_config()
    if saved_voices:
        set_vits_voices(saved_voices)

    print("╔══════════════════════════════════════════╗")
    print("║  Financial Short Video Generator  v1.0  ║")
    print("╚══════════════════════════════════════════╝")

    # Show active voice engine
    if saved_voices.get("gugugaga"):
        print(f"\n  Voices: gugugaga={saved_voices['gugugaga']}  meowchan={saved_voices['meowchan']}")
    else:
        print("\n  Voices: NOT CONFIGURED — press [v] to pick VITS speakers before generating")

    print("\nSelect a topic:")
    for k, v in TOPICS.items():
        print(f"  [{k}] {v}")
    print("  [5] Custom topic")
    print("  [t] Test mode — no Haiku cost (cached dialogue + cached/fresh audio)")
    print("  [r] Rebuild mode — force-regenerate audio from cached dialogue")
    print("  [v] Configure anime voices (VITS Umamusume)")
    print("  [q] Quit")

    choice = input("\n> ").strip().lower()

    if choice == "q":
        return

    if choice == "v":
        configure_voices()
        return

    # ── Test / Rebuild mode ───────────────────────────────────────────────────
    if choice in ("t", "r"):
        force_regen = (choice == "r")
        label = "Rebuild mode" if force_regen else "Test mode"
        print(f"\n{label}: loading from {TEST_ASSETS_DIR}/ (no Haiku credits used)")
        try:
            dialogue_data, audio_data, chart_paths, image_paths = load_test_assets(force_regen_audio=force_regen)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        topic = dialogue_data.get("topic", "Test Topic")
        total = sum(r["duration"] for r in audio_data)
        print(f"  Dialogue: {len(audio_data)} lines | Duration: {total:.1f}s "
              f"| Charts: {len(chart_paths)} | Images: {len(image_paths)}")

        visuals = _interleave_visuals(image_paths, chart_paths)

        out_dir = make_output_dir()
        print(f"\nOutput folder: {out_dir}/")
        print("\n" + "-" * 45)
        print("Composing video (MoviePy)")
        print("-" * 45)

        video_path = os.path.join(out_dir, "output.mp4")
        compose_video(
            audio_data=audio_data,
            topic=topic,
            chart_path=chart_paths[0] if chart_paths else None,
            image_paths=visuals,
            output_path=video_path,
        )

        print("\n" + "=" * 45)
        print("Done!")
        print(f"  Video: {video_path}")
        print("=" * 45)

        _prompt_youtube_upload(video_path, dialogue_data)
        return

    # ── Normal mode ───────────────────────────────────────────────────────────
    if choice == "5":
        topic = input("Enter custom topic: ").strip()
        if not topic:
            print("Topic cannot be empty!")
            return
    elif choice in TOPICS:
        topic = TOPICS[choice]
    else:
        print("Invalid choice")
        return

    out_dir = make_output_dir()
    print(f"\nOutput folder: {out_dir}/")

    print("\n" + "-" * 45)
    print("Step 1/5  Fetch news & generate dialogue")
    print("-" * 45)
    dialogue_data = fetch_news_and_generate_dialogue(topic)
    dialogue_path = os.path.join(out_dir, "dialogue.json")
    with open(dialogue_path, "w", encoding="utf-8") as f:
        json.dump(dialogue_data, f, ensure_ascii=False, indent=2)
    print(f"Dialogue saved: {dialogue_path}")

    print("\n" + "-" * 45)
    print("Step 2/5  Generate voices (TTS @ 1.3x)")
    print("-" * 45)
    audio_dir  = os.path.join(out_dir, "audio")
    audio_data = generate_audio_files(dialogue_data["dialogue"], output_dir=audio_dir)

    total_dur = sum(item["duration"] for item in audio_data)
    print(f"  Dialogue: {len(audio_data)} lines, {total_dur:.1f}s total")

    print("\n" + "-" * 45)
    print("Step 3/5  Generate charts (dashboard + line + candlestick + bar)")
    print("-" * 45)
    try:
        chart_paths = generate_chart_set(topic, output_dir=out_dir)
    except Exception as e:
        print(f"Chart generation failed ({e}), continuing without charts.")
        chart_paths = []

    # Total visuals = #dialogue lines, split between charts and news photos.
    image_count = max(0, len(audio_data) - len(chart_paths))
    print(f"\n  Visual budget: {len(audio_data)} slots = "
          f"{len(chart_paths)} charts + {image_count} news photos")

    print("\n" + "-" * 45)
    print(f"Step 4/5  Search news images (multi-API agent, {image_count}x)")
    print("-" * 45)
    if image_count > 0:
        try:
            image_paths = generate_news_images(
                dialogue_data.get("topic", topic),
                output_dir=out_dir,
                count=image_count,
            )
        except Exception as e:
            print(f"Image search failed ({e}), continuing without images.")
            image_paths = []
    else:
        print("  Skipping — charts already fill all dialogue slots.")
        image_paths = []

    visuals = _interleave_visuals(image_paths, chart_paths)

    print("\n" + "-" * 45)
    print("Step 5/5  Compose video (MoviePy)")
    print("-" * 45)
    video_path = os.path.join(out_dir, "output.mp4")
    compose_video(
        audio_data=audio_data,
        topic=dialogue_data.get("topic", topic),
        chart_path=chart_paths[0] if chart_paths else None,  # fallback only
        image_paths=visuals,
        output_path=video_path,
    )

    print("\n" + "=" * 45)
    print("All done!")
    print(f"  Video:    {video_path}")
    print(f"  Dialogue: {dialogue_path}")
    if visuals:
        print(f"  Visuals:  {len(visuals)} "
              f"({len(chart_paths)} charts + {len(image_paths)} photos)")
    print("=" * 45)

    _prompt_youtube_upload(video_path, dialogue_data)


if __name__ == "__main__":
    main()
