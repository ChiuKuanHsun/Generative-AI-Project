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
from chart_generator import generate_chart
from video_composer import compose_video

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
    """Interactive speaker picker with search filter."""
    print(f"\n  Picking voice for: {character}")
    if current:
        print(f"  Current: {current}  (press Enter to keep)")

    while True:
        query = input("  Search (partial name, or Enter to list all): ").strip().lower()
        matches = [s for s in speakers if query in s.lower()] if query else speakers

        if not matches:
            print("  No matches. Try a different search term.")
            continue

        # Show up to 30 results
        shown = matches[:30]
        for i, name in enumerate(shown, 1):
            print(f"    [{i:2d}] {name}")
        if len(matches) > 30:
            print(f"    ... and {len(matches) - 30} more. Refine your search.")

        choice = input(f"\n  Enter number (1-{len(shown)}), search again, or Enter to keep current: ").strip()

        if not choice and current:
            return current
        if choice.isdigit() and 1 <= int(choice) <= len(shown):
            return shown[int(choice) - 1]
        print("  Invalid input — try again.")


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

def make_output_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"output_{ts}"
    Path(out_dir).mkdir(exist_ok=True)
    return out_dir


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


def load_test_assets(force_regen_audio: bool = False) -> tuple[dict, list, str | None]:
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

    chart_path = os.path.join(TEST_ASSETS_DIR, "chart.png")
    if not os.path.exists(chart_path):
        chart_path = None

    return dialogue_data, audio_data, chart_path


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
            dialogue_data, audio_data, chart_path = load_test_assets(force_regen_audio=force_regen)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        topic = dialogue_data.get("topic", "Test Topic")
        total = sum(r["duration"] for r in audio_data)
        print(f"  Dialogue: {len(audio_data)} lines | Duration: {total:.1f}s | Chart: {'yes' if chart_path else 'no'}")

        out_dir = make_output_dir()
        print(f"\nOutput folder: {out_dir}/")
        print("\n" + "-" * 45)
        print("Composing video (MoviePy)")
        print("-" * 45)

        video_path = os.path.join(out_dir, "output.mp4")
        compose_video(
            audio_data=audio_data,
            topic=topic,
            chart_path=chart_path,
            output_path=video_path,
        )

        print("\n" + "=" * 45)
        print("Done!")
        print(f"  Video: {video_path}")
        print("=" * 45)
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
    print("Step 1/4  Fetch news & generate dialogue")
    print("-" * 45)
    dialogue_data = fetch_news_and_generate_dialogue(topic)
    dialogue_path = os.path.join(out_dir, "dialogue.json")
    with open(dialogue_path, "w", encoding="utf-8") as f:
        json.dump(dialogue_data, f, ensure_ascii=False, indent=2)
    print(f"Dialogue saved: {dialogue_path}")

    print("\n" + "-" * 45)
    print("Step 2/4  Generate voices (TTS)")
    print("-" * 45)
    audio_dir  = os.path.join(out_dir, "audio")
    audio_data = generate_audio_files(dialogue_data["dialogue"], output_dir=audio_dir)

    print("\n" + "-" * 45)
    print("Step 3/4  Generate chart (Plotly)")
    print("-" * 45)
    chart_path = os.path.join(out_dir, "chart.png")
    try:
        generate_chart(topic, output_path=chart_path)
    except Exception as e:
        print(f"Chart generation failed ({e}), skipping chart.")
        chart_path = None

    print("\n" + "-" * 45)
    print("Step 4/4  Compose video (MoviePy)")
    print("-" * 45)
    video_path = os.path.join(out_dir, "output.mp4")
    compose_video(
        audio_data=audio_data,
        topic=dialogue_data.get("topic", topic),
        chart_path=chart_path,
        output_path=video_path,
    )

    print("\n" + "=" * 45)
    print("All done!")
    print(f"  Video:    {video_path}")
    print(f"  Dialogue: {dialogue_path}")
    if chart_path:
        print(f"  Chart:    {chart_path}")
    print("=" * 45)


if __name__ == "__main__":
    main()
