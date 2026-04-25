"""
TTS voice generator — VITS Umamusume via direct HTTP

Bypasses gradio_client to avoid Gradio 3.x / 4.x compatibility issues
with the Plachta/VITS-Umamusume-voice-synthesizer Space.

Setup:
  - Run [v] in main.py to pick a speaker for each character
  - Config is saved to voice_config.json automatically
"""

import os
from pathlib import Path
import requests

try:
    from mutagen.mp3 import MP3
    _HAS_MUTAGEN = True
except ImportError:
    _HAS_MUTAGEN = False

# ── VITS Umamusume config ─────────────────────────────────────────────────────
VITS_SPACE_URL = "https://plachta-vits-umamusume-voice-synthesizer.hf.space"
VITS_LANGUAGE  = "简体中文"   # Other options: "日本語", "English"
VITS_FN_INDEX  = 0           # Gradio function index for main TTS
VITS_TIMEOUT   = 120

# Populated at runtime by set_vits_voices() — called from main.py on startup
VITS_VOICES: dict[str, str] = {
    "gugugaga": "",
    "meowchan": "",
}


def set_vits_voices(voices: dict[str, str]):
    """Apply saved speaker assignments. Called from main.py at startup."""
    VITS_VOICES.update(voices)


def _vits_available() -> bool:
    return bool(VITS_VOICES.get("gugugaga") and VITS_VOICES.get("meowchan"))


# ── Speaker list discovery ────────────────────────────────────────────────────

def get_vits_speakers() -> list[str]:
    """Fetch speaker dropdown choices from the Space's /config endpoint."""
    try:
        print("  Connecting to HuggingFace Space...")
        r = requests.get(f"{VITS_SPACE_URL}/config", timeout=15)
        r.raise_for_status()
        config = r.json()
    except Exception as e:
        print(f"  Could not fetch config: {e}")
        return []

    for component in config.get("components", []):
        props = component.get("props", {})
        choices = props.get("choices") or []
        if isinstance(choices, list) and len(choices) > 10:
            # Choices may be [(label, value), ...] or plain [label, ...]
            names = [
                str(c[0]) if isinstance(c, (list, tuple)) else str(c)
                for c in choices
            ]
            return sorted(set(names))
    return []


# ── Synthesis via direct HTTP ─────────────────────────────────────────────────

def _vits_synthesize_one(text: str, speaker: str, output_path: str) -> str:
    """POST to /run/predict with (text, speaker, language, speed), download audio."""
    payload = {
        "data":     [text, speaker, VITS_LANGUAGE, 1.0],
        "fn_index": VITS_FN_INDEX,
    }
    r = requests.post(
        f"{VITS_SPACE_URL}/run/predict",
        json=payload,
        timeout=VITS_TIMEOUT,
    )
    r.raise_for_status()
    result = r.json()

    data = result.get("data") or []
    audio_info = next(
        (item for item in data if isinstance(item, dict) and item.get("name")),
        None,
    )
    if audio_info is None:
        raise RuntimeError(f"No audio in VITS response: {result!r}")

    file_url = f"{VITS_SPACE_URL}/file={audio_info['name']}"
    audio_r = requests.get(file_url, timeout=60)
    audio_r.raise_for_status()

    wav_path = output_path.replace(".mp3", ".wav")
    with open(wav_path, "wb") as f:
        f.write(audio_r.content)

    size = os.path.getsize(wav_path)
    if size < 1000:
        raise RuntimeError(f"VITS audio too small ({size} bytes): {result!r}")
    return wav_path


def _vits_synthesize_all(items: list, output_dir: str) -> list[str]:
    # Wipe any stale line_* files from prior failed runs so we never mix outputs
    for stale in Path(output_dir).glob("line_*.*"):
        try:
            stale.unlink()
        except OSError:
            pass

    paths = []
    for i, item in enumerate(items):
        speaker = VITS_VOICES.get(item["role"]) or VITS_VOICES.get("meowchan", "")
        out = os.path.join(output_dir, f"line_{i:03d}.mp3")
        print(f"  [{item['role']}] ({speaker}) {item['line'][:35]}...")
        paths.append(_vits_synthesize_one(item["line"], speaker, out))
    return paths


# ── Diagnostics ───────────────────────────────────────────────────────────────

def diagnose_vits():
    """Inspect the Space's /config and print component/function layout."""
    try:
        r = requests.get(f"{VITS_SPACE_URL}/config", timeout=15)
        r.raise_for_status()
        config = r.json()
    except Exception as e:
        print(f"Error fetching config: {e}")
        return

    print(f"\n=== {VITS_SPACE_URL} ===")
    print(f"Gradio version: {config.get('version', 'unknown')}")

    components = config.get("components", [])
    print(f"\nComponents ({len(components)}):")
    for i, c in enumerate(components):
        ctype   = c.get("type", "?")
        props   = c.get("props", {})
        label   = props.get("label", "")
        choices = props.get("choices") or []
        extra   = f"  ({len(choices)} choices)" if choices else ""
        print(f"  [{i}] {ctype:12} label={label!r:30}{extra}")

    fns = config.get("dependencies", [])
    print(f"\nFunctions ({len(fns)}):")
    for i, fn in enumerate(fns):
        print(f"  fn_index={i}  inputs={fn.get('inputs')}  outputs={fn.get('outputs')}")


# ── Audio duration ────────────────────────────────────────────────────────────

def get_audio_duration(path: str) -> float:
    if path.endswith(".mp3") and _HAS_MUTAGEN:
        try:
            return MP3(path).info.length
        except Exception:
            pass
    if path.endswith(".wav"):
        try:
            import wave
            with wave.open(path) as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            pass
    try:
        return max(1.0, os.path.getsize(path) / 16000)
    except Exception:
        return 3.0


def get_mp3_duration(path: str) -> float:
    return get_audio_duration(path)


# ── Main interface ────────────────────────────────────────────────────────────

def generate_audio_files(dialogue: list, output_dir: str = "audio") -> list:
    """Generate audio for each dialogue line. Raises if voices aren't configured."""
    Path(output_dir).mkdir(exist_ok=True)
    print(f"\nGenerating audio ({len(dialogue)} lines)...")

    if not _vits_available():
        raise RuntimeError(
            "VITS voices not configured. Run main.py and press [v] to pick "
            "a speaker for each character."
        )

    print(f"  Using VITS  gugugaga={VITS_VOICES['gugugaga']}  meowchan={VITS_VOICES['meowchan']}")
    actual_paths = _vits_synthesize_all(dialogue, output_dir)

    results = []
    for i, item in enumerate(dialogue):
        path = actual_paths[i]
        duration = get_audio_duration(path)
        results.append({
            "role":       item["role"],
            "line":       item["line"],
            "emotion":    item.get("emotion", ""),
            "audio_path": path,
            "duration":   duration,
        })

    total = sum(r["duration"] for r in results)
    print(f"\nAudio complete. Total: {total:.1f}s")
    return results


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv("api-key.env")

    if "--diagnose" in sys.argv:
        diagnose_vits()
        sys.exit(0)

    test_dialogue = [
        {"role": "gugugaga", "line": "比特币今天又跌了，我的钱包在哭。", "emotion": "sarcastic"},
        {"role": "meowchan", "line": "等等，比特币跌了是什么意思？",       "emotion": "confused"},
        {"role": "gugugaga", "line": "意思就是你的钱包变薄了，就这么简单。", "emotion": "serious"},
    ]

    results = generate_audio_files(test_dialogue, output_dir="audio_test")
    for r in results:
        print(f"{r['role']}: {r['audio_path']} ({r['duration']:.1f}s)")
