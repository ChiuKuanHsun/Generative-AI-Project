"""
TTS voice generator — VITS Umamusume via direct HTTP

Bypasses gradio_client to avoid Gradio 3.x / 4.x compatibility issues
with the Plachta/VITS-Umamusume-voice-synthesizer Space.

Setup:
  - Run [v] in main.py to pick a speaker for each character
  - Config is saved to voice_config.json automatically
"""

import json
import os
import re
from pathlib import Path
import requests

try:
    from mutagen.mp3 import MP3
    _HAS_MUTAGEN = True
except ImportError:
    _HAS_MUTAGEN = False

# Optional Traditional → Simplified converter. Captions display Traditional, but
# VITS-Umamusume's tokenizer is trained on Simplified, so we convert only the
# string sent to TTS.
try:
    from zhconv import convert as _zh_convert
except ImportError:
    _zh_convert = None

# ── VITS Umamusume config ─────────────────────────────────────────────────────
# Space runs Gradio 5.x — endpoints are under /gradio_api with SSE result polling.
VITS_SPACE_URL = "https://plachta-vits-umamusume-voice-synthesizer.hf.space"
VITS_API_PREFIX = "/gradio_api"
VITS_API_NAME  = "tts_fn"     # Gradio api_name for main TTS
VITS_LANGUAGE  = "简体中文"   # Other options: "日本語", "English"
VITS_SPEED     = 1.3          # 1.0 = baseline; 1.3 ≈ 30% faster speech + captions
VITS_TIMEOUT   = 180
 

def _to_simplified(text: str) -> str:
    """Convert Traditional → Simplified for VITS. No-op if zhconv is missing."""
    if _zh_convert is None:
        return text
    try:
        return _zh_convert(text, "zh-cn")
    except Exception:
        return text


# Half-width and full-width parens. Contents inside are caption-only annotations
# (e.g. 谷歌(Google)) and must NOT be spoken by the TTS.
_PARENS_RE = re.compile(r"\([^)]*\)|（[^）]*）")


def _strip_annotations(text: str) -> str:
    return _PARENS_RE.sub("", text)


# Safety net: catches FULL ENGLISH WORDS (brand names, person names) that slipped
# past the dialogue prompt's "中文(English)" rule. Acronyms are intentionally
# absent — short all-caps tokens (AI, GPT, BTC, NVDA, IPO …) pass through to the
# TTS so VITS pronounces them letter-by-letter, the way Mandarin speakers say them.
_FALLBACK_SUBS: dict[str, str] = {
    # Cryptos
    "Bitcoin":   "比特幣",
    "Ethereum":  "以太幣",
    "Solana":    "索拉納",
    "Ripple":    "瑞波幣",
    # Tech megacaps
    "Apple":     "蘋果",
    "Tesla":     "特斯拉",
    "Nvidia":    "輝達",
    "Microsoft": "微軟",
    "Google":    "谷歌",
    "Amazon":    "亞馬遜",
    "Meta":      "臉書",
    "Facebook":  "臉書",
    # AI / general
    "OpenAI":    "開放人工智慧",
    "ChatGPT":   "聊天機器人",
    # People
    "Musk":      "馬斯克",
    "Elon":      "伊隆",
    "Altman":    "奧特曼",
    "Huang":     "黃仁勳",
    "Jensen":    "詹森",
    # Indices
    "Nasdaq":    "那斯達克",
}


def _apply_fallback_subs(text: str) -> str:
    """Translate bare full-word English to Chinese; let acronyms pass through."""
    def replace_one(match: re.Match) -> str:
        token = match.group(0)
        if token in _FALLBACK_SUBS:
            return _FALLBACK_SUBS[token]
        for k, v in _FALLBACK_SUBS.items():
            if k.lower() == token.lower():
                return v
        # No translation found — keep the token. Acronyms get pronounced letter-by-letter
        # by VITS; longer unknown words may sound rough but at least aren't dropped.
        return token

    return re.sub(r"[A-Za-z][A-Za-z0-9&]*", replace_one, text)


def _prepare_for_tts(text: str) -> str:
    """Pipeline: strip caption-only annotations → fallback English subs → Trad→Simp."""
    text = _strip_annotations(text)
    text = _apply_fallback_subs(text)
    text = _to_simplified(text)
    return text

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
        r = requests.get(f"{VITS_SPACE_URL}{VITS_API_PREFIX}/config", timeout=15)
        if r.status_code == 404:
            # Older Gradio versions expose /config without the api prefix
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
    """Gradio 5 SSE flow: POST /call/tts_fn -> event_id -> GET SSE -> download audio."""
    tts_text = _prepare_for_tts(text)
    # Inputs: [text, character, language, speed, symbol_input]
    payload = {"data": [tts_text, speaker, VITS_LANGUAGE, VITS_SPEED, False]}

    call_url = f"{VITS_SPACE_URL}{VITS_API_PREFIX}/call/{VITS_API_NAME}"
    r = requests.post(call_url, json=payload, timeout=VITS_TIMEOUT)
    r.raise_for_status()
    event_id = r.json().get("event_id")
    if not event_id:
        raise RuntimeError(f"VITS did not return event_id: {r.text!r}")

    sse = requests.get(f"{call_url}/{event_id}", stream=True, timeout=VITS_TIMEOUT)
    sse.raise_for_status()

    result = None
    current_event = None
    for raw in sse.iter_lines(decode_unicode=True):
        if raw is None or raw == "":
            continue
        if raw.startswith("event:"):
            current_event = raw.split(":", 1)[1].strip()
        elif raw.startswith("data:"):
            payload_str = raw[len("data:"):].strip()
            if current_event == "complete":
                result = json.loads(payload_str)
                break
            if current_event == "error":
                raise RuntimeError(f"VITS error event: {payload_str}")
    if result is None:
        raise RuntimeError("VITS SSE stream ended without a complete event")

    # result format: ["Success", {"path": ..., "url": ..., "meta": ...}]
    audio_info = next(
        (item for item in result if isinstance(item, dict) and (item.get("url") or item.get("path"))),
        None,
    )
    if audio_info is None:
        raise RuntimeError(f"No audio in VITS response: {result!r}")

    file_url = audio_info.get("url") or f"{VITS_SPACE_URL}{VITS_API_PREFIX}/file={audio_info['path']}"
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
        r = requests.get(f"{VITS_SPACE_URL}{VITS_API_PREFIX}/config", timeout=15)
        if r.status_code == 404:
            r = requests.get(f"{VITS_SPACE_URL}/config", timeout=15)
        r.raise_for_status()
        config = r.json()
    except Exception as e:
        print(f"Error fetching config: {e}")
        return

    print(f"\n=== {VITS_SPACE_URL} ===")
    print(f"Gradio version: {config.get('version', 'unknown')}")
    print(f"Protocol:       {config.get('protocol', 'unknown')}")
    print(f"API prefix:     {config.get('api_prefix', '/')}")

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
        print(f"  fn_index={i}  api_name={fn.get('api_name')!r}  inputs={fn.get('inputs')}  outputs={fn.get('outputs')}")


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
