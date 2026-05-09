"""
Video composer — MoviePy + PIL

Layout (720x1280, portrait, matches re-design.drawio.svg):
  y=0-80     Topic title bar
  y=90-490   gugugaga sprite (top-left, only when gugugaga speaks)
  y=510-820  Center content area (chart or AI-image placeholder)
  y=830-960  Karaoke subtitle band (animated word-by-word highlight)
  y=970-1270 meowchan sprite (bottom-right, only when meowchan speaks)

Only the speaking character is ever rendered — the silent character is hidden.
"""

import os
import random
import re
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from moviepy import (
    AudioFileClip,
    VideoClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy import CompositeAudioClip
from moviepy.video.fx import FadeOut, FadeIn
from moviepy.audio.fx import AudioFadeOut, AudioLoop, MultiplyVolume

# ── Video settings ────────────────────────────────────────────────────────────
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 1280
FPS = 24
BG_COLOR = (10, 10, 18)
BG_VIDEO_DIR = "backgrounds"
BG_DIM = 0.65

# ── Layout boxes (x1, y1, x2, y2) ─────────────────────────────────────────────
TOPIC_BAR    = (0,    0,   720,   80)
GUGUGAGA_BOX = (20,   90,  360,  490)   # top-left
MEOWCHAN_BOX = (380, 970,  700, 1270)   # bottom-right
CONTENT_BOX  = (20,  510,  700,  820)   # center chart / AI image
SUBTITLE_BOX = (30,  830,  690,  960)   # karaoke band

ROLE_COLORS = {
    "gugugaga": (255, 140, 0),
    "meowchan": (0, 200, 180),
}
DEFAULT_COLOR = (160, 160, 160)
ROLE_BOXES = {"gugugaga": GUGUGAGA_BOX, "meowchan": MEOWCHAN_BOX}

CHARACTER_IMAGES = {
    "gugugaga": "characters/character2.png",
    "meowchan": "characters/character1.png",
}

WINDOWS_FONTS = [
    r"C:\Windows\Fonts\msjhbd.ttc",
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
]

SUBTITLE_FONT_SIZE = 34
TOPIC_FONT_SIZE = 22
PLACEHOLDER_FONT_SIZE = 28

# ── Caches ────────────────────────────────────────────────────────────────────
_char_cache: dict[str, Image.Image] = {}
_font_path_cache: str | None = None
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _load_character(name: str) -> Image.Image | None:
    if name in _char_cache:
        return _char_cache[name]
    path = CHARACTER_IMAGES.get(name)
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGBA")
        _char_cache[name] = img
        return img
    return None


def _get_font_path() -> str | None:
    global _font_path_cache
    if _font_path_cache is not None:
        return _font_path_cache
    env_path = os.environ.get("FONT_PATH")
    if env_path and os.path.exists(env_path):
        _font_path_cache = env_path
        return _font_path_cache
    for path in WINDOWS_FONTS:
        if os.path.exists(path):
            _font_path_cache = path
            return _font_path_cache
    return None


def _find_font(size: int) -> ImageFont.FreeTypeFont:
    if size in _font_cache:
        return _font_cache[size]
    path = _get_font_path()
    if path:
        font = ImageFont.truetype(path, size)
    else:
        print("Warning: No font found. Set FONT_PATH env var.")
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font


# ── Background video helpers ──────────────────────────────────────────────────

def _load_bg_video() -> VideoFileClip | None:
    """Honors BG_VIDEO_PATH env var, else picks any video in backgrounds/."""
    env_path = os.environ.get("BG_VIDEO_PATH")
    if env_path:
        candidates = [env_path] if os.path.exists(env_path) else []
    else:
        bg_dir = Path(BG_VIDEO_DIR)
        candidates = sorted(
            p for ext in ("*.mp4", "*.mov", "*.webm", "*.mkv")
            for p in bg_dir.glob(ext)
        ) if bg_dir.is_dir() else []

    if not candidates:
        print(f"  No background video found in {BG_VIDEO_DIR}/. Using solid color fallback.")
        return None

    path = str(candidates[0])
    clip = VideoFileClip(path).resized((VIDEO_WIDTH, VIDEO_HEIGHT))
    print(f"  Background video loaded: {path} ({clip.duration:.1f}s)")
    return clip


def _random_subclip(bg: VideoFileClip, duration: float) -> VideoFileClip:
    max_start = max(0.0, bg.duration - duration - 0.1)
    start = random.uniform(0, max_start)
    return bg.subclipped(start, start + duration)


# ── Character rendering ──────────────────────────────────────────────────────

def _paste_character(img: Image.Image, role: str) -> None:
    """Paste the speaking character into their assigned corner with a soft glow."""
    box = ROLE_BOXES.get(role)
    if box is None:
        return

    char_img = _load_character(role)
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    if char_img is None:
        # Fallback: filled circle avatar with role label
        draw = ImageDraw.Draw(img)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        r = min(bw, bh) // 2 - 10
        rc = ROLE_COLORS.get(role, DEFAULT_COLOR)
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=rc + (230,))
        draw.text((cx, cy), role, font=_find_font(32),
                  fill=(255, 255, 255, 255), anchor="mm")
        return

    # Fit sprite into the corner box, preserve aspect
    ratio = min(bw / char_img.width, bh / char_img.height)
    target_w = int(char_img.width * ratio)
    target_h = int(char_img.height * ratio)
    resized = char_img.resize((target_w, target_h), Image.LANCZOS)

    # Anchor: gugugaga at top-left of its box, meowchan at bottom-right
    if role == "meowchan":
        px = x2 - target_w
        py = y2 - target_h
    else:
        px = x1
        py = y2 - target_h  # bottom-aligned within top-left box

    rc = ROLE_COLORS.get(role, DEFAULT_COLOR)
    draw = ImageDraw.Draw(img)
    for spread in range(20, 0, -3):
        a = int(50 * (20 - spread) / 20)
        draw.rounded_rectangle(
            [(px - spread, py - spread),
             (px + target_w + spread, py + target_h + spread)],
            radius=18,
            fill=rc + (a,),
        )

    img.paste(resized, (px, py), resized)


# ── Static UI overlay (topic + speaker + chart + subtitle pill) ──────────────

def _make_static_overlay(role: str, topic: str,
                          chart_img: Image.Image | None) -> Image.Image:
    """Pieces of the frame that don't change with time. Subtitle text drawn separately."""
    img = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    rc = ROLE_COLORS.get(role, DEFAULT_COLOR)

    # Topic bar
    tx1, ty1, tx2, ty2 = TOPIC_BAR
    draw.rectangle([(tx1, ty1), (tx2, ty2)], fill=(10, 10, 25, 215))
    draw.text(((tx1 + tx2) // 2, (ty1 + ty2) // 2), topic,
              font=_find_font(TOPIC_FONT_SIZE),
              fill=(200, 200, 220, 255), anchor="mm")

    # Speaker only (silent character is hidden)
    _paste_character(img, role)

    # Center content
    cx1, cy1, cx2, cy2 = CONTENT_BOX
    cw, ch = cx2 - cx1, cy2 - cy1
    if chart_img is not None:
        # Cover mode: fill content box, center-crop — no black letterbox bars
        src = chart_img.copy().convert("RGBA")
        scale = max(cw / src.width, ch / src.height)
        new_w, new_h = int(src.width * scale), int(src.height * scale)
        scaled = src.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - cw) // 2
        top  = (new_h - ch) // 2
        cropped = scaled.crop((left, top, left + cw, top + ch))
        # Rounded-corner clip via alpha mask
        mask = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(mask).rounded_rectangle([(0, 0), (cw - 1, ch - 1)], radius=14, fill=255)
        cropped.putalpha(mask)
        img.paste(cropped, (cx1, cy1), cropped)
        draw.rounded_rectangle([(cx1, cy1), (cx2, cy2)], radius=14,
                                outline=(80, 80, 120, 160), width=2)
    else:
        draw.rounded_rectangle([(cx1, cy1), (cx2, cy2)], radius=14,
                                fill=(5, 5, 15, 180),
                                outline=(80, 80, 100, 200), width=2)
        draw.text(((cx1 + cx2) // 2, (cy1 + cy2) // 2),
                  "[ AI image content ]",
                  font=_find_font(PLACEHOLDER_FONT_SIZE),
                  fill=(120, 120, 140, 220), anchor="mm")

    # Subtitle pill (background only — text drawn in karaoke layer)
    sx1, sy1, sx2, sy2 = SUBTITLE_BOX
    draw.rounded_rectangle([(sx1, sy1), (sx2, sy2)], radius=20,
                            fill=(10, 10, 25, 215),
                            outline=rc + (220,), width=2)

    # Responsible AI disclaimer watermark (bottom-left, outside both character boxes)
    wm_text = "AI 生成內容 · 僅供參考"
    wm_font = _find_font(16)
    bbox = draw.textbbox((0, 0), wm_text, font=wm_font)
    wm_w = bbox[2] - bbox[0]
    wm_h = bbox[3] - bbox[1]
    wm_x, wm_y = 14, VIDEO_HEIGHT - wm_h - 14
    pad = 5
    draw.rounded_rectangle(
        [(wm_x - pad, wm_y - pad), (wm_x + wm_w + pad, wm_y + wm_h + pad)],
        radius=6, fill=(0, 0, 0, 140),
    )
    draw.text((wm_x, wm_y), wm_text, font=wm_font, fill=(210, 210, 210, 210))

    return img


# ── Karaoke subtitle ─────────────────────────────────────────────────────────

def _is_cjk(text: str) -> bool:
    return any('一' <= c <= '鿿' for c in text)


# Match (English) or （...）groups — caption-only annotations, never karaoke-driven.
_ANNOTATION_RE = re.compile(r"\([^)]*\)|（[^）]*）")
ANNOTATION_COLOR = (170, 200, 240, 230)  # light blue: always visible, never animated


def _build_unit_timings(text: str, duration: float):
    """Returns list of (unit, start, end, speakable). Annotations like (Google) are
    one non-speakable display unit each; karaoke timing flows only through speakable
    units (CJK chars or English words)."""
    if not text:
        return []

    cjk_mode = _is_cjk(text)
    parts = _ANNOTATION_RE.split(text)
    annotations = _ANNOTATION_RE.findall(text)

    units: list[tuple[str, bool]] = []
    # Interleave: parts[0], annotations[0], parts[1], annotations[1], ..., parts[-1]
    for i, part in enumerate(parts):
        if part:
            if cjk_mode:
                # CJK chars per-char; group consecutive ASCII letters/digits as one unit
                j = 0
                while j < len(part):
                    c = part[j]
                    if c.isspace():
                        j += 1
                    elif c.isascii() and (c.isalnum() or c in "$%&-.,"):
                        k = j
                        while k < len(part) and part[k].isascii() and (
                            part[k].isalnum() or part[k] in "$%&-.,"
                        ):
                            k += 1
                        units.append((part[j:k], True))
                        j = k
                    else:
                        units.append((c, True))
                        j += 1
            else:
                for w in part.split():
                    if w:
                        units.append((w, True))
        if i < len(annotations):
            units.append((annotations[i], False))

    speakable_count = sum(1 for _, s in units if s)
    if speakable_count == 0:
        # All annotations or all whitespace — no karaoke, but still render text
        return [(u, 0.0, 0.0, sp) for u, sp in units]

    per = duration / speakable_count
    timings = []
    speak_idx = 0
    for unit_text, speakable in units:
        if speakable:
            timings.append((unit_text, speak_idx * per, (speak_idx + 1) * per, True))
            speak_idx += 1
        else:
            timings.append((unit_text, 0.0, 0.0, False))
    return timings


def _layout_subtitle(timings, font, max_w):
    """Compute (idx, x, y, unit, speakable) positions and total height."""
    if not timings:
        return [], 0

    is_cjk = _is_cjk("".join(u for u, _, _, _ in timings))
    space_w = font.getbbox(" ")[2] - font.getbbox(" ")[0] if not is_cjk else 0
    line_h = font.size + 10

    widths = [font.getbbox(u)[2] - font.getbbox(u)[0] for u, _, _, _ in timings]

    lines = []
    cur_line, cur_w = [], 0
    for i, ((unit, _, _, speakable), uw) in enumerate(zip(timings, widths)):
        gap = 0 if (is_cjk or not cur_line) else space_w
        if cur_line and cur_w + gap + uw > max_w:
            lines.append(cur_line)
            cur_line, cur_w = [], 0
            gap = 0
        cur_line.append((i, unit, uw, gap, speakable))
        cur_w += gap + uw
    if cur_line:
        lines.append(cur_line)

    positions = []
    for li, line in enumerate(lines):
        line_w = sum(uw + gap for _, _, uw, gap, _ in line)
        x = (max_w - line_w) // 2
        y = li * line_h
        for idx, unit, uw, gap, speakable in line:
            x += gap
            positions.append((idx, x, y, unit, speakable))
            x += uw

    return positions, len(lines) * line_h


def _make_subtitle_text_layer(positions, total_h: int,
                                current_idx: int, role_color: tuple) -> Image.Image:
    """Render subtitle text with one speakable unit highlighted. Annotations are
    drawn in a constant secondary color regardless of current_idx."""
    sx1, sy1, sx2, sy2 = SUBTITLE_BOX
    sh = sy2 - sy1
    layer = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
    if not positions:
        return layer

    draw = ImageDraw.Draw(layer)
    font = _find_font(SUBTITLE_FONT_SIZE)
    y_offset = sy1 + (sh - total_h) // 2

    for idx, x, y, unit, speakable in positions:
        if not speakable:
            fill = ANNOTATION_COLOR              # english-in-parens: always dim blue
        elif idx == current_idx:
            fill = (255, 215, 90, 255)            # gold: active
        elif idx < current_idx:
            fill = (230, 230, 245, 255)           # spoken
        else:
            fill = (140, 140, 165, 230)           # upcoming
        draw.text((sx1 + x, y_offset + y), unit, font=font, fill=fill)

    return layer


# ── Per-line clip composition ────────────────────────────────────────────────

def _build_dialogue_clip(bg_subclip, role: str, line: str, topic: str,
                          chart_img: Image.Image | None,
                          duration: float):
    """Apply BG dim + static UI + animated karaoke subtitle to bg_subclip."""
    rc = ROLE_COLORS.get(role, DEFAULT_COLOR)

    static_arr = np.array(_make_static_overlay(role, topic, chart_img))
    static_alpha = static_arr[:, :, 3:4].astype(np.float32) / 255.0
    static_rgb = static_arr[:, :, :3].astype(np.float32)

    timings = _build_unit_timings(line, duration)
    font = _find_font(SUBTITLE_FONT_SIZE)
    sx1, _, sx2, _ = SUBTITLE_BOX
    max_w = sx2 - sx1 - 30
    positions, total_h = _layout_subtitle(timings, font, max_w)

    # Pre-render N+1 subtitle states (idx = -1 means "no current unit yet")
    sub_layers = [
        np.array(_make_subtitle_text_layer(positions, total_h, cur, rc))
        for cur in range(-1, len(timings))
    ]
    sub_alphas = [layer[:, :, 3:4].astype(np.float32) / 255.0 for layer in sub_layers]
    sub_rgbs = [layer[:, :, :3].astype(np.float32) for layer in sub_layers]

    # Time-to-index lookup runs ONLY across speakable units; annotations never
    # become the "current" highlight. speakable_to_timings maps speakable order
    # back to the unit index inside `timings`.
    speakable_to_timings = [i for i, t in enumerate(timings) if t[3]]
    starts = np.array([timings[i][1] for i in speakable_to_timings] or [0.0])
    ends   = np.array([timings[i][2] for i in speakable_to_timings] or [0.0])

    def process(get_frame, t):
        frame = get_frame(t)
        bg = Image.fromarray(frame).convert("RGB")
        bg = ImageEnhance.Brightness(bg).enhance(BG_DIM)
        out = np.array(bg, dtype=np.float32)
        out = out * (1 - static_alpha) + static_rgb * static_alpha

        # Find current speakable unit at time t, then map to the timings index
        if speakable_to_timings:
            mask = (starts <= t) & (t < ends)
            idx = speakable_to_timings[int(np.argmax(mask))] if mask.any() else -1
        else:
            idx = -1
        sa = sub_alphas[idx + 1]
        sr = sub_rgbs[idx + 1]
        out = out * (1 - sa) + sr * sa

        return out.clip(0, 255).astype(np.uint8)

    return bg_subclip.transform(process)


def _build_solid_clip(role: str, line: str, topic: str,
                       chart_img: Image.Image | None,
                       duration: float):
    """Same as dialogue clip but with a solid-color background (no bg video)."""
    rc = ROLE_COLORS.get(role, DEFAULT_COLOR)

    static_arr = np.array(_make_static_overlay(role, topic, chart_img))
    static_alpha = static_arr[:, :, 3:4].astype(np.float32) / 255.0
    static_rgb = static_arr[:, :, :3].astype(np.float32)

    bg_solid = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), BG_COLOR, dtype=np.float32)
    bg_with_ui = bg_solid * (1 - static_alpha) + static_rgb * static_alpha

    timings = _build_unit_timings(line, duration)
    font = _find_font(SUBTITLE_FONT_SIZE)
    sx1, _, sx2, _ = SUBTITLE_BOX
    max_w = sx2 - sx1 - 30
    positions, total_h = _layout_subtitle(timings, font, max_w)

    sub_layers = [
        np.array(_make_subtitle_text_layer(positions, total_h, cur, rc))
        for cur in range(-1, len(timings))
    ]
    sub_alphas = [layer[:, :, 3:4].astype(np.float32) / 255.0 for layer in sub_layers]
    sub_rgbs = [layer[:, :, :3].astype(np.float32) for layer in sub_layers]

    starts = np.array([s for _, s, _ in timings] or [0.0])
    ends = np.array([e for _, _, e in timings] or [0.0])

    def make_frame(t):
        if speakable_to_timings:
            mask = (starts <= t) & (t < ends)
            idx = speakable_to_timings[int(np.argmax(mask))] if mask.any() else -1
        else:
            idx = -1
        out = bg_with_ui * (1 - sub_alphas[idx + 1]) + sub_rgbs[idx + 1] * sub_alphas[idx + 1]
        return out.clip(0, 255).astype(np.uint8)

    return VideoClip(make_frame, duration=duration)


# ── Main compose function ─────────────────────────────────────────────────────

def compose_video(
    audio_data: list,
    topic: str,
    chart_path: str | None = None,
    image_paths: list[str] | str | None = None,
    output_path: str = "output.mp4",
    bgm_path: str | None = None,
) -> str:
    """Stitch dialogue clips into the final mp4.

    image_paths: list of news images for the dialogue content area; rotated across
                 dialogue lines so the same picture isn't on screen the whole video.
                 Falls back to the chart if empty/None. Accepts a single string for
                 backwards compatibility.
    chart_path:  data chart used as the dialogue fallback when no images are
                 available. (Chart is saved separately on disk; not embedded as an
                 end card.)
    """
    print(f"\nComposing video ({len(audio_data)} dialogue segments)...")

    chart_img = None
    if chart_path and os.path.exists(chart_path):
        chart_img = Image.open(chart_path).convert("RGB")

    if isinstance(image_paths, str):
        image_paths = [image_paths]
    image_paths = image_paths or []

    dialogue_imgs: list[Image.Image] = []
    for p in image_paths:
        if p and os.path.exists(p):
            try:
                dialogue_imgs.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"  Could not load {p}: {e}")

    if dialogue_imgs:
        print(f"  Dialogue images: {len(dialogue_imgs)} (rotating across lines)")
    elif chart_img is not None:
        print("  No news images available — using chart in content area.")
        dialogue_imgs = [chart_img]
    else:
        dialogue_imgs = [None]  # placeholder will be drawn

    bg_video = _load_bg_video()
    clips = []

    for i, item in enumerate(audio_data):
        print(f"  [{i+1}/{len(audio_data)}] {item['role']}: {item['line'][:30]}...")
        duration = item["duration"]
        dialogue_img = dialogue_imgs[i % len(dialogue_imgs)]

        if bg_video is not None:
            bg_sub = _random_subclip(bg_video, duration)
            video_clip = _build_dialogue_clip(
                bg_sub, item["role"], item["line"], topic, dialogue_img, duration
            )
        else:
            video_clip = _build_solid_clip(
                item["role"], item["line"], topic, dialogue_img, duration
            )

        audio_clip = AudioFileClip(item["audio_path"])
        clips.append(video_clip.with_audio(audio_clip))

    if bg_video is not None:
        bg_video.close()

    dialogue_final = concatenate_videoclips(clips, method="compose")

    # Fade out the last second of dialogue (video + audio)
    FADE_DUR = 1.0
    dialogue_final = dialogue_final.with_effects([FadeOut(FADE_DUR)])
    if dialogue_final.audio is not None:
        dialogue_final = dialogue_final.with_audio(
            dialogue_final.audio.with_effects([AudioFadeOut(FADE_DUR)])
        )

    # Append outro.mov if present
    outro_clip = None
    outro_path = os.path.join("backgrounds", "outro.mov")
    if os.path.exists(outro_path):
        print("  Loading outro clip...")
        outro_raw = VideoFileClip(outro_path)
        outro_clip = outro_raw.resized((VIDEO_WIDTH, VIDEO_HEIGHT)).with_effects([FadeIn(0.5)])
        final = concatenate_videoclips([dialogue_final, outro_clip], method="compose")
    else:
        print("  (backgrounds/outro.mov not found — skipping outro)")
        final = dialogue_final

    # Mix in BGM
    BGM_VOLUME = 0.15
    bgm_clip = None
    _bgm_file: str | None = None

    if bgm_path == "":
        print("  BGM disabled.")
    elif bgm_path is not None:
        _bgm_file = bgm_path
    else:
        # Auto-scan fallback
        for bgm_name in ("bgm.mp3", "bgm.wav", "bgm.m4a"):
            candidate = os.path.join("backgrounds", bgm_name)
            if os.path.exists(candidate):
                _bgm_file = candidate
                break

    if _bgm_file and os.path.exists(_bgm_file):
        print(f"  Loading BGM: {_bgm_file}")
        bgm_clip = AudioFileClip(_bgm_file)
        bgm_clip = bgm_clip.with_effects([
            AudioLoop(duration=final.duration),
            MultiplyVolume(BGM_VOLUME),
            AudioFadeOut(1.5),
        ])
        if final.audio is not None:
            final = final.with_audio(CompositeAudioClip([final.audio, bgm_clip]))
        else:
            final = final.with_audio(bgm_clip)
        print(f"  BGM mixed in at {int(BGM_VOLUME*100)}% volume")
    elif bgm_path is None and _bgm_file is None:
        print("  (no BGM found in backgrounds/ — skipping)")

    print(f"\nRendering video ({final.duration:.1f}s)... this may take a few minutes")
    try:
        final.write_videofile(
            output_path,
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )
    finally:
        final.close()
        dialogue_final.close()
        if outro_clip is not None:
            try:
                outro_clip.close()
            except Exception:
                pass
        if bgm_clip is not None:
            try:
                bgm_clip.close()
            except Exception:
                pass
        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass

    print(f"Video saved: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Run main.py to execute the full pipeline.")
