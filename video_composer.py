"""
Video composer
MoviePy + PIL: GTA background video + character sprites + subtitles + chart

Install: pip install moviepy==2.1.1 Pillow numpy
Font: auto-detects Windows CJK fonts, or set FONT_PATH env var
"""

import os
import random
import textwrap
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from moviepy import (
    AudioFileClip,
    ImageClip,
    VideoFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

# ── Video settings ────────────────────────────────────────────────────────────
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 1280
FPS = 24
BG_COLOR = (10, 10, 18)          # Fallback solid background if no video found
BG_VIDEO_PATH = "backgrounds/gta_footage.mp4"
BG_DIM = 0.35                    # Background brightness (0=black, 1=full)

# Character accent colors
ROLE_COLORS = {
    "gugugaga": (255, 140, 0),   # Orange
    "meowchan": (0, 200, 180),   # Teal
}
DEFAULT_COLOR = (160, 160, 160)

# ── Font detection ────────────────────────────────────────────────────────────
WINDOWS_FONTS = [
    r"C:\Windows\Fonts\msjhbd.ttc",   # Microsoft JhengHei Bold
    r"C:\Windows\Fonts\msjh.ttc",     # Microsoft JhengHei
    r"C:\Windows\Fonts\arial.ttf",    # Arial fallback
    r"C:\Windows\Fonts\calibri.ttf",  # Calibri fallback
]

# ── Character sprite settings ─────────────────────────────────────────────────
CHARACTER_IMAGES = {
    "gugugaga": "characters/character2.png",   # Penguin girl
    "meowchan": "characters/character1.png",   # Cat-ear girl
}
CHAR_HEIGHT = 400       # Character sprite height (px)
CHAR_MAX_WIDTH = 310    # Max width per character (prevents overlap)
CHAR_SPEAK_SCALE = 1.0  # Scale when speaking
CHAR_SILENT_SCALE = 0.78  # Scale when silent
CHAR_SILENT_DIM = 0.35    # Brightness when silent

_char_cache: dict[str, Image.Image] = {}


def _load_character(name: str) -> Image.Image | None:
    if name in _char_cache:
        return _char_cache[name]
    path = CHARACTER_IMAGES.get(name)
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGBA")
        _char_cache[name] = img
        return img
    return None


_font_path_cache: str | None = None
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


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
        print("Warning: No font found. Text may render as boxes. Set FONT_PATH env var.")
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font


# ── Background video helpers ──────────────────────────────────────────────────

def _load_bg_video() -> VideoFileClip | None:
    """Load background video. Returns None if not found."""
    path = os.environ.get("BG_VIDEO_PATH", BG_VIDEO_PATH)
    if not os.path.exists(path):
        print(f"  Background video not found: {path}. Using solid color fallback.")
        return None
    clip = VideoFileClip(path).resized((VIDEO_WIDTH, VIDEO_HEIGHT))
    print(f"  Background video loaded: {path} ({clip.duration:.1f}s)")
    return clip


def _random_subclip(bg: VideoFileClip, duration: float) -> VideoFileClip:
    """Pick a random segment from the background video matching the given duration."""
    max_start = max(0.0, bg.duration - duration - 0.1)
    start = random.uniform(0, max_start)
    return bg.subclipped(start, start + duration)


def _apply_ui_overlay(bg_subclip: VideoFileClip, role: str, line: str,
                       topic: str, chart_img: Image.Image | None) -> VideoFileClip:
    """Composite static UI overlay (title + characters + bubble + chart) onto every video frame."""
    ui_overlay = _make_ui_overlay(role, line, topic, chart_img)
    ui_array = np.array(ui_overlay)  # H x W x 4 (RGBA)

    def process_frame(frame: np.ndarray) -> np.ndarray:
        # Dim background
        bg = Image.fromarray(frame).convert("RGB")
        bg = ImageEnhance.Brightness(bg).enhance(BG_DIM)
        bg_arr = np.array(bg, dtype=np.float32)
        # Alpha composite UI
        alpha = ui_array[:, :, 3:4].astype(np.float32) / 255.0
        ui_rgb = ui_array[:, :, :3].astype(np.float32)
        result = bg_arr * (1 - alpha) + ui_rgb * alpha
        return result.clip(0, 255).astype(np.uint8)

    return bg_subclip.image_transform(process_frame)


def _wrap_line(text: str, font_size: int, max_px: int) -> list[str]:
    """Wrap a single dialogue line for the subtitle bubble.
    Chinese has no spaces so we split by character count; English uses word-wrap."""
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        per_line = max(8, max_px // font_size)
        return [text[i:i + per_line] for i in range(0, len(text), per_line)]
    words_per_line = max(10, max_px // (font_size // 2))
    return textwrap.wrap(text, width=words_per_line) or [text]


def _make_ui_overlay(
    role: str,
    line: str,
    topic: str,
    chart_img: Image.Image | None = None,
) -> Image.Image:
    """
    Generate RGBA overlay (transparent background, opaque UI elements).

    Layout (720x1280):
      0   - 75  : Top title bar (semi-transparent)
      85  - 500 : Character sprites
      500 - 570 : Character name labels
      570 - 875 : Dialogue bubble
      875 - 1280: Chart area
    """
    img = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    role_color = ROLE_COLORS.get(role, DEFAULT_COLOR)

    # 1. Top title bar
    draw.rectangle([(0, 0), (VIDEO_WIDTH, 75)], fill=(10, 10, 25, 210))
    draw.text(
        (VIDEO_WIDTH // 2, 38),
        topic,
        font=_find_font(22),
        fill=(200, 200, 220, 255),
        anchor="mm",
    )

    # 2. Character sprites
    roles_order = ["gugugaga", "meowchan"]
    positions = [VIDEO_WIDTH // 4, 3 * VIDEO_WIDTH // 4]
    char_area_top = 85
    char_area_bottom = 500

    font_small = _find_font(18)

    for r_name, x_center in zip(roles_order, positions):
        r_color = ROLE_COLORS.get(r_name, DEFAULT_COLOR)
        is_speaking = r_name == role

        char_img = _load_character(r_name)

        if char_img is not None:
            # Scale sprite (cap width to prevent overlap)
            scale = CHAR_SPEAK_SCALE if is_speaking else CHAR_SILENT_SCALE
            target_h = int(CHAR_HEIGHT * scale)
            ratio = target_h / char_img.height
            target_w = int(char_img.width * ratio)
            if target_w > CHAR_MAX_WIDTH:
                target_w = CHAR_MAX_WIDTH
                target_h = int(char_img.height * (CHAR_MAX_WIDTH / char_img.width))
            resized = char_img.resize((target_w, target_h), Image.LANCZOS)

            # Dim silent character
            if not is_speaking:
                r_ch, g_ch, b_ch, a_ch = resized.split()
                rgb = Image.merge("RGB", (r_ch, g_ch, b_ch))
                rgb = ImageEnhance.Brightness(rgb).enhance(CHAR_SILENT_DIM)
                resized = Image.merge("RGBA", (*rgb.split(), a_ch))

            # Glow halo for speaking character
            if is_speaking:
                glow_w = target_w + 20
                glow_h = target_h + 20
                gx = x_center - glow_w // 2
                gy = char_area_bottom - target_h - 10
                for spread in range(18, 0, -3):
                    a = int(60 * (18 - spread) / 18)
                    draw.rounded_rectangle(
                        [(gx - spread, gy - spread),
                         (gx + glow_w + spread, gy + glow_h + spread)],
                        radius=16,
                        fill=r_color + (a,),
                    )

            # Paste sprite (bottom-aligned to char_area_bottom)
            paste_x = x_center - target_w // 2
            paste_y = char_area_bottom - target_h
            img.paste(resized, (paste_x, paste_y), resized)

        else:
            # Fallback: draw circle avatar
            avatar_y = (char_area_top + char_area_bottom) // 2
            avatar_r = 90
            fill_color = r_color + (230,) if is_speaking else tuple(c // 3 for c in r_color) + (180,)
            draw.ellipse(
                [(x_center - avatar_r, avatar_y - avatar_r),
                 (x_center + avatar_r, avatar_y + avatar_r)],
                fill=fill_color,
            )
            draw.text(
                (x_center, avatar_y),
                r_name,
                font=_find_font(28),
                fill=(255, 255, 255, 255) if is_speaking else (100, 100, 100, 200),
                anchor="mm",
            )

        # Character name label
        label_y = char_area_bottom + 22
        draw.text(
            (x_center, label_y),
            r_name,
            font=_find_font(24),
            fill=r_color + (255,) if is_speaking else (140, 140, 140, 180),
            anchor="mm",
        )

        # "Speaking" indicator
        if is_speaking:
            draw.text(
                (x_center, label_y + 32),
                "▶ speaking",
                font=font_small,
                fill=r_color + (255,),
                anchor="mm",
            )

    # 3. Dialogue bubble
    bubble_top = 570
    bubble_bottom = 875

    draw.rounded_rectangle(
        [(40, bubble_top), (VIDEO_WIDTH - 40, bubble_bottom)],
        radius=20,
        fill=(10, 10, 25, 200),
        outline=role_color + (255,),
        width=2,
    )

    font_sub = _find_font(34)
    wrapped = _wrap_line(line, font_size=34, max_px=620)
    text_block_height = len(wrapped) * 50
    text_y_start = (bubble_top + bubble_bottom) // 2 - text_block_height // 2

    for j, text_line in enumerate(wrapped):
        draw.text(
            (VIDEO_WIDTH // 2, text_y_start + j * 52),
            text_line,
            font=font_sub,
            fill=(240, 240, 240, 255),
            anchor="mm",
        )

    # 4. Chart area
    if chart_img is not None:
        chart_area_top = 875
        chart_area_height = VIDEO_HEIGHT - chart_area_top - 20
        chart_area_width = VIDEO_WIDTH - 40

        draw.rectangle(
            [(20, chart_area_top - 5), (VIDEO_WIDTH - 20, VIDEO_HEIGHT - 10)],
            fill=(5, 5, 15, 190),
        )

        chart_resized = chart_img.copy().convert("RGBA")
        chart_resized.thumbnail((chart_area_width, chart_area_height), Image.LANCZOS)
        cx = (VIDEO_WIDTH - chart_resized.width) // 2
        cy = chart_area_top + (chart_area_height - chart_resized.height) // 2
        img.paste(chart_resized, (cx, cy), chart_resized)
    else:
        draw.rectangle([(0, 875), (VIDEO_WIDTH, VIDEO_HEIGHT)], fill=(5, 5, 15, 160))
        draw.text(
            (VIDEO_WIDTH // 2, (875 + VIDEO_HEIGHT) // 2),
            "Finance News Short Video",
            font=_find_font(20),
            fill=(80, 80, 100, 200),
            anchor="mm",
        )

    return img


# ── Main compose function ─────────────────────────────────────────────────────

def compose_video(
    audio_data: list,
    topic: str,
    chart_path: str | None = None,
    output_path: str = "output.mp4",
    end_card_duration: float = 4.0,
) -> str:
    """
    Compose the final video.

    Args:
        audio_data: return value from tts_generator.generate_audio_files()
        topic: news headline (shown in title bar)
        chart_path: path to chart PNG (optional)
        output_path: output .mp4 path
        end_card_duration: seconds to show the final chart card
    """
    print(f"\nComposing video ({len(audio_data)} dialogue segments)...")

    chart_img = None
    if chart_path and os.path.exists(chart_path):
        chart_img = Image.open(chart_path).convert("RGB")

    bg_video = _load_bg_video()
    clips = []

    for i, item in enumerate(audio_data):
        print(f"  [{i+1}/{len(audio_data)}] {item['role']}: {item['line'][:30]}...")
        duration = item["duration"]

        if bg_video is not None:
            bg_sub = _random_subclip(bg_video, duration)
            video_clip = _apply_ui_overlay(bg_sub, item["role"], item["line"], topic, chart_img)
        else:
            frame = _make_frame_solid(item["role"], item["line"], topic, chart_img)
            video_clip = ImageClip(frame).with_duration(duration)

        audio_clip = AudioFileClip(item["audio_path"])
        clips.append(video_clip.with_audio(audio_clip))

    # End card
    if chart_img is not None:
        print("  Adding end chart card...")
        if bg_video is not None:
            bg_sub = _random_subclip(bg_video, end_card_duration)
            end_clip = _apply_end_card_overlay(bg_sub, topic, chart_img)
        else:
            end_frame = _make_end_card_solid(topic, chart_img)
            end_clip = ImageClip(end_frame).with_duration(end_card_duration)
        clips.append(end_clip)

    if bg_video is not None:
        bg_video.close()

    final = concatenate_videoclips(clips, method="compose")

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
        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass

    print(f"Video saved: {output_path}")
    return output_path


def _apply_end_card_overlay(bg_subclip: VideoFileClip, topic: str,
                              chart_img: Image.Image) -> VideoFileClip:
    """End card: full-screen chart overlaid on background video."""
    ui = _make_end_card_ui(topic, chart_img)
    ui_array = np.array(ui)

    def process_frame(frame: np.ndarray) -> np.ndarray:
        bg = Image.fromarray(frame).convert("RGB")
        bg = ImageEnhance.Brightness(bg).enhance(BG_DIM * 0.7)
        bg_arr = np.array(bg, dtype=np.float32)
        alpha = ui_array[:, :, 3:4].astype(np.float32) / 255.0
        ui_rgb = ui_array[:, :, :3].astype(np.float32)
        result = bg_arr * (1 - alpha) + ui_rgb * alpha
        return result.clip(0, 255).astype(np.uint8)

    return bg_subclip.image_transform(process_frame)


def _make_end_card_ui(topic: str, chart_img: Image.Image) -> Image.Image:
    """End card RGBA overlay."""
    img = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.rectangle([(0, 0), (VIDEO_WIDTH, 80)], fill=(10, 10, 25, 220))
    draw.text(
        (VIDEO_WIDTH // 2, 40),
        topic,
        font=_find_font(24),
        fill=(200, 200, 230, 255),
        anchor="mm",
    )

    chart_area_top = 120
    chart_area_bottom = VIDEO_HEIGHT - 120
    draw.rectangle(
        [(15, chart_area_top - 10), (VIDEO_WIDTH - 15, chart_area_bottom + 10)],
        fill=(5, 5, 15, 200),
    )

    chart_area_w = VIDEO_WIDTH - 40
    chart_area_h = chart_area_bottom - chart_area_top
    chart_resized = chart_img.copy().convert("RGBA")
    chart_resized.thumbnail((chart_area_w, chart_area_h), Image.LANCZOS)
    cx = (VIDEO_WIDTH - chart_resized.width) // 2
    cy = chart_area_top + (chart_area_h - chart_resized.height) // 2
    img.paste(chart_resized, (cx, cy), chart_resized)

    draw.rectangle([(0, VIDEO_HEIGHT - 80), (VIDEO_WIDTH, VIDEO_HEIGHT)], fill=(10, 10, 25, 200))
    draw.text(
        (VIDEO_WIDTH // 2, VIDEO_HEIGHT - 40),
        "Data for reference only. Invest responsibly.",
        font=_find_font(18),
        fill=(150, 150, 170, 220),
        anchor="mm",
    )

    return img


# ── Fallback solid-color background ──────────────────────────────────────────

def _make_frame_solid(role, line, topic, chart_img=None) -> np.ndarray:
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), BG_COLOR)
    overlay = _make_ui_overlay(role, line, topic, chart_img).convert("RGBA")
    base = img.convert("RGBA")
    composite = Image.alpha_composite(base, overlay)
    return np.array(composite.convert("RGB"))


def _make_end_card_solid(topic, chart_img) -> np.ndarray:
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), (8, 8, 15))
    overlay = _make_end_card_ui(topic, chart_img).convert("RGBA")
    base = img.convert("RGBA")
    composite = Image.alpha_composite(base, overlay)
    return np.array(composite.convert("RGB"))


if __name__ == "__main__":
    print("Run main.py to execute the full pipeline.")
