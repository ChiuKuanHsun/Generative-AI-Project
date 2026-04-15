"""
影片合成器
MoviePy + PIL 合成：背景 + 角色指示 + 中文字幕 + 圖表

安裝：pip install moviepy==2.1.1 Pillow numpy

字體：自動偵測 Windows 系統中文字體，或指定 FONT_PATH 環境變數
"""

import os
import textwrap
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

# ── 影片設定 ──────────────────────────────────────────────────────────────────
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 1280
FPS = 24
BG_COLOR = (10, 10, 18)          # 深藍黑背景

# 角色顏色
ROLE_COLORS = {
    "老王": (255, 140, 0),        # 橘色
    "小咪": (0, 200, 180),        # 青色
}
DEFAULT_COLOR = (160, 160, 160)

# ── 字體偵測 ─────────────────────────────────────────────────────────────────
WINDOWS_FONTS = [
    r"C:\Windows\Fonts\msjhbd.ttc",   # 微軟正黑體 Bold
    r"C:\Windows\Fonts\msjh.ttc",     # 微軟正黑體
    r"C:\Windows\Fonts\mingliu.ttc",  # 細明體
    r"C:\Windows\Fonts\simsun.ttc",   # 新細明體
]


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
        print("⚠️ 找不到中文字體，字幕可能顯示為方塊。請設定 FONT_PATH 環境變數。")
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font


# ── 畫面生成函式 ──────────────────────────────────────────────────────────────

def _make_frame(
    role: str,
    line: str,
    topic: str,
    chart_img: Image.Image | None = None,
) -> np.ndarray:
    """
    生成單一對話行的靜態畫面（numpy array, RGB）。

    佈局（720x1280）：
      0  -  80  : 頂部標題欄
      80 - 450  : 角色區域（兩個頭像，發言者高亮）
      450 - 900 : 字幕區域
      900 - 1280: 圖表區域（若有）
    """
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    role_color = ROLE_COLORS.get(role, DEFAULT_COLOR)

    # ── 1. 頂部標題欄 ────────────────────────────────────────────────────────
    draw.rectangle([(0, 0), (VIDEO_WIDTH, 75)], fill=(20, 20, 35))
    font_title = _find_font(22)
    draw.text((VIDEO_WIDTH // 2, 38), topic, font=font_title, fill=(200, 200, 220), anchor="mm")

    # ── 2. 角色頭像區 ────────────────────────────────────────────────────────
    roles_order = ["老王", "小咪"]
    positions = [VIDEO_WIDTH // 4, 3 * VIDEO_WIDTH // 4]  # x 中心
    avatar_y = 260  # 圓心 y
    avatar_r = 100  # 半徑

    font_name = _find_font(28)
    font_small = _find_font(18)

    for r_name, x_center in zip(roles_order, positions):
        r_color = ROLE_COLORS.get(r_name, DEFAULT_COLOR)
        is_speaking = r_name == role

        # 光暈（發言者）：由外到內漸層，alpha 從 0 增大到 80
        if is_speaking:
            for radius in range(avatar_r + 30, avatar_r - 1, -2):
                alpha = int(80 * (avatar_r + 30 - radius) / 30)
                glow_color = tuple(max(0, min(255, int(c * alpha / 80))) for c in r_color)
                draw.ellipse(
                    [(x_center - radius, avatar_y - radius),
                     (x_center + radius, avatar_y + radius)],
                    outline=glow_color,
                    width=2,
                )

        # 頭像圓
        fill_color = r_color if is_speaking else tuple(c // 3 for c in r_color)
        draw.ellipse(
            [(x_center - avatar_r, avatar_y - avatar_r),
             (x_center + avatar_r, avatar_y + avatar_r)],
            fill=fill_color,
        )

        # 角色名
        draw.text(
            (x_center, avatar_y),
            r_name,
            font=_find_font(36),
            fill=(255, 255, 255) if is_speaking else (100, 100, 100),
            anchor="mm",
        )

        # 「說話中」標示
        if is_speaking:
            draw.text(
                (x_center, avatar_y + avatar_r + 28),
                "▶ 說話中",
                font=font_small,
                fill=r_color,
                anchor="mm",
            )

    # ── 3. 對話泡泡 ──────────────────────────────────────────────────────────
    bubble_top = 420
    bubble_bottom = 860
    bubble_left = 40
    bubble_right = VIDEO_WIDTH - 40
    bubble_padding = 30

    draw.rounded_rectangle(
        [(bubble_left, bubble_top), (bubble_right, bubble_bottom)],
        radius=20,
        fill=(20, 20, 35),
        outline=role_color,
        width=2,
    )

    # 字幕文字（自動換行）
    font_sub = _find_font(36)
    max_chars = 14  # 每行最多字數（中文）
    wrapped = textwrap.wrap(line, width=max_chars)

    text_block_height = len(wrapped) * 50
    text_y_start = (bubble_top + bubble_bottom) // 2 - text_block_height // 2

    for j, text_line in enumerate(wrapped):
        draw.text(
            (VIDEO_WIDTH // 2, text_y_start + j * 52),
            text_line,
            font=font_sub,
            fill=(240, 240, 240),
            anchor="mm",
        )

    # ── 4. 圖表區 ────────────────────────────────────────────────────────────
    if chart_img is not None:
        chart_area_top = 875
        chart_area_height = VIDEO_HEIGHT - chart_area_top - 20
        chart_area_width = VIDEO_WIDTH - 40

        # 縮放圖表至區域內
        chart_resized = chart_img.copy()
        chart_resized.thumbnail((chart_area_width, chart_area_height), Image.LANCZOS)
        cx = (VIDEO_WIDTH - chart_resized.width) // 2
        cy = chart_area_top + (chart_area_height - chart_resized.height) // 2
        img.paste(chart_resized, (cx, cy))
    else:
        # 底部提示欄
        draw.rectangle([(0, 875), (VIDEO_WIDTH, VIDEO_HEIGHT)], fill=(15, 15, 25))
        font_hint = _find_font(20)
        draw.text(
            (VIDEO_WIDTH // 2, (875 + VIDEO_HEIGHT) // 2),
            "財經新聞短影音",
            font=font_hint,
            fill=(80, 80, 100),
            anchor="mm",
        )

    return np.array(img)


# ── 主合成函式 ────────────────────────────────────────────────────────────────

def compose_video(
    audio_data: list,
    topic: str,
    chart_path: str | None = None,
    output_path: str = "output.mp4",
    end_card_duration: float = 4.0,
) -> str:
    """
    合成完整影片。

    Args:
        audio_data: tts_generator.generate_audio_files() 的回傳值
        topic: 新聞主題（顯示在標題欄）
        chart_path: 圖表 PNG 路徑（可選）
        output_path: 輸出 .mp4 路徑
        end_card_duration: 最後圖表卡片顯示秒數

    Returns:
        output_path
    """
    print(f"\n🎬 開始合成影片（{len(audio_data)} 個對話片段）...")

    # 載入圖表
    chart_img = None
    if chart_path and os.path.exists(chart_path):
        chart_img = Image.open(chart_path).convert("RGB")

    clips = []

    for i, item in enumerate(audio_data):
        print(f"  [{i+1}/{len(audio_data)}] {item['role']}: {item['line'][:20]}...")

        frame = _make_frame(
            role=item["role"],
            line=item["line"],
            topic=topic,
            chart_img=chart_img,
        )

        duration = item["duration"]
        img_clip = (
            ImageClip(frame)
            .with_duration(duration)
        )

        audio_clip = AudioFileClip(item["audio_path"])
        clip_with_audio = img_clip.with_audio(audio_clip)
        clips.append(clip_with_audio)

    # 結尾圖表卡（全螢幕圖表）
    if chart_img is not None:
        print("  📊 加入結尾圖表卡...")
        end_frame = _make_end_card(topic, chart_img)
        end_clip = ImageClip(end_frame).with_duration(end_card_duration)
        clips.append(end_clip)

    # 拼接
    final = concatenate_videoclips(clips, method="compose")

    print(f"\n⏳ 輸出影片中（{final.duration:.1f}s）... 這可能需要幾分鐘")
    final.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )

    print(f"✅ 影片已輸出：{output_path}")
    return output_path


def _make_end_card(topic: str, chart_img: Image.Image) -> np.ndarray:
    """生成結尾圖表全螢幕卡片。"""
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), (8, 8, 15))
    draw = ImageDraw.Draw(img)

    # 頂部標題
    draw.rectangle([(0, 0), (VIDEO_WIDTH, 80)], fill=(20, 20, 40))
    draw.text(
        (VIDEO_WIDTH // 2, 40),
        topic,
        font=_find_font(24),
        fill=(200, 200, 230),
        anchor="mm",
    )

    # 圖表（佔中間大部分）
    chart_area_top = 120
    chart_area_bottom = VIDEO_HEIGHT - 120
    chart_area_w = VIDEO_WIDTH - 40
    chart_area_h = chart_area_bottom - chart_area_top

    chart_resized = chart_img.copy()
    chart_resized.thumbnail((chart_area_w, chart_area_h), Image.LANCZOS)
    cx = (VIDEO_WIDTH - chart_resized.width) // 2
    cy = chart_area_top + (chart_area_h - chart_resized.height) // 2
    img.paste(chart_resized, (cx, cy))

    # 底部提示
    draw.text(
        (VIDEO_WIDTH // 2, VIDEO_HEIGHT - 50),
        "數據僅供參考，投資有風險",
        font=_find_font(18),
        fill=(80, 80, 100),
        anchor="mm",
    )

    return np.array(img)


if __name__ == "__main__":
    # 測試（需要先跑 tts_generator 產生音檔）
    print("請透過 main.py 執行完整流程。")
