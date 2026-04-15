"""
自動生成財經短影音 - 主程式
完整流程：新聞抓取 → 對話生成 → TTS → 圖表 → 影片合成

使用方式：
    pip install -r requirements.txt
    python main.py
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# 載入 API 金鑰
load_dotenv("api-key.env")

from news_fetcher import fetch_news_and_generate_dialogue, TOPICS
from tts_generator import generate_audio_files
from chart_generator import generate_chart
from video_composer import compose_video

# ── 輸出資料夾設定 ─────────────────────────────────────────────────────────────
def make_output_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"output_{ts}"
    Path(out_dir).mkdir(exist_ok=True)
    return out_dir


def main():
    print("╔══════════════════════════════════════╗")
    print("║   財經新聞短影音自動生成器  v1.0     ║")
    print("╚══════════════════════════════════════╝")

    # ── Step 1: 選擇主題 ──────────────────────────────────────────────────────
    print("\n選擇新聞主題：")
    for k, v in TOPICS.items():
        print(f"  [{k}] {v}")
    print("  [5] 自訂主題")
    print("  [q] 離開")

    choice = input("\n> ").strip()
    if choice == "q":
        return
    if choice == "5":
        topic = input("輸入自訂主題：").strip()
        if not topic:
            print("主題不能是空的！")
            return
    elif choice in TOPICS:
        topic = TOPICS[choice]
    else:
        print("無效選項")
        return

    out_dir = make_output_dir()
    print(f"\n📁 輸出資料夾：{out_dir}/")

    # ── Step 2: 新聞抓取 + 對話生成 ───────────────────────────────────────────
    print("\n" + "─" * 45)
    print("Step 1/4  抓取新聞並生成對話腳本")
    print("─" * 45)

    dialogue_data = fetch_news_and_generate_dialogue(topic)
    dialogue_path = os.path.join(out_dir, "dialogue.json")
    with open(dialogue_path, "w", encoding="utf-8") as f:
        json.dump(dialogue_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 對話腳本已存至：{dialogue_path}")

    # ── Step 3: TTS 語音生成 ──────────────────────────────────────────────────
    print("\n" + "─" * 45)
    print("Step 2/4  生成語音（edge-tts）")
    print("─" * 45)

    audio_dir = os.path.join(out_dir, "audio")
    audio_data = generate_audio_files(dialogue_data["dialogue"], output_dir=audio_dir)

    # ── Step 4: 圖表生成 ──────────────────────────────────────────────────────
    print("\n" + "─" * 45)
    print("Step 3/4  生成圖表（Plotly）")
    print("─" * 45)

    chart_path = os.path.join(out_dir, "chart.png")
    try:
        generate_chart(topic, output_path=chart_path)
    except Exception as e:
        print(f"⚠️ 圖表生成失敗（{e}），跳過圖表，繼續合成影片。")
        chart_path = None

    # ── Step 5: 影片合成 ──────────────────────────────────────────────────────
    print("\n" + "─" * 45)
    print("Step 4/4  合成影片（MoviePy）")
    print("─" * 45)

    video_path = os.path.join(out_dir, "output.mp4")
    compose_video(
        audio_data=audio_data,
        topic=dialogue_data.get("topic", topic),
        chart_path=chart_path,
        output_path=video_path,
    )

    # ── 完成 ──────────────────────────────────────────────────────────────────
    print("\n" + "═" * 45)
    print("🎉 全部完成！")
    print(f"   影片：{video_path}")
    print(f"   腳本：{dialogue_path}")
    if chart_path:
        print(f"   圖表：{chart_path}")
    print("═" * 45)


if __name__ == "__main__":
    main()
