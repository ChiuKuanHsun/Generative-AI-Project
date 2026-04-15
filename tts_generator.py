"""
TTS 語音生成器
使用 edge-tts 將對話 JSON 轉換成 MP3 音檔

安裝：pip install edge-tts
"""

import asyncio
import os
from pathlib import Path

try:
    from mutagen.mp3 import MP3
    _HAS_MUTAGEN = True
except ImportError:
    _HAS_MUTAGEN = False


VOICES = {
    "老王": "zh-TW-YunJheNeural",   # 成熟男聲
    "小咪": "zh-TW-HsiaoChenNeural",  # 年輕女聲
}


async def _synthesize_one(text: str, voice: str, output_path: str):
    """合成單句語音並存檔"""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


async def _synthesize_all(items: list, output_dir: str):
    """並行合成所有對話行"""
    tasks = []
    for i, item in enumerate(items):
        voice = VOICES.get(item["role"], VOICES["小咪"])
        path = os.path.join(output_dir, f"line_{i:03d}.mp3")
        tasks.append(_synthesize_one(item["line"], voice, path))
    await asyncio.gather(*tasks)


def get_mp3_duration(path: str) -> float:
    """取得 MP3 檔案時長（秒）"""
    if _HAS_MUTAGEN:
        try:
            audio = MP3(path)
            return audio.info.length
        except Exception:
            pass
    # fallback: 粗估每個字 0.3 秒（用檔案大小估算）
    try:
        size = os.path.getsize(path)
        return max(1.0, size / 16000)  # 128kbps ≈ 16000 bytes/s
    except Exception:
        return 3.0


def generate_audio_files(dialogue: list, output_dir: str = "audio") -> list:
    """
    為每行對話生成 MP3 音檔。

    Args:
        dialogue: 來自 news_fetcher 的 dialogue list
        output_dir: 音檔輸出資料夾

    Returns:
        list of dicts:
        {
            "role": str,
            "line": str,
            "emotion": str,
            "audio_path": str,
            "duration": float,  # 秒
        }
    """
    Path(output_dir).mkdir(exist_ok=True)
    print(f"\n🎙️ 開始生成語音 ({len(dialogue)} 行)...")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 已有 event loop（如 Jupyter）：用 nest_asyncio 或新執行緒
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(asyncio.run, _synthesize_all(dialogue, output_dir)).result()
    else:
        asyncio.run(_synthesize_all(dialogue, output_dir))

    results = []
    for i, item in enumerate(dialogue):
        path = os.path.join(output_dir, f"line_{i:03d}.mp3")
        duration = get_mp3_duration(path)
        results.append({
            "role": item["role"],
            "line": item["line"],
            "emotion": item.get("emotion", ""),
            "audio_path": path,
            "duration": duration,
        })
        print(f"  ✅ [{item['role']}] {item['line'][:20]}... ({duration:.1f}s)")

    total = sum(r["duration"] for r in results)
    print(f"\n🎧 語音生成完成，總時長：{total:.1f} 秒")
    return results


if __name__ == "__main__":
    # 測試用
    from dotenv import load_dotenv
    load_dotenv("api-key.env")

    test_dialogue = [
        {"role": "老王", "line": "比特幣今天又跌了，跌得我心裡毛毛的。", "emotion": "sarcastic"},
        {"role": "小咪", "line": "老王，比特幣跌是什麼意思啊？", "emotion": "confused"},
        {"role": "老王", "line": "就是你的口袋變薄了，懂嗎？", "emotion": "serious"},
    ]

    results = generate_audio_files(test_dialogue, output_dir="audio_test")
    for r in results:
        print(f"{r['role']}: {r['audio_path']} ({r['duration']:.1f}s)")
