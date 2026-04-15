"""
財經新聞抓取器 + 雙人對話生成
使用 Claude API + web_search 工具

安裝依賴：
    pip install anthropic

使用方式：
    export ANTHROPIC_API_KEY="your_key_here"
    python news_fetcher.py
"""

import anthropic
import json
import re
import time
from datetime import datetime


def _extract_json(text: str) -> dict:
    """從可能含有多餘文字或 markdown 的回應中提取 JSON。"""
    # 優先嘗試 code block（```json ... ``` 或 ``` ... ```）
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if match:
        return json.loads(match.group(1))
    # fallback：找第一個 { ... } 大物件
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"找不到 JSON 內容，原始回應：{text[:200]}")

TOPICS = {
    "1": "今日加密貨幣市場動態，包含比特幣、以太坊最新價格走勢",
    "2": "今日美股科技股動態，包含輝達、蘋果、特斯拉",
    "3": "今日台股大盤與台積電最新動態",
    "4": "今日 AI 產業最新重大新聞",
}

SYSTEM_PROMPT = """你是一個財經新聞分析系統。你的任務是：
1. 搜尋最新的財經新聞（今日或近期）
2. 整理成兩個角色的對話：
   - 角色A「老王」：資深金融老鳥，說話直接、帶點毒舌，喜歡用數據說話
   - 角色B「小咪」：剛入門的投資小白，問問題很直白，偶爾說錯話被糾正
3. 對話要自然、口語化、有梗，適合短影音格式
4. 每人說 3-5 句話，輪流對話共 6-10 輪

嚴格只輸出以下 JSON 格式，不要有任何其他文字：
{
  "topic": "新聞主題",
  "summary": "一句話摘要這則新聞的核心內容",
  "dialogue": [
    {"role": "老王", "line": "台詞內容", "emotion": "calm|excited|sarcastic|serious"},
    {"role": "小咪", "line": "台詞內容", "emotion": "confused|surprised|curious|learning"}
  ]
}"""


def fetch_news_and_generate_dialogue(topic: str, retries: int = 2) -> dict:
    """
    呼叫 Claude API，用 web_search 搜尋新聞並生成對話 JSON。
    失敗時自動重試 retries 次。
    """
    client = anthropic.Anthropic()

    print(f"\n🔍 搜尋中：{topic}")
    print("=" * 50)

    last_err = None
    for attempt in range(1 + retries):
        if attempt > 0:
            print(f"  ↩️ 重試第 {attempt} 次...")
            time.sleep(2)
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1000,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"請搜尋並分析：{topic}"}
                ]
            )

            text_blocks = [b for b in response.content if b.type == "text"]
            if not text_blocks:
                raise ValueError("API 沒有回傳文字內容")

            raw_text = "".join(b.text for b in text_blocks)
            return _extract_json(raw_text)

        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
        except anthropic.APIError as e:
            last_err = e

    raise RuntimeError(f"新聞抓取失敗（已重試 {retries} 次）：{last_err}")


def save_dialogue(data: dict, filename: str = None) -> str:
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dialogue_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filename


def print_dialogue(data: dict):
    print(f"\n📰 主題：{data['topic']}")
    print(f"💡 摘要：{data['summary']}")
    print("\n── 對話內容 ──")
    emoji_map = {"老王": "👴", "小咪": "🐱"}
    for item in data["dialogue"]:
        emoji = emoji_map.get(item["role"], "💬")
        print(f"\n{emoji} [{item['role']}] ({item.get('emotion', '')})")
        print(f"   {item['line']}")


def main():
    print("╔══════════════════════════════╗")
    print("║   財經新聞抓取 + 對話生成器  ║")
    print("╚══════════════════════════════╝")

    while True:
        print("\n選擇新聞主題：")
        for k, v in TOPICS.items():
            print(f"  [{k}] {v}")
        print("  [5] 自訂主題")
        print("  [q] 離開")

        choice = input("\n> ").strip()

        if choice == "q":
            print("掰掰！")
            break
        if choice == "5":
            topic = input("輸入自訂主題：").strip()
            if not topic:
                print("主題不能是空的！")
                continue
        elif choice in TOPICS:
            topic = TOPICS[choice]
        else:
            print("無效選項")
            continue

        try:
            data = fetch_news_and_generate_dialogue(topic)
            print_dialogue(data)
            save = input("\n\n💾 要存成 JSON 檔嗎？(y/N) ").strip().lower()
            if save == "y":
                filename = save_dialogue(data)
                print(f"✅ 已存至：{filename}")
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON 解析失敗：{e}")
        except Exception as e:
            print(f"\n❌ 錯誤：{e}")


if __name__ == "__main__":
    main()
