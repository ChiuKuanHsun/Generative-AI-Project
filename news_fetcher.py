"""
Financial news fetcher + dual-character dialogue generator
Uses Claude API with web_search tool.

Usage:
    export ANTHROPIC_API_KEY="your_key_here"
    python news_fetcher.py
"""

import anthropic
import json
import re
import time
from datetime import datetime


def _strip_citations(data: dict) -> dict:
    """Remove <cite/>, <cite id=...>, </cite> tags injected by web_search."""
    pattern = re.compile(r"</?cite[^>]*>", re.IGNORECASE)
    for item in data.get("dialogue", []):
        item["line"] = pattern.sub("", item["line"]).strip()
    if "summary" in data:
        data["summary"] = pattern.sub("", data["summary"]).strip()
    if "topic" in data:
        data["topic"] = pattern.sub("", data["topic"]).strip()
    return data


def _extract_json(text: str) -> dict:
    """Extract JSON from a response that may contain extra text or markdown code blocks."""
    # Try code block first
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
    if match:
        return json.loads(match.group(1))
    # Try full { ... } object
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON found in response: {text[:200]}")
    raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Truncated JSON — trim to last complete dialogue entry and close the structure
        last_complete = raw.rfind('},')
        if last_complete == -1:
            raise ValueError(f"JSON too malformed to recover: {raw[:200]}")
        repaired = raw[:last_complete + 1] + "\n  ]\n}"
        return json.loads(repaired)


TOPICS = {
    "1": "Today's crypto market update including Bitcoin and Ethereum price movements",
    "2": "Today's US tech stocks update including Nvidia, Apple, and Tesla",
    "3": "Today's AI industry major news and breakthroughs",
    "4": "Today's global financial markets and economic news",
}

SYSTEM_PROMPT = """You are a financial news analysis system. Your task is:
1. Search for the latest financial news (today or recent)
2. Write a dialogue between two characters in TRADITIONAL CHINESE (繁體中文 — NEVER 简体):
   - Character A "gugugaga": 資深金融專家，說話直接略帶諷刺，喜歡用數據支撐觀點
   - Character B "meowchan": 投資新手，問題天真，有時說錯被糾正，充滿好奇
3. Keep dialogue natural, punchy, and entertaining — suitable for short-form video
4. 6-8 turns total. Each line MUST be 25 Traditional Chinese characters or fewer
   (parentheses content does NOT count toward the 25-char limit).
5. Output every Chinese character in Traditional form
   (用「資料」不用「数据」, 用「臺灣」不用「台湾」, 用「網絡」不用「网络」).

6. PROPER NOUNS — two rules depending on word shape:

   (a) FULL WORDS / brand names (Bitcoin, Tesla, Apple, OpenAI, Google, Nvidia, Musk, …)
       → write the standard Traditional Chinese name first, with the English original
         in half-width parentheses on FIRST mention only:
            比特幣(Bitcoin)、特斯拉(Tesla)、蘋果(Apple)、開放人工智慧(OpenAI)、
            谷歌(Google)、輝達(Nvidia)、微軟(Microsoft)、馬斯克(Musk)、
            以太幣(Ethereum)、亞馬遜(Amazon)
       → Later mentions in the same dialogue may drop the parentheses.
       → The text inside parentheses appears in captions but is NOT spoken.

   (b) ACRONYMS (all-caps, 2-5 letters: AI, GPT, IPO, ETF, USD, EUR, BTC, NVDA, AAPL, …)
       → write them BARE, exactly as the acronym, NOT inside parentheses.
       → They are read out letter-by-letter by the TTS (e.g. "AI" → "ei-ai",
         "GPT" → "jee-pee-tee"), which is how Mandarin speakers actually pronounce
         these in news media.
       → Examples of correct usage:
            "AI 概念股大漲三成"
            "美股 NVDA 收盤上漲百分之五"
            "IPO 市場今年表現亮眼"
            "GPT 模型升級了"
       → Do NOT wrap acronyms in parentheses — they need to be spoken.

   Edge case: when both forms exist (e.g. company "Nvidia" + ticker "NVDA"),
   write the company as 輝達(Nvidia) on first mention, the ticker bare as NVDA.

Output ONLY the following JSON format, no other text:
{
  "topic": "news headline in Traditional Chinese (parentheses allowed for English)",
  "summary": "one-sentence summary in Traditional Chinese",
  "dialogue": [
    {"role": "gugugaga", "line": "dialogue line in Traditional Chinese", "emotion": "calm|excited|sarcastic|serious"},
    {"role": "meowchan", "line": "dialogue line in Traditional Chinese", "emotion": "confused|surprised|curious|learning"}
  ]
}"""


def fetch_news_and_generate_dialogue(topic: str, retries: int = 2) -> dict:
    """
    Call Claude API with web_search to fetch news and generate dialogue JSON.
    Automatically retries on failure.
    """
    client = anthropic.Anthropic()

    print(f"\nSearching: {topic}")
    print("=" * 50)

    last_err = None
    for attempt in range(1 + retries):
        if attempt > 0:
            print(f"  Retrying (attempt {attempt})...")
            time.sleep(2)
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2000,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"Search and analyze: {topic}"}
                ]
            )

            text_blocks = [b for b in response.content if b.type == "text"]
            if not text_blocks:
                raise ValueError("API returned no text content")

            raw_text = "".join(b.text for b in text_blocks)
            return _strip_citations(_extract_json(raw_text))

        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
        except anthropic.APIError as e:
            last_err = e

    raise RuntimeError(f"News fetch failed after {retries} retries: {last_err}")


def save_dialogue(data: dict, filename: str = None) -> str:
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dialogue_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filename


def print_dialogue(data: dict):
    print(f"\nTopic: {data['topic']}")
    print(f"Summary: {data['summary']}")
    print("\n-- Dialogue --")
    emoji_map = {"gugugaga": "🐧", "meowchan": "🐱"}
    for item in data["dialogue"]:
        emoji = emoji_map.get(item["role"], "💬")
        print(f"\n{emoji} [{item['role']}] ({item.get('emotion', '')})")
        print(f"   {item['line']}")


def main():
    print("╔══════════════════════════════════════╗")
    print("║   Financial News Dialogue Generator  ║")
    print("╚══════════════════════════════════════╝")

    while True:
        print("\nSelect a topic:")
        for k, v in TOPICS.items():
            print(f"  [{k}] {v}")
        print("  [5] Custom topic")
        print("  [q] Quit")

        choice = input("\n> ").strip()

        if choice == "q":
            print("Bye!")
            break
        if choice == "5":
            topic = input("Enter custom topic: ").strip()
            if not topic:
                print("Topic cannot be empty!")
                continue
        elif choice in TOPICS:
            topic = TOPICS[choice]
        else:
            print("Invalid choice")
            continue

        try:
            data = fetch_news_and_generate_dialogue(topic)
            print_dialogue(data)
            save = input("\n\nSave as JSON? (y/N) ").strip().lower()
            if save == "y":
                filename = save_dialogue(data)
                print(f"Saved to: {filename}")
        except json.JSONDecodeError as e:
            print(f"\nJSON parse error: {e}")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
