"""
圖表生成器
Claude API 搜尋數據 → Plotly 固定模板 → chart.png

安裝：pip install anthropic plotly kaleido
"""

import anthropic
import json
import re
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _extract_json(text: str) -> dict:
    """從可能含有多餘文字或 markdown 的回應中提取 JSON。"""
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if match:
        return json.loads(match.group(1))
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"找不到 JSON 內容，原始回應：{text[:200]}")


CHART_SYSTEM_PROMPT = """你是財經數據分析師。根據剛才的新聞主題，搜尋並提取最新的關鍵數字指標。

嚴格只輸出以下 JSON 格式，不要有其他文字：
{
  "title": "圖表標題（簡短，例如：今日加密貨幣行情）",
  "items": [
    {
      "label": "指標名稱（例如：BTC）",
      "value": 數值（浮點數，例如：85234.5）,
      "change_pct": 漲跌幅百分比（浮點數，例如：-2.3 表示跌2.3%）,
      "unit": "單位（例如：USD、TWD、%）"
    }
  ]
}

規則：
- 最多 4 個指標
- value 和 change_pct 必須是數字，不是字串
- 如果找不到確切數字，用最接近的估計值"""


def fetch_chart_data(topic: str, retries: int = 2) -> dict:
    """
    呼叫 Claude API 搜尋圖表數據，回傳結構化 JSON。
    失敗時自動重試 retries 次。
    """
    client = anthropic.Anthropic()
    print(f"\n📊 搜尋圖表數據：{topic[:30]}...")

    last_err = None
    for attempt in range(1 + retries):
        if attempt > 0:
            print(f"  ↩️ 重試第 {attempt} 次...")
            time.sleep(2)
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                system=CHART_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"搜尋並整理：{topic} 的關鍵數字指標"}],
            )

            text_blocks = [b for b in response.content if b.type == "text"]
            raw = "".join(b.text for b in text_blocks).strip()
            data = _extract_json(raw)
            print(f"  ✅ 取得 {len(data['items'])} 個指標：{[i['label'] for i in data['items']]}")
            return data

        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
        except anthropic.APIError as e:
            last_err = e

    raise RuntimeError(f"圖表數據抓取失敗（已重試 {retries} 次）：{last_err}")


def render_chart(data: dict, output_path: str = "chart.png", width: int = 720, height: int = 300) -> str:
    """
    使用 Plotly 固定深色模板渲染圖表，輸出為 PNG。

    Args:
        data: fetch_chart_data() 回傳的 dict
        output_path: 輸出圖片路徑
        width: 圖片寬度（配合影片寬度）
        height: 圖片高度

    Returns:
        output_path
    """
    items = data.get("items", [])
    if not items:
        raise ValueError("圖表數據為空")

    n = len(items)
    fig = make_subplots(rows=1, cols=n, subplot_titles=None)

    for i, item in enumerate(items, 1):
        change = item.get("change_pct", 0)
        ref_value = item["value"] / (1 + change / 100) if change != -100 else item["value"]

        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=item["value"],
                delta={
                    "reference": ref_value,
                    "valueformat": ".2f",
                    "increasing": {"color": "#00e676"},
                    "decreasing": {"color": "#ff5252"},
                    "suffix": "%",
                    "relative": True,
                },
                title={
                    "text": f"<b>{item['label']}</b><br>"
                            f"<span style='font-size:0.65em;color:#aaaaaa'>{item['unit']}</span>",
                    "font": {"size": 18},
                },
                number={
                    "valueformat": ",.2f",
                    "font": {"size": 28, "color": "white"},
                },
                domain={
                    "x": [(i - 1) / n + 0.02, i / n - 0.02],
                    "y": [0.1, 0.9],
                },
            )
        )

    fig.update_layout(
        title={
            "text": f"<b>{data['title']}</b>",
            "font": {"size": 16, "color": "#cccccc"},
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_dark",
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font={"color": "white"},
        width=width,
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    fig.write_image(output_path, scale=1.5)
    print(f"  ✅ 圖表已存至：{output_path}")
    return output_path


def generate_chart(topic: str, output_path: str = "chart.png", width: int = 720) -> str:
    """
    一鍵生成：搜尋數據 + 渲染圖表。
    """
    data = fetch_chart_data(topic)
    return render_chart(data, output_path=output_path, width=width)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv("api-key.env")

    path = generate_chart("今日比特幣與以太坊最新價格走勢")
    print(f"圖表輸出：{path}")
