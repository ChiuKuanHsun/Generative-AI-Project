"""
Chart generator — multi-type dispatcher
Types: dashboard (price cards), line, candlestick, bar
Data: CoinGecko free API for prices; Claude (no web_search) for asset ID resolution;
      Claude web_search only for dashboard card values.

Install: pip install anthropic plotly kaleido requests
"""
from __future__ import annotations

import anthropic
import json
import re
import time
import requests
from datetime import datetime, timezone
from typing import Literal

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Palette — one place to tweak colors ──────────────────────────────────────
CHART_COLORS: dict = {
    "up":          "#00ff88",
    "down":        "#ff3355",
    "bg":          "#111111",
    "grid":        "rgba(255,255,255,0.08)",
    "text":        "#cccccc",
    "title":       "#ffffff",
    "lines":       ["#00ff88", "#3399ff", "#ff9933", "#cc44ff"],  # multi-series
    "glow_alpha":  0.2,
    "vol_alpha":   0.7,
}

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


# ── JSON utility ──────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
    if match:
        return json.loads(match.group(1))
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON found: {text[:200]}")
    raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        last = raw.rfind("},")
        if last == -1:
            raise
        return json.loads(raw[:last + 1] + "\n  ]\n}")


# ── Shared dark-theme layout ──────────────────────────────────────────────────

def _base_layout(title: str, width: int = 1280, height: int = 720) -> dict:
    """Base Plotly layout dict shared by all renderers."""
    return dict(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=20, color=CHART_COLORS["title"]),
            x=0.5, xanchor="center",
        ),
        template="plotly_dark",
        paper_bgcolor=CHART_COLORS["bg"],
        plot_bgcolor=CHART_COLORS["bg"],
        font=dict(color=CHART_COLORS["text"]),
        width=width,
        height=height,
        margin=dict(l=60, r=60, t=70, b=50),
    )


# ── CoinGecko helpers ─────────────────────────────────────────────────────────

def _cg_get(path: str, params: dict | None = None) -> any:
    """GET from CoinGecko free API with one rate-limit retry."""
    url = f"{COINGECKO_BASE}{path}"
    headers = {"accept": "application/json"}
    r = requests.get(url, params=params, timeout=15, headers=headers)
    if r.status_code == 429:
        print("  CoinGecko rate limited — waiting 30s...")
        time.sleep(30)
        r = requests.get(url, params=params, timeout=15, headers=headers)
    r.raise_for_status()
    return r.json()


_DEFAULT_COINS = ["bitcoin", "ethereum", "solana", "ripple", "binancecoin",
                  "cardano", "dogecoin", "polkadot"]

def _resolve_coin_ids(topic: str, max_coins: int = 4,
                      fallback: bool = True) -> list[str]:
    """Ask Claude (no web_search) to identify CoinGecko coin IDs for the topic.
    Falls back to top-8 popular coins when Claude returns [] and fallback=True."""
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        system=(
            f"Return ONLY a JSON array of up to {max_coins} CoinGecko coin ID slugs "
            "(lowercase, e.g. \"bitcoin\", \"ethereum\", \"solana\"). "
            "If the topic covers stocks or non-crypto assets, return []. "
            "For vague topics like 'top winners/losers/compare', return the 4-8 most "
            "liquid coins: bitcoin, ethereum, solana, ripple, binancecoin, etc. "
            "Output the JSON array only — no other text."
        ),
        messages=[{"role": "user", "content": f"Identify coins for: {topic}"}],
    )
    m = re.search(r"\[.*?\]", resp.content[0].text.strip(), re.DOTALL)
    if m:
        try:
            ids = json.loads(m.group(0))[:max_coins]
            if ids:
                return ids
        except Exception:
            pass
    if fallback:
        return _DEFAULT_COINS[:max_coins]
    return []


def _cg_sparkline(coin_id: str, days: int = 7) -> tuple[list[float], list[str]]:
    """Return (prices, iso_dates) from hourly market_chart."""
    data = _cg_get(
        f"/coins/{coin_id}/market_chart",
        {"vs_currency": "usd", "days": days, "interval": "hourly"},
    )
    prices = [p[1] for p in data["prices"]]
    dates  = [
        datetime.fromtimestamp(p[0] / 1000, tz=timezone.utc).isoformat()
        for p in data["prices"]
    ]
    return prices, dates


def _cg_ohlc_with_volume(coin_id: str, days: int = 14) -> list[dict]:
    """
    Fetch 4-hourly OHLC and merge best-effort hourly volumes.
    Returns list of {t, o, h, l, c, v}.
    """
    ohlc_raw = _cg_get(f"/coins/{coin_id}/ohlc", {"vs_currency": "usd", "days": days})
    time.sleep(0.5)
    chart_raw = _cg_get(
        f"/coins/{coin_id}/market_chart",
        {"vs_currency": "usd", "days": days, "interval": "hourly"},
    )
    # Round to nearest hour in ms
    vol_map: dict[int, float] = {
        round(v[0] / 3_600_000) * 3_600_000: v[1]
        for v in chart_raw.get("total_volumes", [])
    }
    result = []
    for ts_ms, o, h, l, c in ohlc_raw:
        vol = 0.0
        for off_h in range(-4, 5):
            key = round(ts_ms / 3_600_000 + off_h) * 3_600_000
            if key in vol_map:
                vol = vol_map[key]
                break
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        result.append({"t": dt.isoformat(), "o": o, "h": h, "l": l, "c": c, "v": vol})
    return result


def _cg_markets(coin_ids: list[str]) -> list[dict]:
    """Current price + 24h change for a list of coin IDs."""
    return _cg_get("/coins/markets", {
        "vs_currency": "usd",
        "ids":         ",".join(coin_ids),
        "order":       "market_cap_desc",
        "per_page":    len(coin_ids),
        "page":        1,
    })


# ── Chart-type inference ──────────────────────────────────────────────────────

def _infer_chart_type(topic: str) -> str:
    t = topic.lower()
    if any(k in t for k in ["over time", "past week", "7 day", "7-day", "trend",
                              "history", "historical", "performance over"]):
        return "line"
    if any(k in t for k in ["ohlc", "candlestick", "candle", "technical", "chart analysis"]):
        return "candlestick"
    if any(k in t for k in ["compare", "comparison", "winners", "losers",
                              "ranking", " vs ", "versus", "top ", "best", "worst"]):
        return "bar"
    return "dashboard"


# ── Per-type fetch functions ──────────────────────────────────────────────────

_DASHBOARD_PROMPT = """You are a financial data analyst. Based on the given news topic,
search and extract the latest key numerical indicators.

Output ONLY the following JSON format, no other text:
{
  "title": "chart title (short)",
  "items": [
    {
      "label": "indicator name (e.g. BTC)",
      "value": numeric value (float),
      "change_pct": percent change (float, -2.3 means -2.3%),
      "unit": "unit (e.g. USD, %)"
    }
  ]
}
Rules: max 4 indicators; value and change_pct must be numbers."""


def _fetch_dashboard(topic: str, client: anthropic.Anthropic) -> dict:
    """Claude web_search for live price cards + optional CoinGecko sparklines."""
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        system=_DASHBOARD_PROMPT,
        messages=[{"role": "user", "content": f"Search and summarize key indicators for: {topic}"}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    data = _extract_json(raw)
    print(f"  Got {len(data['items'])} indicators: {[i['label'] for i in data['items']]}")

    # Best-effort: attach 7-day sparkline prices to each item
    try:
        coin_ids = _resolve_coin_ids(topic)
        if coin_ids:
            markets  = _cg_markets(coin_ids)
            sym_to_id = {m["symbol"].upper(): m["id"] for m in markets}
            for item in data.get("items", []):
                cid = sym_to_id.get(item["label"].upper().replace("$", ""))
                if cid:
                    item["sparkline"], _ = _cg_sparkline(cid, days=7)
                    time.sleep(0.3)
    except Exception:
        pass  # sparklines are cosmetic; never block the render

    return data


def _fetch_line(topic: str) -> dict:
    """7-day hourly prices from CoinGecko.
    Shape: {title, assets: [{label, prices, dates}], normalize}
    """
    coin_ids = _resolve_coin_ids(topic, max_coins=4)
    if not coin_ids:
        raise ValueError("No crypto assets identified for line chart")

    markets   = _cg_markets(coin_ids)
    label_map = {m["id"]: m["symbol"].upper() for m in markets}
    normalize = any(k in topic.lower() for k in ["compare", "vs", "versus",
                                                   "relative", "normalized"])
    assets = []
    for cid in coin_ids:
        prices, dates = _cg_sparkline(cid, days=7)
        assets.append({"label": label_map.get(cid, cid.upper()),
                        "prices": prices, "dates": dates})
        time.sleep(0.5)

    head   = " & ".join(a["label"] for a in assets[:3])
    suffix = " (+ more)" if len(assets) > 3 else ""
    return {"title": f"{head}{suffix} — 7-Day Price", "assets": assets, "normalize": normalize}


def _fetch_candlestick(topic: str) -> dict:
    """14-day 4h OHLC + volume from CoinGecko.
    Shape: {title, label, ohlc: [{t, o, h, l, c, v}]}
    """
    coin_ids = _resolve_coin_ids(topic, max_coins=1)
    if not coin_ids:
        raise ValueError("No crypto asset identified for candlestick chart")

    cid     = coin_ids[0]
    markets = _cg_markets([cid])
    label   = markets[0]["symbol"].upper() if markets else cid.upper()
    ohlc    = _cg_ohlc_with_volume(cid, days=14)
    return {"title": f"{label} — 14-Day OHLC", "label": label, "ohlc": ohlc}


def _fetch_bar(topic: str) -> dict:
    """24h % change across multiple assets from CoinGecko.
    Shape: {title, items: [{label, value, change_pct}], sort_desc}
    """
    coin_ids = _resolve_coin_ids(topic, max_coins=8)
    if not coin_ids:
        raise ValueError("No crypto assets identified for bar chart")

    markets = _cg_markets(coin_ids)
    items   = [
        {
            "label":      m["symbol"].upper(),
            "value":      m["current_price"],
            "change_pct": m.get("price_change_percentage_24h") or 0.0,
        }
        for m in markets
    ]
    sort_desc = any(k in topic.lower() for k in ["winners", "top", "best", "gainers"])
    return {"title": "24h Performance", "items": items, "sort_desc": sort_desc}


# ── Per-type render functions ─────────────────────────────────────────────────

def render_dashboard(data: dict, output_path: str = "chart.png",
                     width: int = 1280, height: int = 720) -> str:
    """Price indicator cards — plain Figure so indicator domains apply to the full canvas."""
    items = data.get("items", [])
    if not items:
        raise ValueError("Chart data is empty")

    n   = len(items)
    fig = go.Figure()  # no make_subplots — indicator domain is relative to whole figure

    for i, item in enumerate(items, 1):
        change    = item.get("change_pct", 0)
        ref_value = item["value"] / (1 + change / 100) if change != -100 else item["value"]
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=item["value"],
            delta={
                "reference":   ref_value,
                "valueformat": ".2f",
                "increasing":  {"color": CHART_COLORS["up"]},
                "decreasing":  {"color": CHART_COLORS["down"]},
                "suffix":      "%",
                "relative":    True,
            },
            title={
                "text": (
                    f"<b>{item['label']}</b><br>"
                    f"<span style='font-size:0.65em;color:#aaaaaa'>{item['unit']}</span>"
                ),
                "font": {"size": 18},
            },
            number={"valueformat": ",.2f", "font": {"size": 28, "color": "white"}},
            domain={"x": [(i - 1) / n + 0.02, i / n - 0.02], "y": [0.25, 0.75]},
        ))

    layout = _base_layout(data["title"], width, height)
    layout.update(margin=dict(l=10, r=10, t=60, b=10))
    fig.update_layout(layout)
    fig.write_image(output_path, scale=1.5)
    print(f"  Dashboard chart saved: {output_path}")
    return output_path


def render_line(data: dict, output_path: str = "chart.png",
                width: int = 1280, height: int = 720) -> str:
    """Multi-asset line chart with per-series glow duplicate."""
    assets    = data["assets"]
    normalize = data.get("normalize", False)
    fig       = go.Figure()

    for i, asset in enumerate(assets):
        color  = CHART_COLORS["lines"][i % len(CHART_COLORS["lines"])]
        dates  = asset["dates"]
        prices = asset["prices"][:]

        if normalize and prices:
            base   = prices[0]
            prices = [p / base * 100 for p in prices]

        # Glow trace first so it renders underneath
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            line=dict(color=color, width=8),
            opacity=CHART_COLORS["glow_alpha"],
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            name=asset["label"],
            line=dict(color=color, width=3),
        ))

    layout = _base_layout(data["title"], width, height)
    layout.update(
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
        xaxis=dict(gridcolor=CHART_COLORS["grid"], showgrid=True, nticks=5),
        yaxis=dict(
            side="right",
            gridcolor=CHART_COLORS["grid"],
            showgrid=True,
            **({"title": "Indexed (base = 100)"} if normalize else {}),
        ),
        margin=dict(l=20, r=80, t=70, b=50),
    )
    fig.update_layout(layout)
    fig.write_image(output_path, scale=1.5)
    print(f"  Line chart saved: {output_path}")
    return output_path


def render_candlestick(data: dict, output_path: str = "chart.png",
                       width: int = 1280, height: int = 720) -> str:
    """OHLC candlestick with volume subplot sharing the x-axis."""
    ohlc   = data["ohlc"]
    dates  = [r["t"] for r in ohlc]
    opens  = [r["o"] for r in ohlc]
    highs  = [r["h"] for r in ohlc]
    lows   = [r["l"] for r in ohlc]
    closes = [r["c"] for r in ohlc]
    vols   = [r.get("v", 0) for r in ohlc]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.03,
    )

    fig.add_trace(go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes,
        name=data["label"],
        increasing=dict(fillcolor=CHART_COLORS["up"],
                        line=dict(color=CHART_COLORS["up"],   width=1)),
        decreasing=dict(fillcolor=CHART_COLORS["down"],
                        line=dict(color=CHART_COLORS["down"], width=1)),
    ), row=1, col=1)

    vol_colors = [
        CHART_COLORS["up"] if c >= o else CHART_COLORS["down"]
        for o, c in zip(opens, closes)
    ]
    fig.add_trace(go.Bar(
        x=dates, y=vols,
        name="Volume",
        marker=dict(color=vol_colors, opacity=CHART_COLORS["vol_alpha"]),
    ), row=2, col=1)

    layout = _base_layout(data["title"], width, height)
    layout.update(
        xaxis_rangeslider_visible=False,
        xaxis2=dict(gridcolor=CHART_COLORS["grid"]),
        yaxis=dict(gridcolor=CHART_COLORS["grid"], side="right"),
        yaxis2=dict(gridcolor=CHART_COLORS["grid"], showticklabels=False),
        showlegend=False,
    )
    fig.update_layout(layout)
    fig.write_image(output_path, scale=1.5)
    print(f"  Candlestick chart saved: {output_path}")
    return output_path


def render_bar(data: dict, output_path: str = "chart.png",
               width: int = 1280, height: int = 720) -> str:
    """Horizontal bar chart sorted by 24h % change, color-coded red/green."""
    items  = sorted(data["items"], key=lambda x: x["change_pct"])  # ascending
    labels = [x["label"]      for x in items]
    values = [x["change_pct"] for x in items]
    colors = [CHART_COLORS["up"] if v >= 0 else CHART_COLORS["down"] for v in values]
    texts  = [f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%" for v in values]

    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker_color=colors,
        text=texts,
        textposition="outside",
        textfont=dict(color=CHART_COLORS["text"], size=13),
        cliponaxis=False,
    ))

    layout = _base_layout(data["title"], width, height)
    layout.update(
        xaxis=dict(
            gridcolor=CHART_COLORS["grid"],
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.3)",
            zerolinewidth=1,
            ticksuffix="%",
        ),
        yaxis=dict(showline=False, gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=80, r=110, t=70, b=50),
        showlegend=False,
    )
    fig.update_layout(layout)
    fig.write_image(output_path, scale=1.5)
    print(f"  Bar chart saved: {output_path}")
    return output_path


# ── Dispatcher tables ─────────────────────────────────────────────────────────

_FETCHERS = {
    "dashboard":   lambda topic, client: _fetch_dashboard(topic, client),
    "line":        lambda topic, _:      _fetch_line(topic),
    "candlestick": lambda topic, _:      _fetch_candlestick(topic),
    "bar":         lambda topic, _:      _fetch_bar(topic),
}

_RENDERERS = {
    "dashboard":   render_dashboard,
    "line":        render_line,
    "candlestick": render_candlestick,
    "bar":         render_bar,
}


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_chart_data(
    topic: str,
    chart_type: Literal["dashboard", "line", "candlestick", "bar", "auto"] = "auto",
    retries: int = 2,
) -> tuple[dict, str]:
    """
    Fetch chart data for the topic.
    Returns (data_dict, resolved_chart_type).
    Falls back to 'dashboard' if a specialised type fails.
    """
    resolved = _infer_chart_type(topic) if chart_type == "auto" else chart_type
    client   = anthropic.Anthropic()
    fetcher  = _FETCHERS.get(resolved, _FETCHERS["dashboard"])

    last_err: Exception | None = None
    for attempt in range(1 + retries):
        if attempt > 0:
            print(f"  Retrying (attempt {attempt})...")
            time.sleep(2)
        try:
            return fetcher(topic, client), resolved
        except (json.JSONDecodeError, ValueError, anthropic.APIError) as e:
            last_err = e
        except Exception as e:
            last_err = e
            if resolved != "dashboard":
                print(f"  {resolved} chart failed ({e}), falling back to dashboard...")
                try:
                    return _fetch_dashboard(topic, client), "dashboard"
                except Exception as e2:
                    last_err = e2

    raise RuntimeError(f"Chart fetch failed after {retries} retries: {last_err}")


def render_chart(
    data: dict,
    chart_type: str,
    output_path: str = "chart.png",
    width: int = 1280,
    height: int = 720,
) -> str:
    """Dispatch to the correct renderer."""
    return _RENDERERS.get(chart_type, render_dashboard)(data, output_path, width, height)


def generate_chart(
    topic: str,
    output_path: str = "chart.png",
    width: int = 1280,
    chart_type: Literal["dashboard", "line", "candlestick", "bar", "auto"] = "auto",
) -> str:
    """One-shot: infer type → fetch → render → save PNG."""
    data, resolved = fetch_chart_data(topic, chart_type)
    return render_chart(data, resolved, output_path=output_path, width=width)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv("api-key.env")

    import pathlib
    out_dir = pathlib.Path("test-results")
    out_dir.mkdir(exist_ok=True)

    tests = [
        ("dashboard",   "Bitcoin and Ethereum latest prices"),
        ("line",        "Bitcoin 7-day performance history"),
        ("candlestick", "Bitcoin technical analysis chart"),
        ("bar",         "Compare top crypto winners today"),
        ("auto",        "Crypto market overview"),
        # Non-crypto — all resolve to dashboard via Claude web_search
        ("auto",        "Nvidia Apple Tesla stock prices today"),
        ("auto",        "Today's AI industry major news and breakthroughs"),
        ("auto",        "Global financial markets and economic news today"),
    ]
    for ct, topic in tests:
        out = str(out_dir / f"test_{ct}_{topic[:20].replace(' ', '_')}.png")
        generate_chart(topic, output_path=out, chart_type=ct)
        print(f"  -> {out}\n")
        time.sleep(2)  # avoid CoinGecko rate limiting between sequential test runs
