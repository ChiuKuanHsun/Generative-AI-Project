# Financial Short Video Generator

Auto-generates vertical short-form videos (720×1280) covering daily financial
news as a Traditional Chinese (繁體中文) dialogue between two anime characters
— **gugugaga** (the experienced analyst) and **meowchan** (the curious
beginner). News fetch, dialogue scripting, voice synthesis, multi-chart
rendering, news image search, video composition, and optional YouTube upload
all happen in one pipeline run.

## Pipeline

```
news → dialogue (LLM) → TTS → charts (4×) → news images → compose → YouTube
```

| Stage | Module | Engine |
|------|--------|--------|
| News + Traditional Chinese dialogue | `news_fetcher.py` | Anthropic Claude API w/ web_search |
| Voice synthesis (anime, 1.3× speed) | `tts_generator.py` | VITS Umamusume (HuggingFace, free) |
| Multi-chart set (dashboard / line / candlestick / bar) | `chart_generator.py` | Plotly + CoinGecko (crypto) + yfinance (stocks/indices) |
| News image search + ranking + quality filter | `image_agent.py` | Wikimedia / Guardian / NYT + Claude Haiku |
| Video composition (karaoke subs, interleaved visuals) | `video_composer.py` | MoviePy 2.1 + Pillow |
| YouTube Shorts upload | `youtube_uploader.py` | YouTube Data API v3 (OAuth) |
| Orchestration / CLI | `main.py` | — |

Output is a vertical 720×1280 MP4 sized for TikTok / YouTube Shorts /
Instagram Reels. Layout:

- **Top bar** — topic title
- **Top-left corner** — `gugugaga` sprite (only when speaking)
- **Center box** — visual rotated per dialogue line; charts and news photos
  are interleaved (chart, photo, chart, photo …) so charts appear throughout
- **Karaoke subtitle band** — speakable units highlight in gold, English-in-
  parentheses annotations stay in muted blue (caption-only, never voiced)
- **Bottom-right corner** — `meowchan` sprite (only when speaking)

Total runtime targets ~30 seconds (no separate end card; the chart appears
within the dialogue rotation).

### Asset class detection

Claude Haiku classifies the topic up front (`crypto` / `stock` / `other`)
and picks tickers — CoinGecko coin IDs for crypto, Yahoo Finance tickers for
stocks/indices. Defaults: tech topics → `NVDA / MSFT / GOOGL / META / AAPL`,
global markets → `^GSPC / ^DJI / ^IXIC / ^N225`.

## Proper-noun handling

The dialogue prompt enforces two rules so on-screen captions stay readable
and TTS pronounces brands correctly:

- **Brand / company names** are written as 中文(English) on first mention —
  e.g. `特斯拉(Tesla)`, `輝達(Nvidia)`. Captions show both, but the TTS
  pipeline strips parenthesized text so only the Chinese is voiced.
- **Acronyms** (AI, GPT, ETF, NVDA, BTC, IPO …) stay bare and are voiced
  letter-by-letter — the way Mandarin news anchors actually pronounce them.

`tts_generator.py` also has a fallback table that translates any unwrapped
brand name (Bitcoin, Tesla, Nvidia, OpenAI, Musk, Nasdaq, …) to its
Traditional Chinese equivalent before synthesis, in case the LLM forgets the
parenthesized form.

## Requirements

- Python 3.10+
- An Anthropic API key (news + dialogue + image ranking + asset detection)
- Internet access for the HuggingFace VITS Space, CoinGecko, and Yahoo
  Finance
- Optional: Guardian Open API key (free, recommended for image variety) and
  NYT Article Search key
- Optional: Google OAuth client (`client_secret.json`) for YouTube upload
- Windows CJK font (auto-detected) or set `FONT_PATH` env var on other OSes
- Background video at `backgrounds/<name>.mp4` (any `.mp4/.mov/.webm/.mkv`
  in `backgrounds/` is auto-picked, or override with `BG_VIDEO_PATH`)
- Character sprites at `characters/character1.png` (meowchan) and
  `characters/character2.png` (gugugaga)

Asset folders (`backgrounds/`, `characters/`, `experiment-assets/`) are
gitignored — supply your own.

## Setup

```bash
pip install -r requirements.txt
```

Create `api-key.env` in the project root:

```ini
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
GUARDIAN_API_KEY=your-guardian-key   # optional, https://bonobo.capi.gutools.co.uk/register/developer
NYT_API_KEY=your-nyt-key             # optional, https://developer.nytimes.com/get-started
```

For YouTube upload (optional):

1. Create a **Desktop OAuth client** in
   [Google Cloud Console](https://console.cloud.google.com/) and download
   the JSON as `client_secret.json` in the project root.
2. Enable the YouTube Data API v3 for that project.
3. First upload opens a browser for consent; the access token is cached in
   `youtube_token.json`. Both files are gitignored.

## Usage

```bash
python main.py
```

Menu:

| Key | Mode |
|-----|------|
| `1`–`4` | Generate from a preset topic (crypto, US tech, AI, global markets) |
| `5` | Custom topic |
| `t` | Test mode — reuse cached dialogue + audio (no API cost) |
| `r` | Rebuild mode — regenerate audio from cached dialogue |
| `v` | Configure VITS speakers for each character |
| `q` | Quit |

Output lands in a timestamped `output_YYYYMMDD_HHMMSS/` folder containing:

```
output.mp4
dialogue.json
chart_dashboard.png
chart_line.png
chart_candlestick.png
chart_bar.png
news_image_00.png … news_image_NN.png
audio/line_000.mp3 … line_NNN.mp3
```

After composing, the CLI prompts to upload as a YouTube Short (privacy
defaults to `public`; choose `unlisted`/`private` to override).

### Voice configuration

Press `v` on first run to pick a VITS speaker for each character. The
HuggingFace Space exposes hundreds of anime/game voices; selections are saved
to `voice_config.json` and applied automatically on subsequent runs.

> **Note** — VITS-Umamusume's tokenizer expects Simplified Chinese, but the
> dialogue and on-screen subtitles are Traditional. The TTS pipeline strips
> caption-only `(English)` annotations, applies fallback brand translations,
> then converts Traditional → Simplified before synthesis (via `zhconv`).
> Captions stay Traditional throughout.

## Reliability features

- **CoinGecko** — per-process throttle (≥1.5 s between calls) plus 429
  back-off honoring `Retry-After`; sparkline + markets results are cached
  per process so a multi-chart run hits each endpoint once.
- **Wikimedia / Guardian / NYT** — per-host throttling with the
  identifiable Wikimedia user-agent string, plus 429/5xx retry.
- **Image quality filter** — pure-black, pure-white, and near-uniform
  candidates are rejected via luminance mean/stddev before download.
- **Chart fallback** — if Claude classifies the topic as `other`, the
  pipeline falls back to default tech tickers (AI/tech keywords) or major
  indices so a video always gets visual data charts.

## Project layout

```
.
├── main.py                  # CLI orchestration (5-step pipeline + upload prompt)
├── news_fetcher.py          # Claude web_search → Traditional Chinese dialogue
├── tts_generator.py         # VITS Umamusume — Gradio 5 SSE @ 1.3×, brand subs
├── chart_generator.py       # Plotly chart set: dashboard + line + candle + bar
├── image_agent.py           # Multi-source news image search + quality filter
├── video_composer.py        # MoviePy composition (karaoke, interleaved visuals)
├── youtube_uploader.py      # YouTube Data API v3 OAuth upload
├── voice_config.json        # Persisted speaker assignments
├── requirements.txt
├── backgrounds/             # (gitignored) background loops
├── characters/              # (gitignored) character sprites
├── experiment-assets/       # (gitignored) cached test inputs + news images
└── output_*/                # (gitignored) generated runs
```
