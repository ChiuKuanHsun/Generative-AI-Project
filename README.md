# Financial Short Video Generator

Auto-generates vertical short-form videos (720×1280) covering daily financial
news as a Traditional Chinese (繁體中文) dialogue between two anime characters
— **gugugaga** (the experienced analyst) and **meowchan** (the curious
beginner). News fetch, script writing, voice synthesis, news image search,
chart rendering, and video composition all happen in one pipeline run.

## Pipeline

```
news fetch → dialogue (LLM) → TTS → news images → chart → video composition
```

| Stage | Module | Engine |
|------|--------|--------|
| News + Traditional Chinese dialogue | `news_fetcher.py` | Anthropic Claude API w/ web_search |
| Voice synthesis (anime, 1.3× speed) | `tts_generator.py` | VITS Umamusume (HuggingFace Space, free) |
| News image search + ranking | `image_agent.py` | Wikimedia / Guardian / NYT + Claude Haiku |
| Price chart | `chart_generator.py` | Plotly + Kaleido |
| Video composition (karaoke subs) | `video_composer.py` | MoviePy 2.1 + Pillow |
| Orchestration / CLI | `main.py` | — |

Output is a vertical 720×1280 MP4 sized for TikTok / YouTube Shorts /
Instagram Reels. Layout:

- **Top bar** — topic title
- **Top-left corner** — `gugugaga` sprite (only when speaking)
- **Center box** — news image, rotated per dialogue line, falls back to chart
- **Karaoke subtitle band** — character-level highlight in gold, synced to TTS
- **Bottom-right corner** — `meowchan` sprite (only when speaking)
- **End card** — full chart with disclaimer

Total runtime targets ~30–35 seconds including a 4-second chart end card.

## Requirements

- Python 3.10+
- An Anthropic API key (news + dialogue + image ranking)
- Internet access for the HuggingFace VITS Space
- Optional: Guardian Open API key (free, recommended for image variety) and
  NYT Article Search key
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

Output lands in a timestamped `output_YYYYMMDD_HHMMSS/` folder containing
`output.mp4`, `dialogue.json`, `chart.png`, `news_image_*.png`, and per-line
audio files.

### Voice configuration

Press `v` on first run to pick a VITS speaker for each character. The
HuggingFace Space exposes hundreds of anime/game voices; selections are saved
to `voice_config.json` and applied automatically on subsequent runs.

> **Note** — VITS-Umamusume's tokenizer expects Simplified Chinese, but the
> dialogue and on-screen subtitles are Traditional. The TTS pipeline converts
> Traditional → Simplified before synthesis (via `zhconv`); captions stay
> Traditional throughout.

## Project layout

```
.
├── main.py                  # CLI orchestration (5-step pipeline)
├── news_fetcher.py          # Claude web_search → Traditional Chinese dialogue
├── tts_generator.py         # VITS Umamusume — Gradio 5 SSE flow @ 1.3×
├── image_agent.py           # Multi-source news image search + Claude ranking
├── chart_generator.py       # Plotly chart rendering
├── video_composer.py        # MoviePy composition (karaoke subs, corner sprites)
├── voice_config.json        # Persisted speaker assignments
├── requirements.txt
├── backgrounds/             # (gitignored) background loops
├── characters/              # (gitignored) character sprites
├── experiment-assets/       # (gitignored) cached test inputs + news images
└── output_*/                # (gitignored) generated runs
```
