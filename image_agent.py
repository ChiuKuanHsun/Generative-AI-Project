"""
Image agent — multi-source news image fetcher with Claude-based ranking.

Searches free image APIs for topic-relevant images, then asks Claude Haiku to
pick the most useful one. Sources:

  • Wikimedia Commons  — no API key needed (great for logos, asset icons)
  • Guardian Open API  — free key (https://bonobo.capi.gutools.co.uk/register/developer)
  • NYT Article Search — optional, free key (https://developer.nytimes.com/get-started)

Set GUARDIAN_API_KEY and (optionally) NYT_API_KEY in api-key.env.

Usage:
    from image_agent import generate_news_image
    path = generate_news_image("Bitcoin price crash today", "news_image.png")
    # returns the saved path, or None if no usable image was found
"""
from __future__ import annotations

import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import anthropic
import numpy as np
import requests
from PIL import Image

# ── Endpoints ─────────────────────────────────────────────────────────────────
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
GUARDIAN_API  = "https://content.guardianapis.com/search"
NYT_API       = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

HTTP_TIMEOUT = 15
# Wikimedia's UA policy asks for an identifiable bot name + use case.
# https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = (
    "FinancialVideoBot/1.0 "
    "(auto-generated short-form video pipeline; educational use; "
    "https://github.com/anthropics/claude-code)"
)

# Minimum seconds between requests to the same host. Wikimedia's file CDN
# (upload.wikimedia.org) is the strict one — its API endpoints are more lenient.
_HOST_MIN_INTERVAL = {
    "upload.wikimedia.org":  1.0,
    "commons.wikimedia.org": 0.3,
}
_last_hit: dict[str, float] = {}

MIN_PIXELS               = 300   # reject tiny thumbnails
MAX_CANDIDATES_PER_QUERY = 6
MAX_TERMS                = 3

# Pixel-stat quality thresholds (tuned to catch pure-black, pure-white,
# and one-color/low-detail diagrams without over-rejecting real photos).
QUALITY_MIN_MEAN_LUM = 15    # reject mostly-black images
QUALITY_MAX_MEAN_LUM = 240   # reject mostly-white / blown-out images
QUALITY_MIN_STDDEV   = 14    # reject solid-color or near-uniform images


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _http_get(url: str, params: dict | None = None,
              max_retries: int = 2) -> requests.Response | None:
    """GET with per-host throttle + 429/5xx retry honoring Retry-After."""
    host = urlparse(url).netloc
    interval = _HOST_MIN_INTERVAL.get(host, 0.0)

    for attempt in range(max_retries + 1):
        if interval > 0:
            wait = interval - (time.time() - _last_hit.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)

        try:
            r = requests.get(
                url, params=params, timeout=HTTP_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            print(f"  HTTP error ({host}): {e}")
            return None

        _last_hit[host] = time.time()

        if r.status_code == 429 and attempt < max_retries:
            try:
                retry_after = int(r.headers.get("Retry-After", "5"))
            except ValueError:
                retry_after = 5
            wait = min(max(retry_after, 1), 30)
            print(f"  {host} rate-limited (429); waiting {wait}s...")
            time.sleep(wait)
            continue

        if r.status_code in (500, 502, 503, 504) and attempt < max_retries:
            wait = 2.0 * (attempt + 1)
            print(f"  HTTP {r.status_code} from {host}; retrying in {wait:.0f}s...")
            time.sleep(wait)
            continue

        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            print(f"  HTTP error ({host}): {e}")
            return None
        return r

    return None


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


# ── Claude: topic → search terms ──────────────────────────────────────────────

_TERMS_SYSTEM = (
    "You convert a news topic to 2-3 short ENGLISH search terms suitable for a "
    "news image API (Guardian, Wikimedia Commons, NYT). Prefer concrete entities "
    "(company names, asset symbols, people, products) over abstract phrases. "
    'Return ONLY a JSON array of strings, no other text. Example: ["Bitcoin", "Ethereum", "cryptocurrency"].'
)


def _extract_search_terms(topic: str) -> list[str]:
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            system=_TERMS_SYSTEM,
            messages=[{"role": "user", "content": topic}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        m = re.search(r"\[[\s\S]*?\]", text)
        if m:
            terms = json.loads(m.group(0))
            cleaned = [t.strip() for t in terms if isinstance(t, str) and t.strip()]
            if cleaned:
                return cleaned[:MAX_TERMS]
    except Exception as e:
        print(f"  Term extraction failed ({e}); using raw topic.")
    return [topic]


# ── Wikimedia Commons ─────────────────────────────────────────────────────────

def _search_wikimedia(term: str) -> list[dict]:
    r = _http_get(WIKIMEDIA_API, {
        "action":     "query",
        "format":     "json",
        "list":       "search",
        "srsearch":   term,
        "srnamespace": 6,                        # File: namespace
        "srlimit":    MAX_CANDIDATES_PER_QUERY,
    })
    if r is None:
        return []
    titles = [hit["title"] for hit in r.json().get("query", {}).get("search", [])]
    if not titles:
        return []

    info_r = _http_get(WIKIMEDIA_API, {
        "action":     "query",
        "format":     "json",
        "prop":       "imageinfo",
        "iiprop":     "url|size|mime",
        "iiurlwidth": 800,                       # rasterized SVG thumbs come back as PNG
        "titles":     "|".join(titles),
    })
    if info_r is None:
        return []

    candidates = []
    pages = info_r.json().get("query", {}).get("pages", {})
    for page in pages.values():
        info_list = page.get("imageinfo", [])
        if not info_list:
            continue
        info = info_list[0]
        if not info.get("mime", "").startswith("image/"):
            continue
        url = info.get("thumburl") or info.get("url")
        if not url:
            continue
        candidates.append({
            "source":      "wikimedia",
            "title":       page.get("title", "").replace("File:", ""),
            "description": "",
            "url":         url,
            "width":       int(info.get("thumbwidth") or info.get("width") or 0),
            "height":      int(info.get("thumbheight") or info.get("height") or 0),
        })
    return candidates


# ── Guardian ──────────────────────────────────────────────────────────────────

def _search_guardian(term: str, api_key: str) -> list[dict]:
    r = _http_get(GUARDIAN_API, {
        "q":             term,
        "show-fields":   "thumbnail,trailText,headline",
        "show-elements": "image",
        "page-size":     MAX_CANDIDATES_PER_QUERY,
        "order-by":      "relevance",
        "api-key":       api_key,
    })
    if r is None:
        return []

    candidates = []
    for art in r.json().get("response", {}).get("results", []):
        fields = art.get("fields", {}) or {}
        url, width, height = None, 0, 0

        # Prefer full-resolution image from elements over the small thumbnail
        for elem in art.get("elements", []) or []:
            if elem.get("type") != "image":
                continue
            assets = elem.get("assets", []) or []
            best = max(
                assets,
                key=lambda a: int((a.get("typeData", {}) or {}).get("width", 0) or 0),
                default=None,
            )
            if best:
                url = best.get("file")
                td  = best.get("typeData", {}) or {}
                width  = int(td.get("width", 0) or 0)
                height = int(td.get("height", 0) or 0)
                break

        if not url:
            url = fields.get("thumbnail")
        if not url:
            continue

        candidates.append({
            "source":      "guardian",
            "title":       fields.get("headline") or art.get("webTitle", ""),
            "description": _strip_html(fields.get("trailText", ""))[:200],
            "url":         url,
            "width":       width,
            "height":      height,
        })
    return candidates


# ── NYT (optional) ────────────────────────────────────────────────────────────

_NYT_HOST = "https://www.nytimes.com/"

def _search_nyt(term: str, api_key: str) -> list[dict]:
    r = _http_get(NYT_API, {"q": term, "api-key": api_key, "page": 0})
    if r is None:
        return []
    docs = r.json().get("response", {}).get("docs", [])
    candidates = []
    for doc in docs[:MAX_CANDIDATES_PER_QUERY]:
        media = doc.get("multimedia")
        media_list = []
        if isinstance(media, list):
            media_list = media
        elif isinstance(media, dict):
            # Newer schema returns a dict with default/caption/credit subkeys
            for key in ("default", "thumbnail"):
                if isinstance(media.get(key), dict):
                    media_list.append(media[key])

        if not media_list:
            continue
        best = max(media_list, key=lambda m: int(m.get("width") or 0) * int(m.get("height") or 0))
        url = best.get("url") or ""
        if not url:
            continue
        if not url.startswith("http"):
            url = _NYT_HOST + url.lstrip("/")
        candidates.append({
            "source":      "nyt",
            "title":       (doc.get("headline") or {}).get("main", ""),
            "description": (doc.get("abstract") or "")[:200],
            "url":         url,
            "width":       int(best.get("width") or 0),
            "height":      int(best.get("height") or 0),
        })
    return candidates


# ── Claude: rank the top N candidates ─────────────────────────────────────────

_PICK_SYSTEM = (
    "You are a visual editor for short-form news videos. Given a topic and candidate "
    "images, return the TOP N images IN ORDER from best to worst. Prefer concrete, "
    "varied subjects (company logos, products, news photos, asset icons). Avoid "
    "book covers, abstract art, paywall placeholders, generic phone/laptop stock "
    "shots, and tiny icons. Pick a DIVERSE set — don't return four near-duplicates. "
    'Return ONLY a JSON array of indices, no other text. Example: [3, 7, 1, 5].'
)


def _pick_top(topic: str, candidates: list[dict], n: int) -> list[dict]:
    """Claude Haiku ranks candidates and returns the top n in order."""
    if not candidates:
        return []
    if len(candidates) <= n:
        return list(candidates)

    listing = "\n".join(
        f"[{i}] source={c['source']} | title={c['title'][:80]!r} | "
        f"desc={c.get('description','')[:80]!r} | "
        f"size={c.get('width',0)}x{c.get('height',0)}"
        for i, c in enumerate(candidates)
    )

    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=_PICK_SYSTEM,
            messages=[{"role": "user",
                       "content": f"Topic: {topic}\nN: {n}\n\nCandidates:\n{listing}"}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        m = re.search(r"\[[^\]]*\]", text)
        if m:
            indices = json.loads(m.group(0))
            picks: list[dict] = []
            seen: set[int] = set()
            for idx in indices:
                idx = int(idx)
                if idx in seen or not (0 <= idx < len(candidates)):
                    continue
                seen.add(idx)
                picks.append(candidates[idx])
                if len(picks) >= n:
                    break
            if picks:
                print(f"  Agent ranked top {len(picks)}: "
                      f"{[c['source'] + '/' + c['title'][:30] for c in picks]}")
                return picks
    except Exception as e:
        print(f"  Ranking failed ({e}); using first {n} candidates.")
    return candidates[:n]


# ── Quality filter ────────────────────────────────────────────────────────────

def _image_quality_ok(img: Image.Image) -> tuple[bool, str]:
    """Reject pure-black / pure-white / near-uniform images via luminance stats."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    mean = float(gray.mean())
    std  = float(gray.std())

    if mean < QUALITY_MIN_MEAN_LUM:
        return False, f"too dark (mean lum {mean:.1f})"
    if mean > QUALITY_MAX_MEAN_LUM:
        return False, f"too bright (mean lum {mean:.1f})"
    if std < QUALITY_MIN_STDDEV:
        return False, f"too uniform (stddev {std:.1f})"
    return True, ""


# ── Download ──────────────────────────────────────────────────────────────────

def _download(url: str, output_path: str) -> str | None:
    try:
        r = _http_get(url)
        if r is None:
            return None
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if img.width < MIN_PIXELS or img.height < MIN_PIXELS:
            print(f"  Rejected ({img.width}x{img.height} too small)")
            return None
        ok, reason = _image_quality_ok(img)
        if not ok:
            print(f"  Rejected ({reason})")
            return None
        img.save(output_path, "PNG")
        print(f"  Saved: {output_path} ({img.width}x{img.height})")
        return output_path
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


# ── Public entrypoint ─────────────────────────────────────────────────────────

def _gather_candidates(topic: str) -> list[dict]:
    terms = _extract_search_terms(topic)
    print(f"  Search terms: {terms}")

    guardian_key = os.environ.get("GUARDIAN_API_KEY")
    nyt_key      = os.environ.get("NYT_API_KEY")

    if not guardian_key:
        print("  (Guardian key not set — set GUARDIAN_API_KEY in api-key.env to enable)")

    candidates: list[dict] = []
    for term in terms:
        candidates.extend(_search_wikimedia(term))
        if guardian_key:
            try:
                candidates.extend(_search_guardian(term, guardian_key))
            except Exception as e:
                print(f"  Guardian error: {e}")
        if nyt_key:
            try:
                candidates.extend(_search_nyt(term, nyt_key))
            except Exception as e:
                print(f"  NYT error: {e}")

    seen, uniq = set(), []
    for c in candidates:
        if c["url"] and c["url"] not in seen:
            seen.add(c["url"])
            uniq.append(c)
    return uniq


def generate_news_images(topic: str, output_dir: str = ".",
                         count: int = 4) -> list[str]:
    """Search free image APIs, rank top N via Claude, download each.
    Returns a list of saved paths (length 0..count). Saves as news_image_00.png, ..."""
    print(f"\n  Searching news images for: {topic}  (target: {count})")

    candidates = _gather_candidates(topic)
    if not candidates:
        print("  No image candidates found.")
        return []

    sources = sorted({c["source"] for c in candidates})
    print(f"  {len(candidates)} candidates from: {', '.join(sources)}")

    # Ask the agent to rank a generous over-count, then fall back to unranked
    # candidates if the agent's picks get filtered out by the quality check.
    over_count = min(len(candidates), max(count * 3, count + 4))
    ranked = _pick_top(topic, candidates, over_count)

    ranked_urls = {c["url"] for c in ranked}
    remainder = [c for c in candidates if c["url"] not in ranked_urls]
    to_try    = ranked + remainder

    saved: list[str] = []
    used_urls: set[str] = set()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for chosen in to_try:
        if len(saved) >= count:
            break
        if chosen["url"] in used_urls:
            continue
        out = os.path.join(output_dir, f"news_image_{len(saved):02d}.png")
        result = _download(chosen["url"], out)
        used_urls.add(chosen["url"])
        if result:
            saved.append(result)

    if len(saved) < count:
        print(f"  Only got {len(saved)}/{count} images "
              f"(some downloads failed or candidates exhausted).")
    else:
        print(f"  Got {len(saved)} images.")
    return saved


def generate_news_image(topic: str, output_path: str = "news_image.png") -> str | None:
    """Single-image convenience wrapper. Returns one path or None."""
    output_dir = os.path.dirname(output_path) or "."
    paths = generate_news_images(topic, output_dir=output_dir, count=1)
    if not paths:
        return None
    if paths[0] != output_path:
        try:
            os.replace(paths[0], output_path)
            return output_path
        except OSError:
            return paths[0]
    return paths[0]


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv("api-key.env")

    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Bitcoin Ethereum cryptocurrency market"
    paths = generate_news_images(topic, output_dir="image_test_out", count=4)
    print(f"\nResult: {len(paths)} images")
    for p in paths:
        print(f"  {p}")
