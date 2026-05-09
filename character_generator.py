"""
character_generator.py — Generate character images via ComfyUI API.

Supports any ComfyUI server (local or remote via ngrok/Kaggle).
Uses SDXL-Turbo for fast generation (4 steps, ~5-10s on T4).
"""
from __future__ import annotations

import asyncio
import json
import random
import uuid
from pathlib import Path
from typing import AsyncIterator

import httpx
import websockets


# ── Workflow builder ──────────────────────────────────────────────────────────

def _build_workflow(prompt: str, seed: int | None = None) -> dict:
    """Build an SDXL-Turbo ComfyUI API workflow dict."""
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)

    # Animagine XL 3.1 — anime-optimized SDXL model
    # Falls back to SDXL Turbo if Animagine isn't installed
    return {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "animagine-xl-3.1.safetensors"},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": 1, "height": 1024, "width": 1024},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                # Animagine quality boosters
                "text": f"masterpiece, best quality, very aesthetic, absurdres, {prompt}",
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": (
                    "lowres, bad anatomy, bad hands, text, error, missing fingers, "
                    "extra digit, fewer digits, cropped, worst quality, low quality, "
                    "normal quality, jpeg artifacts, signature, watermark, username, blurry, "
                    "extra limbs, extra hands, extra arms, deformed hands, mutated hands, "
                    "malformed limbs, fused fingers, too many fingers, poorly drawn face, "
                    "multiple heads, two heads, three heads, duplicate, multiple faces, "
                    "clones, split person, doubled, mirrored, nsfw"
                ),
            },
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 7.0,
                "denoise": 1,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "seed": seed,
                "steps": 28,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "character", "images": ["8", 0]},
        },
    }


# ── Public API ────────────────────────────────────────────────────────────────

async def generate_character_image(
    prompt: str,
    comfyui_url: str,
    save_dir: str = "characters",
    timeout: int = 120,
) -> str:
    """
    Submit a prompt to ComfyUI and download the generated image.
    Returns the local path of the saved PNG.
    """
    base = comfyui_url.rstrip("/")
    workflow = _build_workflow(prompt)

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        # 1. Submit workflow
        r = await client.post(f"{base}/prompt", json={"prompt": workflow})
        r.raise_for_status()
        prompt_id = r.json()["prompt_id"]

        # 2. Poll history until done
        for _ in range(timeout):
            await asyncio.sleep(1)
            hist = await client.get(f"{base}/history/{prompt_id}")
            if hist.status_code == 200:
                data = hist.json()
                if prompt_id in data:
                    break
        else:
            raise TimeoutError(f"ComfyUI timed out after {timeout}s")

        # 3. Find output image
        outputs = data[prompt_id].get("outputs", {})
        img_info = None
        for node_output in outputs.values():
            imgs = node_output.get("images", [])
            if imgs:
                img_info = imgs[0]
                break

        if img_info is None:
            raise RuntimeError("ComfyUI returned no images")

        # 4. Download image
        img_r = await client.get(
            f"{base}/view",
            params={
                "filename": img_info["filename"],
                "subfolder": img_info.get("subfolder", ""),
                "type": img_info.get("type", "output"),
            },
        )
        img_r.raise_for_status()

        # 5. Save locally
        Path(save_dir).mkdir(exist_ok=True)
        out_name = f"ai_char_{uuid.uuid4().hex[:8]}.png"
        out_path = Path(save_dir) / out_name
        out_path.write_bytes(img_r.content)
        return str(out_path)


async def generate_character_image_streaming(
    prompt: str,
    comfyui_url: str,
    save_dir: str = "characters",
    timeout: int = 300,
) -> AsyncIterator[dict]:
    """
    Async generator that streams progress events from ComfyUI via WebSocket.

    Yields dicts of these shapes:
      {"type": "queued",    "prompt_id": str}
      {"type": "progress",  "value": int, "max": int}
      {"type": "executing", "node": str | None}
      {"type": "done",      "path": str}     ← final event
      {"type": "error",     "error": str}
    """
    base = comfyui_url.rstrip("/")
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    workflow = _build_workflow(prompt)
    client_id = uuid.uuid4().hex
    ws_url = f"{ws_base}/ws?clientId={client_id}"

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        # Open WebSocket FIRST so we don't miss early events
        try:
            ws = await websockets.connect(ws_url, max_size=None)
        except Exception as e:
            yield {"type": "error", "error": f"WebSocket connect failed: {e}"}
            return

        try:
            # Submit workflow with our client_id
            try:
                r = await client.post(
                    f"{base}/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                )
                r.raise_for_status()
                prompt_id = r.json()["prompt_id"]
            except Exception as e:
                yield {"type": "error", "error": f"Submit failed: {e}"}
                return

            yield {"type": "queued", "prompt_id": prompt_id}

            # Listen for events
            done = False
            async for raw in ws:
                if not isinstance(raw, str):
                    continue  # skip binary preview frames
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                msg_type = msg.get("type")
                msg_data = msg.get("data", {})

                if msg_type == "progress":
                    yield {
                        "type": "progress",
                        "value": msg_data.get("value", 0),
                        "max":   msg_data.get("max", 1),
                    }
                elif msg_type == "executing":
                    node = msg_data.get("node")
                    yield {"type": "executing", "node": node}
                    # ComfyUI sends node=None when prompt finishes
                    if node is None and msg_data.get("prompt_id") == prompt_id:
                        done = True
                        break
                elif msg_type == "execution_error":
                    yield {"type": "error", "error": str(msg_data)}
                    return
        finally:
            await ws.close()

        if not done:
            yield {"type": "error", "error": "WebSocket closed before completion"}
            return

        # Fetch result from history
        try:
            hist = await client.get(f"{base}/history/{prompt_id}")
            hist.raise_for_status()
            data = hist.json()
        except Exception as e:
            yield {"type": "error", "error": f"History fetch failed: {e}"}
            return

        outputs = data.get(prompt_id, {}).get("outputs", {})
        img_info = None
        for node_output in outputs.values():
            imgs = node_output.get("images", [])
            if imgs:
                img_info = imgs[0]
                break

        if img_info is None:
            yield {"type": "error", "error": "ComfyUI returned no images"}
            return

        # Download generated image
        try:
            img_r = await client.get(
                f"{base}/view",
                params={
                    "filename":  img_info["filename"],
                    "subfolder": img_info.get("subfolder", ""),
                    "type":      img_info.get("type", "output"),
                },
            )
            img_r.raise_for_status()
        except Exception as e:
            yield {"type": "error", "error": f"Image download failed: {e}"}
            return

        Path(save_dir).mkdir(exist_ok=True)
        out_name = f"ai_char_{uuid.uuid4().hex[:8]}.png"
        out_path = Path(save_dir) / out_name
        out_path.write_bytes(img_r.content)

        yield {"type": "done", "path": str(out_path)}


async def test_connection(comfyui_url: str, timeout: int = 8) -> bool:
    """Return True if ComfyUI server responds to /system_stats."""
    base = comfyui_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{base}/system_stats")
            return r.status_code == 200
    except Exception:
        return False
