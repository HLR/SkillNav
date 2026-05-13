"""
Minimal utilities for preparing multimodal inputs for Qwen VL models.

This mirrors the helpers referenced in Qwen's official documentation so code
that expects ``process_vision_info`` can run without depending on the original
package distribution.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Iterable, List, Sequence, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

from PIL import Image


def _load_image_from_source(source) -> Image.Image:
    """Load an image from a PIL.Image, local path, URL, or base64 string."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if not isinstance(source, str):
        raise TypeError(f"Unsupported image source type: {type(source)}")

    # Handle base64 data URLs or raw base64 payloads
    if source.startswith("data:image"):
        header, _, data = source.partition(",")
        source = data
    if _looks_like_base64(source):
        img_bytes = base64.b64decode(source)
        return Image.open(BytesIO(img_bytes)).convert("RGB")

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        with urlopen(source) as resp:  # nosec: B310 - model loading helper
            img_bytes = resp.read()
        return Image.open(BytesIO(img_bytes)).convert("RGB")

    # Fallback to treating the string as a local path
    return Image.open(source).convert("RGB")


def _looks_like_base64(text: str) -> bool:
    """Simple heuristic to detect base64 payloads without padding."""
    if len(text) % 4 != 0:
        return False
    allowed_chars = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    )
    return all(c in allowed_chars for c in text.strip())


def _iter_message_content(messages: Sequence[dict]) -> Iterable[dict]:
    for message in messages or []:
        content = message.get("content", [])
        if isinstance(content, dict):
            content = [content]
        for item in content:
            if isinstance(item, dict):
                yield item


def process_vision_info(messages: Sequence[dict]) -> Tuple[List[Image.Image], list]:
    """
    Extract image inputs (and placeholder video inputs) from chat messages.

    Returns:
        Tuple[List[PIL.Image], List]: ``image_inputs`` and ``video_inputs`` lists
        expected by ``AutoProcessor`` for Qwen VL.
    """
    image_inputs: List[Image.Image] = []
    video_inputs = []

    for content in _iter_message_content(messages):
        content_type = content.get("type")
        if content_type == "image":
            image_inputs.append(_load_image_from_source(content.get("image")))
        elif content_type == "image_url":
            url_info = content.get("image_url") or {}
            image_inputs.append(_load_image_from_source(url_info.get("url")))
        elif content_type == "video":
            video_inputs.append(content.get("video"))

    return image_inputs, video_inputs


__all__ = ["process_vision_info"]
