"""Runtime configuration, read from the environment."""

from __future__ import annotations

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("CONTINUO_DATA_DIR", os.path.join(_ROOT, "data"))
BIBLE_PATH = os.path.join(DATA_DIR, "bible.json")

# Vision judging defaults to the most capable Claude model. Keep vision cost
# below ~10% of the customer's generation spend (per the product brief) by
# swapping to a cheaper tier here if needed.
VISION_MODEL = os.environ.get("CONTINUO_VISION_MODEL", "claude-opus-4-8")


def has_api_key() -> bool:
    """True when an Anthropic credential is available for live vision screening."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
