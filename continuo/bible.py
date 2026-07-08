"""Asset-bible persistence — a tiny JSON-backed store.

One bible per Continuo instance keeps the MVP simple; a multi-project version
would key this by project id and move to a real database.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from . import config
from .models import AssetBible


def load_bible(path: Optional[str] = None) -> Optional[AssetBible]:
    path = path or config.BIBLE_PATH
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return AssetBible.from_dict(json.load(fh))


def save_bible(bible: AssetBible, path: Optional[str] = None) -> None:
    path = path or config.BIBLE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(bible.to_dict(), fh, indent=2)
