"""Tests for asset-bible parsing and persistence."""

import json
import os

from continuo.bible import load_bible, save_bible
from continuo.models import AssetBible


SAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "asset_bible.json")


def test_sample_bible_parses():
    with open(SAMPLE, encoding="utf-8") as fh:
        bible = AssetBible.from_dict(json.load(fh))
    assert bible.project.startswith("Neon Alley")
    assert bible.character("kaito").face_notes.startswith("thin scar")
    assert bible.wardrobe_item("jacket").colors == ["navy blue"]
    assert bible.location("kitchen").key_features[0].startswith("no window")


def test_from_dict_ignores_unknown_fields():
    bible = AssetBible.from_dict(
        {
            "project": "X",
            "characters": [{"id": "a", "name": "A", "bogus": 123}],
        }
    )
    assert bible.characters[0].name == "A"


def test_roundtrip_persistence(tmp_path):
    with open(SAMPLE, encoding="utf-8") as fh:
        bible = AssetBible.from_dict(json.load(fh))
    path = tmp_path / "bible.json"
    save_bible(bible, str(path))
    reloaded = load_bible(str(path))
    assert reloaded is not None
    assert reloaded.to_dict() == bible.to_dict()


def test_load_missing_returns_none(tmp_path):
    assert load_bible(str(tmp_path / "nope.json")) is None
