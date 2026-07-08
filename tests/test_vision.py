"""Tests for the vision screening layer (mock path + context builder)."""

import json
import os

from continuo.models import AssetBible, Shot
from continuo.vision import build_bible_context, screen_shot, _report_from_payload


SAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "asset_bible.json")


def _bible():
    with open(SAMPLE, encoding="utf-8") as fh:
        return AssetBible.from_dict(json.load(fh))


def _shot():
    return Shot(
        id="shot-7",
        shot_number=7,
        prompt="Kaito leans on the counter",
        model="kling",
        expected_characters=["kaito"],
        expected_wardrobe=["jacket"],
        expected_location="kitchen",
    )


def test_context_includes_expected_assets():
    ctx = build_bible_context(_bible(), _shot())
    assert "Kaito" in ctx
    assert "thin scar across the LEFT cheek" in ctx
    assert "navy blue" in ctx
    assert "noodle-bar kitchen" in ctx


def test_mock_screen_returns_bible_referenced_drift(monkeypatch):
    # Force the mock path regardless of environment.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    report = screen_shot(_bible(), _shot(), image_bytes=None)
    assert report.model_used.startswith("mock")
    assert report.shot_id == "shot-7"
    # references the actual wardrobe item and its canonical colour
    ward = [i for i in report.issues if i.dimension == "wardrobe"]
    assert ward and "navy blue" in ward[0].expected
    # high-severity wardrobe drift -> fail status
    assert report.status == "fail"
    assert 0 <= report.score <= 100


def test_report_from_payload_clamps_score():
    payload = {"status": "pass", "score": 250, "summary": "ok", "issues": []}
    rep = _report_from_payload("s1", payload, "claude-opus-4-8")
    assert rep.score == 100
