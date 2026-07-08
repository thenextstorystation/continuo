"""Tests for clip-level screening and cross-frame aggregation."""

import json
import os

from continuo.models import AssetBible, DriftIssue, DriftReport, Shot
from continuo.vision import aggregate_clip, screen_clip


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


def test_aggregate_merges_same_issue_across_frames():
    def frame_report(idx, dims):
        return DriftReport(
            shot_id="s",
            status="warn",
            score=80,
            summary="",
            issues=[
                DriftIssue(dimension=d, severity=sev, description="d", asset_name=name)
                for (d, sev, name) in dims
            ],
        )

    per_frame = [
        frame_report(0, [("wardrobe", "high", "jacket"), ("face", "medium", "Kaito")]),
        frame_report(1, [("wardrobe", "medium", "jacket")]),  # same jacket, lower severity
        frame_report(2, [("wardrobe", "high", "jacket")]),
    ]
    agg = aggregate_clip("s", per_frame, "mock")

    jacket = next(i for i in agg.issues if i.dimension == "wardrobe")
    face = next(i for i in agg.issues if i.dimension == "face")

    # jacket seen in all 3 frames, severity is the worst (high)
    assert jacket.frame_indices == [0, 1, 2]
    assert jacket.severity == "high"
    assert "3/3 frames" in jacket.description and "sustained" in jacket.description
    # face seen in only 1 frame -> intermittent
    assert face.frame_indices == [0]
    assert "1/3 frames" in face.description and "intermittent" in face.description
    # any high issue -> fail; score is the mean of frame scores
    assert agg.status == "fail"
    assert agg.score == 80


def test_screen_clip_mock_distinguishes_sustained_vs_flicker():
    # 3 mock frames: wardrobe drift is sustained (every frame), face drift is
    # intermittent (only even frame indices 0 and 2).
    frames = [(b"", "image/png")] * 3
    agg, per_frame = screen_clip(_bible(), _shot(), frames)
    assert len(per_frame) == 3

    wardrobe = next(i for i in agg.issues if i.dimension == "wardrobe")
    face = next(i for i in agg.issues if i.dimension == "face")
    assert len(wardrobe.frame_indices) == 3  # sustained
    assert face.frame_indices == [0, 2]  # intermittent flicker
    assert agg.status == "fail"


def test_screen_clip_single_frame_preserves_single_still_semantics():
    agg, per_frame = screen_clip(_bible(), _shot(), [])
    assert len(per_frame) == 1
    # single frame report is returned as-is (model's own status), with provenance tagged
    assert agg is per_frame[0]
    assert all(i.frame_indices == [0] for i in agg.issues)
