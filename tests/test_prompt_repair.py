"""Tests for the deterministic prompt-repair engine."""

from continuo.models import DriftIssue
from continuo.prompt_repair import issue_to_correction, repair_prompt


def _wardrobe_issue():
    return DriftIssue(
        dimension="wardrobe",
        severity="high",
        description="Jacket colour has drifted.",
        expected="navy blue bomber jacket",
        observed="red bomber jacket",
        asset_name="bomber jacket",
    )


def _face_issue():
    return DriftIssue(
        dimension="face",
        severity="medium",
        description="Scar on the wrong cheek.",
        expected="scar on the left cheek",
        observed="scar on the right cheek",
        asset_name="Kaito",
    )


def _low_issue():
    return DriftIssue(
        dimension="lighting",
        severity="low",
        description="Slightly warmer grade.",
        expected="cool neon lighting",
        observed="warm lighting",
    )


def test_issue_to_correction_wardrobe():
    pos, neg = issue_to_correction(_wardrobe_issue())
    # the garment description is used directly, without an awkward "wearing" frame
    assert pos == "navy blue bomber jacket"
    assert neg == "red bomber jacket"


def test_kling_uses_negative_prompt_field():
    r = repair_prompt("Kaito leans on the counter", "kling", [_wardrobe_issue(), _face_issue()])
    assert r.model == "kling"
    # positive corrections land in the main prompt with weight syntax
    assert "navy blue bomber jacket" in r.prompt
    assert ":1.3)" in r.prompt
    # drift observations land in the dedicated negative prompt
    assert r.negative_prompt is not None
    assert "red bomber jacket" in r.negative_prompt


def test_seedance_folds_negatives_inline():
    r = repair_prompt("Kaito leans on the counter", "seedance", [_wardrobe_issue()])
    assert r.model == "seedance"
    assert r.negative_prompt is None  # no negative field
    assert "no red bomber jacket" in r.prompt


def test_veo_is_prose_with_continuity_clause():
    r = repair_prompt("Kaito leans on the counter.", "veo", [_wardrobe_issue(), _face_issue()])
    assert r.model == "veo"
    assert r.negative_prompt is None
    assert "Maintain continuity:" in r.prompt
    assert "Do not depict" in r.prompt
    assert "red bomber jacket" in r.prompt


def test_low_severity_issue_is_not_repaired():
    r = repair_prompt("A quiet street", "kling", [_low_issue()])
    assert r.negative_prompt is None
    assert any("unchanged" in n for n in r.notes)


def test_max_chars_truncation():
    long_prompt = "word " * 2000
    r = repair_prompt(long_prompt, "seedance", [_wardrobe_issue()])
    assert len(r.prompt) <= 800


def test_unknown_model_falls_back_to_default():
    r = repair_prompt("test", "nonexistent-model", [_wardrobe_issue()])
    assert r.model == "kling"  # DEFAULT_GRAMMAR
