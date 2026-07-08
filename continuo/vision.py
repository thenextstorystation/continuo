"""Vision-model drift screening.

Screens a single generated frame against the asset bible using a Claude vision
model and returns a structured DriftReport. When no Anthropic credential is
configured it falls back to a deterministic mock so the whole product loop —
bible → screen → report card → prompt repair — is exercisable offline and in
tests.
"""

from __future__ import annotations

import base64
import json
from typing import Optional

from . import config
from .models import (
    AssetBible,
    DriftIssue,
    DriftReport,
    Shot,
)

# JSON schema the vision model must fill. Structured outputs guarantee this
# shape so we never parse free-form prose.
DRIFT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["pass", "warn", "fail"]},
        "score": {"type": "integer"},
        "summary": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {
                        "type": "string",
                        "enum": ["face", "wardrobe", "lighting", "set_geometry", "prop", "other"],
                    },
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "asset_name": {"type": "string"},
                    "description": {"type": "string"},
                    "expected": {"type": "string"},
                    "observed": {"type": "string"},
                },
                "required": [
                    "dimension",
                    "severity",
                    "asset_name",
                    "description",
                    "expected",
                    "observed",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["status", "score", "summary", "issues"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are Continuo, an expert continuity supervisor for AI-generated film. "
    "You are given the canonical 'asset bible' definitions for the characters, "
    "wardrobe, props, and location that are supposed to appear in a shot, plus a "
    "single frame from the generated clip. Compare the frame against the bible and "
    "report every continuity drift you can see, frame-accurately, across these "
    "dimensions: face, wardrobe, lighting, set_geometry, prop. For each drift give "
    "the expected (canonical) appearance and the observed (drifted) appearance. "
    "Only report drift you can actually see in the frame; do not invent assets that "
    "are not visible. Score 0-100 where 100 is perfectly on-model. Use status "
    "'pass' when there is no meaningful drift, 'warn' for minor drift, and 'fail' "
    "when a high-severity drift is present."
)


def build_bible_context(bible: AssetBible, shot: Shot) -> str:
    """Render the expected assets for a shot into a compact text brief."""
    lines: list[str] = [f"PROJECT: {bible.project}", f"SHOT #{shot.shot_number}"]
    lines.append(f"ORIGINAL PROMPT: {shot.prompt}")

    chars = shot.expected_characters or [c.id for c in bible.characters]
    if chars:
        lines.append("\nEXPECTED CHARACTERS:")
        for cid in chars:
            c = bible.character(cid)
            if not c:
                continue
            face = f" | face: {c.face_notes}" if c.face_notes else ""
            lines.append(f"  - {c.name}: {c.description}{face}")

    wardrobe = shot.expected_wardrobe or [w.id for w in bible.wardrobe]
    if wardrobe:
        lines.append("\nEXPECTED WARDROBE:")
        for wid in wardrobe:
            w = bible.wardrobe_item(wid)
            if not w:
                continue
            worn = bible.character(w.worn_by) if w.worn_by else None
            worn_by = f" (worn by {worn.name})" if worn else ""
            colors = f" | colours: {', '.join(w.colors)}" if w.colors else ""
            lines.append(f"  - {w.name}{worn_by}: {w.description}{colors}")

    props = shot.expected_props or [p.id for p in bible.props]
    if props:
        lines.append("\nEXPECTED PROPS:")
        for pid in props:
            p = bible.prop(pid)
            if p:
                lines.append(f"  - {p.name}: {p.description}")

    loc_id = shot.expected_location
    loc = bible.location(loc_id) if loc_id else (bible.locations[0] if bible.locations else None)
    if loc:
        feats = f" | key features: {', '.join(loc.key_features)}" if loc.key_features else ""
        lines.append(f"\nEXPECTED LOCATION:\n  - {loc.name}: {loc.description}{feats}")

    return "\n".join(lines)


def _report_from_payload(shot_id: str, payload: dict, model_used: str) -> DriftReport:
    issues = [DriftIssue.from_dict(x) for x in payload.get("issues", [])]
    score = int(payload.get("score", 0))
    score = max(0, min(100, score))
    return DriftReport(
        shot_id=shot_id,
        status=payload.get("status", "warn"),
        score=score,
        summary=payload.get("summary", ""),
        issues=issues,
        model_used=model_used,
    )


def _mock_report(bible: AssetBible, shot: Shot, frame_index: int = 0) -> DriftReport:
    """Deterministic stand-in used when no API key is set.

    It fabricates plausible drift referencing the actual bible so the demo loop
    is coherent, and is clearly labelled so it is never mistaken for a real
    judgement. Drift varies by ``frame_index`` so clip-level aggregation has
    something meaningful to work with: a sustained wardrobe drift in every frame
    versus an intermittent face flicker in only some frames.
    """
    issues: list[DriftIssue] = []

    # Wardrobe colour swap — the canonical "jacket changes colour in shot 7".
    # Sustained: present in every sampled frame.
    ward_ids = shot.expected_wardrobe or [w.id for w in bible.wardrobe]
    w = next((bible.wardrobe_item(wid) for wid in ward_ids if bible.wardrobe_item(wid)), None)
    if w:
        canonical = ", ".join(w.colors) if w.colors else w.description
        issues.append(
            DriftIssue(
                dimension="wardrobe",
                severity="high",
                description=f"{w.name} colour has drifted from the bible.",
                expected=f"{w.name} in {canonical}",
                observed=f"{w.name} in an off-model colour",
                asset_id=w.id,
                asset_name=w.name,
            )
        )

    # Face drift — the "scar switches cheeks". Intermittent: only some frames.
    char_ids = shot.expected_characters or [c.id for c in bible.characters]
    c = next(
        (bible.character(cid) for cid in char_ids if bible.character(cid) and bible.character(cid).face_notes),
        None,
    )
    if c and frame_index % 2 == 0:
        issues.append(
            DriftIssue(
                dimension="face",
                severity="medium",
                description=f"{c.name}'s distinguishing facial feature is inconsistent.",
                expected=c.face_notes,
                observed="feature mirrored / on the wrong side",
                asset_id=c.id,
                asset_name=c.name,
            )
        )

    has_high = any(i.severity == "high" for i in issues)
    status = "fail" if has_high else ("warn" if issues else "pass")
    score = 55 if has_high else (78 if issues else 96)
    summary = (
        "MOCK screening (no ANTHROPIC_API_KEY set): fabricated drift referencing the "
        "asset bible so the workflow can be demonstrated offline. Set a credential for "
        "real frame-accurate screening."
    )
    return DriftReport(
        shot_id=shot.id,
        status=status,
        score=score,
        summary=summary,
        issues=issues,
        model_used="mock (no ANTHROPIC_API_KEY)",
    )


def screen_shot(
    bible: AssetBible,
    shot: Shot,
    image_bytes: Optional[bytes] = None,
    media_type: str = "image/png",
    frame_index: int = 0,
) -> DriftReport:
    """Screen one frame against the bible, returning a structured DriftReport."""
    if not config.has_api_key() or image_bytes is None:
        return _mock_report(bible, shot, frame_index=frame_index)

    # Imported lazily so the package stays importable without the SDK installed.
    import anthropic

    client = anthropic.Anthropic()
    context = build_bible_context(bible, shot)
    data = base64.standard_b64encode(image_bytes).decode("utf-8")

    response = client.messages.create(
        model=config.VISION_MODEL,
        max_tokens=4096,
        system=_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    },
                    {
                        "type": "text",
                        "text": "Asset bible for this shot:\n\n"
                        + context
                        + "\n\nCompare the frame above against these definitions and "
                        "report all continuity drift.",
                    },
                ],
            }
        ],
        output_config={"format": {"type": "json_schema", "schema": DRIFT_SCHEMA}},
    )

    text = next((b.text for b in response.content if b.type == "text"), "{}")
    payload = json.loads(text)
    return _report_from_payload(shot.id, payload, model_used=response.model)


# ---------------------------------------------------------------------------
# Clip-level screening — sample N frames and aggregate their drift
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


def aggregate_clip(
    shot_id: str, per_frame: list[DriftReport], model_used: str
) -> DriftReport:
    """Merge per-frame reports into one clip-level DriftReport.

    Issues describing the same drift (same dimension + asset) are merged across
    frames; the merged severity is the worst seen, and ``frame_indices`` records
    exactly which frames it appeared in — so a one-frame flicker is
    distinguishable from a drift sustained across the whole clip.
    """
    frame_count = len(per_frame)
    merged: dict[tuple[str, str], DriftIssue] = {}

    for idx, rep in enumerate(per_frame):
        for iss in rep.issues:
            key = (iss.dimension, (iss.asset_name or "").lower())
            if key not in merged:
                merged[key] = DriftIssue(
                    dimension=iss.dimension,
                    severity=iss.severity,
                    description=iss.description,
                    expected=iss.expected,
                    observed=iss.observed,
                    asset_id=iss.asset_id,
                    asset_name=iss.asset_name,
                    frame_indices=[idx],
                )
            else:
                m = merged[key]
                if idx not in m.frame_indices:
                    m.frame_indices.append(idx)
                if _SEVERITY_RANK[iss.severity] > _SEVERITY_RANK[m.severity]:
                    m.severity = iss.severity

    issues = list(merged.values())

    # Annotate persistence so the report reads like a continuity note.
    if frame_count > 1:
        for m in issues:
            n = len(m.frame_indices)
            kind = "sustained across the clip" if n == frame_count else "intermittent flicker"
            m.description = f"{m.description} (seen in {n}/{frame_count} frames — {kind})"

    ranks = [_SEVERITY_RANK[i.severity] for i in issues]
    if any(r == 3 for r in ranks):
        status = "fail"
    elif any(r == 2 for r in ranks):
        status = "warn"
    else:
        status = "pass"

    score = round(sum(r.score for r in per_frame) / frame_count) if frame_count else 0
    summary = (
        f"Clip screened across {frame_count} frame(s): "
        f"{len(issues)} distinct drift issue(s) after cross-frame aggregation."
    )
    return DriftReport(
        shot_id=shot_id,
        status=status,
        score=score,
        summary=summary,
        issues=issues,
        model_used=model_used,
    )


def screen_clip(
    bible: AssetBible,
    shot: Shot,
    frames: list[tuple[bytes, str]],
) -> tuple[DriftReport, list[DriftReport]]:
    """Screen every sampled frame of a clip and return (aggregated, per_frame).

    ``frames`` is a list of (image_bytes, media_type). An empty list falls back
    to a single (mock) screening so the endpoint still works with no upload.
    Returns the clip-level report plus the individual per-frame reports.
    """
    per_frame: list[DriftReport] = []
    if not frames:
        per_frame = [screen_shot(bible, shot, image_bytes=None, frame_index=0)]
    else:
        for idx, (data, media_type) in enumerate(frames):
            per_frame.append(
                screen_shot(bible, shot, image_bytes=data, media_type=media_type, frame_index=idx)
            )

    # Tag frame provenance on every per-frame issue.
    for idx, rep in enumerate(per_frame):
        for iss in rep.issues:
            if not iss.frame_indices:
                iss.frame_indices = [idx]

    if len(per_frame) == 1:
        # Preserve exact single-still semantics (the model's own status/summary).
        return per_frame[0], per_frame

    model_used = per_frame[0].model_used if per_frame else ""
    aggregated = aggregate_clip(shot.id, per_frame, model_used)
    return aggregated, per_frame
