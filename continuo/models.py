"""Domain models for Continuo — the continuity supervisor for AI filmmaking.

These are plain dataclasses with no third-party dependencies so the core
logic (asset bible, prompt repair, drift reporting) can be imported and
tested without FastAPI, pydantic, or the Anthropic SDK installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

# Drift dimensions Continuo screens for. These mirror the axes a real
# continuity supervisor ("script girl") tracks on set.
DIMENSIONS = ("face", "wardrobe", "lighting", "set_geometry", "prop", "other")
SEVERITIES = ("low", "medium", "high")
STATUSES = ("pass", "warn", "fail")


def _pick(cls, d: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Return only the known keys from ``d`` so unexpected client fields are ignored."""
    return {k: d[k] for k in keys if k in d and d[k] is not None}


# ---------------------------------------------------------------------------
# Asset bible
# ---------------------------------------------------------------------------


@dataclass
class Character:
    id: str
    name: str
    description: str = ""
    face_notes: str = ""  # scars, distinguishing features — the classic drift traps
    reference_images: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Character":
        return cls(**_pick(cls, d, ("id", "name", "description", "face_notes", "reference_images")))


@dataclass
class WardrobeItem:
    id: str
    name: str
    worn_by: Optional[str] = None  # character id
    description: str = ""
    colors: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WardrobeItem":
        return cls(**_pick(cls, d, ("id", "name", "worn_by", "description", "colors")))


@dataclass
class Prop:
    id: str
    name: str
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Prop":
        return cls(**_pick(cls, d, ("id", "name", "description")))


@dataclass
class Location:
    id: str
    name: str
    description: str = ""
    key_features: list[str] = field(default_factory=list)  # set geometry: windows, doors, layout

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Location":
        return cls(**_pick(cls, d, ("id", "name", "description", "key_features")))


@dataclass
class AssetBible:
    project: str
    characters: list[Character] = field(default_factory=list)
    wardrobe: list[WardrobeItem] = field(default_factory=list)
    props: list[Prop] = field(default_factory=list)
    locations: list[Location] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AssetBible":
        return cls(
            project=d.get("project", "Untitled project"),
            characters=[Character.from_dict(x) for x in d.get("characters", [])],
            wardrobe=[WardrobeItem.from_dict(x) for x in d.get("wardrobe", [])],
            props=[Prop.from_dict(x) for x in d.get("props", [])],
            locations=[Location.from_dict(x) for x in d.get("locations", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # convenience lookups -------------------------------------------------
    def character(self, cid: str) -> Optional[Character]:
        return next((c for c in self.characters if c.id == cid), None)

    def wardrobe_item(self, wid: str) -> Optional[WardrobeItem]:
        return next((w for w in self.wardrobe if w.id == wid), None)

    def prop(self, pid: str) -> Optional[Prop]:
        return next((p for p in self.props if p.id == pid), None)

    def location(self, lid: str) -> Optional[Location]:
        return next((loc for loc in self.locations if loc.id == lid), None)


# ---------------------------------------------------------------------------
# Shots
# ---------------------------------------------------------------------------

# Prompt grammars are model-specific; these are the models Continuo sits above.
SUPPORTED_MODELS = ("kling", "seedance", "veo", "sora")


@dataclass
class Shot:
    id: str
    shot_number: int
    prompt: str
    model: str = "kling"
    expected_characters: list[str] = field(default_factory=list)
    expected_wardrobe: list[str] = field(default_factory=list)
    expected_props: list[str] = field(default_factory=list)
    expected_location: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Shot":
        return cls(
            id=str(d.get("id") or f"shot-{d.get('shot_number', 0)}"),
            shot_number=int(d.get("shot_number", 0)),
            prompt=d.get("prompt", ""),
            model=d.get("model", "kling"),
            expected_characters=list(d.get("expected_characters", [])),
            expected_wardrobe=list(d.get("expected_wardrobe", [])),
            expected_props=list(d.get("expected_props", [])),
            expected_location=d.get("expected_location"),
        )


# ---------------------------------------------------------------------------
# Drift reporting
# ---------------------------------------------------------------------------


@dataclass
class DriftIssue:
    dimension: str  # one of DIMENSIONS
    severity: str  # one of SEVERITIES
    description: str
    expected: str = ""
    observed: str = ""
    asset_id: Optional[str] = None
    asset_name: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DriftIssue":
        return cls(
            dimension=d.get("dimension", "other"),
            severity=d.get("severity", "medium"),
            description=d.get("description", ""),
            expected=d.get("expected", ""),
            observed=d.get("observed", ""),
            asset_id=d.get("asset_id"),
            asset_name=d.get("asset_name"),
        )


@dataclass
class DriftReport:
    shot_id: str
    status: str  # pass | warn | fail
    score: int  # 0-100, higher is more consistent
    summary: str
    issues: list[DriftIssue] = field(default_factory=list)
    model_used: str = ""  # vision model id, or a "mock" marker

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DriftReport":
        return cls(
            shot_id=d.get("shot_id", ""),
            status=d.get("status", "warn"),
            score=int(d.get("score", 0)),
            summary=d.get("summary", ""),
            issues=[DriftIssue.from_dict(x) for x in d.get("issues", [])],
            model_used=d.get("model_used", ""),
        )


@dataclass
class RepairedPrompt:
    model: str
    prompt: str
    original_prompt: str
    negative_prompt: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
