"""Per-model prompt grammars.

Each generation model expresses prompts differently — Kling takes a positive
prompt plus a dedicated negative-prompt field and responds to weight syntax;
Seedance is a single comma-separated tag string with no negative field; Veo
and Sora want natural-language cinematic prose. Continuo's repair engine
targets whichever grammar the creator already pays for, which is the whole
"be the referee, not the stadium" pitch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelGrammar:
    key: str
    display_name: str
    style: str  # "tags" | "prose"
    supports_negative_prompt: bool
    max_chars: int
    # Weight syntax applied to correction clauses to emphasise them, or "" if none.
    weight_open: str = ""
    weight_close: str = ""
    notes: str = ""

    def emphasise(self, clause: str) -> str:
        if not self.weight_open:
            return clause
        return f"{self.weight_open}{clause}{self.weight_close}"


GRAMMARS: dict[str, ModelGrammar] = {
    "kling": ModelGrammar(
        key="kling",
        display_name="Kling",
        style="tags",
        supports_negative_prompt=True,
        max_chars=2500,
        weight_open="(",
        weight_close=":1.3)",
        notes="Comma-separated subject + camera tags, dedicated negative prompt, "
        "responds to (clause:weight) emphasis.",
    ),
    "seedance": ModelGrammar(
        key="seedance",
        display_name="Seedance",
        style="tags",
        supports_negative_prompt=False,
        max_chars=800,
        notes="Single comma-separated tag string, no negative field — negatives "
        "are folded inline as 'no X' tags.",
    ),
    "veo": ModelGrammar(
        key="veo",
        display_name="Veo 3",
        style="prose",
        supports_negative_prompt=False,
        max_chars=2000,
        notes="Natural-language cinematic description; corrections woven into prose.",
    ),
    "sora": ModelGrammar(
        key="sora",
        display_name="Sora 2",
        style="prose",
        supports_negative_prompt=False,
        max_chars=2000,
        notes="Descriptive scene prose; continuity constraints stated as directions.",
    ),
}

DEFAULT_GRAMMAR = "kling"


def get_grammar(model_key: str) -> ModelGrammar:
    return GRAMMARS.get((model_key or "").lower(), GRAMMARS[DEFAULT_GRAMMAR])
