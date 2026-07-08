"""Prescriptive prompt-repair engine.

Given the drift issues a screening turned up, plus the original prompt and the
target model, produce a corrected prompt in that model's grammar. This is the
"prescriptive" half of Continuo — it rewrites the prompt instead of merely
flagging the problem.

The repair is deterministic and rule-based so it is fully testable offline and
its output is predictable; the vision screening (which needs a model) is the
only part that reaches for the Anthropic API.
"""

from __future__ import annotations

from typing import Optional

from .grammars import ModelGrammar, get_grammar
from .models import DriftIssue, RepairedPrompt

# Only issues at or above this severity trigger a rewrite. Low-severity drift is
# reported but left alone to avoid over-constraining the prompt.
_REPAIR_SEVERITIES = ("medium", "high")


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip()


def issue_to_correction(issue: DriftIssue) -> tuple[str, Optional[str]]:
    """Turn one drift issue into a (positive clause, negative clause) pair.

    The positive clause restates the canonical/expected appearance; the negative
    clause names the drifted appearance to suppress. Either may be empty.
    """
    name = _clean(issue.asset_name or "")
    expected = _clean(issue.expected)
    observed = _clean(issue.observed)
    dim = issue.dimension

    if dim == "face":
        subj = name or "the character"
        pos = f"{subj}'s face with {expected}" if expected else f"{subj} with consistent facial features"
    elif dim == "wardrobe":
        # `expected` already describes the garment (e.g. "navy blue bomber jacket");
        # use it directly rather than a "X wearing Y" frame that assumes a character name.
        pos = expected or (f"the established {name}" if name else "the established wardrobe")
    elif dim == "lighting":
        pos = f"{expected} lighting" if expected else "lighting matching the previous shots"
    elif dim == "set_geometry":
        loc = name or "the location"
        pos = f"{loc} featuring {expected}" if expected else f"{loc} with consistent set geometry"
    elif dim == "prop":
        pos = expected or (f"the {name}" if name else "the established prop")
    else:  # other / fallback
        pos = expected or (name and f"consistent {name}") or ""

    neg = observed or None
    return _clean(pos), (_clean(neg) if neg else None)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rstrip(", ;.")
    return cut + "…"


def _build_tags(
    grammar: ModelGrammar, original: str, positives: list[str], negatives: list[str]
) -> RepairedPrompt:
    original = _clean(original)
    emphasised = [grammar.emphasise(p) for p in positives]
    parts = [original] + emphasised if original else emphasised

    negative_prompt: Optional[str] = None
    notes: list[str] = []

    if grammar.supports_negative_prompt:
        negative_prompt = ", ".join(dict.fromkeys(negatives)) or None
        if negative_prompt:
            notes.append(f"{grammar.display_name}: continuity fixes added to the negative prompt.")
    else:
        # No negative field — fold negatives inline as 'no X' tags.
        for neg in dict.fromkeys(negatives):
            parts.append(f"no {neg}")
        if negatives:
            notes.append(
                f"{grammar.display_name} has no negative-prompt field — drift folded in as 'no X' tags."
            )

    prompt = ", ".join(p for p in parts if p)
    return RepairedPrompt(
        model=grammar.key,
        prompt=_truncate(prompt, grammar.max_chars),
        original_prompt=original,
        negative_prompt=negative_prompt,
        notes=notes,
    )


def _build_prose(
    grammar: ModelGrammar, original: str, positives: list[str], negatives: list[str]
) -> RepairedPrompt:
    original = _clean(original).rstrip(". ")
    sentences = [original + "."] if original else []

    if positives:
        joined = "; ".join(positives)
        sentences.append(f"Maintain continuity: {joined}.")
    if negatives:
        joined = ", ".join(dict.fromkeys(negatives))
        sentences.append(f"Do not depict {joined}.")

    prompt = " ".join(sentences).strip()
    notes = [f"{grammar.display_name}: continuity constraints woven into the scene description."]
    return RepairedPrompt(
        model=grammar.key,
        prompt=_truncate(prompt, grammar.max_chars),
        original_prompt=original,
        negative_prompt=None,
        notes=notes,
    )


def repair_prompt(
    original_prompt: str, model_key: str, issues: list[DriftIssue]
) -> RepairedPrompt:
    """Rewrite ``original_prompt`` for ``model_key`` to correct the given drift issues."""
    grammar = get_grammar(model_key)

    positives: list[str] = []
    negatives: list[str] = []
    repaired_issue_notes: list[str] = []

    for issue in issues:
        if issue.severity not in _REPAIR_SEVERITIES:
            continue
        pos, neg = issue_to_correction(issue)
        if pos:
            positives.append(pos)
        if neg:
            negatives.append(neg)
        repaired_issue_notes.append(f"[{issue.dimension}] {_clean(issue.description)}")

    # de-dupe while preserving order
    positives = list(dict.fromkeys(positives))
    negatives = list(dict.fromkeys(negatives))

    if grammar.style == "prose":
        result = _build_prose(grammar, original_prompt, positives, negatives)
    else:
        result = _build_tags(grammar, original_prompt, positives, negatives)

    if not positives and not negatives:
        result.notes.insert(
            0, "No medium/high-severity drift to repair — prompt returned unchanged."
        )
    else:
        result.notes.extend(repaired_issue_notes)

    return result
