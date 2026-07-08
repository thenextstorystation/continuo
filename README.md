# 🎬 Continuo

**The continuity supervisor for AI filmmaking** — a model-agnostic QC layer that
catches and fixes drift across every generated shot.

Every AI filmmaker fights the same war: the hero's jacket changes colour in shot 7,
the scar switches cheeks, the kitchen gains a window. Continuo is the digital
equivalent of the on-set continuity supervisor ("script girl") that AI production
pipelines don't have yet. You register an **asset bible** (characters, wardrobe,
props, locations); every generated clip is screened by a vision model against that
bible; drift is flagged frame-accurately; and Continuo **rewrites the offending
prompt in the grammar of whichever model you use** and offers a corrected
regeneration.

> Be the referee, not the stadium: Continuo is model-agnostic (works with whatever
> you already pay for — Kling, Seedance, Veo, Sora), post-hoc (catches what
> generation-time tricks miss), and prescriptive (rewrites the prompt instead of
> just flagging it).

## The loop

```
Asset bible ─▶ Screen a frame ─▶ Report card ─▶ Prompt repair ─▶ One-click regeneration
 (canonical)   (vision vs bible)  (drift+score)  (per-model grammar)  (provider, metered)
```

1. **Register** the asset bible — the canonical definitions every shot is judged against.
2. **Screen** a generated frame. A Claude vision model compares it to the bible and
   returns a structured drift report across five dimensions — `face`, `wardrobe`,
   `lighting`, `set_geometry`, `prop` — each with the expected vs. observed appearance.
3. **Read the report card** — an overall status (`pass` / `warn` / `fail`), a
   consistency score, and a per-issue breakdown.
4. **Repair the prompt** — Continuo rewrites the prompt to correct medium/high drift,
   in the target model's own grammar (Kling's negative-prompt field, Seedance's inline
   tags, Veo/Sora prose).
5. **Regenerate** — submit the corrected prompt straight back to a text-to-video
   provider in one click. Defaults to a dry-run provider (no external call, no credits);
   point it at a real provider (Kling / Higgsfield) via env. Every screening and
   regeneration is metered so QC spend stays visible against generation spend.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional — enables live, frame-accurate vision screening.
# Without a credential Continuo runs a clearly-labelled mock so the whole loop still works.
export ANTHROPIC_API_KEY=sk-ant-...

uvicorn continuo.app:app --reload
# open http://127.0.0.1:8000  → "Load sample" → "Register bible" → "Screen for drift"
```

## API

| Method & path      | Purpose                                                    |
| ------------------ | ---------------------------------------------------------- |
| `POST /api/bible`  | Register / replace the asset bible (JSON body)             |
| `GET  /api/bible`  | Fetch the current bible                                    |
| `GET  /api/grammars` | List supported model grammars                            |
| `POST /api/screen` | Screen a frame (`multipart`: `shot` JSON + optional `frame`) → `DriftReport` |
| `POST /api/repair` | Rewrite a prompt to fix drift → `RepairedPrompt`           |
| `POST /api/regenerate` | Submit a corrected prompt for regeneration → `RegenJob` |
| `GET  /api/providers` | List regeneration providers and which is active         |
| `GET  /api/usage`  | QC metering summary (screenings, regenerations, est. cost) |
| `GET  /api/health` | Service + vision + provider status                         |

### Example: screen then repair (mock mode, no key needed)

```bash
curl -s -X POST localhost:8000/api/bible \
  -H 'content-type: application/json' -d @sample_data/asset_bible.json

curl -s -X POST localhost:8000/api/screen \
  -F 'shot={"id":"shot-7","shot_number":7,"prompt":"Kaito leans on the counter","model":"kling","expected_wardrobe":["jacket"],"expected_characters":["kaito"]}'
```

## Architecture

```
continuo/
  models.py         dataclass domain model (bible, shot, drift report) — zero deps
  grammars.py       per-model prompt grammars (Kling / Seedance / Veo / Sora)
  prompt_repair.py  deterministic, prescriptive prompt-repair engine
  vision.py         Claude vision screening + structured drift report (mock fallback)
  providers.py      one-click regeneration (dry-run default + HTTP adapter)
  metering.py       QC usage accounting + vision-cost estimate
  bible.py          JSON-backed asset-bible store
  app.py            FastAPI service + report-card UI
  static/index.html the report card
tests/              offline unit tests (no API key required)
```

Vision judging uses Claude (`claude-opus-4-8` by default, override with
`CONTINUO_VISION_MODEL`) with **structured outputs**, so the drift report always
matches a fixed JSON schema. Because drift-checking rides on a generic vision model,
Continuo improves automatically as those models improve.

## Tests

```bash
pip install -r requirements-dev.txt
pytest            # runs fully offline — the vision layer is exercised via its mock path
```

## Roadmap

- Wire a live provider adapter end to end (Kling / Higgsfield) behind the HTTP provider
- Clip-level screening (sample N frames per shot) instead of single stills
- Reference-image embedding match for faces, and Stripe push from the metering layer
- Expand QC beyond continuity: audio sync, physics errors, hands

## Status

MVP scaffold — the full screen → report → repair loop works end to end, offline via a
mock screener and live with an Anthropic credential.
