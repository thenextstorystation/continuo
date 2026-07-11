"""Runtime configuration, read from the environment."""

from __future__ import annotations

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("CONTINUO_DATA_DIR", os.path.join(_ROOT, "data"))
BIBLE_PATH = os.path.join(DATA_DIR, "bible.json")
USAGE_PATH = os.path.join(DATA_DIR, "usage.json")

# QC metering. Estimated cost of one live vision screening, used to surface the
# "keep QC below ~10% of generation spend" guardrail from the product brief.
VISION_COST_PER_SCREEN = float(os.environ.get("CONTINUO_VISION_COST_PER_SCREEN", "0.02"))

# One-click corrected regeneration. Defaults to a dry-run provider that never
# calls out or spends credits; set CONTINUO_REGEN_PROVIDER=http with an endpoint
# + key to submit to a real text-to-video provider (Kling / Higgsfield / ...).
REGEN_PROVIDER = os.environ.get("CONTINUO_REGEN_PROVIDER", "dry_run")
REGEN_ENDPOINT = os.environ.get("CONTINUO_REGEN_ENDPOINT")
REGEN_API_KEY = os.environ.get("CONTINUO_REGEN_API_KEY", "")

# Polling behaviour for submit-then-poll providers (Kling / Higgsfield).
REGEN_POLL_INTERVAL = float(os.environ.get("CONTINUO_REGEN_POLL_INTERVAL", "2.0"))
REGEN_MAX_POLLS = int(os.environ.get("CONTINUO_REGEN_MAX_POLLS", "30"))

# Concrete provider endpoints. Empty by default — set the pair for the provider
# you use, plus its API key, and select it via CONTINUO_REGEN_PROVIDER. The
# status URL is a template containing '{id}'. Verify the exact URLs and payload
# against the provider's current API docs before going live.
KLING_SUBMIT_URL = os.environ.get("CONTINUO_KLING_SUBMIT_URL", "")
KLING_STATUS_URL = os.environ.get("CONTINUO_KLING_STATUS_URL", "")
KLING_API_KEY = os.environ.get("CONTINUO_KLING_API_KEY", "")

HIGGSFIELD_SUBMIT_URL = os.environ.get("CONTINUO_HIGGSFIELD_SUBMIT_URL", "")
HIGGSFIELD_STATUS_URL = os.environ.get("CONTINUO_HIGGSFIELD_STATUS_URL", "")
HIGGSFIELD_API_KEY = os.environ.get("CONTINUO_HIGGSFIELD_API_KEY", "")

# Usage-based billing. Events go to Stripe when STRIPE_API_KEY is set, else to a
# local JSONL ledger. Set CONTINUO_BILLING=off to drop them entirely.
BILLING_MODE = os.environ.get("CONTINUO_BILLING", "auto")
BILLING_LEDGER_PATH = os.path.join(DATA_DIR, "billing_ledger.jsonl")
STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "")
STRIPE_CUSTOMER = os.environ.get("STRIPE_CUSTOMER")

# Vision judging defaults to the most capable Claude model. Keep vision cost
# below ~10% of the customer's generation spend (per the product brief) by
# swapping to a cheaper tier here if needed.
VISION_MODEL = os.environ.get("CONTINUO_VISION_MODEL", "claude-opus-4-8")


def has_api_key() -> bool:
    """True when an Anthropic credential is available for live vision screening."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
