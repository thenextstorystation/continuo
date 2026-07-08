"""Usage-based billing push.

The metering layer emits a billing event per billable action (live vision
screening, corrected regeneration). This module routes those events to a sink:

- ``StripeSink`` pushes Stripe meter events when a key is configured — the
  per-minute / per-screening billing the product monetises on.
- ``LedgerSink`` (default when no Stripe key) appends events to a local JSONL
  ledger you can reconcile or replay.
- ``NullSink`` drops events (used in tests and when billing is off).

Billing never blocks the core QC flow: the meter wraps every push in a
try/except so a billing failure can't break a screening.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional, Protocol

from . import config


class BillingSink(Protocol):
    def push(self, meter: str, quantity: int, customer: Optional[str] = None) -> bool: ...


class NullSink:
    def push(self, meter: str, quantity: int, customer: Optional[str] = None) -> bool:
        return True


class LedgerSink:
    """Append billing events to a local JSONL ledger."""

    def __init__(self, path: Optional[str] = None):
        self.path = path if path is not None else config.BILLING_LEDGER_PATH

    def push(self, meter: str, quantity: int, customer: Optional[str] = None) -> bool:
        if not self.path:
            return False
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "meter": meter,
            "quantity": quantity,
            "customer": customer or config.STRIPE_CUSTOMER,
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
        return True


class StripeSink:
    """Push Stripe billing meter events."""

    def __init__(self, api_key: str, customer: Optional[str] = None):
        self.api_key = api_key
        self.customer = customer

    def push(self, meter: str, quantity: int, customer: Optional[str] = None) -> bool:
        import stripe  # lazy — only needed when Stripe billing is active

        stripe.api_key = self.api_key
        stripe.billing.MeterEvent.create(
            event_name=meter,
            payload={
                "stripe_customer_id": customer or self.customer or "",
                "value": str(quantity),
            },
        )
        return True


def _stripe_available() -> bool:
    try:
        import stripe  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def get_sink() -> BillingSink:
    """Pick the active billing sink from configuration.

    Stripe when a key is set and the SDK is importable; otherwise the local
    ledger. Set ``CONTINUO_BILLING=off`` to drop events entirely.
    """
    if config.BILLING_MODE.lower() == "off":
        return NullSink()
    if config.STRIPE_API_KEY and _stripe_available():
        return StripeSink(config.STRIPE_API_KEY, config.STRIPE_CUSTOMER)
    return LedgerSink(config.BILLING_LEDGER_PATH)
