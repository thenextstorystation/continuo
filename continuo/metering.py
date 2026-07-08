"""QC metering.

Tracks how much screening and regeneration the instance has done, and estimates
vision cost — the metric the product's value story hinges on (QC must stay well
below the customer's generation spend). This is the accounting layer a
per-minute billing integration (e.g. Stripe) would push from.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any, Optional

from . import config
from .billing import BillingSink, NullSink

# Billing meter names (Stripe event names / ledger keys).
METER_SCREENING = "continuo.vision_screening"
METER_REGENERATION = "continuo.regeneration"


@dataclass
class Usage:
    shots_screened: int = 0
    live_screenings: int = 0
    mock_screenings: int = 0
    regenerations: int = 0
    est_vision_cost_usd: float = 0.0


class Meter:
    def __init__(self, path: Optional[str] = None, sink: Optional[BillingSink] = None):
        self.path = path if path is not None else config.USAGE_PATH
        self.sink: BillingSink = sink if sink is not None else NullSink()
        self._lock = threading.Lock()
        self.usage = self._load()

    def _bill(self, meter: str, quantity: int) -> None:
        # Billing must never break the core QC flow.
        try:
            self.sink.push(meter, quantity)
        except Exception:  # noqa: BLE001
            pass

    def _load(self) -> Usage:
        if self.path and os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            fields = set(Usage.__annotations__)
            return Usage(**{k: v for k, v in raw.items() if k in fields})
        return Usage()

    def _save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self.usage), fh, indent=2)

    def record_screen(self, live: bool) -> None:
        with self._lock:
            self.usage.shots_screened += 1
            if live:
                self.usage.live_screenings += 1
                self.usage.est_vision_cost_usd += config.VISION_COST_PER_SCREEN
            else:
                self.usage.mock_screenings += 1
            self._save()
        # Only live screenings are billable (mock costs nothing).
        if live:
            self._bill(METER_SCREENING, 1)

    def record_regen(self) -> None:
        with self._lock:
            self.usage.regenerations += 1
            self._save()
        self._bill(METER_REGENERATION, 1)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            data = asdict(self.usage)
        data["est_vision_cost_usd"] = round(self.usage.est_vision_cost_usd, 4)
        data["cost_per_screen_usd"] = config.VISION_COST_PER_SCREEN
        return data
