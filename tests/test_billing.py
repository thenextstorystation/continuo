"""Tests for the billing sinks and metering integration."""

import json

import continuo.config as cfg
from continuo.billing import LedgerSink, NullSink, StripeSink, get_sink
from continuo.metering import METER_REGENERATION, METER_SCREENING, Meter


def _read_ledger(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_ledger_sink_appends_events(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    sink = LedgerSink(path)
    assert sink.push("continuo.vision_screening", 1, customer="cus_1")
    assert sink.push("continuo.regeneration", 2)
    rows = _read_ledger(path)
    assert [r["meter"] for r in rows] == ["continuo.vision_screening", "continuo.regeneration"]
    assert rows[0]["quantity"] == 1 and rows[0]["customer"] == "cus_1"


def test_null_sink_is_noop():
    assert NullSink().push("x", 1) is True


def test_get_sink_defaults_to_ledger_without_stripe_key(monkeypatch):
    monkeypatch.setattr(cfg, "BILLING_MODE", "auto")
    monkeypatch.setattr(cfg, "STRIPE_API_KEY", "")
    assert isinstance(get_sink(), LedgerSink)


def test_get_sink_off_returns_null(monkeypatch):
    monkeypatch.setattr(cfg, "BILLING_MODE", "off")
    assert isinstance(get_sink(), NullSink)


def test_get_sink_falls_back_to_ledger_when_stripe_missing(monkeypatch):
    # A key is set but the stripe SDK isn't installed -> graceful ledger fallback.
    monkeypatch.setattr(cfg, "BILLING_MODE", "auto")
    monkeypatch.setattr(cfg, "STRIPE_API_KEY", "sk_test_123")
    monkeypatch.setattr("continuo.billing._stripe_available", lambda: False)
    assert isinstance(get_sink(), LedgerSink)


def test_get_sink_selects_stripe_when_available(monkeypatch):
    monkeypatch.setattr(cfg, "BILLING_MODE", "auto")
    monkeypatch.setattr(cfg, "STRIPE_API_KEY", "sk_test_123")
    monkeypatch.setattr("continuo.billing._stripe_available", lambda: True)
    assert isinstance(get_sink(), StripeSink)


def test_meter_bills_live_screen_and_regen_but_not_mock(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    m = Meter(path=str(tmp_path / "usage.json"), sink=LedgerSink(ledger))

    m.record_screen(live=False)  # mock -> not billed
    m.record_screen(live=True)   # billed
    m.record_regen()             # billed

    rows = _read_ledger(ledger)
    meters = [r["meter"] for r in rows]
    assert meters == [METER_SCREENING, METER_REGENERATION]


def test_billing_failure_never_breaks_metering(tmp_path):
    class BoomSink:
        def push(self, *a, **k):
            raise RuntimeError("stripe is down")

    m = Meter(path=str(tmp_path / "usage.json"), sink=BoomSink())
    m.record_screen(live=True)  # must not raise
    m.record_regen()  # must not raise
    assert m.summary()["live_screenings"] == 1
