"""Tests for QC metering."""

import continuo.config as cfg
from continuo.metering import Meter


def test_records_and_estimates_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "VISION_COST_PER_SCREEN", 0.02)
    m = Meter(path=str(tmp_path / "usage.json"))

    m.record_screen(live=True)
    m.record_screen(live=False)
    m.record_regen()

    s = m.summary()
    assert s["shots_screened"] == 2
    assert s["live_screenings"] == 1
    assert s["mock_screenings"] == 1
    assert s["regenerations"] == 1
    # only live screenings incur estimated vision cost
    assert s["est_vision_cost_usd"] == 0.02
    assert s["cost_per_screen_usd"] == 0.02


def test_usage_persists_across_instances(tmp_path):
    path = str(tmp_path / "usage.json")
    m1 = Meter(path=path)
    m1.record_screen(live=True)
    m1.record_regen()

    m2 = Meter(path=path)  # fresh instance loads from disk
    s = m2.summary()
    assert s["shots_screened"] == 1
    assert s["regenerations"] == 1


def test_mock_screen_is_free(tmp_path):
    m = Meter(path=str(tmp_path / "usage.json"))
    m.record_screen(live=False)
    assert m.summary()["est_vision_cost_usd"] == 0.0
