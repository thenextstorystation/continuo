"""Tests for the regeneration provider layer."""

from continuo.providers import (
    DryRunProvider,
    HttpRegenProvider,
    available_providers,
    get_provider,
)


def test_dry_run_never_calls_out():
    p = DryRunProvider()
    job = p.regenerate(model="kling", prompt="a corrected prompt", negative_prompt="red jacket")
    assert job.status == "dry_run"
    assert job.provider == "dry_run"
    assert job.prompt == "a corrected prompt"
    assert job.negative_prompt == "red jacket"
    assert job.output_url is None
    assert job.id.startswith("rgn_")
    assert "no credits spent" in job.detail


def test_get_provider_defaults_to_dry_run(monkeypatch):
    monkeypatch.delenv("CONTINUO_REGEN_PROVIDER", raising=False)
    # config is read at import time, so patch the resolved values directly
    import continuo.config as cfg

    monkeypatch.setattr(cfg, "REGEN_PROVIDER", "dry_run")
    monkeypatch.setattr(cfg, "REGEN_ENDPOINT", None)
    assert get_provider().key == "dry_run"


def test_get_provider_selects_http_when_configured(monkeypatch):
    import continuo.config as cfg

    monkeypatch.setattr(cfg, "REGEN_PROVIDER", "http")
    monkeypatch.setattr(cfg, "REGEN_ENDPOINT", "https://example.test/generate")
    monkeypatch.setattr(cfg, "REGEN_API_KEY", "secret")
    provider = get_provider()
    assert isinstance(provider, HttpRegenProvider)
    assert provider.endpoint == "https://example.test/generate"


def test_http_provider_returns_failed_on_bad_endpoint():
    # An unreachable endpoint should surface as a failed job, not raise.
    p = HttpRegenProvider("http://127.0.0.1:1/nope", "k")
    job = p.regenerate(model="kling", prompt="x")
    assert job.status == "failed"
    assert "failed" in job.detail.lower()


def test_available_providers_lists_both():
    info = available_providers()
    keys = {p["key"] for p in info["providers"]}
    assert {"dry_run", "http"} <= keys
    assert "active" in info
