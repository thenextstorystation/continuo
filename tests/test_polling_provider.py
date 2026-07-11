"""Tests for the submit-then-poll regeneration providers (Kling / Higgsfield)."""

import continuo.config as cfg
from continuo.providers import (
    PollingProvider,
    ProviderProfile,
    get_provider,
)


class FakeTransport:
    """Simulates an async video API: one submit, then a scripted list of polls."""

    def __init__(self, submit=(200, {"id": "job1", "status": "queued"}), polls=None):
        self.submit_response = submit
        self.poll_responses = list(polls or [])
        self.posts = []
        self.gets = []

    def post(self, url, payload, headers):
        self.posts.append((url, payload))
        return self.submit_response

    def get(self, url, headers):
        self.gets.append(url)
        if self.poll_responses:
            return self.poll_responses.pop(0)
        return (200, {"status": "processing"})


def _profile(supports_negative=True, max_polls=30):
    return ProviderProfile(
        key="kling",
        display_name="Kling",
        submit_url="https://api.kling.test/submit",
        status_url="https://api.kling.test/jobs/{id}",
        api_key="k",
        supports_negative=supports_negative,
        poll_interval=0.0,
        max_polls=max_polls,
    )


def _noop_sleep(_):
    return None


def test_submit_then_poll_to_completion():
    t = FakeTransport(
        submit=(200, {"id": "job1", "status": "queued"}),
        polls=[
            (200, {"status": "processing"}),
            (200, {"status": "completed", "output_url": "https://cdn.test/v.mp4"}),
        ],
    )
    p = PollingProvider(_profile(), transport=t, sleep=_noop_sleep)
    job = p.regenerate(model="kling", prompt="fixed prompt", negative_prompt="red jacket")

    assert job.status == "succeeded"
    assert job.id == "job1"
    assert job.output_url == "https://cdn.test/v.mp4"
    assert "after 2 poll(s)" in job.detail
    # negative prompt is sent (Kling supports it) and the status URL is templated
    assert t.posts[0][1]["negative_prompt"] == "red jacket"
    assert t.gets[0] == "https://api.kling.test/jobs/job1"


def test_immediate_completion_needs_no_polls():
    t = FakeTransport(submit=(200, {"id": "j", "status": "completed", "output_url": "u"}))
    p = PollingProvider(_profile(), transport=t, sleep=_noop_sleep)
    job = p.regenerate(model="kling", prompt="x")
    assert job.status == "succeeded"
    assert job.output_url == "u"
    assert t.gets == []  # never polled


def test_failure_status_surfaces_as_failed():
    t = FakeTransport(
        submit=(200, {"id": "j", "status": "queued"}),
        polls=[(200, {"status": "failed"})],
    )
    p = PollingProvider(_profile(), transport=t, sleep=_noop_sleep)
    job = p.regenerate(model="kling", prompt="x")
    assert job.status == "failed"
    assert "failed" in job.detail


def test_submit_http_error():
    t = FakeTransport(submit=(500, {"error": "boom"}))
    p = PollingProvider(_profile(), transport=t, sleep=_noop_sleep)
    job = p.regenerate(model="kling", prompt="x")
    assert job.status == "failed"
    assert "HTTP 500" in job.detail


def test_timeout_returns_submitted():
    # never reaches a terminal status within max_polls
    t = FakeTransport(
        submit=(200, {"id": "j", "status": "queued"}),
        polls=[(200, {"status": "processing"})] * 5,
    )
    p = PollingProvider(_profile(max_polls=2), transport=t, sleep=_noop_sleep)
    job = p.regenerate(model="kling", prompt="x")
    assert job.status == "submitted"
    assert "Still processing after 2 poll(s)" in job.detail


def test_negative_prompt_omitted_when_unsupported():
    t = FakeTransport(submit=(200, {"id": "j", "status": "completed"}))
    p = PollingProvider(_profile(supports_negative=False), transport=t, sleep=_noop_sleep)
    p.regenerate(model="higgsfield", prompt="x", negative_prompt="red jacket")
    assert "negative_prompt" not in t.posts[0][1]


def test_get_provider_selects_kling_when_configured(monkeypatch):
    monkeypatch.setattr(cfg, "REGEN_PROVIDER", "kling")
    monkeypatch.setattr(cfg, "KLING_SUBMIT_URL", "https://api.kling.test/submit")
    monkeypatch.setattr(cfg, "KLING_STATUS_URL", "https://api.kling.test/jobs/{id}")
    monkeypatch.setattr(cfg, "KLING_API_KEY", "k")
    provider = get_provider()
    assert isinstance(provider, PollingProvider)
    assert provider.key == "kling"


def test_get_provider_falls_back_to_dry_run_when_kling_unconfigured(monkeypatch):
    monkeypatch.setattr(cfg, "REGEN_PROVIDER", "kling")
    monkeypatch.setattr(cfg, "KLING_SUBMIT_URL", "")
    assert get_provider().key == "dry_run"
