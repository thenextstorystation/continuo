"""One-click corrected regeneration.

Continuo screens a shot, repairs the prompt, and can then submit the corrected
prompt straight back to a text-to-video provider — closing the loop from
"drift detected" to "fixed take" in one click.

Providers are pluggable. The default is a dry-run provider that never calls out
and never spends credits, so the loop is fully demonstrable offline. Setting
``CONTINUO_REGEN_PROVIDER=http`` with an endpoint + key routes to a real
provider (Kling, Higgsfield, etc.) via a conventional REST shape.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Protocol

from . import config


@dataclass
class RegenJob:
    id: str
    provider: str
    model: str
    status: str  # dry_run | submitted | succeeded | failed
    prompt: str
    negative_prompt: Optional[str] = None
    output_url: Optional[str] = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _job_id() -> str:
    return "rgn_" + uuid.uuid4().hex[:12]


class DryRunProvider:
    key = "dry_run"
    display_name = "Dry run (no external call)"

    def regenerate(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "16:9",
        duration: int = 5,
    ) -> RegenJob:
        return RegenJob(
            id=_job_id(),
            provider=self.key,
            model=model,
            status="dry_run",
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_url=None,
            detail=(
                f"Simulated submission — no video generated, no credits spent. Would submit a "
                f"{duration}s {aspect_ratio} clip to '{model}'. Set CONTINUO_REGEN_PROVIDER=http "
                "with CONTINUO_REGEN_ENDPOINT and CONTINUO_REGEN_API_KEY to enable a live provider."
            ),
        )


class HttpRegenProvider:
    """Generic REST adapter for a text-to-video provider.

    POSTs a conventional JSON body to ``endpoint`` with a bearer token. Field
    names follow the common shape (model/prompt/negative_prompt/aspect_ratio/
    duration); adjust per the specific provider's API if needed.
    """

    key = "http"
    display_name = "HTTP provider (Kling / Higgsfield / …)"

    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key

    def regenerate(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "16:9",
        duration: int = 5,
    ) -> RegenJob:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        job_id = _job_id()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - configured endpoint
                body = json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            return RegenJob(
                id=job_id, provider=self.key, model=model, status="failed",
                prompt=prompt, negative_prompt=negative_prompt,
                detail=f"Provider returned HTTP {exc.code}: {exc.reason}",
            )
        except Exception as exc:  # noqa: BLE001 - surface any transport error to the caller
            return RegenJob(
                id=job_id, provider=self.key, model=model, status="failed",
                prompt=prompt, negative_prompt=negative_prompt,
                detail=f"Request failed: {exc}",
            )

        return RegenJob(
            id=str(body.get("id") or job_id),
            provider=self.key,
            model=model,
            status=body.get("status", "submitted"),
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_url=body.get("output_url") or body.get("video_url"),
            detail="Submitted to live provider.",
        )


# ---------------------------------------------------------------------------
# Submit-then-poll providers (Kling / Higgsfield)
# ---------------------------------------------------------------------------
#
# Real text-to-video APIs are asynchronous: you POST a generation request, get a
# job id, then poll a status endpoint until the clip is ready. PollingProvider
# implements that flow. The HTTP transport is injectable so the logic is tested
# against a fake — no live credentials required to prove correctness.


class Transport(Protocol):
    def post(self, url: str, payload: dict, headers: dict) -> tuple[int, dict]: ...
    def get(self, url: str, headers: dict) -> tuple[int, dict]: ...


class UrllibTransport:
    """Default transport backed by urllib (stdlib, no extra dependency)."""

    def _request(self, url, headers, data=None, method="GET"):
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - configured URL
                body = json.loads(resp.read().decode("utf-8") or "{}")
                return resp.status, body
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read().decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                body = {"error": exc.reason}
            return exc.code, body

    def post(self, url, payload, headers):
        return self._request(url, headers, data=json.dumps(payload).encode("utf-8"), method="POST")

    def get(self, url, headers):
        return self._request(url, headers, method="GET")


@dataclass
class ProviderProfile:
    key: str
    display_name: str
    submit_url: str
    status_url: str  # template containing '{id}'
    api_key: str
    supports_negative: bool = True
    poll_interval: float = 2.0
    max_polls: int = 30
    submit_id_field: str = "id"
    status_field: str = "status"
    output_field: str = "output_url"
    done_values: tuple[str, ...] = ("completed", "succeeded", "success", "done")
    failed_values: tuple[str, ...] = ("failed", "error", "canceled", "cancelled")


def _build_payload(profile, model, prompt, negative, aspect_ratio, duration) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
    }
    if negative and profile.supports_negative:
        payload["negative_prompt"] = negative
    return payload


class PollingProvider:
    def __init__(
        self,
        profile: ProviderProfile,
        transport: Optional[Transport] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.profile = profile
        self.key = profile.key
        self.display_name = profile.display_name
        self.transport = transport or UrllibTransport()
        self.sleep = sleep

    def regenerate(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "16:9",
        duration: int = 5,
    ) -> RegenJob:
        pf = self.profile
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {pf.api_key}",
        }
        payload = _build_payload(pf, model, prompt, negative_prompt, aspect_ratio, duration)
        job_id = _job_id()

        try:
            code, body = self.transport.post(pf.submit_url, payload, headers)
        except Exception as exc:  # noqa: BLE001
            return RegenJob(job_id, pf.key, model, "failed", prompt, negative_prompt,
                            detail=f"Submit failed: {exc}")
        if code >= 400:
            return RegenJob(job_id, pf.key, model, "failed", prompt, negative_prompt,
                            detail=f"Submit returned HTTP {code}: {body}")

        job_id = str(body.get(pf.submit_id_field) or job_id)
        status = str(body.get(pf.status_field, "submitted")).lower()
        output = body.get(pf.output_field)

        polls = 0
        while (
            status not in pf.done_values
            and status not in pf.failed_values
            and polls < pf.max_polls
        ):
            self.sleep(pf.poll_interval)
            polls += 1
            try:
                code, body = self.transport.get(pf.status_url.format(id=job_id), headers)
            except Exception as exc:  # noqa: BLE001
                return RegenJob(job_id, pf.key, model, "failed", prompt, negative_prompt,
                                output, f"Poll failed: {exc}")
            status = str(body.get(pf.status_field, status)).lower()
            output = body.get(pf.output_field) or output

        if status in pf.done_values:
            return RegenJob(job_id, pf.key, model, "succeeded", prompt, negative_prompt,
                            output, f"Completed after {polls} poll(s).")
        if status in pf.failed_values:
            return RegenJob(job_id, pf.key, model, "failed", prompt, negative_prompt,
                            output, f"Provider reported status '{status}'.")
        return RegenJob(job_id, pf.key, model, "submitted", prompt, negative_prompt,
                        output, f"Still processing after {polls} poll(s); job {job_id}.")


def kling_profile() -> ProviderProfile:
    return ProviderProfile(
        key="kling",
        display_name="Kling",
        submit_url=config.KLING_SUBMIT_URL,
        status_url=config.KLING_STATUS_URL,
        api_key=config.KLING_API_KEY,
        supports_negative=True,  # Kling has a dedicated negative prompt
        poll_interval=config.REGEN_POLL_INTERVAL,
        max_polls=config.REGEN_MAX_POLLS,
    )


def higgsfield_profile() -> ProviderProfile:
    return ProviderProfile(
        key="higgsfield",
        display_name="Higgsfield",
        submit_url=config.HIGGSFIELD_SUBMIT_URL,
        status_url=config.HIGGSFIELD_STATUS_URL,
        api_key=config.HIGGSFIELD_API_KEY,
        supports_negative=False,
        poll_interval=config.REGEN_POLL_INTERVAL,
        max_polls=config.REGEN_MAX_POLLS,
    )


def get_provider():
    """Return the configured regeneration provider (dry-run unless one is set up)."""
    choice = config.REGEN_PROVIDER.lower()
    if choice == "kling" and config.KLING_SUBMIT_URL:
        return PollingProvider(kling_profile())
    if choice == "higgsfield" and config.HIGGSFIELD_SUBMIT_URL:
        return PollingProvider(higgsfield_profile())
    if choice == "http" and config.REGEN_ENDPOINT:
        return HttpRegenProvider(config.REGEN_ENDPOINT, config.REGEN_API_KEY)
    return DryRunProvider()


def available_providers() -> dict[str, Any]:
    active = get_provider().key
    return {
        "active": active,
        "providers": [
            {"key": DryRunProvider.key, "display_name": DryRunProvider.display_name},
            {"key": HttpRegenProvider.key, "display_name": HttpRegenProvider.display_name},
            {"key": "kling", "display_name": "Kling (submit + poll)"},
            {"key": "higgsfield", "display_name": "Higgsfield (submit + poll)"},
        ],
    }
