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
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Optional

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


def get_provider():
    """Return the configured regeneration provider (dry-run unless a live one is set up)."""
    if config.REGEN_PROVIDER.lower() == "http" and config.REGEN_ENDPOINT:
        return HttpRegenProvider(config.REGEN_ENDPOINT, config.REGEN_API_KEY)
    return DryRunProvider()


def available_providers() -> dict[str, Any]:
    active = get_provider().key
    return {
        "active": active,
        "providers": [
            {"key": DryRunProvider.key, "display_name": DryRunProvider.display_name},
            {"key": HttpRegenProvider.key, "display_name": HttpRegenProvider.display_name},
        ],
    }
