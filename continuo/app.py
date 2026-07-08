"""Continuo web service.

Exposes the continuity-QC loop over HTTP and serves the report-card UI:

    POST /api/bible     register/replace the asset bible
    GET  /api/bible     fetch the current bible
    GET  /api/grammars  list supported model grammars
    POST /api/screen    screen a frame against the bible -> DriftReport
    POST /api/repair    rewrite a prompt to fix drift -> RepairedPrompt
"""

from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import bible as bible_store
from . import config
from .models import AssetBible, DriftIssue, Shot
from .grammars import GRAMMARS
from .metering import Meter
from .prompt_repair import repair_prompt
from .providers import available_providers, get_provider
from .vision import screen_clip

app = FastAPI(title="Continuo", version="0.1.0")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# In-memory cache of the bible, backed by the JSON store on disk.
_bible: Optional[AssetBible] = bible_store.load_bible()
_meter = Meter()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": "0.1.0",
        "live_vision": config.has_api_key(),
        "vision_model": config.VISION_MODEL,
        "bible_loaded": _bible is not None,
        "regen_provider": get_provider().key,
    }


@app.get("/api/grammars")
def grammars() -> dict:
    return {
        "grammars": [
            {
                "key": g.key,
                "display_name": g.display_name,
                "style": g.style,
                "supports_negative_prompt": g.supports_negative_prompt,
                "max_chars": g.max_chars,
                "notes": g.notes,
            }
            for g in GRAMMARS.values()
        ]
    }


@app.get("/api/bible")
def get_bible() -> dict:
    if _bible is None:
        raise HTTPException(status_code=404, detail="No asset bible registered yet.")
    return _bible.to_dict()


@app.post("/api/bible")
async def set_bible(payload: dict) -> dict:
    global _bible
    try:
        bible = AssetBible.from_dict(payload)
    except Exception as exc:  # noqa: BLE001 - surface parse errors to the client
        raise HTTPException(status_code=400, detail=f"Invalid bible: {exc}") from exc
    bible_store.save_bible(bible)
    _bible = bible
    return {"ok": True, "bible": bible.to_dict()}


@app.post("/api/screen")
async def screen(
    shot: str = Form(...),
    frame: Optional[UploadFile] = File(None),
    frames: list[UploadFile] = File(default=[]),
) -> JSONResponse:
    """Screen one or more sampled frames of a shot against the bible.

    Accepts a single ``frame`` and/or a list of ``frames`` (clip sampling). With
    no upload it runs a single mock screening.
    """
    if _bible is None:
        raise HTTPException(status_code=400, detail="Register an asset bible first.")
    try:
        shot_obj = Shot.from_dict(json.loads(shot))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid shot payload: {exc}") from exc

    uploads = ([frame] if frame is not None else []) + list(frames)
    frame_tuples: list[tuple[bytes, str]] = [
        (await f.read(), f.content_type or "image/png") for f in uploads
    ]

    report, per_frame = screen_clip(_bible, shot_obj, frame_tuples)
    for r in per_frame:  # cost accrues per vision call (per frame)
        _meter.record_screen(live=not r.model_used.startswith("mock"))

    body = report.to_dict()
    body["frame_count"] = len(per_frame)
    body["per_frame"] = [r.to_dict() for r in per_frame]
    return JSONResponse(body)


@app.post("/api/repair")
async def repair(payload: dict) -> dict:
    original = payload.get("original_prompt", "")
    model = payload.get("model", "kling")
    issues = [DriftIssue.from_dict(x) for x in payload.get("issues", [])]
    if not original:
        raise HTTPException(status_code=400, detail="original_prompt is required.")
    repaired = repair_prompt(original, model, issues)
    return repaired.to_dict()


@app.get("/api/providers")
def providers() -> dict:
    return available_providers()


@app.get("/api/usage")
def usage() -> dict:
    return _meter.summary()


@app.post("/api/regenerate")
async def regenerate(payload: dict) -> dict:
    """Submit a corrected prompt for one-click regeneration.

    Accepts an already-repaired prompt (``prompt`` [+ ``negative_prompt``]), or an
    ``original_prompt`` + ``issues`` pair which is repaired here first.
    """
    model = payload.get("model", "kling")
    prompt = (payload.get("prompt") or "").strip()
    negative_prompt = payload.get("negative_prompt")

    if not prompt:
        original = (payload.get("original_prompt") or "").strip()
        if not original:
            raise HTTPException(
                status_code=400, detail="Provide either 'prompt' or 'original_prompt'."
            )
        issues = [DriftIssue.from_dict(x) for x in payload.get("issues", [])]
        repaired = repair_prompt(original, model, issues)
        prompt, negative_prompt = repaired.prompt, repaired.negative_prompt

    job = get_provider().regenerate(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        aspect_ratio=payload.get("aspect_ratio", "16:9"),
        duration=int(payload.get("duration", 5)),
    )
    _meter.record_regen()
    return job.to_dict()


# Serve any additional static assets (kept last so /api routes win).
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
