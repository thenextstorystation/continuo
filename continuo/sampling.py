"""Automatic frame sampling from an uploaded clip.

Screens should be run against a clip, not a hand-picked still. This module
extracts N evenly-spaced frames from a generated video so the clip-level
screening in ``vision.py`` has frames to work with.

Video decoding is an optional extra: it prefers a system ``ffmpeg`` on PATH and
falls back to the static binary shipped by the ``imageio-ffmpeg`` wheel. If
neither is present, ``extract_frames`` raises ``FfmpegUnavailable`` with an
install hint — the core package never hard-depends on a decoder.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


class FfmpegUnavailable(RuntimeError):
    """Raised when no ffmpeg binary can be found for clip decoding."""


def ffmpeg_exe() -> Optional[str]:
    """Return a usable ffmpeg path: system binary first, then the bundled one."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional dependency not installed
        return None


def frame_timestamps(duration: float, n: int) -> list[float]:
    """N evenly-spaced sample times (seconds) across a clip of ``duration``.

    Uses frame midpoints ((i + 0.5)/n) so the first and last frames aren't taken
    at the exact clip boundaries (which are often black or truncated).
    """
    n = max(1, n)
    if duration <= 0:
        return [0.0]
    horizon = max(0.0, duration - 0.05)  # keep the last sample inside the clip
    return [min((i + 0.5) / n * duration, horizon) for i in range(n)]


def _probe_duration(exe: str, path: str) -> float:
    # ffmpeg prints stream info (incl. Duration) to stderr when no output is given.
    proc = subprocess.run(
        [exe, "-nostdin", "-i", path],
        capture_output=True,
        text=True,
        check=False,
    )
    m = _DURATION_RE.search(proc.stderr or "")
    if not m:
        return 0.0
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def _extract_one(exe: str, path: str, t: float) -> Optional[bytes]:
    proc = subprocess.run(
        [
            exe, "-nostdin", "-loglevel", "error",
            "-ss", f"{t:.3f}", "-i", path,
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    return proc.stdout or None


def extract_frames(
    video_bytes: bytes, n: int = 5, suffix: str = ".mp4"
) -> list[tuple[bytes, str]]:
    """Extract up to ``n`` evenly-spaced PNG frames from a clip.

    Returns a list of (png_bytes, "image/png"). Raises ``FfmpegUnavailable`` if no
    decoder is present.
    """
    exe = ffmpeg_exe()
    if not exe:
        raise FfmpegUnavailable(
            "Clip decoding needs ffmpeg. Install a system ffmpeg, or "
            "`pip install imageio-ffmpeg` to use the bundled binary."
        )

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(video_bytes)
        tmp.flush()
        tmp.close()
        duration = _probe_duration(exe, tmp.name)
        frames: list[tuple[bytes, str]] = []
        for t in frame_timestamps(duration, n):
            png = _extract_one(exe, tmp.name, t)
            if png:
                frames.append((png, "image/png"))
        return frames
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
