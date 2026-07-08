"""Tests for clip frame sampling."""

import shutil
import subprocess

import pytest

from continuo.sampling import extract_frames, ffmpeg_exe, frame_timestamps


def test_frame_timestamps_are_evenly_spaced_midpoints():
    ts = frame_timestamps(10.0, 5)
    assert ts == [1.0, 3.0, 5.0, 7.0, 9.0]


def test_frame_timestamps_single_and_zero_duration():
    assert frame_timestamps(0.0, 5) == [0.0]
    assert len(frame_timestamps(4.0, 1)) == 1


def test_frame_timestamps_last_sample_stays_inside_clip():
    ts = frame_timestamps(2.0, 3)
    assert all(t <= 2.0 - 0.05 + 1e-9 for t in ts)


_FF = ffmpeg_exe()


@pytest.mark.skipif(_FF is None, reason="no ffmpeg available")
def test_extract_frames_from_generated_clip(tmp_path):
    # Generate a 2s test-pattern clip with the same ffmpeg binary.
    clip = tmp_path / "test.mp4"
    subprocess.run(
        [_FF, "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=10",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    frames = extract_frames(clip.read_bytes(), n=3)
    assert len(frames) == 3
    for data, media_type in frames:
        assert media_type == "image/png"
        assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
