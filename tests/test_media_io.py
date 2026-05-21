"""Stage 2 — media_io (ffmpeg/ffprobe wrapper) tests."""

from pathlib import Path

import pytest

from videodub.errors import MediaIOError
from videodub.media_io import extract_audio, probe, remux


def test_probe_reads_streams_and_duration(sample_video: Path):
    info = probe(sample_video)

    assert info.path == sample_video
    assert info.duration == pytest.approx(1.0, abs=0.2)
    assert info.has_video and info.has_audio
    assert len(info.video_streams) == 1
    assert len(info.audio_streams) == 1
    assert info.video_streams[0].width == 160
    assert info.video_streams[0].height == 120


def test_probe_missing_file_raises():
    with pytest.raises(MediaIOError, match="file not found"):
        probe("does/not/exist.mp4")


def test_extract_audio_defaults_to_16k_mono(sample_video: Path, tmp_path: Path):
    out = extract_audio(sample_video, tmp_path / "audio.wav")

    assert out.exists()
    info = probe(out)
    assert not info.has_video  # the video stream was dropped
    assert info.has_audio
    assert info.audio_streams[0].sample_rate == 16000
    assert info.audio_streams[0].channels == 1
    assert info.duration == pytest.approx(1.0, abs=0.2)


def test_extract_audio_honours_custom_sample_rate(sample_video: Path, tmp_path: Path):
    out = extract_audio(sample_video, tmp_path / "audio8k.wav", sr=8000)

    assert probe(out).audio_streams[0].sample_rate == 8000


def test_extract_audio_creates_missing_parent_dir(sample_video: Path, tmp_path: Path):
    out = extract_audio(sample_video, tmp_path / "nested" / "deep" / "audio.wav")

    assert out.exists()


def test_extract_audio_missing_input_raises(tmp_path: Path):
    with pytest.raises(MediaIOError, match="input video not found"):
        extract_audio("nope.mp4", tmp_path / "audio.wav")


def test_remux_replaces_audio_keeping_video(sample_video: Path, tmp_path: Path):
    audio = extract_audio(sample_video, tmp_path / "audio.wav")
    out = remux(sample_video, audio, tmp_path / "remuxed.mp4")

    assert out.exists()
    info = probe(out)
    assert info.has_video and info.has_audio
    assert info.duration == pytest.approx(1.0, abs=0.2)
    assert info.video_streams[0].width == 160


def test_extract_audio_on_non_media_file_raises(tmp_path: Path):
    junk = tmp_path / "not-a-video.mp4"
    junk.write_text("this is plain text, not a video")

    with pytest.raises(MediaIOError, match="ffmpeg"):
        extract_audio(junk, tmp_path / "audio.wav")
