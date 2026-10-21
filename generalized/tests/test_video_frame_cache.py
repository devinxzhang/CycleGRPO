from pathlib import Path

import torch

from generalized.verl.utils.video_frame_cache import (
    VideoFrameCacheError,
    current_protocol,
    load_cache_entry,
    validate_cache_entry_files,
    write_cache_entry,
)


def _configure(monkeypatch, root: Path) -> dict:
    monkeypatch.setenv("CYCLEGRPO_FRAME_CACHE_ROOT", str(root))
    monkeypatch.setenv("CYCLEGRPO_VIDEO_TOTAL_PIXELS", "8388608")
    monkeypatch.setenv("CYCLEGRPO_VIDEO_MIN_FRAMES", "4")
    monkeypatch.setenv("CYCLEGRPO_VIDEO_MAX_FRAMES", "768")
    return current_protocol(2048, 786432, 2.0)


def test_round_trip_is_exact(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    video = tmp_path / "source.mp4"
    video.write_bytes(b"synthetic-video-identity")
    protocol = _configure(monkeypatch, root)
    frames = torch.randint(0, 256, (8, 3, 28, 56), dtype=torch.uint8).float()
    metadata = {
        "fps": 24.0,
        "frames_indices": list(range(8)),
        "total_num_frames": 96,
        "video_backend": "decord",
    }
    write_cache_entry(root, str(video), frames, metadata, 2.0, protocol)
    validate_cache_entry_files(root, str(video), protocol)

    restored, restored_metadata, sample_fps = load_cache_entry(
        str(video), 2048, 786432, 2.0, requested_nframes=8
    )
    assert restored.dtype == torch.float32
    assert torch.equal(restored, frames)
    assert restored_metadata == metadata
    assert sample_fps == 2.0


def test_strict_mode_rejects_protocol_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    video = tmp_path / "source.mp4"
    video.write_bytes(b"synthetic-video-identity")
    protocol = _configure(monkeypatch, root)
    frames = torch.zeros((4, 3, 28, 28), dtype=torch.float32)
    write_cache_entry(root, str(video), frames, {"fps": 24.0}, 2.0, protocol)
    monkeypatch.setenv("CYCLEGRPO_REQUIRE_FRAME_CACHE", "true")

    try:
        load_cache_entry(str(video), 2048, 602112, 2.0)
    except VideoFrameCacheError as exc:
        assert "protocol mismatch" in str(exc)
    else:
        raise AssertionError("strict cache mode accepted an incompatible protocol")
