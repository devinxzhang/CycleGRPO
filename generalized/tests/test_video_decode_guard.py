import os
import time

import pytest

from generalized.verl.utils.video_decode_guard import VideoDecodeTimeout, guarded_video_decode


def test_native_decode_timeout_is_killable_and_helper_restarts():
    started = time.monotonic()
    with pytest.raises(VideoDecodeTimeout, match="synthetic-bad-video.mp4"):
        guarded_video_decode(
            "_test_sleep",
            30.0,
            timeout_s=0.1,
            media_path="synthetic-bad-video.mp4",
            context="unit-test-timeout",
        )
    # ``spawn`` startup imports can take a couple of seconds on a busy CI
    # host.  The important invariant is that we do not wait for the 30-second
    # native operation to return.
    assert time.monotonic() - started < 5.0

    # The timed-out child was killed.  A subsequent request must start a new
    # helper and complete normally rather than inheriting the stuck decoder.
    # This timeout has to cover a cold ``spawn`` start, which imports torch and
    # can take far longer than the decode itself on a loaded host; what is under
    # test here is the restart, not its latency.
    assert (
        guarded_video_decode(
            "_test_sleep",
            0.0,
            timeout_s=60.0,
            media_path="synthetic-good-video.mp4",
            context="unit-test-restart",
        )
        is None
    )


def test_real_video_decode_when_smoke_path_is_provided():
    video = os.environ.get("CYCLEGRPO_SMOKE_VIDEO")
    if not video:
        pytest.skip("CYCLEGRPO_SMOKE_VIDEO is not set")
    frames, metadata = guarded_video_decode(
        "video",
        video,
        3136,
        262144,
        2.0,
        return_video_metadata=True,
        nframes=4,
        timeout_s=60,
        media_path=video,
        context="unit-test-real-video",
    )
    assert frames.shape[0] == 4
    assert metadata["total_num_frames"] > 0

    frame_images, timestamps = guarded_video_decode(
        "open_o3_frames",
        video,
        nframes=4,
        key_timestamps=[1.0],
        timeout_s=60,
        media_path=video,
        context="unit-test-open-o3-frames",
    )
    assert len(frame_images) >= 4
    assert len(frame_images) == len(timestamps)
