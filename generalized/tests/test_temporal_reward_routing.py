import numpy as np

from generalized.verl.utils.reward_routing import is_temporal_video_sample


def test_tg_multi_source_is_temporal_after_compact_transport():
    batch = {
        "multi_modal_data": np.array(
            [{"videos": ["video.mp4"]}], dtype=object
        )
    }
    assert is_temporal_video_sample(batch, 0, "tg_multi_merged")


def test_explicit_temporal_source_fails_closed_without_media_metadata():
    assert is_temporal_video_sample({}, 0, "tg_multi_merged")


def test_nested_video_is_detected_for_other_video_source():
    batch = {
        "multi_modal_data": np.array(
            [{"videos": ["video.mp4"]}], dtype=object
        )
    }
    assert is_temporal_video_sample(batch, 0, "another_video_source")


def test_nested_image_is_not_misrouted_to_temporal_reward():
    batch = {
        "multi_modal_data": np.array(
            [{"images": ["image.jpg"]}], dtype=object
        )
    }
    assert not is_temporal_video_sample(batch, 0, "dam_cyclegrpo")
