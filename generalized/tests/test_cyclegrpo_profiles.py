from generalized.verl.utils.cyclegrpo_profiles import (
    CAPTIONING_SOURCES,
    CYCLE_SOURCES,
    LOCALIZATION_SOURCES,
    NFRAME_PRESERVING_SOURCES,
    PROFILES,
    SPATIOTEMPORAL_CYCLE_SOURCES,
    CapabilityLevel,
    TrainingRole,
    get_profile,
    validate_registry,
)


def test_each_capability_has_three_parallel_roles():
    validate_registry()
    for capability in CapabilityLevel:
        roles = {p.role for p in PROFILES.values() if p.capability is capability}
        assert roles == set(TrainingRole)


def test_cycle_sources_are_disjoint_from_single_direction_controls():
    assert CYCLE_SOURCES.isdisjoint(CAPTIONING_SOURCES)
    assert CYCLE_SOURCES.isdisjoint(LOCALIZATION_SOURCES)
    assert CAPTIONING_SOURCES.isdisjoint(LOCALIZATION_SOURCES)


def test_legacy_source_strings_and_semantics_stay_stable():
    expected = {
        "dam_cyclegrpo": ("spatial", "images", "mask_iou"),
        "tg_multi_merged": ("temporal", "videos", "temporal_iou"),
        "stgr_trajectory_cyclegrpo": (
            "spatial-temporal",
            "videos",
            "trajectory_space_time_iou",
        ),
    }
    for source, semantics in expected.items():
        profile = get_profile(source)
        assert profile is not None
        assert (profile.capability.value, profile.media_key, profile.overlap) == semantics


def test_spatiotemporal_cycle_sources_preserve_frame_metadata():
    assert SPATIOTEMPORAL_CYCLE_SOURCES <= NFRAME_PRESERVING_SOURCES
    assert all(
        get_profile(source).role is TrainingRole.CYCLE
        for source in SPATIOTEMPORAL_CYCLE_SOURCES
    )
