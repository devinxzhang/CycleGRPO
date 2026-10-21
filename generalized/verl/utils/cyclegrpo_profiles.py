"""Canonical task registry for the three CycleGRPO capability levels.

The registry is dependency-free so data loading, rollout, training, and reward
code can share one source of truth without import cycles. Legacy source strings
remain stable; this module only gives them explicit, parallel semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

VIDEO_CAPTION_ANTI_TIME_INSTRUCTION = (
    "\nImportant: In your answer, do not mention any explicit time information "
    "(timestamps, seconds, minutes, frame indices, time ranges, or phrases like "
    "'during 0-3 seconds'). Replace them with neutral phrases like "
    "'in this segment'. Output description only."
)


class CapabilityLevel(str, Enum):
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    SPATIOTEMPORAL = "spatial-temporal"


class TrainingRole(str, Enum):
    CYCLE = "cycle"
    UNDERSTANDING = "understanding"
    LOCALIZATION = "localization"


@dataclass(frozen=True)
class CycleGRPOProfile:
    source: str | None
    capability: CapabilityLevel
    role: TrainingRole
    media_key: str
    overlap: str
    query_interface: str = "standard"
    preserve_nframes: bool = False


SPATIOTEMPORAL_SOURCE = "stgr_cyclegrpo"
ATOMIC_SPATIOTEMPORAL_SOURCE = "stgr_atomic_cyclegrpo"
TRAJECTORY_SPATIOTEMPORAL_SOURCE = "stgr_trajectory_cyclegrpo"
TRAJECTORY_UNDERSTANDING_SOURCE = "stgr_trajectory_understanding_grpo"
TRAJECTORY_LOCALIZATION_SOURCE = "stgr_trajectory_localization_grpo"
OPEN_O3_UNDERSTANDING_SOURCE = "open_o3_understanding_grpo"


_PROFILES = (
    CycleGRPOProfile(None, CapabilityLevel.SPATIAL, TrainingRole.CYCLE, "images", "mask_iou"),
    CycleGRPOProfile("dam_cyclegrpo", CapabilityLevel.SPATIAL, TrainingRole.CYCLE, "images", "mask_iou"),
    CycleGRPOProfile("dam_captioning", CapabilityLevel.SPATIAL, TrainingRole.UNDERSTANDING, "images", "llm_judge"),
    CycleGRPOProfile("dam_grounding", CapabilityLevel.SPATIAL, TrainingRole.LOCALIZATION, "images", "mask_iou"),
    CycleGRPOProfile("tg_multi_merged", CapabilityLevel.TEMPORAL, TrainingRole.CYCLE, "videos", "temporal_iou"),
    CycleGRPOProfile("tg_captioning", CapabilityLevel.TEMPORAL, TrainingRole.UNDERSTANDING, "videos", "llm_judge"),
    CycleGRPOProfile("tg_grounding", CapabilityLevel.TEMPORAL, TrainingRole.LOCALIZATION, "videos", "temporal_iou"),
    CycleGRPOProfile(
        SPATIOTEMPORAL_SOURCE,
        CapabilityLevel.SPATIOTEMPORAL,
        TrainingRole.CYCLE,
        "videos",
        "space_time_support_iou",
        preserve_nframes=True,
    ),
    CycleGRPOProfile(
        ATOMIC_SPATIOTEMPORAL_SOURCE,
        CapabilityLevel.SPATIOTEMPORAL,
        TrainingRole.CYCLE,
        "videos",
        "atomic_space_time_iou",
        query_interface="atomic",
        preserve_nframes=True,
    ),
    CycleGRPOProfile(
        TRAJECTORY_SPATIOTEMPORAL_SOURCE,
        CapabilityLevel.SPATIOTEMPORAL,
        TrainingRole.CYCLE,
        "videos",
        "trajectory_space_time_iou",
        query_interface="trajectory",
        preserve_nframes=True,
    ),
    CycleGRPOProfile(
        TRAJECTORY_UNDERSTANDING_SOURCE,
        CapabilityLevel.SPATIOTEMPORAL,
        TrainingRole.UNDERSTANDING,
        "videos",
        "llm_judge",
        query_interface="trajectory",
        preserve_nframes=True,
    ),
    CycleGRPOProfile(
        TRAJECTORY_LOCALIZATION_SOURCE,
        CapabilityLevel.SPATIOTEMPORAL,
        TrainingRole.LOCALIZATION,
        "videos",
        "trajectory_space_time_iou",
        preserve_nframes=True,
    ),
    CycleGRPOProfile(
        OPEN_O3_UNDERSTANDING_SOURCE,
        CapabilityLevel.SPATIOTEMPORAL,
        TrainingRole.UNDERSTANDING,
        "videos",
        "answer_accuracy",
        preserve_nframes=True,
    ),
)

PROFILES = {profile.source: profile for profile in _PROFILES}
CYCLE_SOURCES = frozenset(p.source for p in _PROFILES if p.role is TrainingRole.CYCLE)
SPATIAL_SOURCES = frozenset(p.source for p in _PROFILES if p.capability is CapabilityLevel.SPATIAL)
TEMPORAL_SOURCES = frozenset(p.source for p in _PROFILES if p.capability is CapabilityLevel.TEMPORAL)
SPATIOTEMPORAL_SOURCES = frozenset(
    p.source for p in _PROFILES if p.capability is CapabilityLevel.SPATIOTEMPORAL
)
SPATIOTEMPORAL_CYCLE_SOURCES = frozenset(
    p.source
    for p in _PROFILES
    if p.capability is CapabilityLevel.SPATIOTEMPORAL and p.role is TrainingRole.CYCLE
)
NFRAME_PRESERVING_SOURCES = frozenset(p.source for p in _PROFILES if p.preserve_nframes)
QUERY_CONDITIONED_SOURCES = frozenset(
    p.source for p in _PROFILES if p.query_interface != "standard"
)
CAPTIONING_SOURCES = frozenset(
    p.source for p in _PROFILES if p.role is TrainingRole.UNDERSTANDING
)
LOCALIZATION_SOURCES = frozenset(
    p.source for p in _PROFILES if p.role is TrainingRole.LOCALIZATION
)

# DenseWorld shares the historical image nested-rollout machinery.
NESTED_CYCLE_SOURCES = frozenset(
    {"denseworld_single", "denseworld_multiple"} | set(CYCLE_SOURCES)
)


def get_profile(source: str | None) -> CycleGRPOProfile | None:
    """Return the registered semantics for a dataset source, if known."""

    return PROFILES.get(source)


def is_capability(source: str | None, capability: CapabilityLevel) -> bool:
    profile = get_profile(source)
    return profile is not None and profile.capability is capability


def is_cycle_source(source: str | None) -> bool:
    profile = get_profile(source)
    return profile is not None and profile.role is TrainingRole.CYCLE


def validate_registry() -> None:
    """Fail fast if a future edit breaks the parallel profile contract."""

    if len(PROFILES) != len(_PROFILES):
        raise ValueError("CycleGRPO profile sources must be unique")
    for capability in CapabilityLevel:
        roles = {p.role for p in _PROFILES if p.capability is capability}
        missing = set(TrainingRole) - roles
        if missing:
            names = sorted(role.value for role in missing)
            raise ValueError(f"{capability.value} is missing roles: {names}")


validate_registry()
