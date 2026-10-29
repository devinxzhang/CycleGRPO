"""Guard against the spatial and video runtimes drifting apart.

`verl/` and `generalized/verl/` are two copies of the same engine. Most of their
modules are byte-identical, so a fix applied to one copy and not the other is a
silent bug in the other runtime. This test fails when a module that both copies
share stops matching, and it keeps the list of deliberately divergent modules
honest.
"""

import hashlib
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPATIAL = "verl"
VIDEO = "generalized/verl"

# Modules whose two copies are allowed to differ. The video runtime genuinely
# needs its own dataset, trainer, FSDP worker and rollout; the rest of this list
# is drift that nobody has reconciled yet. Shrinking it is the goal -- adding to
# it means accepting that the two runtimes behave differently on purpose.
ALLOW_DIVERGENT = frozenset(
    {
        "__init__.py",
        "models/monkey_patch.py",
        "models/transformers/qwen2_vl.py",
        "models/transformers/qwen3_vl.py",
        "single_controller/ray/base.py",
        "trainer/config.py",
        "trainer/core_algos.py",
        "trainer/data_loader.py",
        "trainer/main.py",
        "trainer/ray_trainer.py",
        "utils/dataset.py",
        "utils/logger/logger.py",
        "utils/torch_functional.py",
        "workers/actor/config.py",
        "workers/actor/dp_actor.py",
        "workers/fsdp_workers.py",
        "workers/reward/function.py",
        "workers/rollout/config.py",
        "workers/rollout/vllm_rollout_spmd.py",
    }
)


def tracked_modules(prefix):
    """Relative paths of the .py files git tracks under one runtime."""
    out = subprocess.run(
        ["git", "ls-files", prefix], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return {path[len(prefix) + 1 :]: REPO_ROOT / path for path in out if path.endswith(".py")}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def runtimes():
    spatial, video = tracked_modules(SPATIAL), tracked_modules(VIDEO)
    assert spatial and video, "no tracked .py files found; is this a git checkout?"
    return spatial, video


def test_shared_modules_stay_identical(runtimes):
    spatial, video = runtimes
    drifted = sorted(
        name
        for name in set(spatial) & set(video)
        if name not in ALLOW_DIVERGENT and digest(spatial[name]) != digest(video[name])
    )
    assert not drifted, (
        "These modules are shared by both runtimes but no longer match:\n  "
        + "\n  ".join(f"{SPATIAL}/{name} != {VIDEO}/{name}" for name in drifted)
        + "\nApply the change to both copies, or add the path to ALLOW_DIVERGENT "
        "if the runtimes are meant to differ here."
    )


def test_allow_list_does_not_name_identical_modules(runtimes):
    spatial, video = runtimes
    reconciled = sorted(
        name
        for name in ALLOW_DIVERGENT
        if name in spatial and name in video and digest(spatial[name]) == digest(video[name])
    )
    assert not reconciled, (
        "These modules now match in both runtimes; drop them from ALLOW_DIVERGENT "
        "so they stay guarded:\n  " + "\n  ".join(reconciled)
    )


def test_allow_list_has_no_stale_entries(runtimes):
    spatial, video = runtimes
    stale = sorted(name for name in ALLOW_DIVERGENT if name not in spatial or name not in video)
    assert not stale, "ALLOW_DIVERGENT names modules that no longer exist in both runtimes:\n  " + "\n  ".join(stale)
