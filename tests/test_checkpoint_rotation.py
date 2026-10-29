import pytest

from verl.utils.checkpoint import remove_obsolete_ckpt


def make_checkpoints(root, steps):
    for step in steps:
        (root / f"global_step_{step}").mkdir(parents=True)
        (root / f"global_step_{step}" / "actor").mkdir()
    return sorted(int(p.name.rsplit("_", 1)[1]) for p in root.glob("global_step_*"))


def remaining(root):
    return sorted(int(p.name.rsplit("_", 1)[1]) for p in root.glob("global_step_*"))


def test_save_limit_keeps_only_the_newest_when_no_best_is_pinned(tmp_path):
    make_checkpoints(tmp_path, [1, 2, 3])
    remove_obsolete_ckpt(str(tmp_path), global_step=3, best_global_step=None, save_limit=1)
    assert remaining(tmp_path) == [3]


def test_a_pinned_best_checkpoint_survives_rotation(tmp_path):
    # With validation on, the best-scoring step is kept on top of the limit.
    make_checkpoints(tmp_path, [1, 2, 3])
    remove_obsolete_ckpt(str(tmp_path), global_step=3, best_global_step=1, save_limit=1)
    assert remaining(tmp_path) == [1, 3]


@pytest.mark.parametrize("save_limit", [0, -1])
def test_rotation_is_disabled_without_a_limit(tmp_path, save_limit):
    make_checkpoints(tmp_path, [1, 2, 3])
    remove_obsolete_ckpt(str(tmp_path), global_step=3, best_global_step=None, save_limit=save_limit)
    assert remaining(tmp_path) == [1, 2, 3]


def test_the_current_step_is_never_removed(tmp_path):
    make_checkpoints(tmp_path, [5])
    remove_obsolete_ckpt(str(tmp_path), global_step=5, best_global_step=None, save_limit=1)
    assert remaining(tmp_path) == [5]
