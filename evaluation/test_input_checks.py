import pytest

from input_checks import find_input_problems, require_inputs


def test_reports_unset_placeholder_and_missing_paths(tmp_path):
    present = tmp_path / "present.json"
    present.write_text("[]")
    problems = find_input_problems([
        ("dataset", str(present), "file"),
        ("root dir", str(tmp_path), "dir"),
        ("unset", None, "dir"),
        ("placeholder", "<PATH_TO_COCO2014>/train2014", "dir"),
        ("missing", str(tmp_path / "absent.pth"), "file"),
        ("file given as dir", str(present), "dir"),
    ])
    assert problems == [
        "unset is not set (got None)",
        "placeholder is not set (got '<PATH_TO_COCO2014>/train2014')",
        f"missing not found: {tmp_path / 'absent.pth'}",
        f"file given as dir not found: {present}",
    ]


def test_require_inputs_exits_with_all_problems(tmp_path):
    with pytest.raises(SystemExit) as exc:
        require_inputs([("a", str(tmp_path / "a"), "file"), ("b", "", "dir")])
    assert "a not found" in str(exc.value) and "b is not set" in str(exc.value)
    require_inputs([("root", str(tmp_path), "dir")])
