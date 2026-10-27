import json

from gs_result_store import count_current_results, load_cases, load_current_result, make_run_id, save_result


MASK = {"size": [4, 4], "counts": "04L4"}


def setup_run(tmp_path, weights=b"v1"):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text('{"idx": 0}\n\n{"idx": 1}\n')
    vq = tmp_path / "vq.pth"
    vq.write_bytes(weights)
    cases, dataset_bytes = load_cases(dataset)
    return cases, make_run_id(dataset_bytes, "model", str(vq), "sam2")


def test_resume_only_counts_results_from_the_same_run(tmp_path):
    cases, run_id = setup_run(tmp_path)
    save_result(tmp_path, run_id, "model", {"idx": 0, "predicted_segmentation": MASK})
    assert load_current_result(tmp_path, cases[0], run_id)["model_path"] == "model"
    assert count_current_results(tmp_path, cases, run_id) == 1
    assert not list(tmp_path.glob(".*.tmp"))

    _, new_run_id = setup_run(tmp_path, weights=b"v2")  # same path, updated weights
    assert new_run_id != run_id
    assert count_current_results(tmp_path, cases, new_run_id) == 0


def test_legacy_and_truncated_results_are_redone(tmp_path):
    cases, run_id = setup_run(tmp_path)
    (tmp_path / "0.json").write_text(json.dumps({"idx": 0, "predicted_segmentation": MASK}))
    (tmp_path / "1.json").write_text('{"idx": 1, "predicted_segm')
    assert count_current_results(tmp_path, cases, run_id) == 0
