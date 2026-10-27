"""GroundingSuite result identity and resume helpers; no model dependencies required."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def load_cases(dataset_path):
    with open(dataset_path, "rb") as f:
        dataset_bytes = f.read()
    cases = [json.loads(line) for line in dataset_bytes.decode("utf-8").splitlines() if line.strip()]
    if not cases or not all(isinstance(case, dict) and "idx" in case for case in cases):
        raise ValueError(f"GroundingSuite dataset must be nonempty JSONL with an idx per line: {dataset_path}")
    return cases, dataset_bytes


def _resource_identity(value):
    """Include local file metadata so an updated checkpoint invalidates a resume."""
    path = Path(value).expanduser()
    if not path.exists():
        return {"identifier": value}
    path = path.resolve()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return {
        "path": str(path),
        "files": [
            (str(p.relative_to(path)) if path.is_dir() else p.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in files
        ],
    }


def make_run_id(dataset_bytes, model_path, vq_sam2_path, sam2_path):
    identity = {
        "format_version": 1,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "model": _resource_identity(model_path),
        "vq_sam2": _resource_identity(vq_sam2_path),
        "sam2": _resource_identity(sam2_path),
        "generation": {"max_new_tokens": 512, "do_sample": False},
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def result_path(save_dir, idx):
    return Path(save_dir) / f"{idx}.json"


def load_current_result(save_dir, case, run_id):
    """Return the saved prediction only if it belongs to this run and is well formed."""
    try:
        with result_path(save_dir, case["idx"]).open(encoding="utf-8") as f:
            result = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict) or result.get("run_id") != run_id or result.get("idx") != case["idx"]:
        return None
    mask = result.get("predicted_segmentation")
    if (
        not isinstance(mask, dict)
        or not isinstance(mask.get("counts"), str)
        or not isinstance(mask.get("size"), list)
        or len(mask["size"]) != 2
    ):
        return None
    return result


def count_current_results(save_dir, cases, run_id):
    return sum(load_current_result(save_dir, case, run_id) is not None for case in cases)


def save_result(save_dir, run_id, model_path, prediction):
    """Write-then-rename so an interrupted shard never leaves a truncated result."""
    record = dict(prediction, run_id=run_id, model_path=model_path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=save_dir, prefix=f".{prediction['idx']}.", suffix=".tmp", delete=False
        ) as f:
            temp_path = f.name
            json.dump(record, f)
        os.replace(temp_path, result_path(save_dir, prediction["idx"]))
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def _main():
    parser = argparse.ArgumentParser(description="Count valid GroundingSuite outputs for a run")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--vq_sam2_path", required=True)
    parser.add_argument("--sam2_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()
    cases, dataset_bytes = load_cases(args.dataset)
    run_id = make_run_id(dataset_bytes, args.model_path, args.vq_sam2_path, args.sam2_path)
    complete = count_current_results(args.save_dir, cases, run_id)
    print(f"{complete}/{len(cases)}")
    if args.require_complete and complete != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
