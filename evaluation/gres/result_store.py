"""GRES result identity and resume helpers; no model dependencies required."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def load_cases(dataset_path):
    with open(dataset_path, "rb") as f:
        dataset_bytes = f.read()
    data = json.loads(dataset_bytes)
    if not isinstance(data, list) or not data or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"GRES dataset must be a nonempty list of objects: {dataset_path}")
    return [dict(item, case_id=index) for index, item in enumerate(data)], dataset_bytes


def _resource_identity(value):
    """Include local file metadata so an updated checkpoint invalidates a resume."""
    path = Path(value).expanduser()
    if not path.exists():
        return {"identifier": value}
    path = path.resolve()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return {
        "path": str(path),
        "files": [(str(p.relative_to(path)) if path.is_dir() else p.name,
                   p.stat().st_size, p.stat().st_mtime_ns) for p in files],
    }


def make_run_id(dataset_bytes, model_path, vq_sam2_path, sam2_path):
    identity = {
        "format_version": 1,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "model": _resource_identity(model_path),
        "vq_sam2": _resource_identity(vq_sam2_path),
        "sam2": _resource_identity(sam2_path),
        "generation": {"max_new_tokens": 128, "do_sample": False},
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def result_path(save_dir, case_id):
    return Path(save_dir) / f"{case_id}.json"


def valid_compressed_rle(mask):
    """Check COCO run syntax and pixel count without decoding a full mask."""
    if not isinstance(mask, dict):
        return False
    size = mask.get("size")
    encoded = mask.get("counts")
    if (not isinstance(size, list) or len(size) != 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in size)
            or not isinstance(encoded, str) or not encoded):
        return False
    area = size[0] * size[1]
    counts = []
    position = 0
    total = 0
    while position < len(encoded):
        value = 0
        shift = 0
        while True:
            if position >= len(encoded) or shift > 60:
                return False
            char = ord(encoded[position]) - 48
            position += 1
            if not 0 <= char <= 63:
                return False
            value |= (char & 0x1F) << shift
            shift += 5
            if not char & 0x20:
                if char & 0x10:
                    value |= -1 << shift
                break
        if len(counts) > 2:
            value += counts[-2]
        if value < 0 or value > area - total:
            return False
        counts.append(value)
        total += value
    return total == area


def load_current_result(save_dir, case, run_id):
    try:
        with result_path(save_dir, case["case_id"]).open(encoding="utf-8") as f:
            result = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict):
        return None
    if (result.get("run_id") != run_id or result.get("case_id") != case["case_id"]
            or result.get("gt_masks") != case["segmentation"]):
        return None
    pred = result.get("pred_masks")
    if not valid_compressed_rle(pred) or pred["size"] != case["segmentation"].get("size"):
        return None
    return result


def count_current_results(save_dir, cases, run_id):
    return sum(load_current_result(save_dir, case, run_id) is not None for case in cases)


def save_result(save_dir, case, run_id, model_path, prediction):
    record = dict(prediction, case_id=case["case_id"], run_id=run_id,
                  model_path=model_path)
    path = result_path(save_dir, case["case_id"])
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=save_dir,
                                         prefix=f".{case['case_id']}.", suffix=".tmp",
                                         delete=False) as f:
            temp_path = f.name
            json.dump(record, f)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def _main():
    parser = argparse.ArgumentParser(description="Count valid GRES outputs for a run")
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
