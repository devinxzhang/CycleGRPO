"""Identify GCG inference inputs and validate resumable JSON results."""

import argparse
import hashlib
import json
from pathlib import Path


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_identity(value):
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


def make_run_id(model_path, vq_sam2_path, sam2_path):
    identity = {
        "format_version": 1,
        "model": _resource_identity(model_path),
        "vq_sam2": _resource_identity(vq_sam2_path),
        "sam2": _resource_identity(sam2_path),
        "generation": {"max_new_tokens": 512, "do_sample": False},
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def image_sha256(image_path):
    return _sha256_file(image_path)


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


def output_is_complete(path, image_id, run_id, image_hash):
    try:
        with Path(path).open(encoding="utf-8") as source:
            result = json.load(source)
    except (OSError, ValueError):
        return False
    return (
        isinstance(result, dict)
        and result.get("image_id") == image_id
        and result.get("run_id") == run_id
        and result.get("image_sha256") == image_hash
        and isinstance(result.get("caption"), str)
        and isinstance(result.get("phrases"), list)
        and all(isinstance(phrase, str) for phrase in result["phrases"])
        and isinstance(result.get("pred_masks"), list)
        and all(valid_compressed_rle(mask) for mask in result["pred_masks"])
    )


def count_current_results(image_folder, save_dir, run_id):
    images = sorted(path for path in Path(image_folder).iterdir() if path.is_file())
    image_ids = [path.stem for path in images]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Image filenames have duplicate stems; GCG outputs would collide")
    complete = sum(
        output_is_complete(Path(save_dir) / f"{path.stem}.json", path.stem,
                           run_id, image_sha256(path))
        for path in images
    )
    return complete, len(images)


def _main():
    parser = argparse.ArgumentParser(description="Verify GCG outputs for the current inputs")
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--vq_sam2_path", required=True)
    parser.add_argument("--sam2_path", required=True)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()
    run_id = make_run_id(args.model_path, args.vq_sam2_path, args.sam2_path)
    complete, total = count_current_results(args.image_folder, args.save_dir, run_id)
    print(f"GCG output check: {complete}/{total} current image IDs")
    if args.require_complete and complete != total:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
