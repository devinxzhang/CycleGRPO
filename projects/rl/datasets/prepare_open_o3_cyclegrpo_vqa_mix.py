"""Add Open-o3's verifiable VQA routes to the audited CycleGRPO mixture.

The base CycleGRPO parquet is preserved byte-for-byte.  This script appends
all 13,000 General-video-QA MCQ rows and all 2,000 free-form rows from the
released Open-o3 RL set.  Training uses exact option accuracy for MCQ and the
released mean ROUGE-1/2/L F1 reward for free-form answers; no LLM judge is
required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from generalized.verl.utils.cyclegrpo_profiles import OPEN_O3_UNDERSTANDING_SOURCE
from projects.rl.datasets.prepare_open_o3_spatiotemporal_cyclegrpo import DEFAULT_DATA_ROOT
from projects.rl.datasets.prepare_open_o3_unified_cyclegrpo import resolve_media_path


VQA_TASKS = frozenset({"General video QA MCQ", "General video QA Free-form"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vqa_prompt(example: dict[str, Any]) -> str:
    question = str(example.get("question") or "").strip()
    if example["task"] == "General video QA MCQ":
        instruction = "Output only the single correct option letter (A, B, C, or D)."
    else:
        instruction = "Answer the question directly and concisely. Output the answer only."
    return f"<video>\n{question}\n{instruction}"


def vqa_row(example: dict[str, Any], media: Path) -> dict[str, Any]:
    return {
        "videos": [str(media)],
        "cap_problem": vqa_prompt(example),
        "cap_answer": str(example.get("answer") or "").strip(),
        "seg_problem": "",
        "seg_answer": "",
        "masks": None,
        "source": OPEN_O3_UNDERSTANDING_SOURCE,
        "extra_info": {
            "open_o3_id": str(example.get("id") or ""),
            "open_o3_source": str(example.get("source") or ""),
            "open_o3_task": str(example.get("task") or ""),
            "question": str(example.get("question") or ""),
            "answer": str(example.get("answer") or ""),
            "video_path": str(example.get("video_path") or ""),
            "vqa_reward": (
                "exact_option_accuracy"
                if example["task"] == "General video QA MCQ"
                else "mean_rouge1_rouge2_rougeL_f1"
            ),
        },
        "cyclegrpo_route": "understanding",
    }


def build(args: argparse.Namespace) -> None:
    base_path = args.base_parquet.resolve()
    input_json = (args.input_json or args.data_root / "json_data" / "STGR-RL.json").resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()

    base = pd.read_parquet(base_path)
    examples = json.loads(input_json.read_text(encoding="utf-8"))
    selected = [example for example in examples if str(example.get("task") or "") in VQA_TASKS]

    task_counts = Counter(str(example["task"]) for example in selected)
    if task_counts != Counter({"General video QA MCQ": 13000, "General video QA Free-form": 2000}):
        raise AssertionError(f"Unexpected released VQA counts: {task_counts}")

    rows = []
    missing = []
    for example in selected:
        media = resolve_media_path(args.data_root.resolve(), example)
        if args.validate_paths and not media.is_file():
            missing.append(str(media))
            continue
        if not str(example.get("question") or "").strip():
            raise AssertionError(f"Empty VQA question: {example.get('id')}")
        if not str(example.get("answer") or "").strip():
            raise AssertionError(f"Empty VQA answer: {example.get('id')}")
        rows.append(vqa_row(example, media))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} VQA videos; first: {missing[:5]}")

    vqa = pd.DataFrame(rows)
    # concat(sort=False) keeps every query-conditioning column from clean-v2;
    # VQA rows receive nulls for fields that do not apply to them.
    mixed = pd.concat([base, vqa], ignore_index=True, sort=False)
    expected_rows = len(base) + 15000
    if len(mixed) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, got {len(mixed)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mixed.to_parquet(output_path, index=False)
    manifest = {
        "version": "open_o3_cyclegrpo_clean_v2_plus_vqa",
        "base": str(base_path),
        "base_sha256": sha256(base_path),
        "open_o3_rl_input": str(input_json),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "base_cycle_rows": len(base),
        "vqa_rows": len(vqa),
        "total_rows": len(mixed),
        "vqa_fraction": len(vqa) / len(mixed),
        "task_counts": dict(task_counts),
        "reward_protocol": {
            "General video QA MCQ": "exact option accuracy",
            "General video QA Free-form": "mean ROUGE-1/2/L F1 (Open-o3 official)",
            "llm_judge": False,
        },
        "base_rows_modified": False,
        "validation_split": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument(
        "--base-parquet",
        type=Path,
        default=Path("rl_dataset/open_o3_unified_cyclegrpo_clean_v2_train.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl_dataset/open_o3_unified_cyclegrpo_clean_v2_plus_vqa_train.parquet"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("rl_dataset/open_o3_unified_cyclegrpo_clean_v2_plus_vqa_manifest.json"),
    )
    parser.add_argument("--validate-paths", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
