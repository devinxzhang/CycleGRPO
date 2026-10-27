"""Read-only checks for public video training; never starts Ray or GPU work."""
import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


def check_checkpoint(path):
    path = Path(path)
    for name in ("config.json", "tokenizer_config.json"):
        if not (path / name).is_file():
            raise ValueError(f"Missing checkpoint metadata: {path / name}")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
    else:
        shards = {"model.safetensors"}
    for name in shards:
        target = path / name
        if not target.is_file() or not target.stat().st_size:
            raise ValueError(f"Missing or empty model shard: {target}")
    return len(shards)


def check_data(path, profile):
    # Only the footer and the source column are read; video rows can be large.
    parquet = pq.ParquetFile(path)
    num_rows = parquet.metadata.num_rows
    if num_rows == 0:
        raise ValueError("Empty training parquet")
    columns = set(parquet.schema_arrow.names)
    required = {"source", "cap_problem", "cap_answer", "seg_problem", "seg_answer", "videos"}
    missing = required - columns
    if missing:
        raise ValueError(f"Missing parquet columns: {sorted(missing)}")
    sources = set(parquet.read(columns=["source"]).column("source").to_pylist())
    if profile.startswith("temporal_") and sources != {"tg_multi_merged"}:
        raise ValueError(f"Expected temporal cycle sources, got {sources}")
    if profile == "openo3_mixed":
        expected = {"dam_cyclegrpo", "tg_multi_merged", "stgr_cyclegrpo", "open_o3_understanding_grpo"}
        if sources != expected or "sampling_weight" not in columns:
            raise ValueError("Open-o3 main recipe requires all three cycle routes, auxiliary QA, and sampling_weight")
    return num_rows, sorted(sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=("temporal_qwen", "temporal_omtg", "openo3_mixed"))
    args = parser.parse_args()
    import generalized.verl as verl
    runtime = Path(__file__).resolve().parent
    if not Path(verl.__file__).resolve().is_relative_to(runtime):
        raise ValueError("Wrong verl selected: run via generalized/train.sh or export its PYTHONPATH")
    for name in ("ROLLOUT_N", "CYCLEGRPO_INNER_ROLLOUT_N"):
        if int(os.environ[name]) < 2:
            raise ValueError(f"{name} must be >= 2 for group-relative optimization")
    if int(os.environ["ROLLOUT_BATCH_SIZE"]) % int(os.environ["GLOBAL_BATCH_SIZE"]):
        raise ValueError("ROLLOUT_BATCH_SIZE must be divisible by GLOBAL_BATCH_SIZE")
    shards = check_checkpoint(os.environ["MODEL_PATH"])
    rows, sources = check_data(os.environ["TRAIN_FILE"], args.profile)
    print(json.dumps({"profile": args.profile, "model_shards": shards, "rows": rows,
                      "sources": sources, "runtime": "generalized/verl"}, indent=2))


if __name__ == "__main__":
    main()
