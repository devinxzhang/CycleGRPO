"""Check spatial CycleGRPO RL parquet files without loading them fully.

Reads the parquet footer for the schema and row count, then validates only the
first ``--sample`` rows: required columns, prompt/image consistency, mask-token
syntax, and that referenced images exist.

    python projects/rl/datasets/check_rl_parquet.py \
        data/cyclegrpo/denseworld_train.parquet data/cyclegrpo/gres_no_target_train.parquet
"""

import argparse
import os
import re
import sys

import pyarrow.parquet as pq


REQUIRED_COLUMNS = ("images", "cap_problem", "seg_answer", "source")
# A SAMTok 256x2 region: start token, one code per codebook depth, end token.
MASK_TOKEN = re.compile(r"<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>")
CODEBOOK_SIZE = 256


def _row_problems(row, image_dir):
    problems = []
    images = row["images"] or []
    if not images:
        problems.append("no images")
    for image in images:
        path = image if os.path.isabs(image) or image_dir is None else os.path.join(image_dir, image)
        if not os.path.isfile(path):
            problems.append(f"image not found: {path}")
    if (row["cap_problem"] or "").count("<image>") != len(images):
        problems.append(f"cap_problem has {row['cap_problem'].count('<image>')} <image> tags for {len(images)} images")
    answer = row["seg_answer"] or ""
    if row["source"] == "gres_no_target":
        if answer != "<answer>No target.</answer>":
            problems.append(f"no-target seg_answer is {answer!r}")
    else:
        regions = MASK_TOKEN.findall(answer)
        if not regions:
            problems.append("seg_answer contains no <|mt_start|>..<|mt_end|> region")
        for first, second in regions:
            if not (0 <= int(first) < CODEBOOK_SIZE <= int(second) < 2 * CODEBOOK_SIZE):
                problems.append(f"mask codes out of range: {first}, {second}")
    return problems


def check_parquet(path, sample=32, image_dir=None):
    """Return (num_rows, sources_in_sample, problems) for one parquet file."""
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema_arrow.names)
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        return parquet.metadata.num_rows, set(), [f"missing columns: {missing}"]
    if parquet.metadata.num_rows == 0:
        return 0, set(), ["parquet has no rows"]
    batch = next(parquet.iter_batches(batch_size=sample, columns=list(REQUIRED_COLUMNS)))
    problems, sources = [], set()
    for index, row in enumerate(batch.to_pylist()):
        sources.add(row["source"])
        problems.extend(f"row {index}: {problem}" for problem in _row_problems(row, image_dir))
    return parquet.metadata.num_rows, sources, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquets", nargs="+")
    parser.add_argument("--sample", type=int, default=32, help="Rows to validate from the start of each file.")
    parser.add_argument(
        "--image_dir",
        default=None,
        help="Base directory for relative image paths (data.image_dir); default: current directory.",
    )
    args = parser.parse_args()
    failed = False
    for path in args.parquets:
        if not os.path.isfile(path):
            print(f"FAIL {path}: file not found")
            failed = True
            continue
        rows, sources, problems = check_parquet(path, args.sample, args.image_dir)
        status = "FAIL" if problems else "OK"
        print(f"{status} {path}: {rows} rows, sources in first {args.sample}: {sorted(sources)}")
        for problem in problems[:10]:
            print(f"    {problem}")
        if len(problems) > 10:
            print(f"    ... {len(problems) - 10} more")
        failed |= bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
