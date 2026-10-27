import pyarrow as pa
import pyarrow.parquet as pq

from projects.rl.datasets.check_rl_parquet import check_parquet


REGION = "<|mt_start|><|mt_0087|><|mt_0290|><|mt_end|>"


def write(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)
    return str(path)


def row(image, **overrides):
    base = dict(
        images=[str(image)],
        cap_problem=f"<image>\nDescribe {REGION}.",
        seg_answer=f"<answer>{REGION}</answer>",
        source="denseworld_multiple",
    )
    return {**base, **overrides}


def test_valid_cycle_and_no_target_rows(tmp_path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"")
    path = write(
        tmp_path / "ok.parquet",
        [
            row(image),
            row(image, seg_answer="<answer>No target.</answer>", source="gres_no_target"),
        ],
    )
    assert check_parquet(path) == (2, {"denseworld_multiple", "gres_no_target"}, [])


def test_reports_row_problems(tmp_path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"")
    path = write(
        tmp_path / "bad.parquet",
        [
            row(tmp_path / "missing.jpg"),
            row(image, images=[str(image), str(image)]),
            row(image, seg_answer="<answer><|mt_start|><|mt_0300|><|mt_0001|><|mt_end|></answer>"),
            row(image, seg_answer="<answer>a dog</answer>", source="gres_no_target"),
        ],
    )
    _, _, problems = check_parquet(path)
    assert problems == [
        f"row 0: image not found: {tmp_path / 'missing.jpg'}",
        "row 1: cap_problem has 1 <image> tags for 2 images",
        "row 2: mask codes out of range: 0300, 0001",
        "row 3: no-target seg_answer is '<answer>a dog</answer>'",
    ]


def test_missing_columns_short_circuit(tmp_path):
    path = write(tmp_path / "cols.parquet", [{"images": ["x.jpg"], "source": "x"}])
    assert check_parquet(path) == (1, set(), ["missing columns: ['cap_problem', 'seg_answer']"])
