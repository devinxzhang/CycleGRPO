"""GRES resume must never accept results from another dataset or model."""

import json
import tempfile
import unittest
from pathlib import Path

from result_store import (
    count_current_results,
    load_cases,
    load_current_result,
    make_run_id,
    save_result,
    valid_compressed_rle,
)


class ResultStoreTest(unittest.TestCase):
    def test_run_identity_and_result_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "cases.json"
            mask = {"size": [2, 2], "counts": "04"}
            dataset.write_text(json.dumps([{"segmentation": mask, "phrase": "object"}]))
            cases, dataset_bytes = load_cases(dataset)
            run_id = make_run_id(dataset_bytes, "model-a", "vq-a", "sam-a")
            self.assertEqual(count_current_results(root, cases, run_id), 0)

            save_result(root, cases[0], run_id, "model-a", {"gt_masks": mask, "pred_masks": mask})
            self.assertIsNotNone(load_current_result(root, cases[0], run_id))
            self.assertEqual(count_current_results(root, cases, run_id), 1)
            self.assertIsNone(load_current_result(root, cases[0], make_run_id(dataset_bytes, "model-b", "vq-a", "sam-a")))

            changed_dataset = json.dumps([{"segmentation": mask, "phrase": "another object"}]).encode()
            self.assertNotEqual(run_id, make_run_id(changed_dataset, "model-a", "vq-a", "sam-a"))

            result_path = root / "0.json"
            record = json.loads(result_path.read_text())
            record["pred_masks"]["size"] = [1, 4]
            result_path.write_text(json.dumps(record))
            self.assertIsNone(load_current_result(root, cases[0], run_id))

            record["pred_masks"] = {"size": [2, 2], "counts": "garbage"}
            result_path.write_text(json.dumps(record))
            self.assertIsNone(load_current_result(root, cases[0], run_id))

    def test_invalid_compressed_rle_is_not_complete(self):
        self.assertTrue(valid_compressed_rle({"size": [2, 2], "counts": "04"}))
        self.assertTrue(valid_compressed_rle({"size": [2, 2], "counts": "4"}))
        self.assertTrue(valid_compressed_rle({"size": [2, 2], "counts": "0111"}))
        self.assertTrue(valid_compressed_rle({"size": [4, 8], "counts": "P1"}))
        self.assertFalse(valid_compressed_rle({"size": [2, 2], "counts": "garbage"}))
        self.assertFalse(valid_compressed_rle({"size": [2, 2], "counts": "0"}))

    def test_empty_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "empty.json"
            dataset.write_text("[]")
            with self.assertRaises(ValueError):
                load_cases(dataset)


if __name__ == "__main__":
    unittest.main()
