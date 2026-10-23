"""GCG resume identity must follow the model and image contents."""

import json
import tempfile
import unittest
from pathlib import Path

from gcg_result_store import count_current_results, image_sha256, make_run_id, output_is_complete


class ResultStoreTest(unittest.TestCase):
    def test_changed_model_or_image_invalidates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_folder = root / "images"
            save_dir = root / "results"
            image_folder.mkdir()
            save_dir.mkdir()
            image = image_folder / "example.png"
            image.write_bytes(b"image A")
            run_id = make_run_id("model-a", "vq-a", "sam-a")
            image_hash = image_sha256(image)
            result = {
                "image_id": "example",
                "run_id": run_id,
                "image_sha256": image_hash,
                "caption": "object",
                "phrases": ["object"],
                "pred_masks": [{"size": [2, 2], "counts": "04"}],
            }
            result_path = save_dir / "example.json"
            result_path.write_text(json.dumps(result))

            self.assertTrue(output_is_complete(result_path, "example", run_id, image_hash))
            self.assertEqual(count_current_results(image_folder, save_dir, run_id), (1, 1))
            self.assertEqual(count_current_results(image_folder, save_dir,
                                                   make_run_id("model-b", "vq-a", "sam-a")), (0, 1))

            image.write_bytes(b"image B")
            self.assertEqual(count_current_results(image_folder, save_dir, run_id), (0, 1))

            image.write_bytes(b"image A")
            result["pred_masks"][0]["counts"] = "garbage"
            result_path.write_text(json.dumps(result))
            self.assertEqual(count_current_results(image_folder, save_dir, run_id), (0, 1))


if __name__ == "__main__":
    unittest.main()
