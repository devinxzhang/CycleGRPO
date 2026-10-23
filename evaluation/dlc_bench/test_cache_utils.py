import unittest

from cache_utils import evaluation_cache_key, reusable_responses


class EvaluationCacheTest(unittest.TestCase):
    def setUp(self):
        self.prediction = "A red cup"
        self.questions = [
            {"question": "What color?", "choices": [["Red", 1], ["Blue", 0]]}
        ]
        self.judge = {"model": "judge-v1", "temperature": 0.0}

    def key(self, **changes):
        inputs = {
            "prediction": self.prediction,
            "questions": self.questions,
            "prompt_template": "{pred_caption}: {question_text_str}",
            "judge": self.judge,
            "images": ["image", "mask"],
        }
        inputs.update(changes)
        return evaluation_cache_key(**inputs)

    def test_changed_judge_inputs_invalidate_cache(self):
        original = self.key()
        self.assertNotEqual(original, self.key(prediction="A blue cup"))
        self.assertNotEqual(original, self.key(questions=[]))
        self.assertNotEqual(original, self.key(prompt_template="new prompt"))
        self.assertNotEqual(original, self.key(judge={"model": "judge-v2"}))
        self.assertNotEqual(original, self.key(images=["image", "new mask"]))

    def test_only_complete_matching_responses_are_reused(self):
        cache_key = self.key()
        entry = {"cache_key": cache_key, "response": ["A"]}
        self.assertEqual(reusable_responses(entry, cache_key, 1), ["A"])
        self.assertIsNone(reusable_responses(entry, "changed", 1))
        self.assertIsNone(reusable_responses(entry, cache_key, 2))
        self.assertIsNone(reusable_responses({"response": ["A"]}, cache_key, 1))
        self.assertIsNone(
            reusable_responses({"cache_key": cache_key, "response": [""]}, cache_key, 1)
        )


if __name__ == "__main__":
    unittest.main()
