"""Content-addressed response caching for DLC judge evaluations."""

import hashlib
import json


def evaluation_cache_key(prediction, questions, prompt_template, judge, images=None):
    """Identify everything sent to the judge, excluding credentials."""

    inputs = {
        "prediction": prediction,
        "questions": questions,
        "prompt_template": prompt_template,
        "judge": judge,
    }
    if images is not None:
        image_hash = hashlib.sha256()
        for encoded_image in images:
            image_bytes = encoded_image.encode("ascii")
            image_hash.update(len(image_bytes).to_bytes(8, "big"))
            image_hash.update(image_bytes)
        inputs["images_sha256"] = image_hash.hexdigest()

    serialized = json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def reusable_responses(entry, cache_key, question_count):
    """Reject legacy, stale, partial, or malformed cached responses."""

    if not isinstance(entry, dict) or entry.get("cache_key") != cache_key:
        return None
    responses = entry.get("response")
    if not isinstance(responses, list) or len(responses) != question_count:
        return None
    if not all(isinstance(response, str) and response.strip() for response in responses):
        return None
    return responses
