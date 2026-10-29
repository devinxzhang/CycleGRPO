import random
import re

from verl.utils.dataset import sample_single_target_from_multi_target


A = "<|mt_start|><|mt_0003|><|mt_0311|><|mt_end|>"
B = "<|mt_start|><|mt_0086|><|mt_0483|><|mt_end|>"
C = "<|mt_start|><|mt_0120|><|mt_0400|><|mt_end|>"
ZA = "<|mt_start|><|mt_0198|><|mt_0300|><|mt_end|>"
ZC = "<|mt_start|><|mt_0003|><|mt_0332|><|mt_end|>"


def multi_target(targets, zooms):
    """Build a prompt the way prepare_dw_rl_dataset.py does; zooms maps target index -> zoom token."""
    prompt = (
        "<image>\nCould you please give me a detailed description of the following regions? "
        + ", ".join(targets[:-1])
        + f", and {targets[-1]}. Please respond with interleaved segmentation masks for the corresponding parts of the answer."
    )
    images = ["full.jpg"]
    for idx in sorted(zooms):
        prompt += f" Zoom in on {targets[idx]} with the perspective as <image>, {zooms[idx]}."
        images.append(f"crop{idx}.jpg")
    return prompt, "<answer>" + ", ".join(targets) + "</answer>", images


def test_one_target_keeps_only_its_zoom_in():
    prompt, answer, images = multi_target([A, B, C], {0: ZA, 2: ZC})
    seen = set()
    for seed in range(50):
        random.seed(seed)
        new_prompt, new_answer, new_images = sample_single_target_from_multi_target(
            prompt, answer, images, max_targets=1
        )
        (target,) = re.findall(r"<\|mt_start\|>.*?<\|mt_end\|>", new_answer)
        seen.add(target)
        assert new_prompt.count("<image>") == len(new_images)
        if target == A:
            assert new_images == ["full.jpg", "crop0.jpg"] and new_prompt.endswith(
                f"Zoom in on {A} with the perspective as <image>, {ZA}."
            )
        elif target == B:
            assert new_images == ["full.jpg"] and "Zoom in" not in new_prompt
        else:
            assert new_images == ["full.jpg", "crop2.jpg"] and new_prompt.endswith(
                f"Zoom in on {C} with the perspective as <image>, {ZC}."
            )
    assert seen == {A, B, C}


def test_duplicate_tokens_map_zoom_in_by_position():
    # Two regions can quantize to the same token; only the second one has a zoom-in.
    prompt, answer, images = multi_target([A, A, B], {1: ZA})
    _, _, all_images = sample_single_target_from_multi_target(prompt, answer, images, max_targets=3)
    assert all_images == images
    for seed in range(50):
        random.seed(seed)
        new_prompt, _, new_images = sample_single_target_from_multi_target(prompt, answer, images, max_targets=1)
        assert new_prompt.count("<image>") == len(new_images) <= 2
        assert new_prompt.count("Zoom in") == len(new_images) - 1


def test_prompts_without_regions_are_unchanged():
    prompt = "<image>\nCould you provide a segmentation mask for the dog in this image?"
    answer = "<answer>No target.</answer>"
    assert sample_single_target_from_multi_target(prompt, answer, ["coco.jpg"], max_targets=1) == (
        prompt,
        answer,
        ["coco.jpg"],
    )
