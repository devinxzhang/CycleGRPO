# --------------------------------------------------------
# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
# Grasp Any Region Project - InternVL3.5 region captioning
# --------------------------------------------------------

import argparse
import json
import os
import re

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from torchvision import transforms as T
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoTokenizer

TORCH_DTYPE_MAP = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def extract_final_output(text: str) -> str:
    """Remove think tags and normalize whitespace."""
    if not isinstance(text, str):
        return text

    answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        text = answer_match.group(1)

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference of InternVL3.5 with BBox prompting on DLC-Bench."
    )

    parser.add_argument(
        "--model_path",
        help="HF model name or local path",
        default="OpenGVLab/InternVL3_5-4B",
    )
    parser.add_argument(
        "--cache_name",
        help="Cache name to save model outputs.",
        default="internvl3.5_bbox",
    )
    parser.add_argument(
        "--data_type",
        help="Data dtype",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="bf16",
    )
    parser.add_argument(
        "--anno_file",
        help="Path to the annotation file.",
        default="evaluation/dlc_bench/annotations/annotations.json",
    )
    parser.add_argument(
        "--image_folder",
        help="The folder of images",
        default="evaluation/dlc_bench/annotations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible text generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum generated tokens",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=448,
        help="Input size for InternVL image encoder",
    )
    return parser.parse_args()


def build_transform(image_size: int):
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def select_ann(coco, img_id, area_min=None, area_max=None):
    cat_ids = coco.getCatIds()
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)

    if area_min is not None:
        ann_ids = [
            ann_id for ann_id in ann_ids if coco.anns[ann_id]["area"] >= area_min
        ]

    if area_max is not None:
        ann_ids = [
            ann_id for ann_id in ann_ids if coco.anns[ann_id]["area"] <= area_max
        ]

    return ann_ids


def build_question_and_images(image: Image.Image, bbox_str: str, boxes_occupied_ratio: float, x1, y1, x2, y2):
    """Keep the same zoom-in policy as qwen/gemma scripts."""
    ori_width, ori_height = image.size

    if boxes_occupied_ratio < 0.3:
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        if bbox_w < 140:
            x1 = x1 - (140 - bbox_w) // 2
            x2 = x2 + (140 - bbox_w) // 2
        if bbox_h < 140:
            y1 = y1 - (140 - bbox_h) // 2
            y2 = y2 + (140 - bbox_h) // 2
        x1 = int(max(0, x1))
        x2 = int(min(ori_width, x2))
        y1 = int(max(0, y1))
        y2 = int(min(ori_height, y2))

        cropped_image = image.crop((x1, y1, x2, y2))
        crop_width, crop_height = cropped_image.size

        if crop_width > crop_height and crop_width < 280:
            ratio = 280 / crop_height
            new_height = 280
            new_width = int(crop_width * ratio)
            resized_crop_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        elif crop_height > crop_width and crop_height < 280:
            ratio = 280 / crop_width
            new_width = 280
            new_height = int(crop_height * ratio)
            resized_crop_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        elif crop_height == crop_width and crop_width < 280:
            ratio = 280 / crop_height
            new_height = 280
            new_width = int(crop_width * ratio)
            resized_crop_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            resized_crop_image = None

        question = (
            f"The first image is the full scene. The second image is a zoomed-in crop of the region at "
            f"bounding box {bbox_str} (coordinates normalized to 0-1000). "
            "Provide a detailed description of this cropped region. "
            "Output only the final description, no preamble."
        )
        image_list = [image, cropped_image if resized_crop_image is None else resized_crop_image]
    else:
        question = (
            f"Provide a detailed description of the region at bounding box {bbox_str} "
            "(coordinates normalized to 0-1000). Output only the final description, no preamble."
        )
        image_list = [image]

    return question, image_list


def build_internvl_prompt(question: str, num_images: int) -> str:
    """Construct prompt with one <image> token per image for InternVL chat."""
    if num_images <= 1:
        return f"<image>\n{question}"

    header = []
    for idx in range(num_images):
        header.append(f"Image-{idx + 1}: <image>")
    return "\n".join(header) + "\n" + question


def main():
    args = parse_args()
    dtype = TORCH_DTYPE_MAP[args.data_type]
    torch.manual_seed(args.seed)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    transform = build_transform(args.image_size)

    model_outputs = {}
    cache_name = args.cache_name

    coco = COCO(args.anno_file)
    img_ids = list(coco.imgs.keys())
    num_anns = len(coco.anns)
    pbar = tqdm(total=num_anns)

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }

    for img_id in img_ids:
        ann_ids = select_ann(coco, img_id)
        img_info = coco.loadImgs(img_id)[0]

        for ann_id in ann_ids:
            if ann_id in model_outputs:
                pbar.update(1)
                continue

            anns = coco.loadAnns([ann_id])
            mask = coco.annToMask(anns[0])

            img_path = os.path.join(args.image_folder, "images", img_info["file_name"])
            image = Image.open(img_path).convert("RGB")
            ori_width, ori_height = image.size

            masks_tensor = torch.from_numpy(np.ascontiguousarray(mask.copy()))[None]
            boxes = torchvision.ops.masks_to_boxes(masks_tensor)
            x1, y1, x2, y2 = boxes.squeeze(0).cpu().numpy().tolist()

            boxes_w = x2 - x1
            boxes_h = y2 - y1
            boxes_area = boxes_h * boxes_w
            image_area = ori_height * ori_width
            boxes_occupied_ratio = boxes_area / max(image_area, 1)

            norm_x1 = int(x1 / ori_width * 1000)
            norm_y1 = int(y1 / ori_height * 1000)
            norm_x2 = int(x2 / ori_width * 1000)
            norm_y2 = int(y2 / ori_height * 1000)
            bbox_str = f"[{norm_x1}, {norm_y1}, {norm_x2}, {norm_y2}]"

            question, image_list = build_question_and_images(
                image=image,
                bbox_str=bbox_str,
                boxes_occupied_ratio=boxes_occupied_ratio,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )

            pixel_values = torch.stack([transform(img) for img in image_list], dim=0)
            model_device = next(model.parameters()).device
            pixel_values = pixel_values.to(model_device, dtype=dtype)
            num_patches_list = [1] * len(image_list)
            prompt = build_internvl_prompt(question, num_images=len(image_list))

            # Preferred InternVL chat API with num_patches_list for multi-image.
            try:
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    prompt,
                    generation_config,
                    num_patches_list=num_patches_list,
                )
            except TypeError:
                # Compatibility fallback for variants with a simpler chat signature.
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    prompt,
                    generation_config,
                )

            outputs = extract_final_output(response)
            print(outputs)

            model_outputs[ann_id] = outputs
            pbar.update(1)

    pbar.close()

    os.makedirs("evaluation/dlc_bench/model_outputs", exist_ok=True)
    out_path = f"evaluation/dlc_bench/model_outputs/{cache_name}.json"
    with open(out_path, "w") as file:
        json.dump(model_outputs, file, indent=4, ensure_ascii=False)

    print(f"Saved {len(model_outputs)} outputs to {out_path}")


if __name__ == "__main__":
    main()
