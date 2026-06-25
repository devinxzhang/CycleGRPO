# --------------------------------------------------------
# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
# Grasp Any Region Project - Llama 3.1 (text-only) baseline
# --------------------------------------------------------

import argparse
import json
import os
import re

import torch
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TORCH_DTYPE_MAP = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)


def clean_output(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference of Meta-Llama-3.1-8B-Instruct on DLC-Bench (text-only baseline)."
    )

    parser.add_argument(
        "--model_path",
        help="HF model name or local path",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--cache_name",
        help="Cache name to save model outputs.",
        default="llama3.1_bbox",
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
        default="evaluation/DLC-Bench/annotations/annotations.json",
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
        default=256,
        help="Maximum generated tokens",
    )
    return parser.parse_args()


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


def build_messages(question: str):
    return [
        {
            "role": "system",
            "content": (
                "You are a text-only assistant. You do not have access to image pixels. "
                "Given only region metadata, provide a conservative short region description. "
                "If the information is insufficient, output: insufficient visual information."
            ),
        },
        {"role": "user", "content": question},
    ]


def main():
    args = parse_args()
    dtype = TORCH_DTYPE_MAP[args.data_type]
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map="auto",
    ).eval()

    print("[WARN] Meta-Llama-3.1-8B-Instruct is text-only and cannot directly see images.")

    model_outputs = {}
    coco = COCO(args.anno_file)
    img_ids = list(coco.imgs.keys())
    num_anns = len(coco.anns)
    pbar = tqdm(total=num_anns)

    for img_id in img_ids:
        ann_ids = select_ann(coco, img_id)
        img_info = coco.loadImgs(img_id)[0]
        ori_width = img_info["width"]
        ori_height = img_info["height"]

        for ann_id in ann_ids:
            if ann_id in model_outputs:
                pbar.update(1)
                continue

            ann = coco.loadAnns([ann_id])[0]
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h

            norm_x1 = int(x1 / max(ori_width, 1) * 1000)
            norm_y1 = int(y1 / max(ori_height, 1) * 1000)
            norm_x2 = int(x2 / max(ori_width, 1) * 1000)
            norm_y2 = int(y2 / max(ori_height, 1) * 1000)
            bbox_str = f"[{norm_x1}, {norm_y1}, {norm_x2}, {norm_y2}]"

            area_ratio = float((w * h) / max(ori_width * ori_height, 1))
            question = (
                f"Image size: {ori_width}x{ori_height}. "
                f"Target bounding box (normalized to 0-1000): {bbox_str}. "
                f"Approximate area ratio: {area_ratio:.4f}. "
                "Describe the likely region content in one short sentence."
            )

            messages = build_messages(question)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            response = clean_output(response)
            print(response)

            model_outputs[ann_id] = response
            pbar.update(1)

    pbar.close()
    os.makedirs("evaluation/DLC-Bench/model_outputs", exist_ok=True)
    out_path = f"evaluation/DLC-Bench/model_outputs/{args.cache_name}.json"
    with open(out_path, "w") as f:
        json.dump(model_outputs, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(model_outputs)} outputs to {out_path}")


if __name__ == "__main__":
    main()
