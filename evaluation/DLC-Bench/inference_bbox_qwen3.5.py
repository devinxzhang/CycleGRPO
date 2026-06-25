# --------------------------------------------------------
# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
# Grasp Any Region Project
# Written by Haochen Wang
# --------------------------------------------------------

import argparse
import base64
import io
import json
import os
import re

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

TORCH_DTYPE_MAP = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)


def extract_final_output(text: str) -> str:
    """Keep only final answer text and remove explicit thinking traces."""
    if not isinstance(text, str):
        return text

    # If model provides structured answer, prioritize answer section.
    answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        text = answer_match.group(1)

    # Remove think tags and their content if present.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = text.replace("<|im_end|>", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference of Qwen3.5-VL with BBox on DLC-Bench."
    )

    parser.add_argument(
        "--model_path",
        help="HF model name or local path",
        default="Qwen/Qwen3.5-VL-4B-Instruct",
    )
    parser.add_argument(
        "--cache_name",
        help="Cache name to save model outputs.",
        default="qwen3.5vl_bbox",
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
        "--image_folder",
        help="The folder of images",
        default="evaluation/DLC-Bench/annotations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible text generation",
    )
    return parser.parse_args()


def decode_mask(object_masks, ori_height, ori_width):
    binary_masks = []
    for object_mask in object_masks:
        if isinstance(object_mask, dict):
            if isinstance(object_mask["counts"], list):
                object_mask = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
            m = mask_utils.decode(object_mask)
            m = m.astype(np.uint8).squeeze()
        elif object_mask:
            rles = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
            rle = mask_utils.merge(rles)
            m = mask_utils.decode(rle).astype(np.uint8).squeeze()
        else:
            m = np.zeros((ori_height, ori_width), dtype=np.uint8)
        binary_masks.append(m)
    return binary_masks


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


def main():
    args = parse_args()
    _ = TORCH_DTYPE_MAP[args.data_type]
    torch.manual_seed(args.seed)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    ).eval()

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    model_outputs = {}
    cache_name = args.cache_name

    coco = COCO(args.anno_file)
    img_ids = list(coco.imgs.keys())
    num_anns = len(coco.anns)
    pbar = tqdm(total=num_anns)

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

            binary_masks = [mask]
            masks_tensor = torch.stack(
                [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks]
            )
            boxes = torchvision.ops.masks_to_boxes(masks_tensor)

            x1, y1, x2, y2 = boxes.squeeze().cpu().numpy().tolist()
            boxes_w = x2 - x1
            boxes_h = y2 - y1
            boxes_area = boxes_h * boxes_w
            image_area = ori_height * ori_width
            boxes_occupied_ratio = boxes_area / image_area

            norm_x1 = int(x1 / ori_width * 1000)
            norm_y1 = int(y1 / ori_height * 1000)
            norm_x2 = int(x2 / ori_width * 1000)
            norm_y2 = int(y2 / ori_height * 1000)
            bbox_str = f"[{norm_x1}, {norm_y1}, {norm_x2}, {norm_y2}]"

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
                    f"Given a detailed description of the region at bounding box {bbox_str}. "
                    "Zoom in with the perspective as"
                )

                buffer = io.BytesIO()
                if resized_crop_image is None:
                    cropped_image.save(buffer, format="JPEG")
                else:
                    resized_crop_image.save(buffer, format="JPEG")
                buffer.seek(0)
                crop_b64 = base64.b64encode(buffer.read()).decode("utf-8")

                with open(img_path, "rb") as f:
                    global_b64 = base64.b64encode(f.read()).decode()

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{global_b64}",
                            },
                            {"type": "text", "text": question},
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{crop_b64}",
                            },
                            {
                                "type": "text",
                                "text": ", give a detailed description of this cropped region.",
                            },
                        ],
                    }
                ]
            else:
                question = f"Given a detailed description of the region at bounding box {bbox_str}."
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{b64}",
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    top_p=1.0,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            outputs = extract_final_output(output_text[0])
            print(outputs)

            model_outputs[ann_id] = outputs
            pbar.update(1)

    pbar.close()

    os.makedirs("evaluation/DLC-Bench/model_outputs", exist_ok=True)
    with open(f"evaluation/DLC-Bench/model_outputs/{cache_name}.json", "w") as file:
        json.dump(model_outputs, file, indent=4, ensure_ascii=False)

    print(f"Cache name: {cache_name}")


if __name__ == "__main__":
    main()
