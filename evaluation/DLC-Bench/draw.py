# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2025) B-
# ytedance Inc..
# *************************************************************************

# Adapted from https://github.com/NVlabs/describe-anything/blob/main/evaluation/eval_model_outputs.py

# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import io
import json
import os

import inflect
import numpy as np
import openai
from PIL import Image
from pycocotools.coco import COCO
from scipy import ndimage
from tqdm import tqdm


def mask_to_box(mask_np):
    mask_coords = np.argwhere(mask_np)
    y0, x0 = mask_coords.min(axis=0)
    y1, x1 = mask_coords.max(axis=0) + 1

    h = y1 - y0
    w = x1 - x0

    return x0, y0, w, h


def encode_pil_image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model outputs")
    parser.add_argument(
        "--pred", type=str, help="Path to the prediction JSON file", required=True
    )
    parser.add_argument(
        "--data-root", type=str, default="evaluation/DLC-Bench/annotations"
    )
    args = parser.parse_args()

    with open(args.pred) as f:
        data_pred = json.load(f)

    annotations_file = os.path.join(args.data_root, "annotations.json")
    coco = COCO(annotations_file)

    with open(annotations_file, "r") as f:
        data = json.load(f)

    # Load samtok and cycle_grpo captions
    samtok_caption_file = './quantitive_comparison/dlc_bench/samtok.json'
    grpo_caption_file = './quantitive_comparison/dlc_bench/cycle_grpo.json'
    
    with open(samtok_caption_file, 'r') as f:
        samtok_captions = json.load(f)
    with open(grpo_caption_file, 'r') as f:
        grpo_captions = json.load(f)

    for sample in tqdm(data_pred):
        sample_id = str(sample["id"])
        for item in data["annotations"]:
            if int(item["id"]) == int(sample_id):
                img_id = item["image_id"]

        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.data_root, "images", img_info["file_name"])
        img = Image.open(img_path)

        anns = coco.loadAnns([int(sample_id)])
        mask_np = coco.annToMask(anns[0]).astype(bool)

        img_np = np.array(img)
        pil_mask = Image.fromarray((mask_np * 255).astype(np.uint8))

        save_folder = './quantitive_comparison/dlc_bench/picked_cases_vis'
        os.makedirs(save_folder, exist_ok=True)

        # 创建带有mask overlay的图像
        # 定义mask颜色 (绿色) 和透明度
        mask_color = np.array([0, 255, 0], dtype=np.uint8)  # 绿色
        alpha = 0.5  # mask区域透明度
        darken_factor = 0.4  # 非mask区域变暗系数 (0-1, 越小越暗)

        # 创建输出图像
        overlay_img = img_np.copy()
        
        # 将非mask区域变暗
        overlay_img[~mask_np] = (img_np[~mask_np] * darken_factor).astype(np.uint8)
        
        # 将mask区域覆盖为半透明颜色
        overlay_img[mask_np] = (
            (1 - alpha) * img_np[mask_np] + alpha * mask_color
        ).astype(np.uint8)

        # 可选：在mask边界画轮廓
        # 找到mask边界
        eroded = ndimage.binary_erosion(mask_np, iterations=2)
        boundary = mask_np & ~eroded
        # 用更深的颜色画边界
        boundary_color = np.array([0, 200, 0], dtype=np.uint8)  # 深绿色
        overlay_img[boundary] = boundary_color

        # 保存结果
        result_img = Image.fromarray(overlay_img)
        save_path = os.path.join(save_folder, f"{sample_id}.png")
        result_img.save(save_path)
        
        # 保存原图
        original_save_path = os.path.join(save_folder, f"{sample_id}_original.png")
        img.save(original_save_path)
        
        # 保存samtok和cycle_grpo的caption
        samtok_caption = samtok_captions.get(sample_id, "N/A")
        grpo_caption = grpo_captions.get(sample_id, "N/A")
        
        caption_save_path = os.path.join(save_folder, f"{sample_id}_captions.txt")
        with open(caption_save_path, 'w', encoding='utf-8') as f:
            f.write(f"Sample ID: {sample_id}\n")
            f.write("=" * 50 + "\n\n")
            f.write("SAMTok Caption:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{samtok_caption}\n\n")
            f.write("Cycle-GRPO Caption:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{grpo_caption}\n")
        
        print(f"Saved: {save_path}")