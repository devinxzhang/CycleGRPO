import argparse
import os
import sys
import collections
import os.path as osp
import random
import copy
from typing import Dict, List, Any, Tuple
from PIL import Image
import numpy as np
import torch
import torchvision
from pycocotools import mask as mask_utils
import json
import tqdm
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import io
import re
import uuid
import hydra
import base64
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageColor
from datasets import Dataset, DatasetDict, Sequence
from datasets import Image as ImageData
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

import mmengine
from mmengine.dataset import BaseDataset
import pycocotools.mask as mask_util

from transformers import Sam2Processor, Sam2Model
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from torchvision.transforms.functional import resize, to_pil_image
def parse_mask_token(mask_2d: str, codebook_size: int = 256) -> list[int]:
    pattern = r'<\|mt_(\d+)\|>'
    matches = re.findall(pattern, mask_2d)
    codes = [int(m) for m in matches]
    return codes

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length
    def apply_image(self, image: np.ndarray) -> np.ndarray:
        img = Image.fromarray(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

def overlay_mask_on_image(image: Image.Image, mask: np.ndarray, color_hex="#00FF00", alpha=150):
    base_rgba = image.convert("RGBA")
    H, W = base_rgba.size[1], base_rgba.size[0]
    mask_bool = mask > 0
    overlay_np = np.zeros((H, W, 4), dtype=np.uint8)
    rgb_color = ImageColor.getrgb(color_hex)
    overlay_np[mask_bool, 0] = rgb_color[0]
    overlay_np[mask_bool, 1] = rgb_color[1]
    overlay_np[mask_bool, 2] = rgb_color[2]
    overlay_np[mask_bool, 3] = alpha
    overlay_pil = Image.fromarray(overlay_np, "RGBA")
    combined_img = Image.alpha_composite(base_rgba, overlay_pil)
    return combined_img.convert("RGB")

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('--model_path', 
        default="Qwen/Qwen3-VL-4B-SAMTok",
        help='hf model path.')
    parser.add_argument(
        '--vq_sam2_path',
        default="Qwen/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--dataset',
        default='data/GroundingME',
        help='Specify a ref dataset')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of threads for parallel processing')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for GPU inference')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    random.seed(42)

    # 加载parquet数据
    parquet_path = "rl_dataset/denseworld_10img_samples_train.parquet"
    ds = load_dataset("parquet", data_files=parquet_path, split="train")

    # 初始化 VQ-SAM2 模型用于 decode mask token
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'
    
    with hydra.initialize(version_base=None, config_path="../../../projects/transformers/vq_sam2/sam2/sam2_configs"):
        sam2_config = SAM2Config(
            cfg_path="sam2.1_hiera_l.yaml",
            ckpt_path="Qwen/sam2.1_hiera_large.pt",
        )
        
        vq_sam2_config = VQ_SAM2Config(
            sam2_config=sam2_config,
            codebook_size=CODEBOOK_SIZE,
            codebook_depth=CODEBOOK_DEPTH,
            shared_codebook=False,
            latent_dim=256,
        )
    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()
    state = torch.load(args.vq_sam2_path, map_location="cpu")
    vq_sam2.load_state_dict(state)
    vq_sam2_image_processor = DirectResize(1024)
    
    print("VQ-SAM2 model loaded successfully")

    os.makedirs("vis_mask_overlay", exist_ok=True)

    for idx, item in enumerate(tqdm(ds, desc="Visualizing")):
        # 原图
        image_path = item['images'][0]
        image = Image.open(image_path).convert("RGB")
        mask_token = None
        if item['cap_problem'] and '<|mt_start|>' in item['cap_problem']:
            mask_token = re.search(r'<\|mt_start\|>.*?<\|mt_end\|>', item['cap_problem'])
            if mask_token:
                mask_token = mask_token.group(0)
        if mask_token is None or mask_token == 'None':
            # 无mask，直接保存原图
            image.save(f"vis_mask_overlay/{idx}_orig_nomask.jpg")
            continue
        # decode mask
        codes = parse_mask_token(mask_token, CODEBOOK_SIZE)
        if len(codes) != CODEBOOK_DEPTH:
            continue
        original_codes = [codes[i] - i * CODEBOOK_SIZE for i in range(len(codes))]
        sam2_image = np.array(image)
        sam2_image = vq_sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
        quant_codes = torch.tensor(original_codes, dtype=torch.long).unsqueeze(0).cuda()
        with torch.no_grad():
            pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_codes)
            pred_masks = pred_masks.detach()
            pred_masks = torch.nn.functional.interpolate(pred_masks, size=(image.height, image.width), mode='bilinear')
            pred_masks = pred_masks > 0.5
            pred_masks = pred_masks[0, 0, :, :].cpu().numpy().astype(np.uint8)
        overlayed = overlay_mask_on_image(image, pred_masks)
        overlayed.save(f"vis_mask_overlay/{idx}_orig_overlay.jpg")
        # crop图
        if len(item['images']) > 1:
            crop_path = item['images'][1]
            crop_image = Image.open(crop_path).convert("RGB")
            # zoomin mask token
            zoomin_mask_token = None
            if item['cap_problem'] and '<|mt_start|>' in item['cap_problem']:
                zoomin_mask_token = re.findall(r'<\|mt_start\|>.*?<\|mt_end\|>', item['cap_problem'])
                if len(zoomin_mask_token) > 1:
                    zoomin_mask_token = zoomin_mask_token[1]
                else:
                    zoomin_mask_token = None
            if zoomin_mask_token:
                codes = parse_mask_token(zoomin_mask_token, CODEBOOK_SIZE)
                if len(codes) == CODEBOOK_DEPTH:
                    original_codes = [codes[i] - i * CODEBOOK_SIZE for i in range(len(codes))]
                    crop_sam2_image = np.array(crop_image)
                    crop_sam2_image = vq_sam2_image_processor.apply_image(crop_sam2_image)
                    crop_sam2_pixel_values = torch.from_numpy(crop_sam2_image).permute(2, 0, 1).contiguous()
                    crop_sam2_pixel_values = crop_sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
                    quant_codes = torch.tensor(original_codes, dtype=torch.long).unsqueeze(0).cuda()
                    with torch.no_grad():
                        pred_masks = vq_sam2.forward_with_codes(crop_sam2_pixel_values, quant_codes)
                        pred_masks = pred_masks.detach()
                        pred_masks = torch.nn.functional.interpolate(pred_masks, size=(crop_image.height, crop_image.width), mode='bilinear')
                        pred_masks = pred_masks > 0.5
                        pred_masks = pred_masks[0, 0, :, :].cpu().numpy().astype(np.uint8)
                    overlayed_crop = overlay_mask_on_image(crop_image, pred_masks)
                    overlayed_crop.save(f"vis_mask_overlay/{idx}_crop_overlay.jpg")
            crop_image.save(f"vis_mask_overlay/{idx}_crop.jpg")

if __name__ == "__main__":
    main()
