import os
import sys
import collections
import os.path as osp
import random
import copy
from typing import Dict, List
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

from datasets import Dataset, DatasetDict, Sequence
from datasets import Image as ImageData

import mmengine
from mmengine.dataset import BaseDataset

from xtuner.model.utils import guess_load_checkpoint

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize
from projects.vlm.qwen2_5_vl_vq_sam2.datasets.coconut_meta import COCO_META
from projects.vlm.qwen2_5_vl_vq_sam2.datasets.gcg_process import glamm_granf_map_fn


def decode_mask(object_masks, ori_height, ori_width):
    binary_masks = []
    for object_mask in object_masks:
        if isinstance(object_mask, dict):
            if isinstance(object_mask["counts"], list):
                # convert to compressed RLE
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

def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union

def generate_data(conversation_items):
    for conv_item in conversation_items:
        image = Image.open(os.path.join(conv_item['image']), "r")
        yield {
            "images": [image],
            "problem": conv_item['conversations'][0]['value'],
            "answer": conv_item['conversations'][1]['value'],
            'masks': conv_item['segmentations'],
            'source': conv_item['source'],
        }

def extract_obj_tags(text):
    pattern = re.compile(r"<obj_\d+>")
    matches = pattern.findall(text)
    result = []
    for m in matches:
        result.append(m)
    return result

def extract_obj_ids(text):
    pattern = re.compile(r"<obj_(\d+)>")
    return [int(m) for m in pattern.findall(text)]

def main():
    random.seed(42)
    reasonseg_files = ["./data/vq_sam2_data_256x2_0927/mask_generation_reasonseg1326.json"]
    # reasonseg_files = ["./data/vq_sam2_data_256x2_0927/mask_generation_reasonseg1326.json","./data/vq_sam2_data_256x2_0927/mask_generation_reasonseg1326.json","./data/vq_sam2_data_256x2_0927/mask_generation_reasonseg1326.json","./data/vq_sam2_data_256x2_0927/mask_generation_reasonseg1326.json", "./data/vq_sam2_data_256x2_0927/mask_generation_reasonseg61k.json"]
    all_json_data = []
    for reasonseg_file in reasonseg_files:
        with open(reasonseg_file, 'r') as f:
            json_data = json.load(f)
            all_json_data.extend(json_data)

    random.shuffle(all_json_data)
    
    shard_items = []
    for data_dict in all_json_data:
        ret_data_dict = copy.deepcopy(data_dict)
        ret_data_dict.update({'source': 'reasonseg', 'segmentations': []})
        shard_items.append(ret_data_dict)
    
    trainset = Dataset.from_generator(generate_data, gen_kwargs={"conversation_items": shard_items})
    dataset = DatasetDict({"train": trainset}).cast_column("images", Sequence(ImageData()))
    # dataset.push_to_hub("zhouyik/grandf1k")
    dataset["train"].to_parquet(f"rl_dataset/reasonseg{len(shard_items)//1000}k_train.parquet")

if __name__ == "__main__":
    main()

    