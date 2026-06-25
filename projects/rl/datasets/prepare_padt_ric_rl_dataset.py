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
    ric_ann_file = "./data/tokenmask_data_256x2_cot_format/mask_generation_padt_ric561k.json"
    with open(ric_ann_file, 'r') as f:
        json_data = json.load(f)
    
    shard_items = []
    image_ric_dict = {}
    for item in json_data:
        if item['image'] not in image_ric_dict:
            image_ric_dict[item['image']] = []
        image_ric_dict[item['image']].append(item)
    
    select_items = []
    for image_path, item_list in image_ric_dict.items():
        item = random.choice(item_list)
        select_items.append(item)

    random.shuffle(select_items)
    
    shard_items = []
    # for data_dict in select_items[:40000]:
    for data_dict in select_items[:200]:
        ret_data_dict = copy.deepcopy(data_dict)
        ret_data_dict.update({'source': 'padt_gcg', 'segmentations': []})
        shard_items.append(ret_data_dict)
    
    trainset = Dataset.from_generator(generate_data, gen_kwargs={"conversation_items": shard_items})
    dataset = DatasetDict({"train": trainset}).cast_column("images", Sequence(ImageData()))
    # dataset.push_to_hub("zhouyik/grandf1k")
    # dataset["train"].to_parquet("rl_dataset/padtgcg40k_train.parquet")
    dataset["train"].to_parquet("rl_dataset/padtgcg0k_train.parquet")

if __name__ == "__main__":
    main()

    