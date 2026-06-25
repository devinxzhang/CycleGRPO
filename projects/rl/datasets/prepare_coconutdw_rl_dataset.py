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
    GCG_QUESTIONS = [
        '<image>\nCould you please give me a detail description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
        '<image>\nCan you provide a detail description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.',
        '<image>\nPlease describe the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
        '<image>\nCould you give a detail explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
        '<image>\nCould you give me a detail explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
        '<image>\nCould you provide me with a detail analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
    ]
    
    coconutdw_ann_file = "./cold_start_data/detail_gcg_rl_source.json"

    with open(coconutdw_ann_file, 'r') as f:
        json_data = json.load(f)

    shard_items = []
    
    for index in tqdm.tqdm(list(range(len(json_data)))):
        data_dict = copy.deepcopy(json_data[index])

        image_path = data_dict['image']
        mask_annotation = data_dict['mask_annotation']
        image_caption = data_dict['image_caption']
        # obj_tags = extract_obj_tags(image_caption)
        obj_ids = extract_obj_ids(image_caption)

        segms = [mask_annotation[f"{obj_id}"]['rle'] for obj_id in obj_ids]

        for image_id, mask_anno in mask_annotation.items():
            mask_token = mask_anno['mask_token']
            image_caption = image_caption.replace(f"<obj_{image_id}>", mask_token)

        question = random.choice(GCG_QUESTIONS)

        conversation = []
        conversation.append({'from': 'human', 'value': question})
        conversation.append({'from': 'gpt', 'value': image_caption})

        ret_data_dict = {
            'image': image_path,
            'conversations': conversation,
            'segmentations': segms,
            'source': 'dw_gcg',
        }
        
        shard_items.append(ret_data_dict)
    
    trainset = Dataset.from_generator(generate_data, gen_kwargs={"conversation_items": shard_items})
    dataset = DatasetDict({"train": trainset}).cast_column("images", Sequence(ImageData()))
    # dataset.push_to_hub("zhouyik/grandf1k")
    dataset["train"].to_parquet("rl_dataset/dwgcg4k_train.parquet")

if __name__ == "__main__":
    main()

    