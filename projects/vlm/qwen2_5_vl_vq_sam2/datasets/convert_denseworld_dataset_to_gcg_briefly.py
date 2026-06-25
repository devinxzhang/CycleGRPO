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


GCG_QUESTIONS = [
    '<image>\nCould you please give me a description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer. Do not use \"<|object_ref_start|>...<|object_ref_end|>\" tags.',
    '<image>\nCan you provide a description of the this image? Please output with interleaved segmentation masks for the corresponding phrases. Do not use \"<|object_ref_start|>...<|object_ref_end|>\" tags.',
    '<image>\nPlease describe the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer. Do not use \"<|object_ref_start|>...<|object_ref_end|>\" tags.',
    '<image>\nCould you give a explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases. Do not use \"<|object_ref_start|>...<|object_ref_end|>\" tags.',
    '<image>\nCould you give me an explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases. Do not use \"<|object_ref_start|>...<|object_ref_end|>\" tags.',
    '<image>\nCould you provide me with a analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer. Do not use \"<|object_ref_start|>...<|object_ref_end|>\" tags.',
]

def main(task_id):

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    temp_save_root = "./temp_data_256x2_0927/denseworld/"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)

    data_path = './data/denseworld_1m/final_annotations_quant_codes_256x2'

    with open("./data/denseworld_task_files.json", 'r') as f:
        task_files = json.load(f)

    dataset_name = 'dw'

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    chunk_idx = task_id
    n = len(task_files)
    chunk_size = (n+7) // 8
    start = chunk_idx * chunk_size
    end = min((chunk_idx + 1) * chunk_size, n)
    for anno_file in tqdm.tqdm(task_files[start:end]):
        if not os.path.exists(os.path.join(data_path, anno_file)):
            continue
        with open(os.path.join(data_path, anno_file), 'r') as f:
            json_data = json.load(f)
        
        image_name = json_data['image_name']
        image_id = os.path.basename(image_name).split('.')[0]
        image_path = None
        for image_root in ['./data/sam_full', './data/object365_full/images/train', './data/V3Det/images']:
            if os.path.exists(os.path.join(image_root, image_name)):
                image_path = os.path.join(image_root, image_name)
        if image_path is None:
            print(image_name, " is not found!!!")
            continue

        objects_anns = json_data['objects_anns']

        id_2_codes = {}

        skip_this_one = False
        for obj_id, obj_ann in objects_anns.items():
            # iou = obj_ann['iou']
            # if iou < 0.4:
            #     skip_this_one = True
            #     break

            quant_codes = obj_ann['quant_codes']

            sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes]) + MT_END_TOKEN

            id_2_codes[obj_id] = sam2_tokens
        
        # if skip_this_one:
        #     continue

        grounded_caption = json_data['grounded_caption']
        for obj_id, codes in id_2_codes.items():
            grounded_caption = grounded_caption.replace(obj_id, codes)
        
        conversation = []
        conversation.append({'from': 'human', 'value': random.choice(GCG_QUESTIONS)})
        conversation.append({'from': 'gpt', 'value': grounded_caption})

        ret_data_dict = {
            'image': image_path,
            'conversations': conversation,
        }

        # with open(os.path.join(temp_save_root, f"{image_id}.json"), 'w') as f:
        #     json.dump(ret_data_dict, f)

        shard_items.append(ret_data_dict)
        count += 1

        if count % shard_size == 0:
            shard_idx += 1
            out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-chunk{task_id}-{shard_idx:05d}.json")
            with open(out_path, "w") as f:
                json.dump(shard_items, f)
            shard_items.clear()
            print(f"[SAVE] {out_path} ({count} items)", flush=True)
    
    # 收尾
    if shard_items:
        shard_idx += 1
        out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-chunk{task_id}-{shard_idx:05d}.json")
        with open(out_path, "w") as f:
            json.dump(shard_items, f)
        shard_items.clear()
        print(f"[SAVE] {out_path} (final, total={count})", flush=True) 

if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)