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


SEG_QUESTIONS = [
    "Locate the object described as: {caption}. Return its segmentation mask.",
    "Find \"{caption}\" in the image and output the pixel mask.",
    "Segment the region corresponding to \"{caption}\". Provide the mask only.",
    "Identify \"{caption}\" and give the segmentation mask of that object.",
    "From the image, extract the mask for \"{caption}\".",
    "Find and segment \"{caption}\"; respond with the mask.",
    "Return the segmentation mask of the target: {caption}.",
    "Segment the pixels that belong to \"{caption}\" and return the mask.",
]

ANSWER_LIST = [
    "It is {SEG}.",
    "Sure, {SEG}.",
    "Sure, it is {SEG}.",
    "Sure, the segmentation result is {SEG}.",
    "{SEG}.",
]

def main(task_id):

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    temp_save_root = "./temp_data/denseworld/"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)

    data_path = './data/denseworld_1m/final_annotations_quant_codes_x61'
    
    with open("./data/denseworld_task_files.json", 'r') as f:
        task_files = json.load(f)

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

        for obj_id, obj_ann in objects_anns.items():
            if os.path.exists(os.path.join(temp_save_root, f"{image_id}_{obj_id}.json")):
                continue
            description = obj_ann['caption']
            iou = obj_ann['iou']
            if iou < 0.8:
                continue
            quant_codes = obj_ann['quant_codes']

            sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes]) + MT_END_TOKEN
            
            question = random.choice(SEG_QUESTIONS).format(caption=description)
            question = "<image>\n" + question
            answer = random.choice(ANSWER_LIST).format(SEG=sam2_tokens)

            conversation = []
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': answer})

            ret_data_dict = {
                'image': image_path,
                'conversations': conversation,
            }

            with open(os.path.join(temp_save_root, f"{image_id}_{obj_id}.json"), 'w') as f:
                json.dump(ret_data_dict, f)
            break  # one case per image

if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)