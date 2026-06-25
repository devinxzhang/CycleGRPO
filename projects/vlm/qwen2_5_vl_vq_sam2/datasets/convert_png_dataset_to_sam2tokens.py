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
from skimage import io
import re

import mmengine
from mmengine.dataset import BaseDataset

from xtuner.model.utils import guess_load_checkpoint

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize

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

QUESTIONS = [
    "<image>\nDo panoptic narrative grounding for this image.",
    "<image>\nDescribe this image and provide corresponding masks for mentioned nouns."
]



def main(task_id):
    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    temp_save_root = "./temp_data_256x2_0927/png/"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)

    dataset_name = 'png'

    sam2_config = SAM2Config(
        cfg_path="sam2.1_hiera_l.yaml",
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
    )

    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=256,
    )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    pretrained_pth = "pretrained_weights/iter_129437_256x2.pth"
    pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

    pretrained_state_dict_new = {}
    for key in pretrained_state_dict.keys():
        new_key = copy.deepcopy(key)
        if key.startswith('hf_model.'):
            new_key = new_key[len('hf_model.'):]
        pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
    
    vq_sam2.load_state_dict(pretrained_state_dict_new)

    sam2_image_processor = DirectResize(1024)


    png_anno_file = "./data/png_data/png_coco_train2017.json"
    panoseg_anno_json_file = "./data/coco/annotations/panoptic_train2017.json"
    panoseg_seg_file_dir = "./data/coco/annotations/panoptic_train2017"

    with open(panoseg_anno_json_file, 'r') as f:
        panoseg_json_data = json.load(f)
    image_dict = {item['id']: item['file_name'] for item in panoseg_json_data['images']}

    with open(png_anno_file, 'r') as f:
        json_data = json.load(f)
    
    chunk_size = (len(json_data)+7) // 8
    _start_ = task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = len(json_data) if _end_ > len(json_data) else _end_

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    for index in tqdm.tqdm(list(range(len(json_data)))[_start_:_end_]):
        data_dict = json_data[index]

        image_id = int(data_dict['image_id'])
        png_segments = data_dict['segments']

        image_file = image_dict[image_id]
        image_path = os.path.join("./data/coco/train2017", image_file)

        panoptic_segm = io.imread(
            osp.join(
                panoseg_seg_file_dir,
                "{:012d}.png".format(image_id),
            )
        )
        panoptic_segm = (
            panoptic_segm[:, :, 0]
            + panoptic_segm[:, :, 1] * 256
            + panoptic_segm[:, :, 2] * 256 ** 2
        )

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        png_caption = ''
        for png_segment in png_segments:
            utterance = png_segment['utterance']
            segment_ids = png_segment['segment_ids']
            
            if utterance == '.':
                png_caption = png_caption[:-1] + f"{utterance} "
            else:
                png_caption += f"{utterance} "

            if len(segment_ids) == 0:
                continue
            
            binary_masks = []
            for seg_id in segment_ids:
                binary_mask = panoptic_segm == int(seg_id)
                assert np.sum(binary_mask) > 0
                binary_masks.append(binary_mask)
            
            masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])

            boxes = torchvision.ops.masks_to_boxes(masks)
            whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
            boxes = boxes / whwh
            boxes = boxes.to(vq_sam2.device)
            masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
            num_ins = len(masks)

            with torch.no_grad():
                vq_sam2_output = vq_sam2(
                    sam2_pixel_values.repeat(num_ins, 1, 1, 1),
                    masks,
                    boxes,
                    reconstruct_mask=False,
                )
                quant_codes = vq_sam2_output.quant_codes.detach()
            
            quant_codes = quant_codes.cpu().numpy().astype(np.int32).tolist()
            remap_quant_codes = []
            for _quant_codes in quant_codes:
                _quant_codes = _quant_codes[0]
                remap_quant_codes.append([depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes)])
            quant_codes = remap_quant_codes
            
            sam2_tokens_list = []
            for _quant_codes in quant_codes:
                sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes]) + MT_END_TOKEN
                sam2_tokens_list.append(sam2_tokens)
            sam2_tokens_list_str = '(' + ', '.join(sam2_tokens_list) + ')'
            png_caption += f"{sam2_tokens_list_str} "
        
        png_caption = png_caption.strip()
        
        question = random.choice(QUESTIONS)
        conversations = []
        conversations.append({'from': 'human', 'value': question})
        conversations.append({'from': 'gpt', 'value': png_caption})

        ret_data_dict = {
            'image': image_path,
            'conversations': conversations,
        }

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