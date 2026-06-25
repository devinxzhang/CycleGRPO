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
import hydra

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



def main():
    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    with hydra.initialize(version_base=None, config_path="../../../projects/transformers/vq_sam2/sam2/sam2_configs"):
        sam2_config = SAM2Config(
            cfg_path="sam2.1_hiera_l.yaml",
            ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
        )
        
        vq_sam2_config = VQ_SAM2Config(
            sam2_config=sam2_config,
            codebook_size=CODEBOOK_SIZE,
            codebook_depth=CODEBOOK_DEPTH,
            shared_codebook=False,
            latent_dim=256,
        )
    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    state = torch.load("./pretrained_weights/mask_tokenizer_256x2.pth", map_location="cpu")
    vq_sam2.load_state_dict(state)

    sam2_image_processor = DirectResize(1024)

    grandf_image_root = "./data/glamm_data/images/grandf/train/"
    grandf_ann_file = "./data/glamm_data/annotations/GranDf_HA_GCG_train.json"

    with open(grandf_ann_file, 'r') as f:
        json_data = json.load(f)

    shard_items = []
    
    for index in tqdm.tqdm(list(range(len(json_data)))):
        data_dict = copy.deepcopy(json_data[index])
        result_dict = glamm_granf_map_fn(data_dict)
        data_dict.update(result_dict)

        image_file = data_dict['file_name']
        image_path = os.path.join(grandf_image_root, image_file)
        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        segms = [x[0] for x in data_dict['masks']]
        masks = decode_mask(segms, ori_height, ori_width)
        masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
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

        question = data_dict['conversation'][0]['input']
        answer = data_dict['conversation'][0]['output']
        seg_pattern = f'\[SEG\]'
        seg_matches = list(re.finditer(seg_pattern, answer))
        if len(seg_matches) != len(sam2_tokens_list):
            continue
        result = answer
        for i in range(len(seg_matches)-1, -1, -1):
            match = seg_matches[i]
            start, end = match.span()
            result = result[:start] + sam2_tokens_list[i] + result[end:]
        result = result.replace('<p>', '<|object_ref_start|>').replace('</p>', '<|object_ref_end|>')

        assert question[-1] == '.'
        question = question[:-1] + ' and highlight the phrases.'

        conversation = []
        conversation.append({'from': 'human', 'value': question})
        conversation.append({'from': 'gpt', 'value': result})

        ret_data_dict = {
            'image': image_path,
            'conversations': conversation,
            'segmentations': segms,
            'source': 'glamm_gcg',
        }

        shard_items.append(ret_data_dict)
    
    trainset = Dataset.from_generator(generate_data, gen_kwargs={"conversation_items": shard_items})
    dataset = DatasetDict({"train": trainset}).cast_column("images", Sequence(ImageData()))
    # dataset.push_to_hub("zhouyik/grandf1k")
    dataset["train"].to_parquet("zhouyik_backup/grandf1k_train.parquet")

if __name__ == "__main__":
    main()

    