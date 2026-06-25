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

import mmengine
from mmengine.dataset import BaseDataset

from xtuner.model.utils import guess_load_checkpoint

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize
from projects.vlm.vq_sam2.datasets.coco_panoseg_dataset import load_coco_panoptic_json, _get_coco_panoptic_meta, read_image
from projects.vlm.vq_sam2.datasets.coco_category import COCO_CATEGORIES


def sort_mask_indices(masks_t: torch.Tensor, mode: str = "ltr-ttb") -> np.ndarray:
    """
    根据实例的几何位置给出排序索引。
    Args:
        masks_t: [N, H, W] 的 torch.bool/uint8 张量（每个实例一个二值mask）
        mode:
            - "ltr-ttb": left-to-right, then top-to-bottom（先按x中心，再按y中心）
            - "ttb-ltr": top-to-bottom, then left-to-right（先按y中心，再按x中心）
            - "tlbr":    purely by top-left (y1, x1) 先y后x（行优先）
    Returns:
        order: numpy 数组，形状 [N]，是重排索引
    """
    # 利用bbox/中心点作为排序依据（更稳定、无需遍历像素）
    boxes = torchvision.ops.masks_to_boxes(masks_t)  # [N,4] (x1,y1,x2,y2)
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    xc = ((x1 + x2) * 0.5).cpu().numpy()
    yc = ((y1 + y2) * 0.5).cpu().numpy()
    y1n = y1.cpu().numpy()
    x1n = x1.cpu().numpy()

    if mode == "ltr-ttb":
        # 先x后y：主键x_center，次键y_center
        order = np.lexsort((yc, xc))
    elif mode == "ttb-ltr":
        # 先y后x：主键y_center，次键x_center
        order = np.lexsort((xc, yc))
    elif mode == "tlbr":
        # 以bbox左上角先y后x（更像“逐行”）
        order = np.lexsort((x1n, y1n))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return order

QUESTION_LIST = [
    "<image>\nSegment every instance that belongs to the following categories: {class_name}",
    "<image>\nLocate every instance that belongs to the following categories: {class_name}. Report segmentation masks in JSON format."
]


def clear_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    import gc
    gc.collect()


def main(task_id):
    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'
    
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    temp_save_root = "./temp_data_256x2_0927/cocopano/"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    dataset_name = "cocopano"

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

    coco_pano_meta = _get_coco_panoptic_meta()

    dataset = load_coco_panoptic_json('./data/coco/annotations/panoptic_train2017.json', './data/coco/train2017/', './data/coco/annotations/panoptic_train2017/', coco_pano_meta)

    n = len(dataset)
    chunk_size = (n+31) // 32
    _start_ = task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = n if _end_ > n else _end_

    for index in tqdm.tqdm(list(range(n))[_start_:_end_]):
        data_dict = dataset[index]

        image_file = data_dict['file_name']
        image_id = os.path.basename(image_file).split('.')[0]
        if os.path.exists(os.path.join(temp_save_root, f"{image_id}.json")):
            print("file exists............")
            continue

        isthing_dict = {item['name']: item['isthing'] for item in COCO_CATEGORIES}

        # decode masks
        pan_seg_gt = read_image(data_dict.pop('pan_seg_file_name'), "RGB")
        segments_info = data_dict['segments_info']

        from panopticapi.utils import rgb2id

        pan_seg_gt = rgb2id(pan_seg_gt)

        class_names = []
        masks = []
        for segment_info in segments_info:
            class_name = segment_info["category_name"]
            if not segment_info["iscrowd"]:
                class_names.append(class_name)
                masks.append(pan_seg_gt == segment_info["id"])
        if len(masks) == 0:
            print(len(masks) == 0)
            continue

        image = Image.open(image_file).convert('RGB')
        ori_width, ori_height = image.size

        categories_name_to_masks = {}
        for cls_name, mask in zip(class_names, masks):
            if cls_name not in categories_name_to_masks:
                categories_name_to_masks[cls_name] = []
            categories_name_to_masks[cls_name].append(mask)
        
        conversation = []
        answer = "```json\n[{mask_2d}]\n```"
        mask_2d_str = ''
        class_names = []
        for category_name, category_masks in categories_name_to_masks.items():
            sam2_image = np.array(image)
            sam2_image = sam2_image_processor.apply_image(sam2_image)
            sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
            sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

            masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in category_masks])
            valid_masks = masks.sum(-1).sum(-1) > 0
            masks = masks[valid_masks]

            if len(masks) == 0:
                print("len(masks) == 0!!!")
                continue

            try:
                order = sort_mask_indices(masks, mode="ltr-ttb")
            except:
                order = np.arange(masks.shape[0])
            
            masks = masks[torch.as_tensor(order, dtype=torch.long)]

            try:
                boxes = torchvision.ops.masks_to_boxes(masks)
            except:
                print("Error at boxes = torchvision.ops.masks_to_boxes(masks)")
                continue

            whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
            boxes = boxes / whwh
            boxes = boxes.to(vq_sam2.device)
            masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
            num_ins = len(masks)

            skip_this_one = False
            try:
                with torch.no_grad():
                    vq_sam2_output = vq_sam2(
                        sam2_pixel_values.repeat(num_ins, 1, 1, 1),
                        masks,
                        boxes,
                        reconstruct_mask=False,
                    )
                    quant_codes = vq_sam2_output.quant_codes
                    # pred_masks = vq_sam2_output.pred_masks
            except torch.OutOfMemoryError:
                print("num_ins is too large: ", num_ins, "; will be split into blocks (size 10)")
                NUM_BLOCKS = num_ins // 10
                if NUM_BLOCKS * 10 < num_ins:
                    NUM_BLOCKS += 1
                block_quant_codes = []
                # block_pred_masks = []
                for block_idx in range(NUM_BLOCKS):
                    start_idx = block_idx * 10
                    end_idx = min(start_idx + 10, num_ins)
                    try:
                        with torch.no_grad():
                            vq_sam2_output = vq_sam2(
                                sam2_pixel_values[start_idx:end_idx],
                                masks[start_idx:end_idx],
                                boxes[start_idx:end_idx],
                                reconstruct_mask=False,
                            )
                    except torch.OutOfMemoryError:
                        skip_this_one = True
                        break
                    block_quant_codes.append(vq_sam2_output.quant_codes)
                if skip_this_one:
                    print("skip this one!!!")
                    continue
                quant_codes = torch.cat(block_quant_codes, dim=0)

            except Exception as e:
                print("sam2_pixel_values.repeat(num_ins, 1, 1, 1).shape: ", sam2_pixel_values.repeat(num_ins, 1, 1, 1).shape)
                continue
            
            if len(quant_codes) == 0:
                continue

            quant_codes = quant_codes.cpu().numpy().astype(np.int32).tolist()
            remap_quant_codes = []
            for _quant_codes in quant_codes:
                _quant_codes = _quant_codes[0]
                remap_quant_codes.append([depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes)])
            quant_codes = remap_quant_codes

            if skip_this_one:
                print("skip this one")
                continue
           
            for _quant_codes_ in quant_codes:
                sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes_]) + MT_END_TOKEN
                item_str = "{\"mask_2d\": " + sam2_tokens + ", \"label\": \"" + category_name + "\"}"
                mask_2d_str += item_str + ",\n"
                if category_name not in class_names:
                    class_names.append(category_name)
      
        if mask_2d_str == '':
            print("mask_2d_str is None........")
            continue

        mask_2d_str = mask_2d_str[:-len(",\n")]
        answer = answer.format(mask_2d=mask_2d_str)

        category_name_str = ', '.join(class_names)
        question = random.choice(QUESTION_LIST).format(class_name=category_name_str)

        conversation.append({'from': 'human', 'value': question})
        conversation.append({'from': 'gpt', 'value': answer})
        
        ret_data_dict = {
            'image': image_file,
            'conversations': conversation,
        }

        # image_id = os.path.basename(image_file.split('.')[0])

        with open(os.path.join(temp_save_root, f"{image_id}.json"), 'w') as f:
            json.dump(ret_data_dict, f)

        clear_gpu_memory()


if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)