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
from projects.vlm.qwen2_5_vl_vq_sam2.datasets.coconut_meta import COCO_META


def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union

def clear_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    import gc
    gc.collect()


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
        if len(binary_masks) == 0:
            binary_masks.append(np.zeros((ori_height, ori_width), dtype=np.uint8))
        masks = np.stack(binary_masks, axis=0)
        return masks


def main(task_id):

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    temp_save_root = "./temp_data_256x2_0927/denseworld/"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)

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

    data_path = "./data/denseworld_1m/final_annotations"

    # denseworld_task_files = os.listdir(data_path)
    # with open('./data/denseworld_task_files.json', 'w') as f:
    #     json.dump(denseworld_task_files, f)
    # exit(0)

    with open('./data/denseworld_task_files.json', 'r') as f:
        task_files = json.load(f)

    chunk_idx = task_id
    n = len(task_files)
    chunk_size = (n+31) // 32
    start = chunk_idx * chunk_size
    end = min((chunk_idx + 1) * chunk_size, n)
    for anno_file in tqdm.tqdm(task_files[start:end]):
        if os.path.exists(os.path.join('./data/denseworld_1m/final_annotations_quant_codes_256x2', anno_file)):
            continue
        with open(os.path.join(data_path, anno_file), 'r') as f:
            json_data = json.load(f)
        image_name = json_data['image_name']
        image_path = None
        for image_root in ['./data/sam_full', './data/object365_full/images/train', './data/V3Det/images']:
            if os.path.exists(os.path.join(image_root, image_name)):
                image_path = os.path.join(image_root, image_name)
        if image_path is None:
            print(image_name, " is not found!!!")
            continue
        
        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        objects_anns = json_data['objects_anns']
        mask_ids = [k for k, _ in objects_anns.items()]
        segms = [item['segmentation'] for _, item in objects_anns.items()]
        masks = decode_mask(segms, ori_height, ori_width)
        masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
        if len(masks) == 0:
            print("len(masks) == 0!!!")
            continue

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        try:
            boxes = torchvision.ops.masks_to_boxes(masks)
        except Exception as e:
            continue
        
        whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
        boxes = boxes / whwh
        boxes = boxes.to(vq_sam2.device)
        masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
        num_ins = len(masks)

        if len(masks) > 50:
            print("too many objects, skip this one.::: ", len(masks))
            continue

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
            block_pred_masks = []
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
                # block_pred_masks.append(vq_sam2_output.pred_masks)
            if skip_this_one:
                continue
            quant_codes = torch.cat(block_quant_codes, dim=0)
            # pred_masks = torch.cat(block_pred_masks, dim=0)

            # print("num_ins is too large: ", num_ins)
            # exit(0)
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

        # # verify the quality of the quant_codes
        # # pred_masks = vq_sam2_output.pred_masks
        # try:
        #     pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height//4, ori_width//4), mode='bilinear')
        #     pred_masks = pred_masks > 0.5
        # except torch.OutOfMemoryError:
        #     NUM_BLOCKS = pred_masks.shape[0] // 10
        #     if NUM_BLOCKS * 10 < pred_masks.shape[0]:
        #         NUM_BLOCKS += 1
        #     resized_pred_masks = []
        #     for block_idx in range(NUM_BLOCKS):
        #         start_idx = block_idx * 10
        #         end_idx = start_idx + 10
        #         end_idx = pred_masks.shape[0] if end_idx > pred_masks.shape[0] else end_idx
        #         chunk_pred_masks = pred_masks[start_idx:end_idx]
        #         chunk_pred_masks = torch.nn.functional.interpolate(chunk_pred_masks, size=(ori_height//4, ori_width//4), mode='bilinear')
        #         chunk_pred_masks = chunk_pred_masks > 0.5
        #         resized_pred_masks.append(chunk_pred_masks)
        #     pred_masks = torch.cat(resized_pred_masks)
        # masks = torch.stack(masks, dim=0).to(torch.float16)
        # try:
        #     if min(masks.shape[-2], masks.shape[-1]) > 2048:
        #         print("masks toooooooooooo large: ", masks.shape)
        #         continue
        #     masks = torch.nn.functional.interpolate(masks, size=(ori_height//4, ori_width//4), mode='bilinear')
        #     masks = masks > 0.5
        # except torch.OutOfMemoryError:
        #     NUM_BLOCKS = masks.shape[0] // 10
        #     if NUM_BLOCKS * 10 < masks.shape[0]:
        #         NUM_BLOCKS += 1
        #     resized_masks = []
        #     for block_idx in range(NUM_BLOCKS):
        #         start_idx = block_idx * 10
        #         end_idx = start_idx + 10
        #         end_idx = masks.shape[0] if end_idx > masks.shape[0] else end_idx
        #         chunk_masks = masks[start_idx:end_idx]
        #         chunk_masks = torch.nn.functional.interpolate(chunk_masks, size=(ori_height//4, ori_width//4), mode='bilinear')
        #         chunk_masks = chunk_masks > 0.5
        #         resized_masks.append(chunk_masks)
        #     masks = torch.cat(resized_masks)
        # iou_list = []
        # for pred_mask, target_mask in zip(pred_masks, masks):
        #     iou = mask_iou(pred_mask, target_mask)
        #     iou_list.append(iou[0][0].item())
        
        ret_json_data = copy.deepcopy(json_data)

        # obj_name_2_iou = {obj_name: iou_v for obj_name, iou_v in zip(mask_ids, iou_list)}
        obj_name_2_quant_codes = {obj_name: _quant_codes_ for obj_name, _quant_codes_ in zip(mask_ids, quant_codes)}

        ret_objects_anns = {}
        for obj_name, item in objects_anns.items():
            item_copy = copy.deepcopy(item)
            item_copy.update({
                # 'iou': obj_name_2_iou[obj_name],
                'quant_codes': obj_name_2_quant_codes[obj_name],
            })
            ret_objects_anns[obj_name] = item_copy
        
        ret_json_data.update({'objects_anns': ret_objects_anns})

        with open(os.path.join('./data/denseworld_1m/final_annotations_quant_codes_256x2', anno_file), 'w') as f:
            json.dump(ret_json_data, f)


if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)


