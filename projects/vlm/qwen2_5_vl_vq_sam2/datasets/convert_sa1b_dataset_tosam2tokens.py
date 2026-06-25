import sys
import torch
import torchvision
import copy
from PIL import Image
import numpy as np
import os
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize
from projects.vlm.vq_sam2.datasets import CoCoPanoSegValDataset, SA1BValDataset
from projects.vlm.vq_sam2.datasets.coco_category import COCO_CATEGORIES

from transformers import AutoProcessor

from xtuner.model.utils import guess_load_checkpoint

def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union


QUESTION_BOX2MASK_LIST = [
    "Please segment the object enclosed by the box {bbox_2d}",
    "Extract a mask for the object inside the bounding box {bbox_2d}",
    "Produce a pixel-level segmentation for the region within {bbox_2d}",
    "Generate a binary mask for the object located at {bbox_2d}",
    "Create the segmentation mask for the item bounded by {bbox_2d}",
    "Segment the foreground object inside the rectangle {bbox_2d}",
    "Return the mask corresponding to the object in {bbox_2d}",
    "Provide a per-pixel mask for the box area {bbox_2d}",
    "Derive the object mask from the ROI {bbox_2d}",
    "Isolate and segment the object within {bbox_2d}",
    "Produce a tight segmentation for the target inside {bbox_2d}",
    "Output a mask that covers the object enclosed by {bbox_2d}",
    "Compute the segmentation mask for the entity at {bbox_2d}",
    "Please return an instance mask for the box {bbox_2d}",
    "Segment only the object indicated by {bbox_2d}, excluding background.",
]

ANSWER_BOX2MASK_LIST = [
    "Sure, {SEG}.",
    "{SEG}.",
]


QUESTION_MASK2BOX_LIST = [
    "{mask_2d}\nPlease compute the bounding box that tightly encloses the given mask.",
    "{mask_2d}\nReturn the object's box from the mask as [x1, y1, x2, y2].",
    "Extract the bounding rectangle for the mask region {mask_2d}.",
    "{mask_2d}\nFind the mask's tight bounding box in absolute pixel coordinates.",
    "{mask_2d}\nFrom the mask, return its smallest enclosing axis-aligned rectangle.",
    "Provide the bounding box enclosing all connected components in the mask {mask_2d}."
]

ANSWER_MASK2BOX_LIST = [
    "Sure, {bbox_2d}",
    "{bbox_2d}.",
]



def main(task_id):

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    temp_save_root = "./temp_data/sa1b_box2mask_mask2box/"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    
    sam2_config = SAM2Config(
        cfg_path="sam2.1_hiera_l.yaml",
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
    )

    CODEBOOK_SIZE = 1024
    CODEBOOK_DEPTH = 4
    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=1024,
        codebook_depth=4,
        shared_codebook=False,
        latent_dim=256,
    )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    pretrained_pth = "work_dirs/vq_sam2_codebookx4depthx1024sizex256dimxunsharex1MT_x60_1_with_box_input/iter_17923.pth"
    pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

    pretrained_state_dict_new = {}
    for key in pretrained_state_dict.keys():
        new_key = copy.deepcopy(key)
        if key.startswith('hf_model.'):
            new_key = new_key[len('hf_model.'):]
        pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
    
    vq_sam2.load_state_dict(pretrained_state_dict_new)

    sam2_image_processor=dict(
        type=DirectResize,
        target_length=1024,
    )

    DATA_ROOT = ''
    sam_info_json = "./data/sam_info.json"

    val_dataset = SA1BValDataset(
        image_folder='',
        preprocessor=sam2_image_processor,
        multi_targets=False,
        repeats=1.0,
        fast_load=True,
        sam_info_json=sam_info_json,
        scan_record_folder='./left_sa1b_indices/vq_sam2_codebookx4depthx1024sizex256dimxunsharex1MT_datasetxsa1bx10xcoconutx10xentityx10xpixelwebx10/'
    )

    chunk_idx = task_id
    n = len(val_dataset)
    chunk_size = (n+15) // 16
    start = chunk_idx * chunk_size
    end = min((chunk_idx + 1) * chunk_size, n)
    indices_list = list(range(len(val_dataset)))[start:end]
    for idx in tqdm(indices_list):
        data = val_dataset[idx]
        image_file = data['image_file']
        image_name = os.path.basename(image_file).split('.jpg')[0]
        masks = data['masks']
        image = Image.open(image_file)
        width, height = image.size
        all_quant_codes = []

        turn_idx = 0
        max_turns = 3
        conversation = []
        for mask in masks:
            if turn_idx == max_turns:
                break

            val_item = val_dataset.prepare_mask_input(image_file, mask)
            pixel_values = val_item['pixel_values']
            masks = val_item['masks']
            ori_boxes = val_item['boxes'].to(vq_sam2.device)
            boxes = val_item['boxes'][0].cpu().numpy().tolist()
            assert len(boxes) == 4

            pixel_values = pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

            masks = [masks.to(vq_sam2.device)]

            with torch.no_grad():
                vq_sam2_output = vq_sam2(
                    pixel_values,
                    masks,
                    ori_boxes,
                )
            quant_codes = vq_sam2_output.quant_codes.squeeze().cpu().numpy().astype(np.int32).tolist()
            remap_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(quant_codes)]
            quant_codes = remap_quant_codes
            
            pred_masks = vq_sam2_output.pred_masks
            pred_masks = torch.nn.functional.interpolate(pred_masks, size=(height, width), mode='bilinear')
            pred_masks = pred_masks > 0.5
            pred_masks = pred_masks[0].cpu().numpy().astype(np.uint8)
        
            target_mask = masks[0].cpu().numpy().astype(np.uint8)

            iou = mask_iou(torch.from_numpy(target_mask), torch.from_numpy(pred_masks))
            if iou[0][0].item() < 0.8:
                continue

            mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes]) + MT_END_TOKEN

            if random.random() < 0.5:
                # box to mask
                question = random.choice(QUESTION_BOX2MASK_LIST)
                answer = random.choice(ANSWER_BOX2MASK_LIST).format(SEG=mask_tokens_str)
                if turn_idx == 0:
                    question = "<image>\n" + question
                conversation.append({'from': 'human', 'value': question, 'bbox_2d': boxes})
                conversation.append({'from': 'gpt', 'value': answer})
            else:
                question = random.choice(QUESTION_MASK2BOX_LIST).format(mask_2d=mask_tokens_str)
                answer = random.choice(ANSWER_MASK2BOX_LIST)
                if turn_idx == 0:
                    question = "<image>\n" + question
                conversation.append({'from': 'human', 'value': question})
                conversation.append({'from': 'gpt', 'value': answer, 'bbox_2d': boxes})
            turn_idx += 1

        ret_data_dict = {
            'image': image_file,
            'conversations': conversation,
        }

        with open(os.path.join(temp_save_root, f"{image_name}.json"), 'w') as f:
            json.dump(ret_data_dict, f)

if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)
