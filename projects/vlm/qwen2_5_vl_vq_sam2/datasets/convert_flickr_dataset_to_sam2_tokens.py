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

import mmengine
from mmengine.dataset import BaseDataset

from xtuner.model.utils import guess_load_checkpoint

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize
from projects.vlm.qwen2_5_vl_vq_sam2.datasets.coconut_meta import COCO_META
from projects.vlm.qwen2_5_vl_vq_sam2.datasets.gcg_process import glamm_flickr_map_fn


from types import MethodType
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.data.detection_utils import read_image, _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.utils.visualizer import GenericMask
import matplotlib.colors as mplc
def draw_instance_predictions_cache(self, labels, np_masks, jittering: bool = True):
    """
    Draw instance-level prediction results on an image.
    Args:
        predictions (Instances): the output of an instance detection/segmentation
            model. Following fields will be used to draw:
            "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        jittering: if True, in color mode SEGMENTATION, randomly jitter the colors per class
            to distinguish instances from the same class
    Returns:
        output (VisImage): image object with visualizations.
    """
    boxes = None
    scores = None
    classes = None
    keypoints = None

    masks = [GenericMask(x, self.output.height, self.output.width) for x in np_masks]

    if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
        colors = (
            [self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes]
            if jittering
            else [
                tuple(mplc.to_rgb([x / 255 for x in self.metadata.thing_colors[c]]))
                for c in classes
            ]
        )

        alpha = 0.8
    else:
        colors = None
        alpha = 0.5
    
    alpha = 0.4

    self.overlay_instances(
        masks=masks,
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=colors,
        alpha=alpha,
    )
    return self.output


def visualize(input_image, cat_masks, tags):
    if tags is None:
        left_tags = [f'{i}' for i in range(len(cat_masks))]
    else:
        left_tags = tags

    unique_tags = list(set(left_tags))
    text_prompt = ','.join(unique_tags)
    metadata = MetadataCatalog.get("__unused_ape_" + text_prompt)
    metadata.thing_classes = unique_tags
    metadata.stuff_classes = unique_tags

    result_masks = cat_masks
    input_image = _apply_exif_orientation(input_image)
    input_image = convert_PIL_to_numpy(input_image, "BGR")
    visualizer = Visualizer(input_image[:, :, ::-1], metadata, instance_mode=ColorMode.IMAGE)
    visualizer.draw_instance_predictions = MethodType(draw_instance_predictions_cache, visualizer)
    vis_output = visualizer.draw_instance_predictions(labels=left_tags, np_masks=result_masks)
    output_image = vis_output.get_image()
    output_image = Image.fromarray(output_image)

    return output_image


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



def main(task_id):
    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    temp_save_root = "./temp_data_256x2_0927/gcg/"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)

    dataset_name = "flickr"

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

    flickr_image_root = "./data/glamm_data/images/flickr30k/Flickr30K/"
    flickr_ann_file = "./data/glamm_data/annotations/flickr_mergedGT_GCG_train.json"

    
    # load the json file
    def filter_images(data_infos, min_size):
        return [i for i, info in enumerate(data_infos) if min(info['width'], info['height']) >= min_size]
    from pycocotools.coco import COCO
    coco = COCO(flickr_ann_file)
    image_ids = coco.getImgIds()
    data_infos = []
    total_ann_ids = []
    removed_img_count = 0
    for img_id in image_ids:
        info = coco.loadImgs([img_id])[0]
        if len(info['caption'].split(' ')) < 3:
            removed_img_count += 1
            continue
        info['filename'] = info['file_name'].split('_')[-1]
        info['height'] = int(info['height'])
        info['width'] = int(info['width'])
        data_infos.append(info)
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        total_ann_ids.extend(ann_ids)
    assert len(set(total_ann_ids)) == len(total_ann_ids), f"Non-unique annotation IDs in '{flickr_ann_file}'!"
    print(f'Removed {removed_img_count} images.')
    data_infos = [data_infos[i] for i in filter_images(data_infos, min_size=32)]
    # obtain_annotations
    for data_info in data_infos:
        ann_ids = coco.getAnnIds(imgIds=data_info['id'])
        ann_info = coco.loadAnns(ann_ids)
        data_info.update({'ann_info': ann_info})

    chunk_size = (len(data_infos)+7) // 8
    _start_ = task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = len(data_infos) if _end_ > len(data_infos) else _end_

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    for index in tqdm.tqdm(list(range(len(data_infos)))[_start_:_end_]):
        data_dict = copy.deepcopy(data_infos[index])
        result_dict = glamm_flickr_map_fn(data_dict)
        data_dict.update(result_dict)

        image_file = os.path.basename(data_dict['file_name'])
        if ".jpg" in image_file:
            image_id = image_file.split(".")[0]
        elif ".png" in image_file:
            image_id = image_file.split(".")[0]
        else:
            raise ValueError(f"Unsupported image format: {image_file}")
        
        if os.path.exists(os.path.join(temp_save_root, f"{image_id}_{index}_flickr.json")):
            print("file exits........")
            continue
        
        image_path = os.path.join(flickr_image_root, data_dict['file_name'])
        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        masks = decode_mask(data_dict['masks'], ori_height, ori_width)

        # output_image = visualize(image, masks, None)
        # output_image.save(f"test_flickr_{image_id}.jpg")
        # if index < 100:
        #     continue
        # else:
        #     exit(0)

        masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])

        try:
            boxes = torchvision.ops.masks_to_boxes(masks)
        except Exception as e:
            continue
        
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
            quant_codes = vq_sam2_output.quant_codes

        skip_this_one = False
        # try:
        #     with torch.no_grad():
        #         vq_sam2_output = vq_sam2(
        #             sam2_pixel_values.repeat(num_ins, 1, 1, 1),
        #             masks,
        #             boxes,
        #             multimask_output=False,
        #         )
        #         quant_codes = vq_sam2_output.quant_codes
        #         # pred_masks = vq_sam2_output.pred_masks
        # except torch.OutOfMemoryError:
        #     print("num_ins is too large: ", num_ins, "; will be split into blocks (size 10)")
        #     NUM_BLOCKS = num_ins // 10
        #     if NUM_BLOCKS * 10 < num_ins:
        #         NUM_BLOCKS += 1
        #     block_quant_codes = []
        #     block_pred_masks = []
        #     for block_idx in range(NUM_BLOCKS):
        #         start_idx = block_idx * 10
        #         end_idx = min(start_idx + 10, num_ins)
        #         try:
        #             with torch.no_grad():
        #                 vq_sam2_output = vq_sam2(
        #                     sam2_pixel_values[start_idx:end_idx],
        #                     masks[start_idx:end_idx],
        #                     boxes[start_idx:end_idx],
        #                     multimask_output=False,
        #                 )
        #         except torch.OutOfMemoryError:
        #             skip_this_one = True
        #             break
        #         block_quant_codes.append(vq_sam2_output.quant_codes)
        #         # block_pred_masks.append(vq_sam2_output.pred_masks)
        #     if skip_this_one:
        #         continue
        #     quant_codes = torch.cat(block_quant_codes, dim=0)
        #     # pred_masks = torch.cat(block_pred_masks, dim=0)

        #     # print("num_ins is too large: ", num_ins)
        #     # exit(0)
        # except Exception as e:
        #     print("sam2_pixel_values.repeat(num_ins, 1, 1, 1).shape: ", sam2_pixel_values.repeat(num_ins, 1, 1, 1).shape)
        #     continue
        
        # if len(quant_codes) == 0:
        #     continue

        quant_codes = quant_codes.cpu().numpy().astype(np.int32).tolist()
        remap_quant_codes = []
        for _quant_codes in quant_codes:
            _quant_codes = _quant_codes[0]
            remap_quant_codes.append([depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes)])
        quant_codes = remap_quant_codes
        
        # # verify the quality of the quant_codes
        # pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
        # pred_masks = pred_masks > 0.5
        # skip_this_one = False
        # for pred_mask, target_mask in zip(pred_masks, masks):
        #     iou = mask_iou(pred_mask, target_mask)
        #     if iou[0][0].item() < 0.5:
        #         skip_this_one = True
        #         break
        
        # if skip_this_one:
        #     print("skip this one============")
        #     continue

        sam2_tokens_list = []
        for _quant_codes in quant_codes:
            sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes]) + MT_END_TOKEN
            sam2_tokens_list.append(sam2_tokens)

        question = data_dict['conversation'][0]['input']
        answer = data_dict['conversation'][0]['output']
        # answer:
        # '<p> a pink, oval shaped bowl filled with brown rice and veggies </p> [SEG] is visible. there is also <p> a container with vegetables and a slice of linterleaved segmentation masks for the corresponding parme in it </p> [SEG].'
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

        conversation = []
        conversation.append({'from': 'human', 'value': question})
        conversation.append({'from': 'gpt', 'value': result})

        ret_data_dict = {
            'image': image_path,
            'conversations': conversation,
        }

        shard_items.append(ret_data_dict)
        count += 1

        # with open(os.path.join(temp_save_root, f"{image_id}_{index}_flickr.json"), 'w') as f:
        #     json.dump(ret_data_dict, f)

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

    