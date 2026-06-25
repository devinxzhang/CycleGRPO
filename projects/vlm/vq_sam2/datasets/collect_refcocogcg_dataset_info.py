import os
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
from projects.vlm.qwen2_5_vl_vq_sam2.datasets.gcg_process import glamm_refcocog_map_fn


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

def encode_binary_mask(bin_mask_bool):
    # 跳过空 mask，避免 encode 的边界行为
    if not np.any(bin_mask_bool):
        return None
    # pycocotools 期望的是 Fortran 连续的 0/1 uint8，形状 HxW
    m = np.asfortranarray(bin_mask_bool.astype(np.uint8, copy=False))
    rle = mask_utils.encode(m)
    # 某些版本返回的是{'counts': bytes, 'size': [H, W]}
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def main():

    # all_items = []
    # for json_file in os.listdir('./any_other_seg_data'):
    #     with open('./any_other_seg_data', json_file) as f:
    #         json_data = json.load(f)
    #         all_items.extend(json_data)
    # with open('./data/sft_segdata_info_part1.json', 'w') as f:
    #     json.dump(all_items, f)
    # exit(0)

    # with open("", 'r') as f:
    #     json_data = json.load(f)
    # random_idx = random.randint(0, len(json_data)-1)
    # item = json_data[random_idx]
    # image_file = item['image_file']
    # segm = [item['segmentation']]

    # image = Image.open(image_file).convert('RGB')
    # ori_width, ori_height = image.size
    # masks = decode_mask([segm], ori_height, ori_width)
    # output_image = visualize(image, masks, [""]*len(masks))
    # output_image.save('any_other_seg_data.jpg')
    # exit(0)

    dataset_name = 'refcocogcg'

    temp_save_root = "./any_other_seg_data"
    os.makedirs(temp_save_root, exist_ok=True)

    refcocog_image_root = "./data/glamm_data/images/coco2014/train2014/"
    refcocog_ann_file = "./data/glamm_data/annotations/RefCOCOg_GCG_train.json"

    with open(refcocog_ann_file, 'r') as f:
        json_data = json.load(f)
    for idx in range(len(json_data)):
        _id = list(json_data[idx].keys())[0]
        json_data[idx] = json_data[idx][_id]
        json_data[idx].update({'id': _id})
    
    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    for index in tqdm.tqdm(range(len(json_data))):
        data_dict = copy.deepcopy(json_data[index])
        result_dict = glamm_refcocog_map_fn(data_dict)
        data_dict.update(result_dict)

        image_file = data_dict['file_name']
        if ".jpg" in image_file:
            image_id = image_file.split(".")[0]
        elif ".png" in image_file:
            image_id = image_file.split(".")[0]
        else:
            raise ValueError(f"Unsupported image format: {image_file}")
        
        image_path = os.path.join(refcocog_image_root, image_file)
        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        masks = decode_mask(data_dict['masks'], ori_height, ori_width)

        for bin_mask in masks:
            try:
                assert len(bin_mask.shape) ==2
                rle = encode_binary_mask(bin_mask.astype(np.bool))
                if rle is None:
                    # 空实例，跳过但记录
                    # print(f"[WARN] empty mask seg_id={seg_id} file={image_file}", flush=True)
                    continue

                shard_items.append({
                    "image_file": image_path,
                    "segmentation": rle,
                })
                count += 1

                if count % shard_size == 0:
                    shard_idx += 1
                    out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}.json")
                    with open(out_path, "w") as f:
                        json.dump(shard_items, f)
                    shard_items.clear()
                    print(f"[SAVE] {out_path} ({count} items)", flush=True)

            except Exception as e:
                # 如果 pycocotools 在 C 层崩溃，这里是抓不到的；但大多数数据问题能在这儿被捕到
                print(f"[ERROR] ...", flush=True)
                continue
    
     # 收尾
    if shard_items:
        shard_idx += 1
        out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}.json")
        with open(out_path, "w") as f:
            json.dump(shard_items, f)
        shard_items.clear()
        print(f"[SAVE] {out_path} (final, total={count})", flush=True) 

if __name__ == "__main__":
    main()

    