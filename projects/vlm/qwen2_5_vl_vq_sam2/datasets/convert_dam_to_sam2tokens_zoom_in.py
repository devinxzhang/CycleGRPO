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
import uuid

import mmengine
from mmengine.dataset import BaseDataset

from xtuner.model.utils import guess_load_checkpoint

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize
from projects.vlm.qwen2_5_vl_vq_sam2.datasets.coconut_meta import COCO_META


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


QUESTION_LIST = [
    "Given a detailed description of this region {SEG}. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "Provide a thorough description of this region {SEG}. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "Describe the region {SEG} in detail. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "Provide detailed information about this region {SEG}. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "Provide a detailed caption of this region {SEG}. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "{SEG}\nGive a detailed description of the masked region. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "{SEG}\nProvide a detailed description of the masked region. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "{SEG}\nDescribe the masked area comprehensively. Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
    "{SEG}\nWhat are the details of the masked area? Zoom in with the perspective as <image>, {ZOOM_IN_SEG}.",
]



def main(task_id):

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    temp_save_root = "./temp_data_256x2_0927/dam_zoom_in/"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    dataset_name = "dam_zoom_in"

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


    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0


    dam_data_root = "./data/dam_data"
    split_name_list = ["COCOStuff", "LVIS", "Mapillary", "OpenImages", "PACO", "SAM"]
    # split_name_list = ["SAM"]

    split_image_folder = {
        'COCOStuff': './data/',
        'LVIS': './data/',
        'Mapillary': './data/dam_data/Mapillary/images/',
        'OpenImages': './data/dam_data/OpenImages/images/',
        'PACO': './data/coco/',
        'SAM': './data/dam_data/SAM/images/'
    }

    for split_name in split_name_list:
        
        split_path = os.path.join(dam_data_root, split_name)
        annotation_path = os.path.join(split_path, "annotations.json")

        with open(annotation_path, 'r') as f:
            annotations_dict = json.load(f)

        rows = len(annotations_dict)
        chunk_size = (rows+7) // 8
        _start_ = task_id * chunk_size
        _end_ = _start_ + chunk_size
        _end_ = rows if _end_ > rows else _end_
        
        index = 0

        all_items = list(annotations_dict.items())
        subset = all_items[_start_:_end_]
        for _, data_dict in tqdm.tqdm(subset):
            for item in data_dict:
                caption = item['caption']
                if split_name == 'OpenImages':
                    image_id = item['img_id']
                    image_file = f"{image_id}.jpg"
                else:
                    image_file = os.path.basename(item['image'])
                    if '.jpg' in image_file:
                        image_id = image_file.split('.')[0]
                    elif '.png' in image_file:
                        image_id = image_file.split('.')[0]
                    else:
                        raise ValueError(f"Unsupported image format: {image_file}")
                ann_id = item['ann_id']
                
                if os.path.exists(os.path.join(temp_save_root, f"{image_id}_{ann_id}_{split_name}.json")):
                    print("file exists.............")
                    continue
                if split_name in ['COCOStuff', 'LVIS', 'PACO']:
                    image_path = os.path.join(split_image_folder[split_name], item['image'])
                elif split_name in ['Mapillary', 'OpenImages', 'SAM']:
                    image_path = os.path.join(split_image_folder[split_name], image_file)
                image = Image.open(image_path).convert('RGB')
                ori_width, ori_height = image.size

                sam2_image = np.array(image)
                sam2_image = sam2_image_processor.apply_image(sam2_image)
                sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
                sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

                binary_masks = decode_mask([item['mask_rle']], ori_height, ori_width)

                # output_image = visualize(image, masks, None)
                # output_image.save(f'test_dam_{image_id}_{ann_id}.jpg')
                # index += 1
                # if index < 100:
                #     continue
                # else:
                #     exit(0)

                masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])
                boxes = torchvision.ops.masks_to_boxes(masks)
                x1, y1, x2, y2 = boxes.squeeze().cpu().numpy().tolist()
                boxes_w = boxes[:, 2] - boxes[:, 0]
                boxes_h = boxes[:, 3] - boxes[:, 1]
                boxes_area = boxes_h * boxes_w
                image_area = ori_height * ori_width
                boxes_occupied_ratio = boxes_area / image_area

                if boxes_occupied_ratio[0].item() > 0.2:
                    continue

                whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
                boxes = boxes / whwh
                boxes = boxes.to(vq_sam2.device)
                masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
                
                with torch.no_grad():
                    vq_sam2_output = vq_sam2(
                        sam2_pixel_values,
                        masks,
                        boxes,
                        reconstruct_mask=False,
                    )

                quant_codes = vq_sam2_output.quant_codes.squeeze().detach().cpu().numpy().astype(np.int32).tolist()
                remap_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(quant_codes)]
                quant_codes = remap_quant_codes

                # visualize global mask
                # pred_masks = vq_sam2_output.pred_masks
                # pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
                # pred_masks = pred_masks > 0.5
                # pred_masks = pred_masks[0].cpu().numpy().astype(np.uint8)
                # target_mask = masks[0].cpu().numpy().astype(np.uint8)

                # global_recon_mask = visualize(image, pred_masks, ['recon'])
                # global_gt_mask = visualize(image, target_mask, ['gt'])

                # zoom in mask and image
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                if bbox_w < 140:
                    x1 = x1 - (140 - bbox_w) // 2
                    x2 = x2 + (140 - bbox_w) // 2
                if bbox_h < 140:
                    y1 = y1 - (140 - bbox_h) // 2
                    y2 = y2 + (140 - bbox_h) // 2
                x1 = int(max(0, x1))
                x2 = int(min(ori_width, x2))
                y1 = int(max(0, y1))
                y2 = int(min(ori_height, y2))
                

                cropped_image = image.crop((x1, y1, x2, y2))
                crop_width, crop_height = cropped_image.size

                # scale up the long edge
                
                if crop_width > crop_height and crop_width < 280:
                    ratio = 280 / crop_height
                    new_height = 280
                    new_width = int(crop_width * ratio)
                    resized_crop_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                elif crop_height > crop_width and crop_height < 280:
                    ratio = 280 / crop_width
                    new_width = 280
                    new_height = int(crop_height * ratio)
                    resized_crop_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                else:
                    new_height = new_width = None
                    resized_crop_image = None

                if resized_crop_image is None:
                    cropped_sam2_image = np.array(cropped_image)
                    cropped_sam2_image = sam2_image_processor.apply_image(cropped_sam2_image)
                    cropped_sam2_pixel_values = torch.from_numpy(cropped_sam2_image).permute(2, 0, 1).contiguous()
                    cropped_sam2_pixel_values = cropped_sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
                else:
                    cropped_sam2_image = np.array(resized_crop_image)
                    cropped_sam2_image = sam2_image_processor.apply_image(cropped_sam2_image)
                    cropped_sam2_pixel_values = torch.from_numpy(cropped_sam2_image).permute(2, 0, 1).contiguous()
                    cropped_sam2_pixel_values = cropped_sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

                cropped_masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy()[y1:y2, x1:x2])) for x in binary_masks])
                assert cropped_masks.shape[-2] == crop_height and cropped_masks.shape[-1] == crop_width

                if resized_crop_image is not None:
                    resized_crop_masks = torch.nn.functional.interpolate(cropped_masks.unsqueeze(0), size=(new_height, new_width), mode='bilinear')
                    resized_crop_masks = resized_crop_masks[0] > 0.5
                    cropped_masks = resized_crop_masks
                crop_height, crop_width = cropped_masks.shape[-2:]
                cropped_boxes = torchvision.ops.masks_to_boxes(cropped_masks)
                crop_whwh = torch.as_tensor([[crop_width, crop_height, crop_width, crop_height]])
                cropped_boxes = cropped_boxes / crop_whwh
                cropped_boxes = cropped_boxes.to(vq_sam2.device)
                cropped_masks = [m.unsqueeze(0).to(vq_sam2.device) for m in cropped_masks]

                with torch.no_grad():
                    cropped_vq_sam2_output = vq_sam2(
                        cropped_sam2_pixel_values,
                        cropped_masks,
                        cropped_boxes,
                        reconstruct_mask=True,
                    )
                
                crop_quant_codes = cropped_vq_sam2_output.quant_codes.squeeze().detach().cpu().numpy().astype(np.int32).tolist()
                remap_crop_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(crop_quant_codes)]
                crop_quant_codes = remap_crop_quant_codes

                # # visualize local mask
                # if resized_crop_image is not None:
                #     pred_crop_masks = cropped_vq_sam2_output.pred_masks
                #     pred_crop_masks = torch.nn.functional.interpolate(pred_crop_masks, size=(crop_height, crop_width), mode='bilinear')
                #     pred_crop_masks = pred_crop_masks > 0.5
                #     pred_crop_masks = pred_crop_masks[0].cpu().numpy().astype(np.uint8)
                #     local_recon_mask = visualize(resized_crop_image, pred_crop_masks, ['recon'])

                #     # global_recon_mask.save('dam_global_recon_mask.jpg')
                #     # global_gt_mask.save('dam_global_gt_mask.jpg')
                #     local_recon_mask.save('dam_local_resize_recon_mask.jpg')
                #     exit(0)
                # continue

                mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes]) + MT_END_TOKEN
                crop_mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in crop_quant_codes]) + MT_END_TOKEN
                question = random.choice(QUESTION_LIST).format(SEG=mask_tokens_str, ZOOM_IN_SEG=crop_mask_tokens_str)
                question = "<image>\n" + question

                conversation = []
                conversation.append({'from': 'human', 'value': question})
                conversation.append({'from': 'gpt', 'value': caption})

                # save crop image
                random_uuid = uuid.uuid4()
                crop_image_file = f"./data/dam_data/dam_crop_images/{image_id}_{random_uuid}.jpg"
                if resized_crop_image is not None:
                    resized_crop_image.save(crop_image_file)
                else:
                    cropped_image.save(crop_image_file)
                    
                ret_data_dict = {
                    'image': [image_path, crop_image_file],
                    'conversations': conversation,
                }

                shard_items.append(ret_data_dict)
                count += 1

                # print(ret_data_dict)
                # exit(0)

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
