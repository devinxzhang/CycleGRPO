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
import uuid
from datasets import load_from_disk, concatenate_datasets
import base64
import io
from io import BytesIO
from datasets import Dataset, Features, Sequence, Value
import datasets

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

GLOBAL_QUESTION_LIST = [
    "Given a detailed description of this region {SEG}.",
    "Provide a thorough description of this region {SEG}.",
    "Describe the region {SEG} in detail.",
    "Provide detailed information about this region {SEG}.",
    "Provide a detailed caption of this region {SEG}.",
    "{SEG}\nGive a detailed description of the masked region.",
    "{SEG}\nProvide a detailed description of the masked region.",
    "{SEG}\nDescribe the masked area comprehensively.",
    "{SEG}\nWhat are the details of the masked area?",
]

def to_bytes(img, format="PNG", pil_mode=None):
    # img 可以是 PIL 或 numpy
    if isinstance(img, np.ndarray):
        # 保证是 uint8, HWC
        if img.dtype != np.uint8:
            raise ValueError("Expect uint8 ndarray for image encoding.")
        img = Image.fromarray(img)
    if pil_mode is not None and img.mode != pil_mode:
        img = img.convert(pil_mode)
    bio = BytesIO()
    img.save(bio, format=format)  # "PNG" / "JPEG" / "WEBP"...
    return bio.getvalue()

def generate_unique_image_filename(directory="./data/gar_sam_zoom_in_images", prefix="gar", suffix="_global.jpg"):
    """
    生成一个唯一的图片文件名。
    """
    while True:
        # 1. 将 UUID 对象转换为字符串
        # 2. 移除连字符，得到一个纯粹的十六进制字符串
        # 3. 可以选择截断，但为了更高的唯一性，建议使用更长的部分或整个 UUID
        random_uuid_str = uuid.uuid4().hex # .hex 属性直接返回不带连字符的字符串
        # 如果确实需要更短的，可以截断，但请注意唯一性风险
        # random_uuid_str = uuid.uuid4().hex[:8] 
        image_file_name = f'{prefix}_{random_uuid_str}{suffix}'
        image_file_path = os.path.join(directory, image_file_name)
        if not os.path.exists(image_file_path):
            return image_file_path

def main(task_id):

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    temp_save_root = "./temp_data_256x2_0927/gar_multi_region_zoom_in/"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    dataset_name = "gar_multi_region_zoom_in"

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

    dataset = load_from_disk("./data/HaochenWang/Grasp-Any-Region-Dataset/Relation-Dataset")

    rows = len(dataset)
    chunk_size = (rows+7) // 8
    _start_ = task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = rows if _end_ > rows else _end_

    for row_id in tqdm.tqdm(list(range(rows))[_start_:_end_]):
        # if row_id < _start_ + 30000:
        #     continue
        data_dict = dataset[row_id]
        assert len(data_dict['conversations']) == 2
        from_human = data_dict['conversations'][0]['value']
        from_gpt = data_dict['conversations'][1]['value']

        image_path = data_dict['image']
        if isinstance(image_path, Image.Image):
            image = image_path
        elif isinstance(image_path, dict):
            image = Image.open(BytesIO(image_path["bytes"]))
        elif image_path.startswith("data:base64,"):
            base64_str = image_path.replace("data:base64,", "")
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        ori_width, ori_height = image.size

        image_file = generate_unique_image_filename(prefix='gar')
        image.save(image_file)
        
        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        binary_masks = decode_mask(data_dict['mask_rle'], ori_height, ori_width)
        if len(binary_masks) == 0:
            continue

        masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])
        boxes = torchvision.ops.masks_to_boxes(masks)
        whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
        boxes = boxes / whwh
        boxes = boxes.to(vq_sam2.device)
        masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
        
        with torch.no_grad():
            vq_sam2_output = vq_sam2(
                sam2_pixel_values.repeat(len(masks), 1, 1, 1),
                masks,
                boxes,
                reconstruct_mask=False,
            )

        quant_codes = vq_sam2_output.quant_codes.squeeze().detach().cpu().numpy().astype(np.int32).tolist()
        global_mask_token_str_dict = {}
        local_mask_token_str_dict = {}
        image_files = [image_file]
        if isinstance(quant_codes[0], list):
            remap_quant_codes = []
            for _quant_codes in quant_codes:
                remap_quant_codes.append([depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes)])
            quant_codes = remap_quant_codes

            for box_id in range(len(masks)):
                x1, y1, x2, y2 = boxes[box_id].squeeze().cpu().numpy().tolist()
                boxes_w = x2 - x1
                boxes_h = y2 - y1
                boxes_area = boxes_h * boxes_w
                boxes_occupied_ratio = boxes_area
                x1 = x1 * ori_width
                x2 = x2 * ori_width
                y1 = y1 * ori_height
                y2 = y2 * ori_height

                mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes[box_id]]) + MT_END_TOKEN
                global_mask_token_str_dict[f'<Prompt{box_id}>'] = mask_tokens_str
                if boxes_occupied_ratio > 0.2:
                    continue

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

                # resize the short edge
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
                elif crop_height == crop_width and crop_width < 280:
                    ratio = 280 / crop_height
                    new_height = 280
                    new_width = int(crop_width * ratio)
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

                cropped_masks = torch.stack([torch.from_numpy(np.ascontiguousarray(binary_masks[box_id].copy()[y1:y2, x1:x2]))])
                assert cropped_masks.shape[-2] == crop_height and cropped_masks.shape[-1] == crop_width

                if resized_crop_image is not None:
                    resized_crop_masks = torch.nn.functional.interpolate(cropped_masks.unsqueeze(0), size=(new_height, new_width), mode='bilinear')
                    resized_crop_masks = resized_crop_masks[0] > 0.5
                    cropped_masks = resized_crop_masks
                crop_height, crop_width = cropped_masks.shape[-2:]
                try:
                    cropped_boxes = torchvision.ops.masks_to_boxes(cropped_masks)
                except:
                    print(binary_masks[box_id].sum())
                    print(resized_crop_image is None)
                    exit(0)
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
                crop_mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in crop_quant_codes]) + MT_END_TOKEN
                local_mask_token_str_dict[f'<Prompt{box_id}>'] = crop_mask_tokens_str

                # save crop image
                if resized_crop_image is not None:
                    cropped_image_file = generate_unique_image_filename(prefix='gar', suffix='_local.jpg')
                    resized_crop_image.save(cropped_image_file)
                else:
                    cropped_image_file = generate_unique_image_filename(prefix='gar', suffix='_local.jpg')
                    cropped_image.save(cropped_image_file)
                image_files.append(cropped_image_file)
                
            for k, v in global_mask_token_str_dict.items():
                from_human = from_human.replace(k, k+v)
            zoom_in_str = ''
            for k, v in local_mask_token_str_dict.items():
                zoom_in_str += ' Zoom in ' + k + ': <image>, ' + v + '.'
            from_human = from_human + zoom_in_str
            question = "<image>\n" + from_human

            conversation = []
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': from_gpt})

            ret_data_dict = {
                'image': image_files,
                'conversations': conversation,
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
        else:
            remap_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(quant_codes)]
            quant_codes = remap_quant_codes

            x1, y1, x2, y2 = boxes.squeeze().cpu().numpy().tolist()
            boxes_w = x2 - x1
            boxes_h = y2 - y1
            boxes_area = boxes_h * boxes_w
            boxes_occupied_ratio = boxes_area
            x1 = x1 * ori_width
            x2 = x2 * ori_width
            y1 = y1 * ori_height
            y2 = y2 * ori_height

            if boxes_occupied_ratio > 0.2:
                mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes]) + MT_END_TOKEN
                # question = random.choice(GLOBAL_QUESTION_LIST).format(SEG=mask_tokens_str)
                from_human = from_human.replace('<Prompt0>', '<Prompt0>'+mask_tokens_str)
                question = "<image>\n" + from_human
                conversation = []
                conversation.append({'from': 'human', 'value': question})
                conversation.append({'from': 'gpt', 'value': from_gpt})
                ret_data_dict = {
                    'image': image_file,
                    'conversations': conversation,
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
                continue
            
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

            # resize the short edge
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
            elif crop_height == crop_width and crop_width < 280:
                ratio = 280 / crop_height
                new_height = 280
                new_width = int(crop_width * ratio)
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

            mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes]) + MT_END_TOKEN
            crop_mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in crop_quant_codes]) + MT_END_TOKEN
            
            from_human = from_human.replace('<Prompt0>', '<Prompt0>'+mask_tokens_str)
            from_human = from_human + f' Zoom in <Prompt0>: <image>, {crop_mask_tokens_str}.'
            question = "<image>\n" + from_human

            conversation = []
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': from_gpt})

            # save crop image
            if resized_crop_image is not None:
                cropped_image_file = generate_unique_image_filename(prefix='gar', suffix='_local.jpg')
                resized_crop_image.save(cropped_image_file)
            else:
                cropped_image_file = generate_unique_image_filename(prefix='gar', suffix='_local.jpg')
                cropped_image.save(cropped_image_file)
                
            ret_data_dict = {
                'image': [image_file, cropped_image_file],
                'conversations': conversation,
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
        print(f"[SAVE] {out_path} ({count} items)", flush=True)

if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)
