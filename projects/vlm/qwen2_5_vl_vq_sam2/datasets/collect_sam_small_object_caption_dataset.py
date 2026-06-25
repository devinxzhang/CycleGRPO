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


# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag

np.random.seed(42)

def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image

def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image

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

def main(task_id):
    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    temp_save_root = "./temp_data_256x2_0927/sam_zoom_in/"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    dataset_name = "sam_zoom_in"

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

    coconut_dw = "./data/sam_obj_cap_gar"
    all_json_files = ['sa_000000_sub_001.parquet', 'sa_000000_sub_002.parquet', 'sa_000000_sub_005.parquet', 'sa_000000_sub_007.parquet', 'sa_000001_sub_004.parquet', 'sa_000001_sub_005.parquet', 'sa_000001_sub_006.parquet', 'sa_000001_sub_007.parquet', 'sa_000001_sub_008.parquet', 'sa_000001_sub_009.parquet', 'sa_000001_sub_010.parquet', 'sa_000002_sub_000.parquet', 'sa_000002_sub_008.parquet', 'sa_000002_sub_009.parquet', 'sa_000002_sub_010.parquet', 'sa_000003_sub_000.parquet', 'sa_000003_sub_001.parquet', 'sa_000003_sub_002.parquet', 'sa_000003_sub_003.parquet', 'sa_000003_sub_004.parquet', 'sa_000003_sub_005.parquet', 'sa_000003_sub_007.parquet']
    num_files = len(all_json_files)
    chunk_size = (num_files+7) // 8
    _start_ = task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = num_files if _end_ > num_files else _end_

    image_count = 0
    for parquet_file in all_json_files[_start_:_end_]:
        if not parquet_file.endswith('.parquet'):
            continue
        parquet_path = os.path.join(coconut_dw, parquet_file)
        parquet_f = pq.ParquetFile(parquet_path)
        data = parquet_f.read().to_pandas()

        for _, row in data.iterrows():
            # dict_keys(['mask', 'segments_info', 'image_info', 'image_caption', 'image'])
            image_count += 1
            print(f"============>>>{image_count} / {1024 * (_end_ - _start_)}")
            row_dict = row.to_dict()
            image = Image.open(BytesIO(row_dict['image']))
            ori_width, ori_height = image.size
            ann_list = json.loads(row_dict['ann'])
            global_img_bytes = to_bytes(image, format="PNG")
            global_image_dict = {"bytes": global_img_bytes, "path": None}

            for item in ann_list:
                segm = item['ann']['segmentation']
                caption = item['caption']

                sam2_image = np.array(image)
                sam2_image = sam2_image_processor.apply_image(sam2_image)
                sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
                sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

                binary_masks = decode_mask([segm], ori_height, ori_width)

                masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])
                boxes = torchvision.ops.masks_to_boxes(masks)
                x1, y1, x2, y2 = boxes.squeeze().cpu().numpy().tolist()
                boxes_w = boxes[:, 2] - boxes[:, 0]
                boxes_h = boxes[:, 3] - boxes[:, 1]
                boxes_area = boxes_h * boxes_w
                image_area = ori_height * ori_width
                boxes_occupied_ratio = boxes_area / image_area

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

                quant_codes = vq_sam2_output.quant_codes.detach().squeeze().detach().cpu().numpy().astype(np.int32).tolist()
                remap_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(quant_codes)]
                quant_codes = remap_quant_codes
                if boxes_occupied_ratio[0].item() > 0.2:
                    mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes]) + MT_END_TOKEN
                    question = random.choice(GLOBAL_QUESTION_LIST).format(SEG=mask_tokens_str)
                    question = "<image>\n" + question
                    conversation = []
                    conversation.append({'from': 'human', 'value': question})
                    conversation.append({'from': 'gpt', 'value': caption})

                    ret_data_dict = {
                        'image': [global_image_dict],
                        'conversations': conversation,
                    }
                    shard_items.append(ret_data_dict)
                    count += 1

                    if count % shard_size == 0:
                        shard_idx += 1
                        features = Features({
                            "image": Sequence(datasets.Image()),             # 列是一个 Image() 的序列
                            "conversations": Sequence(Value("string")),
                        })
                        ds = Dataset.from_list(shard_items, features=features)
                        out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-chunk{task_id}-{shard_idx:05d}")
                        ds.save_to_disk(out_path)
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
                question = random.choice(QUESTION_LIST).format(SEG=mask_tokens_str, ZOOM_IN_SEG=crop_mask_tokens_str)
                question = "<image>\n" + question

                conversation = []
                conversation.append({'from': 'human', 'value': question})
                conversation.append({'from': 'gpt', 'value': caption})

                # save crop image
                if resized_crop_image is not None:
                    img_bytes = to_bytes(resized_crop_image, format="PNG")
                    cropped_image_dict = {"bytes": img_bytes, "path": None}
                else:
                    img_bytes = to_bytes(cropped_image, format="PNG")
                    cropped_image_dict = {"bytes": img_bytes, "path": None}
                
                
                ret_data_dict = {
                    'image': [global_image_dict, cropped_image_dict],
                    'conversations': conversation,
                }

                shard_items.append(ret_data_dict)
                count += 1

                if count % shard_size == 0:
                    shard_idx += 1
                    features = Features({
                        "image": Sequence(datasets.Image()),             # 列是一个 Image() 的序列
                        "conversations": Sequence(Value("string")),
                    })
                    ds = Dataset.from_list(shard_items, features=features)
                    out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-chunk{task_id}-{shard_idx:05d}")
                    ds.save_to_disk(out_path)
                    shard_items.clear()
                    print(f"[SAVE] {out_path} ({count} items)", flush=True)

    # 收尾
    if shard_items:
        shard_idx += 1
        features = Features({
            "image": Sequence(datasets.Image()),             # 列是一个 Image() 的序列
            "conversations": Sequence(Value("string")),
        })
        ds = Dataset.from_list(shard_items, features=features)
        out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-chunk{task_id}-{shard_idx:05d}")
        ds.save_to_disk(out_path)
        shard_items.clear()
        print(f"[SAVE] {out_path} ({count} items)", flush=True)


if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)