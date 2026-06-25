import argparse
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
import uuid
import hydra
import base64
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageColor
from datasets import Dataset, DatasetDict, Sequence
from datasets import Image as ImageData
from datasets import load_dataset

import mmengine
from mmengine.dataset import BaseDataset
import pycocotools.mask as mask_util
from xtuner.model.utils import guess_load_checkpoint

from transformers import Sam2Processor, Sam2Model
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from torchvision.transforms.functional import resize, to_pil_image


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

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

def apply_overlay_and_box(base_image, mask_np, box, color_hex="#00FF00", alpha=150):
    """
    将 Mask 以半透明叠加的方式贴到原图，并画上 BBox。
    不使用 plt，纯用 PIL 和 Numpy 实现。
    """
    # 1. 准备底图 (转换为 RGBA 以便进行 alpha 合成)
    base_rgba = base_image.convert("RGBA")
    H, W = base_rgba.size[1], base_rgba.size[0]

    # ensure mask is boolean
    mask_bool = mask_np > 0

    # 2. 创建 Mask 叠加层
    # 创建一个全透明的黑色图像 (H, W, 4)
    overlay_np = np.zeros((H, W, 4), dtype=np.uint8)

    # 解析颜色 (例如 #00FF00 -> (0, 255, 0))
    rgb_color = ImageColor.getrgb(color_hex)

    # 在 Mask 为 True 的区域填充颜色和透明度
    # 这里的 alpha 控制 Mask 的透明度 (0-255, 越大越不透明)
    overlay_np[mask_bool, 0] = rgb_color[0] # R
    overlay_np[mask_bool, 1] = rgb_color[1] # G
    overlay_np[mask_bool, 2] = rgb_color[2] # B
    overlay_np[mask_bool, 3] = alpha        # A

    # 将 numpy 数组转回 PIL 图像
    overlay_pil = Image.fromarray(overlay_np, "RGBA")

    # 3. 合成图像 (将半透明 Mask 盖在底图上)
    # alpha_composite 要求两张图都是 RGBA
    combined_img = Image.alpha_composite(base_rgba, overlay_pil)

    # 4. 绘制 Bounding Box
    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(combined_img)
    # 绘制矩形框，outline 是边框颜色，width 是线宽
    # input_box 格式直接就是 [x0, y0, x1, y1]，可以直接用
    draw.rectangle(box, outline="red", width=5)

    # 5. 转回 RGB (方便保存为 jpg 等格式)
    final_result_rgb = combined_img.convert("RGB")

    return final_result_rgb

def concat_side_by_side(img_left, img_right):
    """
    将两张 PIL 图片横向并排拼接
    """
    w1, h1 = img_left.size
    w2, h2 = img_right.size
    
    # 1. 创建新画布：宽度相加，高度取最大值
    new_w = w1 + w2
    new_h = max(h1, h2)
    # 创建RGB模式的空白图
    new_img = Image.new('RGB', (new_w, new_h), color=(255, 255, 255)) 
    
    # 2. 粘贴图片
    new_img.paste(img_left, (0, 0))      # 第一张贴在左边 (0,0)
    new_img.paste(img_right, (w1, 0))    # 第二张贴在右边 (w1, 0)
    
    return new_img

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('--model_path', 
        default="Qwen/Qwen3-VL-4B-SAMTok",
        help='hf model path.')
    parser.add_argument(
        '--vq_sam2_path',
        default="Qwen/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--dataset',
        default='data/GroundingME',
        help='Specify a ref dataset')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "facebook/sam2.1-hiera-large"
    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id).to(device)
    model.eval()


    # build vq-sam2 model
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    with hydra.initialize(version_base=None, config_path="../../../projects/transformers/vq_sam2/sam2/sam2_configs"):
        sam2_config = SAM2Config(
            cfg_path="sam2.1_hiera_l.yaml",
            ckpt_path="Qwen/sam2.1_hiera_large.pt",
        )
        
        vq_sam2_config = VQ_SAM2Config(
            sam2_config=sam2_config,
            codebook_size=CODEBOOK_SIZE,
            codebook_depth=CODEBOOK_DEPTH,
            shared_codebook=False,
            latent_dim=256,
        )
    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()
    state = torch.load(args.vq_sam2_path, map_location="cpu")
    vq_sam2.load_state_dict(state)
    vq_sam2_image_processor = DirectResize(1024)

    PROMPT_TEMPLATE = """<image>\nAll spatial relationships are defined from the viewer's perspective, where 'front' means closer to the viewer and 'back' means farther from the viewer. Please provide the segmentation mask of the object the following statement describes:
    {description}
    Ensure that all details mentioned about the object are accurate. Provide at most one segmentation mask. If a matching object is found, provide its segmentation mask in the format `<|mt_start|><|mt_xxxx|><|mt_xxxx|><|mt_end|>`. If no matching object is found, output null."""

    data_list = []
    # Load the dataset
    dataset = load_dataset("data/GroundingME", split="test")
    # Access a sample
    for img_idx, sample in enumerate(tqdm(dataset)):
        image = sample["image"]
        description = sample["description"]
        bbox = sample["bbox"]  # Ground truth [x1, y1, x2, y2]
        if bbox is not None:
            category = sample["subtask_l1"]  # Discriminative/Spatial/Limited/Rejection
            assert category in ["Discriminative", "Spatial", "Limited"]
            width = sample["width"]
            height = sample["height"]

            item = {}

            ori_width, ori_height = image.size

            inputs = processor(
                images=image,
                input_boxes=[[bbox]],   # 注意：是 List[List[box]]
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.image_processor.post_process_masks(
                outputs.pred_masks,
                original_sizes=inputs["original_sizes"],
                reshaped_input_sizes=inputs["reshaped_input_sizes"]
            )
            masks = results[0]  # Shape: [num_boxes, num_masks_per_box, H, W]
            scores = outputs.iou_scores
            best_mask_idx = torch.argmax(scores[0, 0]) # 找到这一组里分数最高的 mask 索引
            final_mask = masks[0, best_mask_idx].cpu().numpy().astype(np.uint8) # 取出 mask 并转为 numpy

            ############### start VQ_SAM2 ################
            MT_START_TOKEN = '<|mt_start|>'
            MT_END_TOKEN = '<|mt_end|>'
            MT_CONTEXT_TOKEN = '<|mt_{}|>'

            ori_width, ori_height = image.size

            sam2_image = np.array(image)
            sam2_image = vq_sam2_image_processor.apply_image(sam2_image)
            sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
            sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

            binary_masks = [final_mask]

            masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])

            boxes = torchvision.ops.masks_to_boxes(masks)
            x1, y1, x2, y2 = boxes.squeeze().cpu().numpy().tolist()
            boxes_w = x2 - x1
            boxes_h = y2 - y1
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

            quant_codes = vq_sam2_output.quant_codes.squeeze().cpu().numpy().astype(np.int32).tolist()
            remap_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(quant_codes)]
            quant_codes = remap_quant_codes
            global_mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes]) + MT_END_TOKEN

            if boxes_occupied_ratio < 0.3:
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
                    # continue

                if resized_crop_image is None:
                    cropped_sam2_image = np.array(cropped_image)
                    cropped_sam2_image = vq_sam2_image_processor.apply_image(cropped_sam2_image)
                    cropped_sam2_pixel_values = torch.from_numpy(cropped_sam2_image).permute(2, 0, 1).contiguous()
                    cropped_sam2_pixel_values = cropped_sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
                else:
                    cropped_sam2_image = np.array(resized_crop_image)
                    cropped_sam2_image = vq_sam2_image_processor.apply_image(cropped_sam2_image)
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
                zoom_in_mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in crop_quant_codes]) + MT_END_TOKEN
                if resized_crop_image is None:
                    zoomin_image = cropped_image
                else:
                    zoomin_image = resized_crop_image
                item['images'] = [image, zoomin_image]
                item['cap_problem'] = '<image>\nProvide a detailed description of this region {Global_SEG}. Zoom in with the perspective as <image>, {Zoomin_SEG}.'.format(Global_SEG=global_mask_tokens_str, Zoomin_SEG=zoom_in_mask_tokens_str)
            else:
                item['images'] = image
                item['cap_problem'] = '<image>\nProvide a detailed description of this region {Global_SEG}.'.format(Global_SEG=global_mask_tokens_str)
            
            item['cap_answer'] = '<think>\n\n</think><answer>{CAPTION}</answer>'.format(CAPTION=description)
            item['seg_problem'] = PROMPT_TEMPLATE.format(description=description)
            item['seg_answer'] = '<think>\n\n</think><answer>{SEGMENTATION}</answer>'.format(SEGMENTATION=global_mask_tokens_str)
            
            fortran_mask = np.asfortranarray(final_mask)
            rle_dict = mask_util.encode(fortran_mask)
            rle_string = rle_dict['counts'].decode('utf-8')
            entity_rle = {'size': [int(x) for x in rle_dict['size']], 'counts': rle_string}
            item['masks'] = entity_rle

            item['source'] = 'groundingme'
            data_list.append(item)

    trainset = Dataset.from_list(data_list)
    trainset = DatasetDict({"train": trainset}).cast_column("images", Sequence(ImageData()))
    trainset["train"].to_parquet(f"rl_dataset/groundingme_train.parquet")

            ## For Visualization
            # result_image = apply_overlay_and_box(image, final_mask, bbox, color_hex="#33FF33", alpha=160)
            # comparison_image = concat_side_by_side(image, result_image)
            # output_path = f"sam2_result_overlay_{img_idx}.jpg"
            # comparison_image.save(output_path)
            # print(f"Result saved to {output_path}")
            # print()


if __name__ == "__main__":
    main()

    