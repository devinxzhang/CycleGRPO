import argparse
import os
import sys
import collections
import os.path as osp
import random
import copy
from typing import Dict, List, Any, Tuple
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

import mmengine
from mmengine.dataset import BaseDataset
import pycocotools.mask as mask_util

from transformers import Sam2Processor, Sam2Model
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from torchvision.transforms.functional import resize, to_pil_image


def get_image_and_all_masks(sample: dict[str, Any]):
    """
    从样本中获取image path和所有的mask_2d
    支持两种格式:
    1. mask_2d直接嵌入文本中: <|mt_start|><|mt_xxxx|><|mt_xxxx|><|mt_end|>
    2. mask_2d在```json```代码块中
    
    Args:
        sample: 包含'image'和'conversations'字段的样本字典
        
    Returns:
        tuple: (image_path, list[mask_2d])
               如果没有找到mask_2d，返回 (image_path, [])
    """
    # 1. 获取images路径
    image_path = sample['image']
    if isinstance(image_path, list):
        image_path = image_path[0]
    
    # 2. 从gpt回答中提取mask_2d
    gpt_value = sample['conversations'][1]['value']
    
    # 方式1: 尝试从```json```代码块中提取
    json_match = re.search(r'```json\s*(.*?)\s*```', gpt_value, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            all_mask_2d = [item['mask_2d'] for item in result]
            if all_mask_2d:
                return image_path, all_mask_2d
        except:
            pass
    
    # 方式2: 直接从文本中提取mask_2d格式
    pattern = r'<\|mt_start\|><\|mt_\d+\|><\|mt_\d+\|><\|mt_end\|>'
    all_mask_2d = re.findall(pattern, gpt_value)
    if all_mask_2d:
        return image_path, all_mask_2d
    
    return image_path, []


def parse_mask_token(mask_2d: str, codebook_size: int = 256) -> list[int]:
    """
    从mask token字符串中解析出quant codes
    
    Args:
        mask_2d: 如 <|mt_start|><|mt_0012|><|mt_0268|><|mt_end|>
        codebook_size: codebook大小，默认256
        
    Returns:
        list[int]: 解析出的codes列表
    """
    pattern = r'<\|mt_(\d+)\|>'
    matches = re.findall(pattern, mask_2d)
    codes = [int(m) for m in matches]
    return codes


def build_sample_dict_single_object(
    image_path: str, 
    mask_token: str = None, 
    zoomin_image_path: str = None,
    zoomin_mask_token: str = None,
) -> dict[str, Any]:
    """
    为单个物体构建训练样本dict
    
    Args:
        image_path: 原图路径
        mask_token: 该物体的mask token，可以为None表示没有mask
        zoomin_image_path: zoom-in图像路径（可选）
        zoomin_mask_token: zoom-in的mask token（可选）
        
    Returns:
        dict: 包含['images', 'cap_problem', 'cap_answer', 'seg_problem', 'seg_answer', 'masks', 'source']
    """
    # 构建图像列表
    image_paths = [image_path]
    
    # 处理 mask_token 为 None 的情况
    if mask_token is None:
        cap_problem = f'<image>\nProvide a detailed description of this region {None}.'
        seg_answer = f'<answer>{None}</answer>'
    else:
        cap_problem = f'<image>\nProvide a detailed description of this region {mask_token}.'
        seg_answer = f'<answer>{mask_token}</answer>'
        
        # 添加zoom-in信息（仅当有mask_token时）
        if zoomin_image_path and zoomin_mask_token:
            cap_problem += f' Zoom in with the perspective as <image>, {zoomin_mask_token}.'
            image_paths.append(zoomin_image_path)
    
    return {
        'images': image_paths,
        'cap_problem': cap_problem,
        'cap_answer': None,
        'seg_problem': None,
        'seg_answer': seg_answer,
        'masks': None,
        'source': 'denseworld_multiple',
    }


def build_sample_dict_multi_object(
    image_path: str,
    mask_tokens: List[str],
    zoomin_info: List[Tuple[str, str]] = None,  # [(zoomin_image_path, zoomin_mask_token), ...] 与 mask_tokens 一一对应，None 表示无 zoom-in
) -> dict[str, Any]:
    """
    为多个物体构建训练样本dict，支持小目标的zoom-in
    
    Args:
        image_path: 原图路径
        mask_tokens: 多个物体的mask token列表
        zoomin_info: 与 mask_tokens 一一对应的 zoom-in 信息列表，每个元素是 (zoomin_image_path, zoomin_mask_token) 或 None
        
    Returns:
        dict: 包含['images', 'cap_problem', 'cap_answer', 'seg_problem', 'seg_answer', 'masks', 'source']
    """
    image_paths = [image_path]
    
    if zoomin_info is None:
        zoomin_info = [None] * len(mask_tokens)
    
    if not mask_tokens or len(mask_tokens) == 0:
        cap_problem = f'<image>\nCould you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.'
        seg_answer = '<answer></answer>'
    else:
        # 构建 mask tokens 字符串
        if len(mask_tokens) == 1:
            tokens_str = mask_tokens[0]
        elif len(mask_tokens) == 2:
            tokens_str = f'{mask_tokens[0]} and {mask_tokens[1]}'
        else:
            # 3个或更多：token1, token2, ..., and tokenN
            tokens_str = ', '.join(mask_tokens[:-1]) + ', and ' + mask_tokens[-1]
        
        cap_problem = f'<image>\nCould you please give me a detailed description of the following regions? {tokens_str}. Please respond with interleaved segmentation masks for the corresponding parts of the answer.'
        
        # 添加 zoom-in 信息
        for idx, zi in enumerate(zoomin_info):
            if zi is not None:
                zoomin_image_path, zoomin_mask_token = zi
                image_paths.append(zoomin_image_path)
                cap_problem += f' Zoom in on {mask_tokens[idx]} with the perspective as <image>, {zoomin_mask_token}.'
        
        seg_answer = '<answer>' + ', '.join(mask_tokens) + '</answer>'
    
    return {
        'images': image_paths,
        'cap_problem': cap_problem,
        'cap_answer': None,
        'seg_problem': None,
        'seg_answer': seg_answer,
        'masks': None,
        'source': 'denseworld_multiple',
    }


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
    parser.add_argument(
        '--vq_sam2_path',
        default="Qwen/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of threads for parallel processing')
    args = parser.parse_args()
    return args


def preprocess_sample(sample: dict, image_cache: dict = None) -> Tuple[str, List[str]]:
    """
    预处理单个样本，提取 image_path 和所有 mask_2d
    返回 (image_path, [mask_2d_1, mask_2d_2, ...])
    """
    image_path, all_mask_2d = get_image_and_all_masks(sample)
    return (image_path, all_mask_2d)


def load_image_worker(task: Tuple[int, str]) -> Tuple[int, Image.Image, int, int]:
    """
    多线程加载图像
    """
    idx, image_path = task
    try:
        image = Image.open(image_path).convert("RGB")
        ori_width, ori_height = image.size
        return (idx, image, ori_width, ori_height)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return (idx, None, 0, 0)


def save_crop_worker(task: Tuple[str, Image.Image]) -> str:
    """
    多线程保存裁剪图像，如果文件已存在则跳过
    """
    save_path, image = task
    try:
        if os.path.exists(save_path):
            return save_path  # 文件已存在，跳过保存
        image.save(save_path)
        return save_path
    except Exception as e:
        print(f"Error saving {save_path}: {e}")
        return None


def main():
    args = parse_args()
    random.seed(42)

    # 初始化 VQ-SAM2 模型用于 decode mask token
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'
    
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
    
    print("VQ-SAM2 model loaded successfully")

    # 读取 denseworld 数据集
    denseworld_path = "./data/tokenmask_data_256x2_cot_format/mask_generation_denseworld872k_clean_repeat_pattern.json"
    print(f"Loading dataset from: {denseworld_path}")
    with open(denseworld_path, 'r') as f:
        denseworld_data = json.load(f)
    
    denseworld_data = denseworld_data[15000:20000]  # 仅用于测试，实际使用时注释掉
    print(f"Total samples: {len(denseworld_data)}")
    
    # 处理全部样本 872485
    num_samples = len(denseworld_data)
    print(f"Using all {num_samples} samples")
    
    # Step 1: 预处理，提取所有 (image_path, [mask_2d_list]) 对
    print("Step 1: Extracting all image-mask pairs...")
    all_samples = []  # [(image_path, [mask_2d_1, mask_2d_2, ...]), ...]
    no_mask_count = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(preprocess_sample, sample) for sample in denseworld_data]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            image_path, mask_list = future.result()
            if len(mask_list) > 0:
                all_samples.append((image_path, mask_list))
            else:
                # 没有mask的图也保留，mask_list为空列表
                all_samples.append((image_path, []))
                no_mask_count += 1
    
    print(f"Images without masks: {no_mask_count}")
    
    print(f"Total images with masks: {len(all_samples)}")
    total_masks = sum(len(masks) for _, masks in all_samples)
    print(f"Total masks: {total_masks}")
    
    # Step 2+3: 边处理边加载图片
    print("Step 2: Processing images and generating samples (streaming)...")
    data_list = []
    small_object_count = 0
    crop_save_dir = "rl_dataset/denseworld_5k_crops_multiple"
    os.makedirs(crop_save_dir, exist_ok=True)
    crops_to_save = []  # [(save_path, image), ...]

    for image_path, all_mask_2d in tqdm(all_samples, desc="Processing images"):
        try:
            image = Image.open(image_path).convert("RGB")
            ori_width, ori_height = image.size
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            continue

        # 如果没有mask，生成一个空mask token列表的样本
        if len(all_mask_2d) == 0:
            # item = build_sample_dict_multi_object(
            #     image_path=image_path,
            #     mask_tokens=[],
            #     zoomin_info=[],
            # )
            # data_list.append(item)
            continue

        # 收集该图像的所有有效 mask token 和对应的 zoom-in 信息
        valid_mask_tokens = []
        valid_zoomin_info = []  # 与 valid_mask_tokens 一一对应
        
        # 准备图像用于VQ-SAM2（只处理一次）
        sam2_image = np.array(image)
        sam2_image = vq_sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        # 对每个 mask 验证有效性并收集
        for mask_idx, mask_2d in enumerate(all_mask_2d):
            try:
                # 解析mask token并decode
                codes = parse_mask_token(mask_2d, CODEBOOK_SIZE)
                if len(codes) != CODEBOOK_DEPTH:
                    continue

                # 还原原始codes（去除depth offset）
                original_codes = [codes[i] - i * CODEBOOK_SIZE for i in range(len(codes))]
                quant_codes = torch.tensor(original_codes, dtype=torch.long).unsqueeze(0).cuda()

                # Decode mask
                with torch.no_grad():
                    pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_codes)
                    pred_masks = pred_masks.detach()
                    pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
                    pred_masks = pred_masks > 0.5
                    pred_masks = pred_masks[0, 0, :, :].cpu().numpy().astype(np.uint8)

                final_mask = pred_masks

                # 检查 mask 是否为空
                if final_mask.sum() == 0:
                    continue

                # 当前物体的 zoom-in 信息
                zoomin_image_path = None
                zoomin_mask_token = None

                # 计算bbox和面积占比
                masks_tensor = torch.from_numpy(final_mask).unsqueeze(0)
                boxes = torchvision.ops.masks_to_boxes(masks_tensor)
                x1, y1, x2, y2 = boxes.squeeze().cpu().numpy().tolist()
                boxes_w = x2 - x1
                boxes_h = y2 - y1
                boxes_area = boxes_h * boxes_w
                image_area = ori_height * ori_width
                boxes_occupied_ratio = boxes_area / image_area

                # 判断是否需要zoom-in（面积占比小于30%）
                if boxes_occupied_ratio < 0.3:
                    small_object_count += 1

                    # 扩展bbox，确保不小于140像素
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

                    # Crop图像
                    cropped_image = image.crop((x1, y1, x2, y2))
                    crop_width, crop_height = cropped_image.size

                    # 如果crop太小，需要resize
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

                    # 为crop图像计算新的mask token
                    if resized_crop_image is None:
                        cropped_sam2_image = np.array(cropped_image)
                    else:
                        cropped_sam2_image = np.array(resized_crop_image)

                    cropped_sam2_image = vq_sam2_image_processor.apply_image(cropped_sam2_image)
                    cropped_sam2_pixel_values = torch.from_numpy(cropped_sam2_image).permute(2, 0, 1).contiguous()
                    cropped_sam2_pixel_values = cropped_sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

                    # Crop mask
                    cropped_mask = final_mask[y1:y2, x1:x2]
                    cropped_mask_tensor = torch.from_numpy(np.ascontiguousarray(cropped_mask))

                    if resized_crop_image is not None:
                        cropped_mask_tensor = torch.nn.functional.interpolate(
                            cropped_mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                            size=(new_height, new_width), 
                            mode='bilinear'
                        )
                        cropped_mask_tensor = (cropped_mask_tensor[0, 0] > 0.5)

                    crop_height_final, crop_width_final = cropped_mask_tensor.shape[-2:]
                    cropped_mask_for_box = cropped_mask_tensor.unsqueeze(0)
                    cropped_box = torchvision.ops.masks_to_boxes(cropped_mask_for_box)
                    crop_whwh = torch.as_tensor([[crop_width_final, crop_height_final, crop_width_final, crop_height_final]])
                    cropped_box = cropped_box / crop_whwh
                    cropped_box = cropped_box.to(vq_sam2.device)
                    cropped_mask_input = cropped_mask_tensor.unsqueeze(0).to(vq_sam2.device)

                    with torch.no_grad():
                        cropped_vq_sam2_output = vq_sam2(
                            cropped_sam2_pixel_values,
                            [cropped_mask_input],
                            cropped_box,
                            reconstruct_mask=False,
                        )

                    crop_quant_codes = cropped_vq_sam2_output.quant_codes.squeeze().detach().cpu().numpy().astype(np.int32).tolist()
                    remap_crop_quant_codes = [depth_idx * CODEBOOK_SIZE + quant_code for depth_idx, quant_code in enumerate(crop_quant_codes)]
                    zoom_in_mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in remap_crop_quant_codes]) + MT_END_TOKEN

                    # 保存crop图像路径
                    crop_filename = f"{uuid.uuid4().hex}.jpg"
                    crop_save_path = os.path.join(crop_save_dir, crop_filename)

                    if resized_crop_image is None:
                        crops_to_save.append((crop_save_path, cropped_image.copy()))
                    else:
                        crops_to_save.append((crop_save_path, resized_crop_image.copy()))

                    zoomin_image_path = crop_save_path
                    zoomin_mask_token = zoom_in_mask_tokens_str

                # mask 有效，加入列表
                valid_mask_tokens.append(mask_2d)
                if zoomin_image_path is not None:
                    valid_zoomin_info.append((zoomin_image_path, zoomin_mask_token))
                else:
                    valid_zoomin_info.append(None)

            except Exception as e:
                print(f"Error processing mask {mask_idx} in {image_path}: {e}")
                continue
        
        # 为该图像生成一个包含所有有效 mask token 的样本
        if len(valid_mask_tokens) > 0:
            item = build_sample_dict_multi_object(
                image_path=image_path,
                mask_tokens=valid_mask_tokens,
                zoomin_info=valid_zoomin_info,
            )
            data_list.append(item)
    
    # Step 4: 多线程保存裁剪图像
    print(f"Step 4: Saving {len(crops_to_save)} crop images in parallel...")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(save_crop_worker, task) for task in crops_to_save]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Saving crops"):
            future.result()
    
    print(f"Total images processed: {len(all_samples)}")
    print(f"Total masks: {total_masks}")
    print(f"Successfully generated {len(data_list)} samples")
    print(f"Small objects with zoom-in: {small_object_count}")
    
    # 已改为边处理边加载图片，无需清理 image_cache
    
    # 保存为 parquet 文件（全量样本）
    trainset = Dataset.from_list(data_list)
    num_images_k = len(all_samples) // 1000
    num_samples = len(data_list)
    output_path = f"rl_dataset/denseworld_{num_images_k}k_img_{num_samples}_samples_train.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trainset.to_parquet(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()

    