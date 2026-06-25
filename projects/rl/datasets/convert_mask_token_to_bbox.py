import argparse
import os
import re
from typing import Dict, List, Any, Tuple
from PIL import Image
import numpy as np
import torch
import torchvision
import json
import tqdm
import hydra
from tqdm import tqdm
from datasets import Dataset, load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from torchvision.transforms.functional import to_pil_image


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))


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


def format_bbox_str(x1: float, y1: float, x2: float, y2: float, ori_width: int, ori_height: int) -> str:
    """
    将bbox归一化到[0, 1000]范围并格式化为字符串
    """
    norm_x1 = int(x1 / ori_width * 1000)
    norm_y1 = int(y1 / ori_height * 1000)
    norm_x2 = int(x2 / ori_width * 1000)
    norm_y2 = int(y2 / ori_height * 1000)
    return f"[{norm_x1}, {norm_y1}, {norm_x2}, {norm_y2}]"


def parse_args():
    parser = argparse.ArgumentParser(description='Convert mask tokens to bbox in parquet dataset')
    parser.add_argument(
        '--input_parquet',
        default="rl_dataset/denseworld_10k_img_45977_samples_train.parquet",
        help='Input parquet file path.')
    parser.add_argument(
        '--output_parquet',
        default=None,
        help='Output parquet file path. If None, will auto-generate based on input.')
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
    
    if args.output_parquet is None:
        base_name = os.path.splitext(args.input_parquet)[0]
        args.output_parquet = f"{base_name}_bbox.parquet"
    
    return args


def main():
    args = parse_args()

    # 初始化 VQ-SAM2 模型用于 decode mask token
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
    
    print("VQ-SAM2 model loaded successfully")

    # 读取 parquet 文件
    print(f"Loading dataset from: {args.input_parquet}")
    dataset = load_dataset("parquet", data_files=args.input_parquet, split="train")
    print(f"Total samples: {len(dataset)}")

    # 提取所有 mask token 的正则表达式
    mask_token_pattern = r'<\|mt_start\|><\|mt_\d+\|><\|mt_\d+\|><\|mt_end\|>'
    
    converted_data = []
    failed_count = 0
    
    for sample_idx, sample in enumerate(tqdm(dataset, desc="Converting samples")):
        try:
            images = sample['images']
            cap_problem = sample['cap_problem']
            seg_answer = sample['seg_answer']
            
            # 获取主图像路径（第一张图）
            main_image_path = images[0]
            
            # 加载图像获取尺寸
            image = Image.open(main_image_path).convert("RGB")
            ori_width, ori_height = image.size
            
            # 准备图像用于VQ-SAM2
            sam2_image = np.array(image)
            sam2_image = vq_sam2_image_processor.apply_image(sam2_image)
            sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
            sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
            
            # 找出所有 mask token
            mask_tokens_in_problem = re.findall(mask_token_pattern, cap_problem)
            mask_tokens_in_answer = re.findall(mask_token_pattern, seg_answer)
            
            # 合并所有唯一的 mask token
            all_mask_tokens = list(set(mask_tokens_in_problem + mask_tokens_in_answer))
            
            # 为每个 mask token 计算对应的 bbox
            token_to_bbox = {}
            
            for mask_token in all_mask_tokens:
                try:
                    # 解析 mask token
                    codes = parse_mask_token(mask_token, CODEBOOK_SIZE)
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
                    
                    # 检查 mask 是否为空
                    if pred_masks.sum() == 0:
                        continue
                    
                    # 计算 bbox
                    masks_tensor = torch.from_numpy(pred_masks).unsqueeze(0)
                    boxes = torchvision.ops.masks_to_boxes(masks_tensor)
                    x1, y1, x2, y2 = boxes.squeeze().cpu().numpy().tolist()
                    
                    # 格式化 bbox 字符串
                    bbox_str = format_bbox_str(x1, y1, x2, y2, ori_width, ori_height)
                    token_to_bbox[mask_token] = bbox_str
                    
                except Exception as e:
                    print(f"Error decoding mask token {mask_token}: {e}")
                    continue
            
            # 替换 cap_problem 和 seg_answer 中的 mask token
            new_cap_problem = cap_problem
            new_seg_answer = seg_answer
            
            # Step 1: 先处理 zoom-in 部分的 mask token
            # 格式: "Zoom in on <mask_token1> with the perspective as <image>, <mask_token2>."
            # mask_token1 需要替换为 bbox，mask_token2 需要替换为描述性文本
            zoomin_full_pattern = (
                r'(Zoom in on )(' + mask_token_pattern + r')( with the perspective as <image>, )' 
                + mask_token_pattern + r'(\.|,)'
            )
            new_cap_problem = re.sub(
                zoomin_full_pattern, 
                r'\1\2\3give a detailed description of this region\4', 
                new_cap_problem
            )
            
            # Step 2: 替换剩余的 mask token 为 bbox
            for mask_token, bbox_str in token_to_bbox.items():
                # 使用 re.escape 确保 mask token 中的特殊字符被正确转义
                escaped_token = re.escape(mask_token)
                new_cap_problem = re.sub(escaped_token, bbox_str, new_cap_problem)
                new_seg_answer = re.sub(escaped_token, bbox_str, new_seg_answer)
            
            # 检查是否还有未替换的 mask token
            remaining_tokens = re.findall(mask_token_pattern, new_cap_problem + new_seg_answer)
            if remaining_tokens:
                print(f"Warning: Sample {sample_idx} has {len(remaining_tokens)} unreplaced mask tokens, skipping...")
                failed_count += 1
                continue
            
            # 构建新样本（保留 zoom-in 图像）
            new_sample = {
                'images': images,  # 保留所有图像，包括 zoom-in
                'cap_problem': new_cap_problem,
                'cap_answer': sample.get('cap_answer', None),
                'seg_problem': sample.get('seg_problem', None),
                'seg_answer': new_seg_answer,
                'masks': sample.get('masks', None),
                'source': sample.get('source', 'denseworld_bbox'),
            }
            
            converted_data.append(new_sample)
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            failed_count += 1
            continue
    
    print(f"\nConversion complete:")
    print(f"  Successfully converted: {len(converted_data)}")
    print(f"  Failed: {failed_count}")
    
    # 保存为 parquet 文件
    output_dataset = Dataset.from_list(converted_data)
    output_dataset.to_parquet(args.output_parquet)
    print(f"Saved to {args.output_parquet}")


if __name__ == "__main__":
    main()
