import argparse
import os
import json
import re
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import pandas as pd
import hydra
from typing import List, Tuple

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from torchvision.transforms.functional import to_pil_image


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))


def parse_mask_token(mask_2d: str, codebook_size: int = 256) -> list[int]:
    """从mask token字符串中解析出quant codes"""
    pattern = r'<\|mt_(\d+)\|>'
    matches = re.findall(pattern, mask_2d)
    codes = [int(m) for m in matches]
    return codes


def decode_mask_token(mask_2d: str, vq_sam2, sam2_pixel_values, ori_height: int, ori_width: int, codebook_size: int = 256) -> np.ndarray:
    """解析mask token并decode成mask"""
    codes = parse_mask_token(mask_2d, codebook_size)
    if len(codes) != 2:  # CODEBOOK_DEPTH = 2
        return None
    
    original_codes = [codes[i] - i * codebook_size for i in range(len(codes))]
    quant_codes = torch.tensor(original_codes, dtype=torch.long).unsqueeze(0).cuda()
    
    with torch.no_grad():
        pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_codes)
        pred_masks = pred_masks.detach()
        pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
        pred_masks = pred_masks > 0.5
        pred_masks = pred_masks[0, 0, :, :].cpu().numpy().astype(np.uint8)
    
    return pred_masks


def overlay_all_masks_on_image(image: Image.Image, masks: List[np.ndarray], colors: List[str], alpha: int = 150) -> Image.Image:
    """将多个mask overlay到同一张图像上"""
    base_rgba = image.convert("RGBA")
    H, W = base_rgba.size[1], base_rgba.size[0]
    
    for mask, color_hex in zip(masks, colors):
        mask_bool = mask > 0
        overlay_np = np.zeros((H, W, 4), dtype=np.uint8)
        
        rgb_color = ImageColor.getrgb(color_hex)
        overlay_np[mask_bool, 0] = rgb_color[0]
        overlay_np[mask_bool, 1] = rgb_color[1]
        overlay_np[mask_bool, 2] = rgb_color[2]
        overlay_np[mask_bool, 3] = alpha
        
        overlay_pil = Image.fromarray(overlay_np, "RGBA")
        base_rgba = Image.alpha_composite(base_rgba, overlay_pil)
    
    return base_rgba.convert("RGB")


def overlay_mask_on_image(image: Image.Image, mask: np.ndarray, color_hex: str = "#FF00FF", alpha: int = 150) -> Image.Image:
    """将mask overlay到图像上"""
    base_rgba = image.convert("RGBA")
    H, W = base_rgba.size[1], base_rgba.size[0]
    
    mask_bool = mask > 0
    overlay_np = np.zeros((H, W, 4), dtype=np.uint8)
    
    rgb_color = ImageColor.getrgb(color_hex)
    overlay_np[mask_bool, 0] = rgb_color[0]
    overlay_np[mask_bool, 1] = rgb_color[1]
    overlay_np[mask_bool, 2] = rgb_color[2]
    overlay_np[mask_bool, 3] = alpha
    
    overlay_pil = Image.fromarray(overlay_np, "RGBA")
    combined_img = Image.alpha_composite(base_rgba, overlay_pil)
    
    return combined_img.convert("RGB")


def concat_images_grid(images: List[Image.Image], cols: int = 2) -> Image.Image:
    """将多张图片拼接成网格"""
    if not images:
        return None
    
    # 将所有图片resize到相同大小
    target_h = 300
    resized = []
    for img in images:
        w, h = img.size
        ratio = target_h / h
        new_w = int(w * ratio)
        resized.append(img.resize((new_w, target_h), Image.Resampling.LANCZOS))
    
    # 计算网格大小
    rows = (len(resized) + cols - 1) // cols
    col_widths = [0] * cols
    row_heights = [target_h] * rows
    
    for idx, img in enumerate(resized):
        col = idx % cols
        col_widths[col] = max(col_widths[col], img.width)
    
    total_w = sum(col_widths)
    total_h = sum(row_heights)
    
    # 创建画布并粘贴图片
    grid = Image.new('RGB', (total_w, total_h), color=(255, 255, 255))
    
    for idx, img in enumerate(resized):
        row = idx // cols
        col = idx % cols
        x = sum(col_widths[:col])
        y = sum(row_heights[:row])
        grid.paste(img, (x, y))
    
    return grid


def main():
    parser = argparse.ArgumentParser(description='Visualize parquet dataset')
    parser.add_argument('parquet_path', help='Path to parquet file')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--vq_sam2_path', default="Qwen/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth", help='vq-sam2 model path')
    parser.add_argument('--output_dir', default='visualization_output', help='Output directory for visualizations')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化VQ-SAM2模型
    print("Loading VQ-SAM2 model...")
    CODEBOOK_SIZE = 256
    with hydra.initialize(version_base=None, config_path="../../../projects/transformers/vq_sam2/sam2/sam2_configs"):
        sam2_config = SAM2Config(
            cfg_path="sam2.1_hiera_l.yaml",
            ckpt_path="Qwen/sam2.1_hiera_large.pt",
        )
        vq_sam2_config = VQ_SAM2Config(
            sam2_config=sam2_config,
            codebook_size=CODEBOOK_SIZE,
            codebook_depth=2,
            shared_codebook=False,
            latent_dim=256,
        )
    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()
    state = torch.load(args.vq_sam2_path, map_location="cpu")
    vq_sam2.load_state_dict(state)
    vq_sam2_image_processor = DirectResize(1024)
    print("VQ-SAM2 model loaded")
    
    # 读取parquet文件
    print(f"Loading parquet file: {args.parquet_path}")
    df = pd.read_parquet(args.parquet_path)
    print(f"Total samples: {len(df)}")
    
    # 可视化前N个样本
    num_viz = min(args.num_samples, len(df))
    for sample_idx in range(num_viz):
        print(f"\n{'='*80}")
        print(f"Visualizing sample {sample_idx + 1}/{num_viz}")
        print(f"{'='*80}")
        
        row = df.iloc[sample_idx]
        
        # 解析images列表
        images_str = row['images']
        if isinstance(images_str, str):
            images_list = eval(images_str)  # 转换字符串列表
        else:
            images_list = images_str
        
        print(f"Images: {images_list}")
        print(f"cap_problem:\n{row['cap_problem']}\n")
        print(f"seg_answer:\n{row['seg_answer']}\n")
        
        # 加载原图
        original_image_path = images_list[0]
        try:
            original_image = Image.open(original_image_path).convert('RGB')
            ori_width, ori_height = original_image.size
        except Exception as e:
            print(f"Error loading original image {original_image_path}: {e}")
            continue
        
        # 准备sam2输入
        sam2_image = np.array(original_image)
        sam2_image = vq_sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
        
        # 提取mask tokens
        seg_answer = row['seg_answer']
        mask_tokens = re.findall(r'<\|mt_start\|><\|mt_\d+\|><\|mt_\d+\|><\|mt_end\|>', seg_answer)
        print(f"Found {len(mask_tokens)} mask tokens in seg_answer")
        
        # 提取zoom-in mask tokens（从cap_problem中提取）
        cap_problem = row['cap_problem']
        zoomin_pattern = r'Zoom in on (<\|mt_start\|><\|mt_\d+\|><\|mt_\d+\|><\|mt_end\|>) with the perspective as <image>, (<\|mt_start\|><\|mt_\d+\|><\|mt_\d+\|><\|mt_end\|>)\.'
        zoomin_matches = re.findall(zoomin_pattern, cap_problem)
        zoomin_mask_tokens = [m[1] for m in zoomin_matches]  # 第二个是crop图上的mask token
        print(f"Found {len(zoomin_mask_tokens)} zoom-in mask tokens")
        
        # 解码所有原图mask
        colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FF8800", "#8800FF", "#00FF88", "#FF8888"]
        all_masks = []
        mask_colors = []
        
        for token_idx, mask_token in enumerate(mask_tokens):
            color = colors[token_idx % len(colors)]
            print(f"  Token {token_idx + 1}/{len(mask_tokens)}: {mask_token}")
            
            try:
                mask = decode_mask_token(mask_token, vq_sam2, sam2_pixel_values, ori_height, ori_width, CODEBOOK_SIZE)
                if mask is not None and mask.sum() > 0:
                    all_masks.append(mask)
                    mask_colors.append(color)
                    print(f"    -> Decoded successfully, mask area: {mask.sum()} pixels")
                else:
                    print(f"    -> Empty mask or decode failed")
            except Exception as e:
                print(f"    -> Error decoding: {e}")
        
        # 保存原图所有mask overlay到一起的图
        if len(all_masks) > 0:
            combined_overlay = overlay_all_masks_on_image(original_image, all_masks, mask_colors, alpha=150)
            original_overlay_path = os.path.join(args.output_dir, f"sample_{sample_idx:03d}_original_all_masks.jpg")
            combined_overlay.save(original_overlay_path)
            print(f"\nOriginal image with all masks saved to: {original_overlay_path}")
        
        # 处理zoom-in图像：加载crop图并overlay对应的mask
        crop_image_paths = images_list[1:] if len(images_list) > 1 else []
        for crop_idx, crop_path in enumerate(crop_image_paths):
            print(f"\n  Crop image {crop_idx + 1}: {crop_path}")
            try:
                crop_img = Image.open(crop_path).convert('RGB')
                crop_w, crop_h = crop_img.size
                print(f"    -> Loaded successfully, size: {crop_w}x{crop_h}")
                
                # 对应的zoom-in mask token
                if crop_idx < len(zoomin_mask_tokens):
                    zoomin_token = zoomin_mask_tokens[crop_idx]
                    print(f"    -> Zoom-in mask token: {zoomin_token}")
                    
                    # 准备crop图的sam2输入
                    crop_sam2_image = np.array(crop_img)
                    crop_sam2_image = vq_sam2_image_processor.apply_image(crop_sam2_image)
                    crop_sam2_pixel_values = torch.from_numpy(crop_sam2_image).permute(2, 0, 1).contiguous()
                    crop_sam2_pixel_values = crop_sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
                    
                    # 解码zoom-in mask
                    crop_mask = decode_mask_token(zoomin_token, vq_sam2, crop_sam2_pixel_values, crop_h, crop_w, CODEBOOK_SIZE)
                    if crop_mask is not None and crop_mask.sum() > 0:
                        crop_overlay = overlay_mask_on_image(crop_img, crop_mask, color_hex="#FF00FF", alpha=150)
                        crop_overlay_path = os.path.join(args.output_dir, f"sample_{sample_idx:03d}_crop_{crop_idx:02d}_overlay.jpg")
                        crop_overlay.save(crop_overlay_path)
                        print(f"    -> Crop overlay saved to: {crop_overlay_path}")
                    else:
                        # 保存原始crop图
                        crop_only_path = os.path.join(args.output_dir, f"sample_{sample_idx:03d}_crop_{crop_idx:02d}.jpg")
                        crop_img.save(crop_only_path)
                        print(f"    -> Empty mask, crop image saved to: {crop_only_path}")
                else:
                    # 没有对应的zoom-in token，保存原始crop图
                    crop_only_path = os.path.join(args.output_dir, f"sample_{sample_idx:03d}_crop_{crop_idx:02d}.jpg")
                    crop_img.save(crop_only_path)
                    print(f"    -> No zoom-in token, crop image saved to: {crop_only_path}")
                    
            except Exception as e:
                print(f"    -> Error loading: {e}")
        
        print(f"\n{'='*80}")
        print(f"Sample {sample_idx + 1} visualization complete!")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
