import os
import re
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
import hydra

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
    if mask_2d is None or mask_2d == "None":
        return []
    pattern = r'<\|mt_(\d+)\|>'
    matches = re.findall(pattern, mask_2d)
    codes = [int(m) for m in matches]
    return codes


def extract_mask_token_from_text(text: str) -> str:
    """从文本中提取mask token"""
    if "None" in text:
        return None
    pattern = r'<\|mt_start\|><\|mt_\d+\|><\|mt_\d+\|><\|mt_end\|>'
    matches = re.findall(pattern, text)
    if matches:
        return matches[0]
    return None


def overlay_mask_on_image(image: Image.Image, mask: np.ndarray, color=(0, 255, 0), alpha=128):
    """将mask以半透明方式叠加到图像上"""
    image_rgba = image.convert("RGBA")
    H, W = image.size[1], image.size[0]
    
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    mask_bool = mask > 0
    overlay[mask_bool, 0] = color[0]
    overlay[mask_bool, 1] = color[1]
    overlay[mask_bool, 2] = color[2]
    overlay[mask_bool, 3] = alpha
    
    overlay_img = Image.fromarray(overlay, "RGBA")
    combined = Image.alpha_composite(image_rgba, overlay_img)
    return combined.convert("RGB")


def add_text_to_image(image: Image.Image, text: str, position=(10, 10), color=(255, 0, 0)):
    """在图像上添加文字"""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # 添加背景框使文字更清晰
    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=(255, 255, 255))
    draw.text(position, text, fill=color, font=font)
    return image


def concat_images_horizontal(images: list, labels: list = None):
    """水平拼接多张图片"""
    if not images:
        return None
    
    # 统一高度
    max_height = max(img.height for img in images)
    resized_images = []
    for img in images:
        if img.height != max_height:
            ratio = max_height / img.height
            new_width = int(img.width * ratio)
            img = img.resize((new_width, max_height), Image.Resampling.LANCZOS)
        resized_images.append(img)
    
    total_width = sum(img.width for img in resized_images)
    combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    x_offset = 0
    for i, img in enumerate(resized_images):
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return combined


def main():
    # 初始化 VQ-SAM2 模型
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
    state = torch.load("Qwen/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth", map_location="cpu")
    vq_sam2.load_state_dict(state)
    vq_sam2_image_processor = DirectResize(1024)
    
    print("VQ-SAM2 model loaded successfully")
    
    # 读取 parquet 文件
    parquet_path = "rl_dataset/denseworld_test_247_samples_train.parquet"
    print(f"Loading dataset from: {parquet_path}")
    dataset = load_dataset("parquet", data_files=parquet_path, split="train")
    
    print(f"Total samples: {len(dataset)}")
    
    # 创建输出目录
    output_dir = "rl_dataset/denseworld_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个样本
    for idx, sample in enumerate(dataset):
        print(f"\n--- Sample {idx} ---")
        
        images_paths = sample['images']
        cap_problem = sample['cap_problem']
        seg_answer = sample['seg_answer']
        
        print(f"  Images: {images_paths}")
        print(f"  cap_problem: {cap_problem[:100]}...")
        print(f"  seg_answer: {seg_answer}")
        
        # 提取mask token
        # 从 cap_problem 中提取原图的 mask token
        main_mask_token = extract_mask_token_from_text(cap_problem.split('.')[0])
        
        # 从 cap_problem 中提取 zoom-in 的 mask token（如果有）
        zoomin_mask_token = None
        if "Zoom in" in cap_problem:
            zoomin_part = cap_problem.split("Zoom in")[1]
            zoomin_mask_token = extract_mask_token_from_text(zoomin_part)
        
        visualization_images = []
        labels = []
        
        # 处理原图
        main_image_path = images_paths[0]
        try:
            main_image = Image.open(main_image_path).convert("RGB")
            ori_width, ori_height = main_image.size
            
            if main_mask_token is None:
                # 没有目标
                vis_image = main_image.copy()
                vis_image = add_text_to_image(vis_image, "NO MASK / NO TARGET", position=(10, 10), color=(255, 0, 0))
                visualization_images.append(vis_image)
                labels.append("Original (No Target)")
                print(f"  -> No mask token for main image")
            else:
                # 有目标，decode mask
                codes = parse_mask_token(main_mask_token, CODEBOOK_SIZE)
                if len(codes) == CODEBOOK_DEPTH:
                    # 准备图像
                    sam2_image = np.array(main_image)
                    sam2_image = vq_sam2_image_processor.apply_image(sam2_image)
                    sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
                    sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
                    
                    # 还原原始codes
                    original_codes = [codes[i] - i * CODEBOOK_SIZE for i in range(len(codes))]
                    quant_codes = torch.tensor(original_codes, dtype=torch.long).unsqueeze(0).cuda()
                    
                    # Decode mask
                    with torch.no_grad():
                        pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_codes)
                        pred_masks = pred_masks.detach()
                        pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
                        pred_masks = pred_masks > 0.5
                        pred_masks = pred_masks[0, 0, :, :].cpu().numpy().astype(np.uint8)
                    
                    # Overlay mask
                    vis_image = overlay_mask_on_image(main_image, pred_masks, color=(0, 255, 0), alpha=128)
                    vis_image = add_text_to_image(vis_image, f"Main: {main_mask_token}", position=(10, 10), color=(0, 128, 0))
                    visualization_images.append(vis_image)
                    labels.append("Original + Mask")
                    print(f"  -> Decoded main mask: {main_mask_token}")
                else:
                    vis_image = main_image.copy()
                    vis_image = add_text_to_image(vis_image, "Invalid mask token", position=(10, 10), color=(255, 0, 0))
                    visualization_images.append(vis_image)
                    labels.append("Original (Invalid)")
        except Exception as e:
            print(f"  -> Error loading main image: {e}")
            continue
        
        # 处理 zoom-in 图（如果有）
        if len(images_paths) > 1 and zoomin_mask_token is not None:
            zoomin_image_path = images_paths[1]
            try:
                zoomin_image = Image.open(zoomin_image_path).convert("RGB")
                zoomin_width, zoomin_height = zoomin_image.size
                
                codes = parse_mask_token(zoomin_mask_token, CODEBOOK_SIZE)
                if len(codes) == CODEBOOK_DEPTH:
                    # 准备图像
                    sam2_image = np.array(zoomin_image)
                    sam2_image = vq_sam2_image_processor.apply_image(sam2_image)
                    sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
                    sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
                    
                    # 还原原始codes
                    original_codes = [codes[i] - i * CODEBOOK_SIZE for i in range(len(codes))]
                    quant_codes = torch.tensor(original_codes, dtype=torch.long).unsqueeze(0).cuda()
                    
                    # Decode mask
                    with torch.no_grad():
                        pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_codes)
                        pred_masks = pred_masks.detach()
                        pred_masks = torch.nn.functional.interpolate(pred_masks, size=(zoomin_height, zoomin_width), mode='bilinear')
                        pred_masks = pred_masks > 0.5
                        pred_masks = pred_masks[0, 0, :, :].cpu().numpy().astype(np.uint8)
                    
                    # Overlay mask
                    vis_image = overlay_mask_on_image(zoomin_image, pred_masks, color=(255, 165, 0), alpha=128)
                    vis_image = add_text_to_image(vis_image, f"Zoom: {zoomin_mask_token}", position=(10, 10), color=(255, 100, 0))
                    visualization_images.append(vis_image)
                    labels.append("Zoom-in + Mask")
                    print(f"  -> Decoded zoom-in mask: {zoomin_mask_token}")
            except Exception as e:
                print(f"  -> Error loading zoom-in image: {e}")
        
        # 拼接并保存
        if visualization_images:
            combined = concat_images_horizontal(visualization_images)
            output_path = os.path.join(output_dir, f"sample_{idx:04d}.jpg")
            combined.save(output_path, quality=95)
            print(f"  -> Saved to {output_path}")
    
    print(f"\nVisualization complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
