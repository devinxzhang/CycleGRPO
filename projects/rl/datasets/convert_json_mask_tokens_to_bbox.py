#!/usr/bin/env python3
"""Convert mask tokens in a JSON file to bbox strings using VQ-SAM2.

Input JSON format: a list of samples where each sample contains at least:
  - 'image': list of image paths (first image used to get size)
  - 'conversations': list of dicts with 'from' and 'value' fields

This script preserves all fields and only replaces mask token substrings
inside conversation 'value' strings with bbox strings like [x1, y1, x2, y2]
where coordinates are normalized to [0, 1000].

Example:
  python convert_json_mask_tokens_to_bbox.py --input input.json --output output.json \
    --vq_sam2_path Qwen/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth
"""
import argparse
import json
import os
import re
from typing import List

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torchvision

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        img = Image.fromarray(image)
        return np.array(img.resize((self.target_length, self.target_length)))


def parse_mask_token(mask_2d: str) -> List[int]:
    pattern = r'<\|mt_(\d+)\|>'
    matches = re.findall(pattern, mask_2d)
    return [int(m) for m in matches]


def format_bbox_str(x1: float, y1: float, x2: float, y2: float, ori_w: int, ori_h: int) -> str:
    norm_x1 = int(x1 / ori_w * 1000)
    norm_y1 = int(y1 / ori_h * 1000)
    norm_x2 = int(x2 / ori_w * 1000)
    norm_y2 = int(y2 / ori_h * 1000)
    return f"[{norm_x1}, {norm_y1}, {norm_x2}, {norm_y2}]"


def parse_args():
    p = argparse.ArgumentParser(description='Convert mask tokens inside JSON samples to bbox strings')
    p.add_argument('--input', required=True, help='Input JSON file (list of samples)')
    p.add_argument('--output', required=False, help='Output JSON file (default: <input>_bbox.json)')
    p.add_argument('--vq_sam2_path', default='Qwen/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth',
                   help='Path to VQ-SAM2 model weights')
    p.add_argument('--sam_ckpt', default='Qwen/sam2.1_hiera_large.pt', help='SAM checkpoint used by VQ-SAM2')
    p.add_argument('--num_workers', type=int, default=1, help='Number of threads (not used)')
    return p.parse_args()


def main():
    args = parse_args()

    out_path = args.output if args.output else os.path.splitext(args.input)[0] + '_bbox.json'

    # VQ-SAM2 config (assumes same config files available in repo)
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2

    # initialize VQ-SAM2
    import hydra
    with hydra.initialize(version_base=None, config_path="../../../projects/transformers/vq_sam2/sam2/sam2_configs"):
        sam2_config = SAM2Config(cfg_path="sam2.1_hiera_l.yaml", ckpt_path=args.sam_ckpt)
        vq_sam2_config = VQ_SAM2Config(
            sam2_config=sam2_config,
            codebook_size=CODEBOOK_SIZE,
            codebook_depth=CODEBOOK_DEPTH,
            shared_codebook=False,
            latent_dim=256,
        )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()
    state = torch.load(args.vq_sam2_path, map_location='cpu')
    vq_sam2.load_state_dict(state)
    image_processor = DirectResize(1024)

    print('VQ-SAM2 loaded')

    # token pattern like <|mt_start|><|mt_0123|><|mt_0456|><|mt_end|>
    mask_token_pattern = r'<\|mt_start\|><\|mt_\d+\|><\|mt_\d+\|><\|mt_end\|>'

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    converted = []
    failed = 0

    for idx, sample in tqdm(enumerate(data), total=len(data), desc="Converting samples"):
        try:
            images = sample.get('image') or sample.get('images')
            if not images or len(images) == 0:
                print(f"Sample {idx} has no images, skipping")
                failed += 1
                continue

            main_image_path = images[0]
            img = Image.open(main_image_path).convert('RGB')
            ori_w, ori_h = img.size

            # prepare image for vq_sam2
            sam2_image = np.array(img)
            sam2_image = image_processor.apply_image(sam2_image)
            sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous().unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

            # collect mask tokens from all conversation values
            convo_text = ''
            for c in sample.get('conversations', []):
                if 'value' in c and isinstance(c['value'], str):
                    convo_text += c['value'] + '\n'

            tokens_in_text = re.findall(mask_token_pattern, convo_text)
            unique_tokens = list(set(tokens_in_text))

            token_to_bbox = {}

            for token in unique_tokens:
                try:
                    codes = parse_mask_token(token)
                    if len(codes) != CODEBOOK_DEPTH:
                        # skip unexpected depth
                        continue

                    original_codes = [codes[i] - i * CODEBOOK_SIZE for i in range(len(codes))]
                    quant_codes = torch.tensor(original_codes, dtype=torch.long).unsqueeze(0).cuda()

                    with torch.no_grad():
                        pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_codes)
                        pred_masks = pred_masks.detach()
                        pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_h, ori_w), mode='bilinear')
                        pred_masks = pred_masks > 0.5
                        pred_mask = pred_masks[0, 0, :, :].cpu().numpy().astype(np.uint8)

                    if pred_mask.sum() == 0:
                        continue

                    masks_tensor = torch.from_numpy(pred_mask).unsqueeze(0)
                    boxes = torchvision.ops.masks_to_boxes(masks_tensor)
                    x1, y1, x2, y2 = boxes.squeeze().cpu().numpy().tolist()
                    bbox_str = format_bbox_str(x1, y1, x2, y2, ori_w, ori_h)
                    token_to_bbox[token] = bbox_str
                except Exception as e:
                    print(f"Error decoding token {token} in sample {idx}: {e}")
                    continue

            # replace tokens inside conversation values
            new_sample = dict(sample)  # shallow copy
            new_convos = []
            for c in sample.get('conversations', []):
                new_c = dict(c)
                if 'value' in new_c and isinstance(new_c['value'], str):
                    text = new_c['value']
                    for tok, bbox_str in token_to_bbox.items():
                        text = re.sub(re.escape(tok), bbox_str, text)
                    new_c['value'] = text
                new_convos.append(new_c)
            new_sample['conversations'] = new_convos

            # check for remaining tokens
            remaining = re.findall(mask_token_pattern, '\n'.join([c.get('value','') for c in new_convos]))
            if remaining:
                print(f"Warning: sample {idx} has {len(remaining)} remaining mask tokens; skipping sample")
                failed += 1
                continue

            converted.append(new_sample)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            failed += 1
            continue

    print(f"Converted: {len(converted)}, Failed: {failed}")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print('Saved to', out_path)


if __name__ == '__main__':
    main()
