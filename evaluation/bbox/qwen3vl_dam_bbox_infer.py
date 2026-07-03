import argparse
import copy
import math
import os
import torch
import torchvision
import tqdm
from pycocotools import mask as mask_utils
import numpy as np
import random
import re
from PIL import Image
import json
import uuid
import base64
import io
from tqdm import tqdm

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from torchvision.transforms.functional import to_pil_image


def parse_args():
    parser = argparse.ArgumentParser(description='DAM with BBox')
    parser.add_argument(
        '--model_path',
        default="Qwen/Qwen3-VL-4B",
        help='hf model path.')
    parser.add_argument(
        '--data_path',
        default='./data/DLC-Bench/DLC-bench.json',
        help='DLC-Bench json path.')
    parser.add_argument(
        '--image_folder',
        default='./data/DLC-Bench/images',
        help='Image folder path.')
    parser.add_argument(
        '--output',
        default='./results/dam_bbox.json',
        help='Output json path.')
    args = parser.parse_args()
    return args


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


def main():
    args = parse_args()

    # Build Qwen3-VL model (vanilla, no VQ-SAM2)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    # Load DLC-Bench data
    with open(args.data_path, 'r') as f:
        eval_samples = json.load(f)
    
    all_items = []
    for eval_sample in tqdm(eval_samples):
        image_name = eval_sample['image_name']
        save_mask_samples = []
        
        for mask_sample in eval_sample['mask_samples']:
            mask_anno = mask_sample['segmentation']
            category_name = mask_sample['class_name']

            image_path = os.path.join(args.image_folder, image_name)

            image = Image.open(image_path).convert('RGB')
            ori_width, ori_height = image.size

            # Decode mask and get bounding box
            binary_masks = decode_mask([mask_anno], ori_height, ori_width)
            masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])
            boxes = torchvision.ops.masks_to_boxes(masks)
            x1, y1, x2, y2 = boxes.squeeze().cpu().numpy().tolist()
            
            # Calculate box area ratio
            boxes_w = x2 - x1
            boxes_h = y2 - y1
            boxes_area = boxes_h * boxes_w
            image_area = ori_height * ori_width
            boxes_occupied_ratio = boxes_area / image_area

            # Format bbox as normalized coordinates [0, 1000]
            norm_x1 = int(x1 / ori_width * 1000)
            norm_y1 = int(y1 / ori_height * 1000)
            norm_x2 = int(x2 / ori_width * 1000)
            norm_y2 = int(y2 / ori_height * 1000)
            bbox_str = f"[{norm_x1}, {norm_y1}, {norm_x2}, {norm_y2}]"

            if boxes_occupied_ratio < 0.3:
                # Zoom in logic for small regions
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

                # Resize if too small
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
                    resized_crop_image = None

                # Prepare cropped image for zoom-in
                buffer = io.BytesIO()
                if resized_crop_image is None:
                    cropped_image.save(buffer, format='JPEG')
                else:
                    resized_crop_image.save(buffer, format='JPEG')
                buffer.seek(0)
                crop_b64 = base64.b64encode(buffer.read()).decode("utf-8")

                with open(image_path, "rb") as f:
                    global_b64 = base64.b64encode(f.read()).decode()

                question = f"Given a detailed description of the region at bounding box {bbox_str}. Zoom in with the perspective as"

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{global_b64}",
                            },
                            {"type": "text", "text": question},
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{crop_b64}",
                            },
                            {"type": "text", "text": ", give a detailed description of this cropped region."},
                        ],
                    }
                ]
            else:
                # Use global image with bbox
                question = f"Given a detailed description of the region at bounding box {bbox_str}."
                with open(image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{b64}",
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=1024,
                do_sample=False,
                top_p=1.0,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            save_sample = copy.deepcopy(mask_sample)
            save_sample.update({'pred_caption': output_text[0].replace('<|im_end|>', '')})
            save_mask_samples.append(save_sample)
            
        copy_eval_sample = copy.deepcopy(eval_sample)
        copy_eval_sample.update({'mask_samples': save_mask_samples})
        all_items.append(copy_eval_sample)

    print(len(all_items), " items")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_items, f, indent=4)

    print(f"Saved results to {args.output}")


if __name__ == '__main__':     
    main()
