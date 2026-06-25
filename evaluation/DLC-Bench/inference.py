# --------------------------------------------------------
# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
# Grasp Any Region Project
# Written by Haochen Wang
# --------------------------------------------------------

import argparse
import json
import os
import copy
import base64
import io

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from torchvision.transforms.functional import to_pil_image
import hydra

from projects.transformers.vq_sam2 import SAM2Config, VQ_SAM2Config, VQ_SAM2

TORCH_DTYPE_MAP = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)

# VQ-SAM2 constants
MT_START_TOKEN = '<|mt_start|>'
MT_END_TOKEN = '<|mt_end|>'
MT_CONTEXT_TOKEN = '<|mt_{}|>'
CODEBOOK_SIZE = 256
CODEBOOK_DEPTH = 2


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference of Grasp Any Region models on DLC-Bench."
    )

    parser.add_argument(
        "--model_path",
        help="HF model name or path",
        default="zhouyik/Qwen3-VL-4B-SAMTok-dam",
    )
    parser.add_argument(
        "--vq_sam2_path",
        help="vq-sam2 model path.",
        default="Qwen/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth",
    )
    parser.add_argument(
        "--sam2_path",
        help="sam2 model path.",
        default="Qwen/sam2.1_hiera_large.pt",
    )
    parser.add_argument(
        "--cache_name",
        help="cache name to save model outputs.",
        default="qwen3vl_samtok",
    )
    parser.add_argument(
        "--data_type",
        help="data dtype",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="bf16",
    )
    parser.add_argument(
        "--anno_file",
        help="path to the annotation file.",
        default="evaluation/DLC-Bench/annotations/annotations.json",
    )
    parser.add_argument(
        "--image_folder",
        help="the folder of images",
        default="evaluation/DLC-Bench/annotations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible text generation",
    )
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


def encode_mask_to_tokens(vq_sam2, sam2_image_processor, image, binary_masks, ori_width, ori_height):
    """Encode a mask to VQ-SAM2 tokens."""
    sam2_image = np.array(image)
    sam2_image = sam2_image_processor.apply_image(sam2_image)
    sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
    sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

    masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])
    boxes = torchvision.ops.masks_to_boxes(masks)
    
    whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
    boxes_normalized = boxes / whwh
    boxes_normalized = boxes_normalized.to(vq_sam2.device)
    masks_list = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
    
    with torch.no_grad():
        vq_sam2_output = vq_sam2(
            sam2_pixel_values,
            masks_list,
            boxes_normalized,
            reconstruct_mask=False,
        )

    quant_codes = vq_sam2_output.quant_codes.squeeze().cpu().numpy().astype(np.int32).tolist()
    remap_quant_codes = [depth_idx * CODEBOOK_SIZE + quant_code for depth_idx, quant_code in enumerate(quant_codes)]
    mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in remap_quant_codes]) + MT_END_TOKEN
    
    return mask_tokens_str, boxes


def select_ann(coco, img_id, area_min=None, area_max=None):
    cat_ids = coco.getCatIds()
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)

    if area_min is not None:
        ann_ids = [
            ann_id for ann_id in ann_ids if coco.anns[ann_id]["area"] >= area_min
        ]

    if area_max is not None:
        ann_ids = [
            ann_id for ann_id in ann_ids if coco.anns[ann_id]["area"] <= area_max
        ]

    return ann_ids


def main():
    args = parse_args()
    data_dtype = TORCH_DTYPE_MAP[args.data_type]
    torch.manual_seed(args.seed)

    # Build Qwen3VL model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    # Build vq-sam2 model
    # Use absolute path for hydra config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../../projects/transformers/vq_sam2/sam2/sam2_configs')
    config_path = os.path.abspath(config_path)
    
    with hydra.initialize_config_dir(version_base=None, config_dir=config_path):
        sam2_config = SAM2Config(
            cfg_path="sam2.1_hiera_l.yaml",
            ckpt_path=args.sam2_path,
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

    sam2_image_processor = DirectResize(1024)

    model_outputs = {}
    cache_name = args.cache_name

    # This coco instance is actually an o365 subset. This is for code reuse.
    coco = COCO(args.anno_file)
    img_ids = list(coco.imgs.keys())
    num_anns = len(coco.anns)
    pbar = tqdm(total=num_anns)

    for img_id in img_ids:
        ann_ids = select_ann(coco, img_id)
        img_info = coco.loadImgs(img_id)[0]

        for i, ann_id in enumerate(ann_ids):
            if ann_id in model_outputs.keys():
                pbar.update(1)
                continue

            anns = coco.loadAnns([ann_id])
            mask = coco.annToMask(anns[0])

            img_path = os.path.join(args.image_folder, "images", img_info["file_name"])
            image = Image.open(img_path).convert('RGB')
            ori_width, ori_height = image.size

            # Encode mask to tokens (wrap mask in a list)
            binary_masks = [mask]
            global_mask_tokens_str, boxes = encode_mask_to_tokens(
                vq_sam2, sam2_image_processor, image, binary_masks, ori_width, ori_height
            )

            # Calculate box info for zoom-in decision
            x1, y1, x2, y2 = boxes.squeeze().cpu().numpy().tolist()
            boxes_w = x2 - x1
            boxes_h = y2 - y1
            boxes_area = boxes_h * boxes_w
            image_area = ori_height * ori_width
            boxes_occupied_ratio = boxes_area / image_area

            # Prepare messages based on zoom-in decision
            if boxes_occupied_ratio < 0.3:
                # Zoom in case
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

                # Resize cropped image if needed
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

                # Process cropped image for VQ-SAM2
                if resized_crop_image is None:
                    cropped_sam2_image = np.array(cropped_image)
                else:
                    cropped_sam2_image = np.array(resized_crop_image)
                cropped_sam2_image = sam2_image_processor.apply_image(cropped_sam2_image)
                cropped_sam2_pixel_values = torch.from_numpy(cropped_sam2_image).permute(2, 0, 1).contiguous()
                cropped_sam2_pixel_values = cropped_sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

                cropped_masks = torch.stack([torch.from_numpy(np.ascontiguousarray(mask.copy()[y1:y2, x1:x2]))])
                
                if resized_crop_image is not None:
                    resized_crop_masks = torch.nn.functional.interpolate(cropped_masks.unsqueeze(0), size=(new_height, new_width), mode='bilinear')
                    resized_crop_masks = resized_crop_masks[0] > 0.5
                    cropped_masks = resized_crop_masks
                
                crop_height_mask, crop_width_mask = cropped_masks.shape[-2:]
                cropped_boxes = torchvision.ops.masks_to_boxes(cropped_masks)
                crop_whwh = torch.as_tensor([[crop_width_mask, crop_height_mask, crop_width_mask, crop_height_mask]])
                cropped_boxes = cropped_boxes / crop_whwh
                cropped_boxes = cropped_boxes.to(vq_sam2.device)
                cropped_masks_list = [m.unsqueeze(0).to(vq_sam2.device) for m in cropped_masks]

                with torch.no_grad():
                    cropped_vq_sam2_output = vq_sam2(
                        cropped_sam2_pixel_values,
                        cropped_masks_list,
                        cropped_boxes,
                        reconstruct_mask=True,
                    )
                
                crop_quant_codes = cropped_vq_sam2_output.quant_codes.squeeze().detach().cpu().numpy().astype(np.int32).tolist()
                remap_crop_quant_codes = [depth_idx * CODEBOOK_SIZE + quant_code for depth_idx, quant_code in enumerate(crop_quant_codes)]
                zoom_in_mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in remap_crop_quant_codes]) + MT_END_TOKEN

                question = "Given a detailed description of this region {SEG}. Zoom in with the perspective as ".format(SEG=global_mask_tokens_str)
                
                # Encode images to base64
                buffer = io.BytesIO()
                if resized_crop_image is None:
                    cropped_image.save(buffer, format='JPEG')
                else:
                    resized_crop_image.save(buffer, format='JPEG')
                buffer.seek(0)
                crop_b64 = base64.b64encode(buffer.read()).decode("utf-8")

                with open(img_path, "rb") as f:
                    global_b64 = base64.b64encode(f.read()).decode()

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
                            {"type": "text", "text": f", {zoom_in_mask_tokens_str}."},
                        ],
                    }
                ]
            else:
                # No zoom in case
                question = "Given a detailed description of this region {SEG}.".format(SEG=global_mask_tokens_str)
                with open(img_path, "rb") as f:
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

            # Apply chat template and generate
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            with torch.no_grad():
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

            outputs = output_text[0].replace('<|im_end|>', '').strip()
            print(outputs)  # Print model output for this image

            model_outputs[ann_id] = outputs
            pbar.update(1)
    pbar.close()

    with open(f"evaluation/DLC-Bench/model_outputs/{cache_name}.json", "w") as file:
        json.dump(model_outputs, file, indent=4, ensure_ascii=False)

    print(f"Cache name: {cache_name}")


if __name__ == "__main__":
    main()
