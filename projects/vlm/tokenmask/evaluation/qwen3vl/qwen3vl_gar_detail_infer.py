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
import hydra
import base64
import io

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


def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--vq_sam2_path',
        default="pretrained_weights/iter_175473.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--dataset',
        default='./data/PaDT-MLLM/RefCOCO/refcoco_val.json',
        help='Specify a ref dataset')
    parser.add_argument('--task_id', '--task-id', type=int, default=0)
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

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    # build qwen25vl model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    # build vq-sam2 model
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    with hydra.initialize(version_base=None, config_path="../../../../../projects/transformers/vq_sam2/sam2/sam2_configs"):
        sam2_config = SAM2Config(
            cfg_path="sam2.1_hiera_l.yaml",
            ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
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

    with open('./data/Grasp-Any-Region/evaluation/GAR-Bench/annotations/GAR-Bench-Caption-Detailed.json', 'r') as f:
        eval_samples = json.load(f)
    
    all_items = []
    for eval_sample in eval_samples:
        image_file = eval_sample['image']
        image_path = os.path.join('./data/Grasp-Any-Region/evaluation/GAR-Bench/annotations/', image_file)
        segm1 = eval_sample['mask_rles'][0]
        segm2 = eval_sample['mask_rles'][1]
        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        binary_masks = decode_mask([segm1, segm2], ori_height, ori_width)
        masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])
        boxes = torchvision.ops.masks_to_boxes(masks)
        whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
        boxes = boxes / whwh
        boxes = boxes.to(vq_sam2.device)
        masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        with torch.no_grad():
            vq_sam2_output = vq_sam2(
                sam2_pixel_values.repeat(len(masks), 1, 1, 1),
                masks,
                boxes,
                reconstruct_mask=False,
            )
            quant_codes = vq_sam2_output.quant_codes.squeeze().cpu().numpy().astype(np.int32).tolist()
        
        region1_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(quant_codes[0])]
        region2_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(quant_codes[1])]
        region1_quant_codes_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in region1_quant_codes]) + MT_END_TOKEN
        region2_quant_codes_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in region2_quant_codes]) + MT_END_TOKEN

        # # region1 zoom in
        # zoom_in_quant_codes_str_list = []
        # zoom_in_images = []
        # for box_id in range(len(boxes)):
        #     x1, y1, x2, y2 = boxes[box_id].cpu().numpy().tolist()
        #     bbox_w = x2 - x1
        #     bbox_h = y2 - y1
        #     if bbox_w < 140:
        #         x1 = x1 - (140 - bbox_w) // 2
        #         x2 = x2 + (140 - bbox_w) // 2
        #     if bbox_h < 140:
        #         y1 = y1 - (140 - bbox_h) // 2
        #         y2 = y2 + (140 - bbox_h) // 2
        #     x1 = int(max(0, x1))
        #     x2 = int(min(ori_width, x2))
        #     y1 = int(max(0, y1))
        #     y2 = int(min(ori_height, y2))

        #     cropped_image = image.crop((x1, y1, x2, y2))
        #     crop_width, crop_height = cropped_image.size

        #     if crop_width > crop_height and crop_width < 280:
        #         ratio = 280 / crop_height
        #         new_height = 280
        #         new_width = int(crop_width * ratio)
        #         resized_crop_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        #     elif crop_height > crop_width and crop_height < 280:
        #         ratio = 280 / crop_width
        #         new_width = 280
        #         new_height = int(crop_height * ratio)
        #         resized_crop_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        #     elif crop_height == crop_width and crop_width < 280:
        #         ratio = 280 / crop_height
        #         new_height = 280
        #         new_width = int(crop_width * ratio)
        #         resized_crop_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        #     else:
        #         new_height = new_width = None
        #         resized_crop_image = None

        #     if resized_crop_image is None:
        #         cropped_sam2_image = np.array(cropped_image)
        #         cropped_sam2_image = sam2_image_processor.apply_image(cropped_sam2_image)
        #         cropped_sam2_pixel_values = torch.from_numpy(cropped_sam2_image).permute(2, 0, 1).contiguous()
        #         cropped_sam2_pixel_values = cropped_sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
        #     else:
        #         cropped_sam2_image = np.array(resized_crop_image)
        #         cropped_sam2_image = sam2_image_processor.apply_image(cropped_sam2_image)
        #         cropped_sam2_pixel_values = torch.from_numpy(cropped_sam2_image).permute(2, 0, 1).contiguous()
        #         cropped_sam2_pixel_values = cropped_sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        #     cropped_masks = torch.stack([torch.from_numpy(np.ascontiguousarray(binary_masks[box_id].copy()[y1:y2, x1:x2]))])
        #     assert cropped_masks.shape[-2] == crop_height and cropped_masks.shape[-1] == crop_width

        #     if resized_crop_image is not None:
        #         resized_crop_masks = torch.nn.functional.interpolate(cropped_masks.unsqueeze(0), size=(new_height, new_width), mode='bilinear')
        #         resized_crop_masks = resized_crop_masks[0] > 0.5
        #         cropped_masks = resized_crop_masks
        #     crop_height, crop_width = cropped_masks.shape[-2:]
        #     cropped_boxes = torchvision.ops.masks_to_boxes(cropped_masks)
        #     crop_whwh = torch.as_tensor([[crop_width, crop_height, crop_width, crop_height]])
        #     cropped_boxes = cropped_boxes / crop_whwh
        #     cropped_boxes = cropped_boxes.to(vq_sam2.device)
        #     cropped_masks = [m.unsqueeze(0).to(vq_sam2.device) for m in cropped_masks]

        #     with torch.no_grad():
        #         cropped_vq_sam2_output = vq_sam2(
        #             cropped_sam2_pixel_values,
        #             cropped_masks,
        #             cropped_boxes,
        #             reconstruct_mask=True,
        #         )
            
        #     crop_quant_codes = cropped_vq_sam2_output.quant_codes.squeeze().detach().cpu().numpy().astype(np.int32).tolist()
        #     remap_crop_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(crop_quant_codes)]
        #     crop_quant_codes_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in remap_crop_quant_codes]) + MT_END_TOKEN
        #     zoom_in_quant_codes_str_list.append(crop_quant_codes_str)
            
        #     buffer = io.BytesIO()
        #     if resized_crop_image is None:
        #         cropped_image.save(buffer, format='JPEG')
        #     else:
        #         resized_crop_image.save(buffer, format='JPEG')
        #     buffer.seek(0)
        #     b64 = base64.b64encode(buffer.read()).decode("utf-8")
        #     zoom_in_images.append(b64)
        
        with open(image_path, "rb") as f:
            global_b64 = base64.b64encode(f.read()).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{global_b64}",
                    },
                    {"type": "text", "text": f"1. Describe {region1_quant_codes_str} in detail 2. Describe the relationship between {region1_quant_codes_str} and {region2_quant_codes_str}."},
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
            do_sample=False,  # 关闭采样，使用贪婪解码
            top_p=1.0,  # 配合do_sample=False使用
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("Assistant: ", output_text)
        response = output_text[0].replace('<|im_end|>', '')

        save_item = copy.deepcopy(eval_sample)
        save_item.update({'model_output': response})
        all_items.append(save_item)

    print(len(all_items), " items")
    
    with open('./result_qwen3vl_4b_gar_detail.json', 'w') as f:
        json.dump(all_items, f, indent=4)

if __name__ == '__main__':     
    main()
