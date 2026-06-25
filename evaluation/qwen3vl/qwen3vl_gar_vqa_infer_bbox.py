import argparse
import copy
import os
import torch
import torchvision
import tqdm
from pycocotools import mask as mask_utils
import numpy as np
import re
from PIL import Image
import json
import base64

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description='GAR VQA with BBox')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument('--output', help='output file name')
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


def mask_to_normalized_bbox(mask, normalized_scale=1000):
    """
    从二值 mask 计算归一化的 bbox。
    
    Args:
        mask: np.ndarray, 形状为 (H, W) 的二值 mask
        normalized_scale: 归一化尺度，默认 1000
    
    Returns:
        str: "[x1, y1, x2, y2]" 格式的字符串，坐标归一化到 [0, normalized_scale]
    """
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask.copy())).unsqueeze(0)
    
    # 获取 bbox: [x1, y1, x2, y2] in pixel coordinates
    try:
        bbox = torchvision.ops.masks_to_boxes(mask_tensor)[0]  # (4,)
    except:
        # 如果 mask 为空，返回 [0, 0, 0, 0]
        return "[0, 0, 0, 0]"
    
    h, w = mask.shape
    x1, y1, x2, y2 = bbox.numpy()
    
    # 归一化到 [0, normalized_scale]
    x1_norm = int(x1 / w * normalized_scale)
    y1_norm = int(y1 / h * normalized_scale)
    x2_norm = int(x2 / w * normalized_scale)
    y2_norm = int(y2 / h * normalized_scale)
    
    return f"[{x1_norm}, {y1_norm}, {x2_norm}, {y2_norm}]"


def main():
    args = parse_args()

    # build qwen3vl model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    with open('../Grasp-Any-Region/evaluation/GAR-Bench/annotations/GAR-Bench-VQA.json', 'r') as f:
        eval_samples = json.load(f)
    
    all_items = []
    for eval_sample in tqdm.tqdm(eval_samples):
        image_file = eval_sample['image']
        image_path = os.path.join('../Grasp-Any-Region/evaluation/GAR-Bench/annotations/', image_file)
        seg = []
        for mask_idx, mask_rle in enumerate(eval_sample["mask_rles"]):
            seg.append(mask_rle)

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        # 解码 mask 并转换为 bbox
        binary_masks = decode_mask(seg, ori_height, ori_width)
        
        # 将每个 mask 转换为归一化的 bbox 字符串
        region_bbox_strs = []
        for mask in binary_masks:
            bbox_str = mask_to_normalized_bbox(mask)
            region_bbox_strs.append(bbox_str)
        
        with open(image_path, "rb") as f:
            global_b64 = base64.b64encode(f.read()).decode()

        question_str = f"Question: {eval_sample['question']}\nOptions:"
        for op in eval_sample["choices"]:
            question_str += f"\n{op}"
        question_str += "\nAnswer with the correct option's letter directly."

        # Replace <Prompti> in question_str with region_bbox_strs[i]
        def replace_prompti(match):
            idx = int(match.group(1))
            if idx < len(region_bbox_strs):
                # 使用 bbox 格式: "the region at bounding box [x1, y1, x2, y2]"
                return f"the region at bounding box {region_bbox_strs[idx]}"
            return match.group(0)
        
        question_str = re.sub(r"<Prompt(\d+)>", replace_prompti, question_str)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{global_b64}",
                    },
                    {"type": "text", "text": question_str},
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
            top_p=1.0,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("Assistant: ", output_text)
        response = output_text[0].replace('<|im_end|>', '')

        save_item = copy.deepcopy(eval_sample)
        save_item.update({
            'model_output': response,
            'region_bboxes': region_bbox_strs  # 保存使用的 bbox
        })
        all_items.append(save_item)

    print(len(all_items), " items")
    
    with open(args.output, 'w') as f:
        json.dump(all_items, f, indent=4)


if __name__ == '__main__':     
    main()
