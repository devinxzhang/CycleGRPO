import argparse
import os
import torch
import torchvision
import tqdm
import re
from PIL import Image
import json

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import smart_resize


def parse_args():
    parser = argparse.ArgumentParser(description='GroundingSuite with BBox (Qwen2.5-VL)')
    parser.add_argument(
        '--model_path',
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help='hf model path.')
    parser.add_argument(
        '--save_dir',
        default='./results/groundingsuite_bbox_qwen25vl/',
        help='save path')
    parser.add_argument(
        '--dataset',
        default='./data/GroundingSuiteEval/GroundingSuite-Eval.jsonl',
        help='Specify a ref dataset')
    parser.add_argument('--task_id', '--task-id', type=int, default=0)
    parser.add_argument('--num_tasks', '--num-tasks', type=int, default=1)
    args = parser.parse_args()
    return args


def extract_bbox_from_response(response_str: str):
    """
    从模型响应字符串中提取 bbox 坐标。

    Args:
        response_str: 模型生成的响应字符串，可能包含 [x1, y1, x2, y2] 格式的 bbox

    Returns:
        tuple: (x1, y1, x2, y2) 归一化坐标 (0-1000)，如果未找到则返回 None
    """
    # 匹配 [x1, y1, x2, y2] 格式，支持可选的空格
    bbox_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    match = re.search(bbox_pattern, response_str)

    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return (x1, y1, x2, y2)

    return None


def bbox_to_pixel_coords(bbox, ori_height, ori_width, resized_height, resized_width):
    """
    Qwen2.5-VL 原生输出的是 smart_resize 之后图像的绝对像素坐标，
    需要根据 resized -> original 的尺度比例缩放回原图坐标。

    Args:
        bbox: (x1, y1, x2, y2) Qwen2.5-VL 输出的 bbox（resized 图像绝对像素坐标）
        ori_height: 原图高度
        ori_width: 原图宽度
        resized_height: smart_resize 后的高度
        resized_width: smart_resize 后的宽度

    Returns:
        list: [x1, y1, x2, y2] 原图像素坐标
    """
    x1, y1, x2, y2 = bbox

    x_scale = ori_width / resized_width
    y_scale = ori_height / resized_height

    x1_pixel = int(x1 * x_scale)
    y1_pixel = int(y1 * y_scale)
    x2_pixel = int(x2 * x_scale)
    y2_pixel = int(y2 * y_scale)

    x1_pixel = max(0, min(ori_width - 1, x1_pixel))
    y1_pixel = max(0, min(ori_height - 1, y1_pixel))
    x2_pixel = max(0, min(ori_width, x2_pixel))
    y2_pixel = max(0, min(ori_height, y2_pixel))

    return [x1_pixel, y1_pixel, x2_pixel, y2_pixel]


def main():
    args = parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    all_data_dict = []
    with open(args.dataset, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            all_data_dict.append(obj)

    rows = len(all_data_dict)
    chunk_size = (rows + args.num_tasks - 1) // args.num_tasks
    _start_ = args.task_id * chunk_size
    _end_ = min(_start_ + chunk_size, rows)

    # Qwen2.5-VL 的原生 grounding prompt：输出 resized 图像的绝对像素坐标 JSON
    BBOX_PROMPT_TEMPLATE = """Outline the position of {caption} and output the bounding box coordinates in JSON format."""

    image_processor = processor.image_processor
    ip_min_pixels = getattr(image_processor, 'min_pixels', None)
    ip_max_pixels = getattr(image_processor, 'max_pixels', None)
    ip_patch_size = image_processor.patch_size
    ip_merge_size = image_processor.merge_size

    for data_dict in tqdm.tqdm(all_data_dict[_start_:_end_]):
        image_file = data_dict['image_path']
        item_idx = data_dict['idx']
        label = data_dict['label']
        caption = data_dict['caption']
        class_id = data_dict['class_id']

        image_path = os.path.join('./data/GroundingSuiteEval', image_file)
        # Replace path for coco images
        image_path = image_path.replace('./data/ref_seg/grefs/coco2014/train2014', '<PATH_TO_COCO2014>/train2014')

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        # Qwen2.5-VL 的 bbox 输出在 smart_resize 后的像素坐标空间
        resized_height, resized_width = smart_resize(
            ori_height,
            ori_width,
            factor=ip_patch_size * ip_merge_size,
            min_pixels=ip_min_pixels,
            max_pixels=ip_max_pixels,
        )

        # Check if result already exists
        if os.path.exists(f"{args.save_dir}/{item_idx}.json"):
            print(f"File {item_idx}.json exists, skipping...")
            continue

        question = BBOX_PROMPT_TEMPLATE.format(caption=caption)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
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
            max_new_tokens=256,
            do_sample=False,
            top_p=1.0,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(f"Caption: {caption}")
        print(f"Assistant: {output_text[0]}")

        # Extract bbox from output
        bbox = extract_bbox_from_response(output_text[0])

        if bbox is None:
            prediction = {
                'idx': item_idx,
                'image_path': image_file,
                'box': [0, 0, 0, 0],
                'predicted_box': [0, 0, 0, 0],
                'class_id': class_id,
                'raw_output': output_text[0]
            }
        else:
            # Qwen2.5-VL 原生输出 resized 图像的绝对像素坐标，需要缩放回原图
            pred_box = bbox_to_pixel_coords(bbox, ori_height, ori_width, resized_height, resized_width)

            prediction = {
                'idx': item_idx,
                'image_path': image_file,
                'box': pred_box,
                'predicted_box': pred_box,
                'class_id': class_id,
                'resized_bbox': list(bbox),
                'resized_size': [resized_height, resized_width],
                'raw_output': output_text[0]
            }

        with open(f"{args.save_dir}/{item_idx}.json", 'w') as f:
            json.dump(prediction, f)

    print(f"Finished processing {_end_ - _start_} samples.")


if __name__ == "__main__":
    main()
