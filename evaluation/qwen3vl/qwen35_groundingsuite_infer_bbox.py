import argparse
import os
import torch
import torchvision
import tqdm
import re
from PIL import Image
import json

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from transformers import AutoProcessor


NO_THINK_PREFIX = "<think>\n\n</think>\n\n"


def parse_args():
    parser = argparse.ArgumentParser(description='GroundingSuite with BBox (Qwen3.5)')
    parser.add_argument(
        '--model_path',
        default="Qwen/Qwen3.5-VL-8B-Instruct",
        help='hf model path.')
    parser.add_argument(
        '--save_dir',
        default='./results/groundingsuite_bbox_qwen35/',
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


def bbox_to_pixel_coords(bbox, height, width, normalized_scale=1000):
    """
    将归一化的 bbox 坐标转换为像素坐标。

    Args:
        bbox: (x1, y1, x2, y2) 归一化坐标 (0-normalized_scale)
        height: 图像高度
        width: 图像宽度
        normalized_scale: 归一化尺度，默认 1000

    Returns:
        list: [x1, y1, x2, y2] 像素坐标
    """
    x1, y1, x2, y2 = bbox

    x1_pixel = int(x1 / normalized_scale * width)
    y1_pixel = int(y1 / normalized_scale * height)
    x2_pixel = int(x2 / normalized_scale * width)
    y2_pixel = int(y2 / normalized_scale * height)

    # 确保坐标在有效范围内
    x1_pixel = max(0, min(width - 1, x1_pixel))
    y1_pixel = max(0, min(height - 1, y1_pixel))
    x2_pixel = max(0, min(width, x2_pixel))
    y2_pixel = max(0, min(height, y2_pixel))

    return [x1_pixel, y1_pixel, x2_pixel, y2_pixel]


def main():
    args = parse_args()

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
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

    # BBox prompt template
    BBOX_PROMPT_TEMPLATE = """Please carefully check the image and detect the object this sentence describes: {caption}
Provide the bounding box in the format [x1, y1, x2, y2] where coordinates are normalized to [0, 1000]. If no matching object is found, output null."""

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

        # Check if result already exists
        if os.path.exists(f"{args.save_dir}/{item_idx}.json"):
            print(f"File {item_idx}.json exists, skipping...")
            continue

        question = NO_THINK_PREFIX + BBOX_PROMPT_TEMPLATE.format(caption=caption)

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
            # Convert normalized bbox to pixel coordinates
            pred_box = bbox_to_pixel_coords(bbox, ori_height, ori_width)

            prediction = {
                'idx': item_idx,
                'image_path': image_file,
                'box': pred_box,
                'predicted_box': pred_box,
                'class_id': class_id,
                'normalized_bbox': list(bbox),
                'raw_output': output_text[0]
            }

        with open(f"{args.save_dir}/{item_idx}.json", 'w') as f:
            json.dump(prediction, f)

    print(f"Finished processing {_end_ - _start_} samples.")


if __name__ == "__main__":
    main()
