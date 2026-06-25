"""
GroundingME Evaluation Script

A standalone script to evaluate vision-language models on the GroundingME benchmark.
Loads data from HuggingFace, calls models via OpenAI-compatible API, and computes evaluation metrics.

Usage:
    python evaluate.py --api-url <api_base_url> --api-key <your_key> --model-name <model_name> [--workers <num>] [--output <output_file>] [--limit <num_samples>]

Examples:
    # Local vLLM server (single worker)
    python evaluate.py --api-url http://localhost:8000/v1 --api-key dummy --model-name Qwen/Qwen3-VL-8B-Thinking
    
    # With concurrent workers (faster evaluation)
    python evaluate.py --api-url http://localhost:8000/v1 --api-key dummy --model-name Qwen/Qwen3-VL-8B-Thinking --workers 16
"""

import argparse
import os
import json
import glob
import pandas as pd
import re
import torch
import numpy as np
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pycocotools.mask as mask_util
from PIL import Image, ImageDraw, ImageColor
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from projects.vlm.vq_sam2.models import DirectResize
from projects.vlm.vq_sam2.models import VQ_SAM2, VQ_SAM2Config, SAM2Config


PROMPT_TEMPLATE = """All spatial relationships are defined from the viewer's perspective, where 'front' means closer to the viewer and 'back' means farther from the viewer. Please provide the bounding box coordinate of the object the following statement describes:
{description}
Ensure that all details mentioned about the object are accurate. Provide at most one bounding box. If a matching object is found, provide its bounding box as a JSON in the format {{"bbox_2d": [x1, y1, x2, y2]}}. If no matching object is found, output {{"bbox_2d": null}}."""

def apply_overlay_and_box(base_image, mask_np, box, color_hex="#00FF00", alpha=150):
    """
    将 Mask 以半透明叠加的方式贴到原图，并画上 BBox。
    不使用 plt，纯用 PIL 和 Numpy 实现。
    """
    # 1. 准备底图 (转换为 RGBA 以便进行 alpha 合成)
    base_rgba = base_image.convert("RGBA")
    H, W = base_rgba.size[1], base_rgba.size[0]

    # ensure mask is boolean
    mask_bool = mask_np > 0

    # 2. 创建 Mask 叠加层
    # 创建一个全透明的黑色图像 (H, W, 4)
    overlay_np = np.zeros((H, W, 4), dtype=np.uint8)

    # 解析颜色 (例如 #00FF00 -> (0, 255, 0))
    rgb_color = ImageColor.getrgb(color_hex)

    # 在 Mask 为 True 的区域填充颜色和透明度
    # 这里的 alpha 控制 Mask 的透明度 (0-255, 越大越不透明)
    overlay_np[mask_bool, 0] = rgb_color[0] # R
    overlay_np[mask_bool, 1] = rgb_color[1] # G
    overlay_np[mask_bool, 2] = rgb_color[2] # B
    overlay_np[mask_bool, 3] = alpha        # A

    # 将 numpy 数组转回 PIL 图像
    overlay_pil = Image.fromarray(overlay_np, "RGBA")

    # 3. 合成图像 (将半透明 Mask 盖在底图上)
    # alpha_composite 要求两张图都是 RGBA
    combined_img = Image.alpha_composite(base_rgba, overlay_pil)

    # 4. 绘制 Bounding Box
    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(combined_img)
    # 绘制矩形框，outline 是边框颜色，width 是线宽
    # input_box 格式直接就是 [x0, y0, x1, y1]，可以直接用
    draw.rectangle(box.tolist(), outline="red", width=5)

    # 5. 转回 RGB (方便保存为 jpg 等格式)
    final_result_rgb = combined_img.convert("RGB")

    return final_result_rgb

def concat_side_by_side(img_left, img_right):
    """
    将两张 PIL 图片横向并排拼接
    """
    w1, h1 = img_left.size
    w2, h2 = img_right.size
    
    # 1. 创建新画布：宽度相加，高度取最大值
    new_w = w1 + w2
    new_h = max(h1, h2)
    # 创建RGB模式的空白图
    new_img = Image.new('RGB', (new_w, new_h), color=(255, 255, 255)) 
    
    # 2. 粘贴图片
    new_img.paste(img_left, (0, 0))      # 第一张贴在左边 (0,0)
    new_img.paste(img_right, (w1, 0))    # 第二张贴在右边 (w1, 0)
    
    return new_img

def parse_bbox(text: str) -> List[float]:
    """Extract bounding box from model response."""
    try:
        match = re.search(r'\{.*"bbox_2d".*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            bbox = data["bbox_2d"]
            if bbox is None:
                return [0, 0, 0, 0]
            if isinstance(bbox, list) and len(bbox) == 4:
                return [float(coord) for coord in bbox]
    except:
        pass
    
    # Fallback: try to extract four numbers
    pattern = r"(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)"
    matches = re.findall(pattern, text)
    if matches:
        return [float(coord) for coord in matches[-1]]
    
    return [0, 0, 0, 0]


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes."""
    if box1 == [0, 0, 0, 0]:
        return 1 if box2 == [0, 0, 0, 0] else 0
    
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def normalize_bbox(bbox: List[float], width: int, height: int) -> List[float]:
    """Convert normalized or 0-999 range bbox to pixel coordinates."""
    if all(coord <= 1 for coord in bbox):
        return [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
    return [bbox[0] / 999 * width, bbox[1] / 999 * height, bbox[2] / 999 * width, bbox[3] / 999 * height]


def call_model(image, prompt: str, api_config: Dict[str, str]) -> str:
    """
    Call vision-language model via OpenAI-compatible API.
    
    Args:
        image: PIL Image object
        prompt: Text prompt
        api_config: Dictionary containing 'base_url', 'api_key', 'model_name'
    
    Returns:
        Model response text
    """
    import base64
    import io
    from openai import OpenAI
    
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Initialize client with custom base_url if provided
    client = OpenAI(
        api_key=api_config.get("api_key"),
        base_url=api_config.get("base_url")
    )
    
    response = client.chat.completions.create(
        model=api_config.get("model_name"),
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}},
                {"type": "text", "text": prompt}
            ]
        }],
        max_completion_tokens=20480,
        temperature=0
    )
    
    return response.choices[0].message.content

def local_inference(model, processor, image, prompt, device):
    """
    使用本地加载的模型进行推理
    修改版：不需要 qwen_vl_utils 库，直接处理 PIL 图片
    """
    # 1. 构建对话格式 (让 apply_chat_template 处理文本中的 placeholder)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image}, # 这里主要为了让 template 生成 <|image_pad|> 占位符
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 2. 生成包含特殊 token 的文本 prompt
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 3. 直接调用 processor
    # 我们知道 messages 里只有一张图，所以直接把 image 放入列表传给 images 参数
    inputs = processor(
        text=[text],
        images=[image], # 直接传 PIL Image 对象列表
        videos=None,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)

    # 4. 推理生成
    # max_new_tokens 不需要太大，因为只输出 bbox
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=128, 
        temperature=0.01, 
        do_sample=False
    )
    
    # 5. 截取新生成的 token (去掉 input 部分)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text


def evaluate_sample(sample: Dict[str, Any], api_config: Dict[str, str]) -> Dict[str, Any]:
    """Evaluate a single sample."""
    # Prepare inputs
    image = sample["image"].convert("RGB")
    prompt = PROMPT_TEMPLATE.format(description=sample["description"])
    
    # Get model prediction
    response = call_model(image, prompt, api_config)
    pred_bbox = parse_bbox(response)
    
    # Get ground truth
    gt_bbox = sample["bbox"] if sample["subtask_l1"] != "Rejection" else [0, 0, 0, 0]
    height, width = sample["height"], sample["width"]
    
    # Try different coordinate formats and pick the best one
    pred_candidates = [
        pred_bbox,
        normalize_bbox(pred_bbox, width, height),
    ]
    
    ious = [compute_iou(gt_bbox, pred) for pred in pred_candidates]
    best_idx = ious.index(max(ious))
    best_pred = pred_candidates[best_idx]
    best_iou = ious[best_idx]
    
    # Compute metrics
    acc_50 = float(best_iou >= 0.5)
    acc_75 = float(best_iou >= 0.75)
    acc_90 = float(best_iou >= 0.9)
    
    return {
        "id": sample["id"],
        "subtask_l1": sample["subtask_l1"],
        "subtask_l2": sample["subtask_l2"],
        "iou": best_iou,
        "acc_50": acc_50,
        "acc_75": acc_75,
        "acc_90": acc_90,
        "response": response,
        "pred_bbox": best_pred,
        "gt_bbox": gt_bbox,
    }

def fix_mt_format_comprehensive(text):
    """
    全面修正 <|mt_...> 格式的函数。
    它会处理以下几种情况：
    1. 标记太少 (1个): <|mt_start|><|mt_0198|><|mt_end|> -> <|mt_start|><|mt_0198|><|mt_-1|><|mt_end|>
    2. 标记太少 (1个, 无end): <|mt_start|><|mt_0198|> -> <|mt_start|><|mt_0198|><|mt_-1|><|mt_end|>
    3. 标记太多 (3个或以上): <|mt_start|><|mt_0186|><|mt_0410|><|mt_0186|><|mt_end|> -> <|mt_start|><|mt_0186|><|mt_0410|><|mt_end|>
    4. 正确格式: <|mt_start|><|mt_0044|><|mt_0442|><|mt_end|> -> 不变
    """
    # 规则 1: 处理标记太多的情况 (3个或以上)
    # 捕获前两个，匹配掉多余的，然后用前两个重构
    pattern_too_many = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_\d+\|>)(?:<\|mt_\d+\|>)+<\|mt_end\|>'
    replacement_too_many = r'\1\2\3<|mt_end|>'
    text = re.sub(pattern_too_many, replacement_too_many, text)
    # 规则 2: 处理标记太少的情况 (只有1个，且有<|mt_end|>)
    pattern_too_few_with_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)'
    replacement_too_few = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_with_end, replacement_too_few, text)
    # 规则 3: 处理标记太少的情况 (只有1个，且没有<|mt_end|>)
    # 使用负向前瞻确保后面不是另一个mt_token
    pattern_too_few_no_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)'
    replacement_too_few_no_end = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_no_end, replacement_too_few_no_end, text)
    return text

def mask_to_box_np(mask):
    """
    mask: np.ndarray of shape (H, W), binary (0/1 or bool)
    return: [x_min, y_min, x_max, y_max]
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None  # 或 [0,0,0,0]

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    return np.array([x_min, y_min, x_max, y_max])


def evaluate_sample_local(sample, model, processor, vq_sam2, sam2_processor, device):
    """单个样本评估"""
    image = sample["image"].convert("RGB")
    prompt = PROMPT_TEMPLATE.format(description=sample["description"])
    
    # 调用本地推理
    response = local_inference(model, processor, image, prompt, device)
    # print(response)
    quant_ids = extract_mt_token_ids_v1(response)

    codebook_depth = vq_sam2.config.codebook_depth
    codebook_size = vq_sam2.config.codebook_size
    if len(quant_ids) % codebook_depth != 0:
        print("FORMAT ERROR: ", output_text)
        output_text = [fix_mt_format_comprehensive(output_text[0])]
        print("FIXED OUTPUT TEXT: ", output_text)
        quant_ids = extract_mt_token_ids_v2(output_text[0])

    batch_size = len(quant_ids) // codebook_depth
    remap_quant_ids = []
    tags = []
    for bs_id in range(batch_size):
        chunk_quant_ids = quant_ids[bs_id*codebook_depth:(bs_id+1)*codebook_depth]
        tags.append(f"{chunk_quant_ids[0]}-{chunk_quant_ids[1]}")
        remap_chunk_quant_ids = [quant_id - book_id*codebook_size for book_id, quant_id in enumerate(chunk_quant_ids)]
        code1 = remap_chunk_quant_ids[0]
        code2 = remap_chunk_quant_ids[1]
        if not (code2 >= 0 and code2 < codebook_size):
            code2 = -1
        remap_chunk_quant_ids_error_handle = [code1, code2]
        remap_quant_ids.append(remap_chunk_quant_ids_error_handle)

    batch_size = len(remap_quant_ids)
    sam2_image = np.array(image)
    sam2_image = sam2_processor.apply_image(sam2_image)
    sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
    sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
    sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

    quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

    with torch.no_grad():
        _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)

    gt_bbox = sample["bbox"] if sample["subtask_l1"] != "Rejection" else [0, 0, 0, 0]
    height, width = sample["height"], sample["width"]

    _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(height, width), mode='bilinear')
    _pred_masks = _pred_masks > 0.5
    _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)

    assert _pred_masks.shape[0] == 1, f"currently support batch_size=1, but got {_pred_masks.shape[0]}"
    _pred_masks = _pred_masks[0]

    fortran_mask = np.asfortranarray(_pred_masks)
    rle_dict = mask_util.encode(fortran_mask)
    rle_string = rle_dict['counts'].decode('utf-8')
    entity_rle = {'size': [int(x) for x in rle_dict['size']], 'counts': rle_string}    

    pred_bbox = mask_to_box_np(_pred_masks).tolist()
    
    # result_image = apply_overlay_and_box(image, _pred_masks, pred_bbox, color_hex="#33FF33", alpha=160)
    # comparison_image = concat_side_by_side(image, result_image)
    # output_path = f"samtok_result_overlay.jpg"
    # comparison_image.save(output_path)
    # print(sample["description"])
    # return pred_bbox

    pred_candidates = [pred_bbox, normalize_bbox(pred_bbox, width, height)]
    ious = [compute_iou(gt_bbox, pred) for pred in pred_candidates]
    best_idx = ious.index(max(ious))
    best_pred = pred_candidates[best_idx]
    best_iou = ious[best_idx]
    
    # Compute metrics
    acc_50 = float(best_iou >= 0.5)
    acc_75 = float(best_iou >= 0.75)
    acc_90 = float(best_iou >= 0.9)
    
    return {
        "id": sample["id"],
        "subtask_l1": sample["subtask_l1"],
        "subtask_l2": sample["subtask_l2"],
        "iou": best_iou,
        "acc_50": acc_50,
        "acc_75": acc_75,
        "acc_90": acc_90,
        "response": response,
        "pred_bbox": best_pred,
        "gt_bbox": gt_bbox,
        "pred_mask": entity_rle,
    }

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate evaluation results and compute overall metrics."""
    if not results:
        return {}
    
    # Overall metrics
    metrics = {
        "IoU": sum(r["iou"] for r in results) / len(results),
        "ACC@0.5": sum(r["acc_50"] for r in results) / len(results),
        "ACC@0.75": sum(r["acc_75"] for r in results) / len(results),
        "ACC@0.9": sum(r["acc_90"] for r in results) / len(results),
    }
    
    # Per-category metrics (subtask_l1)
    categories_l1 = {}
    for result in results:
        cat = result["subtask_l1"]
        if cat not in categories_l1:
            categories_l1[cat] = []
        categories_l1[cat].append(result)
    
    for cat, cat_results in categories_l1.items():
        metrics[f"{cat}_ACC@0.5"] = sum(r["acc_50"] for r in cat_results) / len(cat_results)
        metrics[f"{cat}_ACC@0.75"] = sum(r["acc_75"] for r in cat_results) / len(cat_results)
        metrics[f"{cat}_ACC@0.9"] = sum(r["acc_90"] for r in cat_results) / len(cat_results)
    
    # Per-subcategory metrics (subtask_l2)
    categories_l2 = {}
    for result in results:
        if result["subtask_l2"]:
            cat = f"{result['subtask_l1']}_{result['subtask_l2']}"
            if cat not in categories_l2:
                categories_l2[cat] = []
            categories_l2[cat].append(result)
    
    for cat, cat_results in categories_l2.items():
        metrics[f"{cat}_ACC@0.5"] = sum(r["acc_50"] for r in cat_results) / len(cat_results)
    
    return metrics


def evaluate_sample_wrapper(args):
    """Wrapper function for parallel execution."""
    sample, api_config = args
    try:
        return evaluate_sample(sample, api_config), None
    except Exception as e:
        return None, (sample.get('id', 'unknown'), str(e))

def extract_mt_token_ids_v1(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]

def extract_mt_token_ids_v2(text):
    pattern = re.compile(r'<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>')
    matches = pattern.findall(text)
    ret_list = []
    for num1, num2 in matches:
        ret_list.append(int(num1))
        ret_list.append(int(num2))
    return ret_list


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on GroundingME benchmark")
    # parser.add_argument("--api-url", type=str, required=True, help="API base URL")
    # parser.add_argument("--api-key", type=str, required=True, help="API key for authentication")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-4B-SAMTok", help="Model name")
    parser.add_argument("--dataset", type=str, default="data/GroundingME", help="HuggingFace dataset path")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--output", type=str, default="results_groundingme_dual.json", help="Output file for results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 这里以 Qwen2-VL 为例，如果是其他模型请替换对应的类
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # min_pixels/max_pixels 视显存情况调整
    processor = AutoProcessor.from_pretrained(args.model_name, min_pixels=256*28*28, max_pixels=1280*28*28)

    # build vq-sam2 model
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    sam2_config = SAM2Config(
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
    state = torch.load(os.path.join(args.model_name, "mask_tokenizer_256x2.pth"), map_location="cpu")
    vq_sam2.load_state_dict(state)

    sam2_image_processor = DirectResize(1024)

    print(f"Loading dataset: {args.dataset} (split: {args.split})")
    dataset = load_dataset(args.dataset, split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    
    print(f"Evaluating {len(dataset)} samples")
    # print(f"  API URL: {args.api_url}")
    print(f"  Model: {args.model_name}")
    
    results = []
    errors = []
    
    for sample in tqdm(dataset, desc="Evaluating"):
        try:
            result = evaluate_sample_local(sample, model, processor, vq_sam2, sam2_image_processor, device)
            results.append(result)
        except Exception as e:
            error_msg = f"Sample {sample.get('id', 'unknown')}: {e}"
            errors.append(error_msg)
            print(f"\nError: {error_msg}")

    # Compute aggregate metrics
    metrics = aggregate_results(results)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples: {len(dataset)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(errors)}")
    print("-"*50)
    for metric, value in sorted(metrics.items()):
        print(f"{metric:40s}: {value:.4f}")
    
    # Save detailed results
    output_data = {
        "model": args.model_name,
        "dataset": args.dataset,
        "split": args.split,
        "total_samples": len(dataset),
        "successful_samples": len(results),
        "failed_samples": len(errors),
        "metrics": metrics,
        "detailed_results": results,
        "errors": errors if errors else None,
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {args.output}")
    if errors:
        print(f"{len(errors)} samples failed - see 'errors' field in output file")


if __name__ == "__main__":
    main()

