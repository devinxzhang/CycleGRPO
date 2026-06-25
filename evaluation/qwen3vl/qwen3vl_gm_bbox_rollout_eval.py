import argparse
import base64
import os
import io
import torch
import torchvision
import numpy as np

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from pycocotools import mask as mask_utils

from PIL import Image, ImageDraw
import re
import json
import pyarrow.parquet as pq

from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description='GroundingME BBox Rollout Evaluation')
    parser.add_argument(
        '--model_path',
        default="Qwen/Qwen3-VL-4B",
        help='hf model path.')
    parser.add_argument(
        '--save_dir',
        default='./results/gm_bbox_rollout/',
        help='save path')
    parser.add_argument(
        '--index',
        type=int,
        nargs='+',
        default=[0],
        help='Sample indices to evaluate (list of int)')
    parser.add_argument(
        '--cap_rollouts',
        type=int,
        default=6,
        help='Number of caption rollouts')
    parser.add_argument(
        '--seg_rollouts',
        type=int,
        default=6,
        help='Number of segmentation rollouts per caption')
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature')
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p sampling')
    parser.add_argument(
        '--zoom_threshold',
        type=float,
        default=0.3,
        help='Threshold for zoom-in (bbox area ratio)')
    args = parser.parse_args()
    return args


IMAGE_FOLDER = 'rl_dataset/groundingme_train.parquet'


def clean_caption(text):
    """Clean caption: remove special tokens and formatting."""
    text = text.replace('<|im_end|>', '')
    text = text.replace('<think>\n\n</think>\n\n', '')
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<answer>', '').replace('</answer>', '')
    text = text.strip()
    return text


def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.5):
    """Overlay a binary mask on an image with specified color and transparency."""
    image_np = np.array(image).copy()
    mask_np = mask.astype(bool)
    result = image_np.copy()
    result[mask_np] = (alpha * np.array(color) + (1 - alpha) * image_np[mask_np]).astype(np.uint8)
    return Image.fromarray(result)


def mask_to_bbox(mask):
    """Convert binary mask to bounding box [x1, y1, x2, y2]."""
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask.copy())).unsqueeze(0)
    boxes = torchvision.ops.masks_to_boxes(mask_tensor)
    return boxes.squeeze().cpu().numpy().tolist()


def normalize_bbox(bbox, width, height, scale=1000):
    """Normalize bbox to [0, scale] range."""
    x1, y1, x2, y2 = bbox
    norm_x1 = int(x1 / width * scale)
    norm_y1 = int(y1 / height * scale)
    norm_x2 = int(x2 / width * scale)
    norm_y2 = int(y2 / height * scale)
    return [norm_x1, norm_y1, norm_x2, norm_y2]


def main():
    args = parse_args()
    
    # Log collection
    log_lines = []
    
    def log(msg):
        """Print and collect log message."""
        print(msg)
        log_lines.append(msg)

    log(f"Loading model from {args.model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    # Load dataset
    log(f"Loading dataset from {IMAGE_FOLDER}...")
    table = pq.read_table(IMAGE_FOLDER)

    # Process each index
    for sample_index in args.index:
        evaluate_sample(args, model, processor, table, sample_index, log_lines)


def evaluate_sample(args, model, processor, table, sample_index, log_lines):
    """Evaluate a single sample."""
    def log(msg):
        """Print and collect log message."""
        print(msg)
        log_lines.append(msg)

    row = {col: table[col][sample_index].as_py() for col in table.column_names}

    # Get image
    images = []
    for img_info in row['images']:
        image_bytes = img_info['bytes']
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        images.append(img)
    
    w, h = images[0].size
    gt_mask = row['masks']
    gt_mask_decoded = mask_utils.decode(gt_mask)

    # Extract bbox from GT mask
    bbox = mask_to_bbox(gt_mask_decoded)
    x1, y1, x2, y2 = bbox
    norm_bbox = normalize_bbox(bbox, w, h)
    bbox_str = f"[{norm_bbox[0]}, {norm_bbox[1]}, {norm_bbox[2]}, {norm_bbox[3]}]"

    # Calculate bbox area ratio for zoom-in decision
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    bbox_area = bbox_w * bbox_h
    image_area = w * h
    bbox_ratio = bbox_area / image_area

    # Create output directory
    sample_dir = os.path.join(args.save_dir, str(sample_index))
    result_path = os.path.join(sample_dir, 'rollout_results.json')
    
    os.makedirs(sample_dir, exist_ok=True)

    # Save original image
    images[0].save(os.path.join(sample_dir, 'original.png'))

    # Save GT mask overlay
    gt_overlay = overlay_mask_on_image(images[0], gt_mask_decoded, color=(0, 255, 0), alpha=0.5)
    gt_overlay.save(os.path.join(sample_dir, 'gt_mask_overlay.png'))

    def image_to_base64(img):
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def run_inference(messages, do_sample=True):
        """Run model inference and return output text."""
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        return output_text[0].replace('<think>\n\n</think>\n\n', '')

    def extract_bbox_from_text(text):
        """Extract bbox coordinates [x1, y1, x2, y2] from text."""
        # Try to find bbox pattern like [123, 456, 789, 012]
        pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        match = re.search(pattern, text)
        if match:
            return [int(match.group(i)) for i in range(1, 5)]
        return None

    def decode_bbox(seg_pred):
        """Decode bbox prediction, return normalized bbox coords."""
        bbox_coords = extract_bbox_from_text(seg_pred)
        if bbox_coords is None:
            return None, "no_bbox_found"
        
        norm_x1, norm_y1, norm_x2, norm_y2 = bbox_coords
        
        # Validate bbox
        if norm_x2 > norm_x1 and norm_y2 > norm_y1:
            return [norm_x1, norm_y1, norm_x2, norm_y2], "success"
        else:
            return None, "invalid_bbox"
    
    def draw_bbox_on_image(img, bbox_coords, color=(0, 0, 255), width=3):
        """Draw bbox rectangle on image."""
        if bbox_coords is None:
            return None
        norm_x1, norm_y1, norm_x2, norm_y2 = bbox_coords
        # Convert normalized [0, 1000] coordinates to pixel coordinates
        px1 = int(norm_x1 / 1000 * w)
        py1 = int(norm_y1 / 1000 * h)
        px2 = int(norm_x2 / 1000 * w)
        py2 = int(norm_y2 / 1000 * h)
        # Clamp to image bounds
        px1 = max(0, min(w, px1))
        py1 = max(0, min(h, py1))
        px2 = max(0, min(w, px2))
        py2 = max(0, min(h, py2))
        # Draw rectangle on image
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle([px1, py1, px2, py2], outline=color, width=width)
        return img_copy

    def compute_bbox_iou(pred_bbox, gt_bbox):
        """Compute IoU between predicted and ground truth bboxes (normalized coords)."""
        if pred_bbox is None:
            return 0.0
        
        # pred_bbox and gt_bbox are [x1, y1, x2, y2] in normalized [0, 1000] coords
        px1, py1, px2, py2 = pred_bbox
        gx1, gy1, gx2, gy2 = gt_bbox
        
        # Compute intersection
        inter_x1 = max(px1, gx1)
        inter_y1 = max(py1, gy1)
        inter_x2 = min(px2, gx2)
        inter_y2 = min(py2, gy2)
        
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        # Compute union
        pred_area = (px2 - px1) * (py2 - py1)
        gt_area = (gx2 - gx1) * (gy2 - gy1)
        union_area = pred_area + gt_area - inter_area
        
        return inter_area / (union_area + 1e-8)

    # Build captioning messages with bbox prompt
    img_base64 = image_to_base64(images[0])
    
    # Prepare caption prompt based on bbox ratio (zoom-in for small regions)
    if bbox_ratio < args.zoom_threshold:
        # Zoom-in logic for small regions
        crop_x1, crop_y1, crop_x2, crop_y2 = x1, y1, x2, y2
        
        # Expand bbox if too small
        if bbox_w < 140:
            crop_x1 = crop_x1 - (140 - bbox_w) // 2
            crop_x2 = crop_x2 + (140 - bbox_w) // 2
        if bbox_h < 140:
            crop_y1 = crop_y1 - (140 - bbox_h) // 2
            crop_y2 = crop_y2 + (140 - bbox_h) // 2
        
        crop_x1 = int(max(0, crop_x1))
        crop_x2 = int(min(w, crop_x2))
        crop_y1 = int(max(0, crop_y1))
        crop_y2 = int(min(h, crop_y2))
        
        cropped_image = images[0].crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crop_width, crop_height = cropped_image.size
        
        # Resize if too small
        resized_crop_image = None
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
        
        crop_to_encode = resized_crop_image if resized_crop_image is not None else cropped_image
        crop_base64 = image_to_base64(crop_to_encode)
        
        # Save cropped image
        crop_to_encode.save(os.path.join(sample_dir, 'crop_zoom.png'))
        
        cap_question = f"Given a detailed description of the region at bounding box {bbox_str}. Zoom in with the perspective as"
        
        cap_content = [
            {
                "type": "image",
                "image": f"data:image/png;base64,{img_base64}",
            },
            {"type": "text", "text": cap_question},
            {
                "type": "image",
                "image": f"data:image/png;base64,{crop_base64}",
            },
            {"type": "text", "text": ", give a detailed description of this cropped region."},
        ]
        use_zoom = True
    else:
        # Use global image with bbox
        cap_question = f"Given a detailed description of the region at bounding box {bbox_str}."
        cap_content = [
            {
                "type": "image",
                "image": f"data:image/png;base64,{img_base64}",
            },
            {"type": "text", "text": cap_question},
        ]
        use_zoom = False

    cap_messages = [{"role": "user", "content": cap_content}]

    # GT bbox in normalized coordinates for IoU computation
    gt_bbox_norm = norm_bbox  # [0, 1000] normalized

    # Results storage
    all_results = {
        "index": sample_index,
        "gt_bbox": bbox,
        "gt_bbox_norm": gt_bbox_norm,
        "bbox_str": bbox_str,
        "bbox_ratio": float(bbox_ratio),
        "use_zoom": use_zoom,
        "cap_rollouts": args.cap_rollouts,
        "seg_rollouts": args.seg_rollouts,
        "captions": []
    }

    log(f"\n{'='*60}")
    log(f"Evaluating sample {sample_index} (BBox: {bbox_str}, ratio: {bbox_ratio:.3f}, zoom: {use_zoom})")
    log(f"Caption rollouts: {args.cap_rollouts}, Seg rollouts per caption: {args.seg_rollouts}")
    log(f"{'='*60}")

    for cap_idx in range(args.cap_rollouts):
        log(f"\n[Caption {cap_idx+1}/{args.cap_rollouts}]")
        
        # Generate caption
        cap_pred = run_inference(cap_messages, do_sample=True)
        cap_pred_clean = clean_caption(cap_pred)
        log(f"  Caption: {cap_pred_clean}")

        # Build segmentation problem using the generated caption (output bbox)
        seg_problem = f"""All spatial relationships are defined from the viewer's perspective, where 'front' means closer to the viewer and 'back' means farther from the viewer. Please locate the object that matches the following description and provide its bounding box coordinates: {cap_pred_clean} Output the bounding box in the format [x1, y1, x2, y2] where coordinates are normalized to [0, 1000]. If no matching object is found, output null."""

        seg_content = [
            {
                "type": "image",
                "image": f"data:image/png;base64,{img_base64}",
            },
            {"type": "text", "text": seg_problem},
        ]
        seg_messages = [{"role": "user", "content": seg_content}]

        # Segmentation rollouts (using bbox output)
        seg_results = []
        for seg_idx in range(args.seg_rollouts):
            seg_pred = run_inference(seg_messages, do_sample=True)
            pred_bbox, status = decode_bbox(seg_pred)
            iou = compute_bbox_iou(pred_bbox, gt_bbox_norm)
            
            # Save pred bbox overlaid on original image
            if pred_bbox is not None:
                bbox_img = draw_bbox_on_image(images[0], pred_bbox, color=(0, 0, 255), width=3)
                if bbox_img is not None:
                    bbox_path = os.path.join(sample_dir, f'cap_{cap_idx:02d}_seg_{seg_idx:02d}_bbox.png')
                    bbox_img.save(bbox_path)
            
            seg_results.append({
                "seg_idx": seg_idx,
                "seg_pred": seg_pred[:200],
                "pred_bbox": pred_bbox,
                "status": status,
                "iou": float(iou)
            })
            log(f"    Seg {seg_idx+1}/{args.seg_rollouts}: IoU={iou:.4f} ({status}) pred_bbox={pred_bbox}")

        # Compute average IoU for this caption
        seg_ious = [r["iou"] for r in seg_results]
        avg_iou = np.mean(seg_ious)
        
        caption_result = {
            "cap_idx": cap_idx,
            "caption_raw": cap_pred,
            "caption_clean": cap_pred_clean,
            "seg_results": seg_results,
            "avg_iou": float(avg_iou)
        }
        all_results["captions"].append(caption_result)
        log(f"  -> Average IoU for this caption: {avg_iou:.4f}")

    # Compute overall statistics
    all_caption_ious = [c["avg_iou"] for c in all_results["captions"]]
    all_seg_ious = [s["iou"] for c in all_results["captions"] for s in c["seg_results"]]
    
    all_results["overall"] = {
        "mean_caption_iou": float(np.mean(all_caption_ious)),
        "std_caption_iou": float(np.std(all_caption_ious)),
        "max_caption_iou": float(np.max(all_caption_ious)),
        "min_caption_iou": float(np.min(all_caption_ious)),
        "mean_seg_iou": float(np.mean(all_seg_ious)),
        "std_seg_iou": float(np.std(all_seg_ious)),
        "total_segmentations": len(all_seg_ious),
    }

    log(f"\n{'='*60}")
    log(f"Overall Results for sample {sample_index}:")
    log(f"  Mean Caption IoU: {all_results['overall']['mean_caption_iou']:.4f} ± {all_results['overall']['std_caption_iou']:.4f}")
    log(f"  Max Caption IoU: {all_results['overall']['max_caption_iou']:.4f}")
    log(f"  Min Caption IoU: {all_results['overall']['min_caption_iou']:.4f}")
    log(f"  Mean Seg IoU (all {len(all_seg_ious)} segs): {all_results['overall']['mean_seg_iou']:.4f}")
    log(f"{'='*60}")

    # Save results
    result_path = os.path.join(sample_dir, 'rollout_results.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to {result_path}")
    
    # Save log to text file
    log_path = os.path.join(sample_dir, 'rollout_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"Log saved to {log_path}")


if __name__ == '__main__':
    main()
