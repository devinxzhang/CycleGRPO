import argparse
import base64
import os
import io
import torch
import numpy as np
import hydra

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from pycocotools import mask as mask_utils

from PIL import Image
import re
import json
import pyarrow.parquet as pq

from qwen_vl_utils import process_vision_info
from torchvision.transforms.functional import to_pil_image
from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))


def parse_args():
    parser = argparse.ArgumentParser(description='GroundingME Rollout Evaluation')
    parser.add_argument(
        '--model_path',
        default="zhouyik/Qwen3-VL-4B-SAMTok-co",
        help='hf model path.')
    parser.add_argument(
        '--vq_sam2_path',
        default="Qwen/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--sam2_path',
        default="Qwen/sam2.1_hiera_large.pt",
        help='sam2 model path.')
    parser.add_argument(
        '--save_dir',
        default='./results/gm_rollout/',
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
    args = parser.parse_args()
    return args


IMAGE_FOLDER = 'rl_dataset/groundingme_train.parquet'


def extract_mt_token_ids(text):
    """Extract mask token ids from text.
    
    Format: <|mt_start|><|mt_xx|><|mt_xx|><|mt_end|>
    Only extract the first complete mask (2 tokens between start and end).
    """
    # Find the first <|mt_start|>...<|mt_end|> block
    block_pattern = r"<\|mt_start\|>(.*?)<\|mt_end\|>"
    block_match = re.search(block_pattern, text)
    if not block_match:
        return []
    
    # Extract numeric tokens from within the block
    block_content = block_match.group(1)
    token_pattern = r"<\|mt_(\d+)\|>"
    matches = [int(x) for x in re.findall(token_pattern, block_content)]
    
    # Return only first 2 tokens (one mask = 2 tokens with CODEBOOK_DEPTH=2)
    return matches[:2]


def remove_special_tokens(text):
    pattern = r"<\|mt_(start|end|\d{4})\|>"
    return re.sub(pattern, "", text)


def fix_mt_format_comprehensive(text):
    """Fix incomplete <|mt_...> format."""
    # Rule 1: Too many tokens (3+)
    pattern_too_many = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_\d+\|>)(?:<\|mt_\d+\|>)+<\|mt_end\|>'
    replacement_too_many = r'\1\2\3<|mt_end|>'
    text = re.sub(pattern_too_many, replacement_too_many, text)
    # Rule 2: Too few tokens (1, with end)
    pattern_too_few_with_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)'
    replacement_too_few = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_with_end, replacement_too_few, text)
    # Rule 3: Too few tokens (1, no end)
    pattern_too_few_no_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)'
    replacement_too_few_no_end = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_no_end, replacement_too_few_no_end, text)
    return text


def clean_caption(text):
    """Clean caption: remove special tokens and formatting."""
    text = text.replace('<|im_end|>', '')
    text = text.replace('<think>\n\n</think>\n\n', '')
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<answer>', '').replace('</answer>', '')
    text = remove_special_tokens(text)
    text = text.strip()
    return text


def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.5):
    """Overlay a binary mask on an image with specified color and transparency."""
    image_np = np.array(image).copy()
    mask_np = mask.astype(bool)
    result = image_np.copy()
    result[mask_np] = (alpha * np.array(color) + (1 - alpha) * image_np[mask_np]).astype(np.uint8)
    return Image.fromarray(result)


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

    # Build VQ-SAM2 model
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    log("Loading VQ-SAM2 model...")
    with hydra.initialize(version_base=None, config_path='../../projects/transformers/vq_sam2/sam2/sam2_configs'):
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

    # Load dataset
    log(f"Loading dataset from {IMAGE_FOLDER}...")
    table = pq.read_table(IMAGE_FOLDER)

    # Process each index
    for sample_index in args.index:
        evaluate_sample(args, model, processor, vq_sam2, sam2_image_processor, table, sample_index, log_lines, CODEBOOK_SIZE, CODEBOOK_DEPTH)


def evaluate_sample(args, model, processor, vq_sam2, sam2_image_processor, table, sample_index, log_lines, CODEBOOK_SIZE, CODEBOOK_DEPTH):
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
    cap_problem = row['cap_problem'].replace('<image>\n', '').strip()
    gt_mask = row['masks']
    gt_mask_decoded = mask_utils.decode(gt_mask)

    # Create output directory
    sample_dir = os.path.join(args.save_dir, str(sample_index))
    result_path = os.path.join(sample_dir, 'rollout_results.json')
    
    # # Skip if already evaluated
    # if os.path.exists(result_path):
    #     log(f"Sample {args.index} already evaluated, skipping. (Found {result_path})")
    #     return
    
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

    def decode_seg_mask(seg_pred):
        """Decode segmentation prediction to mask."""
        quant_ids = extract_mt_token_ids(seg_pred)
        if len(quant_ids) == 0:
            return None, "no_mask_token"

        if len(quant_ids) % CODEBOOK_DEPTH != 0:
            seg_pred = fix_mt_format_comprehensive(seg_pred)
            quant_ids = extract_mt_token_ids(seg_pred)

        if len(quant_ids) % CODEBOOK_DEPTH != 0:
            return None, "format_error"

        batch_size = len(quant_ids) // CODEBOOK_DEPTH
        MAX_BATCH_SIZE = 16
        if batch_size > MAX_BATCH_SIZE:
            batch_size = MAX_BATCH_SIZE
            quant_ids = quant_ids[:batch_size * CODEBOOK_DEPTH]

        remap_quant_ids = []
        for bs_id in range(batch_size):
            chunk_quant_ids = quant_ids[bs_id*CODEBOOK_DEPTH:(bs_id+1)*CODEBOOK_DEPTH]
            remap_chunk_quant_ids = [quant_id - book_id*CODEBOOK_SIZE for book_id, quant_id in enumerate(chunk_quant_ids)]
            remap_chunk_quant_ids_error_handle = [quant_id if quant_id < CODEBOOK_SIZE else -1 for quant_id in remap_chunk_quant_ids]
            remap_quant_ids.append(remap_chunk_quant_ids_error_handle)

        sam2_image = np.array(images[0])
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
        sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

        quant_ids_tensor = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

        try:
            with torch.no_grad():
                _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids_tensor)
            _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(h, w), mode='bilinear')
            _pred_masks = _pred_masks > 0.5
            _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
            # Merge all masks
            merged_mask = np.any(_pred_masks, axis=0).astype(np.uint8)
            return merged_mask, "success"
        except Exception as e:
            return None, f"decode_error: {str(e)}"

    def compute_iou(pred_mask, gt_mask):
        """Compute IoU between predicted and ground truth masks."""
        if pred_mask is None:
            return 0.0
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return intersection / (union + 1e-8)

    # Build captioning messages
    img_base64 = image_to_base64(images[0])
    cap_content = []
    for img in images:
        cap_content.append({
            "type": "image",
            "image": f"data:image/png;base64,{image_to_base64(img)}",
        })
    cap_content.append({"type": "text", "text": cap_problem})
    cap_messages = [{"role": "user", "content": cap_content}]

    # Results storage
    all_results = {
        "index": sample_index,
        "cap_problem": cap_problem,
        "cap_rollouts": args.cap_rollouts,
        "seg_rollouts": args.seg_rollouts,
        "captions": []
    }

    log(f"\n{'='*60}")
    log(f"Evaluating sample {sample_index}")
    log(f"Caption rollouts: {args.cap_rollouts}, Seg rollouts per caption: {args.seg_rollouts}")
    log(f"{'='*60}")

    for cap_idx in range(args.cap_rollouts):
        log(f"\n[Caption {cap_idx+1}/{args.cap_rollouts}]")
        
        # Generate caption
        cap_pred = run_inference(cap_messages, do_sample=True)
        cap_pred_clean = clean_caption(cap_pred)
        log(f"  Caption: {cap_pred_clean}")

        # Build segmentation problem using the generated caption
        seg_problem = f"""All spatial relationships are defined from the viewer's perspective, where 'front' means closer to the viewer and 'back' means farther from the viewer. Please provide the segmentation mask of the object the following statement describes: {cap_pred_clean} Ensure that all details mentioned about the object are accurate. If a matching object is found, provide its segmentation mask in the format `<|mt_start|><|mt_xxxx|><|mt_xxxx|><|mt_end|>`. If no matching object is found, output null."""

        seg_content = [
            {
                "type": "image",
                "image": f"data:image/png;base64,{img_base64}",
            },
            {"type": "text", "text": seg_problem},
        ]
        seg_messages = [{"role": "user", "content": seg_content}]

        # Segmentation rollouts
        seg_results = []
        for seg_idx in range(args.seg_rollouts):
            seg_pred = run_inference(seg_messages, do_sample=True)
            pred_mask, status = decode_seg_mask(seg_pred)
            iou = compute_iou(pred_mask, gt_mask_decoded)
            
            # Save pred mask overlaid on original image and binary mask
            if pred_mask is not None:
                # Save overlay image
                overlay_img = overlay_mask_on_image(images[0], pred_mask, color=(0, 0, 255), alpha=0.5)
                overlay_path = os.path.join(sample_dir, f'cap_{cap_idx:02d}_seg_{seg_idx:02d}_overlay.png')
                overlay_img.save(overlay_path)
                # Save binary mask
                binary_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
                binary_mask_path = os.path.join(sample_dir, f'cap_{cap_idx:02d}_seg_{seg_idx:02d}_mask.png')
                binary_mask_img.save(binary_mask_path)
            
            seg_results.append({
                "seg_idx": seg_idx,
                "seg_pred": seg_pred[:200],
                "status": status,
                "iou": float(iou)
            })
            log(f"    Seg {seg_idx+1}/{args.seg_rollouts}: IoU={iou:.4f} ({status})")

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
