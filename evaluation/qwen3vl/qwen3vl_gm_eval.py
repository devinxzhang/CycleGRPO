import argparse
import base64
import math
import os
import io
import torch
import tqdm
from pycocotools import mask as mask_utils
import numpy as np
import copy
import hydra

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from PIL import Image
import re
import json
import pyarrow.parquet as pq

from qwen_vl_utils import process_vision_info
from torchvision.transforms.functional import to_pil_image
from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.tokenmask.evaluation.utils import _init_dist_pytorch, get_dist_info, collect_results_cpu


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
    parser = argparse.ArgumentParser(description='GroundingME')
    parser.add_argument(
        '--model_path',
        default="zhouyik/Qwen3-VL-4B-SAMTok-co",
        help='hf model path.')
    parser.add_argument(
        '--vq_sam2_path',
        default="Qwen/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')
    parser.add_argument(
        '--sam2_path',
        default="Qwen/sam2.1_hiera_large.pt",
        help='sam2 model path.')
    parser.add_argument(
        '--save_dir',
        default='./results/gm/',
        help='save path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--task_id', '--task-id', type=int, default=0)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


IMAGE_FOLDER = 'rl_dataset/groundingme_train.parquet'


class GroundingMEInferenceDataset:
    def __init__(self,
                 parquet_path,
                 save_dir=None,
                 ):
        self.parquet_path = parquet_path
        self.save_dir = save_dir

        # Read parquet table
        table = pq.read_table(parquet_path)
        self.data = []
        for i in range(table.num_rows):
            row = {col: table[col][i].as_py() for col in table.column_names}
            row['original_index'] = i  # Store original index for saving
            self.data.append(row)

        if save_dir is not None:
            # filter evaluated
            if os.path.exists(save_dir):
                exists_files = os.listdir(save_dir)
                exists_indices = set(int(_file[:-5]) for _file in exists_files if _file.endswith('.json'))
                self.data = [item for item in self.data if item['original_index'] not in exists_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_dict = {}
        row = self.data[index]
        
        # Get image from bytes (may have multiple images for zoom-in)
        images = []
        for img_info in row['images']:
            image_bytes = img_info['bytes']
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            images.append(img)
        
        # Captioning task
        cap_problem = row['cap_problem'].replace('<image>\n', '').strip()
        cap_answer = row['cap_answer']  # Ground truth caption
        
        # Segmentation task
        seg_problem = row['seg_problem'].replace('<image>\n', '').strip()
        seg_answer = row['seg_answer']  # Ground truth seg answer
        
        data_dict['images'] = images  # List of images
        data_dict['cap_problem'] = cap_problem
        data_dict['cap_answer'] = cap_answer  # GT caption
        data_dict['seg_problem'] = seg_problem
        data_dict['seg_answer'] = seg_answer  # GT seg answer
        data_dict['img_id'] = row['original_index']
        data_dict['gt_mask'] = row['masks']  # RLE format mask for evaluation
        data_dict['source'] = row['source']
        
        return data_dict

def extract_mt_token_ids(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]


def remove_special_tokens(text):
    pattern = r"<\|mt_(start|end|\d{4})\|>"
    return re.sub(pattern, "", text)

def fix_mt_format(text):
    """
    使用正则表达式查找并修正不完整的 <|mt_...> 格式。
    这个函数会处理以下几种情况：
    1. 正确格式: <|mt_start|><|mt_0044|><|mt_0442|><|mt_end|> -> 不变
    2. 错误格式 (缺少第二个token和end): <|mt_start|><|mt_0198|> -> <|mt_start|><|mt_0198|><|mt_9999|><|mt_end|>
    3. 错误格式 (缺少第二个token但有end): <|mt_start|><|mt_0198|><|mt_end|> -> <|mt_start|><|mt_0198|><|mt_9999|><|mt_end|>
    """
    # 模式1: 匹配 <|mt_start|> + 一个token + <|mt_end|>
    # (<\|mt_start\|>) - 捕获组1: <|mt_start|>
    # (<\|mt_\d+\|>) - 捕获组2: <|mt_XXXX|>
    # (<\|mt_end\|>) - 捕获组3: <|mt_end|>
    pattern1 = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)'
    # 替换逻辑1: 在中间插入 <|mt_-1|>
    # \1 代表捕获组1的内容, \2 代表捕获组2的内容
    replacement1 = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern1, replacement1, text)
    # 模式2: 匹配 <|mt_start|> + 一个token，且后面不是另一个mt_token或mt_end
    # (<\|mt_start\|>) - 捕获组1: <|mt_start|>
    # (<\|mt_\d+\|>) - 捕获组2: <|mt_XXXX|>
    # (?!<\|mt_) - 负向前瞻断言，确保后面不是 "<|mt_" 开头，避免匹配到正确格式的前半部分
    pattern2 = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)'
    # 替换逻辑2: 拼接上 <|mt_-1|> 和 <|mt_end|>
    replacement2 = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern2, replacement2, text)
    return text

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


def main():
    args = parse_args()

    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    # build vq-sam2 model
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
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


    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    dataset = GroundingMEInferenceDataset(
        parquet_path=IMAGE_FOLDER,
        save_dir=args.save_dir,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    
    rows = len(per_rank_ids)
    # chunk_size = (rows+7) // 8
    chunk_size = (rows+0) // 1
    _start_ = args.task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = rows if _end_ > rows else _end_

    def images_to_base64(images):
        """Convert list of PIL images to base64 strings."""
        base64_list = []
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            b64 = base64.b64encode(buffered.getvalue()).decode()
            base64_list.append(b64)
        return base64_list

    def run_inference(messages):
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
            do_sample=False, 
            top_p=1.0,  
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        return output_text[0].replace('<think>\n\n</think>\n\n', '')

    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        img_id = data_batch['img_id']
        
        # Skip if already evaluated
        result_path = os.path.join(args.save_dir, str(img_id), 'result.json')
        if os.path.exists(result_path):
            print(f"Sample {img_id} already evaluated, skipping.")
            continue
        
        gt_mask = data_batch['gt_mask']
        images = data_batch['images']
        
        # Get image size from first image
        w, h = images[0].size
        
        # Convert images to base64
        img_base64_list = images_to_base64(images)
        
        # ========== Task 1: Captioning ==========
        cap_problem = data_batch['cap_problem']
        cap_gt = data_batch['cap_answer']
        
        # Build messages for captioning (may have multiple images for zoom-in)
        cap_content = []
        for i, b64 in enumerate(img_base64_list):
            cap_content.append({
                "type": "image",
                "image": f"data:image/png;base64,{b64}",
            })
        cap_content.append({"type": "text", "text": cap_problem})
        
        cap_messages = [{"role": "user", "content": cap_content}]
        
        print(f"\n[{img_id}] Running Captioning...")
        cap_pred = run_inference(cap_messages)
        print(f"Caption GT: {cap_gt[:100]}...")
        print(f"Caption Pred: {cap_pred[:100]}...")
        
        # ========== Task 2: Segmentation ==========
        seg_problem = data_batch['seg_problem']
        seg_gt = data_batch['seg_answer']
        
        # Build messages for segmentation (only first image)
        seg_content = [
            {
                "type": "image",
                "image": f"data:image/png;base64,{img_base64_list[0]}",
            },
            {"type": "text", "text": seg_problem},
        ]
        
        seg_messages = [{"role": "user", "content": seg_content}]
        
        print(f"[{img_id}] Running Segmentation...")
        seg_pred = run_inference(seg_messages)
        print(f"Seg Pred: {seg_pred[:100]}...")
        
        # Decode segmentation mask
        quant_ids = extract_mt_token_ids(seg_pred)
        if len(quant_ids) == 0:
            print("No SEG !!!")
            pred_masks = torch.zeros((1, h, w), dtype=torch.bool)
        else:
            if len(quant_ids) % CODEBOOK_DEPTH != 0:
                print("FORMAT ERROR: ", seg_pred)
                seg_pred = fix_mt_format_comprehensive(seg_pred)
                print("FIXED OUTPUT TEXT: ", seg_pred)
                quant_ids = extract_mt_token_ids(seg_pred)
            
            if len(quant_ids) % CODEBOOK_DEPTH != 0:
                print("Still FORMAT ERROR after fix, using empty mask")
                pred_masks = torch.zeros((1, h, w), dtype=torch.bool)
            else:
                batch_size = len(quant_ids) // CODEBOOK_DEPTH
                # Limit batch_size to avoid CUDA errors (only use first mask if too many)
                MAX_BATCH_SIZE = 16
                if batch_size > MAX_BATCH_SIZE:
                    print(f"WARNING: batch_size={batch_size} too large, truncating to {MAX_BATCH_SIZE}")
                    batch_size = MAX_BATCH_SIZE
                    quant_ids = quant_ids[:batch_size * CODEBOOK_DEPTH]
                remap_quant_ids = []
                for bs_id in range(batch_size):
                    chunk_quant_ids = quant_ids[bs_id*CODEBOOK_DEPTH:(bs_id+1)*CODEBOOK_DEPTH]
                    remap_chunk_quant_ids = [quant_id - book_id*CODEBOOK_SIZE for book_id, quant_id in enumerate(chunk_quant_ids)]
                    remap_chunk_quant_ids_error_handle = [quant_id if quant_id < CODEBOOK_SIZE else -1 for quant_id in remap_chunk_quant_ids]
                    remap_quant_ids.append(remap_chunk_quant_ids_error_handle)

                # Use the first image for mask decoding
                ori_width, ori_height = w, h
                sam2_image = np.array(images[0])
                sam2_image = sam2_image_processor.apply_image(sam2_image)
                sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
                sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
                sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

                quant_ids_tensor = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

                try:
                    with torch.no_grad():
                        _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids_tensor)
                    _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
                    _pred_masks = _pred_masks > 0.5
                    _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
                    pred_masks = torch.from_numpy(_pred_masks).to(torch.bool)
                except:
                    print("VQ-SAM2 decoding error, using empty mask")
                    pred_masks = torch.zeros((1, h, w), dtype=torch.bool)

        # Save result for this sample
        iou = save_groundingme_output(
            args.save_dir,
            img_id,
            images[0],  # original image
            cap_gt,
            cap_pred,
            pred_masks,
            gt_mask
        )
        results.append({'img_id': img_id, 'iou': iou})

    results = collect_results_cpu(results, len(dataset), tmpdir='./groundingme_eval_tmp')
    
    # Compute mIoU across all samples
    if rank == 0 and results is not None:
        all_ious = [r['iou'] for r in results if r is not None]
        mIoU = sum(all_ious) / len(all_ious) if len(all_ious) > 0 else 0.0
        print(f"\n{'='*50}")
        print(f"Total samples: {len(all_ious)}")
        print(f"mIoU: {mIoU:.4f}")
        print(f"{'='*50}")
        
        # Save summary
        summary_path = os.path.join(args.save_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump({'total_samples': len(all_ious), 'mIoU': mIoU}, f, indent=2)
        print(f"Summary saved to {summary_path}")


def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.5):
    """Overlay a binary mask on an image with specified color and transparency."""
    image_np = np.array(image).copy()
    mask_np = mask.astype(bool)
    
    # Create colored overlay
    overlay = image_np.copy()
    overlay[mask_np] = color
    
    # Blend original and overlay
    result = image_np.copy()
    result[mask_np] = (alpha * np.array(color) + (1 - alpha) * image_np[mask_np]).astype(np.uint8)
    
    return Image.fromarray(result)


def clean_cap_gt(text):
    """Clean caption GT: remove <think>...</think><answer>...</answer> format."""
    # Remove <think>...</think> part
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove <answer> and </answer> tags
    text = text.replace('<answer>', '').replace('</answer>', '')
    # Clean up whitespace
    text = text.strip()
    return text


def clean_cap_pred(text):
    """Clean caption prediction: remove <|im_end|> and other special tokens."""
    text = text.replace('<|im_end|>', '')
    text = remove_special_tokens(text)
    text = text.strip()
    return text


def save_groundingme_output(output_dir, img_id, original_image, cap_gt, cap_pred, pred_masks, gt_mask):
    """Save GroundingME evaluation output: original image, mask overlays, and captions."""
    # Create output directory structure
    sample_dir = os.path.join(output_dir, str(img_id))
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Save original image
    original_image.save(os.path.join(sample_dir, 'original.png'))

    # Convert the predicted masks into numpy
    pred_masks_tensor = pred_masks.cpu()
    
    # Merge all predicted masks into one
    if pred_masks_tensor.shape[0] > 0:
        merged_pred_mask = pred_masks_tensor.any(dim=0).numpy().astype(np.uint8)
    else:
        merged_pred_mask = np.zeros((original_image.height, original_image.width), dtype=np.uint8)

    # Decode GT mask
    gt_mask_decoded = mask_utils.decode(gt_mask)

    # 2. Save GT mask overlay (green color)
    gt_overlay = overlay_mask_on_image(original_image, gt_mask_decoded, color=(0, 255, 0), alpha=0.5)
    gt_overlay.save(os.path.join(sample_dir, 'gt_mask_overlay.png'))

    # 3. Save pred mask overlay (red color)
    pred_overlay = overlay_mask_on_image(original_image, merged_pred_mask, color=(255, 0, 0), alpha=0.5)
    pred_overlay.save(os.path.join(sample_dir, 'pred_mask_overlay.png'))

    # Compute IoU
    intersection = np.logical_and(merged_pred_mask, gt_mask_decoded).sum()
    union = np.logical_or(merged_pred_mask, gt_mask_decoded).sum()
    iou = intersection / (union + 1e-8)

    # 4 & 5. Clean and save captions
    cap_gt_clean = clean_cap_gt(cap_gt)
    cap_pred_clean = clean_cap_pred(cap_pred)

    result_dict = {
        "img_id": img_id,
        "cap_gt": cap_gt_clean,
        "cap_pred": cap_pred_clean,
        "iou": float(iou),
    }

    # Save JSON with captions and IoU
    json_path = os.path.join(sample_dir, 'result.json')
    with open(json_path, 'w') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    print(f"Saved result for {img_id}, IoU: {iou:.4f}")
    return iou


def process_and_save_output(output_dir, image_name, text_output, pred_masks):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    text_output = text_output.replace("<s>", "").replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    cleaned_str = re.sub(r'<.*?>', '', text_output)

    pattern = re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>')
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]

    # Remove the [SEG] token
    # cleaned_str = cleaned_str.replace('[SEG]', '')
    cleaned_str = remove_special_tokens(cleaned_str)

    # Strip unnecessary spaces
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    # Convert the predicted masks into RLE format
    pred_masks_tensor = pred_masks.cpu()
    uncompressed_mask_rles = mask_to_rle_pytorch(pred_masks_tensor)
    rle_masks = []
    for m in uncompressed_mask_rles:
        rle_masks.append(coco_encode_rle(m))

    # Create results dictionary
    # print(f"clean_str: {cleaned_str}")
    result_dict = {
        "image_id": image_name[:-4],
        "caption": cleaned_str,
        "phrases": phrases,
        "pred_masks": rle_masks
    }

    # print(cleaned_str)
    # print(phrases)

    output_path = f"{output_dir}/{image_name[:-4]}.json"

    with open(output_path, 'w') as f:
        json.dump(result_dict, f)

    return

def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out

def coco_encode_rle(uncompressed_rle):
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json

    return rle

if __name__ == '__main__':
    main()