import argparse
import copy
import math
import os
import time
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

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info
from torchvision.transforms.functional import to_pil_image

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config

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
    parser = argparse.ArgumentParser(description='GroundingSuite')
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
        default='./results/groundingsuite/',
        help='save path')
    parser.add_argument(
        '--dataset',
        default='./data/GroundingSuiteEval/GroundingSuite-Eval.jsonl',
        help='Specify a ref dataset')
    parser.add_argument('--task_id', '--task-id', type=int, default=0,
                        help='Shard index for this process (0 .. num_tasks-1).')
    parser.add_argument('--num_tasks', '--num-tasks', type=int, default=1,
                        help='Total number of shards / data-parallel processes (one per GPU).')
    parser.add_argument('--gpu_id', '--gpu-id', type=int, default=-1,
                        help='CUDA device to bind this process to. Default -1 => use task_id.')
    parser.add_argument('--merge_out', '--merge-out', default=None,
                        help='Merged JSONL path. Default: <save_dir>_pred.jsonl. '
                             'Once all shards have written their {idx}.json, the last '
                             'one to finish collates them here automatically.')
    parser.add_argument('--no_merge', '--no-merge', action='store_true',
                        help='Disable the automatic merge-to-JSONL step.')
    args = parser.parse_args()
    return args

def load_image_with_retry(path, retries=6, base_delay=0.5):
    """Open an image, retrying transient NAS I/O failures with backoff.

    The coco images live on a slow NAS; under N concurrent shards Image.open can
    raise transient I/O errors. Retrying avoids dropping/crashing on valid images.
    Raises the last error if all retries fail.
    """
    last = None
    for i in range(retries):
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(base_delay * (2 ** i))  # 0.5, 1, 2, 4, 8, 16s
    raise last


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(mask_utils.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

def rle_to_mask(rle):
    mask = []
    for r in rle:
        m = mask_utils.decode(r)
        m = np.uint8(m)
        mask.append(m)
    mask = np.stack(mask, axis=0)
    return mask


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

def find_first_index(arr, value):
    """
    在NumPy数组中找到第一个指定值的第一个出现的索引
    
    参数:
        arr: NumPy数组
        value: 要查找的值
        
    返回:
        第一个匹配值的索引，如果没有找到则返回-1
    """
    # 使用where找到所有匹配值的索引
    indices = np.where(arr == value)[0]
    
    # 返回第一个索引，如果没有找到则返回-1
    return indices[0] if len(indices) > 0 else -1

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

def maybe_merge(all_data_dict, args):
    """Collate per-idx {idx}.json into one JSONL once EVERY sample is done.

    Called by each shard at exit. Only fires when all expected files exist, so
    the last shard to finish triggers it. Writes atomically (tmp + rename) so
    concurrent triggers can't corrupt the output (result is identical anyway).
    """
    if args.no_merge:
        return
    out_path = args.merge_out or (args.save_dir.rstrip("/") + "_pred.jsonl")
    want = [d["idx"] for d in all_data_dict]
    paths = [os.path.join(args.save_dir, f"{i}.json") for i in want]
    if not all(os.path.exists(p) for p in paths):
        n_done = sum(os.path.exists(p) for p in paths)
        print(f"[task {args.task_id}] merge skipped: {n_done}/{len(want)} done, "
              f"other shards still running.")
        return

    records = []
    for p in paths:
        try:
            with open(p) as f:
                records.append(json.load(f))
        except Exception as e:  # noqa: BLE001
            print(f"[merge] unreadable {p}: {e}")

    def _key(r):
        v = r.get("idx")
        try:
            return (0, int(v))
        except (TypeError, ValueError):
            return (1, str(v))
    records.sort(key=_key)

    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp_path, out_path)  # atomic
    print(f"[task {args.task_id}] merged {len(records)} records -> {out_path}")


def main():
    args = parse_args()

    # ---- multi-GPU data parallelism: bind this process to one GPU ----
    gpu_id = args.task_id if args.gpu_id < 0 else args.gpu_id
    torch.cuda.set_device(gpu_id)            # makes every subsequent .cuda() target this GPU
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[task {args.task_id}/{args.num_tasks}] using GPU {gpu_id}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).to(device).eval()

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

    vq_sam2 = VQ_SAM2(vq_sam2_config).to(device).eval()

    state = torch.load(args.vq_sam2_path, map_location="cpu")
    vq_sam2.load_state_dict(state)

    sam2_image_processor = DirectResize(1024)

    # exist_ok=True: many shards may race to create the dir at once
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
    # Data-parallel sharding across `num_tasks` processes (one per GPU).
    chunk_size = math.ceil(rows / args.num_tasks)
    _start_ = args.task_id * chunk_size
    _end_ = min(_start_ + chunk_size, rows)
    print(f"[task {args.task_id}/{args.num_tasks}] rows {_start_}:{_end_} of {rows}")

    count = 0
    for data_dict in tqdm.tqdm(all_data_dict[_start_:_end_]):
        image_file = data_dict['image_path']
        item_idx = data_dict['idx']
        label = data_dict['label']
        caption = data_dict['caption']
        class_id = data_dict['class_id']

        image_path = os.path.join('./data/GroundingSuiteEval', image_file)
        # Replace path for coco images
        # image_path = image_path.replace('data/ref_seg/grefs/coco2014/train2014', 'data/coco/train2014')
        image_path = image_path.replace('./data/ref_seg/grefs/coco2014/train2014', '<PATH_TO_COCO2014>/train2014')
    
        try:
            image = load_image_with_retry(image_path)
        except Exception as e:
            print(f"skip {image_path} after retries: {e}")
            continue
        ori_width, ori_height = image.size

        question = f"Please carefully check the image and detect the object this sentence describes: {caption}"

        if os.path.exists(f"{args.save_dir}/{item_idx}.json"):
            print("file exists.............")
            continue

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
            max_new_tokens=512,
            do_sample=False,  # 关闭采样，使用贪婪解码
            top_p=1.0,  # 配合do_sample=False使用
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print("User: ", phrase)
        print("Assistant: ", output_text)

        quant_ids = extract_mt_token_ids_v1(output_text[0])
        if len(quant_ids) == 0:
            zero_mask = np.zeros((1, ori_height, ori_width)).astype(np.uint8)
            zero_mask = mask_to_rle(zero_mask)[0]
            prediction = {'idx': item_idx, 'image_path': image_file, 'predicted_box': [0, 0, 0, 0], 'predicted_segmentation': zero_mask, 'class_id': class_id}

            with open(f"{args.save_dir}/{item_idx}.json", 'w') as f:
                json.dump(prediction, f)
            continue

        # if len(quant_ids) % CODEBOOK_DEPTH != 0:
        #     print("FORMAT ERROR: ", output_text)
        #     output_text = [fix_mt_format_comprehensive(output_text[0])]
        #     print("FIXED OUTPUT TEXT: ", output_text)
        #     quant_ids = extract_mt_token_ids_v2(output_text[0])
        # assert len(quant_ids) % CODEBOOK_DEPTH == 0
        if len(quant_ids) % CODEBOOK_DEPTH != 0:
            zero_mask = np.zeros((1, ori_height, ori_width)).astype(np.uint8)
            zero_mask = mask_to_rle(zero_mask)[0]
            prediction = {'idx': item_idx, 'image_path': image_file, 'predicted_box': [0, 0, 0, 0], 'predicted_segmentation': zero_mask, 'class_id': class_id}

            with open(f"{args.save_dir}/{item_idx}.json", 'w') as f:
                json.dump(prediction, f)
            continue

        batch_size = len(quant_ids) // CODEBOOK_DEPTH
        remap_quant_ids = []
        for bs_id in range(batch_size):
            chunk_quant_ids = quant_ids[bs_id*CODEBOOK_DEPTH:(bs_id+1)*CODEBOOK_DEPTH]
            remap_chunk_quant_ids = [quant_id - book_id*CODEBOOK_SIZE for book_id, quant_id in enumerate(chunk_quant_ids)]
            code1 = remap_chunk_quant_ids[0]
            code2 = remap_chunk_quant_ids[1]
            if not (code1 >= 0 and code1 < CODEBOOK_SIZE):
                continue
            if not (code2 >= 0 and code2 < CODEBOOK_SIZE):
                code2 = -1
            remap_chunk_quant_ids_error_handle = [code1, code2]
            remap_quant_ids.append(remap_chunk_quant_ids_error_handle)

        batch_size = len(remap_quant_ids)
        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
        sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

        quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

        try:
            with torch.no_grad():
                _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
        except Exception as e:  # never drop into pdb -- would hang background/multi-GPU runs
            print(f"[task {args.task_id}] forward_with_codes failed on idx={item_idx}: {e}")
            zero_mask = np.zeros((1, ori_height, ori_width)).astype(np.uint8)
            zero_mask = mask_to_rle(zero_mask)[0]
            prediction = {'idx': item_idx, 'image_path': image_file, 'predicted_box': [0, 0, 0, 0], 'predicted_segmentation': zero_mask, 'class_id': class_id}

            with open(f"{args.save_dir}/{item_idx}.json", 'w') as f:
                json.dump(prediction, f)
            continue

        _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
        _pred_masks = _pred_masks > 0.5
        _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
        _pred_masks = np.sum(_pred_masks, axis=0).astype(np.uint8)[np.newaxis, :, :]
        _pred_masks = (_pred_masks > 0).astype(np.uint8)
        try:
            pred_box = torchvision.ops.masks_to_boxes(torch.from_numpy(_pred_masks)).cpu().numpy().tolist()
        except:
            pred_box = [0, 0, 0, 0]
        _pred_masks = mask_to_rle(_pred_masks)[0]
        prediction = {'idx': item_idx, 'image_path': image_file, 'predicted_box': pred_box, 'predicted_segmentation': _pred_masks, 'class_id': class_id}
        with open(f"{args.save_dir}/{item_idx}.json", 'w') as f:
            json.dump(prediction, f)

    # once this shard is done, merge into one JSONL iff all shards have finished
    maybe_merge(all_data_dict, args)


if __name__ == "__main__":
    main()