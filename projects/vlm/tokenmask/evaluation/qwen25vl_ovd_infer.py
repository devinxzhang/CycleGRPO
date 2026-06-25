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
from typing import Any, List, Optional

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from dataset import RESDataset
from xtuner.model.utils import guess_load_checkpoint

from qwen_vl_utils import process_vision_info
from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize

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



FENCE_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def _clean_wrappers(s: str) -> str:
    """去掉常见包裹符、收尾空白等。"""
    s = s.replace("<|im_end|>", "")
    s = s.strip()
    # 去掉首尾引号（包括中英文引号）
    quotes = "'\"“”‘’"
    if len(s) >= 2 and s[0] in quotes and s[-1] in quotes:
        s = s[1:-1].strip()
    return s

def _extract_from_code_fence(text: str) -> List[str]:
    return [m.strip() for m in FENCE_JSON_RE.findall(text)]

def _extract_by_bracket_scan(text: str) -> List[str]:
    """
    在全文里用配对括号扫描，提取可能的 JSON 片段（对象或数组）。
    忽略字符串内的括号与转义。
    返回候选片段（可能有嵌套，调用方可按长度降序尝试 json.loads）。
    """
    candidates = []
    open_to_close = {"{": "}", "[": "]"}
    open_set = set(open_to_close.keys())
    close_set = set(open_to_close.values())

    n = len(text)
    for start in range(n):
        ch = text[start]
        if ch not in open_set:
            continue
        stack = [open_to_close[ch]]
        in_string = False
        escape = False
        for i in range(start + 1, n):
            c = text[i]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
                continue
            else:
                if c == '"':
                    in_string = True
                elif c in open_set:
                    stack.append(open_to_close[c])
                elif c in close_set:
                    if not stack or c != stack[-1]:
                        break  # 非法配对，放弃这个起点
                    stack.pop()
                    if not stack:
                        # 成功匹配一段
                        candidates.append(text[start : i + 1])
                        break
    # 去重 & 按长度降序（优先外层最大块）
    uniq = list(dict.fromkeys(candidates))
    uniq.sort(key=len, reverse=True)
    return uniq

def _try_parse_candidates(cands: List[str]) -> List[Any]:
    parsed = []
    for raw in cands:
        cand = _clean_wrappers(raw)
        try:
            parsed.append(json.loads(cand))
        except Exception:
            # 再尝试去掉再次包裹的三引号/反引号之类
            cand2 = cand.strip("`").strip()
            try:
                parsed.append(json.loads(cand2))
            except Exception:
                continue
    return parsed

def parse_first_json(text: str) -> Any:
    """
    提取并解析第一个可用 JSON（先看```json```代码块，再看正文扫描）。
    解析失败会抛出 ValueError。
    """
    text = _clean_wrappers(text)

    # 1) 代码块
    fence_cands = _extract_from_code_fence(text)
    parsed = _try_parse_candidates(fence_cands)
    if parsed:
        return parsed[0]

    # 2) 正文扫描（对象/数组）
    bracket_cands = _extract_by_bracket_scan(text)
    parsed = _try_parse_candidates(bracket_cands)
    if parsed:
        return parsed[0]

    raise ValueError("没有在文本中找到可解析的 JSON。")

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

def main():
    args = parse_args()

    # build qwen25vl model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    # build vq-sam2 model
    sam2_config = SAM2Config(
        cfg_path="sam2.1_hiera_l.yaml",
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
    )

    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=256,
    )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    pretrained_state_dict = guess_load_checkpoint(args.vq_sam2_path)

    pretrained_state_dict_new = {}
    for key in pretrained_state_dict.keys():
        new_key = copy.deepcopy(key)
        if key.startswith('hf_model.'):
            new_key = new_key[len('hf_model.'):]
        pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
    
    vq_sam2.load_state_dict(pretrained_state_dict_new)

    sam2_image_processor = DirectResize(1024)

    with open(args.dataset, 'r') as f:
        eval_items = json.load(f)
    
    for eval_item in tqdm.tqdm(eval_items):
        image_file = eval_item['image_file']
        image_id = eval_item['image_id']
        candidate_category_names = eval_item['candidate_category']

        if os.path.exists(f"./temp_save/ovd/{image_id}.json"):
            try_again = False
            with open(f"./temp_save/ovd/{image_id}.json", 'r') as f:
                json_data = json.load(f)
            if json_data['prediction'] is None:
                try_again = True
            if not try_again:
                print("file exists.............")
                continue

        image_path = os.path.join('./data/coco/val2017', image_file)
        candidate_category_names_str = ", ".join([f"\"{name}\"" for name in candidate_category_names])
        question = "Please carefully check the image and detect the following objects: [" + candidate_category_names_str + "]."

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

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
        
        if try_again:
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=4096,
                do_sample=True,             # 启用采样，这是关键！
                temperature=0.7,            # 调整温度，增加随机性。可以尝试0.5到1.0之间的值
                top_k=50,                   # 考虑概率最高的50个词元
                top_p=0.95,                 # 考虑累积概率达到95%的词元集合
            )
        else:
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=4096,
                do_sample=False,  # 关闭采样，使用贪婪解码
                top_p=1.0,  # 配合do_sample=False使用
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        # print("User: ", phrase)
        print("Assistant: ", output_text)

        results = parse_first_json(output_text[0])
        if isinstance(results, list) and len(results) == 0:
            copy_eval_item = copy.deepcopy(eval_item)
            copy_eval_item.update({'prediction': None})

            with open(f"./temp_save/ovd/{image_id}.json", 'w') as f:
                json.dump(copy_eval_item, f, indent=4)
            continue
        if not (isinstance(results, list) and isinstance(results[0], dict)):
            pattern = r'\{[^{}]*?"mask_2d"\s*:\s*"(.*?)"[^{}]*?"label"\s*:\s*"(.*?)"[^{}]*?\}'
            matches = re.findall(pattern, output_text[0], flags=re.S)
            # matches 是一个元组列表，每个元素是 (mask_2d_value, label_value)
            results = []
            for mask_val, label_val in matches:
                results.append({"mask_2d": mask_val, "label": label_val})

        all_quant_ids = []
        all_labels = []
        exist_mask_2d = []
        for one_row in results:
            try:
                mask_2d = one_row['mask_2d']
                label = one_row['label']
            except:
                print(one_row)
                print(type(one_row))
                exit(0) 

            if mask_2d in exist_mask_2d:
                continue
            exist_mask_2d.append(mask_2d)

            quant_ids = extract_mt_token_ids_v1(mask_2d)
            if len(quant_ids) == 0:
                continue

            if len(quant_ids) % CODEBOOK_DEPTH != 0:
                print("FORMAT ERROR: ", mask_2d)
                mask_2d = [fix_mt_format_comprehensive(mask_2d)]
                print("FIXED OUTPUT TEXT: ", mask_2d)
                try:
                    quant_ids = extract_mt_token_ids_v2(mask_2d)
                except:
                    continue
            # assert len(quant_ids) % CODEBOOK_DEPTH == 0
            if len(quant_ids) % CODEBOOK_DEPTH != 0:
                continue

            batch_size = len(quant_ids) // CODEBOOK_DEPTH
            assert batch_size == 1
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
                all_quant_ids.append(remap_chunk_quant_ids_error_handle)
                all_labels.append(label)

        batch_size = len(all_quant_ids)
        if batch_size == 0:
            copy_eval_item = copy.deepcopy(eval_item)
            copy_eval_item.update({'prediction': None})

            with open(f"./temp_save/ovd/{image_id}.json", 'w') as f:
                json.dump(copy_eval_item, f, indent=4)
            continue
        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
        sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

        quant_ids = torch.LongTensor(all_quant_ids).to(vq_sam2.device)

        with torch.no_grad():
            _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
        _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
        _pred_masks = _pred_masks > 0.5
        # _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)

        save_items = []
        for label, pred_mask in zip(all_labels, _pred_masks):
            try:
                bbox = torchvision.ops.masks_to_boxes(pred_mask)
            except Exception as e:
                continue
            
            mask = pred_mask[0].cpu().numpy().astype(np.uint8)
            bbox = bbox[0].cpu().numpy().tolist()

            rle = mask_utils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")

            save_items.append({'mask': rle, 'bbox': bbox, 'label': label})
        
        if len(save_items) == 0:
            copy_eval_item = copy.deepcopy(eval_item)
            copy_eval_item.update({'prediction': None})

            with open(f"./temp_save/ovd/{image_id}.json", 'w') as f:
                json.dump(copy_eval_item, f, indent=4)
            continue

        copy_eval_item = copy.deepcopy(eval_item)
        copy_eval_item.update({'prediction': save_items})

        with open(f"./temp_save/ovd/{image_id}.json", 'w') as f:
            json.dump(copy_eval_item, f, indent=4)

def merge():
    all_items = []
    for json_file in os.listdir("./temp_save/ovd"):
        if not json_file.endswith('.json'):
            continue
        json_path = os.path.join("./temp_save/ovd", json_file)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        all_items.append(json_data)
    with open("./godx7/DLC-Bench/coco2017_val_ovd_result_7b.json", 'w') as f:
        json.dump(all_items, f, indent=4)

if __name__ == '__main__':
    main()
    merge()

    
        