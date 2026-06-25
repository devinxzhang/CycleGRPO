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

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import mmengine
from mmengine.dataset import BaseDataset
from mmdet.registry import DATASETS
from mmdet.datasets.coco_panoptic import COCOPanoptic, CocoPanopticDataset
from collections import defaultdict


from qwen_vl_utils import process_vision_info


object_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard',
    'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit',
    'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
    'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
    'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone',
    'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
    'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
    'food-other-merged', 'building-other-merged', 'rock-merged',
    'wall-other-merged', 'rug-merged'
]

predicate_classes = [
    'over',
    'in front of',
    'beside',
    'on',
    'in',
    'attached to',
    'hanging from',
    'on back of',
    'falling off',
    'going down',
    'painted on',
    'walking on',
    'running on',
    'crossing',
    'standing on',
    'lying on',
    'sitting on',
    'flying over',
    'jumping over',
    'jumping from',
    'wearing',
    'holding',
    'carrying',
    'looking at',
    'guiding',
    'kissing',
    'eating',
    'drinking',
    'feeding',
    'biting',
    'catching',
    'picking',
    'playing with',
    'chasing',
    'climbing',
    'cleaning',
    'playing',
    'touching',
    'pushing',
    'pulling',
    'opening',
    'cooking',
    'talking to',
    'throwing',
    'slicing',
    'driving',
    'riding',
    'parked on',
    'driving on',
    'about to hit',
    'kicking',
    'swinging',
    'entering',
    'exiting',
    'enclosing',
    'leaning on',
]

def extract_mt_token_ids(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]

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

class COCOPanopticRelation(COCOPanoptic):
    def createIndex(self):
        # create index
        print('creating index...')
        # anns stores 'segment_id -> annotation'
        anns, cats, imgs = {}, {}, {}
        relations = {}

        segments_info = {}

        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann, img_info in zip(self.dataset['annotations'],
                                     self.dataset['images']):
                img_info['segm_file'] = ann['file_name']
                for seg_ann in ann['segments_info']:
                    # to match with instance.json
                    seg_ann['image_id'] = ann['image_id']
                    seg_ann['height'] = img_info['height']
                    seg_ann['width'] = img_info['width']
                    img_to_anns[ann['image_id']].append(seg_ann)
                    # segment_id is not unique in coco dataset orz...
                    if seg_ann['id'] in anns.keys():
                        anns[seg_ann['id']].append(seg_ann)
                    else:
                        anns[seg_ann['id']] = [seg_ann]

                relations[ann['image_id']] = ann['relations']
                segments_info[ann['image_id']] = ann['segments_info']

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    cat_to_imgs[seg_ann['category_id']].append(ann['image_id'])

        print('index created!')

        self.anns = anns
        self.imgToAnns = img_to_anns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats
        self.relations = relations
        self.segments_info = segments_info
        self.relations_categories = self.dataset['relations_categories']
        self.relationID2Categories = {item['id']: item['name'] for item in self.dataset['relations_categories']}

import json
import re
from typing import Any, List, Optional

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

def main():
    args = parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    coco = COCOPanopticRelation('./data/psg_data/psg_val.json')
    img_ids = coco.get_img_ids()
    data_infos = []
    for i in img_ids:
        info = coco.load_imgs([i])[0]
        info['filename'] = info['file_name']
        data_infos.append(info)

    question = "Create a scene graph by identifying triplets of <subject, predicate, object>. Focus on the most prominent interactions."
    candidate_categories = [v['name'] for k, v in coco.cats.items()]
    candidate_predicates = [v['name'] for v in coco.relations_categories]
    candidate_categories_str = "{" + ", ".join(candidate_categories) + "}"
    candidate_predicates_str = "{" + ", ".join(candidate_predicates) + "}"
    candidate_str = "CANDIDATE CATEGORIES: \n" + candidate_categories_str + "\n" + "CANDIDATE PREDICATES: \n" + candidate_predicates_str + "\n"
    question = candidate_str + question

    rows = len(data_infos)
    chunk_size = (rows+15) // 16
    _start_ = args.task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = rows if _end_ > rows else _end_

    for data_info in tqdm.tqdm(data_infos[_start_:_end_]):
        # print(data_info) # {'file_name': 'val2017/000000195045.jpg', 'height': 480, 'width': 640, 'id': 107907, 'segm_file': 'panoptic_val2017/000000195045.png', 'filename': 'val2017/000000195045.jpg'}
        file_name = data_info['file_name']
        image_id = data_info['id']
        image_path = os.path.join("./data/coco", file_name)

        if os.path.exists(f"./temp_save/psg/{image_id}.json"):
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
            max_new_tokens=2048,
            do_sample=False,  # 关闭采样，使用贪婪解码
            top_p=1.0,  # 配合do_sample=False使用
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        print("Assistant: ", output_text[0])

        if not "```<|im_end|>" in output_text[0]:
            print("我还没说完呢！！！！")
            continue

        psg_list = parse_first_json(output_text[0])
        
        save_item = {
            'image_id': image_id,
            'file_name': file_name,
            'psg_list': psg_list,
        }

        with open(f"./temp_save/psg/{image_id}.json", 'w') as f:
            json.dump(save_item, f)

if __name__ == '__main__':
    main()