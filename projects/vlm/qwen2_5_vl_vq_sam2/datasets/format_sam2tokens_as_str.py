import os
import json
import tqdm

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

import random
def insert_random_elements(list_a, list_b, num_to_insert):
    """
    从 list_b 中随机选择指定数量的不在 list_a 中的元素，
    然后随机插入到 list_a 中，并保持 list_a 原始元素的相对顺序。
    :param list_a: 原始列表 A
    :param list_b: 从中选择元素的列表 B
    :param num_to_insert: 希望插入的元素数量
    :return: 一个新的、修改后的列表
    """
    # 1. 找出 B 中不在 A 中的元素
    elements_to_choose_from = list(set(list_b) - set(list_a))
    if not elements_to_choose_from:
        # print("列表B中没有不在列表A中的新元素可供选择。")
        return list_a.copy()
    # 2. 随机选择 num_to_insert 个元素
    # 确保选择的数量不超过可选元素的总数
    k = min(num_to_insert, len(elements_to_choose_from))
    elements_to_insert = random.sample(elements_to_choose_from, k)
    # print(f"从差异元素 {elements_to_choose_from} 中选择了: {elements_to_insert}")
    # 3. 随机插入到 A 的副本中
    new_list = list_a.copy()
    for element in elements_to_insert:
        # 随机选择一个插入位置
        random_index = random.randint(0, len(new_list))
        new_list.insert(random_index, element)
    return new_list

QUESTION_LIST = [
    "<image>\nSegment every instance that belongs to the following categories: {class_name}",
    "<image>\nLocate every instance that belongs to the following categories: {class_name}. Report segmentation masks in JSON format."
]

def main():
    source_path_list = [
        './data/vq_sam2_data_256x2_0927/mask_generation_refseg641k.json',
        './data/vq_sam2_data_256x2_0927/mask_generation_invig505k.json',
        './data/vq_sam2_data_256x2_0927/mask_generation_gcg_exclude_grandf195k.json',
        './data/vq_sam2_data_256x2_0927/mask_generation_gcg_grandf1k.json',
        './data/vq_sam2_data_256x2_0927/mask_generation_v3det183k.json',
        './data/vq_sam2_data_256x2_0927/mask_understanding_dam1458k.json',
        './data/vq_sam2_data_256x2_0927/mask_generation_denseworld900k.json',
        './data/vq_sam2_data_256x2_0927/mask_generation_coconut426k.json',
        './data/vq_sam2_data_256x2_0927/mask_generation_padt_refcoco321k.json',
        './data/vq_sam2_data_256x2_0927/mask_generation_padt_ric561k.json',
        './data/vq_sam2_data_256x2_0927/mask_generation_padt_cocoseg118k.json',
        './data/vq_sam2_data_256x2_0927/mask_generation_segllm1049k.json',
        './data/vq_sam2_data_256x2_0927/mask_generation_grefcoco209k.json',
    ]

    from pycocotools.coco import COCO
    data_path = "./data/V3Det/v3det_2023_v1_train.json"
    coco_api = COCO(data_path)
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    v3det_class_name = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]

    from projects.vlm.qwen2_5_vl_vq_sam2.datasets.coconut_meta import COCO_META
    coconut_class_name = [meta['name'] for meta in COCO_META[1:]]

    for source_path in tqdm.tqdm(source_path_list):
        with open(source_path, 'r') as f:
            source_data_dict_list = json.load(f)

        result_data_dict_list = []
        for data_dict in source_data_dict_list:
            image = data_dict['image']
            conversations = data_dict['conversations']
            
            result_conversations = []
            for turn in conversations:
                role = turn['from']
                content = turn['value']
                if "\"mask_2d\"" in content:
                    content = content.replace("<|mt_start|>", "\"<|mt_start|>").replace("<|mt_end|>", "<|mt_end|>\"")
                
                result_conversations.append({'from': role, 'value': content})
            conversations = result_conversations

            if 'v3det' in source_path:
                assert len(conversations) == 2
                assert conversations[1]['from'] == 'gpt'
                answer = conversations[1]['value']
                if "\"\"" in answer:
                    answer = answer.replace("\"\"", "\"")
                try:
                    parsed_answer = parse_first_json(answer)
                except:
                    print(answer)
                    exit(0)
                if not isinstance(parsed_answer, list):
                    parsed_answer = [parsed_answer]
                try:
                    list_a = list(set([item['label'] for item in parsed_answer]))
                except:
                    print(parsed_answer)
                    exit(0)
                random_num_to_insert = random.randint(0, 10)

                new_candidate_classes = insert_random_elements(list_a, v3det_class_name, random_num_to_insert)

                category_name_str = ', '.join(new_candidate_classes)
                question = random.choice(QUESTION_LIST).format(class_name=category_name_str)
                result_conversations = []
                result_conversations.append({'from': 'human', 'value': question})
                result_conversations.append({'from': 'gpt', 'value': answer})
            elif 'coconut' in source_path:
                assert len(conversations) == 2
                assert conversations[1]['from'] == 'gpt'
                answer = conversations[1]['value']
                if "\"\"" in answer:
                    answer = answer.replace("\"\"", "\"")
                parsed_answer = parse_first_json(answer)
                if not isinstance(parsed_answer, list):
                    parsed_answer = [parsed_answer]

                list_a = list(set([item['label'] for item in parsed_answer]))
                random_num_to_insert = random.randint(0, 10)

                new_candidate_classes = insert_random_elements(list_a, coconut_class_name, random_num_to_insert)

                category_name_str = ', '.join(new_candidate_classes)
                question = random.choice(QUESTION_LIST).format(class_name=category_name_str)
                result_conversations = []
                result_conversations.append({'from': 'human', 'value': question})
                result_conversations.append({'from': 'gpt', 'value': answer})

            result_data_dict_list.append(
                {
                    'image': image,
                    'conversations': result_conversations,
                }
            )
                
        basename = os.path.basename(source_path)
        with open(os.path.join('./data/tokenmask_data_256x2', basename), 'w') as f:
            json.dump(result_data_dict_list, f, indent=4)

if __name__ == "__main__":
    main()