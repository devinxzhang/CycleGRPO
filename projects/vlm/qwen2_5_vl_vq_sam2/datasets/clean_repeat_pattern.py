import os
import json
import re
import tqdm


regex_pattern = r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>"

anno_file = "./data/tokenmask_data_256x2/mask_generation_coconut_gcg129k.json"

with open(anno_file, 'r') as f:
    data_dict_list = json.load(f)

ret_data_dict_list = []
for data_dict in tqdm.tqdm(data_dict_list):
    image = data_dict['image']
    # print(data_dict['conversations'])
    # exit(0)
    from_human = data_dict['conversations'][0]['value']
    from_gpt = data_dict['conversations'][1]['value']

    # from_human = from_human.replace('Do not use \"<|object_ref_start|>...<|object_ref_end|>\" tags', '')

    mask_tokens = re.findall(regex_pattern, from_gpt)

    bad_case = False
    for mask_token in mask_tokens:
        if from_gpt.count(mask_token) > 3:
            bad_case = True
            break
    
    if bad_case:
        continue

    # conversations = []
    # conversations.append({'from': 'human', 'value': from_human})
    # conversations.append({'from': 'gpt', 'value': from_gpt})

    # ret_data_dict = {
    #     'image': image,
    #     'conversations': conversations
    # }

    # ret_data_dict_list.append(ret_data_dict)
    ret_data_dict_list.append(data_dict)

with open(f'./data/tokenmask_data_256x2/mask_generation_coconut_gcg{len(ret_data_dict_list)//1000}k_clean_repeat_pattern.json', 'w') as f:
    json.dump(ret_data_dict_list, f, indent=4)


# def extract_scene_graph_dicts(text):
#     pattern = r'''
#     \{
#     \s*"subject"\s*:\s*\{
#         \s*"mask_2d"\s*:\s*"[^"]*"\s*,\s*
#         \s*"label"\s*:\s*"[^"]*"\s*
#     \}\s*,\s*
#     \s*"predicate"\s*:\s*"[^"]*"\s*,\s*
#     \s*"object"\s*:\s*\{
#         \s*"mask_2d"\s*:\s*"[^"]*"\s*,\s*
#         \s*"label"\s*:\s*"[^"]*"\s*
#     \}
#     \s*\}
#     '''
#     matches = re.findall(pattern, text, flags=re.DOTALL | re.VERBOSE)

#     dicts, text_triplets = [], []
#     for m in matches:
#         try:
#             d = json.loads(m)
#             dicts.append(d)
#             text_triplets.append(m)
#         except json.JSONDecodeError:
#             pass
#     return dicts, text_triplets

# anno_file = "./data/tokenmask_data_256x2/mask_generation_psg45k_v4.json"

# with open(anno_file, 'r') as f:
#     data_dict_list = json.load(f)

# ret_data_dict_list = []
# for data_dict in tqdm.tqdm(data_dict_list):
#     conversations = data_dict['conversations']
#     # print(conversations)
#     # exit(0)
#     from_human = conversations[1]['value']
#     from_gpt = conversations[3]['value']

#     mask_tokens = re.findall(regex_pattern, from_human)
#     bad_case = False
#     for mask_token in mask_tokens:
#         if from_human.count(mask_token) > 1:
#             bad_case = True
#             break
    
#     if bad_case:
#         continue

#     triplets, text_triplets = extract_scene_graph_dicts(from_gpt)
#     text_triplet_dict = {}
#     for text_triplet in text_triplets:
#         if text_triplet not in text_triplet_dict:
#             text_triplet_dict[text_triplet] = 0
#         text_triplet_dict[text_triplet] += 1

#     if max(list(text_triplet_dict.values())) > 1:
#         continue
    
#     ret_data_dict_list.append(data_dict)

# with open(f"./data/tokenmask_data_256x2/mask_generation_psg{len(ret_data_dict_list)//1000}k_v4_clean_repeat_pattern.json", 'w') as f:
#     json.dump(ret_data_dict_list, f, indent=4)



# from collections import Counter

# # 简单 tokenizer（中英混合兼容）
# _token_pattern = re.compile(r"\w+")

# def tokenize(text: str):
#     return _token_pattern.findall(text.lower())

# def has_repetition(tokens, n_min=3, n_max=4, threshold=0.30):
#     """
#     返回 True 表示自我复读。
#     threshold 默认 0.25，可按需调。
#     """
#     L = len(tokens)
#     if L < n_min:
#         return False

#     for n in range(n_min, n_max + 1):
#         if L < n:
#             continue

#         # 快速生成 n-grams
#         ngrams = [tuple(tokens[i:i+n]) for i in range(L - n + 1)]

#         # 用 Counter 强检重复
#         counter = Counter(ngrams)

#         # 计算重复 n-gram 的 token 总数
#         repeated_tokens = 0
#         for c in counter.values():
#             if c > 1:
#                 repeated_tokens += (c - 1) * n

#         if repeated_tokens / L >= threshold:
#             return True

#     return False


# def is_repetitive(response: str) -> bool:
#     """高效版：只返回 True/False。"""
#     tokens = tokenize(response)
#     return has_repetition(tokens)


# anno_file = "./data/tokenmask_data_256x2/mask_understanding_sam_zoom_in_2181k.json"
# with open(anno_file, 'r') as f:
#     data_dict_list = json.load(f)

# ret_data_dict_list = []
# for data_dict in tqdm.tqdm(data_dict_list):
#     from_gpt = data_dict['conversations'][1]['value']
#     if is_repetitive(from_gpt):
#         continue

#     ret_data_dict_list.append(data_dict)

# with open(f"./data/tokenmask_data_256x2/mask_understanding_sam_zoom_in_{len(ret_data_dict_list)//1000}k_clean_repeat_pattern.json", 'w') as f:
#     json.dump(ret_data_dict_list, f, indent=4)

