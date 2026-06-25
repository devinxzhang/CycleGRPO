import os
import json
import tqdm
import time
from openai import OpenAI
import base64
import re

import random
def split_list_random(lst, k=4000, seed=None):
    """
    从 lst 中随机选取 k 个元素作为子集A，剩余作为子集B。
    不放回抽样，A 与 B 不重叠。
    """
    if seed is not None:
        random.seed(seed)  # 可复现实验
    n = len(lst)
    if k > n:
        raise ValueError(f"k={k} 大于列表长度 n={n}")
    indices = random.sample(range(n), k)  # 随机挑 k 个索引
    indices_set = set(indices)
    subset_A = [lst[i] for i in indices]
    subset_B = [lst[i] for i in range(n) if i not in indices_set]
    return subset_A, subset_B

def clean_up():
    QUESTION_TEMPLATE = "{content} A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
    ANSWER_TEMPLATE = "<think> {thinking} </think><answer> {answer} </answer>"
    with open('./cold_start_data/gres_no_target_source.json', 'r') as f:
        data_dict_list = json.load(f)

    all_cold_start_data = []
    for json_file in os.listdir('./cold_start_data/gres_no_targets'):
        json_path = os.path.join('./cold_start_data/gres_no_targets', json_file)

        index = int(json_file.split('.')[0])

        skip_this_case = False
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            raw_cot = json_data['raw_cot']

        pattern = re.compile(r"<COT>(.*?)</COT>", re.S)  # 非贪婪 + 跨行
        matches = pattern.findall(raw_cot)
        if matches:
            # 去掉首尾可能的换行
            cot_content = matches[0].strip()
            if '<tool_call>' in cot_content:
                skip_this_case = True
        else:
            skip_this_case = True
        
        if skip_this_case:
            continue

        data_dict = data_dict_list[index]

        from_human = QUESTION_TEMPLATE.format(content=data_dict['conversations'][0]['value'])
        from_gpt = ANSWER_TEMPLATE.format(thinking=cot_content, answer=data_dict['conversations'][1]['value'])

        conversations = []
        conversations.append({'from': 'human', 'value': from_human})
        conversations.append({'from': 'gpt', 'value': from_gpt})
        ret_data_dict = {
            'image': data_dict['image'],
            'conversations': conversations,
        }
        all_cold_start_data.append(ret_data_dict)
    
    with open(f'./cold_start_data/gres_no_target_cold_start_data{len(all_cold_start_data)//1000}k.json', 'w') as f:
        json.dump(all_cold_start_data, f, indent=4)
    
    print(f"{len(all_cold_start_data)} items")


def load_dataset():
    image_file_item_dict = {}
    with open('./data/tokenmask_data_256x2/mask_generation_grefcoco209k_new.json', 'r') as f:
        data_dict_list = json.load(f)
    
    for data_dict in data_dict_list:
        image = data_dict['image']
        if image not in image_file_item_dict:
            image_file_item_dict[image] = []
        image_file_item_dict[image].append(data_dict)

    rl_subset_keys, cold_start_subset_keys = split_list_random(list(image_file_item_dict.keys()), k=4000, seed=42)
    # rl_subset = {k: v for k, v in image_file_item_dict.items() if k in rl_subset_keys}
    # cold_start_subset = {k: v for k, v in image_file_item_dict.items() if k in cold_start_subset_keys}
    random.seed(42)
    rl_subset = [random.choice(v) for k, v in image_file_item_dict.items() if k in rl_subset_keys]
    cold_start_subset = [random.choice(v) for k, v in image_file_item_dict.items() if k in cold_start_subset_keys]

    with open('./cold_start_data/gres/cold_start_source.json', 'w') as f:
        json.dump(cold_start_subset, f, indent=4)
    
    with open('./cold_start_data/gres/rl_source.json', 'w') as f:
        json.dump(rl_subset, f, indent=4)

def load_zero_targets():
    image_file_item_dict = {}
    with open('./data/tokenmask_data_256x2/mask_generation_grefcoco209k_new.json', 'r') as f:
        data_dict_list = json.load(f)
    
    for data_dict in data_dict_list:
        image = data_dict['image']
        if image not in image_file_item_dict:
            image_file_item_dict[image] = []
        image_file_item_dict[image].append(data_dict)

    rl_subset_keys, cold_start_subset_keys = split_list_random(list(image_file_item_dict.keys()), k=4000, seed=42)
    all_zero_target_items = []
    for image_k in cold_start_subset_keys:
        image_items = image_file_item_dict[image_k]
        for item in image_items:
            from_gpt = item['conversations'][1]['value']
            if '<|mt_start|>' not in from_gpt:
                all_zero_target_items.append(item)
    
    with open("./cold_start_data/gres_no_target_source.json", 'w') as f:
        json.dump(all_zero_target_items, f, indent=4)

    print(f"{len(all_zero_target_items)} items!")    

def main():
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        timeout=3600
    )

    PROMPT_TEMPLATE = "Given a grounding dialogue record: {GRES}\n"\
    "It contains a question and an answer. Your task is to simulate an intermediate chain-of-thought (CoT): after receiving the user's question, first think through the problem, and only then produce the final answer. \n"\
    "Overall requirements:\n"\
    "1. All visual references must be grounded in the visible content of the image; avoid speculation.\n"\
    "2. Analyze all objects in the image carefully.\n"\
    "3. Compare candidates against the target description.\n"\
    "4. Treat any `<|mt_start|><|mt_****|>...<|mt_end|>` sequence directly as the mask of the targets.\n"\
    "5. Wrap the simulated CoT within `<COT>...</COT>` to facilitate downstream parsing.\n"\
    "6. Forbidden any <tool_call> tags.\n"



    # with open('./cold_start_data/gres_cold_start_source.json', 'r') as f:
    #     data_dict_list = json.load(f)
    with open('./cold_start_data/gres_no_target_source.json', 'r') as f:
        data_dict_list = json.load(f)
    
    indices = list(range(0, len(data_dict_list)))
    for idx in tqdm.tqdm(indices):
        if os.path.exists(f'./cold_start_data/gres_no_targets/{idx}.json'):
            continue
        data_dict = data_dict_list[idx]
        image = data_dict['image']
        conversations = data_dict['conversations']

        with open(image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        conversations_str = json.dumps(conversations, ensure_ascii=False)
        question = PROMPT_TEMPLATE.format(GRES=conversations_str)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": question,
                    }
                ]
            }
        ]

        response = client.chat.completions.create(
            model="Qwen/Qwen3-VL-235B-A22B-Instruct",
            messages=messages,
            max_tokens=2048
        )
        cot_str = response.choices[0].message.content
        save_item = {'raw_cot': cot_str}

        with open(f'./cold_start_data/gres_no_targets/{idx}.json', 'w') as f:
            json.dump(save_item, f)

if __name__ == '__main__':
    # main()
    clean_up()
    # load_dataset()
    # load_zero_targets()
