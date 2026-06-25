import os
import json
import tqdm
import random
from openai import OpenAI
import base64
import re

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
    with open('./cold_start_data/detail_gcg_cold_start_source.json', 'r') as f:
        data_dict_list = json.load(f)

    all_cold_start_data = []
    for json_file in os.listdir('./cold_start_data/coconut_dw'):
        json_path = os.path.join('./cold_start_data/coconut_dw', json_file)

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
            if '<|mt_start|>' not in cot_content:
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
    
    with open(f'./cold_start_data/coconut_dw_cold_start_data{len(all_cold_start_data)//1000}k.json', 'w') as f:
        json.dump(all_cold_start_data, f, indent=4)
    
    print(f"{len(all_cold_start_data)} items")

def load_dataset():
    GCG_QUESTIONS = [
        '<image>\nCould you please give me a detail description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
        '<image>\nCan you provide a detail description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.',
        '<image>\nPlease describe the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
        '<image>\nCould you give a detail explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
        '<image>\nCould you give me a detail explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
        '<image>\nCould you provide me with a detail analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
    ]
        
    with open('./data/coconut_dw_source.json', 'r') as f:
        source_items = json.load(f)

    rl_subset, left_items = split_list_random(source_items, k=4000, seed=42)
    cold_start_subset, _ = split_list_random(left_items, k=20000, seed=42)

    cold_start_items = []
    for source_item in cold_start_subset:
        image_path = source_item['image']
        image_caption = source_item['image_caption']
        mask_annotation = source_item['mask_annotation']

        mask_id_2_mask_token = {}
        for mask_id, mask_anno in mask_annotation.items():
            mask_id_2_mask_token[f'<obj_{mask_id}>'] = mask_anno['mask_token']
        
        for mask_id, mask_token in mask_id_2_mask_token.items():
            image_caption = image_caption.replace(mask_id, mask_token)

        conversation = []
        conversation.append({'from': 'human', 'value': random.choice(GCG_QUESTIONS)})
        conversation.append({'from': 'gpt', 'value': image_caption})

        ret_data_dict = {
            'image': image_path,
            'conversations': conversation,
        }
        cold_start_items.append(ret_data_dict)
    
    with open('./cold_start_data/detail_gcg_cold_start_source.json', 'w') as f:
        json.dump(cold_start_items, f, indent=4)
    
    with open('./cold_start_data/detail_gcg_rl_source.json', 'w') as f:
        json.dump(rl_subset, f, indent=4)

def main():
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        timeout=3600
    )

    PROMPT_TEMPLATE = "Given a grounded conversation generation (GCG) dialogue record: {GCG}\n"\
    "It contains a question and an answer. Your task is to simulate an intermediate chain-of-thought (CoT): after receiving the user's question, first think through the problem, and only then produce the final answer. \n"\
    "Overall requirements:\n"\
    "1. All visual references must be grounded in the visible content of the image; avoid speculation.\n"\
    "2. In the CoT, explain step by step how you locate objects, parse the scene, extract attributes and relationships, and clearly provide a short phrase description for each referenced object.\n"\
    "3. Treat any `<|mt_start|><|mt_xxxx|><|mt_xxxx|><|mt_end|>` sequence directly as the mask of the object.\n"\
    "4. In the CoT, you must use both the object mask information (`<|mt_start|><|mt_xxxx|><|mt_xxxx|><|mt_end|>`) and the phrase descriptions.\n"\
    "5. Wrap the simulated CoT within `<COT>...</COT>` to facilitate downstream parsing.\n"\
    "6. Don'r put the final answer inside `<COT>...</COT>`.\n"\
    "7. Forbidden any <tool_call> tags.\n"\

    with open('./cold_start_data/detail_gcg_cold_start_source.json', 'r') as f:
        data_dict_list = json.load(f)

    indices = list(range(0, len(data_dict_list)))
    for idx in tqdm.tqdm(indices):
        if os.path.exists(f'./cold_start_data/coconut_dw/{idx}.json'):
            continue
        data_dict = data_dict_list[idx]
        image = data_dict['image']
        conversations = data_dict['conversations']

        with open(image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        conversations_str = json.dumps(conversations, ensure_ascii=False)
        question = PROMPT_TEMPLATE.format(GCG=conversations_str)

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

        with open(f'./cold_start_data/coconut_dw/{idx}.json', 'w') as f:
            json.dump(save_item, f)

if __name__ == "__main__":
    # load_dataset()
    # main()
    clean_up()