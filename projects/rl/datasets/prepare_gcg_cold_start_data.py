import os
import json
import tqdm
import time
from openai import OpenAI
import base64
import re

def clean_up():
    QUESTION_TEMPLATE = "{content} A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
    ANSWER_TEMPLATE = "<think> {thinking} </think><answer> {answer} </answer>"
    with open('./data/tokenmask_data_256x2/mask_generation_gcg_exclude_grandf195k.json', 'r') as f:
        data_dict_list = json.load(f)

    all_cold_start_data = []
    for json_file in os.listdir('./cold_start_data/gcg'):
        json_path = os.path.join('./cold_start_data/gcg', json_file)

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
    
    with open(f'./cold_start_data/gcg_cold_start_data{len(all_cold_start_data)//1000}k.json', 'w') as f:
        json.dump(all_cold_start_data, f, indent=4)
    
    print(f"{len(all_cold_start_data)} items")

def filter_no_evidence_items(source_file):
    with open(source_file, 'r') as f:
        source_items = json.load(f)
    save_items = []
    for item in source_items:
        from_gpt = item['conversations'][1]['value']
        pattern = re.compile(r"<think>(.*?)</think>", re.S)  # 非贪婪 + 跨行
        matches = pattern.findall(from_gpt)
        if matches:
            # 去掉首尾可能的换行
            cot_content = matches[0].strip()
            if '<|mt_start|>' in cot_content:
                save_items.append(item)
        else:
            continue
    
    with open(source_file.replace('.json', '_evidence.json'), 'w') as f:
        json.dump(save_items, f, indent=4)

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
    "3. Treat any `<|mt_start|><|mt_****|>...<|mt_end|>` sequence directly as the mask of the object wrapped by the phrase `<|object_ref_start|> ... <|object_ref_end|>`.\n"\
    "4. In the CoT, you must use both the object mask information (`<|mt_start|><|mt_****|>...<|mt_end|>`) and the phrase descriptions.\n"\
    "5. Wrap the simulated CoT within `<COT>...</COT>` to facilitate downstream parsing.\n"\
    "6. Don'r put the final answer inside `<COT>...</COT>`.\n"\
    "7. Forbidden any <tool_call> tags.\n"\

    with open('./data/tokenmask_data_256x2/mask_generation_gcg_exclude_grandf195k.json', 'r') as f:
        data_dict_list = json.load(f)
    
    stride = len(data_dict_list) // 10000
    indices = list(range(0, len(data_dict_list), stride))
    for idx in tqdm.tqdm(indices):
        if os.path.exists(f'./cold_start_data/gcg/{idx}.json'):
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

        with open(f'./cold_start_data/gcg/{idx}.json', 'w') as f:
            json.dump(save_item, f)

if __name__ == '__main__':
    # main()
    # clean_up()
    # filter_no_evidence_items('./cold_start_data/gcg_cold_start_data10k.json')
    # filter_no_evidence_items('./cold_start_data/gcg_cold_start_data2k.json')

    with open("./cold_start_data/gcg_cold_start_data2k_evidence.json", 'r') as f:
        subset1 = json.load(f)
    with open('./cold_start_data/gcg_cold_start_data10k_evidence.json', 'r') as f:
        subset2 = json.load(f)
    
    image_file_2_items = {}
    for item in subset1:
        if item['image'] not in image_file_2_items:
            image_file_2_items[item['image']] = item
    for item in subset2:
        if item['image'] not in image_file_2_items:
            image_file_2_items[item['image']] = item
    left_items = list(image_file_2_items.values())
    reformat_question_items = []
    for _, item in image_file_2_items.items():
        from_human = item['conversations'][0]['value']
        from_gpt = item['conversations'][1]['value']
        from_human = from_human.replace('. A conversation between User and Assistant.', ' and highlight the phrases. A conversation between User and Assistant.')
        conversations = []
        conversations.append({'from': 'human', 'value': from_human})
        conversations.append({'from': 'gpt', 'value': from_gpt})
        ret_data_dict = {
            'image': item['image'],
            'conversations': conversations,
        }
        reformat_question_items.append(ret_data_dict)

    with open(f'./cold_start_data/gcg_cold_start_data_with_evidence_{len(reformat_question_items)//1000}k.json', 'w') as f:
        json.dump(reformat_question_items, f, indent=4)
