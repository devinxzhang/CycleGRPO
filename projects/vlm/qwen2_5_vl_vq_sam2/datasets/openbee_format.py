import os
import json
import tqdm


source_root = "<PATH_TO_DATA>/MaskTokenizer/data/tokenmask_data_256x2"
target_root = "<PATH_TO_DATA>/MaskTokenizer/data/tokenmask_data_256x2_cot_format"

for json_file in os.listdir(source_root):
    if not json_file.endswith('.json'):
        continue
    if 'cold_start' in json_file:
        continue

    json_path = os.path.join(source_root, json_file)
    target_path = os.path.join(target_root, json_file)

    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    ret_data_dict_list = []
    for data_dict in tqdm.tqdm(json_data):
        image = data_dict['image']
        conversations = data_dict['conversations']

        new_conversations = []
        for conv in conversations:
            if conv['from'] == 'human':
                new_conversations.append(conv)
            else:
                assert conv['from'] == 'gpt'
                answer = "<think>\n\n</think>\n\n" + conv['value']
                new_conversations.append({"from": 'gpt', 'value': answer})
        ret_data_dict = {
            "image": image,
            "conversations": new_conversations
        }
        ret_data_dict_list.append(ret_data_dict)
    
    with open(target_path, 'w') as f:
        json.dump(ret_data_dict_list, f, indent=4)
