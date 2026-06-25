import os
import json
import tqdm

if __name__ == "__main__":
    with open('./data/tokenmask_data_256x2/mask_generation_grefcoco209k.json', 'r') as f:
        all_data_dict = json.load(f)
    
    all_save_data_dict = []
    for data_dict in tqdm.tqdm(all_data_dict):
        from_human = data_dict['conversations'][0]['value']
        from_gpt = data_dict['conversations'][1]['value']

        token_count = from_gpt.count('<|mt_start|>')
        if token_count <= 1 and 'one of the ' in from_gpt:
            from_gpt = from_gpt.replace('one of the ', '')
        
        conversation = []
        conversation.append({'from': 'human', 'value': from_human})
        conversation.append({'from': 'gpt', 'value': from_gpt})

        save_item = {
            'image': data_dict['image'],
            'conversations': conversation
        }
        all_save_data_dict.append(save_item)
    with open('./data/tokenmask_data_256x2/mask_generation_grefcoco209k_new.json', 'w') as f:
        json.dump(all_save_data_dict, f, indent=4)
