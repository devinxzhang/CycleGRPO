import os
import json
import tqdm
import random

def main():
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
    
    all_data_dict = []
    for source_item in tqdm.tqdm(source_items):
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
        all_data_dict.append(ret_data_dict)
    
    with open(f'./data/tokenmask_data_256x2/mask_generation_coconut_gcg{len(all_data_dict)//1000}k.json', 'w') as f:
        json.dump(all_data_dict, f, indent=4)

if __name__ == "__main__":
    main()