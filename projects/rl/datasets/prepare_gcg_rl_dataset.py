import os
import json
import tqdm
import random
import re
import argparse

from datasets import Dataset, Sequence
from datasets import Image as ImageData



def extract_answer_content(text):
    """
    Extract answer content from response text.
    Handles multiple formats:
    1. <answer>...</answer> tags -> extract content inside
    2. <think>...</think> followed by content -> extract content after </think>
    3. Plain text -> return as is
    """
    # Try <answer>...</answer> format first
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try <think>...</think> format - extract content after </think>
    think_pattern = r"</think>\s*\n*(.+)"
    match = re.search(think_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: return original text
    return text.strip() if text else None


GCG_QUESTIONS = [
    '<image>\nCould you please give me a detail description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    '<image>\nCan you provide a detail description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.',
    '<image>\nPlease describe the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    '<image>\nCould you give a detail explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
    '<image>\nCould you give me a detail explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
    '<image>\nCould you provide me with a detail analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
]


def generate_rl_data(items):
    """Generator function for creating RL dataset samples."""
    for item in items:
        yield {
            "images": item["images"],
            "cap_problem": item["cap_problem"],
            "cap_answer": item["cap_answer"],
            "seg_problem": item["seg_problem"],
            "seg_answer": item["seg_answer"],
            "masks": item["masks"],
            "source": item["source"],
        }


def main():
    parser = argparse.ArgumentParser(description="Prepare GCG RL dataset")
    parser.add_argument(
        "--input_json",
        type=str,
        default="./data/tokenmask_data_256x2_cot_format/mask_generation_gcg_grandf1k.json",
        help="Path to the input JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./rl_dataset",
        help="Output directory for parquet file"
    )
    parser.add_argument(
        "--source_name",
        type=str,
        default="gcg",
        help="Source name for the dataset"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading JSON from: {args.input_json}")
    with open(args.input_json, 'r') as f:
        json_data = json.load(f)

    print(f"Total samples in JSON: {len(json_data)}")

    processed_items = []
    
    for index in tqdm.tqdm(range(len(json_data)), desc="Processing samples"):
        data_dict = json_data[index]

        image_path = data_dict['image']
        from_human = data_dict['conversations'][0]['value']
        from_gpt = data_dict['conversations'][1]['value']

        # Build cap_problem: randomly sample from GCG_QUESTIONS
        cap_problem = random.choice(GCG_QUESTIONS)
        
        # Ensure <image> placeholder is present
        if "<image>" not in cap_problem:
            cap_problem = "<image>\n" + cap_problem

        # Build seg_answer: extract answer content from <answer>...</answer>
        answer_content = extract_answer_content(from_gpt)
        if answer_content:
            seg_answer = f"<answer>{answer_content}</answer>"
        else:
            # Fallback: use full response
            seg_answer = f"<answer>{from_gpt}</answer>"

        # Verify image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image path does not exist: {image_path}, skipping...")
            continue

        ret_data_dict = {
            'images': [image_path],  # Store image path as list
            'cap_problem': cap_problem,
            'cap_answer': None,  # Ground truth caption can be extracted if needed
            'seg_problem': None,  # No separate seg problem for GCG task
            'seg_answer': seg_answer,
            'masks': None,  # Masks are embedded in the answer as tokens
            'source': args.source_name,
        }

        processed_items.append(ret_data_dict)

    print(f"Processed {len(processed_items)} items!")
    
    # Create dataset
    # trainset = Dataset.from_generator(
    #     generate_rl_data, 
    #     gen_kwargs={"items": processed_items}
    # )
    trainset = Dataset.from_list(processed_items)
    # trainset = trainset.cast_column("images", Sequence(ImageData()))
    
    # Save to parquet
    output_filename = f"{args.source_name}_grandf1k_{len(processed_items)}_samples_train.parquet"
    output_path = os.path.join(args.output_dir, output_filename)
    trainset.to_parquet(output_path)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()