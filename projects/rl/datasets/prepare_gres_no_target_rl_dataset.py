import os
import re
import json
import argparse
from typing import List
from PIL import Image
import tqdm

from datasets import Dataset, Sequence
from datasets import Image as ImageData


def extract_mt_token_ids(text):
    """Extract mask token ids from text."""
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]


def extract_answer_content(text):
    """Extract content within <answer>...</answer> tags."""
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


COLOR_LIST = [
    "red", "orange", "yellow", "green", "blue", "purple", "black", "white", 
    "gray", "brown", "pink", "cyan", "magenta", "navy", "maroon", "teal", 
    "olive", "lime", "aqua", "silver", "gold", "beige", "cream", "indigo", 
    "violet", "crimson", "scarlet", "turquoise", "lavender", "plum",
    "light blue", "dark green", "sky blue"
]


def find_colors_regex_english(text):
    """
    Finds color words in an English string using a robust regex pattern.
    Handles word boundaries and is case-insensitive.
    """
    sorted_colors = sorted(COLOR_LIST, key=len, reverse=True)
    pattern = r'\b(' + '|'.join(re.escape(color) for color in sorted_colors) + r')\b'
    found_colors = re.findall(pattern, text, re.IGNORECASE)
    return list(dict.fromkeys(color.lower() for color in found_colors))


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
    parser = argparse.ArgumentParser(description="Prepare GRES no-target RL dataset")
    parser.add_argument(
        "--input_json",
        type=str,
        default="./data/tokenmask_data_256x2_cot_format/gres_no_target_cold_start_data14k.json",
        help="Path to the input JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./rl_dataset",
        help="Output directory for parquet file"
    )
    parser.add_argument(
        "--filter_by_color",
        action="store_true",
        help="If set, only keep samples that mention colors"
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

        # Optional: filter by color mentions
        if args.filter_by_color:
            colors_regex = find_colors_regex_english(from_human)
            if len(colors_regex) == 0:
                continue

        # Build cap_problem: the user's question (already contains <image> placeholder)
        # Remove the CoT instruction suffix if present
        cap_problem = from_human
        cot_instruction = " A conversation between User and Assistant."
        if cot_instruction in cap_problem:
            cap_problem = cap_problem.split(cot_instruction)[0].strip()
        
        # Ensure <image> placeholder is present
        if "<image>" not in cap_problem:
            cap_problem = "<image>\n" + cap_problem

        # Build seg_answer: extract answer content or use "No target."
        answer_content = extract_answer_content(from_gpt)
        if answer_content:
            seg_answer = f"<answer>{answer_content}</answer>"
        else:
            # Fallback: check if it's a "No target" response
            if "No target" in from_gpt or "no target" in from_gpt.lower():
                seg_answer = "<answer>No target.</answer>"
            else:
                seg_answer = f"<answer>{from_gpt}</answer>"

        assert os.path.exists(image_path), f"Image path does not exist: {image_path}"

        ret_data_dict = {
            'images': [image_path],  # Store image path as list; actual loading done in data pipeline
            'cap_problem': cap_problem,
            'cap_answer': None,  # No ground truth caption for this task
            'seg_problem': None,  # No separate seg problem for no-target task
            'seg_answer': seg_answer,
            'masks': None,  # No masks for no-target samples
            'source': 'gres_no_target',
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
    output_filename = f"gres_no_target_{len(processed_items)}_samples_train.parquet"
    output_path = os.path.join(args.output_dir, output_filename)
    trainset.to_parquet(output_path)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()    