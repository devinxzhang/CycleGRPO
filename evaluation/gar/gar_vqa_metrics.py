import argparse
import json
import re


def clean_text(s):
    s = re.sub(r"<think>\n\n</think>\n\n", "", s, flags=re.IGNORECASE).strip()
    # Remove trailing '.' if present (e.g., 'D.' -> 'D')
    if s.endswith("."):
        s = s[:-1].strip()
    return s


def is_correct(record):
    return clean_text(record["model_output"]).lower() == clean_text(record["answer"]).lower()


def main():
    parser = argparse.ArgumentParser(description="Compute VQA metrics from model output JSON.")
    parser.add_argument("json_path", type=str, help="Path to the model output JSON file")
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        model_outputs = json.load(f)

    # directly compute accuracy using exact-matching
    for category in {x["type"] for x in model_outputs}:
        results = [x for x in model_outputs if x["type"] == category]
        total = len(results)
        correct = sum(is_correct(x) for x in results)
        print(f"{category}: [{correct}/{total}]={round(correct / total * 100, 1)}")

    total = len(model_outputs)
    correct = sum(is_correct(x) for x in model_outputs)
    print(f"=> overall: [{correct}/{total}]={round(correct / total * 100, 1)}")


if __name__ == "__main__":
    main()
