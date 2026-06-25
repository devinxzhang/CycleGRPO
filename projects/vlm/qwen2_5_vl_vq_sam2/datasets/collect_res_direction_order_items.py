import os
import json
import tqdm
import re
from typing import Dict, Any, List, Tuple
DIRECTION_WORDS = {
    "left", "right", "top", "bottom", "upper", "lower", "middle", "center", "centre",
    "front", "back", "behind",
}
ORDINAL_WORDS = {
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
}
PHRASE_PATTERNS = [
    r"\bin\s+front\s+of\b",
    r"\bon\s+the\s+left\s+side\b",
    r"\bon\s+the\s+right\s+side\b",
    r"\bat\s+the\s+top\b",
    r"\bat\s+the\s+bottom\b",
    r"\bupper[-\s]?left\b",
    r"\bupper[-\s]?right\b",
    r"\blower[-\s]?left\b",
    r"\blower[-\s]?right\b",
    r"\btop[-\s]?left\b",
    r"\btop[-\s]?right\b",
    r"\bbottom[-\s]?left\b",
    r"\bbottom[-\s]?right\b",
    r"\bmiddle\s+row\b",
    r"\bfront\s+row\b",
    r"\bback\s+row\b",
    r"\bfar\s+left\b",
    r"\bfar\s+right\b",
    r"\bto\s+the\s+left\s+of\b",
    r"\bto\s+the\s+right\s+of\b",
]
ORDINAL_PHRASES = [
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+from\s+the?\s+(left|right)\b",
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)[-\s]+from[-\s]+(left|right)\b",
    r"\b(second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+from\s+(left|right)\b",
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(row|column)\b",
]
ORDINAL_NTH_PATTERNS = [
    r"\b\d+(st|nd|rd|th)\b",
    r"\b\d+(st|nd|rd|th)\s+from\s+the?\s+(left|right)\b",
    r"\b\d+(st|nd|rd|th)\s+(row|column)\b",
]
# 可选：支持 rightmost/leftmost/lowermost/uppermost 等词
MOST_PATTERNS = [
    r"\b(right|left|top|bottom|upper|lower)(?:-)?most\b",
]
def detect_direction_or_order(text: str) -> Dict[str, Any]:
    result = {
        "has_direction_or_order": False,
        "matches": []  # List[Dict[str, Any]]: type, span, text
    }
    if not text:
        return result
    s = text.lower()
    def add_matches(pats: List[str], typ: str):
        for pat in pats:
            for m in re.finditer(pat, s):
                result["matches"].append({
                    "type": typ,
                    "span": m.span(),
                    "text": s[m.start():m.end()],
                    "pattern": pat
                })
    # 短语优先
    add_matches(PHRASE_PATTERNS, "direction_phrase")
    add_matches(ORDINAL_PHRASES, "ordinal_phrase")
    add_matches(ORDINAL_NTH_PATTERNS, "ordinal_nth")
    add_matches(MOST_PATTERNS, "direction_most")
    # 单词
    word_pat = r"\b(" + "|".join(map(re.escape, (DIRECTION_WORDS | ORDINAL_WORDS))) + r")\b"
    for m in re.finditer(word_pat, s):
        result["matches"].append({
            "type": "word",
            "span": m.span(),
            "text": s[m.start():m.end()],
            "pattern": "WORD_SET"
        })
    result["has_direction_or_order"] = len(result["matches"]) > 0
    return result

def main():
    with open('./data/ablation_data/mask_generation_padt_refcoco_512x2_321k.json', 'r') as f:
        data_dict_list = json.load(f)
    
    direction_order_data_dict_list = []
    for data_dict in tqdm.tqdm(data_dict_list):
        question = data_dict['conversations'][0]['value']
        det_result = detect_direction_or_order(question)
        if det_result['has_direction_or_order']:
            direction_order_data_dict_list.append(data_dict)
    num_samples = len(direction_order_data_dict_list)

    with open(f'./data/ablation_data/mask_generation_padt_refcoco_512x2_direction_order_{num_samples//1000}k.json', 'w') as f:
        json.dump(direction_order_data_dict_list, f, indent=4)
    print(f"{num_samples} items!")

if __name__ == '__main__':
    main()
