import re
from typing import Any, Tuple, Optional
import torch
import torch.nn.functional as F
import json
import hydra
import os
import numpy as np
from PIL import Image
from collections import Counter
from nltk.stem.porter import PorterStemmer
from pycocotools import mask as mask_utils
from pycocoevalcap.cider.cider import Cider
from torchvision.transforms.functional import to_pil_image
from projects.transformers.vq_sam2 import SAM2Config, VQ_SAM2Config, VQ_SAM2
from generalized.verl.utils.spatiotemporal import (
    ATOMIC_SPATIOTEMPORAL_SOURCE,
    SPATIOTEMPORAL_SOURCE,
    TRAJECTORY_SPATIOTEMPORAL_SOURCE,
    TRAJECTORY_UNDERSTANDING_SOURCE,
    TRAJECTORY_LOCALIZATION_SOURCE,
    atomic_spatiotemporal_format_reward,
    spatiotemporal_caption_no_leak_reward,
    spatiotemporal_format_reward,
    trajectory_spatiotemporal_format_reward,
    trajectory_spatiotemporal_reward,
)
from generalized.verl.utils.cyclegrpo_profiles import OPEN_O3_UNDERSTANDING_SOURCE

_MODEL_CACHE = {'vq_sam2': None, 'sam2_image_processor': None}

from generalized.reward_function.llm_judge_reward import (
    init_judge_clients,
    compute_caption_judge_rewards_batch,
    compute_graded_caption_judge_rewards_batch,
)

# 模块加载时初始化一次。从 env vars 读 endpoint:
#   export LLM_AS_A_JUDGE_BASES="http://127.0.0.1:8000/v1"
#   export LLM_AS_A_JUDGE_MODEL="qwen-judge"
init_judge_clients()


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))


def extract_mt_token_ids_v1(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]


def mask_token_to_mask(quant_ids, vq_sam2, sam2_image_processor, image):
    batch_size = len(quant_ids) // vq_sam2.config.codebook_depth
    remap_quant_ids = []
    for bs_id in range(batch_size):
        chunk_quant_ids = quant_ids[bs_id*vq_sam2.config.codebook_depth:(bs_id+1)*vq_sam2.config.codebook_depth]
        remap_chunk_quant_ids = [quant_id - book_id*vq_sam2.config.codebook_size for book_id, quant_id in enumerate(chunk_quant_ids)]
        code1 = remap_chunk_quant_ids[0]
        code2 = remap_chunk_quant_ids[1]
        if not (code1 >= 0 and code1 < vq_sam2.config.codebook_size):
            continue
        if not (code2 >= 0 and code2 < vq_sam2.config.codebook_size):
            code2 = -1
        remap_chunk_quant_ids_error_handle = [code1, code2]
        remap_quant_ids.append(remap_chunk_quant_ids_error_handle)

    batch_size = len(remap_quant_ids)
    ori_width, ori_height = image.size
    sam2_image = np.array(image)
    sam2_image = sam2_image_processor.apply_image(sam2_image)
    sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
    sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

    quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

    with torch.no_grad():
        _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
    _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
    _pred_masks = _pred_masks > 0.5
    _pred_masks = _pred_masks[0, 0, :, :]

    return _pred_masks


def extract_mt_token_ids_v2(text):
    pattern = re.compile(r'<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>')
    matches = pattern.findall(text)
    ret_list = []
    for num1, num2 in matches:
        ret_list.append(int(num1))
        ret_list.append(int(num2))
    return ret_list

def fix_mt_format_comprehensive(text):
    """
    全面修正 <|mt_...> 格式的函数。
    它会处理以下几种情况：
    1. 标记太少 (1个): <|mt_start|><|mt_0198|><|mt_end|> -> <|mt_start|><|mt_0198|><|mt_-1|><|mt_end|>
    2. 标记太少 (1个, 无end): <|mt_start|><|mt_0198|> -> <|mt_start|><|mt_0198|><|mt_-1|><|mt_end|>
    3. 标记太多 (3个或以上): <|mt_start|><|mt_0186|><|mt_0410|><|mt_0186|><|mt_end|> -> <|mt_start|><|mt_0186|><|mt_0410|><|mt_end|>
    4. 正确格式: <|mt_start|><|mt_0044|><|mt_0442|><|mt_end|> -> 不变
    """
    # 规则 1: 处理标记太多的情况 (3个或以上)
    # 捕获前两个，匹配掉多余的，然后用前两个重构
    pattern_too_many = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_\d+\|>)(?:<\|mt_\d+\|>)+<\|mt_end\|>'
    replacement_too_many = r'\1\2\3<|mt_end|>'
    text = re.sub(pattern_too_many, replacement_too_many, text)
    # 规则 2: 处理标记太少的情况 (只有1个，且有<|mt_end|>)
    pattern_too_few_with_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)'
    replacement_too_few = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_with_end, replacement_too_few, text)
    # 规则 3: 处理标记太少的情况 (只有1个，且没有<|mt_end|>)
    # 使用负向前瞻确保后面不是另一个mt_token
    pattern_too_few_no_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)'
    replacement_too_few_no_end = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_no_end, replacement_too_few_no_end, text)
    return text

def no_target_check(response: str, ground_truth: str) -> float:
    """
    检查 response 是否不包含任何 mask token 格式。
    如果 response 不包含 <|mt_start|>, <|mt_xxxx|>, <|mt_end|> 任何一个，
    且 ground_truth 在 response 中，则返回 1.0，否则返回 0.0。
    """
    # 检查是否包含 <|mt_start|>, <|mt_\d{4}|>, <|mt_end|>
    has_mt_start = re.search(r'<\|mt_start\|>', response) is not None
    has_mt_token = re.search(r'<\|mt_\d{4}\|>', response) is not None
    has_mt_end = re.search(r'<\|mt_end\|>', response) is not None

    # 如果 response 不包含任何 mask token 相关格式
    no_mask_token = not (has_mt_start or has_mt_token or has_mt_end)

    if no_mask_token and ground_truth in response:
        return 1.0
    elif no_mask_token and ground_truth not in response:
        return 0.2
    return 0.0

def no_target_check_bbox(response: str, ground_truth: str) -> float:
    """
    检查 response 是否不包含任何 bbox 格式。
    如果 response 不包含 [x1, y1, x2, y2] 格式的 bbox，
    且 ground_truth 在 response 中，则返回 1.0，否则返回 0.0。

    Args:
        response: 模型生成的响应字符串
        ground_truth: ground truth 字符串（通常是 "null" 或其他无目标标记）

    Returns:
        float: 1.0 表示正确识别无目标，0.2 表示无 bbox 但 gt 不匹配，0.0 表示包含 bbox
    """
    # 检查是否包含 [x1, y1, x2, y2] 格式的 bbox
    bbox_pattern = r'\[\d+,\s*\d+,\s*\d+,\s*\d+\]'
    has_bbox = re.search(bbox_pattern, response) is not None

    # 如果 response 不包含任何 bbox 格式
    no_bbox = not has_bbox

    if no_bbox and ground_truth in response:
        return 1.0
    elif no_bbox and ground_truth not in response:
        return 0.2
    return 0.0

def cap_no_mask_token_check(response: str) -> float:
    # 首先检查是否包含 <|mt_start|>, <|mt_\d{4}|>, <|mt_end|>
    has_mt_start = re.search(r'<\|mt_start\|>', response) is not None
    has_mt_token = re.search(r'<\|mt_\d{4}\|>', response) is not None
    has_mt_end = re.search(r'<\|mt_end\|>', response) is not None

    # 如果 response 不包含任何 mask token 相关格式
    no_mask_token = not (has_mt_start or has_mt_token or has_mt_end)

    if no_mask_token:
        return 1.0
    return 0.0

def cap_no_bbox_check(response: str) -> float:
    """
    检查 caption response 是否不包含任何 bbox 格式。
    如果 response 不包含 [x1, y1, x2, y2] 格式的 bbox，则返回 1.0，否则返回 0.0。

    这是 cap_no_mask_token_check 的 bbox 版本，用于确保 caption 任务的输出
    不会错误地包含 bbox 坐标。

    Args:
        response: 模型生成的响应字符串

    Returns:
        float: 1.0 表示不包含 bbox（正确），0.0 表示包含 bbox（错误）
    """
    # 检查是否包含 [x1, y1, x2, y2] 格式的 bbox
    bbox_pattern = r'\[\d+,\s*\d+,\s*\d+,\s*\d+\]'
    has_bbox = re.search(bbox_pattern, response) is not None

    if not has_bbox:
        return 1.0
    return 0.0


def cap_no_bbox_no_chinese_check(response: str) -> float:
    """
    检查 caption response 是否同时满足：
    1) 不包含 [x1, y1, x2, y2] 格式的 bbox
    2) 不包含任何中文字符 (CJK Unified Ideographs)

    Returns:
        1.0  -- 两条都满足（合规）
        0.0  -- 至少违反其中一条
    """
    bbox_pattern = r'\[\d+,\s*\d+,\s*\d+,\s*\d+\]'
    has_bbox = re.search(bbox_pattern, response) is not None

    chinese_pattern = r'[一-鿿]'
    has_chinese = re.search(chinese_pattern, response) is not None

    if not has_bbox and not has_chinese:
        return 1.0
    return 0.0


def temporal_list_format_reward(response: str) -> float:
    """Check whether response contains an 'x.x - x.x seconds' interval after </think>/<answer>."""
    if response is None:
        return 0.0

    answer_content = response
    think_end_match = re.search(r'</think>', response, re.I)
    if think_end_match:
        answer_content = response[think_end_match.end():]

    answer_match = re.search(r'<answer>(.*?)</answer>', answer_content, re.I | re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1)

    interval_pattern = r"\d+(?:\.\d+)?\s*[-–—~]\s*\d+(?:\.\d+)?\s*seconds?"
    return 1.0 if re.search(interval_pattern, answer_content, flags=re.I) else 0.0


def extract_temporal_intervals(text: str, only_result: bool = False) -> list[tuple[float, float]]:
    """Extract temporal intervals from GT <time> tags or text outputs like 'x.x - x.x seconds'."""
    if text is None:
        return []

    content = str(text).strip()
    if only_result:
        _, answer_content = extract_think_and_answer_robust(content)
        if answer_content is not None:
            content = answer_content.strip()

    intervals = []

    time_blocks = re.findall(r'<time>(.*?)</time>', content, flags=re.I | re.DOTALL)
    if time_blocks:
        for blk in time_blocks:
            blk = blk.strip()
            match = re.fullmatch(
                r'\s*(\d+(?:\.\d+)?)\s*[-–—~]\s*(\d+(?:\.\d+)?)\s*(?:seconds?|s)?\s*',
                blk,
                flags=re.I,
            )
            if not match:
                continue
            start, end = float(match.group(1)), float(match.group(2))
            if start < end:
                intervals.append((start, end))
        if intervals:
            return intervals

    # Primary format for model output: "x.x - x.x seconds" (comma separated for multi segments).
    text_matches = re.findall(
        r'(\d+(?:\.\d+)?)\s*[-–—~]\s*(\d+(?:\.\d+)?)\s*(?:seconds?|s)',
        content,
        flags=re.I,
    )
    for start, end in text_matches:
        start_val, end_val = float(start), float(end)
        if start_val < end_val:
            intervals.append((start_val, end_val))
    if intervals:
        return intervals

    # Backward-compatible fallback for legacy outputs: [[start, end], ...] or [start, end].
    bracket_matches = re.findall(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', content)
    for start, end in bracket_matches:
        start_val, end_val = float(start), float(end)
        if start_val < end_val:
            intervals.append((start_val, end_val))

    return intervals


def compute_temporal_count_reward(solution_str: str, ground_truth: str) -> tuple[float, int, int]:
    """Reward 1.0 when predicted temporal interval count matches GT count, else 0.0."""
    pred_segments = extract_temporal_intervals(solution_str, only_result=True)
    gt_segments = extract_temporal_intervals(ground_truth, only_result=True)

    pred_count = len(pred_segments)
    gt_count = len(gt_segments)
    count_score = 1.0 if pred_count == gt_count else 0.0
    return count_score, pred_count, gt_count


def temporal_iou_from_text(solution_str: str, ground_truth: str) -> float:
    """tIoU between predicted and GT temporal intervals (multi-segment aware).

    Note: distinct from fsdp_workers.py:compute_temporal_iou_reward, which takes
    already-parsed window lists. This one parses raw text on both sides.
    """
    def merge(intervals: list[tuple[float, float]]) -> list[list[float]]:
        ivs = sorted([[s, e] for s, e in intervals if s < e], key=lambda x: x[0])
        if not ivs:
            return []
        merged = [ivs[0][:]]
        for s, e in ivs[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged

    pred = merge(extract_temporal_intervals(solution_str, only_result=True))
    gt = merge(extract_temporal_intervals(ground_truth, only_result=True))
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0

    intersection = 0.0
    for ps, pe in pred:
        for gs, ge in gt:
            intersection += max(0.0, min(pe, ge) - max(ps, gs))
    union = sum(e - s for s, e in pred) + sum(e - s for s, e in gt) - intersection
    return intersection / union if union > 0 else 0.0


TIMESTAMP_PATTERNS = [
    r'\b\d+(\.\d+)?\s*[-–to]+\s*\d+(\.\d+)?\s*(second|sec|minute|min|s)\b',
    r'\bat\s+\d+(\.\d+)?\s*(second|sec|s)\b',
    r'\bfrom\s+\d+(\.\d+)?\s*(second|sec|minute|to)\b',
    r'\b\d{1,2}:\d{2}\b',  # MM:SS format
    r'\b(at|around|by|in|from|between)\s+\d+\s*(second|sec|minute|min)\b',
]

def timestamp_penalty(caption: str) -> float:
    """Returns 0.0 if no timestamps found, -1.0 if found."""
    text = caption.lower()
    for pattern in TIMESTAMP_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return -1.0
    return 0.0

def extract_think_and_answer_robust(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts the content between <think> and <answer> tags from a string,
    regardless of their order or position, as long as the tags exist.
    Args:
        response (str): The input string, potentially containing <think> and <answer> tags.
    Returns:
        Tuple[Optional[str], Optional[str]]:
            A tuple containing (think_content, answer_content).
            Each element will be a string if found, or None if the corresponding tag is not found.
    """
    think_content = None
    answer_content = None
    # Pattern for <think> tag content
    # re.DOTALL allows '.' to match any character, including newlines.
    # Non-greedy match (.*?) ensures it stops at the first </think>.
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    # Pattern for <answer> tag content
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    # Search for <think> content
    think_match = think_pattern.search(response)
    if think_match:
        think_content = think_match.group(1) # group(1) gets the content of the first capture group
    # Search for <answer> content
    answer_match = answer_pattern.search(response)
    if answer_match:
        answer_content = answer_match.group(1) # group(1) gets the content of the first capture group

    if answer_content is None or think_content is None:
        if '<answer>' in response:
            head, tail = response.split('<answer>', 1)
            if think_content is None:
                think_content = head
            if answer_content is None:
                answer_content = tail
        elif '</think>' in response:
            head, tail = response.split('</think>', 1)
            if think_content is None:
                think_content = head
            if answer_content is None:
                answer_content = tail

    return think_content, answer_content

def downsample_masks_half(masks: torch.Tensor, mode: str = "nearest") -> torch.Tensor:
    """
    Downsample masks to half of original spatial size (H/2, W/2).
    masks: (N, H, W)
    Returns: (N, H//2, W//2)
    """
    if masks.dtype == torch.bool:
        masks = masks.float()
    N, H, W = masks.shape
    # Ensure even division; interpolate handles odd sizes by floor with target
    target_h = max(1, H // 2)
    target_w = max(1, W // 2)
    masks = masks.unsqueeze(1)  # (N,1,H,W)
    masks_ds = F.interpolate(masks, size=(target_h, target_w), mode=mode)
    masks_ds = masks_ds.squeeze(1)
    return masks_ds

def compute_pairwise_iou(gt_masks: torch.Tensor, pred_masks: torch.Tensor, threshold: float = None) -> torch.Tensor:
    """
    Compute pairwise mask IoU between gt_masks and pred_masks.
    gt_masks: (N, H, W), pred_masks: (M, H, W)
    If threshold is not None, binarize masks by threshold.
    Returns IoU matrix of shape (N, M).
    """
    if threshold is not None:
        gt = (gt_masks >= threshold).float()
        pred = (pred_masks >= threshold).float()
    else:
        gt = gt_masks.float()
        pred = pred_masks.float()
    N, H, W = gt.shape
    M = pred.shape[0]
    gt_flat = gt.reshape(N, -1)      # (N, HW)
    pred_flat = pred.reshape(M, -1)  # (M, HW)
    inter = torch.matmul(gt_flat, pred_flat.t())  # (N, M)
    gt_area = gt_flat.sum(dim=1, keepdim=True)    # (N, 1)
    pred_area = pred_flat.sum(dim=1, keepdim=True) # (M, 1)
    union = gt_area + pred_area.t() - inter
    eps = 1e-7
    iou = inter / (union + eps)
    return iou.clamp(0.0, 1.0)

def hungarian_maximize(iou_matrix: torch.Tensor):
    """
    Maximum bipartite matching maximizing IoU via Hungarian algorithm.
    Returns:
      - matches: list of (gt_idx, pred_idx)
      - ious: list of IoU values
    """
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        cost = (1.0 - iou_matrix).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]
        ious = [float(iou_matrix[r, c].item()) for r, c in zip(row_ind, col_ind)]
        return matches, ious
    except Exception:
        # Greedy fallback
        iou = iou_matrix.clone()
        N, M = iou.shape
        vals, idxs = torch.topk(iou.view(-1), k=iou.numel())
        used_rows, used_cols = set(), set()
        matches, ious = [], []
        for val, flat_idx in zip(vals.tolist(), idxs.tolist()):
            if val <= 0:
                break
            r = flat_idx // M
            c = flat_idx % M
            if r in used_rows or c in used_cols:
                continue
            used_rows.add(r)
            used_cols.add(c)
            matches.append((int(r), int(c)))
            ious.append(float(val))
            if len(used_rows) == N or len(used_cols) == M:
                break
        return matches, ious

def bipartite_match_by_mask_iou_half(
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    binarize_threshold: float = 0.5,
    interp_mode: str = "nearest"
):
    """
    Downsample to half size, compute IoU, and perform maximum-weight matching.
    Returns:
      - matches: list[(gt_idx, pred_idx)]
      - ious: list[float]
    """
    assert gt_masks.ndim == 3 and pred_masks.ndim == 3, "Masks must be (N,H,W) and (M,H,W)"
    # Downsample to half of original size
    # gt_ds = downsample_masks_half(gt_masks, mode=interp_mode)
    # pred_ds = downsample_masks_half(pred_masks, mode=interp_mode)
    gt_ds = gt_masks
    pred_ds = pred_masks
    # Compute IoU with optional binarization
    iou_mat = compute_pairwise_iou(gt_ds, pred_ds, threshold=binarize_threshold)
    # Hungarian max matching
    matches, ious = hungarian_maximize(iou_mat)
    return matches, ious

def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0

def mask_token_format_reward(answer_content: str) -> float:
    """Check if the answer contains valid mask token format.

    Args:
        answer_content: The extracted answer content string.

    Returns:
        1.0 if at least one valid mask token pattern is found, 0.0 otherwise.
    """
    if answer_content is None:
        return 0.0
    regex_pattern = r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>"
    pred_mask_tokens = re.findall(regex_pattern, answer_content)
    return 1.0 if len(pred_mask_tokens) > 0 else 0.0

def bbox_format_reward(answer_content: str) -> float:
    """Check if the answer contains valid bbox format.

    检查答案中是否包含有效的 bbox 格式 [x1, y1, x2, y2]。
    这是 mask_token_format_reward 的 bbox 版本。

    Args:
        answer_content: The extracted answer content string.

    Returns:
        1.0 if at least one valid bbox pattern is found, 0.0 otherwise.
    """
    if answer_content is None:
        return 0.0
    # 匹配 [x1, y1, x2, y2] 格式的 bbox
    bbox_pattern = r'\[\d+,\s*\d+,\s*\d+,\s*\d+\]'
    pred_bboxes = re.findall(bbox_pattern, answer_content)
    return 1.0 if len(pred_bboxes) > 0 else 0.0


def extract_bbox_xyxy(text: str) -> Optional[list[int]]:
    if text is None:
        return None
    match = re.search(r'\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?', text)
    if match is None:
        return None
    return [int(v) for v in match.groups()]


def is_valid_bbox_xyxy(box: Optional[list[int]]) -> bool:
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and box[2] > box[0]
        and box[3] > box[1]
    )


def bbox_iou_xyxy(box1: Optional[list[int]], box2: Optional[list[int]], eps: float = 1e-6) -> float:
    if not is_valid_bbox_xyxy(box1) or not is_valid_bbox_xyxy(box2):
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / (union + eps)) if union > 0 else 0.0


def bbox_center_closeness(box1: Optional[list[int]], box2: Optional[list[int]], scale: float = 1000.0) -> float:
    if not is_valid_bbox_xyxy(box1) or not is_valid_bbox_xyxy(box2):
        return 0.0
    center1 = ((box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0)
    center2 = ((box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0)
    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    max_distance = (2.0 ** 0.5) * scale
    return float(max(0.0, 1.0 - distance / max_distance))


def refcocog_branch_caption_reward(reward_input: dict[str, Any]) -> dict[str, float]:
    response = reward_input.get("response") or ""
    _, answer_content = extract_think_and_answer_robust(response)
    response_content = answer_content if answer_content is not None else response
    pred_bbox = extract_bbox_xyxy(response_content)
    target_bbox = extract_bbox_xyxy(reward_input.get("ground_truth"))

    format_score = bbox_format_reward(response_content)
    no_repeat_score = gm_non_repeat_reward(response_content) if response_content else 0.0
    target_iou = bbox_iou_xyxy(pred_bbox, target_bbox)
    center_score = bbox_center_closeness(pred_bbox, target_bbox)

    extra_info = reward_input.get("extra_info", {}) or {}
    pseudo_label = extra_info.get("pseudo_label", {}) or {}
    baseline_branch = extra_info.get("baseline_branch", {}) or {}
    candidate_branches = extra_info.get("candidate_branches", []) or []

    selected_branch_bbox = pseudo_label.get("bbox_norm") or target_bbox
    selected_branch_iou = bbox_iou_xyxy(pred_bbox, selected_branch_bbox)
    baseline_iou = bbox_iou_xyxy(pred_bbox, baseline_branch.get("bbox_norm"))
    best_candidate_iou = 0.0
    for candidate in candidate_branches:
        best_candidate_iou = max(best_candidate_iou, bbox_iou_xyxy(pred_bbox, candidate.get("bbox_norm")))

    collapse_penalty = 0.0
    if baseline_iou >= 0.98 and selected_branch_iou < 0.90:
        collapse_penalty = 1.0

    overall = (
        8.0 * target_iou
        + 1.5 * selected_branch_iou
        + 0.75 * best_candidate_iou
        + 0.5 * center_score
        + 0.5 * format_score
        + 0.25 * no_repeat_score
        - 1.0 * collapse_penalty
    )
    return {
        "cap_overall": overall,
        "refcocog_branch_target_iou": target_iou,
        "refcocog_branch_selected_iou": selected_branch_iou,
        "refcocog_branch_best_candidate_iou": best_candidate_iou,
        "refcocog_branch_center_score": center_score,
        "refcocog_branch_format_score": format_score,
        "refcocog_branch_no_repeat_score": no_repeat_score,
        "refcocog_branch_collapse_penalty": collapse_penalty,
    }

def mask_accuracy_reward(gt_masks: torch.Tensor, pred_masks: torch.Tensor):
    if gt_masks is None and pred_masks is None:
        return 1.0
    if gt_masks is None or pred_masks is None:
        return 0.0
    if gt_masks.ndim == 2:
        gt_masks = gt_masks.unsqueeze(0)
    if pred_masks.ndim == 2:
        pred_masks = pred_masks.unsqueeze(0)

    matches, ious = bipartite_match_by_mask_iou_half(gt_masks, pred_masks)
    denominator = max(len(gt_masks), len(pred_masks))
    return sum(ious) / denominator

def mask_token_accuracy_reward(answer_content: list[str], ground_truth: list[str]):
    if answer_content is None:
        return 0.0
    regex_pattern = r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>"
    target_mask_tokens = re.findall(regex_pattern, ground_truth)
    pred_mask_tokens = re.findall(regex_pattern, answer_content)
    if len(target_mask_tokens) == 0 and len(pred_mask_tokens) == 0:
        return 1.0
    unique_target_mask_tokens = list(set(target_mask_tokens))
    unique_pred_mask_tokens = list(set(pred_mask_tokens))
    max_N = max(len(pred_mask_tokens), len(target_mask_tokens))
    recall_N = 0
    for mask_token in unique_pred_mask_tokens:
        max_match_score = 0.0
        for gt_mask_token in unique_target_mask_tokens:
            match_score = compute_mask_token_accuracy_by_grading(mask_token, gt_mask_token)
            if max_match_score < match_score:
                max_match_score = match_score
        recall_N += max_match_score
    return recall_N / max_N

def compute_mask_token_accuracy_by_grading(one_answer: str, one_ground_truth: str):
    answer_tokens = re.findall(r"<\|[^|]*\|>", one_answer)
    gt_tokens = re.findall(r"<\|[^|]*\|>", one_ground_truth)
    if one_answer == one_ground_truth:
        return 1.0
    elif ''.join(answer_tokens[:-1]) == ''.join(gt_tokens[:-1]):  # both of the two token are matched up
        return 0.8
    elif ''.join(answer_tokens[:-2]) == ''.join(gt_tokens[:-2]): # only the first token are matched up
        return 0.4
    else:
        return 0.0

def extract_gcg_patterns(text):
    """
    Extracts specific pattern substrings and their second parts from a GCG task string.
    Args:
        text (str): The input string containing GCG task information.
    Returns:
        list: A list of tuples, where each tuple is (full_match, second_part).
              full_match is the complete matched string,
              second_part is the substring starting from <|object_ref_end|> to <|mt_end|>.
    """
    # Define the regular expression pattern to match the desired structure.
    # The pattern looks for:
    # 1. <|object_ref_start|>
    # 2. (.*?) - Non-greedy capture of the phrase between start and end object refs.
    # 3. \s*<|object_ref_end|> - The end object ref, preceded by zero or more whitespace characters.
    # 4. \s* - Zero or more whitespace characters between <|object_ref_end|> and <|mt_start|>.
    # 5. (<|mt_start|>(<|mt_\d+|>){2}<|mt_end|>) - This entire part is captured as 'second_part'.
    #    - <|mt_start|> - Literal match for the mt start tag.
    #    - (<|mt_\d+|>){2} - Matches exactly two mt tags, e.g., <|mt_0198|><|mt_0373|>.
    #      - \d+ matches one or more digits.
    #    - <|mt_end|> - Literal match for the mt end tag.
    pattern = r"<\|object_ref_start\|>(.*?)\s*<\|object_ref_end\|>\s*(<\|mt_start\|>(<\|mt_\d+\|>){2}<\|mt_end\|>)"
    # pattern = r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>"
    # Find all non-overlapping matches of the pattern in the text.
    # re.findall returns a list of tuples, where each tuple contains the captured groups.
    # For our pattern, it will return (phrase, second_part_full, mt_code_group_1, mt_code_group_2).
    matches = re.findall(pattern, text)
    results = []
    phrases = []
    # Iterate through each match found.
    # 'phrase' is the content between <|object_ref_start|> and <|object_ref_end|>.
    # 'second_part_full' is the entire <|mt_start|><|mt_...|><|mt_...|><|mt_end|> block.
    # The last two '_' are placeholders for the inner capture groups of the mt codes, which we don't need directly here.
    for phrase, second_part_full, _ in matches:
        # Reconstruct the full matched string by combining the object_ref part and the mt part.
        # .strip() is used on 'phrase' to remove any leading/trailing whitespace that might have been captured.
        # full_match = f"<|object_ref_start|>{phrase.strip()}<|object_ref_end|>{second_part_full}"
        results.append(f"<|object_ref_end|>{second_part_full}")
        words = [word.strip().lower() for word in phrase.split(' ') if word.strip().lower() != '']
        if len(words) > 0:
            phrases.append(' '.join(words))

    return results, phrases

def glamm_gcg_mask_token_accuracy_reward(answer_content: str, ground_truth: str):
    if answer_content is None:
        return 0.0
    target_mask_tokens, _ = extract_gcg_patterns(ground_truth)
    pred_mask_tokens, _ = extract_gcg_patterns(answer_content)

    # print("=========>target_mask_tokens: ", target_mask_tokens)
    # print("=========>pred_mask_tokens: ", pred_mask_tokens)
    # print("=========>answer_content: ", answer_content)

    unique_pred_mask_tokens = list(set(pred_mask_tokens))
    unique_target_mask_tokens = list(set(target_mask_tokens))
    max_N = max(len(pred_mask_tokens), len(target_mask_tokens))
    recall_N = 0
    for mask_token in unique_pred_mask_tokens:
        max_match_score = 0.0
        for gt_mask_token in unique_target_mask_tokens:
            match_score = compute_mask_token_accuracy_by_grading(mask_token, gt_mask_token)
            if max_match_score < match_score:
                max_match_score = match_score
        recall_N += max_match_score
    return recall_N / max_N


def glamm_gcg_mask_token_accuracy_reward_v2(answer_content: str, ground_truth: str):
    if answer_content is None:
        return 0.0
    pattern = r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>"
    target_mask_tokens = re.findall(pattern, ground_truth)
    pred_mask_tokens = re.findall(pattern, answer_content)

    unique_pred_mask_tokens = list(set(pred_mask_tokens))
    unique_target_mask_tokens = list(set(target_mask_tokens))
    max_N = max(len(pred_mask_tokens), len(target_mask_tokens))
    recall_N = 0
    for mask_token in unique_pred_mask_tokens:
        max_match_score = 0.0
        for gt_mask_token in unique_target_mask_tokens:
            match_score = compute_mask_token_accuracy_by_grading(mask_token, gt_mask_token)
            if max_match_score < match_score:
                max_match_score = match_score
        recall_N += max_match_score
    return recall_N / max_N

def glamm_gcg_phrases_accuracy_reward(answer_content: str, ground_truth: str):
    if answer_content is None:
        return 0.0

    _, target_phrases = extract_gcg_patterns(ground_truth)
    _, pred_phrases = extract_gcg_patterns(answer_content)

    if len(target_phrases) == 0:
        return 1.0
    unique_pred_phrases = list(set(pred_phrases))
    max_N = max(len(pred_phrases), len(target_phrases))
    recall_N = 0
    for pred_phrase in unique_pred_phrases:
        if pred_phrase in target_phrases:
            recall_N += 1
    return recall_N / max_N

def psg_accuracy_reward(answer_content: str, ground_truth: str):
    pred_triplets, pred_text_triplets = extract_scene_graph_dicts(answer_content)
    gt_triplets, gt_text_triplets = extract_scene_graph_dicts(ground_truth)

    non_repeat_score = psg_non_repeat_reward(pred_text_triplets)

    max_N = max(len(pred_text_triplets), len(gt_text_triplets))
    recall_N = 0

    scanned_text_triplets = []
    for pred_triplet, pred_text_triplet in zip(pred_triplets, pred_text_triplets):
        if pred_text_triplet in scanned_text_triplets:
            continue
        scanned_text_triplets.append(pred_text_triplet)
        if pred_text_triplet in gt_text_triplets:
            recall_N += 1.0
            continue
        try:
            pred_sub_label = pred_triplet['subject']['label']
            pred_predicate = pred_triplet['predicate']
            pred_obj_label = pred_triplet['object']['label']
            pred_sub_mask_2d = pred_triplet['subject']['mask_2d']
            pred_obj_mask_2d = pred_triplet['object']['mask_2d']
        except:
            continue
        max_score = 0.0
        for gt_triplet in gt_triplets:
            try:
                gt_sub_label = gt_triplet['subject']['label']
                gt_predicate = gt_triplet['predicate']
                gt_obj_label = gt_triplet['object']['label']
                gt_sub_mask_2d = gt_triplet['subject']['mask_2d']
                gt_obj_mask_2d = gt_triplet['object']['mask_2d']
            except:
                continue

            match_score = 0.0
            if f"{pred_sub_label}#{pred_predicate}#{pred_obj_label}" == f"{gt_sub_label}#{gt_predicate}#{gt_obj_label}":
                match_score += 0.5
                sub_mask_token_match_score = compute_mask_token_accuracy_by_grading(pred_sub_mask_2d, gt_sub_mask_2d)
                match_score += sub_mask_token_match_score * 0.25
                obj_mask_token_match_score = compute_mask_token_accuracy_by_grading(pred_obj_mask_2d, gt_obj_mask_2d)
                match_score += obj_mask_token_match_score * 0.25
            if max_score < match_score:
                max_score = match_score
        recall_N += max_score
    return recall_N / max_N, non_repeat_score

def non_repeat_reward(predict_str: str) -> float:
    if predict_str is None:
        return 0.0
    regex_pattern = r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>"
    mask_tokens = re.findall(regex_pattern, predict_str)
    non_repeat_reward = 1.0
    for mask_token in mask_tokens:
        mask_token_count = predict_str.count(mask_token)
        if mask_token_count > 3:
            non_repeat_reward = 0.0
            break
    return non_repeat_reward

def psg_non_repeat_reward(text_triplets: list[str]):
    if len(text_triplets) == 0:
        return 1.0
    text_triplet_dict = {}
    for text_triplet in text_triplets:
        if text_triplet not in text_triplet_dict:
            text_triplet_dict[text_triplet] = 0
        text_triplet_dict[text_triplet] += 1
    if max(list(text_triplet_dict.values())) > 2:
        return 0.0
    else:
        return 1.0

def extract_scene_graph_dicts(text):
    pattern = r'''
    \{
    \s*"subject"\s*:\s*\{
        \s*"mask_2d"\s*:\s*"[^"]*"\s*,\s*
        \s*"label"\s*:\s*"[^"]*"\s*
    \}\s*,\s*
    \s*"predicate"\s*:\s*"[^"]*"\s*,\s*
    \s*"object"\s*:\s*\{
        \s*"mask_2d"\s*:\s*"[^"]*"\s*,\s*
        \s*"label"\s*:\s*"[^"]*"\s*
    \}
    \s*\}
    '''
    matches = re.findall(pattern, text, flags=re.DOTALL | re.VERBOSE)

    dicts, text_triplets = [], []
    for m in matches:
        try:
            d = json.loads(m)
            dicts.append(d)
            text_triplets.append(m)
        except json.JSONDecodeError:
            pass
    return dicts, text_triplets

SCORER_CIDER = Cider()

def compute_semantic_alignment_reward(answer_content: str, gt_cap: str, weight: float = 1.0) -> torch.Tensor:
    """
    针对已提取答案的单条字符串进行语义奖励计算

    Args:
        answer_content: 已解析出的 <answer> 文本内容
        gt_cap: Ground Truth 描述字符串
        weight: 权重系数
    """
    # 鲁棒性处理：确保输入是字符串且非空
    ans_str = str(answer_content).strip() if answer_content else ""
    gt_str = str(gt_cap).strip() if gt_cap else ""

    # 即使是单条，也必须满足 pycocoevalcap 的 dict(list) 格式要求
    res_dict = {0: [ans_str]}
    gts_dict = {0: [gt_str]}

    try:
        # compute_score 返回：(平均分, 各样本分列表)
        _, scores = SCORER_CIDER.compute_score(gts_dict, res_dict)
        # 取出第 0 个位置的分数
        score = scores[0]
    except Exception:
        # 预防计算异常（如词典库未对齐等极端情况）
        score = 0.0

    # 返回 Scalar Tensor
    return (torch.tensor(score, dtype=torch.float32) * weight).item()

def gm_non_repeat_reward(text: str) -> float:
    """
    针对单条文本输入的 3-gram 重复检查奖励

    Args:
        text: 模型生成的描述字符串
    Returns:
        float: 1.0 (通过) 或 0.0 (触发重复)
    """
    # 1. 基础清理：转小写，移除标点符号
    text = str(text).lower()
    clean_text = re.sub(r'[^\w\s]', ' ', text) # 将标点替换为空格，避免单词粘连
    words = clean_text.split()

    # 2. 长度校验：不足 3 个词无法构成 triplet
    if len(words) < 3:
        # 在 Region Captioning 中，太短通常意味着信息量不足
        # 但从“非重复”角度看，它是合格的。这里我们遵循你原有的逻辑或给 1.0
        return 1.0

    # 3. 生成 3-gram (Triplets) 列表
    # 使用滑动窗口：[w1, w2, w3], [w2, w3, w4] ...
    text_triplets = [" ".join(words[i:i+3]) for i in range(len(words)-2)]

    # 4. 统计频率
    triplet_dict = {}
    for triplet in text_triplets:
        triplet_dict[triplet] = triplet_dict.get(triplet, 0) + 1

    # 5. 判定标准：任何三元组出现超过 2 次则判定为重复
    # 例如："on the left ... on the left ... on the left" 触发 0 分
    if len(triplet_dict) > 0 and max(triplet_dict.values()) > 2:
        return 0.0
    else:
        return 1.0


_OPEN_O3_PORTER_STEMMER = PorterStemmer()


def _extract_open_o3_answer(text: str) -> str:
    """Accept both Open-o3's tagged answer and our answer-only interface."""

    value = str(text or "").strip()
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", value, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else value


def _open_o3_rouge_tokens(text: str) -> list[str]:
    """Match ``rouge_score.RougeScorer(..., use_stemmer=True)`` tokenization."""

    value = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
    tokens = []
    for token in value.split():
        if len(token) > 3:
            token = _OPEN_O3_PORTER_STEMMER.stem(token)
        if re.fullmatch(r"[a-z0-9]+", token):
            tokens.append(token)
    return tokens


def _open_o3_fmeasure(overlap: int, pred_count: int, ref_count: int) -> float:
    if overlap <= 0 or pred_count <= 0 or ref_count <= 0:
        return 0.0
    precision = overlap / pred_count
    recall = overlap / ref_count
    return 2.0 * precision * recall / (precision + recall)


def _open_o3_ngram_fmeasure(reference: list[str], prediction: list[str], n: int) -> float:
    ref_ngrams = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
    pred_ngrams = Counter(tuple(prediction[i : i + n]) for i in range(len(prediction) - n + 1))
    overlap = sum((ref_ngrams & pred_ngrams).values())
    return _open_o3_fmeasure(overlap, sum(pred_ngrams.values()), sum(ref_ngrams.values()))


def _open_o3_lcs_length(reference: list[str], prediction: list[str]) -> int:
    # One-row dynamic program keeps long free-form answers inexpensive.
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = [0] * (len(prediction) + 1)
    for ref_token in reference:
        current = [0]
        for j, pred_token in enumerate(prediction, start=1):
            if ref_token == pred_token:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def open_o3_freeform_rouge_reward(reference: str, prediction: str) -> float:
    """Official Open-o3 free-form reward: mean ROUGE-1/2/L F1."""

    ref_tokens = _open_o3_rouge_tokens(reference)
    pred_tokens = _open_o3_rouge_tokens(_extract_open_o3_answer(prediction))
    rouge1 = _open_o3_ngram_fmeasure(ref_tokens, pred_tokens, 1)
    rouge2 = _open_o3_ngram_fmeasure(ref_tokens, pred_tokens, 2)
    rouge_l = _open_o3_fmeasure(
        _open_o3_lcs_length(ref_tokens, pred_tokens), len(pred_tokens), len(ref_tokens)
    )
    return (rouge1 + rouge2 + rouge_l) / 3.0

def mask_iou_reward(pred_masks: np.ndarray, gt_masks: list[str], eps=1e-6):
    if pred_masks is None:
        return 0.0

    gt_mask_ori = mask_utils.decode(gt_masks)
    gt_mask_ori = gt_mask_ori > 0.5

    pred = pred_masks.astype(bool)
    gt   = gt_mask_ori.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()

    return intersection / (union + eps)


def _temporal_dense_reward_weight() -> float:
    """Experiment-gated weight; zero preserves every historical run exactly."""

    value = float(os.environ.get("CYCLEGRPO_TEMPORAL_DENSE_REWARD_WEIGHT", "0"))
    if value < 0:
        raise ValueError("CYCLEGRPO_TEMPORAL_DENSE_REWARD_WEIGHT must be non-negative")
    return value


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1, task: str = 'caption') -> list[dict[str, float]]:

    scores = []
    if task == 'caption':
        # reward_inputs: dict_keys(['response', 'response_length', 'ground_truth', 'pred_masks', 'gt_masks', 'source', 'task', 'image'])
        # reward_inputs[0]['ground_truth']  reward_inputs[0]['response']

        # LLM-as-judge captioning sources (image: dam_captioning, video: tg_captioning).
        # 同一 judge endpoint / 同一 prompt，一次 async batch 算完两边。
        judge_caption_indices = [
            i for i, ri in enumerate(reward_inputs)
            if ri["source"] in ("dam_captioning", "tg_captioning")
        ]
        if judge_caption_indices:
            judge_rewards = compute_caption_judge_rewards_batch(
                responses=[reward_inputs[i]["response"]         for i in judge_caption_indices],
                gts      =[reward_inputs[i]["cap_ground_truth"] for i in judge_caption_indices],
                max_concurrency=64,
                max_votes=1,            # baseline: 单次 deterministic
                temperature=0.0,
            )
            judge_reward_map = dict(zip(judge_caption_indices, judge_rewards))
        else:
            judge_reward_map = {}

        understanding_indices = [
            i for i, ri in enumerate(reward_inputs)
            if ri["source"] == TRAJECTORY_UNDERSTANDING_SOURCE
        ]
        understanding_reward_map = {}
        if understanding_indices:
            understanding_rewards = compute_graded_caption_judge_rewards_batch(
                [reward_inputs[i]["response"] for i in understanding_indices],
                [reward_inputs[i]["cap_ground_truth"] for i in understanding_indices],
                max_concurrency=64,
                temperature=0.0,
                strict=True,
            )
            understanding_reward_map = dict(zip(understanding_indices, understanding_rewards))

        for i, reward_input in enumerate(reward_inputs):
            source = reward_input["source"]

            if source in ['thinking_res', 'thinking_gres', 'thinking_ver', 'reasonseg']:
                format_score = format_reward(reward_input["response"])
                # print("===============>response: ", reward_input["response"])
                think_content, answer_content = extract_think_and_answer_robust(reward_input["response"])
                accuracy_score = mask_token_accuracy_reward(answer_content, reward_input["ground_truth"])
                think_content_no_repeat_score = non_repeat_reward(think_content)
                answer_content_no_repeat_score = non_repeat_reward(answer_content)
                scores.append(
                    {
                        "cap_overall": accuracy_score + format_score + think_content_no_repeat_score + answer_content_no_repeat_score,
                        "format": format_score,
                        "accuracy": accuracy_score,
                        "think_content_no_repeat_score": think_content_no_repeat_score,
                        "answer_content_no_repeat_score": answer_content_no_repeat_score,
                    }
                )
            elif source in ['gcg']:
                accuracy_score = glamm_gcg_mask_token_accuracy_reward(reward_input["response"], reward_input["ground_truth"])
                phrase_score = glamm_gcg_phrases_accuracy_reward(reward_input["response"], reward_input["ground_truth"])
                no_repeat_score = non_repeat_reward(reward_input["response"])
                scores.append(
                    {
                        "cap_overall": accuracy_score + phrase_score + no_repeat_score,
                        "gcg_accuracy": accuracy_score,
                        "gcg_phrase": phrase_score,
                        "no_repeat_score": no_repeat_score,
                    }
                )
            elif source in ['gres_no_target']:
                reward_input["ground_truth"] = 'No target.'
                _, response_answer_content = extract_think_and_answer_robust(reward_input["response"])
                _, gt_answer_content = extract_think_and_answer_robust(reward_input["ground_truth"])
                # accuracy_score = no_target_check(response_answer_content, gt_answer_content)
                accuracy_score = no_target_check_bbox(reward_input["response"], reward_input["ground_truth"])
                no_repeat_score = non_repeat_reward(reward_input["response"])
                scores.append(
                    {
                        "cap_overall": accuracy_score + no_repeat_score,
                        "no_target_accuracy": accuracy_score,
                        "no_repeat_score": no_repeat_score,
                    }
                )
            # elif source in ['psg']:
            #     accuracy_score, no_repeat_score = psg_accuracy_reward(reward_input['response'], reward_input['ground_truth'])
            #     phrase_score = 1.0
            #     scores.append(
            #         {
            #             "cap_overall": accuracy_score + no_repeat_score + phrase_score,
            #             "accuracy": accuracy_score,
            #             "phrase_accuracy": phrase_score,
            #             "no_repeat_score": no_repeat_score,
            #         }
            #     )
            elif source == TRAJECTORY_UNDERSTANDING_SOURCE:
                semantic_score = float(understanding_reward_map[i])
                no_leak_score = spatiotemporal_caption_no_leak_reward(
                    reward_input.get("response")
                )
                no_repeat_score = gm_non_repeat_reward(reward_input.get("response"))
                format_score = float(bool(str(reward_input.get("response") or "").strip()))
                cap_overall = no_leak_score * (
                    10.0 * semantic_score / 4.0 + format_score + no_repeat_score
                )
                scores.append({
                    "cap_overall": cap_overall,
                    "understanding_semantic_0_4": semantic_score,
                    "understanding_no_leak": no_leak_score,
                    "understanding_format": format_score,
                    "understanding_no_repeat": no_repeat_score,
                })
            elif source == OPEN_O3_UNDERSTANDING_SOURCE:
                task_name = (reward_input.get("extra_info") or {}).get(
                    "open_o3_task", ""
                )
                response = str(reward_input.get("response") or "")
                answer_text = _extract_open_o3_answer(response)
                reference = str(reward_input.get("cap_ground_truth") or "").strip()
                if task_name == "General video QA MCQ":
                    # Prefer an explicit final option, then fall back to the
                    # last standalone A-D token. This tolerates short CoT while
                    # keeping the released answer fully verifiable.
                    candidates = re.findall(
                        r"(?:answer\s*(?:is|:)?\s*)?\b([A-D])\b",
                        answer_text.upper(),
                    )
                    accuracy = float(bool(candidates) and candidates[-1] == reference.upper())
                    semantic = accuracy
                else:
                    semantic = open_o3_freeform_rouge_reward(reference, answer_text)
                    accuracy = semantic
                format_score = float(bool(response.strip()))
                no_repeat_score = gm_non_repeat_reward(response)
                scores.append({
                    "cap_overall": 10.0 * semantic + format_score + no_repeat_score,
                    "open_o3_answer_accuracy": accuracy,
                    "open_o3_answer_format": format_score,
                    "open_o3_answer_no_repeat": no_repeat_score,
                })
            elif source == TRAJECTORY_LOCALIZATION_SOURCE:
                joint_score = trajectory_spatiotemporal_reward(
                    reward_input.get("response"),
                    reward_input.get("cap_ground_truth"),
                )
                format_score = trajectory_spatiotemporal_format_reward(
                    reward_input.get("response")
                )
                scores.append({
                    "cap_overall": 10.0 * joint_score + format_score,
                    "localization_trajectory_st_joint": joint_score,
                    "localization_trajectory_st_format": format_score,
                })
            elif source in {
                SPATIOTEMPORAL_SOURCE,
                ATOMIC_SPATIOTEMPORAL_SOURCE,
                TRAJECTORY_SPATIOTEMPORAL_SOURCE,
            }:
                cycle_iou = float(reward_input.get("iou_scores") or 0.0)
                temporal_dense = float(
                    reward_input.get("temporal_dense_scores") or 0.0
                )
                temporal_dense_bonus = (
                    _temporal_dense_reward_weight() * temporal_dense
                )
                no_leak_score = spatiotemporal_caption_no_leak_reward(
                    reward_input.get("response")
                )
                no_repeat_score = gm_non_repeat_reward(reward_input.get("response"))
                # Strong gate: copied timestamps/boxes erase the cycle benefit.
                # The inner prompt is independently sanitized in ray_trainer.py,
                # so coordinate copying cannot close the cycle either.
                cap_overall = (
                    (10.0 * cycle_iou + no_repeat_score) * no_leak_score
                    + no_leak_score
                    + temporal_dense_bonus * no_leak_score
                )
                scores.append(
                    {
                        "cap_overall": cap_overall,
                        "cap_st_iou": cycle_iou,
                        "cap_temporal_dense": temporal_dense,
                        "cap_temporal_dense_bonus": temporal_dense_bonus,
                        "cap_st_no_leak": no_leak_score,
                        "cap_st_no_repeat": no_repeat_score,
                    }
                )
            elif source in ['tg_multi_merged']:
                cap_no_time_score = timestamp_penalty(reward_input['response'])
                cap_no_repeat_score = gm_non_repeat_reward(reward_input['response'])
                iou_scores = reward_input['iou_scores']
                temporal_dense = float(
                    reward_input.get("temporal_dense_scores") or 0.0
                )
                temporal_dense_bonus = (
                    _temporal_dense_reward_weight() * temporal_dense
                )
                # 长度正则：纯 GRPO 组内相对，同 prompt 下偏好更短的 caption。
                # 不引入任何 caption GT —— 仅作格式正则，长度由「组内更短且 IoU 不掉」自行决定。
                # λ=0.002 定标自 warmup_0 log：10*iou 项单样本跨 0~11，精简/臃肿 caption 差约 0.44
                # (~0.044 IoU 等价)，盖不过真实 grounding 收益但能在 IoU 打平时果断选短。
                # 压不住再上调，0.003 为上限。
                # lambda_len = 0.002
                # cap_length_penalty = -lambda_len * reward_input['response_length']
                scores.append(
                    {
                        "cap_overall":  10*iou_scores + temporal_dense_bonus + cap_no_time_score + cap_no_repeat_score,
                        "cap_iou_scores": iou_scores,
                        "cap_temporal_dense": temporal_dense,
                        "cap_temporal_dense_bonus": temporal_dense_bonus,
                        "cap_no_time_score": cap_no_time_score,
                        "cap_no_repeat_score": cap_no_repeat_score,
                        # "cap_length_penalty": cap_length_penalty,
                    }
                )
            elif source in ['dam_captioning']:
                caption_reward = judge_reward_map[i]
                scores.append(
                    {
                        "cap_overall":    caption_reward,
                        "caption_reward": caption_reward,
                    }
                )

            elif source in ['tg_captioning']:
                # Video temporal captioning, LLM-as-judge baseline. Reward 完全由
                # judge (true/false -> 1.0/0.0) 给出，对照 cap_answer (Gemini GT)。
                # 复用 dam_captioning 的 judge prompt —— mask-token 子句对 video
                # 输出是 soft constraint，不会触发。
                caption_reward = judge_reward_map[i]
                print(caption_reward)
                scores.append(
                    {
                        "cap_overall":    caption_reward,
                        "caption_reward": caption_reward,
                    }
                )

            elif source in ['dam_grounding']:
                # Check format correctness (should have the correct mask token format)
                regex_pattern = r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>"
                pred_mask_tokens = re.findall(regex_pattern, reward_input["response"])
                format_score = int(len(pred_mask_tokens) > 0)
                iou_scores = mask_token_accuracy_reward(reward_input["response"], reward_input['seg_ground_truth'])
                scores.append(
                    {
                        "cap_overall":  format_score + iou_scores,
                        "cap_format": format_score,
                        "cap_iou_scores": iou_scores,
                    }
                )

            elif source in ['tg_grounding']:
                # tg_grounding 是独立的 temporal grounding 数据集（cap_problem 即 grounding
                # 提问，cap_answer 为 GT 时间段），非 cycle source，只走 caption 任务、无内层
                # rollout，故 tIoU 直接由 response 与 GT 现算。
                # ⚠️ GT 取自 cap_ground_truth（<- parquet 的 cap_answer）：caption 任务下
                # reward manager 会把 seg_ground_truth 置 None（function.py 只在
                # task=='segmentation' 时透传它），cap_ground_truth 才是唯一可用通道。
                # 奖励结构参考 segmentation 分支的 tg_multi_merged：count 作为 IoU 的乘性门控
                # （段数对→IoU 全奖励，段数错→打 3 折），format 独立加分。
                response = reward_input["response"]
                gt = reward_input.get("cap_ground_truth")
                iou_score = temporal_iou_from_text(response, gt)
                format_score = temporal_list_format_reward(response)
                count_score, pred_count, gt_count = compute_temporal_count_reward(
                    response, gt
                )
                count_gate = 0.3 + 0.7 * count_score
                scores.append(
                    {
                        "cap_overall": 10 * iou_score * count_gate + format_score,
                        "cap_iou_scores": iou_score,
                        "cap_temporal_format_score": format_score,
                        "cap_temporal_count_score": count_score,
                        "cap_temporal_pred_count": pred_count,
                        "cap_temporal_gt_count": gt_count,
                    }
                )

            elif source in ['groundingme', 'denseworld_single', 'denseworld_multiple', 'dam_cyclegrpo'] or source is None:
                # format_score = format_reward(reward_input["response"])
                # response_think_content, response_answer_content = extract_think_and_answer_robust(reward_input["response"])
                # gt_think_content, gt_answer_content = extract_think_and_answer_robust(reward_input["ground_truth"])
                # semantic_alignment_score = compute_semantic_alignment_reward(response_answer_content, gt_answer_content)
                # think_content_no_repeat_score = gm_non_repeat_reward(response_think_content)

                answer_content_no_repeat_score = gm_non_repeat_reward(reward_input["response"])
                # cap_no_mask_token_check_score = cap_no_mask_token_check(reward_input["response"])
                cap_no_bbox_no_chinese_score = cap_no_bbox_no_chinese_check(reward_input["response"])
                iou_scores = reward_input['iou_scores']
                # 方案 A: 乘性 gating —— 输出含 bbox 或中文时，no_repeat 与 iou 奖励一并清零，
                # 让违规直接抹掉所有正向收益，模型无法用 bbox 蹭 iou 分。
                cap_overall = (answer_content_no_repeat_score + 10*iou_scores) * cap_no_bbox_no_chinese_score + cap_no_bbox_no_chinese_score
                scores.append(
                    {
                        "cap_overall": cap_overall,
                        "cap_iou_scores": iou_scores,
                        "cap_answer_content_no_repeat_score": answer_content_no_repeat_score,
                        "cap_no_mask_token_check_score": cap_no_bbox_no_chinese_score,
                    }
                )

            else:
                raise NotImplementedError
    elif task == 'segmentation':
        # ['response', 'response_length', 'ground_truth', 'pred_masks', 'gt_masks', 'source', 'task']
        # reward_inputs[0]['response']: '<think>\n\n</think>\n\n<|mt_start|><|mt_0017|><|mt_0408|><|mt_end|><|im_end|>'
        # reward_inputs[0]['response_length']: 9
        # reward_inputs[0]['ground_truth']: '<think>\n\n</think><answer><|mt_start|><|mt_0214|><|mt_0418|><|mt_end|></answer>'
        # reward_inputs[0]['pred_masks']: None
        # reward_inputs[0]['gt_masks']: None
        # reward_inputs[0]['source']: 'groundingme'
        # reward_inputs[0]['task']: 'segmentation'
        # 'response', 'response_length', 'source', 'task', 'iou_scores', 'mask_token_accuracy']
        from projects.rl.reward_function.tg_reward import tg_reward
        for reward_input in reward_inputs:
            source = reward_input["source"]
            if source == TRAJECTORY_SPATIOTEMPORAL_SOURCE:
                joint_score = float(
                    reward_input.get("mask_token_accuracy") or 0.0
                )
                format_score = trajectory_spatiotemporal_format_reward(
                    reward_input.get("response")
                )
                scores.append(
                    {
                        "seg_overall": 10.0 * joint_score + format_score,
                        "seg_trajectory_st_joint": joint_score,
                        "seg_trajectory_st_format": format_score,
                    }
                )
            elif source == ATOMIC_SPATIOTEMPORAL_SOURCE:
                joint_score = float(
                    reward_input.get("mask_token_accuracy") or 0.0
                )
                format_score = atomic_spatiotemporal_format_reward(
                    reward_input.get("response")
                )
                scores.append(
                    {
                        "seg_overall": 10.0 * joint_score + format_score,
                        "seg_atomic_st_joint": joint_score,
                        "seg_atomic_st_format": format_score,
                    }
                )
            elif source == SPATIOTEMPORAL_SOURCE:
                local_iou = float(reward_input.get("mask_token_accuracy") or 0.0)
                parent_iou = float(reward_input.get("iou_scores") or 0.0)
                coupled_iou = local_iou * parent_iou
                local_temporal_dense = float(
                    reward_input.get("temporal_dense_score") or 0.0
                )
                parent_temporal_dense = float(
                    reward_input.get("temporal_dense_scores") or 0.0
                )
                temporal_dense = 0.5 * (
                    local_temporal_dense + parent_temporal_dense
                )
                temporal_dense_bonus = (
                    _temporal_dense_reward_weight() * temporal_dense
                )
                format_score = spatiotemporal_format_reward(
                    reward_input.get("response")
                )
                scores.append(
                    {
                        "seg_overall": 10.0 * coupled_iou + temporal_dense_bonus + format_score,
                        "seg_st_iou": coupled_iou,
                        "seg_st_local_iou": local_iou,
                        "seg_st_parent_iou": parent_iou,
                        "seg_temporal_dense": temporal_dense,
                        "seg_temporal_dense_bonus": temporal_dense_bonus,
                        "seg_st_format": format_score,
                    }
                )
            elif source in ['tg_multi_merged']:
                iou_score = reward_input["mask_token_accuracy"] * reward_input["iou_scores"]
                local_temporal_dense = float(
                    reward_input.get("temporal_dense_score") or 0.0
                )
                parent_temporal_dense = float(
                    reward_input.get("temporal_dense_scores") or 0.0
                )
                temporal_dense = 0.5 * (
                    local_temporal_dense + parent_temporal_dense
                )
                temporal_dense_bonus = (
                    _temporal_dense_reward_weight() * temporal_dense
                )
                format_score = temporal_list_format_reward(reward_input.get("response"))
                count_score, pred_count, gt_count = compute_temporal_count_reward(
                    reward_input.get("response"), reward_input.get("seg_ground_truth")
                )
                # count 作为 IoU 的乘性门控：段数对 → IoU 全奖励；段数错 → IoU 奖励
                # 打 3 折，保留少量梯度但显著抑制"输出多段/不输出"。format 独立加分。
                # 单独刷 count 拿不到收益（IoU 项被乘 0.3，且 IoU 本身在多段/0 段下会被压低）。
                count_gate = 0.3 + 0.7 * count_score
                scores.append(
                    {
                        "seg_overall": 10 * iou_score * count_gate + temporal_dense_bonus + format_score,
                        "seg_iou_scores": iou_score,
                        "seg_temporal_format_score": format_score,
                        "seg_temporal_count_score": count_score,
                        "seg_temporal_dense": temporal_dense,
                        "seg_temporal_dense_bonus": temporal_dense_bonus,
                        "seg_temporal_pred_count": pred_count,
                        "seg_temporal_gt_count": gt_count,
                    }
                )
            elif source in ['groundingme', 'denseworld_single', 'denseworld_multiple', 'dam_cyclegrpo', None] or source is None:
                # format_score = format_reward(reward_input["response"])
                # _, answer_content = extract_think_and_answer_robust(reward_input["response"])
                iou_score = reward_input["mask_token_accuracy"] * reward_input["iou_scores"]
                # think_content_no_repeat_score = non_repeat_reward(think_content)

                answer_content_no_repeat_score = non_repeat_reward(reward_input["response"])
                is_bbox_target = bool(
                    extract_bbox_xyxy(str(reward_input.get("seg_ground_truth") or ""))
                )
                mask_token_format_correct = (
                    bbox_format_reward(reward_input["response"])
                    if is_bbox_target
                    else mask_token_format_reward(reward_input["response"])
                )

                # answer_content_no_repeat_score = non_repeat_reward(reward_input["response"])
                # mask_token_format_correct = bbox_format_reward(reward_input["response"])
                scores.append(
                    {
                        "seg_overall": 10*iou_score + answer_content_no_repeat_score + mask_token_format_correct,
                        # "seg_format": format_score,
                        "seg_iou_scores": iou_score,
                        "seg_answer_content_no_repeat_score": answer_content_no_repeat_score,
                        "seg_mask_token_format_correct": mask_token_format_correct,
                    }
                )
            else:
                raise NotImplementedError
    return scores


if __name__ == "__main__":
    # text = "```json\n[{\"subject\": {\"mask_2d\": \"<|mt_start|><|mt_0017|><|mt_0322|><|mt_end|>\", \"label\": \"person\"}, \"predicate\": \"beside\", \"object\": {\"mask_2d\": \"<|mt_start|><|mt_0044|><|mt_0438|><|mt_end|>\", \"label\": \"person\"}},\n{\"subject\": {\"mask_2d\": \"<|mt_start|><|mt_0017|><|mt_0322|><|mt_end|>\", \"label\": \"person\"}, \"predicate\": \"standing on\", \"object\": {\"mask_2d\": \"<|mt_start|><|mt_0186|><|mt_0268|><|mt_end|>\", \"label\": \"grass-merged\"}},\n{\"subject\": {\"mask_2d\": \"<|mt_start|><|mt_0044|><|mt_0438|><|mt_end|>\", \"label\": \"person\"}, \"predicate\": \"holding\", \"object\": {\"mask_2d\": \"<|mt_start|><|mt_0116|><|mt_0448|><|mt_end|>\", \"label\": \"umbrella\"}},\n{\"subject\": {\"mask_2d\": \"<|mt_start|><|mt_0044|><|mt_0438|><|mt_end|>\", \"label\": \"person\"}, \"predicate\": \"standing on\", \"object\": {\"mask_2d\": \"<|mt_start|><|mt_0186|><|mt_0268|><|mt_end|>\", \"label\": \"grass-merged\"}},\n{\"subject\": {\"mask_2d\": \"<|mt_start|><|mt_0040|><|mt_0497|><|mt_end|>\", \"label\": \"fence-merged\"}, \"predicate\": \"in front of\", \"object\": {\"mask_2d\": \"<|mt_start|><|mt_0135|><|mt_0306|><|mt_end|>\", \"label\": \"tree-merged\"}},\n{\"subject\": {\"mask_2d\": \"<|mt_start|><|mt_0168|><|mt_0428|><|mt_end|>\", \"label\": \"sky-other-merged\"}, \"predicate\": \"over\", \"object\": {\"mask_2d\": \"<|mt_start|><|mt_0135|><|mt_0306|><|mt_end|>\", \"label\": \"tree-merged\"}}]\n```"
    # triplets = extract_scene_graph_dicts(text)
    # for triplet in triplets:
    #     subject_label = triplet['subject']['label']
    #     predicate = triplet['predicate']
    #     object_label = triplet['object']['label']
    #     print(f"{subject_label}#{predicate}#{object_label}")
    # print(triplets)
    reward_inputs = [{
        "response": '<think>\n\n</think>\n\n<answer>The TV screen displays a user interface with a light blue background. On the left side, there is a vertical menu with several icons and text labels. In the center, there is a large search bar with a magnifying glass icon. Below the search bar, there is a yellow sticky note with text on it. On the right side, there is a circular icon with a blue background and a white symbol inside it.<|im_end|></answer>',
        "ground_truth": '<think>\n\n</think>\n\n<answer>This is a rectangular screen with a smooth, reflective surface, displaying an advertisement with a light, possibly white or light blue, background. The content includes a blurred image of a person with curly hair on the left, and dark text along with a yellow price box and numeric information on the right. Only the lower-right corner of the screen is obscured.</answer>',
        "source": "groundingme",
    }]
    rewards = compute_score(reward_inputs)
    print(rewards)
