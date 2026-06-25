"""
Temporal Grounding Reward Functions

This module provides reward functions for temporal grounding tasks,
including tIoU, format, recall, precision, f1-score, C-Acc, and multi-dimensional caption rewards.

Caption reward uses two-stage LLM-as-a-judge evaluation:
1. Stage 1 (with GT): Evaluate coverage, precision, discriminability
2. Stage 2 (without GT): Counterfactual prediction for consistency check

IMPROVED VERSION:
- Stage 1: Structured analysis with forced GT-Caption mapping and strict scoring
- Stage 2: Robust prediction format with no NONE fallback
- Better parsing and error handling
- More discriminative scoring

Author: Refactored with improved caption evaluation for One-to-Many TG
"""

import os
import re
import json
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment

# ==============================================================================
# Configuration from Environment Variables
# ==============================================================================

model_template = os.getenv("MODEL_TEMPLATE", "qwen").lower()
rm_name = os.getenv("RM_NAME", "Qwen3-30B-A3B")
debug_print = os.getenv("DEBUG_PRINT", "false").lower() == "true"

# Strategy configuration
reward_strategy = os.getenv("TG_REWARD_STRATEGY", "tiou_format").lower().strip()

# Weight configuration
tiou_weight = float(os.getenv("TG_TIOU_WEIGHT", "1.0"))
format_weight = float(os.getenv("TG_FORMAT_WEIGHT", "1.0"))
caption_weight = float(os.getenv("TG_CAPTION_WEIGHT", "1.0"))
recall_weight = float(os.getenv("TG_RECALL_WEIGHT", "1.0"))
precision_weight = float(os.getenv("TG_PRECISION_WEIGHT", "1.0"))
f1_weight = float(os.getenv("TG_F1_WEIGHT", "1.0"))
cacc_weight = float(os.getenv("TG_CACC_WEIGHT", "1.0"))
length_penalty_weight = float(os.getenv("TG_LENGTH_PENALTY_WEIGHT", "0.3"))

# Caption reward dimension weights (IMPROVED - more balanced)
CAPTION_COVERAGE_WEIGHT = float(os.getenv("CAPTION_COVERAGE_WEIGHT", "0.35"))
CAPTION_PRECISION_WEIGHT = float(os.getenv("CAPTION_PRECISION_WEIGHT", "0.20"))
CAPTION_DISCRIMINABILITY_WEIGHT = float(os.getenv("CAPTION_DISCRIMINABILITY_WEIGHT", "0.15"))
CAPTION_COUNTERFACTUAL_WEIGHT = float(os.getenv("CAPTION_COUNTERFACTUAL_WEIGHT", "0.30"))

# Length Penalty thresholds (DAPO-style)
THINK_LENGTH_SOFT_THRESHOLD = int(os.getenv("THINK_LENGTH_SOFT_THRESHOLD", "2000"))
THINK_LENGTH_HARD_THRESHOLD = int(os.getenv("THINK_LENGTH_HARD_THRESHOLD", "5000"))
CAPTION_LENGTH_SOFT_THRESHOLD = int(os.getenv("CAPTION_LENGTH_SOFT_THRESHOLD", "100"))
CAPTION_LENGTH_HARD_THRESHOLD = int(os.getenv("CAPTION_LENGTH_HARD_THRESHOLD", "200"))

# Penalty factors
THINK_OVERLONG_PENALTY_FACTOR = float(os.getenv("THINK_OVERLONG_PENALTY_FACTOR", "1.0"))
CAPTION_OVERLONG_PENALTY_FACTOR = float(os.getenv("CAPTION_OVERLONG_PENALTY_FACTOR", "0.5"))

# Recall threshold (for single-threshold legacy recall)
RECALL_TIOU_THRESHOLD = float(os.getenv("RECALL_TIOU_THRESHOLD", "0.5"))

# Multi-threshold settings for precision/recall/f1
PRF_THRESHOLDS = [0.3, 0.5, 0.7]

# Constants
MAX_RETRIES = 3


# ==============================================================================
# Strategy Classes
# ==============================================================================

@dataclass
class RewardWeights:
    """Weights for different reward components."""
    tiou: float = 1.0
    format: float = 1.0
    caption: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    cacc: float = 0.0
    length_penalty: float = 0.0


class TGRewardStrategy(ABC):
    """Abstract base class for reward strategies."""
    
    @abstractmethod
    def get_enabled_rewards(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_weights(self) -> RewardWeights:
        pass
    
    def get_name(self) -> str:
        return self.__class__.__name__


class TiouFormatCaptionStrategy(TGRewardStrategy):
    """Strategy using all three rewards: tiou + format + caption."""
    
    def __init__(self, tiou_w: float = 1.0, format_w: float = 1.0, caption_w: float = 1.0):
        self._tiou_w = tiou_w
        self._format_w = format_w
        self._caption_w = caption_w
    
    def get_enabled_rewards(self) -> List[str]:
        return ['tiou', 'format', 'caption']
    
    def get_weights(self) -> RewardWeights:
        total = self._tiou_w + self._format_w + self._caption_w
        if total > 0:
            return RewardWeights(
                tiou=self._tiou_w / total,
                format=self._format_w / total,
                caption=self._caption_w / total,
            )
        return RewardWeights(tiou=1/3, format=1/3, caption=1/3)


class TiouFormatRewardStrategy(TGRewardStrategy):
    """Strategy using only tiou + format rewards."""
    
    def __init__(self, tiou_w: float = 1.0, format_w: float = 1.0):
        self._tiou_w = tiou_w
        self._format_w = format_w
    
    def get_enabled_rewards(self) -> List[str]:
        return ['tiou', 'format']
    
    def get_weights(self) -> RewardWeights:
        total = self._tiou_w + self._format_w
        if total > 0:
            return RewardWeights(
                tiou=self._tiou_w / total,
                format=self._format_w / total,
            )
        return RewardWeights(tiou=0.5, format=0.5)


class TiouFormatCaptionLengthStrategy(TGRewardStrategy):
    """Strategy using tiou + format + caption + length_penalty."""
    
    def __init__(self, tiou_w: float = 1.0, format_w: float = 1.0, 
                 caption_w: float = 1.0, length_penalty_w: float = 0.3):
        self._tiou_w = tiou_w
        self._format_w = format_w
        self._caption_w = caption_w
        self._length_penalty_w = length_penalty_w
    
    def get_enabled_rewards(self) -> List[str]:
        return ['tiou', 'format', 'caption', 'length_penalty']
    
    def get_weights(self) -> RewardWeights:
        positive_total = self._tiou_w + self._format_w + self._caption_w
        if positive_total > 0:
            return RewardWeights(
                tiou=self._tiou_w / positive_total,
                format=self._format_w / positive_total,
                caption=self._caption_w / positive_total,
                length_penalty=self._length_penalty_w,
            )
        return RewardWeights(tiou=1/3, format=1/3, caption=1/3, length_penalty=self._length_penalty_w)


class TiouFormatRecallStrategy(TGRewardStrategy):
    """Strategy using tiou + format + recall (single threshold) rewards."""
    
    def __init__(self, tiou_w: float = 1.0, format_w: float = 1.0, recall_w: float = 1.0):
        self._tiou_w = tiou_w
        self._format_w = format_w
        self._recall_w = recall_w
    
    def get_enabled_rewards(self) -> List[str]:
        return ['tiou', 'format', 'recall']
    
    def get_weights(self) -> RewardWeights:
        total = self._tiou_w + self._format_w + self._recall_w
        if total > 0:
            return RewardWeights(
                tiou=self._tiou_w / total,
                format=self._format_w / total,
                recall=self._recall_w / total,
            )
        return RewardWeights(tiou=1/3, format=1/3, recall=1/3)


class TiouFormatRecallAvgStrategy(TGRewardStrategy):
    """Strategy using tiou + format + recall_avg (multi-threshold average) rewards."""
    
    def __init__(self, tiou_w: float = 1.0, format_w: float = 1.0, recall_w: float = 1.0):
        self._tiou_w = tiou_w
        self._format_w = format_w
        self._recall_w = recall_w
    
    def get_enabled_rewards(self) -> List[str]:
        return ['tiou', 'format', 'recall_avg']
    
    def get_weights(self) -> RewardWeights:
        total = self._tiou_w + self._format_w + self._recall_w
        if total > 0:
            return RewardWeights(
                tiou=self._tiou_w / total,
                format=self._format_w / total,
                recall=self._recall_w / total,
            )
        return RewardWeights(tiou=1/3, format=1/3, recall=1/3)


class TiouFormatPrecisionStrategy(TGRewardStrategy):
    """Strategy using tiou + format + precision (multi-threshold average) rewards."""
    
    def __init__(self, tiou_w: float = 1.0, format_w: float = 1.0, precision_w: float = 1.0):
        self._tiou_w = tiou_w
        self._format_w = format_w
        self._precision_w = precision_w
    
    def get_enabled_rewards(self) -> List[str]:
        return ['tiou', 'format', 'precision']
    
    def get_weights(self) -> RewardWeights:
        total = self._tiou_w + self._format_w + self._precision_w
        if total > 0:
            return RewardWeights(
                tiou=self._tiou_w / total,
                format=self._format_w / total,
                precision=self._precision_w / total,
            )
        return RewardWeights(tiou=1/3, format=1/3, precision=1/3)


class TiouFormatF1Strategy(TGRewardStrategy):
    """Strategy using tiou + format + f1 (multi-threshold average) rewards."""
    
    def __init__(self, tiou_w: float = 1.0, format_w: float = 1.0, f1_w: float = 1.0):
        self._tiou_w = tiou_w
        self._format_w = format_w
        self._f1_w = f1_w
    
    def get_enabled_rewards(self) -> List[str]:
        return ['tiou', 'format', 'f1']
    
    def get_weights(self) -> RewardWeights:
        total = self._tiou_w + self._format_w + self._f1_w
        if total > 0:
            return RewardWeights(
                tiou=self._tiou_w / total,
                format=self._format_w / total,
                f1=self._f1_w / total,
            )
        return RewardWeights(tiou=1/3, format=1/3, f1=1/3)


class TiouFormatPRFStrategy(TGRewardStrategy):
    """Strategy using tiou + format + precision + recall + f1 (all multi-threshold average) rewards."""
    
    def __init__(self, tiou_w: float = 1.0, format_w: float = 1.0, 
                 precision_w: float = 1.0, recall_w: float = 1.0, f1_w: float = 1.0):
        self._tiou_w = tiou_w
        self._format_w = format_w
        self._precision_w = precision_w
        self._recall_w = recall_w
        self._f1_w = f1_w
    
    def get_enabled_rewards(self) -> List[str]:
        return ['tiou', 'format', 'precision', 'recall_avg', 'f1']
    
    def get_weights(self) -> RewardWeights:
        total = self._tiou_w + self._format_w + self._precision_w + self._recall_w + self._f1_w
        if total > 0:
            return RewardWeights(
                tiou=self._tiou_w / total,
                format=self._format_w / total,
                precision=self._precision_w / total,
                recall=self._recall_w / total,
                f1=self._f1_w / total,
            )
        return RewardWeights(tiou=1/5, format=1/5, precision=1/5, recall=1/5, f1=1/5)


class TiouFormatCaccStrategy(TGRewardStrategy):
    """Strategy using tiou + format + cacc rewards."""
    
    def __init__(self, tiou_w: float = 1.0, format_w: float = 1.0, cacc_w: float = 1.0):
        self._tiou_w = tiou_w
        self._format_w = format_w
        self._cacc_w = cacc_w
    
    def get_enabled_rewards(self) -> List[str]:
        return ['tiou', 'format', 'cacc']
    
    def get_weights(self) -> RewardWeights:
        total = self._tiou_w + self._format_w + self._cacc_w
        if total > 0:
            return RewardWeights(
                tiou=self._tiou_w / total,
                format=self._format_w / total,
                cacc=self._cacc_w / total,
            )
        return RewardWeights(tiou=1/3, format=1/3, cacc=1/3)


class TiouFormatF1CaccStrategy(TGRewardStrategy):
    """Strategy using tiou + format + f1 + cacc rewards."""
    
    def __init__(self, tiou_w: float = 1.0, format_w: float = 1.0, 
                 f1_w: float = 1.0, cacc_w: float = 1.0):
        self._tiou_w = tiou_w
        self._format_w = format_w
        self._f1_w = f1_w
        self._cacc_w = cacc_w
    
    def get_enabled_rewards(self) -> List[str]:
        return ['tiou', 'format', 'f1', 'cacc']
    
    def get_weights(self) -> RewardWeights:
        total = self._tiou_w + self._format_w + self._f1_w + self._cacc_w
        if total > 0:
            return RewardWeights(
                tiou=self._tiou_w / total,
                format=self._format_w / total,
                f1=self._f1_w / total,
                cacc=self._cacc_w / total,
            )
        return RewardWeights(tiou=1/4, format=1/4, f1=1/4, cacc=1/4)


class TiouFormatF1CaccCaptionLengthStrategy(TGRewardStrategy):
    """
    Strategy using tiou + format + f1 + cacc + caption + length_penalty rewards.
    
    This is the most comprehensive strategy for One-to-Many Temporal Grounding.
    """
    
    def __init__(self, tiou_w: float = 1.0, format_w: float = 1.0, 
                 f1_w: float = 1.0, cacc_w: float = 1.0,
                 caption_w: float = 1.0, length_penalty_w: float = 0.3):
        self._tiou_w = tiou_w
        self._format_w = format_w
        self._f1_w = f1_w
        self._cacc_w = cacc_w
        self._caption_w = caption_w
        self._length_penalty_w = length_penalty_w
    
    def get_enabled_rewards(self) -> List[str]:
        return ['tiou', 'format', 'f1', 'cacc', 'caption', 'length_penalty']
    
    def get_weights(self) -> RewardWeights:
        positive_total = (self._tiou_w + self._format_w + self._f1_w + 
                         self._cacc_w + self._caption_w)
        if positive_total > 0:
            return RewardWeights(
                tiou=self._tiou_w / positive_total,
                format=self._format_w / positive_total,
                f1=self._f1_w / positive_total,
                cacc=self._cacc_w / positive_total,
                caption=self._caption_w / positive_total,
                length_penalty=self._length_penalty_w,
            )
        return RewardWeights(
            tiou=1/5, format=1/5, f1=1/5, cacc=1/5, caption=1/5,
            length_penalty=self._length_penalty_w
        )


def get_current_strategy() -> TGRewardStrategy:
    """Get the current strategy based on environment variable."""
    current_reward_strategy = os.getenv("TG_REWARD_STRATEGY", "tiou_format").lower().strip()
    current_tiou_weight = float(os.getenv("TG_TIOU_WEIGHT", "1.0"))
    current_format_weight = float(os.getenv("TG_FORMAT_WEIGHT", "1.0"))
    current_caption_weight = float(os.getenv("TG_CAPTION_WEIGHT", "1.0"))
    current_recall_weight = float(os.getenv("TG_RECALL_WEIGHT", "1.0"))
    current_precision_weight = float(os.getenv("TG_PRECISION_WEIGHT", "1.0"))
    current_f1_weight = float(os.getenv("TG_F1_WEIGHT", "1.0"))
    current_cacc_weight = float(os.getenv("TG_CACC_WEIGHT", "1.0"))
    current_length_penalty_weight = float(os.getenv("TG_LENGTH_PENALTY_WEIGHT", "0.3"))
    
    normalized_strategy = current_reward_strategy.replace("_", "").replace("-", "").lower()
    
    if normalized_strategy == "tiouformatcaption":
        return TiouFormatCaptionStrategy(current_tiou_weight, current_format_weight, current_caption_weight)
    elif normalized_strategy == "tiouformatcaptionlength":
        return TiouFormatCaptionLengthStrategy(current_tiou_weight, current_format_weight, current_caption_weight, current_length_penalty_weight)
    elif normalized_strategy == "tiouformat":
        return TiouFormatRewardStrategy(current_tiou_weight, current_format_weight)
    elif normalized_strategy == "tiouformatrecall":
        return TiouFormatRecallStrategy(current_tiou_weight, current_format_weight, current_recall_weight)
    elif normalized_strategy == "tiouformatrecallavg":
        return TiouFormatRecallAvgStrategy(current_tiou_weight, current_format_weight, current_recall_weight)
    elif normalized_strategy == "tiouformatprecision":
        return TiouFormatPrecisionStrategy(current_tiou_weight, current_format_weight, current_precision_weight)
    elif normalized_strategy == "tiouformatf1":
        return TiouFormatF1Strategy(current_tiou_weight, current_format_weight, current_f1_weight)
    elif normalized_strategy == "tiouformatprf":
        return TiouFormatPRFStrategy(current_tiou_weight, current_format_weight, current_precision_weight, current_recall_weight, current_f1_weight)
    elif normalized_strategy == "tiouformatcacc":
        return TiouFormatCaccStrategy(current_tiou_weight, current_format_weight, current_cacc_weight)
    elif normalized_strategy == "tiouformatf1cacc":
        return TiouFormatF1CaccStrategy(current_tiou_weight, current_format_weight, current_f1_weight, current_cacc_weight)
    elif normalized_strategy == "tiouformatf1cacccaptionlength":
        return TiouFormatF1CaccCaptionLengthStrategy(
            current_tiou_weight, current_format_weight, current_f1_weight, 
            current_cacc_weight, current_caption_weight, current_length_penalty_weight
        )
    else:
        raise ValueError(f"Unknown reward strategy: {current_reward_strategy}")


# ==============================================================================
# Time Interval Extraction Utilities
# ==============================================================================

def extract_time_intervals(sentence: str, only_result: bool = False) -> List[List[float]]:
    """Extract time intervals from text."""
    if only_result:
        think_end_match = re.search(r'</think>', sentence, re.I)
        if think_end_match:
            sentence = sentence[think_end_match.end():]
    
    intervals = []
    
    # Try <time>start - end</time> format
    time_blocks = re.findall(r'<time>(.*?)</time>', sentence, flags=re.I)
    if time_blocks:
        for blk in time_blocks:
            blk = blk.strip()
            if not blk:
                continue
            m = re.fullmatch(
                r'\s*(\d+(?:\.\d+)?)\s*[-–—~]\s*(\d+(?:\.\d+)?)\s*(?:seconds?|s)?\s*', 
                blk, flags=re.I
            )
            if m:
                intervals.append([float(m.group(1)), float(m.group(2))])
        if intervals:
            return intervals
    
    # Try "From X to Y" format
    from_to_matches = re.findall(r'[Ff]rom\s+(\d+(?:\.\d+)?)\s*s?\s+to\s+(\d+(?:\.\d+)?)\s*s?', sentence)
    if from_to_matches:
        for match in from_to_matches:
            intervals.append([float(match[0]), float(match[1])])
        return intervals
    
    # Try generic "number-number" format
    general_matches = re.findall(r'(\d+(?:\.\d+)?)\s*s?\s*[-–—~]\s*(\d+(?:\.\d+)?)\s*s?', sentence)
    if general_matches:
        for match in general_matches:
            intervals.append([float(match[0]), float(match[1])])
        return intervals
    
    return []


def extract_think_content(text: str) -> str:
    """Extract content within <think></think> tags."""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.I)
    return match.group(1).strip() if match else ""


def extract_captions_with_timestamps(think_content: str) -> List[Dict[str, Any]]:
    """Extract captions with timestamps from think content."""
    captions = []
    pattern = r'<time>\s*(\d+(?:\.\d+)?)\s*[-–—~]\s*(\d+(?:\.\d+)?)\s*(?:seconds?|s)?\s*</time>\s*[,:]?\s*([^\n<]+)'
    for m in re.finditer(pattern, think_content, re.I):
        start, end = float(m.group(1)), float(m.group(2))
        caption = m.group(3).strip().rstrip('.,;:')
        if caption and start < end:
            captions.append({"start": start, "end": end, "caption": caption})
    return captions


# ==============================================================================
# tIoU Calculation
# ==============================================================================

def calculate_iou(gt_windows: List[List[float]], pred_windows: List[List[float]]) -> float:
    """Calculate total IoU between multiple time windows."""
    def merge_intervals(intervals: List[List[float]]) -> List[List[float]]:
        if not intervals:
            return []
        valid_intervals = [[s, e] for s, e in intervals if s < e]
        if not valid_intervals:
            return []
        sorted_intervals = sorted(valid_intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0][:]]
        for current in sorted_intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                merged[-1] = [last[0], max(last[1], current[1])]
            else:
                merged.append(current[:])
        return merged
    
    all_gt = merge_intervals(gt_windows)
    all_pred = merge_intervals(pred_windows)
    
    if not all_gt and not all_pred:
        return 1.0
    if not all_gt or not all_pred:
        return 0.0
    
    total_gt = sum(e - s for s, e in all_gt)
    total_pred = sum(e - s for s, e in all_pred)
    
    intersection = 0.0
    i = j = 0
    while i < len(all_gt) and j < len(all_pred):
        gt_start, gt_end = all_gt[i]
        pred_start, pred_end = all_pred[j]
        intersect_start = max(gt_start, pred_start)
        intersect_end = min(gt_end, pred_end)
        intersection += max(0.0, intersect_end - intersect_start)
        if gt_end < pred_end:
            i += 1
        else:
            j += 1
    
    union = total_gt + total_pred - intersection
    return intersection / union if union > 0 else 0.0


def compute_tiou_reward(solution_str: str, ground_truth: str) -> Tuple[float, List, List]:
    """Compute tIoU reward."""
    pred_windows = extract_time_intervals(solution_str, only_result=True)
    gt_windows = extract_time_intervals(ground_truth, only_result=True)
    
    if not pred_windows and not gt_windows:
        return 1.0, pred_windows, gt_windows
    elif not pred_windows or not gt_windows:
        return 0.0, pred_windows, gt_windows
    else:
        return calculate_iou(gt_windows, pred_windows), pred_windows, gt_windows


# ==============================================================================
# Single-Pair IoU Calculation (for Recall/Precision/F1)
# ==============================================================================

def temporal_iou_single(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    """Calculate tIoU between a single prediction and GT interval."""
    pred_start, pred_end = pred
    gt_start, gt_end = gt
    
    if pred_start >= pred_end or gt_start >= gt_end:
        return 0.0
    
    intersect_start = max(pred_start, gt_start)
    intersect_end = min(pred_end, gt_end)
    intersection = max(0.0, intersect_end - intersect_start)
    
    pred_length = pred_end - pred_start
    gt_length = gt_end - gt_start
    union = pred_length + gt_length - intersection
    
    return intersection / union if union > 0 else 0.0


# ==============================================================================
# Legacy Single-Threshold Recall Calculation
# ==============================================================================

def recall_at_tiou_threshold(
    pred_intervals: List[Tuple[float, float]],
    gt_intervals: List[Tuple[float, float]],
    threshold: float = 0.5,
) -> float:
    """Calculate recall at a given tIoU threshold."""
    if not gt_intervals:
        return 0.0
    if not pred_intervals:
        return 0.0
    
    matched_gt = set()
    for i, gt in enumerate(gt_intervals):
        for pred in pred_intervals:
            tiou = temporal_iou_single(pred, gt)
            if tiou >= threshold:
                matched_gt.add(i)
                break
    
    return len(matched_gt) / len(gt_intervals)


def compute_recall_reward(
    solution_str: str, 
    ground_truth: str, 
    threshold: float = None
) -> Tuple[float, List, List]:
    """Compute recall reward (legacy single-threshold version)."""
    if threshold is None:
        threshold = RECALL_TIOU_THRESHOLD
    
    pred_windows = extract_time_intervals(solution_str, only_result=True)
    gt_windows = extract_time_intervals(ground_truth, only_result=True)
    
    pred_intervals = [(p[0], p[1]) for p in pred_windows]
    gt_intervals = [(g[0], g[1]) for g in gt_windows]
    
    if not gt_intervals or not pred_intervals:
        return 0.0, pred_intervals, gt_intervals
    
    recall_score = recall_at_tiou_threshold(pred_intervals, gt_intervals, threshold)
    return recall_score, pred_intervals, gt_intervals


# ==============================================================================
# Multi-Threshold Precision/Recall/F1 Calculation
# ==============================================================================

def compute_prf_metrics(
    pred_segments: List[Tuple[float, float]],
    gt_segments: List[Tuple[float, float]],
    iou_thresholds: List[float] = None,
) -> Dict[str, float]:
    """Compute P/R/F1 at multiple thresholds using Hungarian matching."""
    if iou_thresholds is None:
        iou_thresholds = PRF_THRESHOLDS
    
    results = {}
    
    if len(pred_segments) == 0 or len(gt_segments) == 0:
        value = 1.0 if len(pred_segments) == 0 and len(gt_segments) == 0 else 0.0
        for th in iou_thresholds:
            results[f'P@{th}'] = value
            results[f'R@{th}'] = value
            results[f'F1@{th}'] = value
        results['precision_avg'] = value
        results['recall_avg'] = value
        results['f1_avg'] = value
        return results
    
    num_preds, num_gts = len(pred_segments), len(gt_segments)
    iou_matrix = np.array([
        [temporal_iou_single(p, g) for g in gt_segments]
        for p in pred_segments
    ])
    
    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
    matched_ious = [iou_matrix[i, j] for i, j in zip(pred_indices, gt_indices)]
    
    precisions, recalls, f1s = [], [], []
    
    for th in iou_thresholds:
        tp = sum(1 for iou in matched_ious if iou >= th)
        precision = tp / num_preds if num_preds > 0 else 0.0
        recall = tp / num_gts if num_gts > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[f'P@{th}'] = precision
        results[f'R@{th}'] = recall
        results[f'F1@{th}'] = f1
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    results['precision_avg'] = sum(precisions) / len(precisions) if precisions else 0.0
    results['recall_avg'] = sum(recalls) / len(recalls) if recalls else 0.0
    results['f1_avg'] = sum(f1s) / len(f1s) if f1s else 0.0
    
    return results


def compute_precision_reward(solution_str: str, ground_truth: str) -> Tuple[float, Dict[str, float]]:
    """Compute precision reward (average of P@0.3, P@0.5, P@0.7)."""
    pred_windows = extract_time_intervals(solution_str, only_result=True)
    gt_windows = extract_time_intervals(ground_truth, only_result=True)
    
    pred_segments = [(p[0], p[1]) for p in pred_windows]
    gt_segments = [(g[0], g[1]) for g in gt_windows]
    
    metrics = compute_prf_metrics(pred_segments, gt_segments)
    return metrics['precision_avg'], metrics


def compute_recall_avg_reward(solution_str: str, ground_truth: str) -> Tuple[float, Dict[str, float]]:
    """Compute recall reward (average of R@0.3, R@0.5, R@0.7)."""
    pred_windows = extract_time_intervals(solution_str, only_result=True)
    gt_windows = extract_time_intervals(ground_truth, only_result=True)
    
    pred_segments = [(p[0], p[1]) for p in pred_windows]
    gt_segments = [(g[0], g[1]) for g in gt_windows]
    
    metrics = compute_prf_metrics(pred_segments, gt_segments)
    return metrics['recall_avg'], metrics


def compute_f1_reward(solution_str: str, ground_truth: str) -> Tuple[float, Dict[str, float]]:
    """Compute F1 reward (average of F1@0.3, F1@0.5, F1@0.7)."""
    pred_windows = extract_time_intervals(solution_str, only_result=True)
    gt_windows = extract_time_intervals(ground_truth, only_result=True)
    
    pred_segments = [(p[0], p[1]) for p in pred_windows]
    gt_segments = [(g[0], g[1]) for g in gt_windows]
    
    metrics = compute_prf_metrics(pred_segments, gt_segments)
    return metrics['f1_avg'], metrics


def compute_prf_all_rewards(solution_str: str, ground_truth: str) -> Tuple[float, float, float, Dict[str, float]]:
    """Compute all PRF rewards at once."""
    pred_windows = extract_time_intervals(solution_str, only_result=True)
    gt_windows = extract_time_intervals(ground_truth, only_result=True)
    
    pred_segments = [(p[0], p[1]) for p in pred_windows]
    gt_segments = [(g[0], g[1]) for g in gt_windows]
    
    metrics = compute_prf_metrics(pred_segments, gt_segments)
    return metrics['precision_avg'], metrics['recall_avg'], metrics['f1_avg'], metrics


# ==============================================================================
# C-Acc (Count Accuracy) Calculation
# ==============================================================================

def compute_cacc_reward(solution_str: str, ground_truth: str) -> Tuple[float, int, int]:
    """Compute C-Acc (Count Accuracy) reward."""
    pred_windows = extract_time_intervals(solution_str, only_result=True)
    gt_windows = extract_time_intervals(ground_truth, only_result=True)
    
    pred_count = len(pred_windows)
    gt_count = len(gt_windows)
    
    cacc_score = 1.0 if pred_count == gt_count else 0.0
    return cacc_score, pred_count, gt_count


# ==============================================================================
# Format Reward Calculation
# ==============================================================================

def _is_valid_time_tag(text: str) -> bool:
    """Check if a <time>...</time> tag has valid format."""
    pattern = r'^<time>\s*\d+(?:\.\d+)?\s*[-–—~]\s*\d+(?:\.\d+)?\s*(?:seconds?|s)?\s*</time>$'
    return bool(re.match(pattern, text.strip(), re.I))


def _extract_all_time_tags(text: str) -> List[str]:
    """Extract all <time>...</time> tags from text."""
    return re.findall(r'<time>.*?</time>', text, re.I | re.DOTALL)


def _validate_time_tags_format(time_tags: List[str]) -> bool:
    """Check if all time tags have valid format."""
    if not time_tags:
        return False
    for tag in time_tags:
        if not _is_valid_time_tag(tag):
            return False
    return True


def compute_format_reward(input_string: str) -> Tuple[float, Dict]:
    """Compute format reward by parsing model output."""
    pred_dict = {}
    input_string = input_string.strip()
    
    if not input_string:
        pred_dict['error'] = 'empty_response'
        return 0.0, pred_dict
    
    think_open_match = re.search(r'<think>', input_string, re.I)
    think_close_match = re.search(r'</think>', input_string, re.I)
    
    has_think_open = think_open_match is not None
    has_think_close = think_close_match is not None
    
    if has_think_open or has_think_close:
        if not (has_think_open and has_think_close):
            pred_dict['error'] = 'mismatched_think_tags'
            return 0.0, pred_dict
        
        if think_open_match.start() >= think_close_match.start():
            pred_dict['error'] = 'think_tags_wrong_order'
            return 0.0, pred_dict
        
        before_think = input_string[:think_open_match.start()]
        time_tags_before = _extract_all_time_tags(before_think)
        if time_tags_before:
            pred_dict['error'] = 'timestamps_before_think'
            return 0.0, pred_dict
        
        think_content = input_string[think_open_match.end():think_close_match.start()]
        pred_dict['think_content'] = think_content.strip()
        
        after_think = input_string[think_close_match.end():]
        pred_dict['after_think'] = after_think.strip()
        
        think_time_tags = _extract_all_time_tags(think_content)
        pred_dict['think_time_tags_count'] = len(think_time_tags)
        
        if think_time_tags:
            if not _validate_time_tags_format(think_time_tags):
                pred_dict['error'] = 'invalid_think_timestamp_format'
                return 0.0, pred_dict
        
        result_time_tags = _extract_all_time_tags(after_think)
        pred_dict['result_time_tags_count'] = len(result_time_tags)
        
        if not result_time_tags:
            pred_dict['error'] = 'no_result_timestamp'
            return 0.0, pred_dict
        
        if not _validate_time_tags_format(result_time_tags):
            pred_dict['error'] = 'invalid_result_timestamp_format'
            return 0.0, pred_dict
        
        result_intervals = extract_time_intervals(after_think, only_result=False)
        if result_intervals:
            pred_dict['result_timestamp'] = result_intervals[0]
        
        pred_dict['format_type'] = 'with_think'
        return 1.0, pred_dict
    
    else:
        pred_dict['format_type'] = 'no_think'
        
        all_time_tags = _extract_all_time_tags(input_string)
        pred_dict['time_tags_count'] = len(all_time_tags)
        
        if not all_time_tags:
            pred_dict['error'] = 'no_timestamp'
            return 0.0, pred_dict
        
        if not _validate_time_tags_format(all_time_tags):
            pred_dict['error'] = 'invalid_timestamp_format'
            return 0.0, pred_dict
        
        result_intervals = extract_time_intervals(input_string, only_result=False)
        if result_intervals:
            pred_dict['result_timestamp'] = result_intervals[0]
        
        return 1.0, pred_dict


# ==============================================================================
# Length Penalty Calculation
# ==============================================================================

def _soft_overlong_penalty(length: int, soft: int, hard: int, penalty_factor: float) -> float:
    """DAPO-style Soft Overlong Punishment."""
    if length <= soft:
        return 0.0
    elif length <= hard:
        progress = (length - soft) / (hard - soft)
        return penalty_factor * progress
    else:
        return penalty_factor


def compute_length_penalty(think_content: str, captions: List[Dict[str, Any]]) -> float:
    """Compute length penalty."""
    think_length = len(think_content) if think_content else 0
    
    think_penalty = _soft_overlong_penalty(
        think_length, 
        THINK_LENGTH_SOFT_THRESHOLD, 
        THINK_LENGTH_HARD_THRESHOLD,
        THINK_OVERLONG_PENALTY_FACTOR
    )
    
    caption_penalty = 0.0
    if captions:
        caption_penalties = [
            _soft_overlong_penalty(
                len(c.get("caption", "")),
                CAPTION_LENGTH_SOFT_THRESHOLD, 
                CAPTION_LENGTH_HARD_THRESHOLD,
                CAPTION_OVERLONG_PENALTY_FACTOR
            )
            for c in captions
        ]
        caption_penalty = sum(caption_penalties) / len(caption_penalties)
    
    return think_penalty + caption_penalty


# ==============================================================================
# IMPROVED Caption Reward Prompts
# ==============================================================================

# Stage 1 Prompt: WITH ground truth - IMPROVED with strict scoring
STAGE1_PROMPT_WITH_GT = """You are a STRICT evaluator for Video Temporal Grounding caption quality.

## Context
Query: "{query}"
Ground Truth: {num_gt_intervals} segment(s) at {gt_intervals_str}
Video duration: ~{video_duration:.0f}s

## Model's Captions:
{caption_list_str}

## EVALUATION TASK

### Step 1: Map each GT to captions
For each GT segment, find the BEST matching caption (if any).
A match requires: (1) temporal overlap, AND (2) caption describes "{query}"

### Step 2: Score STRICTLY using these rules

**COVERAGE (0-10)**: What fraction of GT segments are matched?
- 10 = ALL {num_gt_intervals} GT matched with clear "{query}" descriptions
- 8 = ALL matched, but 1 has weak description
- 6 = ~70% matched
- 4 = ~50% matched  
- 2 = Only 1 matched
- 0 = None matched
⚠️ If ANY GT is missing, score ≤ 8

**PRECISION (0-10)**: How close are boundaries?
- 10 = ALL within 1s of GT
- 8 = Most within 2s
- 6 = Within 3-5s
- 4 = Off by 5-10s
- 2 = Off by >10s
⚠️ Captions much WIDER than GT count as imprecise

**DISCRIMINABILITY (0-10)**: Can occurrences be distinguished?
- 10 = Each has unique context (who/what/when/where)
- 7 = Good context for most
- 4 = Generic (just "person claps")
- 0 = Impossible to distinguish

## Output Format
After analysis, output ONLY valid JSON:
{{"coverage": <int 0-10>, "precision": <int 0-10>, "discriminability": <int 0-10>}}

BE STRICT: Average captions score 4-6, not 8-10."""


# Stage 2 Prompt: WITHOUT ground truth - IMPROVED with robust prediction
STAGE2_PROMPT_NO_GT = """You are predicting video timestamps from text captions ONLY.

## Query: "{query}"

## Captions:
{caption_list_str}

## Task
Find ALL segments where "{query}" occurs based on the captions.

## Rules:
1. Look for captions that DESCRIBE or IMPLY "{query}"
2. Use the caption's timestamp as your prediction
3. If multiple captions match, list all of them
4. If caption text is vague but timestamp overlaps likely target, include it
5. Output format: one segment per line as "start - end"

## Example Output:
10.5 - 15.0
32.0 - 37.0

## Your predictions (list ALL matching segments):"""


# ==============================================================================
# IMPROVED Parsing Functions
# ==============================================================================

def _parse_stage1_response(response_text: str) -> Dict[str, int]:
    """Parse Stage 1 JSON response with multiple fallback strategies."""
    # Strategy 1: Find JSON in code block
    json_match = re.search(r'```(?:json)?\s*(\{[^`]+\})\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return {
                "coverage": min(10, max(0, int(data.get("coverage", 0)))),
                "precision": min(10, max(0, int(data.get("precision", 0)))),
                "discriminability": min(10, max(0, int(data.get("discriminability", 0)))),
            }
        except:
            pass
    
    # Strategy 2: Find standalone JSON object
    json_match = re.search(r'\{[^{}]*"coverage"[^{}]*\}', response_text)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return {
                "coverage": min(10, max(0, int(data.get("coverage", 0)))),
                "precision": min(10, max(0, int(data.get("precision", 0)))),
                "discriminability": min(10, max(0, int(data.get("discriminability", 0)))),
            }
        except:
            pass
    
    # Strategy 3: Extract individual values with regex
    result = {"coverage": 0, "precision": 0, "discriminability": 0}
    
    for key in result.keys():
        match = re.search(rf'{key}["\s:]+(\d+)', response_text, re.I)
        if match:
            result[key] = min(10, int(match.group(1)))
    
    return result


def _parse_stage2_response_multi(response_text: str) -> List[Tuple[float, float]]:
    """Parse Stage 2 time intervals with robust extraction."""
    # Skip obvious "no results" responses
    lower = response_text.lower()
    if len(response_text) < 30 and ('none' in lower or 'no ' in lower or 'cannot' in lower):
        return []
    
    intervals = []
    
    # Pattern: X.X - Y.Y or X - Y
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*[-–—~]\s*(\d+(?:\.\d+)?)', response_text):
        start, end = float(m.group(1)), float(m.group(2))
        if 0 <= start < end <= 10000:
            intervals.append((start, end))
    
    return intervals


# ==============================================================================
# IMPROVED Caption Reward Function
# ==============================================================================

def compute_caption_reward(
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any],
) -> Tuple[float, int, List[float]]:
    """
    Compute caption reward using improved two-stage LLM-as-Judge evaluation.
    
    IMPROVED:
    - Stage 1: Structured analysis with strict scoring guidelines
    - Stage 2: Robust prediction format, no NONE fallback
    - Better parsing with multiple fallback strategies
    
    Returns:
        (combined_score, caption_count, raw_scores)
        raw_scores = [coverage, precision, discriminability, counterfactual] (0-10 scale)
    """
    # Initialize scores
    coverage_score = 0.0
    precision_score = 0.0
    discriminability_score = 0.0
    counterfactual_score = 0.0
    
    # Extract think content and captions
    think_content = extract_think_content(solution_str)
    if not think_content:
        return 0.0, 0, [0.0, 0.0, 0.0, 0.0]
    
    captions = extract_captions_with_timestamps(think_content)
    if not captions:
        return 0.0, 0, [0.0, 0.0, 0.0, 0.0]
    
    # Get query and GT intervals
    query = extra_info.get("query", "")
    if not query:
        return 0.0, len(captions), [0.0, 0.0, 0.0, 0.0]
    
    gt_windows = extract_time_intervals(ground_truth, only_result=True)
    if not gt_windows:
        return 0.0, len(captions), [0.0, 0.0, 0.0, 0.0]
    
    num_gt_intervals = len(gt_windows)
    gt_intervals_str = ", ".join([f"[{g[0]:.1f}s - {g[1]:.1f}s]" for g in gt_windows])
    
    # Estimate video duration
    video_duration = max(c["end"] for c in captions) if captions else 60.0
    video_duration = max(video_duration, max(g[1] for g in gt_windows) + 5.0)
    
    # Format captions for prompts
    caption_list_str = "\n".join([
        f"[{c['start']:.1f}s - {c['end']:.1f}s]: {c['caption']}"
        for c in captions
    ])
    
    # Try to import LLM client
    try:
        from models.llm import AgentModel
        from schema import GenModel
        llm = AgentModel()
        reward_model = GenModel.QWEN3_30B
    except ImportError:
        # Fallback: return heuristic score
        count_ratio = min(len(captions), num_gt_intervals) / max(len(captions), num_gt_intervals, 1)
        return 0.4 * count_ratio, len(captions), [5.0, 5.0, 5.0, 0.0]
    
    # =========================================================================
    # Stage 1: Structured Evaluation with GT
    # =========================================================================
    stage1_prompt = STAGE1_PROMPT_WITH_GT.format(
        query=query,
        num_gt_intervals=num_gt_intervals,
        gt_intervals_str=gt_intervals_str,
        video_duration=video_duration,
        caption_list_str=caption_list_str,
    )
    
    for retry in range(MAX_RETRIES):
        try:
            response = llm.chat([{"role": "user", "content": stage1_prompt}], reward_model)
            parsed = _parse_stage1_response(response.content)
            
            if any(v > 0 for v in parsed.values()):
                coverage_score = parsed["coverage"] / 10.0
                precision_score = parsed["precision"] / 10.0
                discriminability_score = parsed["discriminability"] / 10.0
                
                if debug_print:
                    print(f"[S1] cov={parsed['coverage']}, prec={parsed['precision']}, disc={parsed['discriminability']}")
                break
        except Exception as e:
            if debug_print:
                print(f"[S1 Error] retry {retry + 1}/{MAX_RETRIES}: {e}")
    
    # =========================================================================
    # Stage 2: Counterfactual Prediction (No GT)
    # =========================================================================
    stage2_prompt = STAGE2_PROMPT_NO_GT.format(
        query=query,
        caption_list_str=caption_list_str,
    )
    
    for retry in range(MAX_RETRIES):
        try:
            response = llm.chat([{"role": "user", "content": stage2_prompt}], reward_model)
            judge_predictions = _parse_stage2_response_multi(response.content)
            
            if judge_predictions:
                gt_segments = [(g[0], g[1]) for g in gt_windows]
                # Use F1@0.3 and F1@0.5 average for lenient matching
                prf_metrics = compute_prf_metrics(judge_predictions, gt_segments, [0.3, 0.5])
                counterfactual_score = (prf_metrics.get('F1@0.3', 0) + prf_metrics.get('F1@0.5', 0)) / 2.0
                
                if debug_print:
                    print(f"[S2] preds={len(judge_predictions)}, F1@0.3={prf_metrics.get('F1@0.3',0):.3f}, F1@0.5={prf_metrics.get('F1@0.5',0):.3f}")
                break
            else:
                if debug_print:
                    print(f"[S2] No predictions from: {response.content[:100]}...")
        except Exception as e:
            if debug_print:
                print(f"[S2 Error] retry {retry + 1}/{MAX_RETRIES}: {e}")
    
    # =========================================================================
    # Compute Combined Score
    # =========================================================================
    combined_score = (
        CAPTION_COVERAGE_WEIGHT * coverage_score +
        CAPTION_PRECISION_WEIGHT * precision_score +
        CAPTION_DISCRIMINABILITY_WEIGHT * discriminability_score +
        CAPTION_COUNTERFACTUAL_WEIGHT * counterfactual_score
    )
    
    if debug_print:
        print(f"[Caption] {combined_score:.4f} = cov({coverage_score:.2f}*{CAPTION_COVERAGE_WEIGHT}) + "
              f"prec({precision_score:.2f}*{CAPTION_PRECISION_WEIGHT}) + "
              f"disc({discriminability_score:.2f}*{CAPTION_DISCRIMINABILITY_WEIGHT}) + "
              f"cf1({counterfactual_score:.2f}*{CAPTION_COUNTERFACTUAL_WEIGHT})")
    
    # Return raw scores in 0-10 scale
    raw_scores = [
        coverage_score * 10,
        precision_score * 10,
        discriminability_score * 10,
        counterfactual_score * 10,
    ]
    
    return combined_score, len(captions), raw_scores


# ==============================================================================
# Main Reward Function
# ==============================================================================

def tg_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Main temporal grounding reward function.
    
    Returns:
        Dictionary containing all reward components and metadata.
    """
    if extra_info is None:
        extra_info = {}
    
    strategy = get_current_strategy()
    enabled_rewards = strategy.get_enabled_rewards()
    weights = strategy.get_weights()
    
    if debug_print:
        print(f"[DEBUG] Strategy: {strategy.get_name()}, Enabled: {enabled_rewards}")
    
    result = {
        "score": 0.0,
        "strategy_name": strategy.get_name(),
    }
    
    try:
        if solution_str is None:
            print(f"[WARNING] solution_str is None, data_source: {data_source}")
            solution_str = ""
        
        solution_str_stripped = solution_str.strip()
        if len(solution_str_stripped) == 0:
            print(f"[WARNING] Empty response for data_source: {data_source}")
            return result
        
        # Compute individual rewards based on what's enabled
        tiou_score = 0.0
        format_score = 0.0
        caption_score = 0.0
        recall_score = 0.0
        precision_score = 0.0
        f1_score = 0.0
        cacc_score = 0.0
        length_penalty = 0.0
        
        if 'tiou' in enabled_rewards:
            tiou_score, _, _ = compute_tiou_reward(solution_str, ground_truth)
            result["tiou_score"] = tiou_score
        
        if 'format' in enabled_rewards:
            format_score, _ = compute_format_reward(solution_str)
            result["format_score"] = format_score
        
        # Legacy single-threshold recall
        if 'recall' in enabled_rewards and 'recall_avg' not in enabled_rewards:
            recall_score, _, _ = compute_recall_reward(solution_str, ground_truth)
            result["recall_score"] = recall_score
        
        # Multi-threshold metrics
        if any(r in enabled_rewards for r in ['precision', 'recall_avg', 'f1']):
            precision_score, recall_score, f1_score, prf_metrics = compute_prf_all_rewards(
                solution_str, ground_truth
            )
            
            if 'precision' in enabled_rewards:
                result["precision_score"] = precision_score
                for th in PRF_THRESHOLDS:
                    result[f'P@{th}'] = prf_metrics[f'P@{th}']
            
            if 'recall_avg' in enabled_rewards:
                result["recall_score"] = recall_score
                for th in PRF_THRESHOLDS:
                    result[f'R@{th}'] = prf_metrics[f'R@{th}']
            
            if 'f1' in enabled_rewards:
                result["f1_score"] = f1_score
                for th in PRF_THRESHOLDS:
                    result[f'F1@{th}'] = prf_metrics[f'F1@{th}']
        
        # C-Acc reward
        if 'cacc' in enabled_rewards:
            cacc_score, pred_count, gt_count = compute_cacc_reward(solution_str, ground_truth)
            result["cacc_score"] = cacc_score
            result["pred_count"] = pred_count
            result["gt_count"] = gt_count
        
        if 'caption' in enabled_rewards:
            think_content = extract_think_content(solution_str)
            captions = extract_captions_with_timestamps(think_content) if think_content else []
            
            caption_score, caption_count, raw_scores = compute_caption_reward(
                solution_str, ground_truth, extra_info
            )
            result["caption_score"] = caption_score
            result["caption_count"] = caption_count if caption_count > 0 else len(captions)
            result["caption_raw_scores"] = raw_scores
            # Always add caption_avg_length to ensure consistent keys across batch
            if captions:
                total_caption_length = sum(len(c.get("caption", "")) for c in captions)
                result["caption_avg_length"] = total_caption_length / len(captions)
            else:
                result["caption_avg_length"] = 0.0
        
        if 'length_penalty' in enabled_rewards:
            think_content = extract_think_content(solution_str)
            captions = extract_captions_with_timestamps(think_content) if think_content else []
            length_penalty = compute_length_penalty(think_content, captions)
            result["length_penalty"] = length_penalty
        
        # Combined score
        combined_score = (
            weights.tiou * tiou_score +
            weights.format * format_score +
            weights.caption * caption_score +
            weights.recall * recall_score +
            weights.precision * precision_score +
            weights.f1 * f1_score +
            weights.cacc * cacc_score -
            weights.length_penalty * length_penalty
        )
        combined_score = max(0.0, combined_score)
        result["score"] = combined_score
        
        if debug_print:
            print(f"[Final] score={combined_score:.4f}")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Exception in tg_reward: {traceback.format_exc()}")
        return result


# ==============================================================================
# Main Function for Testing
# ==============================================================================

if __name__ == "__main__":
    import os
    
    os.environ["TG_REWARD_STRATEGY"] = "tiou_format_f1_cacc_caption_length"
    os.environ["DEBUG_PRINT"] = "true"
    
    print("=" * 80)
    print("Testing tiou_format_f1_cacc_caption_length Strategy")
    print("=" * 80)
    
    # Test Case: Multiple GT Intervals
    solution_str = """<think>
<time>0.0 - 10.0 seconds</time> Host introduces the talent show.
<time>10.0 - 15.0 seconds</time> FIRST CLAP: Person in red claps after magic trick.
<time>15.0 - 30.0 seconds</time> Judges give feedback.
<time>30.0 - 35.0 seconds</time> SECOND CLAP: Same person claps after positive comments.
<time>35.0 - 55.0 seconds</time> Next act prepares.
<time>55.0 - 60.0 seconds</time> THIRD CLAP: Final applause as performer bows.
</think>
<time>10.0 - 15.0 seconds</time>
<time>30.0 - 35.0 seconds</time>
<time>55.0 - 60.0 seconds</time>"""
    
    ground_truth = """<time>10.0 - 15.0 seconds</time>
<time>30.0 - 35.0 seconds</time>
<time>55.0 - 60.0 seconds</time>"""
    
    extra_info = {"query": "a person clapping"}
    
    result = tg_reward(
        data_source="test",
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    
    print(f"\nResult: {json.dumps(result, indent=2)}")
    
    # Print strategy info
    print("\n" + "=" * 80)
    print("Strategy Information")
    print("=" * 80)
    strategy = get_current_strategy()
    print(f"Strategy Name: {strategy.get_name()}")
    print(f"Enabled Rewards: {strategy.get_enabled_rewards()}")
    weights = strategy.get_weights()
    print(f"Weights: tiou={weights.tiou:.3f}, format={weights.format:.3f}, "
          f"f1={weights.f1:.3f}, cacc={weights.cacc:.3f}, "
          f"caption={weights.caption:.3f}, length_penalty={weights.length_penalty:.3f}")