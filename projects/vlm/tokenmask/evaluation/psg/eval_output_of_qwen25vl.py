import argparse
import copy
import math
import os
import torch
import torchvision
import tqdm
from pycocotools import mask as mask_utils
import numpy as np
import random
import re
from PIL import Image
import json
import uuid
import torch.nn.functional as F
import hydra

from xtuner.model.utils import guess_load_checkpoint

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize
from projects.vlm.tokenmask.evaluation.psg.relation_utils import Result

from projects.vlm.tokenmask.evaluation.psg.psg_dataset import PanopticSceneGraphDataset


from types import MethodType
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.data.detection_utils import read_image, _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.utils.visualizer import GenericMask
import matplotlib.colors as mplc
def draw_instance_predictions_cache(self, labels, np_masks, jittering: bool = True):
    """
    Draw instance-level prediction results on an image.
    Args:
        predictions (Instances): the output of an instance detection/segmentation
            model. Following fields will be used to draw:
            "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        jittering: if True, in color mode SEGMENTATION, randomly jitter the colors per class
            to distinguish instances from the same class
    Returns:
        output (VisImage): image object with visualizations.
    """
    boxes = None
    scores = None
    classes = None
    keypoints = None

    masks = [GenericMask(x, self.output.height, self.output.width) for x in np_masks]

    if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
        colors = (
            [self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes]
            if jittering
            else [
                tuple(mplc.to_rgb([x / 255 for x in self.metadata.thing_colors[c]]))
                for c in classes
            ]
        )

        alpha = 0.8
    else:
        colors = None
        alpha = 0.5
    
    alpha = 0.4

    self.overlay_instances(
        masks=masks,
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=colors,
        alpha=alpha,
    )
    return self.output


def visualize(input_image, cat_masks, tags):
    if tags is None:
        left_tags = [f'{i}' for i in range(len(cat_masks))]
    else:
        left_tags = tags

    unique_tags = list(set(left_tags))
    text_prompt = ','.join(unique_tags)
    metadata = MetadataCatalog.get("__unused_ape_" + text_prompt)
    metadata.thing_classes = unique_tags
    metadata.stuff_classes = unique_tags

    result_masks = cat_masks
    input_image = _apply_exif_orientation(input_image)
    input_image = convert_PIL_to_numpy(input_image, "BGR")
    visualizer = Visualizer(input_image[:, :, ::-1], metadata, instance_mode=ColorMode.IMAGE)
    visualizer.draw_instance_predictions = MethodType(draw_instance_predictions_cache, visualizer)
    vis_output = visualizer.draw_instance_predictions(labels=left_tags, np_masks=result_masks)
    output_image = vis_output.get_image()
    output_image = Image.fromarray(output_image)

    return output_image




object_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard',
    'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit',
    'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
    'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
    'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone',
    'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
    'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
    'food-other-merged', 'building-other-merged', 'rock-merged',
    'wall-other-merged', 'rug-merged'
]

predicate_classes = [
    'over',
    'in front of',
    'beside',
    'on',
    'in',
    'attached to',
    'hanging from',
    'on back of',
    'falling off',
    'going down',
    'painted on',
    'walking on',
    'running on',
    'crossing',
    'standing on',
    'lying on',
    'sitting on',
    'flying over',
    'jumping over',
    'jumping from',
    'wearing',
    'holding',
    'carrying',
    'looking at',
    'guiding',
    'kissing',
    'eating',
    'drinking',
    'feeding',
    'biting',
    'catching',
    'picking',
    'playing with',
    'chasing',
    'climbing',
    'cleaning',
    'playing',
    'touching',
    'pushing',
    'pulling',
    'opening',
    'cooking',
    'talking to',
    'throwing',
    'slicing',
    'driving',
    'riding',
    'parked on',
    'driving on',
    'about to hit',
    'kicking',
    'swinging',
    'entering',
    'exiting',
    'enclosing',
    'leaning on',
]

def extract_mt_token_ids(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]

def find_first_index(arr, value):
    """
    在NumPy数组中找到第一个指定值的第一个出现的索引
    
    参数:
        arr: NumPy数组
        value: 要查找的值
        
    返回:
        第一个匹配值的索引，如果没有找到则返回-1
    """
    # 使用where找到所有匹配值的索引
    indices = np.where(arr == value)[0]
    
    # 返回第一个索引，如果没有找到则返回-1
    return indices[0] if len(indices) > 0 else -1


def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument(
        '--vq_sam2_path',
        default="pretrained_weights/iter_175473.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--dataset',
        default='./data/PaDT-MLLM/RefCOCO/refcoco_val.json',
        help='Specify a ref dataset')
    args = parser.parse_args()
    return args


def parse_output_text(output_text):
    pass

from collections import defaultdict
def _dedup_triplets_based_on_iou(s_labels, o_labels, r_labels, s_mask_pred, o_mask_pred):
    relation_classes = defaultdict(lambda: [])
    for k, (s_l, o_l, r_l) in enumerate(zip(s_labels, o_labels, r_labels)):
        relation_classes[(s_l.item(), o_l.item(),
                            r_l.item())].append(k)
    s_binary_masks = s_mask_pred.to(torch.float).flatten(1)
    o_binary_masks = o_mask_pred.to(torch.float).flatten(1)

    def dedup_triplets(triplets_ids, s_binary_masks, o_binary_masks, keep_tri):
        while len(triplets_ids) > 1:
            base_s_mask = s_binary_masks[triplets_ids[0]].unsqueeze(0)
            base_o_mask = o_binary_masks[triplets_ids[0]].unsqueeze(0)
            other_s_masks = s_binary_masks[triplets_ids[1:]]
            other_o_masks = o_binary_masks[triplets_ids[1:]]
            # calculate ious
            s_ious = base_s_mask.mm(other_s_masks.transpose(
                0, 1))/((base_s_mask+other_s_masks) > 0).sum(-1)
            o_ious = base_o_mask.mm(other_o_masks.transpose(
                0, 1))/((base_o_mask+other_o_masks) > 0).sum(-1)
            ids_left = []
            for s_iou, o_iou, other_id in zip(s_ious[0], o_ious[0], triplets_ids[1:]):
                if (s_iou > 0.5) & (o_iou > 0.5):
                    keep_tri[other_id] = False
                else:
                    ids_left.append(other_id)
            triplets_ids = ids_left
        return keep_tri

    keep_tri = torch.ones_like(
        r_labels, dtype=torch.bool, device=r_labels.device)
    for triplets_ids in relation_classes.values():
        if len(triplets_ids) > 1:
            keep_tri = dedup_triplets(
                triplets_ids, s_binary_masks, o_binary_masks, keep_tri)

    return keep_tri

def _dedup_objects_based_on_iou(labels, masks_binary):
    '''
    Parameters
    ----------
    labels: (K, )
    masks_binary: (K, H, W)
        each pixel in masks contains 0 or 1

    Return
    ------
    keep_num: int
    old2new_map: dict
        {int: int}
    new2old_map: dict
        {int: List[int]}
    '''
    thing_classes = defaultdict(lambda: [])
    thing_dedup = defaultdict(lambda: [])
    stuff_merge = defaultdict(lambda: [])
    for k, label in enumerate(labels):
        if label.item() < 80:
            thing_classes[label.item()].append(k)
        else:
            stuff_merge[label.item()].append(k)

    masks_binary = masks_binary.to(torch.float).flatten(1)
    for thing_ids in thing_classes.values():
        if len(thing_ids) > 1:
            while len(thing_ids) > 1:
                base_mask = masks_binary[thing_ids[0:1]]
                other_masks = masks_binary[thing_ids[1:]]
                ious = base_mask.mm(other_masks.transpose(
                    0, 1)) / ((base_mask + other_masks) > 0).sum(-1)
                ids_left = []
                thing_dedup[thing_ids[0]].append(thing_ids[0])
                for iou, other_id in zip(ious[0], thing_ids[1:]):
                    if iou > 0.5:
                        thing_dedup[thing_ids[0]].append(other_id)
                    else:
                        ids_left.append(other_id)
                thing_ids = ids_left
            if len(thing_ids) == 1:
                thing_dedup[thing_ids[0]].append(thing_ids[0])
        else:
            thing_dedup[thing_ids[0]].append(thing_ids[0])

    keep_num = len(thing_dedup.keys()) + len(stuff_merge.keys())
    old2new_map = dict()
    new2old_map = dict()
    new_id = 0
    for thing_ids in thing_dedup.values():
        new2old_map[new_id] = thing_ids
        for thing_id in thing_ids:
            old2new_map[thing_id] = new_id
        new_id += 1
    for stuff_ids in stuff_merge.values():
        new2old_map[new_id] = stuff_ids
        for stuff_id in stuff_ids:
            old2new_map[stuff_id] = new_id
        new_id += 1

    return keep_num, old2new_map, new2old_map

from mmdet.evaluation.functional.panoptic_utils import INSTANCE_OFFSET
def _get_results_single(
    s_cls_score, o_cls_score, r_cls_score,
    s_bbox_pred, o_bbox_pred,
    s_mask_pred, o_mask_pred):

    s_cls_score = s_cls_score.to(s_bbox_pred.device)
    o_cls_score = o_cls_score.to(o_bbox_pred.device)
    r_cls_score = r_cls_score.to(s_bbox_pred.device)

    # max_per_img = 100

    mask_size = (s_mask_pred.shape[-2], s_mask_pred.shape[-1])

    ###################
    # sub/obj/rel cls #
    ###################
    # 0-based label input for objects, self.num_classes as default background class
    s_logits = F.softmax(s_cls_score, dim=-1)[..., :-1]
    o_logits = F.softmax(o_cls_score, dim=-1)[..., :-1]
    s_scores, s_labels = s_logits.max(-1)
    o_scores, o_labels = o_logits.max(-1)

    # 1-based label input for relationships, 0 as default no relationship class
    r_lgs = F.softmax(r_cls_score, dim=-1)
    r_logits = r_lgs[..., 1:]
    # Top K
    num_relations = len(predicate_classes)
    max_per_img = r_logits.shape[0]
    r_scores, r_indexes = r_logits.reshape(-1).topk(max_per_img)
    r_labels = r_indexes % num_relations + 1
    triplet_index = r_indexes // num_relations

    s_scores = s_scores[triplet_index]
    s_labels = s_labels[triplet_index]
    s_bbox_pred = s_bbox_pred[triplet_index]
    s_mask_pred = s_mask_pred[triplet_index]

    o_scores = o_scores[triplet_index]
    o_labels = o_labels[triplet_index]
    o_bbox_pred = o_bbox_pred[triplet_index]
    o_mask_pred = o_mask_pred[triplet_index]

    r_dists = r_lgs.reshape(-1, num_relations + 1)[triplet_index]

    if os.getenv('EVAL_PAN_RELS', 'false').lower() == 'true':
        keep = (s_scores > 0.5) & (o_scores > 0.5) & (r_scores > 0.1)
    else:
        keep = (s_scores > 0.0) & (o_scores > 0.0) & (r_scores > 0.0)
    s_scores = s_scores[keep]
    s_labels = s_labels[keep]
    s_bbox_pred = s_bbox_pred[keep]
    s_mask_pred = s_mask_pred[keep]
    o_scores = o_scores[keep]
    o_labels = o_labels[keep]
    o_bbox_pred = o_bbox_pred[keep]
    o_mask_pred = o_mask_pred[keep]
    r_scores = r_scores[keep]
    r_labels = r_labels[keep]
    r_dists = r_dists[keep]

    ################
    # sub/obj bbox #
    ################
    # s_bboxes = bbox_cxcywh_to_xyxy(s_bbox_pred)
    # s_bboxes[:, 0::2] = s_bboxes[:, 0::2] * img_shape[1]
    # s_bboxes[:, 1::2] = s_bboxes[:, 1::2] * img_shape[0]
    # s_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
    # s_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
    # if rescale:
    #     s_bboxes /= s_bboxes.new_tensor(scale_factor)
    # s_bboxes = torch.cat((s_bboxes, s_scores.unsqueeze(1)), -1)
    s_bboxes = torch.cat((s_bbox_pred, s_scores.unsqueeze(1).to(s_bbox_pred.device)), -1)

    # o_bboxes = bbox_cxcywh_to_xyxy(o_bbox_pred)
    # o_bboxes[:, 0::2] = o_bboxes[:, 0::2] * img_shape[1]
    # o_bboxes[:, 1::2] = o_bboxes[:, 1::2] * img_shape[0]
    # o_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
    # o_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
    # if rescale:
    #     o_bboxes /= o_bboxes.new_tensor(scale_factor)
    # o_bboxes = torch.cat((o_bboxes, o_scores.unsqueeze(1)), -1)
    o_bboxes = torch.cat((o_bbox_pred, o_scores.unsqueeze(1).to(o_bbox_pred.device)), -1)
    
    ################
    # sub/obj mask #
    ################
    # s_mask_pred = F.interpolate(s_mask_pred.unsqueeze(1),
    #                             size=mask_size).squeeze(1)
    # o_mask_pred = F.interpolate(o_mask_pred.unsqueeze(1),
    #                             size=mask_size).squeeze(1)
    s_mask_pred_logits = s_mask_pred
    o_mask_pred_logits = o_mask_pred
    # s_mask_pred = torch.sigmoid(s_mask_pred) > 0.85
    # o_mask_pred = torch.sigmoid(o_mask_pred) > 0.85
    s_mask_pred = s_mask_pred > 0
    o_mask_pred = o_mask_pred > 0

    ########################
    # triplets deduplicate #
    ########################
    keep_tri = _dedup_triplets_based_on_iou(
        s_labels, o_labels, r_labels, s_mask_pred, o_mask_pred)

    ###################
    # complete output #
    ###################
    # object score, (2*n, )
    s_scores = s_scores[keep_tri]
    o_scores = o_scores[keep_tri]
    complete_scores = torch.cat((s_scores, o_scores), 0)
    # object label, (2*n, )
    s_labels = s_labels[keep_tri]
    o_labels = o_labels[keep_tri]
    complete_labels = torch.cat((s_labels, o_labels), 0)
    # object bbox, (2*n, 5)
    s_bboxes = s_bboxes[keep_tri]
    o_bboxes = o_bboxes[keep_tri]
    complete_bboxes = torch.cat((s_bboxes, o_bboxes), 0)
    # object mask, (2*n, h, w)
    s_mask_pred_logits = s_mask_pred_logits[keep_tri]
    o_mask_pred_logits = o_mask_pred_logits[keep_tri]
    complete_masks_logits = torch.cat(
        (s_mask_pred_logits, o_mask_pred_logits), 0)
    s_mask_pred = s_mask_pred[keep_tri]
    o_mask_pred = o_mask_pred[keep_tri]
    complete_masks_binary = torch.cat((s_mask_pred, o_mask_pred), 0)
    # relation (n, )
    r_labels = r_labels[keep_tri]
    r_scores = r_scores[keep_tri]
    r_dists = r_dists[keep_tri]
    # (n, 2)
    complete_rel_pairs = torch.arange(keep_tri.sum()*2,
                                        dtype=r_labels.dtype,
                                        device=r_labels.device).reshape(2, -1).T
    complete_r_scores = r_scores
    complete_r_labels = r_labels
    complete_r_dists = r_dists
    complete_triplets = torch.cat(
        (complete_rel_pairs, complete_r_labels.unsqueeze(-1)), dim=1)

    complete_masks_score = complete_scores.view(
        -1, 1, 1) * complete_masks_logits
    h, w = complete_masks_score.shape[-2:]

    ################
    # panoptic seg #
    ################
    num_classes = len(object_classes)
    assert num_classes + 1 == s_cls_score.shape[-1]
    panoptic_seg = torch.full(
        (h, w), num_classes, dtype=torch.int32, device=complete_masks_score.device)

    if complete_labels.numel() == 0:
        new_labels = torch.tensor([0])
        new_bboxes = torch.zeros((1, 5))
        new_masks_binary = panoptic_seg.unsqueeze(0).cpu().to(torch.long)
        new_rel_pairs = torch.arange(len(complete_labels), dtype=torch.int).to(
            complete_masks_binary.device).reshape(2, -1).T
        new_r_scores = complete_r_scores
        new_r_labels = complete_r_labels
        new_r_dists = complete_r_dists
        new_triplets = torch.tensor([0, 0, 0]).view(-1, 3)
        panoptic_seg = torch.ones(mask_size).cpu().to(torch.long)
    else:
        # 1. generation panoptic seg
        # 2. assign each subject/object to panoptic seg
        keep_num, old2new_map, new2old_map = _dedup_objects_based_on_iou(
            complete_labels, complete_masks_binary)

        new_scores = complete_scores.new_zeros((keep_num))
        new_labels = complete_labels.new_zeros((keep_num))
        new_bboxes = complete_bboxes.new_zeros((keep_num, 5))
        new_masks_logits = complete_masks_logits.new_zeros(
            (keep_num, h, w))
        new_masks_score = complete_masks_score.new_zeros((keep_num, h, w))

        new_rel_pairs = torch.zeros_like(complete_rel_pairs)
        new_r_scores = complete_r_scores
        new_r_labels = complete_r_labels
        new_r_dists = complete_r_dists

        for k, v in new2old_map.items():
            new_scores[k] = complete_scores[v].mean(dim=0)
            new_labels[k] = complete_labels[v[0]]
            new_bboxes[k] = complete_bboxes[v].mean(dim=0)
            new_masks_logits[k] = complete_masks_logits[v].to(torch.float32).mean(dim=0)
            new_masks_score[k] = complete_masks_score[v].to(torch.float32).mean(dim=0)
        new_masks_binary = new_masks_score > 0.8

        for ii in range(complete_rel_pairs.shape[0]):
            for jj in range(complete_rel_pairs.shape[1]):
                new_rel_pairs[ii,
                                jj] = old2new_map[complete_rel_pairs[ii, jj].item()]
        new_triplets = torch.cat(
            (new_rel_pairs, new_r_labels.unsqueeze(-1)), dim=1)

        mask_score, mask_ids = new_masks_score.max(dim=0)
        instance_id = 1
        for k in range(new_labels.shape[0]):
            pred_class = new_labels[k].to(torch.long)
            isthing = pred_class < 80
            mask = mask_ids == k
            mask_area = mask.sum().item()
            original_area = (new_masks_logits[k] >= 0.5).sum().item()
            filter_low_score = True
            if filter_low_score:
                mask = mask & (new_masks_logits[k] >= 0.5)
            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < 0.8:
                    continue
                if not isthing:
                    # different stuff regions of same class will be
                    # merged here, and stuff share the instance_id 0.
                    panoptic_seg[mask] = panoptic_seg[mask] * \
                        0 + pred_class
                else:
                    panoptic_seg[mask] = panoptic_seg[mask] * 0 + \
                        (pred_class + instance_id * INSTANCE_OFFSET)
                    instance_id += 1

    return complete_labels, complete_bboxes, complete_masks_binary, complete_rel_pairs, complete_r_scores, complete_r_labels, complete_r_dists, complete_triplets, \
        new_labels, new_bboxes, new_masks_binary, new_rel_pairs, new_r_scores, new_r_labels, new_r_dists, new_triplets, panoptic_seg
    

def triplet2Result(triplets, use_mask, eval_pan_rels=os.getenv('EVAL_PAN_RELS', 'false').lower() == 'true'):
    if isinstance(triplets, Result):
        return triplets
    
    if True:
        complete_labels, complete_bboxes, complete_masks_binary, complete_rel_pairs, complete_r_scores, complete_r_labels, complete_r_dists, complete_triplets, \
            new_labels, new_bboxes, new_masks_binary, new_rel_pairs, new_r_scores, new_r_labels, new_r_dists, new_triplets, panoptic_seg = triplets
        complete_labels = complete_labels.detach().cpu().numpy()
        complete_bboxes = complete_bboxes.detach().cpu().numpy()
        complete_masks_binary = complete_masks_binary.detach().cpu().numpy()
        complete_rel_pairs = complete_rel_pairs.detach().cpu().numpy()
        complete_r_scores = complete_r_scores.detach().cpu().numpy()
        complete_r_labels = complete_r_labels.detach().cpu().numpy()
        complete_r_dists = complete_r_dists.detach().cpu().numpy()
        complete_triplets = complete_triplets.detach().cpu().numpy()
        new_labels = new_labels.detach().cpu().numpy()
        new_bboxes = new_bboxes.detach().cpu().numpy()
        new_masks_binary = new_masks_binary.detach().cpu().numpy()
        new_rel_pairs = new_rel_pairs.detach().cpu().numpy()
        new_r_scores = new_r_scores.detach().cpu().numpy()
        new_r_labels = new_r_labels.detach().cpu().numpy()
        new_r_dists = new_r_dists.detach().cpu().numpy()
        new_triplets = new_triplets.detach().cpu().numpy()
        panoptic_seg = panoptic_seg.detach().cpu().numpy()
        # if eval_pan_rels:
        return Result(refine_bboxes=new_bboxes,  # (2*n, 5)
                        labels=new_labels+1,  # (2*n)
                        formatted_masks=dict(
                            pan_results=panoptic_seg),  # (h, w)
                        rel_pair_idxes=new_rel_pairs,  # (n, 2)
                        rel_scores=new_r_scores,  # (n)
                        rel_labels=new_r_labels,  # (n)
                        rel_dists=new_r_dists,  # (n, 57)
                        pan_results=panoptic_seg,  # (h, w)
                        masks=new_masks_binary,  # (2*n, h, w)
                        rels=new_triplets)  # (n, 3)
    #     else:
    #         return Result(refine_bboxes=complete_bboxes,  # (2*n, 5)
    #                       labels=complete_labels+1,  # (2*n)
    #                       formatted_masks=dict(
    #                           pan_results=panoptic_seg),  # (h, w)
    #                       rel_pair_idxes=complete_rel_pairs,  # (n, 2)
    #                       rel_scores=complete_r_scores,  # (n)
    #                       rel_labels=complete_r_labels,  # (n)
    #                       rel_dists=complete_r_dists,  # (n, 57)
    #                       pan_results=panoptic_seg,  # (h, w)
    #                       masks=complete_masks_binary,  # (2*n, h, w)
    #                       rels=complete_triplets)  # (n, 3)
    # else:
    #     bboxes, labels, rel_pairs, r_labels, r_dists = triplets
    #     labels = labels.detach().cpu().numpy()
    #     bboxes = bboxes.detach().cpu().numpy()
    #     rel_pairs = rel_pairs.detach().cpu().numpy()
    #     r_labels = r_labels.detach().cpu().numpy()
    #     r_dists = r_dists.detach().cpu().numpy()
    #     return Result(
    #         refine_bboxes=bboxes,
    #         labels=labels,
    #         formatted_masks=dict(pan_results=None),
    #         rel_pair_idxes=rel_pairs,
    #         rel_dists=r_dists,
    #         rel_labels=r_labels,
    #         pan_results=None,
    #     )


def main():
    args = parse_args()

    # build vq-sam2 model
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    with hydra.initialize(version_base=None, config_path="../../../../../projects/transformers/vq_sam2/sam2/sam2_configs"):
        sam2_config = SAM2Config(
            cfg_path="sam2.1_hiera_l.yaml",
            ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
        )
        
        vq_sam2_config = VQ_SAM2Config(
            sam2_config=sam2_config,
            codebook_size=CODEBOOK_SIZE,
            codebook_depth=CODEBOOK_DEPTH,
            shared_codebook=False,
            latent_dim=256,
        )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    state = torch.load(args.vq_sam2_path, map_location="cpu")
    vq_sam2.load_state_dict(state)

    sam2_image_processor = DirectResize(1024)

    # output
    num_classes = len(object_classes)
    num_relations = len(predicate_classes)
    object_class_2_object_id = {cls_name: id for id, cls_name in enumerate(object_classes)}
    relation_class_2_relation_id = {rel_name: id for id, rel_name in enumerate(predicate_classes)}
    
    all_psg_list = []
    for json_file in os.listdir(args.dataset):
        json_path = os.path.join(args.dataset, json_file)
        with open(json_path, 'r') as f:
            pred_psg = json.load(f)
            all_psg_list.append(pred_psg)

    from mmengine.fileio.io import load
    dataset = load("./data/psg_data/psg_val.json")
    image_id_2_annos = {}
    for item in dataset['annotations']:
        image_id_2_annos[item['image_id']] = item
    relations_categories = dataset['relations_categories']

    category_id_2_category_name = {item['id']: item['name'] for item in dataset['categories']}
    image_id_2_pan_seg_file = {item['image_id']: os.path.join('./data/coco/annotations', item['file_name']) for item in dataset['annotations']}
    
    
    all_sg_results = {}
    for psg_info in tqdm.tqdm(all_psg_list):
        sub_outputs_class = []
        obj_outputs_class = []
        rel_outputs_class = []
        sub_outputs_mask = []
        obj_outputs_mask = []
        sub_outputs_bbox = []
        obj_outputs_bbox = []

        file_name = psg_info['file_name']
        image_id = psg_info['image_id']
        image_path = os.path.join('./data/coco', file_name)
        raw_triplet_list = psg_info['psg_list']

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size
        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        if not isinstance(raw_triplet_list, list):
            raw_triplet_list = [raw_triplet_list]
        for triplet_id, one_triplet in enumerate(raw_triplet_list):
            subject_label = one_triplet['subject']['label'].strip()
            if subject_label in object_class_2_object_id:
                subject_id = object_class_2_object_id[subject_label]
            else:
                print("subject_label: ", subject_label, " is not included in the candidate categories!!!")
                continue
            object_label = one_triplet['object']['label'].strip()
            if object_label in object_class_2_object_id:
                object_id = object_class_2_object_id[object_label]
            else:
                print("object_label: ", object_label, " is not included in the candidate categories!!!")
                continue
            relation_label = one_triplet['predicate'].strip()
            if relation_label in relation_class_2_relation_id:
                relation_id = relation_class_2_relation_id[relation_label]
            else:
                print("relation_label: ", relation_label, " is not included in the candidate predicate classes!!!")
                continue

            sub_output_class = torch.zeros((num_classes+1,), dtype=torch.float32)
            obj_output_class = torch.zeros((num_classes+1,), dtype=torch.float32)
            rel_output_class = torch.zeros((num_relations+1,), dtype=torch.float32)

            sub_output_class[subject_id] = 10.0
            obj_output_class[object_id] = 10.0
            # 1-based label input for relationships, 0 as default no relationship class
            rel_output_class[relation_id+1] = 10.0

            sub_outputs_class.append(sub_output_class)
            obj_outputs_class.append(obj_output_class)
            rel_outputs_class.append(rel_output_class)

            subject_mask_token = one_triplet['subject']['mask_2d']
            object_mask_token = one_triplet['object']['mask_2d']

            #===================subject
            subject_quant_ids = extract_mt_token_ids(subject_mask_token)
            if len(subject_quant_ids) == 0:
                continue
            
            remap_subject_quant_ids = np.array([-1 for _ in range(CODEBOOK_DEPTH)])
            for quant_id in subject_quant_ids:
                depth_idx = quant_id // CODEBOOK_SIZE
                remap_subject_quant_ids[depth_idx] = quant_id % CODEBOOK_SIZE

            truncated_idx = find_first_index(remap_subject_quant_ids, -1)
            if truncated_idx != -1:
                remap_subject_quant_ids[truncated_idx:] = -1
            if remap_subject_quant_ids[0] == -1:
                continue
            subject_quant_ids = torch.LongTensor(remap_subject_quant_ids).to(vq_sam2.device).unsqueeze(0)

            subject_pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, subject_quant_ids)
            subject_pred_masks = torch.nn.functional.interpolate(subject_pred_masks, size=(ori_height, ori_width), mode='bilinear')
            subject_pred_masks = subject_pred_masks > 0.5
            subject_pred_masks = subject_pred_masks[0, 0, :, :]
            sub_outputs_mask.append(subject_pred_masks)
            try:
                sub_outputs_bbox.append(torchvision.ops.masks_to_boxes(subject_pred_masks.unsqueeze(0))[0])
            except:
                sub_outputs_bbox.append(torch.zeros((4,), dtype=torch.float32, device='cuda'))

            #===================object
            object_quant_ids = extract_mt_token_ids(object_mask_token)
            if len(object_mask_token) == 0:
                zero_mask = torch.zeros((ori_height, ori_width), dtype=torch.float32)
                obj_outputs_mask.append(zero_mask)
                obj_outputs_bbox.append(torch.zeros((4,), dtype=torch.float32, device='cuda'))
                continue
            
            remap_object_quant_ids = np.array([-1 for _ in range(CODEBOOK_DEPTH)])
            for quant_id in object_quant_ids:
                depth_idx = quant_id // CODEBOOK_SIZE
                remap_object_quant_ids[depth_idx] = quant_id % CODEBOOK_SIZE
            
            truncated_idx = find_first_index(remap_object_quant_ids, -1)
            if truncated_idx != -1:
                remap_object_quant_ids[truncated_idx:] = -1
            if remap_object_quant_ids[0] == -1:
                zero_mask = torch.zeros((ori_height, ori_width), dtype=torch.float32)
                obj_outputs_mask.append(zero_mask)
                obj_outputs_bbox.append(torch.zeros((4,), dtype=torch.float32, device='cuda'))
                continue
            object_quant_ids = torch.LongTensor(remap_object_quant_ids).to(vq_sam2.device).unsqueeze(0)

            object_pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, object_quant_ids)
            object_pred_masks = torch.nn.functional.interpolate(object_pred_masks, size=(ori_height, ori_width), mode='bilinear')
            object_pred_masks = object_pred_masks > 0.5
            object_pred_masks = object_pred_masks[0, 0, :, :]
            obj_outputs_mask.append(object_pred_masks)
            try:
                obj_outputs_bbox.append(torchvision.ops.masks_to_boxes(object_pred_masks.unsqueeze(0))[0])
            except:
                obj_outputs_bbox.append(torch.zeros((4,), dtype=torch.float32, device='cuda'))

            
        #     sub_obj_masks = np.stack((subject_pred_masks.cpu().numpy(), object_pred_masks.cpu().numpy()), axis=0)
        #     tags = ['sub', 'obj']
        #     output_images = visualize(image, sub_obj_masks, tags)
        #     output_images.save(f'psg_triplet_{triplet_id}.jpg')
        #     print(f"=====>triplet#{triplet_id}: ", one_triplet)
        # print("=======>image_id: ", image_id)
        # exit(0)

        

        # image_annos = image_id_2_annos[image_id]
        # for triplet_id, one_triplet in enumerate(image_annos['relations']):
        #     sub_segment_info = image_annos['segments_info'][one_triplet[0]]
        #     obj_segment_info = image_annos['segments_info'][one_triplet[1]]
        #     relation_class = relations_categories[one_triplet[2]]

        #     pan_seg_file = image_id_2_pan_seg_file[image_id]
        #     from panopticapi.utils import rgb2id
        #     pan_seg_gt = read_image(pan_seg_file, "RGB")
        #     pan_seg_gt = rgb2id(pan_seg_gt)

            # sub_mask = pan_seg_gt == sub_segment_info['id']
            # obj_mask = pan_seg_gt == obj_segment_info['id']

        #     sub_obj_masks = np.stack((sub_mask, obj_mask), axis=0)
        #     tags = ['sub', 'obj']
        #     output_images = visualize(image, sub_obj_masks, tags)
        #     output_images.save(f'psg_triplet_anno_{triplet_id}.jpg')
        #     print(f"<sub, obj, rel>#{triplet_id}: ", (category_id_2_category_name[sub_segment_info['category_id']], category_id_2_category_name[obj_segment_info['category_id']], relation_class['name']))
        # exit(0)

            

        sub_outputs_class = torch.stack(sub_outputs_class).unsqueeze(0).unsqueeze(0) # num_decoder_layer, batch_size, num_queries, num_classes+1
        obj_outputs_class = torch.stack(obj_outputs_class).unsqueeze(0).unsqueeze(0) # num_decoder_layer, batch_size, num_queries, num_classes+1
        rel_outputs_class = torch.stack(rel_outputs_class).unsqueeze(0).unsqueeze(0) # num_decoder_layer, batch_size, num_queries, num_relations+1
        sub_outputs_mask = torch.stack(sub_outputs_mask).unsqueeze(0).unsqueeze(0) # lbqhw
        obj_outputs_mask = torch.stack(obj_outputs_mask).unsqueeze(0).unsqueeze(0) # lbqhw
        sub_outputs_bbox = torch.stack(sub_outputs_bbox).unsqueeze(0).unsqueeze(0) # lbq4
        obj_outputs_bbox = torch.stack(obj_outputs_bbox).unsqueeze(0).unsqueeze(0) # lbq4

        all_cls_scores = dict(sub=sub_outputs_class,
                                obj=obj_outputs_class,
                                rel=rel_outputs_class)
        all_bbox_preds = dict(sub=sub_outputs_bbox,
                                obj=obj_outputs_bbox)
        all_mask_preds = dict(sub=sub_outputs_mask,
                                obj=obj_outputs_mask)
        # output_dict = dict()
        # output_dict['all_cls_scores'] = all_cls_scores
        # output_dict['all_bbox_preds'] = all_bbox_preds
        # output_dict['all_mask_preds'] = all_mask_preds


        # get_results
        s_cls_score = all_cls_scores['sub'][-1, 0, ...]
        o_cls_score = all_cls_scores['obj'][-1, 0, ...]
        r_cls_score = all_cls_scores['rel'][-1, 0, ...]
        s_bbox_pred = all_bbox_preds['sub'][-1, 0, ...]
        o_bbox_pred = all_bbox_preds['obj'][-1, 0, ...]
        s_mask_pred = all_mask_preds['sub'][-1, 0, ...]
        o_mask_pred = all_mask_preds['obj'][-1, 0, ...]
        triplets = _get_results_single(s_cls_score, o_cls_score,
                                            r_cls_score, s_bbox_pred,
                                            o_bbox_pred, s_mask_pred,
                                            o_mask_pred)
        sg_results = triplet2Result(triplets, True)
        # all_sg_results.append(sg_results)
        all_sg_results[image_id] = sg_results
    
    eval_dataset = PanopticSceneGraphDataset(
        ann_file="./data/psg_data/psg_val.json",
        img_prefix="./data/coco",
        seg_prefix="./data/coco",
        pipeline=None,
        split='test',
        all_bboxes=True,
    )

    evaluation1 = dict(metric=['sgdet'],
                  relation_mode=True,
                  classwise=True,
                  iou_thrs=0.5,
                  detection_method='pan_seg')

    # evaluation2 = dict(metric=['PQ'],
    #                 relation_mode=True,
    #                 classwise=True,
    #                 iou_thrs=0.5,
    #                 detection_method='pan_seg')
    
    image_ids = []
    all_sg_results_v = []
    for k, v in all_sg_results.items():
        image_ids.append(k)
        all_sg_results_v.append(v)
        
    metric1 = eval_dataset.evaluate(all_sg_results_v, image_ids, **evaluation1)
    # metric2 = eval_dataset.evaluate(all_sg_results, **evaluation2)

    # result_str = 'epoch=xx, PQ={:.2f}\nR/mR@20={:.2f}/{:.2f}\nR/mR@50={:.2f}/{:.2f}\nR/mR@100={:.2f}/{:.2f}'.format(
    #     metric2['PQ'],
    #     metric1['sgdet_recall_R_20'] * 100,
    #     metric1['sgdet_mean_recall_mR_20'] * 100,
    #     metric1['sgdet_recall_R_50'] * 100,
    #     metric1['sgdet_mean_recall_mR_50'] * 100,
    #     metric1['sgdet_recall_R_100'] * 100,
    #     metric1['sgdet_mean_recall_mR_100'] * 100,
    # )
    result_str = 'epoch=xx\nR/mR@20={:.2f}/{:.2f}\nR/mR@50={:.2f}/{:.2f}\nR/mR@100={:.2f}/{:.2f}'.format(
        metric1['sgdet_recall_R_20'] * 100,
        metric1['sgdet_mean_recall_mR_20'] * 100,
        metric1['sgdet_recall_R_50'] * 100,
        metric1['sgdet_mean_recall_mR_50'] * 100,
        metric1['sgdet_recall_R_100'] * 100,
        metric1['sgdet_mean_recall_mR_100'] * 100,
    )
    print(result_str)


if __name__ == '__main__':
    main()