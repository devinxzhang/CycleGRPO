import os
import sys
import collections
import os.path as osp
import random
import copy
from typing import Dict, List
from PIL import Image
import numpy as np
import torch
import torchvision
from pycocotools import mask as mask_utils
import json
import tqdm
import glob

import mmengine
from mmengine.dataset import BaseDataset

from xtuner.model.utils import guess_load_checkpoint

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize


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
    
    alpha = 0.5

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

def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union


def decode_mask(object_masks, ori_height, ori_width):
    binary_masks = []
    for object_mask in object_masks:
        if isinstance(object_mask, dict):
            if isinstance(object_mask["counts"], list):
                # convert to compressed RLE
                object_mask = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
            m = mask_utils.decode(object_mask)
            m = m.astype(np.uint8).squeeze()
        elif object_mask:
            rles = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
            rle = mask_utils.merge(rles)
            m = mask_utils.decode(rle).astype(np.uint8).squeeze()
        else:
            m = np.zeros((ori_height, ori_width), dtype=np.uint8)
        binary_masks.append(m)
    return binary_masks

import cv2
def get_mask_from_json(json_path, img):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    height, width = img.shape[:2]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence

SHORT_QUESTION_LIST = [
    "<image>\n" + "Can you segment the {class_name} in this image?",
    "<image>\n" + "Please segment the {class_name} in this image.",
    "<image>\n" + "What is {class_name} in this image? Please respond with segmentation mask.",
    "<image>\n" + "What is {class_name} in this image? Please output segmentation mask.",
]

LONG_QUESTION_LIST = [
    "<image>\n" + "{sent} Please respond with segmentation mask.",
    "<image>\n" + "{sent} Please output segmentation mask.",
    "<image>\n" + "{sent} Provide the segmentation mask.",
    "<image>\n" + "{sent} Output the segmentation mask.",
    "<image>\n" + "{sent} Please show the segmentation mask.",
    "<image>\n" + "{sent} I'd appreciate segmentation masks.",
    "<image>\n" + "{sent} Please highlight the segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

ANSWER_LIST = [
    "It is {seg}.",
    "Sure, {seg}.",
    "Sure, it is {seg}.",
    "Sure, the segmentation result is {seg}.",
    "{seg}.",
]

MR_SINGLE_ANSWER_LIST = [
    "{class_name} is [SEG].",
]

MR_MULTI_ANSWER_LIST = [
    "{class_name} are {seg}, separately.",
    "{class_name} are {seg}.",
    "Sure, {class_name} are {seg}, separately.",
    "Sure, {class_name} are {seg}.",
    "the segmentation result of {class_name} are {seg}.",
    "the segmentation result of {class_name} are {seg}, separately.",
    "Sure, the segmentation result of {class_name} are {seg}.",
    "Sure, the segmentation result of {class_name} are {seg}, separately.",
]

def main(task_id):
    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    dataset_name = "mmr"
    temp_save_root = "./temp_data_256x2_0927/mmr/"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)

    sam2_config = SAM2Config(
        cfg_path="sam2.1_hiera_l.yaml",
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
    )

    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=256,
    )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    pretrained_pth = "pretrained_weights/iter_129437_256x2.pth"
    pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

    pretrained_state_dict_new = {}
    for key in pretrained_state_dict.keys():
        new_key = copy.deepcopy(key)
        if key.startswith('hf_model.'):
            new_key = new_key[len('hf_model.'):]
        pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
    
    vq_sam2.load_state_dict(pretrained_state_dict_new)

    sam2_image_processor = DirectResize(1024)

    with open("./data/mmr_data/MMR_train.json", "r") as f:
        all_data_dict = json.load(f)
    
    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    rows = len(all_data_dict)
    chunk_size = (rows+7) // 8
    _start_ = task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = rows if _end_ > rows else _end_

    for image_info in tqdm.tqdm(all_data_dict[_start_:_end_]):
        if "file_name" in image_info:    
            image_root = "./data/coco"
            image_path = os.path.join(image_root, image_info["file_name"])
            
        anns = image_info['annotations']
        question = image_info['questions']
        gt_answer = image_info['answers']
        text_answers = image_info['text_answers']

        masks = []

        sampled_inds = list(range(len(question)))
        sampled_sents = np.vectorize(question.__getitem__)(sampled_inds).tolist()

        sampled_answers = gt_answer
        sampled_masks = masks
        sampled_text_answers = text_answers

        image_name = image_path.split("/")[-1]
        questions = []
        answers = []

        seg_token = '[SEG]'
        skip_this_case = False
        print_this_case = False
        if len(question) != 0:
            for text, answer_list, text_answer in zip(sampled_sents, sampled_answers, sampled_text_answers):
                question_template = random.choice(LONG_QUESTION_LIST)
                questions.append(question_template.format(sent=text))

                for answer in answer_list:
                    rle = answer["segmentation"]
                    m = mask_utils.decode(rle)
                    if len(m.shape) > 2:
                        # m = np.sum(m, axis=2)
                        print_this_case = True
                    if len(m.shape) == 2:
                        m = m[np.newaxis, :, :]
                    m = m.astype(np.uint8)
                    masks.append(m)

                if len(text_answer) != 0:
                    if text_answer.count('{seg}') != len(answer_list):
                        skip_this_case = True
                        break

                    _text_answer = text_answer.format(seg=seg_token)
                    answers.append(_text_answer)

        if skip_this_case:
            continue

        if print_this_case:
            masks = np.concatenate(masks, axis=0)
            tags = []
            for mask_id, _masks_ in enumerate(masks):
                if len(_masks_) == 1:
                    tags.append(f"{mask_id}")
                else:
                    for _mask_id_ in range(len(_masks_)):
                        tags.append(f"{mask_id}_{_mask_id_}")
            output_image = visualize(image, masks, tags)
            output_image(f"mmr_multi_parts_{count}.jpg")
            count += 1
            continue
        else:
            continue
        masks = np.stack(sampled_masks, axis=0)

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
        try:
            boxes = torchvision.ops.masks_to_boxes(masks)
        except:
            print("Error at boxes = torchvision.ops.masks_to_boxes(masks)")
            continue

        whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
        boxes = boxes / whwh
        boxes = boxes.to(vq_sam2.device)
        masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
        num_ins = len(masks)
        
        with torch.no_grad():
            vq_sam2_output = vq_sam2(
                sam2_pixel_values.repeat(num_ins, 1, 1, 1),
                masks,
                boxes,
                reconstruct_mask=False,
            )
            quant_codes = vq_sam2_output.quant_codes
            # pred_masks = vq_sam2_output.pred_masks
            # pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
            # pred_masks = pred_masks > 0.5
            # pred_masks = pred_masks[:, 0].cpu().numpy().astype(np.uint8)
        
        # output_image = visualize(image, pred_masks, [""]*len(pred_masks))
        # output_image.save("./mmr_mask.jpg")
        # exit(0)
        
        quant_codes = quant_codes.cpu().numpy().astype(np.int32).tolist()
        remap_quant_codes = []
        for _quant_codes in quant_codes:
            _quant_codes = _quant_codes[0]
            remap_quant_codes.append([depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes)])
        quant_codes = remap_quant_codes

        mask_token_list = []
        for _quant_codes_ in quant_codes:
            sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes_]) + MT_END_TOKEN
            mask_token_list.append(sam2_tokens)
        
        seg_id = 0
        for question, answer in zip(questions, answers):
            token_count = answer.count('[SEG]')
            for _ in range(token_count):
                answer = answer.replace('[SEG]', mask_token_list[seg_id], 1)
                seg_id += 1
            
            conversations = []
            conversations.append({'from': 'human', 'value': question})
            conversations.append({'from': 'gpt', 'value': answer})

            ret_data_dict = {
                'image': image_path,
                'conversations': conversations,
            }

            shard_items.append(ret_data_dict)
            count += 1
            if count % shard_size == 0:
                shard_idx += 1
                out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}-chunk{task_id}.json")
                with open(out_path, "w") as f:
                    json.dump(shard_items, f)
                shard_items.clear()
                print(f"[SAVE] {out_path} ({count} items)", flush=True)
    
    # 收尾
    if shard_items:
        shard_idx += 1
        out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}-chunk{task_id}.json")
        with open(out_path, "w") as f:
            json.dump(shard_items, f)
        shard_items.clear()
        print(f"[SAVE] {out_path} (final, total={count})", flush=True) 

if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)
        