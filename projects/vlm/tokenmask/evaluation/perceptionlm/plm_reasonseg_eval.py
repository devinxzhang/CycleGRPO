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
import glob
import base64
import io
import hydra

from transformers import AutoModelForImageTextToText, AutoProcessor

from qwen_vl_utils import process_vision_info
from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config

from torchvision.transforms.functional import resize, to_pil_image
class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

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
    
    alpha = 0.8

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


def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--vq_sam2_path',
        default="pretrained_weights/iter_175473.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--dataset',
        default='./data/PaDT-MLLM/RefCOCO/refcoco_val.json',
        help='Specify a ref dataset')
    parser.add_argument('--task_id', '--task-id', type=int, default=0)
    args = parser.parse_args()
    return args

IMAGE_FOLDER = './data/glamm_data/images/coco2014/train2014/'

def extract_mt_token_ids_v1(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]

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

def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(mask_utils.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

def rle_to_mask(rle):
    mask = []
    for r in rle:
        m = mask_utils.decode(r)
        m = np.uint8(m)
        mask.append(m)
    mask = np.stack(mask, axis=0)
    return mask

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
    # masks = []
    for i in sort_inform:
        # mask = np.zeros((height, width), dtype=np.uint8)

        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)
        # masks.append((mask == 1).astype(np.uint8))

    return mask, comments, is_sentence
    # return masks, comments, is_sentence


def metric():
    from projects.vlm.qwen2_5_vl_vq_sam2.evaluation.utils import REFER, Summary, AverageMeter, intersectionAndUnionGPU, master_only
    
    trackers = {
        "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
        "union": AverageMeter("Union", ":6.3f", Summary.SUM),
        "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
    }
    for json_file in os.listdir('./temp_save/reasonseg'):
        with open(os.path.join('./temp_save/reasonseg', json_file), 'r') as f:
            data_dict = json.load(f)

        intersection, union, accuracy_iou = 0.0, 0.0, 0.0
        masks = data_dict['prediction_masks']
        if len(masks) > 1:
            print("pred: ", masks)
            print("gt: ",  data_dict['gt_masks'])
            exit(0)
        _masks = []
        for mask in masks:
            if mask is not None:
                mask = rle_to_mask([mask])
            _masks.append(mask)
        targets = data_dict['gt_masks']
        _targets = rle_to_mask(targets)

        for i_item, _mask in enumerate(_masks):
            if _mask is None:
                continue

            _target = _targets[i_item: i_item+1]
            for prediction, target in zip(_mask, _target):
                prediction = torch.from_numpy(prediction).int().cuda()
                target = torch.from_numpy(target).int().cuda()
                try:
                    intersect, union_, _ = intersectionAndUnionGPU(
                        prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                    )
                except:
                    print("pred.shape: ", prediction.shape)
                    print("target.shape: ", target.shape)
                    continue
                intersection += intersect
                union += union_
                accuracy_iou += intersect / (union_ + 1e-5)
                accuracy_iou[union_ == 0] += 1.0
        try:
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            accuracy_iou = accuracy_iou.cpu().numpy() / _targets.shape[0]
        except:
            accuracy_iou = accuracy_iou / _targets.shape[0]
        trackers["intersection"].update(intersection)
        trackers["union"].update(union)
        trackers["gIoU"].update(accuracy_iou, n=_targets.shape[0])

    cur_results = {'pixel_intersection': trackers["intersection"].sum[1],
                    'pixel_union': trackers["union"].sum[1],
                    'gIoU': trackers["gIoU"].avg[1],
                    'mask_counts': trackers["gIoU"].count,
                    }
    class_iou = cur_results['pixel_intersection'] / (cur_results['pixel_union'] + 1e-10)
    global_iou = cur_results['gIoU']

    print('============================================', 'current')
    print('CIoU: {}, GIoU: {}'.format(class_iou, global_iou), 'current')
    print('============================================', 'current')
    return {'Acc': class_iou}

import numpy as np
from scipy.optimize import linear_sum_assignment
def match_masks(gt_masks, pred_masks):
    """
    将基准真相掩码(gt_masks)与预测掩码(pred_masks)进行最优匹配。
    这个函数解决了指派问题，其中成本是基于掩码之间的交并比(IoU)计算的。
    它旨在找到一个匹配方案，使得所有配对的IoU之和最大化，等同于
    (1 - IoU)的总成本最小化。
    参数:
    gt_masks (np.ndarray): 形状为 (n, h, w) 的布尔或整数(0/1)数组，代表基准真相掩码。
    pred_masks (np.ndarray): 形状为 (m, h, w) 的布尔或整数(0/1)数组，代表预测掩码。
                               其中 m >= n。
    返回:
    list[tuple(np.ndarray, np.ndarray)]: 一个元组列表，每个元组包含一对匹配的
                                         (gt_mask, pred_mask)。
    """
    n, h, w = gt_masks.shape
    m = pred_masks.shape[0]
    # 确保输入是布尔类型以进行位运算，这样更高效
    gt_masks = gt_masks.astype(bool)
    pred_masks = pred_masks.astype(bool)
    # 初始化成本矩阵。成本定义为 1 - IoU
    # IoU = intersection / union
    # 我们希望最大化 IoU，等同于最小化 1 - IoU
    cost_matrix = np.zeros((n, m))
    # 逐对计算 IoU 并填充成本矩阵
    for i in range(n):
        for j in range(m):
            gt_mask = gt_masks[i]
            pred_mask = pred_masks[j]
            # 使用位运算高效计算交集和并集
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            # 计算 IoU，避免除以零的错误
            iou = intersection / union if union > 0 else 0
            # 成本是 1 - IoU
            cost_matrix[i, j] = 1 - iou
    # 使用匈牙利算法解决指派问题. [1, 8]
    # linear_sum_assignment 会找到成本总和最小的配对. [1]
    # 它返回两组索引：gt_indices 对应 gt_masks 的行，pred_indices 对应 pred_masks 的列
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
    # 创建匹配对的列表
    matched_pairs = []
    for i in range(len(gt_indices)):
        gt_idx = gt_indices[i]
        pred_idx = pred_indices[i]
        # 将原始的、未经类型转换的掩码添加到配对列表中
        matched_pairs.append((gt_masks[gt_idx], pred_masks[pred_idx]))
    return matched_pairs

def main():
    args = parse_args()

    # build plm model
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    processor.image_processor.max_num_tiles = 8
    model = AutoModelForImageTextToText.from_pretrained(args.model_path).to("cuda")

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


    images = glob.glob(
        os.path.join(
            args.dataset, "*.jpg"
        )
    )
    jsons = [path.replace(".jpg", ".json") for path in images]

    count = 0
    for image_path, json_path in zip(images, jsons):
        print(f"{count+1} / {len(images)}")
        # if count not in [12, 77, 102, 106, 110]:
        #     count += 1
        #     continue
        cv2_image = cv2.imread(image_path)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, cv2_image)
        # assert is_sentence

        text = sampled_sents[0].strip()
        if is_sentence:
            question = "{} Please output segmentation mask.".format(text)
        else:
            question = "Can you segment the {} in this image?".format(text.lower())

        batch_size = 1
        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        gt_mask = mask_json[np.newaxis, :, :] == 1
        gt_mask_backup = gt_mask
        gt_mask = mask_to_rle(gt_mask)

        # gt_mask_backup = np.stack(mask_json, axis=0)
        # output_image = visualize(image, gt_mask_backup, ['']*len(gt_mask_backup))
        # output_image.save(f'lisa_cases/reasonseg_{count}_gt.jpg')
        # count += 1
        # continue

        # resize long edge to 1024
        if ori_width > ori_height and ori_width > 1024:
            new_width = 1024
            new_height = int(ori_height / ori_width * 1024)
        elif ori_height > ori_width and ori_height > 1024:
            new_height = 1024
            new_width = int(ori_width / ori_height * 1024)
        elif ori_height == ori_width and ori_height > 1024:
            new_height = 1024
            new_width = 1024
        else:
            new_height = ori_height
            new_width = ori_width
        
        resized_image = image.resize((new_width, new_height), resample=Image.BICUBIC)
        buffer = io.BytesIO()
        resized_image.save(buffer, format='JPEG')
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{b64}",
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        generate_ids = model.generate(**inputs, max_new_tokens=256)
        input_length = inputs["input_ids"].shape[1]
        generate_ids_without_inputs = generate_ids[:, input_length:]
        output_text = processor.batch_decode(generate_ids_without_inputs, skip_special_tokens=True)
        print("Assistant: ", output_text)

        quant_ids = extract_mt_token_ids_v1(output_text[0])
        if len(quant_ids) == 0:
            zero_mask = np.zeros((1, ori_height, ori_width)).astype(np.uint8)
            zero_mask = mask_to_rle(zero_mask)
            prediction = {'image_id': count, 'gt_masks': gt_mask, 'prediction_masks': zero_mask}
            with open(f"./temp_save/reasonseg/{count}.json", 'w') as f:
                json.dump(prediction, f)
            count += 1
            continue
        
        if len(quant_ids) % CODEBOOK_DEPTH != 0:
            print("FORMAT ERROR: ", output_text)
            output_text = [fix_mt_format_comprehensive(output_text[0])]
            print("FIXED OUTPUT TEXT: ", output_text)
            quant_ids = extract_mt_token_ids_v2(output_text[0])
        if len(quant_ids) % CODEBOOK_DEPTH != 0:
            zero_mask = np.zeros((1, ori_height, ori_width)).astype(np.uint8)
            zero_mask = mask_to_rle(zero_mask)
            prediction = {'image_id': count, 'gt_masks': gt_mask, 'prediction_masks': zero_mask}
            with open(f"./temp_save/reasonseg/{count}.json", 'w') as f:
                json.dump(prediction, f)
            count += 1
            continue
        
        batch_size = len(quant_ids) // CODEBOOK_DEPTH
        remap_quant_ids = []
        for bs_id in range(batch_size):
            chunk_quant_ids = quant_ids[bs_id*CODEBOOK_DEPTH:(bs_id+1)*CODEBOOK_DEPTH]
            remap_chunk_quant_ids = [quant_id - book_id*CODEBOOK_SIZE for book_id, quant_id in enumerate(chunk_quant_ids)]
            code1 = remap_chunk_quant_ids[0]
            code2 = remap_chunk_quant_ids[1]
            if not (code1 >= 0 and code1 < CODEBOOK_SIZE):
                continue
            if not (code2 >= 0 and code2 < CODEBOOK_SIZE):
                code2 = -1
            remap_chunk_quant_ids_error_handle = [code1, code2]
            remap_quant_ids.append(remap_chunk_quant_ids_error_handle)

        batch_size = len(remap_quant_ids)
        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
        sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

        quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

        with torch.no_grad():
            _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
        _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
        _pred_masks = _pred_masks > 0.5
        _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
        _pred_masks = np.sum(_pred_masks, axis=0).astype(np.uint8)[np.newaxis, :, :]
        _pred_masks = (_pred_masks > 0).astype(np.uint8)

        # output_image = visualize(image, _pred_masks, ['']*len(_pred_masks))
        # output_image.save(f'lisa_cases/reasonseg_{count}_pred.jpg')

        _pred_masks = mask_to_rle(_pred_masks)
        prediction = {'image_id': count, 'gt_masks': gt_mask, 'prediction_masks': _pred_masks}
        with open(f"./temp_save/reasonseg/{count}.json", 'w') as f:
            json.dump(prediction, f)
        count += 1

        
if __name__ == '__main__':
    main()
    print(metric())