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

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from dataset import RESDataset
from xtuner.model.utils import guess_load_checkpoint

from qwen_vl_utils import process_vision_info
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


def load_dataset():
    with open('./data/mmr_data/MMR_val.json', 'r') as f:
        reason_seg_data = json.load(f)
    
    all_eval_items = []
    for idx in range(len(reason_seg_data)):
        image_info = reason_seg_data[idx]
        if "file_name" in image_info:
            image_path = os.path.join("./data/coco", image_info["file_name"])
        anns = image_info['annotations']
        sampled_sents = image_info['questions'] 
        gt_answers = image_info['answers']
        sampled_answers = image_info['text_answers']
        
        is_sentence = True

        i = 0
        while i < len(sampled_sents):
            text = sampled_sents[i].strip()
            _seg = sampled_answers[i].format(seg="[SEG]")

            question = "{} Please output segmentation mask.".format(text)
            answer = "{}.".format(_seg)

            gt_answer = gt_answers[i]

            # masks = []
            # for answer in gt_answer:
            #     rle = answer["segmentation"]
            #     m = mask_utils.decode(rle)
            #     if len(m.shape) > 2:
            #         m = np.sum(m, axis=2)
            #     m = m.astype(np.uint8)
            #     masks.append(m)
            
            eval_item = {
                'image': image_path,
                'question': question,
                'answer': answer,
                'anno': gt_answer,
            }
            all_eval_items.append(eval_item)

            i += 1
    
    with open("./data/mmr_data/MMR_val_sampled.json", 'w') as f:
        json.dump(all_eval_items, f)


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


def metric():
    from projects.vlm.qwen2_5_vl_vq_sam2.evaluation.utils import REFER, Summary, AverageMeter, intersectionAndUnionGPU, master_only
    
    trackers = {
        "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
        "union": AverageMeter("Union", ":6.3f", Summary.SUM),
        "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
    }
    for json_file in os.listdir('./temp_save/mmr'):
        with open(os.path.join('./temp_save/mmr', json_file), 'r') as f:
            data_dict = json.load(f)

        intersection, union, accuracy_iou = 0.0, 0.0, 0.0
        masks = data_dict['pred_masks']
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
                intersect, union_, _ = intersectionAndUnionGPU(
                    prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                )
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


def main():
    args = parse_args()

    # build qwen25vl model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    # build vq-sam2 model
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

    pretrained_state_dict = guess_load_checkpoint(args.vq_sam2_path)

    pretrained_state_dict_new = {}
    for key in pretrained_state_dict.keys():
        new_key = copy.deepcopy(key)
        if key.startswith('hf_model.'):
            new_key = new_key[len('hf_model.'):]
        pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
    
    vq_sam2.load_state_dict(pretrained_state_dict_new)

    sam2_image_processor = DirectResize(1024)


    # dataset
    all_data_dict = []
    case_id = 0
    with open(args.dataset, 'r') as f:
        json_data = json.load(f)
        for item in json_data:
            item.update({'case_id': case_id})
            all_data_dict.append(item)
            case_id += 1

    rows = len(all_data_dict)
    chunk_size = (rows+7) // 8
    _start_ = args.task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = rows if _end_ > rows else _end_

    for data_dict in tqdm.tqdm(all_data_dict[_start_:_end_]):
        image_path = data_dict['image']
        question = data_dict['question']
        mask_annos = data_dict['anno']
        case_id = data_dict['case_id']

        gt_masks = []
        for answer in mask_annos:
            rle = answer["segmentation"]
            m = mask_utils.decode(rle)
            if len(m.shape) > 2:
                m = np.sum(m, axis=2)
            m = m.astype(np.uint8)
            gt_masks.append(m)
        gt_masks = np.stack(gt_masks, axis=0)
        gt_rles = mask_to_rle(gt_masks)
        num_gt_masks = len(gt_masks)
        
        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512,
            do_sample=False,  # 关闭采样，使用贪婪解码
            top_p=1.0,  # 配合do_sample=False使用
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        print("Assistant: ", output_text)

        quant_ids = extract_mt_token_ids(output_text[0])
        if len(quant_ids) == 0:
            zero_mask = np.zeros((num_gt_masks, ori_height, ori_width)).astype(np.uint8)
            zero_mask = mask_to_rle(zero_mask)
            prediction = {'gt_masks': gt_rles, 'pred_masks': zero_mask}
            with open(f"./temp_save/mmr/{case_id}.json", 'w') as f:
                json.dump(prediction, f)
            continue

        if len(quant_ids) % CODEBOOK_DEPTH != 0:
            print("FORMAT ERROR: ", output_text)
            output_text = [fix_mt_format_comprehensive(output_text[0])]
            print("FIXED OUTPUT TEXT: ", output_text)
            quant_ids = extract_mt_token_ids(output_text[0])
        assert len(quant_ids) % CODEBOOK_DEPTH == 0
        batch_size = len(quant_ids) // CODEBOOK_DEPTH
        remap_quant_ids = []
        for bs_id in range(batch_size):
            chunk_quant_ids = quant_ids[bs_id*CODEBOOK_DEPTH:(bs_id+1)*CODEBOOK_DEPTH]
            remap_chunk_quant_ids = [quant_id - book_id*CODEBOOK_SIZE for book_id, quant_id in enumerate(chunk_quant_ids)]
            remap_chunk_quant_ids_error_handle = [quant_id if quant_id < CODEBOOK_SIZE else -1 for quant_id in remap_chunk_quant_ids]
            remap_quant_ids.append(remap_chunk_quant_ids_error_handle)

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size
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

        if len(gt_masks) == 1 and len(_pred_masks) == 1:
            _pred_masks = mask_to_rle(_pred_masks)
            prediction = {'gt_masks': gt_rles, 'pred_masks': _pred_masks}
            with open(f"./temp_save/mmr/{case_id}.json", 'w') as f:
                json.dump(prediction, f)
            continue

        if len(_pred_masks) < len(gt_masks):
            zero_mask = np.zeros((len(gt_masks) - len(_pred_masks), ori_height, ori_width)).astype(np.uint8)
            _pred_masks = np.concatenate((_pred_masks, zero_mask), axis=0)
        
        # _pred_masks = _pred_masks[:len(gt_masks)]
        # _pred_masks = mask_to_rle(_pred_masks)
        # prediction = {'gt_masks': gt_rles, 'pred_masks': _pred_masks}
        # with open(f"./temp_save/mmr/{case_id}.json", 'w') as f:
        #     json.dump(prediction, f)

        macthed_pairs = match_masks(gt_masks, _pred_masks)

        final_gt_masks = [gt for (gt, pred) in macthed_pairs]
        final_pred_masks = [pred for (gt, pred) in macthed_pairs]

        final_gt_masks = mask_to_rle(final_gt_masks)
        final_pred_masks = mask_to_rle(final_pred_masks)
        prediction = {'gt_masks': final_gt_masks, 'pred_masks': final_pred_masks}
        with open(f"./temp_save/mmr/{case_id}.json", 'w') as f:
            json.dump(prediction, f)
        

if __name__ == "__main__":
    # load_dataset()
    main()
    # metric()




        






