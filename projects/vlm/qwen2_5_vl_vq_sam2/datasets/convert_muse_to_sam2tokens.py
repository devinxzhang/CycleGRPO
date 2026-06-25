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
    
    alpha = 0.3

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



LONG_QUESTION_LIST = [
    "<image>\n" + "{sent} Please respond with segmentation mask.",
    "<image>\n" + "{sent} Please output segmentation mask.",
    "<image>\n" + "{sent} Provide the segmentation mask.",
    "<image>\n" + "{sent} Output the segmentation mask.",
    "<image>\n" + "{sent} Please show the segmentation mask.",
    "<image>\n" + "{sent} I'd appreciate segmentation masks.",
    "<image>\n" + "{sent} Please highlight the segmentation mask.",
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
    
    dataset_name = "muse"
    temp_save_root = "./temp_data_256x2_0927/muse/"
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

    with open('./data/MUSE/MUSE_train.json', 'r') as f:
        all_data_dict = json.load(f)

    print("===========>TOTAL_ITEMS: ", len(all_data_dict))

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    num_classes_per_sample = 3
    seg_token_num = 1

    rows = len(all_data_dict)
    chunk_size = (rows+7) // 8
    _start_ = task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = rows if _end_ > rows else _end_

    for data_dict in tqdm.tqdm(all_data_dict[_start_:_end_]):
        if 'file_name' in data_dict:
            image_root = './data/coco/train2014'
            image_path = os.path.join(image_root, data_dict['file_name'])
        else:
            if 'train2017' in data_dict['coco_url']:
                image_root = './data/coco/train2017'
                image_path = os.path.join(image_root, data_dict['coco_url'].split('/')[-1])
            else:
                image_root = './data/coco/val2017'
                image_path = os.path.join(image_root, data_dict['coco_url'].split('/')[-1])
        
        anns = data_dict['ann_list']
        question = data_dict['questions'] if 'questions' in data_dict else None
        gt_answer = data_dict['answers'] if 'answers' in data_dict else None
        if question is not None:
            text_answers = data_dict['text_answers'] if 'text_answers' in data_dict else [None] * len(gt_answer)
        else:
            text_answers = None
        
        if len(anns) == 0:
            continue

        category_ids = [ann['category_id'] for ann in anns]
        category_ids = list(set(category_ids))
        sampled_num = min(num_classes_per_sample, len(category_ids))
        sampled_category_ids = np.random.choice(category_ids, size=sampled_num, replace=False)

        masks = []
        sampled_sents = question
        sampled_answers = gt_answer
        sampled_masks = masks
        sample_text_answers = text_answers

        image_name = image_path.split("/")[-1]
        questions = []
        answers = []
        use_assign_list = []
        seg_token = ["[SEG{}]".format(i) for i in range(seg_token_num)]
        seg_token = ' '.join(seg_token)

        skip_this_case = False

        if question is not None:
            for text, answer_list, text_answer in zip(sampled_sents, sampled_answers, sample_text_answers):
                # if is_sentence:
                question_template = random.choice(LONG_QUESTION_LIST)
                questions.append(question_template.format(sent=text))
                
                for answer in answer_list:
                    rle = mask_utils.frPyObjects(answer["segmentation"], data_dict["height"], data_dict["width"])
                    m = mask_utils.decode(rle)
                    if len(m.shape) > 2:
                        # assert m.shape[-1] == 1, m.shape
                        m = np.sum(m, axis=2)  # so
                    m = m.astype(np.uint8)
                    masks.append(m)

        
                use_assign = False
                if text_answer is not None:
                    if text_answer.count('{seg}') != len(answer_list):
                        skip_this_case = True
                        break
                    try:
                        _text_answer = text_answer.format(seg='[SEG]') if seg_token_num == 1 else text_answer.format(seg=seg_token)
                    except:
                        skip_this_case = True
                        break
                    answers.append(_text_answer)
                    use_assign_list.append(False)
                else:
                    target_list = [a['rephrased_name'] if (random.random() > 0.1 and 'rephrased_name' in a) else a['category_name'] for a in answer_list ]
                    target_answer = []
                    separate_answer = random.randint(0, 1)
                    _seg = ['[SEG]'] * len(target_list)
                    if len(target_list) > 1:
                        part1 = ', '.join(_seg[:-1])
                        part2 = ' and ' + _seg[-1]
                        _seg = part1 + part2 
                    else:
                        _seg = _seg[0]
                    
                    if separate_answer:
                        choice_list = MR_SINGLE_ANSWER_LIST
                        answer_temp = random.choice(choice_list) if seg_token_num == 1 else random.choice(choice_list).replace('[SEG]', seg_token)
                        use_assign = False if "{class_name}" in answer_temp else True
                        for i, sampled_cls in enumerate(target_list):
                            _answer_temp = answer_temp.format(class_name=sampled_cls) if "{class_name}" in answer_temp else answer_temp
                            target_answer.append(_answer_temp[:-1])
                        if len(target_answer) > 1:
                            part1 = ', '.join(target_answer[:-1])
                            part2 = ' and ' + target_answer[-1]
                            target_answer = part1 + part2 + '.'
                        else:
                            target_answer = target_answer[0] + '.'
                    else:
                        answer_temp = random.choice(MR_MULTI_ANSWER_LIST)
                        _answer_temp = answer_temp.format(class_name=', '.join(target_list).lower(), seg=_seg) if "{class_name}" in answer_temp else answer_temp.format(seg=_seg)
                        use_assign = False if "{class_name}" in answer_temp else True
                        _answer_temp = _answer_temp if seg_token_num == 1 else _answer_temp.replace('[SEG]', seg_token)
                        target_answer = _answer_temp

                    answers.append(target_answer)
                    use_assign_list.append(use_assign)
            
        else:
            skip_this_case =True
            print('question is None')
            exit(0)
        
        if skip_this_case:
            continue

        # masks = np.stack(sampled_masks[:3], axis=0)

        # image = Image.open(image_path).convert('RGB')
        # ori_width, ori_height = image.size

        # output_image = visualize(image, masks, [f"{idx+1}" for idx in range(len(sampled_masks[:3]))])
        # output_image.save('./muse.jpg')
        
        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        if len(sampled_masks) == 0:
            print('len(sampled_masks) == 0')
            continue
        masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in sampled_masks])
        try:
            boxes = torchvision.ops.masks_to_boxes(masks)
        except:
            print("contain empty mask.")
            continue
        whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
        boxes = boxes / whwh
        boxes = boxes.to(vq_sam2.device)
        masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
        num_ins = len(masks)

        with torch.no_grad():
            try:
                vq_sam2_output = vq_sam2(
                    sam2_pixel_values.repeat(num_ins, 1, 1, 1),
                    masks,
                    boxes,
                    reconstruct_mask=False,
                )
                quant_codes = vq_sam2_output.quant_codes
            except:
                print("vq_sam2 error!!")
                continue
        quant_codes = quant_codes.cpu().numpy().astype(np.int32).tolist()
        remap_quant_codes = []
        for _quant_codes in quant_codes:
            _quant_codes = _quant_codes[0]
            remap_quant_codes.append([depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes)])
        quant_codes = remap_quant_codes

        sam2_tokens_list = []
        for _quant_codes_ in quant_codes:
            sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes_]) + MT_END_TOKEN
            sam2_tokens_list.append(sam2_tokens)
        
        i = 0
        seg_i = 0
        while i < len(questions):
            conversations = []
            conversations.append({'from': 'human', 'value': questions[i]})
            seg_answer = answers[i]
            token_count = seg_answer.count('[SEG]')
            for _ in range(token_count):
                seg_answer = seg_answer.replace('[SEG]', sam2_tokens_list[seg_i], 1)
                seg_i += 1

            conversations.append({'from': 'gpt', 'value': seg_answer})
            i += 1

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
        
        assert seg_i == len(sam2_tokens_list)

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




