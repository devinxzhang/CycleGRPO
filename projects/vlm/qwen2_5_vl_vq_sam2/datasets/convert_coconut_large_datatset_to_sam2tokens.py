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
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import io

import mmengine
from mmengine.dataset import BaseDataset

from xtuner.model.utils import guess_load_checkpoint

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize
from projects.vlm.qwen2_5_vl_vq_sam2.datasets.coconut_meta import COCO_META


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

def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union


def sort_mask_indices(masks_t: torch.Tensor, mode: str = "ltr-ttb") -> np.ndarray:
    """
    根据实例的几何位置给出排序索引。
    Args:
        masks_t: [N, H, W] 的 torch.bool/uint8 张量（每个实例一个二值mask）
        mode:
            - "ltr-ttb": left-to-right, then top-to-bottom（先按x中心，再按y中心）
            - "ttb-ltr": top-to-bottom, then left-to-right（先按y中心，再按x中心）
            - "tlbr":    purely by top-left (y1, x1) 先y后x（行优先）
    Returns:
        order: numpy 数组，形状 [N]，是重排索引
    """
    # 利用bbox/中心点作为排序依据（更稳定、无需遍历像素）
    boxes = torchvision.ops.masks_to_boxes(masks_t)  # [N,4] (x1,y1,x2,y2)
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    xc = ((x1 + x2) * 0.5).cpu().numpy()
    yc = ((y1 + y2) * 0.5).cpu().numpy()
    y1n = y1.cpu().numpy()
    x1n = x1.cpu().numpy()

    if mode == "ltr-ttb":
        # 先x后y：主键x_center，次键y_center
        order = np.lexsort((yc, xc))
    elif mode == "ttb-ltr":
        # 先y后x：主键y_center，次键x_center
        order = np.lexsort((xc, yc))
    elif mode == "tlbr":
        # 以bbox左上角先y后x（更像“逐行”）
        order = np.lexsort((x1n, y1n))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return order


# QUESTION_LIST = [
#     "Can you segment the \"{class_name}\" in this image?",
#     "Please segment the \"{class_name}\" in this image.",
#     "What is \"{class_name}\" in this image? Please respond with segmentation mask.",
#     "What is \"{class_name}\" in this image? Please output segmentation mask.",
# ]

# ANSWER_LIST = [
#     "Sure, {SEG}.",
#     "{SEG}.",
# ]

QUESTION_LIST = [
    "<image>\nSegment every instance that belongs to the following categories: {class_name}",
    "<image>\nLocate every instance that belongs to the following categories: {class_name}. Report segmentation masks in JSON format."
]

def clear_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    import gc
    gc.collect()


def main(task_id):

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    temp_save_root = "./temp_data_256x2_0927/coconut/"
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

    image_root = "./data/object365/"
    pano_image_root = "<PATH_TO_DATA>/segmentation_datasets/coconut/xdeng77/coconut_large/panoptic_object365/"
    anno_file = "<PATH_TO_DATA>/segmentation_datasets/coconut/xdeng77/coconut_large/panseg_object365_train_v2.json"
    with open(anno_file, 'r') as f:
        anno_data = json.load(f)
    
    categories = anno_data['categories']
    anno_info_list = anno_data['annotations']

    coco_id_to_name = {meta['id']: meta['name'] for meta in categories}
    category_isthing = {meta['name']: meta['isthing'] for meta in categories}

    chunk_size = (len(anno_info_list)+31) // 32
    _start_ = task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = len(anno_info_list) if _end_ > len(anno_info_list) else _end_


    for anno_info in tqdm.tqdm(anno_info_list[_start_:_end_]):
        segments_info = anno_info['segments_info']
        if 'object365_file_name' in anno_info:
            image_id = anno_info['object365_file_name']
        elif 'file_name' in anno_info:
            image_id = anno_info['file_name']
        else:
            ValueError(f"image_id not found in anno_info: {anno_info}")

        if '.png' in image_id:
            image_id = image_id.split('.png')[0]
        if '.jpg' in image_id:
            image_id = image_id.split('.jpg')[0]

        if os.path.exists(os.path.join(temp_save_root, f"{image_id}.json")):
            print("file exists...................")
            continue

        # if os.path.exists(os.path.join(temp_save_root, f"{image_id}.json")):
        #     continue

        patch_list = ['patch17', 'patch23', 'patch25', 'patch28', 'patch32', 'patch35', 'patch38', 'patch40', 'patch42', 'patch44', 'patch50']
        image_path = None
        for patch_name in patch_list:
            if os.path.exists(os.path.join(image_root, patch_name, f"{image_id}.jpg")):
                image_path = os.path.join(image_root, patch_name, f"{image_id}.jpg")
                break
        if image_path is None:
            continue
        
        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size
        
        pano_file = os.path.join(pano_image_root, f"{image_id}.png")
        mask_image = Image.open(pano_file)
        mask_image_np = np.array(mask_image)[:, :, 0]

        categories_name_to_masks = {}
        for segment_info in segments_info:
            category_id = segment_info['category_id']
            if category_id == 0:
                continue
            isthing = segment_info['isthing']
            mask = mask_image_np == segment_info['id']
            if coco_id_to_name[category_id] not in categories_name_to_masks:
                categories_name_to_masks[coco_id_to_name[category_id]] = []
            categories_name_to_masks[coco_id_to_name[category_id]].append(mask)

        turn_idx = 0
        conversation = []
        answer = "```json\n[{mask_2d}]\n```"
        mask_2d_str = ''
        class_names = []
        for category_name, category_masks in categories_name_to_masks.items():
            sam2_image = np.array(image)
            sam2_image = sam2_image_processor.apply_image(sam2_image)
            sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
            sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

            masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in category_masks])
            valid_masks = masks.sum(-1).sum(-1) > 0
            masks = masks[valid_masks]

            if len(masks) == 0:
                print("len(masks) == 0!!!")
                continue

            try:
                order = sort_mask_indices(masks, mode="ltr-ttb")
            except:
                order = np.arange(masks.shape[0])
            
            masks = masks[torch.as_tensor(order, dtype=torch.long)]
        
            try:
                boxes = torchvision.ops.masks_to_boxes(masks)
            except Exception as e:
                continue
            
            whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
            boxes = boxes / whwh
            boxes = boxes.to(vq_sam2.device)
            masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
            num_ins = len(masks)

            skip_this_one = False
            try:
                with torch.no_grad():
                    vq_sam2_output = vq_sam2(
                        sam2_pixel_values.repeat(num_ins, 1, 1, 1),
                        masks,
                        boxes,
                        reconstruct_mask=False,
                    )
                    quant_codes = vq_sam2_output.quant_codes
                    # pred_masks = vq_sam2_output.pred_masks
            except torch.OutOfMemoryError:
                print("num_ins is too large: ", num_ins, "; will be split into blocks (size 10)")
                NUM_BLOCKS = num_ins // 10
                if NUM_BLOCKS * 10 < num_ins:
                    NUM_BLOCKS += 1
                block_quant_codes = []
                # block_pred_masks = []
                for block_idx in range(NUM_BLOCKS):
                    start_idx = block_idx * 10
                    end_idx = min(start_idx + 10, num_ins)
                    try:
                        with torch.no_grad():
                            vq_sam2_output = vq_sam2(
                                sam2_pixel_values[start_idx:end_idx],
                                masks[start_idx:end_idx],
                                boxes[start_idx:end_idx],
                                reconstruct_mask=False,
                            )
                    except torch.OutOfMemoryError:
                        skip_this_one = True
                        break
                    block_quant_codes.append(vq_sam2_output.quant_codes)
                    # block_pred_masks.append(vq_sam2_output.pred_masks)
                if skip_this_one:
                    continue
                quant_codes = torch.cat(block_quant_codes, dim=0)
                # pred_masks = torch.cat(block_pred_masks, dim=0)

                # print("num_ins is too large: ", num_ins)
                # exit(0)
            except Exception as e:
                # print("sam2_pixel_values.repeat(num_ins, 1, 1, 1).shape: ", sam2_pixel_values.repeat(num_ins, 1, 1, 1).shape)
                # import uuid
                # save_obj = {
                #     'image_file': image_path,
                #     'category_name': category_name,
                #     "height": ori_height,
                #     "width": ori_width,
                # }

                # rles = []
                # for m in masks:
                #     m_np = m.detach().to("cpu").squeeze(0).numpy().astype(np.uint8)
                #     rle = mask_utils.encode(np.asfortranarray(m_np))
                #     if isinstance(rle["counts"], bytes):
                #         rle["counts"] = rle['counts'].decode("ascii")
                #     rles.append(rle)
                # save_obj['segmentation'] = rles

                # random_tag = uuid.uuid4().hex[:8]
                # _save_path = os.path.join("./temp_data/too_many_obj_cases", f"{image_id}_{random_tag}.json")
                # with open(_save_path, "w") as f:
                #     json.dump(save_obj, f)
                # print(f"[fallback saved] {_save_path}")
                continue
            
            if len(quant_codes) == 0:
                continue

            quant_codes = quant_codes.cpu().numpy().astype(np.int32).tolist()
            remap_quant_codes = []
            for _quant_codes in quant_codes:
                _quant_codes = _quant_codes[0]
                remap_quant_codes.append([depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes)])
            quant_codes = remap_quant_codes

            # verify the quality of the quant_codes
            # pred_masks = vq_sam2_output.pred_masks
            # try:
            #     pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height//4, ori_width//4), mode='bilinear')
            #     pred_masks = pred_masks > 0.5
            # except torch.OutOfMemoryError:
            #     NUM_BLOCKS = pred_masks.shape[0] // 10
            #     if NUM_BLOCKS * 10 < pred_masks.shape[0]:
            #         NUM_BLOCKS += 1
            #     resized_pred_masks = []
            #     for block_idx in range(NUM_BLOCKS):
            #         start_idx = block_idx * 10
            #         end_idx = start_idx + 10
            #         end_idx = pred_masks.shape[0] if end_idx > pred_masks.shape[0] else end_idx
            #         chunk_pred_masks = pred_masks[start_idx:end_idx]
            #         chunk_pred_masks = torch.nn.functional.interpolate(chunk_pred_masks, size=(ori_height//4, ori_width//4), mode='bilinear')
            #         chunk_pred_masks = chunk_pred_masks > 0.5
            #         resized_pred_masks.append(chunk_pred_masks)
            #     pred_masks = torch.cat(resized_pred_masks)
            # masks = torch.stack(masks, dim=0).to(torch.float16)
            # try:
            #     if min(masks.shape[-2], masks.shape[-1]) > 2048:
            #         print("masks toooooooooooo large: ", masks.shape)
            #         continue
            #     masks = torch.nn.functional.interpolate(masks, size=(ori_height//4, ori_width//4), mode='bilinear')
            #     masks = masks > 0.5
            # except torch.OutOfMemoryError:
            #     NUM_BLOCKS = masks.shape[0] // 10
            #     if NUM_BLOCKS * 10 < masks.shape[0]:
            #         NUM_BLOCKS += 1
            #     resized_masks = []
            #     for block_idx in range(NUM_BLOCKS):
            #         start_idx = block_idx * 10
            #         end_idx = start_idx + 10
            #         end_idx = masks.shape[0] if end_idx > masks.shape[0] else end_idx
            #         chunk_masks = masks[start_idx:end_idx]
            #         chunk_masks = torch.nn.functional.interpolate(chunk_masks, size=(ori_height//4, ori_width//4), mode='bilinear')
            #         chunk_masks = chunk_masks > 0.5
            #         resized_masks.append(chunk_masks)
            #     masks = torch.cat(resized_masks)

            # skip_this_one = False
            # for pred_mask, target_mask in zip(pred_masks, masks):
            #     iou = mask_iou(pred_mask, target_mask)
            #     if iou[0][0].item() < 0.5:
            #         skip_this_one = True
            #         break
            
            # pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
            # pred_masks =  pred_masks > 0.5
            # if skip_this_one:
            #     print("skip this one==========================")
            #     continue
            # else:
            #     output_image = visualize(image, pred_masks.cpu().numpy()[:,0], [""]*len(pred_masks))
            #     output_image.save(f"pred_mask_{image_id}.jpg")
            #     output_image = visualize(image, torch.cat(masks).cpu().numpy(), [""]*len(masks))
            #     output_image.save(f"target_mask_{image_id}.jpg")
            #     image.save(f"source_image_{image_id}.jpg")

            # question = random.choice(QUESTION_LIST).format(class_name=category_name)
            # if turn_idx == 0:
            #     question = "<image>\n" + question

            # sam2_tokens_list = []
            # for _quant_codes in quant_codes:
            #     sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes]) + MT_END_TOKEN
            #     sam2_tokens_list.append(sam2_tokens)
            # sam2_tokens_str = ', '.join(sam2_tokens_list)
            
            # answer = random.choice(ANSWER_LIST).format(SEG=sam2_tokens_str)
            # conversation.append({'from': 'human', 'value': question})
            # conversation.append({'from': 'gpt', 'value': answer})
            # turn_idx += 1

            for _quant_codes_ in quant_codes:
                sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes_]) + MT_END_TOKEN
                item_str = "{\"mask_2d\": " + sam2_tokens + ", \"label\": \"" + category_name + "\"}"
                mask_2d_str += item_str + ",\n"
                if category_name not in class_names:
                    class_names.append(category_name)
        
        # if len(conversation) == 0:
        #     continue

        if mask_2d_str == '':
            continue

        mask_2d_str = mask_2d_str[:-len(",\n")]
        answer = answer.format(mask_2d=mask_2d_str)

        category_name_str = ', '.join(class_names)
        question = random.choice(QUESTION_LIST).format(class_name=category_name_str)

        conversation.append({'from': 'human', 'value': question})
        conversation.append({'from': 'gpt', 'value': answer})
        
        ret_data_dict = {
            'image': image_path,
            'conversations': conversation,
        }
        
        with open(os.path.join(temp_save_root, f"{image_id}.json"), 'w') as f:
            json.dump(ret_data_dict, f)


if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)