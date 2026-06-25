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
    
    alpha = 0.0

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
    
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    temp_save_root = "./temp_data_256x2_0927/coconut/"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    dataset_name = "coconut"

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

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    sample_files = [
        "<PATH_TO_DATA>/segmentation_datasets/coconut/xdeng77/coconut_b/data/train-00000-of-00004.parquet",
        "<PATH_TO_DATA>/segmentation_datasets/coconut/xdeng77/coconut_b/data/train-00001-of-00004.parquet",
        "<PATH_TO_DATA>/segmentation_datasets/coconut/xdeng77/coconut_b/data/train-00002-of-00004.parquet",
        "<PATH_TO_DATA>/segmentation_datasets/coconut/xdeng77/coconut_b/data/train-00003-of-00004.parquet",
    ]

    coco_id_to_name = {meta['id']: meta['name'] for meta in COCO_META}
    category_isthing = {meta['name']: meta['isthing'] for meta in COCO_META}

    # index = 0
    for sample_file in sample_files:
        parquet_file = pq.ParquetFile(sample_file)
        data = parquet_file.read().to_pandas()
        rows = data.shape[0]

        chunk_size = (rows+31) // 32
        _start_ = task_id * chunk_size
        _end_ = _start_ + chunk_size
        _end_ = rows if _end_ > rows else _end_

        subset = data.iloc[_start_:_end_]
        subset_rows = subset.shape[0]

        row_idx = 0
        for row in subset.itertuples():
            print(f"===========================Processing row {row_idx+1} of {subset_rows}===========================")
            # if row_idx+1 in [1223, 2197, 2630, 6061, 2373, 9453, 1100, 10757, 10384, 5729, 12662, 12901, 6362, 637, 13642, 7596]:
            #     row_idx += 1
            #     continue
            row_idx += 1
            masks = getattr(row, 'mask')
            segments_info = getattr(row, 'segments_info')
            image_info = getattr(row, 'image_info')

            image_file = image_info['file_name']
            if '.jpg' in image_file:
                image_id = image_file.split('.jpg')[0]
            elif '.png' in image_file:
                image_id = image_file.split('.png')[0]
            else:
                raise ValueError(f"Unsupported image format: {image_file}")
            
            if os.path.exists(os.path.join(temp_save_root, f"{image_id}.json")):
                print("file exists.................")
                continue
            
            image_path = os.path.join("./data/coco/train2017", image_file)
            if not os.path.exists(image_path):
                image_path = os.path.join("./data/coco/unlabeled2017", image_file)
                if not os.path.exists(image_path):
                    print(image_path, "is not found!!!")
                    continue
            image = Image.open(image_path).convert('RGB')
            ori_width, ori_height = image.size

            mask_image = Image.open(io.BytesIO(masks['bytes']))
            mask_image_np = np.array(mask_image)[:, :, 0]

            categories_name_to_masks = {}
            for segment_info in segments_info['segments_info']:
                category_id = segment_info['category_id']
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
                except:
                    print("Error at boxes = torchvision.ops.masks_to_boxes(masks)")
                    continue

                whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
                boxes = boxes / whwh
                boxes = boxes.to(vq_sam2.device)
                masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
                num_ins = len(masks)
                # if num_ins > 10:
                #     print("===================num_ins is too large: ", num_ins)
                #     output_image = visualize(image, category_masks, [category_name]*num_ins)
                #     output_image.save(f'test_coconut_to_much_objects_{image_id}.jpg')
                #     exit(0)
                # if num_ins > 10:
                #     continue

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
                            # print("num_ins is too large: ", end_idx-start_idx, "; will be split into blocks (size 1)")
                            # sub_block_quant_codes = []
                            # sub_block_pred_masks = []
                            # for sub_block_idx in range(end_idx-start_idx):
                            #     try:
                            #         vq_sam2_output = vq_sam2(
                            #             sam2_pixel_values[start_idx+sub_block_idx:start_idx+sub_block_idx+1],
                            #             masks[start_idx+sub_block_idx:start_idx+sub_block_idx+1],
                            #             boxes[start_idx+sub_block_idx:start_idx+sub_block_idx+1],
                            #         )
                            #         sub_block_quant_codes.append(vq_sam2_output.quant_codes)
                            #         sub_block_pred_masks.append(vq_sam2_output.pred_masks)
                            #     except torch.OutOfMemoryError:
                            #         skip_this_one = True
                            #         break
                            # if skip_this_one:
                            #     break
                            # block_quant_codes.append(torch.cat(sub_block_quant_codes, dim=0))
                            # block_pred_masks.append(torch.cat(sub_block_pred_masks, dim=0))
                            skip_this_one = True
                            break
                        block_quant_codes.append(vq_sam2_output.quant_codes)
                        # block_pred_masks.append(vq_sam2_output.pred_masks)
                    if skip_this_one:
                        print("skip this one!!!")
                        continue
                    quant_codes = torch.cat(block_quant_codes, dim=0)
                    # pred_masks = torch.cat(block_pred_masks, dim=0)

                    # print("num_ins is too large: ", num_ins)
                    # exit(0)
                except Exception as e:
                    print("sam2_pixel_values.repeat(num_ins, 1, 1, 1).shape: ", sam2_pixel_values.repeat(num_ins, 1, 1, 1).shape)
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
                # pred_masks = vq_sam2_output.pred_masks
                # pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
                # pred_masks = pred_masks > 0.5
                # pred_masks = pred_masks[0].cpu().numpy().astype(np.uint8)
                # target_mask = masks[0].cpu().numpy().astype(np.uint8)
                
                # skip_this_one = False
                # for pred_mask, target_mask in zip(pred_masks, masks):
                #     iou = mask_iou(pred_mask, target_mask)
                #     if iou[0][0].item() < 0.5:
                #         skip_this_one = True
                #         break
                
                if skip_this_one:
                    # output_image = visualize(image, pred_masks[:, 0].cpu().numpy(), [""]*len(pred_masks))
                    # output_image.save(f"pred_mask_{row_idx}.jpg")
                    # output_image = visualize(image, torch.cat(masks).cpu().numpy(), [""]*len(masks))
                    # output_image.save(f"target_mask_{row_idx}.jpg")
                    # image.save(f"source_image_{row_idx}.jpg")
                    print("skip this one")
                    # exit(0)
                    continue
                # else:
                #     output_image = visualize(image, pred_masks[:, 0].cpu().numpy(), [""]*len(pred_masks))
                #     output_image.save(f"pred_mask_{row_idx}.jpg")
                #     output_image = visualize(image, torch.cat(masks).cpu().numpy(), [""]*len(masks))
                #     output_image.save(f"target_mask_{row_idx}.jpg")
                #     # image.save(f"source_image_{row_idx}.jpg")


                # question = random.choice(QUESTION_LIST).format(class_name=category_name)
                # if turn_idx == 0:
                #     question = "<image>\n" + question

                # answer = "```json\n[{mask_2d}]\n```"
                # mask_2d_str = ''
                # for _quant_codes_ in quant_codes:
                #     sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes_]) + MT_END_TOKEN
                #     item_str = "{\"mask_2d\": " + sam2_tokens + ", \"label\": \"" + category_name + "\"}"
                #     mask_2d_str += item_str + ",\n"
                # mask_2d_str = mask_2d_str[:-len(",\n")]
                # answer = answer.format(mask_2d=mask_2d_str)

                # conversation.append({'from': 'human', 'value': question})
                # conversation.append({'from': 'gpt', 'value': answer})
                # turn_idx += 1

                # for binary_mask in binary_masks:
                #     rle = mask_utils.encode(np.array(binary_mask[:, :, None], order="F", dtype="uint8"))[0]
                #     rle["counts"] = rle["counts"].decode("utf-8")
                #     rles.append(rle)

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

            # print(ret_data_dict)
            # exit(0)

            with open(os.path.join(temp_save_root, f"{image_id}.json"), 'w') as f:
                json.dump(ret_data_dict, f)

            clear_gpu_memory()

            # shard_items.append(ret_data_dict)
            # count += 1

            # if count % shard_size == 0:
            #     shard_idx += 1
            #     out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-chunk{task_id}-{shard_idx:05d}.json")
            #     with open(out_path, "w") as f:
            #         json.dump(shard_items, f)
            #     shard_items.clear()
            #     print(f"[SAVE] {out_path} ({count} items)", flush=True)

    # # 收尾
    # if shard_items:
    #     shard_idx += 1
    #     out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-chunk{task_id}-{shard_idx:05d}.json")
    #     with open(out_path, "w") as f:
    #         json.dump(shard_items, f)
    #     shard_items.clear()
    #     print(f"[SAVE] {out_path} (final, total={count})", flush=True) 


if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)