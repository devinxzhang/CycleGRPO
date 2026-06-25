import random
import pandas as pd
import os
import sys
import json
import tqdm
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image

import copy
import torch
import torchvision
import pyarrow.parquet as pq
import pyarrow as pa
import io

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize


from xtuner.model.utils import guess_load_checkpoint



def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    frames = []

    frame_id = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

        frame_id += 1

    cap.release()
    return frames

def decode_masklet(masklet):
    masks = []
    for _rle in masklet:
        mask = mask_utils.decode(_rle)
        masks.append(mask)
    return masks

def select_valid_random_frames(masklets):
    # masklets形状为(n_frames, h, w)
    
    # 计算每帧mask的面积（非零元素数量）
    mask_areas = np.sum(masklets > 0, axis=(1, 2))  # 结果形状为(n_frames,)
    
    # 筛选出有效帧的索引（面积大于0的帧）
    valid_indices = np.where(mask_areas > 0)[0]
    
    # 如果没有有效帧，返回空结果
    if len(valid_indices) == 0:
        return np.array([]), np.array([])
    
    # 确定要选择的帧数（1-3，但不超过有效帧总数）
    max_possible = min(3, len(valid_indices))
    num_frames_to_select = np.random.randint(1, max_possible + 1)
    
    # 从有效帧中随机选择
    selected_indices = np.random.choice(valid_indices, size=num_frames_to_select, replace=False)
    
    # 返回选中的mask和对应的frame_id
    return masklets[selected_indices], selected_indices

def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union


QUESTION = [
    "Given the video, the subject {SUBJECT}, segment the full timeline into contiguous intervals and output a JSON array of (start, end, caption, masklets) that covers the entire duration.",
    "Produce a complete, gap-free sequence of (start, end, caption, masklets) for the subject {SUBJECT}.",
    "Track subject {SUBJECT} across the whole video and summarize behavior in segments. For each segment, return (start, end, caption, masklets).",
    "Return a JSON timeline of subject {SUBJECT} as segments (start, end, caption, masklets). Cover the entire video, keep segments contiguous.",
]



def main(task_id):

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'
    
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    temp_save_root = "./temp_data/rdcap/"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)

    sam2_config = SAM2Config(
        cfg_path="sam2.1_hiera_l.yaml",
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
    )


    CODEBOOK_SIZE = 1024
    CODEBOOK_DEPTH = 4
    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=256,
    )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    pretrained_pth = "./pretrained_weights/iter_17923.pth"
    pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

    pretrained_state_dict_new = {}
    for key in pretrained_state_dict.keys():
        new_key = copy.deepcopy(key)
        if key.startswith('hf_model.'):
            new_key = new_key[len('hf_model.'):]
        pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
    
    vq_sam2.load_state_dict(pretrained_state_dict_new)

    sam2_image_processor = DirectResize(1024)


    sav_root = "./data/sam_v_full"
    sav_frame_root = "./data/sam_v_frames"
    if not os.path.exists(sav_frame_root):
        os.makedirs(sav_frame_root)

    rdcap_parquet = "./data/PLM-Video-Human/rdcap/plm_rdcap_train.parquet"
    df = pd.read_parquet(rdcap_parquet, engine="pyarrow")
    row_index = 0
    num_rows = df.shape[0]
    for row in df.itertuples(index=True, name='RowData'):
        video_file = getattr(row, 'video')
        masklet_id = getattr(row, 'masklet_id')
        total_frames = getattr(row, 'total_frames')
        dense_captions = getattr(row, 'dense_captions')

        print(dense_captions)

        video_id_str = video_file.split(".mp4")[0].split("sav_")[-1]
        video_id_int = int(video_id_str)
        split_id = video_id_int // 1000
        split_name = f"sav_"+str(split_id).zfill(3)

        print(f"================{row_index+1} / {num_rows}=================")
        row_index += 1

        if os.path.exists(os.path.join(temp_save_root, f"sav_{video_id_int}_{masklet_id}.json")):
            print("file exists.................")
            continue

        if not os.path.exists(os.path.join(sav_frame_root, video_id_str)):
            os.makedirs(os.path.join(sav_frame_root, video_id_str))

        video_path = os.path.join(sav_root, split_name, "sav_train", split_name, video_file)
        anno_path = os.path.join(sav_root, split_name, "sav_train", split_name, video_file.replace(".mp4", "_auto.json"))
        if not os.path.exists(video_path) or not os.path.exists(anno_path):
            print("FILES ARE NOT FOUND!!!", video_path, anno_path)
            continue


        video_frames = get_video_frames(video_path)

        video_frames = video_frames[::4] # list, item.shape == h, w, 3
        ori_height, ori_width = video_frames[0].shape[:2]

        # mask annotation
        with open(anno_path, 'r') as f:
            mask_data = json.load(f)
        masklets = decode_masklet(mask_data['masklet'])
        masklets = np.stack(masklets, axis=0)  # (n_frames, h, w, n_obj)

        assert len(video_frames) == len(masklets) == total_frames, f"total_frames={total_frames}, len(video_frames)={len(video_frames)}, len(masklets)={len(masklets)}"

        masklet_id_2_idx = {_masklet_id_: idx for idx, _masklet_id_ in enumerate(mask_data["masklet_id"])}
        masklet_idx = masklet_id_2_idx[masklet_id]

        masklet_for_idx = masklets[:, :, :, masklet_id]

        print(np.sum(masklet_for_idx, axis=(1, 2)))
        exit(0)


        select_masks, select_frame_ids = select_valid_random_frames(masklet_for_idx)
        if len(select_masks) == 0:
            print("!!!!!!!!!!!!!!!!!this case has no valid masklets!!!!!!!!!!!!!!!")
            continue
        
        frame_id_2_quant_codes_dict = {}
        frame_id_2_image_path = {}
        for frame_id, mask in enumerate(masklet_for_idx):
            frame_image = Image.fromarray(video_frames[frame_id]).convert('RGB')

            if not os.path.exists(os.path.join(sav_frame_root, f"sav_{video_id_str}")):
                os.makedirs(os.path.join(sav_frame_root, f"sav_{video_id_str}"))


            frame_image.save(os.path.join(sav_frame_root, f"sav_{video_id_str}", f"frame_{frame_id}.jpg"))
            frame_id_2_image_path[frame_id] = os.path.join(sav_frame_root, f"sav_{video_id_str}", f"frame_{frame_id}.jpg")

            sam2_image = np.array(frame_image)
            sam2_image = sam2_image_processor.apply_image(sam2_image)
            sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
            sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

            if np.sum(mask) == 0:
                frame_id_2_quant_codes_dict[frame_id] = None
                continue
            mask_tensor = torch.tensor(mask).unsqueeze(0)
            boxes = torchvision.ops.masks_to_boxes(mask_tensor)

            whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
            boxes = boxes / whwh
            boxes = boxes.to(vq_sam2.device)
            mask_tensor = [m.unsqueeze(0).to(vq_sam2.device) for m in mask_tensor]

            with torch.no_grad():
                vq_sam2_output = vq_sam2(
                    sam2_pixel_values,
                    mask_tensor,
                    boxes,
                )
            
            quant_codes = vq_sam2_output.quant_codes.squeeze().cpu().numpy().astype(np.int32).tolist()
            remap_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(quant_codes)]
            quant_codes = remap_quant_codes

            # pred_masks = vq_sam2_output.pred_masks
            # pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
            # pred_masks = pred_masks > 0.5
            # pred_masks = pred_masks[0].cpu().numpy().astype(np.uint8)
            # target_mask = mask_tensor[0].cpu().numpy().astype(np.uint8)
            frame_id_2_quant_codes_dict[frame_id] = quant_codes

            torch.cuda.empty_cache()

        print(frame_id_2_quant_codes_dict)
        exit(0)
        
        item_str_list = []
        for frame_id in select_frame_ids:
            item_str = "{\"frame_id\": " + f"{frame_id}" + ", \"mask_2d\": [" + ", ".join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in frame_id_2_quant_codes_dict[frame_id]]) + "]}"
            item_str_list.append(item_str)
        item_list_str = "[" + ",\n".join(item_str_list) + "]"

        question = "<video>\n" + random.choice(QUESTION).format(SUBJECT=item_list_str)

        answer = "```json\n["
        answer_segment_str_list = []
        for segment in dense_captions:
            start_frame = segment['start_frame']
            end_frame = segment['end_frame']
            masklets_code_str_list = []
            for frame_id in range(start_frame, end_frame+1):
                quant_codes_str = "{\"frame_id\": " + f"{frame_id}" + ", \"mask_2d\": [" + ", ".join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in frame_id_2_quant_codes_dict[frame_id]]) + "]}"
                masklets_code_str_list.append(quant_codes_str)
            masklets_str = "[" + ",\n".join(masklets_code_str_list) + "]"
            segment_str = "{\"start\": " + str(segment["start_frame"]) + ", \"end\": " + str(segment["end_frame"]) + ", \"caption\": " + segment["caption"] + ", \"masklets\": " + masklets_str + "}"
            answer_segment_str_list.append(segment_str)
        
        answer_segment_list_str = ",\n".join(answer_segment_str_list)
        answer = answer + answer_segment_list_str + "]\n```"

        conversation = [
            {'from': 'human', 'value': question},
            {'from': 'gpt', 'value': answer},
        ]
        ret_data_dict = {
            'video': [frame_id_2_image_path[frame_id] for frame_id in range(total_frames)],
            'conversations': conversation,
        }

        print(ret_data_dict)
        exit(0)
        
        with open(os.path.join(temp_save_root, f"sav_{video_id_int}_{masklet_id}.json"), 'w') as f:
            json.dump(ret_data_dict, f)


if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)