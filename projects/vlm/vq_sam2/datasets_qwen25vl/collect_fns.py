from typing import Dict, Sequence

import numpy as np
import torch
from xtuner.model.utils import guess_load_checkpoint

PAD_TOKEN_ID = 151643
IGNORE_INDEX = -100
MODEL_MAX_LENGTH = 8192

def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor

def qwen25vl_vqsam2_collate_fn(instances: Sequence[Dict]):
    input_ids, labels, position_ids, attention_mask = tuple(
        [instance[key] for instance in instances]
        for key in ("input_ids", "labels", "position_ids", "attention_mask")
    )
    input_ids = [ids.squeeze(0) for ids in input_ids]
    labels = [ids.squeeze(0) for ids in labels]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=PAD_TOKEN_ID
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX
    )
    position_ids = pad_and_cat(position_ids)
    input_ids = input_ids[:, :MODEL_MAX_LENGTH]
    labels = labels[:, :MODEL_MAX_LENGTH]
    position_ids = position_ids[:, :MODEL_MAX_LENGTH]
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(PAD_TOKEN_ID),
    )
    images = list(
        instance["pixel_values"]
        for instance in instances
        if "pixel_values" in instance
    )
    videos = list(
        instance["pixel_values_videos"]
        for instance in instances
        if "pixel_values_videos" in instance
    )
    if len(images) != 0:
        concat_images = torch.cat([image for image in images], dim=0)
        grid_thw = [
            instance["image_grid_thw"]
            for instance in instances
            if "image_grid_thw" in instance
        ]
        grid_thw = torch.cat(grid_thw, dim=0)
    else:
        concat_images = None
        grid_thw = None

    if len(videos) != 0:
        concat_videos = torch.cat([video for video in videos], dim=0)
        video_grid_thw = [
            instance["video_grid_thw"]
            for instance in instances
            if "video_grid_thw" in instance
        ]
        video_grid_thw = torch.cat(video_grid_thw, dim=0)
    else:
        concat_videos = None
        video_grid_thw = None

    batch["pixel_values"] = concat_images
    batch["image_grid_thw"] = grid_thw
    batch["pixel_values_videos"] = concat_videos
    batch["video_grid_thw"] = video_grid_thw
    batch["position_ids"] = position_ids


    # handle sam2 inputs
    has_mask = any(inst.get('masks', None) is not None for inst in instances)
    if not has_mask:
        batch['masks'] = None
        batch['sam2_pixel_values'] = None
    else:
        sam2_pixel_values = []
        masks = []
        for example in instances:
            if example.get('sam2_pixel_values', None) is not None:
                sam2_pixel_values.extend(example['sam2_pixel_values'])
                masks.extend(example['masks'])
        batch['sam2_pixel_values'] = torch.stack(sam2_pixel_values, dim=0)
        batch['masks'] = masks
    
    # pretrained_pth = "./pretrained_weights/iter_17923_resampled_256x4.pth"
    # pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
    # codebooks_0 = pretrained_state_dict['quantizer.codebooks.0.weight'].detach().cpu()[:-1, :]
    # codebooks_1 = pretrained_state_dict['quantizer.codebooks.1.weight'].detach().cpu()[:-1, :]
    # codebooks_2 = pretrained_state_dict['quantizer.codebooks.2.weight'].detach().cpu()[:-1, :]
    # codebooks_3 = pretrained_state_dict['quantizer.codebooks.3.weight'].detach().cpu()[:-1, :]
    # batch["codebook_embeds"] = [codebooks_0, codebooks_1, codebooks_2, codebooks_3]

    
    return {'data': batch, 'data_samples': None}

