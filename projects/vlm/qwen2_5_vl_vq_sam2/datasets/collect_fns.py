from typing import Dict, Sequence

import numpy as np
import torch
import itertools 
import copy

from torch.nn.utils.rnn import pad_sequence

from xtuner.parallel.sequence import (get_sequence_parallel_world_size,
                                      pad_for_sequence_parallel)

DEFAULT_PAD_TOKEN_INDEX = 151643
IGNORE_INDEX = -100
model_max_length = 8192

"""
input_ids .shape:  torch.Size([1, 658])
labels .shape:  torch.Size([1, 658])
position_ids .shape:  torch.Size([3, 1, 658])
attention_mask :  [658]
pixel_values .shape:  torch.Size([1380, 1176])
image_grid_thw .shape:  torch.Size([1, 3])
"""


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor

def qwen2_5_vl_vq_sam2_collate_fn(instances: Sequence[Dict]):

    input_ids = [instance['input_ids'].squeeze(0) for instance in instances]
    labels = [instance['labels'].squeeze(0) for instance in instances]
    position_ids = [instance['position_ids'] for instance in instances]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=DEFAULT_PAD_TOKEN_INDEX)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    padded_position_ids = pad_and_cat(position_ids)
    if padded_input_ids.shape[-1] > model_max_length:
        padded_input_ids = padded_input_ids[:, :model_max_length]
        padded_labels = padded_labels[:, :model_max_length]
        padded_position_ids = padded_position_ids[:, :, :model_max_length]
    attention_mask = padded_input_ids.ne(DEFAULT_PAD_TOKEN_INDEX)

    if any("pixel_values" in instance for instance in instances):
        pixel_values = torch.cat([instance['pixel_values'] for instance in instances if "pixel_values" in instance], dim=0)
        image_grid_thw = torch.cat([instance['image_grid_thw'] for instance in instances if "image_grid_thw" in instance], dim=0)
    else:
        pixel_values = None
        image_grid_thw = None
    
    batch = dict(
        input_ids=copy.deepcopy(padded_input_ids),
        labels=copy.deepcopy(padded_labels),
        attention_mask=copy.deepcopy(attention_mask),
        position_ids=copy.deepcopy(padded_position_ids),
        pixel_values=copy.deepcopy(pixel_values),
        image_grid_thw=copy.deepcopy(image_grid_thw),
    )

    return {'data': batch, 'data_samples': None}


def qwen2_5_vl_vq_sam2_collate_fn_data_flatten(instances: Sequence[Dict]):

    input_ids = torch.cat([instance['input_ids'] for instance in instances], dim=1)
    labels = torch.cat([instance['labels'] for instance in instances], dim=1)
    position_ids = torch.cat([instance['position_ids'] for instance in instances], dim=2)
    attention_mask = list(
        itertools.chain(
            *(
                instance["attention_mask"]
                for instance in instances
                if "attention_mask" in instance
            )
        )
    )
    seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
    cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)

    packed_data_dict = {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "attention_mask": cumsum_seq_lens,
    }

    if any("pixel_values" in instance for instance in instances):
        pixel_values = torch.cat([instance['pixel_values'] for instance in instances if "pixel_values" in instance], dim=0)
        image_grid_thw = torch.cat([instance['image_grid_thw'] for instance in instances if "image_grid_thw" in instance], dim=0)

        packed_data_dict.update({
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        })

    return {'data': packed_data_dict, 'data_samples': None}



