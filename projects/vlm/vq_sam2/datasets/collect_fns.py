from typing import Dict, Sequence

import numpy as np
import torch


def vq_sam2_collate_fn(instances: Sequence[Dict]):
    
    pixel_values = []
    # image_grid_thw = []
    # sam2_pixel_values = []
    masks = []
    boxes = []

    for example in instances:
        pixel_values.append(example['pixel_values'])
        # image_grid_thw.append(example['image_grid_thw'])
        # sam2_pixel_values.append(example['sam2_pixel_values'])
        if isinstance(example['masks'], list):
            if isinstance(example['masks'][0], np.ndarray):
                _masks = np.stack(example['masks'], axis=0)
                _masks = torch.from_numpy(_masks)
                masks.append(_masks)
            else:
                masks.append(torch.stack(example['masks'], dim=0))
        else:
            masks.append(example['masks'])

        boxes.append(example['boxes'])
    
    data_dict = {
        'pixel_values': torch.stack(pixel_values, dim=0),
        'masks': masks,
        'boxes': torch.cat(boxes, dim=0)
    }

    return {'data': data_dict, 'data_samples': None}
