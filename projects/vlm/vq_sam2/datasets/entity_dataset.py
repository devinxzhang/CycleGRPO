import os
import json
import numpy as np
from PIL import Image
import random
import copy
import shutil
import cv2
from pycocotools import mask as mask_utils

import torch
import torchvision
from torch.utils.data import Dataset

from datasets import Dataset as HFDataset
from datasets import DatasetDict

from xtuner.dataset.huggingface import build_origin_dataset
from xtuner.registry import BUILDER


class EntityDataset(Dataset):

    def __init__(
        self,
        image_folder,
        preprocessor=None,
        repeats=1,
        entity_info_json=None,
        mini=None,
    ):
        super().__init__()
        self.mini = mini
        self.entity_info_json = entity_info_json
        self.image_folder = image_folder
        self.repeats = repeats

        self.preprocessor = BUILDER.build(preprocessor)
        json_data = self.load_data()
        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        self.text_data = build_origin_dataset(json_data, 'train')

        self._max_refetch = 1000

    def load_data(self):
        with open(self.entity_info_json, "r") as f:
            data_list = json.load(f)
        if self.mini is not None:
            data_list = data_list[:self.mini]
        for item in data_list:
            if self.image_folder is not None and self.image_folder != '':
                item.update({'image_file': os.path.join(self.image_folder, item['image_file'])})
        assert len(data_list) > 0, f"No data found in {self.coconut_info_json}"
        return data_list
    
    @property
    def modality_length(self):
        length_list = [100] * int(len(self.text_data) * self.repeats)
        return length_list
    
    def __len__(self):
        return int(len(self.text_data) * self.repeats)

    @property
    def real_len(self):
        return len(self.text_data)
    
    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))
    
    def decode_mask(self, object_masks, ori_height, ori_width):
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
        if len(binary_masks) == 0:
            binary_masks.append(np.zeros((ori_height, ori_width), dtype=np.uint8))
        masks = np.stack(binary_masks, axis=0)
        return masks

    def prepare_data(self, index):
        data_dict = self.text_data[index]

        image_file = data_dict['image_file']

        if not os.path.exists(image_file):
            print(image_file, " lost.")
            return None
        
        segms = [data_dict['segmentation']]
        if len(segms) == 0:
            return None
        
        image = Image.open(image_file).convert('RGB')
        ori_width, ori_height = image.size

        masks = self.decode_mask(segms, ori_height, ori_width)

        out_data_dict = {}

        # process sam image input
        sam2_image = np.array(image)
        sam2_image = self.preprocessor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        out_data_dict['pixel_values'] = sam2_pixel_values

        masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
        if masks.sum() == 0:
            return None
        
        masks = masks.sum(dim=0).to(torch.bool)
        out_data_dict['masks'] = masks.unsqueeze(0)

        boxes = torchvision.ops.masks_to_boxes(masks.unsqueeze(0))
        whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
        out_data_dict['boxes'] = boxes / whwh

        return out_data_dict
    
    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            if self.repeats >= 1:
                real_index = index % self.real_len
            else:
                real_index = random.randint(0, self.real_len-1)
            try:
                data = self.prepare_data(real_index)
            except:
                data = None
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data