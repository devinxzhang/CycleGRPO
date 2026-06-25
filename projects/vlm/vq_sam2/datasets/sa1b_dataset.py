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


class SA1BDataset(Dataset):

    def __init__(
        self,
        image_folder,
        preprocessor=None,
        multi_targets=False,
        repeats=1,
        fast_load=False,
        sam_info_json=None,
        mini=None,
        scan_record_folder=None,
    ):
        super().__init__()
        self.mini = mini
        self.fast_load = fast_load
        if self.fast_load:
            assert sam_info_json is not None
        self.sam_info_json = sam_info_json
        self.image_folder = image_folder
        self.multi_targets = multi_targets
        self.repeats = repeats
        self.scan_record_folder=scan_record_folder

        self.preprocessor = BUILDER.build(preprocessor)
        if fast_load:
            json_data = self.fast_load_data()
        else:
            json_data = self.load_data()
        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        self.text_data = build_origin_dataset(json_data, 'train')

        self._max_refetch = 1000

    def fast_load_data(self):
        with open(self.sam_info_json, "r") as f:
            data_list = json.load(f)
        if self.mini is not None:
            data_list = data_list[:self.mini]
        for item in data_list:
            if self.image_folder is not None and self.image_folder != '':
                item.update({'image_file': os.path.join(self.image_folder, item['image_file']),
                            'json_file': os.path.join(self.image_folder, item['json_file'])})
        assert len(data_list) > 0, f'No data found in {self.image_folder} and sub-folders of {self.image_folder}'
        return data_list

    def load_data(self):
        data_list = []
        dir_list = os.listdir(self.image_folder)
        for dir_name in dir_list:
            if os.path.isfile(os.path.join(self.image_folder, dir_name)):
                if dir_name.endswith('.json'):
                    json_path = os.path.join(self.image_folder, dir_name)
                    image_name = dir_name.replace('.json', '.jpg')
                    image_path = os.path.join(self.image_folder, image_name)
                    data_list.append({'image_file': image_path, 'json_file': json_path})
            elif os.path.isdir(os.path.join(self.image_folder, dir_name)):
                dir_path = os.path.join(self.image_folder, dir_name)
                for json_file in os.listdir(dir_path):
                    if json_file.endswith('.json'):
                        json_path = os.path.join(dir_path, json_file)
                        image_name = json_file.replace('.json', '.jpg')
                        image_path = os.path.join(dir_path, image_name)
                        data_list.append({'image_file': image_path, 'json_file': json_path})
        assert len(data_list) > 0, f'No data found in {self.image_folder} and sub-folders of {self.image_folder}'
        
        # if self.multi_targets:
        #     return data_list

        # entity_list = []
        # for data_dict in data_list:
        #     json_file = data_dict['json_file']
        #     with open(json_file, 'r') as f:
        #         json_data = json.load(f)
        #     for anno in json_data['annotations']:
        #         entity_list.append({'image_file': data_dict['image_file'], 'json_file': json_file, 'id': anno['id']})
        # return entity_list
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
    
    def filter_nested_masks(self, masks):
        if len(masks) <= 1:
            return masks
        
        areas = np.sum(masks, axis=(1, 2))
        
        sorted_indices = np.argsort(areas)[::-1]
        sorted_masks = masks[sorted_indices]
        
        keep_indices = []
        
        for i, mask in enumerate(sorted_masks):
            is_nested = False
            
            for j in keep_indices:
                larger_mask = sorted_masks[j]

                intersection = np.logical_and(mask, larger_mask)
                intersection_area = np.sum(intersection)
                current_area = np.sum(mask)
                
                if intersection_area == current_area and current_area > 0:
                    is_nested = True
                    break
            
            if not is_nested:
                keep_indices.append(i)
        
        filtered_masks = sorted_masks[keep_indices]
        return filtered_masks
    
    def prepare_data(self, index, fine=False):
        data_dict = self.text_data[index]

        image_file = data_dict['image_file']
        json_file = data_dict['json_file']

        image_id = os.path.basename(image_file).split('.jpg')[0]

        if not os.path.exists(image_file) or not os.path.exists(json_file):
            print(image_file, "---", json_file)
            return None
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
        except:
            return None

        # if self.multi_targets:
        #     segms = [anno['segmentation'] for anno in json_data['annotations']]
        # else:
        #     all_ids = [anno["id"] for anno in json_data['annotations']]
        #     random.shuffle(all_ids)
        #     entity_id = all_ids[0]
        #     segms = []
        #     for anno in json_data['annotations']:
        #         if anno['id'] == entity_id:
        #             segms.append(anno['segmentation'])
        segms = [anno['segmentation'] for anno in json_data['annotations']]
        if len(segms) == 0:
            return None
        if not self.multi_targets:
            if os.path.exists(os.path.join(self.scan_record_folder, f"{image_id}.json")):
                with open(os.path.join(self.scan_record_folder, f"{image_id}.json"), 'r') as f:
                    filtered_indices = json.load(f)
                if len(filtered_indices) == 0:
                    if fine:
                        print(f"all masks in {image_file} have been scanned.")
                        filtered_indices = list(range(len(segms)))
                    else:
                        filtered_indices = []
            else:
                filtered_indices = list(range(len(segms)))
            if len(filtered_indices) == 0:
                return None
            random_idx = random.choice(filtered_indices)
            segms = [segms[random_idx]]
            left_indices = [x for x in filtered_indices if x != random_idx]
            with open(os.path.join(self.scan_record_folder, f"{image_id}.json"), 'w') as f:
                json.dump(left_indices, f)
        else:
            if len(segms) > 10:
                random_idxs = np.random.choice(len(segms), size=10)
                segms = [segms[_random_idx] for _random_idx in random_idxs]
        
        image = Image.open(image_file).convert('RGB')
        ori_width, ori_height = image.size

        masks = self.decode_mask(segms, ori_height, ori_width)

        out_data_dict = {}

        # process sam image input
        sam2_image = np.array(image)
        sam2_image = self.preprocessor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        out_data_dict['pixel_values'] = sam2_pixel_values

        # process masks
        if self.multi_targets:
            filtered_masks = self.filter_nested_masks(masks)
            if len(filtered_masks) == 0:
                return None
            
            num_masks = len(filtered_masks)
            num_to_select = random.randint(2, 8)
            if num_to_select > num_masks:
                num_to_select = num_masks
            
            selected_indices = random.sample(range(num_masks), num_to_select)
            selected_masks = filtered_masks[selected_indices]
            masks = selected_masks
        
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
                if _ == self._max_refetch:
                    data = self.prepare_data(real_index, fine=True)
                else:
                    data = self.prepare_data(real_index)
            except:
                data = None
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            
            return data

