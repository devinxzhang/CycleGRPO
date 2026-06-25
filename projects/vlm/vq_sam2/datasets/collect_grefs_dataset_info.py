import collections
import os
import os.path as osp
import random
from typing import Dict, List
import json
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils
import torch
import copy
import tqdm
import torchvision

import mmengine
from mmengine.dataset import BaseDataset

from xtuner.model.utils import guess_load_checkpoint

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize



SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

ANSWER_LIST = [
    "It is {SEG}.",
    "Sure, {SEG}.",
    "Sure, it is {SEG}.",
    "Sure, the segmentation result is {SEG}.",
    "{SEG}.",
]

NO_TARGETS_ANSWER_LIST = [
    "No target."
]


class GRefCoCoDataset(BaseDataset):
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 split_file: str,
                 data_prefix=dict(img_path='train2014/'),
                 split: str = 'train',
                 text_mode: str = 'random',
                 **kwargs):
        self.split_file = split_file
        self.split = split
        self.text_mode = text_mode

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            **kwargs,
        )
        

    def _join_prefix(self):
        if not mmengine.is_abs(self.split_file) and self.split_file:
            self.split_file = osp.join(self.data_root, self.split_file)

        return super()._join_prefix()
    
    def _init_refs(self):
        """Initialize the refs for GRefCOCO."""
        anns, imgs = {}, {}
        for ann in self.instances['annotations']:
            anns[ann['id']] = ann
        for img in self.instances['images']:
            imgs[img['id']] = img

        anns[-1] = {"segmentation": None, "area": 0.0, "iscrowd": 0, "bbox": None, "category_id": -1, "id": -1}

        refs, ref_to_ann = {}, {}
        for ref in self.splits:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            # add mapping related to ref            
            refs[ref_id] = ref
            ref_to_ann[ref_id] = [anns[_ann_id] for _ann_id in ann_id]
            assert len(ref_to_ann[ref_id]) == len(ann_id)

        self.refs = refs
        self.ref_to_ann = ref_to_ann

    def load_data_list(self) -> List[dict]:
        """Load data list.
        Specially, there are no_targets items, where ref['ann_id'] = [-1]
        """
        self.splits = json.load(open(self.split_file, 'rb'))
        self.instances = mmengine.load(self.ann_file, file_format='json')
        self._init_refs()
        img_prefix = self.data_prefix['img_path']

        ref_ids = [
            ref['ref_id'] for ref in self.splits if ref['split'] == self.split
        ]
        image_id_list = []
        for ref_id in ref_ids:
            image_id_list.append(self.refs[ref_id]['image_id'])
        image_annot = {}
        for i in range(len(self.instances['images'])):
            image_annot[self.instances['images'][i]
                        ['id']] = self.instances['images'][i]
        images = []
        for image_id in list(set(image_id_list)):
            images += [image_annot[image_id]]

        grounding_dict = collections.defaultdict(list)
        for ref_id in ref_ids:
            ref = self.refs[ref_id]
            ann_list = [copy.deepcopy(e) for e in self.ref_to_ann[ref_id]]
            ann_list[0].update(ref)
            image_id = ref['image_id']
            grounding_dict[image_id].append(ann_list)
        
        data_list = []

        join_path = mmengine.fileio.get_file_backend(img_prefix).join_path
        for image in images:
            img_id = image['id']
            instances = []
            sentences = []
            anno_ids = []
            for grounding_anno in grounding_dict[img_id]:
                texts = [x['raw'].lower() for x in grounding_anno[0]['sentences']]
                # random select one text
                if self.text_mode == 'random':
                    idx = random.randint(0, len(texts) - 1)
                    text = [texts[idx]]
                # concat all texts
                elif self.text_mode == 'concat':
                    text = [''.join(texts)]
                # select the first text
                elif self.text_mode == 'select_first':
                    text = [texts[0]]
                # use all texts
                elif self.text_mode == 'original':
                    text = texts
                else:
                    raise ValueError(f'Invalid text mode "{self.text_mode}".')
                ins = [{
                    'mask': [_grounding_anno['segmentation'] for _grounding_anno in grounding_anno],
                    'ignore_flag': 0
                }] * len(text)
                instances.extend(ins)
                sentences.extend(text)
                anno_ids.extend([grounding_anno[0]['ann_id']]*len(text))
            data_info = {
                'img_path': join_path(img_prefix, image['file_name']),
                'img_id': img_id,
                'instances': instances,
                'text': sentences,
                'anno_ids': anno_ids,
            }
            data_list.append(data_info)

        if len(data_list) == 0:
            raise ValueError(f'No sample in split "{self.split}".')

        return data_list

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

def encode_binary_mask(bin_mask_bool):
    # 跳过空 mask，避免 encode 的边界行为
    if not np.any(bin_mask_bool):
        return None
    # pycocotools 期望的是 Fortran 连续的 0/1 uint8，形状 HxW
    m = np.asfortranarray(bin_mask_bool.astype(np.uint8, copy=False))
    rle = mask_utils.encode(m)
    # 某些版本返回的是{'counts': bytes, 'size': [H, W]}
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def main():

    dataset_name = 'grefs'

    temp_save_root = "./any_other_seg_data"
    os.makedirs(temp_save_root, exist_ok=True)


    dataset = GRefCoCoDataset(
        data_root='./data/ref_seg/grefs',
        ann_file='instances.json',
        split_file='grefs(unc).json',
        data_prefix=dict(img_path='coco2014/train2014/'),
    )

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    for index in tqdm.tqdm(range(len(dataset))):
        data_dict = dataset.prepare_data(index)

        image_path = data_dict['img_path']
        image_file = os.path.basename(image_path)
        if '.jpg' in image_file:
            image_id = image_file.split('.jpg')[0]
        elif '.png' in image_file:
            image_id = image_file.split('.png')[0]
        else:
            raise ValueError(f'Invalid image file "{image_file}".')
        
        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        instances, text, anno_ids = data_dict['instances'], data_dict['text'], data_dict['anno_ids']

        index = np.random.choice(range(len(instances)), 3, replace=True)
        conversation = []
        turn_idx = 0
        for idx in index:
            inst = instances[idx]
            phrase = text[idx].lower()
            if '.' == phrase[-1]:
                phrase = phrase[:-1]

            if inst["mask"] is None or inst["mask"][0] is None:
                continue

            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
            
            assert len(inst["mask"]) == len(anno_ids[idx])

            binary_masks = decode_mask(inst["mask"], ori_height, ori_width)
            assert len(binary_masks) == len(inst["mask"])
            for m in binary_masks:
                binary_mask += m
            
            binary_mask = binary_mask>0
            
            assert len(binary_mask.shape) == 2

            try:
                rle = encode_binary_mask(binary_mask.astype(np.bool))
                if rle is None:
                    # 空实例，跳过但记录
                    # print(f"[WARN] empty mask seg_id={seg_id} file={image_file}", flush=True)
                    continue

                shard_items.append({
                    "image_file": image_path,
                    "segmentation": rle,
                })
                count += 1

                if count % shard_size == 0:
                    shard_idx += 1
                    out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}.json")
                    with open(out_path, "w") as f:
                        json.dump(shard_items, f)
                    shard_items.clear()
                    print(f"[SAVE] {out_path} ({count} items)", flush=True)

            except Exception as e:
                # 如果 pycocotools 在 C 层崩溃，这里是抓不到的；但大多数数据问题能在这儿被捕到
                print(f"[ERROR] ...", flush=True)
                continue

    # 收尾
    if shard_items:
        shard_idx += 1
        out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}.json")
        with open(out_path, "w") as f:
            json.dump(shard_items, f)
        shard_items.clear()
        print(f"[SAVE] {out_path} (final, total={count})", flush=True) 


if __name__ == "__main__":
    main()






