
import os
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

# Copyright (c) OpenMMLab. All rights reserved.
class RefCocoDataset(BaseDataset):
    """RefCOCO dataset.

    The `Refcoco` and `Refcoco+` dataset is based on
    `ReferItGame: Referring to Objects in Photographs of Natural Scenes
    <http://tamaraberg.com/papers/referit.pdf>`_.

    The `Refcocog` dataset is based on
    `Generation and Comprehension of Unambiguous Object Descriptions
    <https://arxiv.org/abs/1511.02283>`_.

    Args:
        ann_file (str): Annotation file path.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str): Prefix for training data.
        split_file (str): Split file path.
        split (str): Split name. Defaults to 'train'.
        text_mode (str): Text mode. Defaults to 'random'.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 split_file: str,
                 data_prefix: Dict,
                 split: str = 'train',
                 text_mode: str = 'random',
                 **kwargs):
        self.split_file = split_file
        self.split = split

        assert text_mode in ['original', 'random', 'concat', 'select_first']
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
        """Initialize the refs for RefCOCO."""
        anns, imgs = {}, {}
        for ann in self.instances['annotations']:
            anns[ann['id']] = ann
        for img in self.instances['images']:
            imgs[img['id']] = img

        refs, ref_to_ann = {}, {}
        for ref in self.splits:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            # add mapping related to ref
            refs[ref_id] = ref
            ref_to_ann[ref_id] = anns[ann_id]

        self.refs = refs
        self.ref_to_ann = ref_to_ann

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        self.splits = mmengine.load(self.split_file, file_format='pkl')
        self.instances = mmengine.load(self.ann_file, file_format='json')
        self._init_refs()
        img_prefix = self.data_prefix['img_path']

        ref_ids = [
            ref['ref_id'] for ref in self.splits if ref['split'] == self.split
        ]
        full_anno = []
        for ref_id in ref_ids:
            ref = self.refs[ref_id]
            ann = self.ref_to_ann[ref_id]
            ann.update(ref)
            full_anno.append(ann)

        image_id_list = []
        final_anno = {}
        for anno in full_anno:
            image_id_list.append(anno['image_id'])
            final_anno[anno['ann_id']] = anno
        annotations = [value for key, value in final_anno.items()]

        coco_train_id = []
        image_annot = {}
        for i in range(len(self.instances['images'])):
            coco_train_id.append(self.instances['images'][i]['id'])
            image_annot[self.instances['images'][i]
                        ['id']] = self.instances['images'][i]

        images = []
        for image_id in list(set(image_id_list)):
            images += [image_annot[image_id]]

        data_list = []

        grounding_dict = collections.defaultdict(list)
        for anno in annotations:
            image_id = int(anno['image_id'])
            grounding_dict[image_id].append(anno)

        join_path = mmengine.fileio.get_file_backend(img_prefix).join_path
        for image in images:
            img_id = image['id']
            instances = []
            sentences = []
            for grounding_anno in grounding_dict[img_id]:
                texts = [x['raw'].lower() for x in grounding_anno['sentences']]
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
                    'mask': grounding_anno['segmentation'],
                    'ignore_flag': 0
                }] * len(text)
                instances.extend(ins)
                sentences.extend(text)
            data_info = {
                'img_path': join_path(img_prefix, image['file_name']),
                'img_id': img_id,
                'instances': instances,
                'text': sentences
            }
            data_list.append(data_info)

        if len(data_list) == 0:
            raise ValueError(f'No sample in split "{self.split}".')

        return data_list
    

def main():
    # dataset = RefCocoDataset(
    #     data_root='./data/ref_seg/refcoco',
    #     data_prefix=dict(img_path='coco2014/train2014/'),
    #     pipeline=None,
    #     ann_file='instances.json',
    #     split_file='refs(unc).p',
    # )
    # dataset_name = 'refcoco'

    # dataset = RefCocoDataset(
    #     data_root='./data/ref_seg/refcoco+',
    #     data_prefix=dict(img_path='coco2014/train2014/'),
    #     pipeline=None,
    #     ann_file='instances.json',
    #     split_file='refs(unc).p',
    # )
    # dataset_name = 'refcocoplus'

    dataset = RefCocoDataset(
        data_root='./data/ref_seg/refcocog',
        data_prefix=dict(img_path='coco2014/train2014/'),
        pipeline=None,
        ann_file='instances.json',
        split_file='refs(umd).p',
    )
    dataset_name = 'refcocog'

    temp_save_root = "./any_other_seg_data"
    os.makedirs(temp_save_root, exist_ok=True)

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    for index in tqdm.tqdm(range(len(dataset))):
        data_dict = dataset.prepare_data(index)

        image_path = data_dict['img_path']
        
        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        instances, text = data_dict['instances'], data_dict['text']

        # process masks
        for idx, inst in enumerate(instances):
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
            for seg in inst['mask']:
                rles = mask_utils.frPyObjects([seg], ori_height, ori_width)
                m = mask_utils.decode(rles)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()

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
                print(f"[ERROR] encode failed seg_id={seg_id} file={image_path}: {e}", flush=True)
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
    # 可选：降低原生库线程数，规避奇怪的并发崩溃
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()