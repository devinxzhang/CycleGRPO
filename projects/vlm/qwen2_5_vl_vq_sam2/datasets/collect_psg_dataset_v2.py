import numpy as np
from collections import defaultdict

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

import mmengine
from mmengine.dataset import BaseDataset

from xtuner.model.utils import guess_load_checkpoint

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize

from mmdet.registry import DATASETS
from mmdet.datasets.coco_panoptic import COCOPanoptic, CocoPanopticDataset

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
    
    alpha = 0.3

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


class COCOPanopticRelation(COCOPanoptic):
    def createIndex(self):
        # create index
        print('creating index...')
        # anns stores 'segment_id -> annotation'
        anns, cats, imgs = {}, {}, {}
        relations = {}

        segments_info = {}

        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann, img_info in zip(self.dataset['annotations'],
                                     self.dataset['images']):
                img_info['segm_file'] = ann['file_name']
                for seg_ann in ann['segments_info']:
                    # to match with instance.json
                    seg_ann['image_id'] = ann['image_id']
                    seg_ann['height'] = img_info['height']
                    seg_ann['width'] = img_info['width']
                    img_to_anns[ann['image_id']].append(seg_ann)
                    # segment_id is not unique in coco dataset orz...
                    if seg_ann['id'] in anns.keys():
                        anns[seg_ann['id']].append(seg_ann)
                    else:
                        anns[seg_ann['id']] = [seg_ann]

                relations[ann['image_id']] = ann['relations']
                segments_info[ann['image_id']] = ann['segments_info']

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    cat_to_imgs[seg_ann['category_id']].append(ann['image_id'])

        print('index created!')

        self.anns = anns
        self.imgToAnns = img_to_anns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats
        self.relations = relations
        self.segments_info = segments_info
        self.relations_categories = self.dataset['relations_categories']
        self.relationID2Categories = {item['id']: item['name'] for item in self.dataset['relations_categories']}
        
    
# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag

def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image

def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray):
            an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    with open(file_name, "rb") as f:
        image = Image.open(f)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        image = _apply_exif_orientation(image)
        return convert_PIL_to_numpy(image, format)
    raise ValueError(f"Failed to read image at: {file_name}")


QUESTIONS = [
    "Generate a scene graph for this image. Identify the main objects and describe their relationships to each other.",
    "Create a scene graph by identifying triplets of <subject, predicate, object>. Focus on the most prominent interactions.",
    "What objects are in this image and how are they interacting? List them as a scene graph.",
    "Generate a structured scene graph for the provided image.",
    "Your task is to generate a scene graph for the image.",
    "Generate a scene graph for this image and format the output as a list of JSON objects."
]


def main(task_id):
    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    temp_save_root = "./temp_data_256x2_0927/psg_multiround"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    dataset_name = "psg_multiround"

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


    coco = COCOPanopticRelation('./data/psg_data/psg_tra.json')
    cat_ids = coco.get_cat_ids()
    cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    categories = coco.cats
    img_ids = coco.get_img_ids()
    data_infos = []
    for i in img_ids:
        info = coco.load_imgs([i])[0]
        info['filename'] = info['file_name']
        if 'segm_file' in info:
            info['segm_file'] = info['segm_file']
        else:
            info['segm_file'] = info['filename'].replace('jpg', 'png')
        data_infos.append(info)
    imgid2relations = coco.relations


    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    rows = len(data_infos)
    chunk_size = (rows+7) // 8
    _start_ = task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = rows if _end_ > rows else _end_

    for data_info in tqdm.tqdm(data_infos[_start_:_end_]):
        file_name = data_info['file_name']
        image_path = os.path.join('./data/coco', file_name)
        segm_file = data_info['segm_file']
        segm_path = os.path.join('./data/coco/annotations', segm_file)

        img_id = data_info['id']
        relation_annos = imgid2relations[img_id]

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        from panopticapi.utils import rgb2id
        pan_seg_gt = read_image(segm_path, "RGB")
        pan_seg_gt = rgb2id(pan_seg_gt)

        candidate_categories = [v['name'] for k, v in coco.cats.items()]
        candidate_predicates = [v['name'] for v in coco.relations_categories]
        candidate_categories_str = "{" + ", ".join(candidate_categories) + "}"
        candidate_predicates_str = "{" + ", ".join(candidate_predicates) + "}"

        question = "<image>\nPlease carefully check the image and detect the following objects: " + candidate_categories_str

        segment_id_2_mask_tokens = {}
        category_name_to_masks = {}
        for segment_info in coco.segments_info[img_id]:
            mask = pan_seg_gt == segment_info['id']
            category_name = coco.cats[segment_info['category_id']]['name']
            masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in [mask]])
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

            with torch.no_grad():
                vq_sam2_output = vq_sam2(
                    sam2_pixel_values.repeat(num_ins, 1, 1, 1),
                    masks,
                    boxes,
                    reconstruct_mask=False,
                )
                quant_codes = vq_sam2_output.quant_codes
            
            quant_codes = quant_codes.cpu().numpy().astype(np.int32).tolist()
            remap_quant_codes = []
            for _quant_codes in quant_codes:
                _quant_codes = _quant_codes[0]
                remap_quant_codes.append([depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes)])
            quant_codes = remap_quant_codes

            sam2token_list = []
            for _quant_codes_ in quant_codes:
                sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes_]) + MT_END_TOKEN
                sam2token_list.append(sam2_tokens)
            
            segment_id_2_mask_tokens[segment_info['id']] = sam2token_list[0]
            if category_name not in category_name_to_masks:
                category_name_to_masks[category_name] = []
            category_name_to_masks[category_name].append(sam2token_list[0])
        
        # first round QA
        answer = "```json\n[{mask_2d}]\n```"
        mask_2d_str = ''
        for category_name, mask_token_list in category_name_to_masks.items():
            for mask_tokens in mask_token_list:
                item_str = "{\"mask_2d\": " + "\"" + mask_tokens + "\"" + ", \"label\": \"" + category_name + "\"}"
                mask_2d_str += item_str + ",\n"
        if mask_2d_str == '':
            continue
        mask_2d_str = mask_2d_str[:-len(",\n")]
        answer = answer.format(mask_2d=mask_2d_str)

        conversations = []
        conversations.append({'from': 'human', 'value': question})
        conversations.append({'from': 'gpt', 'value': answer})

        # second round QA
        question = "CANDIDATE PREDICATES: \n" + candidate_predicates_str + "\n" + random.choice(QUESTIONS)
        answer = "```json\n[{mask_2d}]\n```"
        mask_2d_str = ''
        for triplet in relation_annos:
            sub_segment_info = coco.segments_info[img_id][triplet[0]]
            obj_segment_info = coco.segments_info[img_id][triplet[1]]
            sub_mask_tokens = segment_id_2_mask_tokens[sub_segment_info['id']]
            obj_mask_tokens = segment_id_2_mask_tokens[obj_segment_info['id']]
            sub_category = coco.cats[sub_segment_info['category_id']]['name']
            obj_category = coco.cats[obj_segment_info['category_id']]['name']
            relation_category = coco.relations_categories[triplet[2]]['name']

            sub_item_str = "{\"mask_2d\": " + "\"" + sub_mask_tokens + "\"" + ", \"label\": \"" + sub_category + "\"}"
            obj_item_str = "{\"mask_2d\": " + "\"" + obj_mask_tokens + "\"" + ", \"label\": \"" + obj_category + "\"}"
            item_str = "{\"subject\": " + sub_item_str + ", \"predicate\": " + "\"" + relation_category + "\"" + ", \"object\": " + obj_item_str + "}"
            mask_2d_str += item_str + ",\n"

        if mask_2d_str == '':
            continue

        mask_2d_str = mask_2d_str[:-len(",\n")]
        answer = answer.format(mask_2d=mask_2d_str)

        conversations.append({'from': 'human', 'value': question})
        conversations.append({'from': 'gpt', 'value': answer})

        ret_data_dict = {
            'image': image_path,
            'conversations': conversations,
        }
        shard_items.append(ret_data_dict)
        count += 1

        if count % shard_size == 0:
            shard_idx += 1
            out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}-chunk{task_id}.json")
            with open(out_path, "w") as f:
                json.dump(shard_items, f)
            shard_items.clear()
            print(f"[SAVE] {out_path} ({count} items)", flush=True)

    # 收尾
    if shard_items:
        shard_idx += 1
        out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}-chunk{task_id}.json")
        with open(out_path, "w") as f:
            json.dump(shard_items, f)
        shard_items.clear()
        print(f"[SAVE] {out_path} (final, total={count})", flush=True)

if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)
