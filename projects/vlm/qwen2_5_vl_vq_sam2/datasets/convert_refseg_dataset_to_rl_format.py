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
import re

import mmengine
from mmengine.dataset import BaseDataset

from xtuner.model.utils import guess_load_checkpoint

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize


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

def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union


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

def extract_mt_token_ids(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]

def find_first_index(arr, value):
    """
    在NumPy数组中找到第一个指定值的第一个出现的索引
    
    参数:
        arr: NumPy数组
        value: 要查找的值
        
    返回:
        第一个匹配值的索引，如果没有找到则返回-1
    """
    # 使用where找到所有匹配值的索引
    indices = np.where(arr == value)[0]
    
    # 返回第一个索引，如果没有找到则返回-1
    return indices[0] if len(indices) > 0 else -1

def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union



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

system_prompt_lens = (
    r"You are a helpful assistant. Find the object that best matches the description provided by the user and provide its mask."
    r"Please:\n"
    r"1. Analyze all objects in the image carefully\n"
    r"2. Compare candidates against the target description\n"
    r"3. Select the most closely matching object\n"
    r"4. Provide precise mask\n"
    r"Format your response as:\n"
    r"<think>\n"
    r"[Your step-by-step analysis and reasoning]\n"
    r"</think>\n"
    r"<answer>\n"
    r"[Your final answer]\n"
    r"</answer>\n"
)

system_prompt = (
    r"A conversation between User and Assistant. The user asks a question, and the Assistant solves it."
    r"The assistant first thinks about the reasoning process in the mind and then provides the user"
    r"with the answer. The reasoning process and answer are enclosed within <think> </think> and"
    r"<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>"
    r"<answer> answer here </answer>."
)


def main():

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    dataset = RefCocoDataset(
        data_root='./data/ref_seg/refcoco',
        data_prefix=dict(img_path='coco2014/train2014/'),
        pipeline=None,
        ann_file='instances.json',
        split_file='refs(unc).p',
    )
    temp_save_root = "./temp_rl_data_256x2_1002/ref_seg/refcoco"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    dataset_name = "refcoco"

    # dataset = RefCocoDataset(
    #     data_root='./data/ref_seg/refcoco+',
    #     data_prefix=dict(img_path='coco2014/train2014/'),
    #     pipeline=None,
    #     ann_file='instances.json',
    #     split_file='refs(unc).p',
    # )
    # temp_save_root = "./temp_rl_data_256x2_1002/ref_seg/refcoco+"
    # if not os.path.exists(temp_save_root):
    #     os.makedirs(temp_save_root)
    # dataset_name = "refcocop"

    # dataset = RefCocoDataset(
    #     data_root='./data/ref_seg/refcocog',
    #     data_prefix=dict(img_path='coco2014/train2014/'),
    #     pipeline=None,
    #     ann_file='instances.json',
    #     split_file='refs(umd).p',
    # )
    # temp_save_root = "./temp_rl_data_256x2_1002/ref_seg/refcocog"
    # if not os.path.exists(temp_save_root):
    #     os.makedirs(temp_save_root)
    # dataset_name = "refcocog"

    # dataset = RefCocoDataset(
    #     data_root='./data/ref_seg/refclef',
    #     data_prefix=dict(img_path='saiapr_tc-12/'),
    #     pipeline=None,
    #     ann_file='instances.json',
    #     split_file='refs(unc).p',
    # )
    # temp_save_root = "./temp_rl_data_256x2_1002/ref_seg/refclef"
    # if not os.path.exists(temp_save_root):
    #     os.makedirs(temp_save_root)
    # dataset_name = "refclef"


    # build qwen25vl model
    model_path = "work_dirs/qwen25vl_3b_t2m_m2t_v2/hf_ckpt100k"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(model_path)

    
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

    for index in tqdm.tqdm(range(len(dataset))):
        data_dict = dataset.prepare_data(index)
       
        image_path = data_dict['img_path']

        image_file = os.path.basename(image_path)
        if '.jpg' in image_file:
            image_id = image_file.split('.jpg')[0]
        elif '.png' in image_file:
            image_id = image_file.split('.png')[0]
        else:
            raise ValueError(f"Unsupported image format: {image_file}")
        
        if os.path.exists(os.path.join(temp_save_root, f"{image_id}.json")):
            continue

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        instances, text = data_dict['instances'], data_dict['text']

        # process masks

        scaned_phrases = []
        
        turn_idx = 0
        for idx, inst in enumerate(instances):
            phrase = text[idx].lower()
            if '.' == phrase[-1]:
                phrase = phrase[:-1]
            if '|' in phrase or len(phrase) <= 1:
                continue
            if phrase in scaned_phrases:
                continue
            
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
            for seg in inst['mask']:
                try:
                    rles = mask_utils.frPyObjects([seg], ori_height, ori_width)
                    m = mask_utils.decode(rles)
                    m = m.astype(np.uint8)
                    binary_mask += m.squeeze()
                except:
                    m = decode_mask([seg], ori_height, ori_width)
                    binary_mask += m[0]
           
            # output_image = visualize(image, binary_mask[np.newaxis, :, :], [""])
            # output_image.save('./refgta_ins.jpg')
            # print("===========>phrase: ", phrase)
            # exit(0)


            # construct rl prompt
            question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text", "text": system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=1024,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            
            quant_ids = extract_mt_token_ids(output_text[0])
        
            batch_size = 1
            remap_quant_ids = np.array([-1 for _ in range(2)])
            for quant_id in quant_ids:
                depth_idx = quant_id // 256
                remap_quant_ids[depth_idx] = quant_id % 256
            truncated_idx = find_first_index(remap_quant_ids, -1)
            if truncated_idx != -1:
                remap_quant_ids[truncated_idx:] = -1
            quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device).unsqueeze(0)
            _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
            _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
            _pred_masks = _pred_masks > 0.5

            iou = mask_iou(torch.from_numpy(binary_mask).unsqueeze(0), _pred_masks[:, 0, :, :].cpu())
            if iou[0][0].item() > 0.9:
                print("=====>iou: ", iou[0][0].item())
                continue

    
            masks = [binary_mask]
            masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])

            boxes = torchvision.ops.masks_to_boxes(masks)
            whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
            boxes = boxes / whwh
            boxes = boxes.to(vq_sam2.device)
            masks = [masks.to(vq_sam2.device)]
            
            with torch.no_grad():
                vq_sam2_output = vq_sam2(
                    sam2_pixel_values,
                    masks,
                    boxes,
                    reconstruct_mask=False,
                )
        
            quant_codes = vq_sam2_output.quant_codes.squeeze().cpu().numpy().astype(np.int32).tolist()

            generated_quant_codes = quant_ids.squeeze().cpu().numpy().astype(np.int32).tolist()
            reconstructed_quant_codes = quant_codes
            if generated_quant_codes[0] == reconstructed_quant_codes[0] and generated_quant_codes[1] == reconstructed_quant_codes[1]:
                print("=====>match with reconstructed mask.")
                continue

            remap_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(quant_codes)]
            quant_codes = remap_quant_codes
 
            sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes]) + MT_END_TOKEN
            answer = "```json\n[{mask_2d}]\n```"
            item_str = "{\"mask_2d\": " + sam2_tokens + ", \"label\": \"" + phrase + "\"}"
            answer = answer.format(mask_2d=item_str)

            rle = mask_utils.encode(np.array(binary_mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")

            ret_data_dict = {
                'image': image_path,
                'question': question,
                'answer': answer,
                'source': 'refcoco',
                'segmentation': [rle],
                'mask_tokens': sam2_tokens,
            }
            scaned_phrases.append(phrase)

            shard_items.append(ret_data_dict)
            count += 1

            if count % shard_size == 0:
                shard_idx += 1
                out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}.json")
                with open(out_path, "w") as f:
                    json.dump(shard_items, f)
                shard_items.clear()
                print(f"[SAVE] {out_path} ({count} items)", flush=True)

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




