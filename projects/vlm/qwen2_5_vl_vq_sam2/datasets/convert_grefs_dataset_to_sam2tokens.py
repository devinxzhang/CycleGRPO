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

def sort_mask_indices(masks_t: torch.Tensor, mode: str = "ltr-ttb") -> np.ndarray:
    """
    根据实例的几何位置给出排序索引。
    Args:
        masks_t: [N, H, W] 的 torch.bool/uint8 张量（每个实例一个二值mask）
        mode:
            - "ltr-ttb": left-to-right, then top-to-bottom（先按x中心，再按y中心）
            - "ttb-ltr": top-to-bottom, then left-to-right（先按y中心，再按x中心）
            - "tlbr":    purely by top-left (y1, x1) 先y后x（行优先）
    Returns:
        order: numpy 数组，形状 [N]，是重排索引
    """
    # 利用bbox/中心点作为排序依据（更稳定、无需遍历像素）
    boxes = torchvision.ops.masks_to_boxes(masks_t)  # [N,4] (x1,y1,x2,y2)
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    xc = ((x1 + x2) * 0.5).cpu().numpy()
    yc = ((y1 + y2) * 0.5).cpu().numpy()
    y1n = y1.cpu().numpy()
    x1n = x1.cpu().numpy()

    if mode == "ltr-ttb":
        # 先x后y：主键x_center，次键y_center
        order = np.lexsort((yc, xc))
    elif mode == "ttb-ltr":
        # 先y后x：主键y_center，次键x_center
        order = np.lexsort((xc, yc))
    elif mode == "tlbr":
        # 以bbox左上角先y后x（更像“逐行”）
        order = np.lexsort((x1n, y1n))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return order

def main():

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    dataset = GRefCoCoDataset(
        data_root='./data/ref_seg/grefs',
        ann_file='instances.json',
        split_file='grefs(unc).json',
        data_prefix=dict(img_path='coco2014/train2014/'),
    )
    temp_save_root = "./temp_data_256x2_0927/ref_seg/grefs"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    dataset_name = "grefs"
    

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
            raise ValueError(f'Invalid image file "{image_file}".')
        
        if os.path.exists(os.path.join(temp_save_root, f"{image_id}.json")):
            continue
        
        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        instances, text, anno_ids = data_dict['instances'], data_dict['text'], data_dict['anno_ids']

        # index = np.random.choice(range(len(instances)), 3, replace=True)
        indices= list(range(len(instances)))
        
        # turn_idx = 0
        for idx in indices:
            inst = instances[idx]
            phrase = text[idx].lower()
            if '.' == phrase[-1]:
                phrase = phrase[:-1]

            # binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
            if inst["mask"] is None or inst["mask"][0] is None:
                question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
                # if turn_idx == 0:
                #     question = "<image>\n" + question
                question = "<image>\n" + question
                
                answer = random.choice(NO_TARGETS_ANSWER_LIST)

                conversation = []
                conversation.append({'from': 'human', 'value': question})
                conversation.append({'from': 'gpt', 'value': answer})
                ret_data_dict = {
                    'image': image_path,
                    'conversations': conversation,
                    'segmentation': None,
                    'segmentation_image_indices': None,
                }
                shard_items.append(ret_data_dict)
                count += 1

                if count % shard_size == 0:
                    shard_idx += 1
                    out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}.json")
                    with open(out_path, "w") as f:
                        json.dump(shard_items, f)
                    shard_items.clear()
                    print(f"[SAVE] {out_path} ({count} items)", flush=True)
                continue

            
            assert len(inst["mask"]) == len(anno_ids[idx])

            binary_masks = decode_mask(inst["mask"], ori_height, ori_width)

            # if len(binary_masks) > 1:
            #     output_image = visualize(image, binary_masks, [""]*len(binary_masks))
            #     output_image.save('./grefs_multiple_ins.jpg')
            #     exit(0)

            assert len(binary_masks) == len(inst["mask"])
            # for m in binary_masks:
            #     binary_mask += m

            masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])
            try:
                order = sort_mask_indices(masks, mode="ltr-ttb")
            except:
                order = np.arange(masks.shape[0])

            masks = masks[torch.as_tensor(order, dtype=torch.long)]

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

            question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
            question = "<image>\n" + question

            answer = "```json\n[{mask_2d}]\n```"
            mask_2d_str = ''
            for _quant_codes_ in quant_codes:
                sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes_]) + MT_END_TOKEN
                item_str = "{\"mask_2d\": " + sam2_tokens + ", \"label\": \"" + phrase + "\"}"
                mask_2d_str += item_str + ",\n"
            mask_2d_str = mask_2d_str[:-len(",\n")]
            answer = answer.format(mask_2d=mask_2d_str)

            conversation = []
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': answer})
            # turn_idx += 1

            rles = []
            for binary_mask in binary_masks:
                rle = mask_utils.encode(np.array(binary_mask[:, :, None], order="F", dtype="uint8"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")
                rles.append(rle)
        
            ret_data_dict = {
                'image': image_path,
                'conversations': conversation,
                'segmentation': rles,
                'segmentation_image_indices': [0 for _ in range(len(rles))],
            }
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






