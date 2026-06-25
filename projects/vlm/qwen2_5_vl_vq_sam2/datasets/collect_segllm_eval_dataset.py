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
import re
import uuid
from torch.utils.data import Dataset

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
    
    alpha = 0

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

all_bracket_types = ['IMAGE256', 'MASK-ENCODE', 'BOX-ENCODE', 'MASK-DECODE']
def find_brackets(x):
    matches =  re.compile('\[[^\]]+\]').findall(x)
    # filter out any bad matches
    filtered_matches = []
    for bracket in matches:
        if any([x in bracket for x in all_bracket_types]):
            filtered_matches.append(bracket)
        else:
            print("Ignoring bad bracket:", bracket)
    return filtered_matches

def get_replacement_len(s):
    if 'IMAGE256' in s:
        return 256
    # elif 'MASK-ENCODE' in s:
    #     return Constant.MASK_ENCODE_LEN
    else:
        return 1  
    
class REPLACEMENT_TYPE:
    INPUT = 0
    BASE = 1
    GEN = 2
    SEG = 3


from pycocotools.mask import decode, frPyObjects, merge
from pycocotools.coco import COCO
class MultiRoundDataset(Dataset):
    def __init__(self):
        super(MultiRoundDataset, self).__init__()

        
        with open("./data/segllm_data/annotation_folder/visual_genome/vg_masks_train_new.json", "r") as f:
            json_data = json.load(f)
        self.visual_genome = json_data

        with open("./data/segllm_data/annotation_folder/description_based_coco/seg_mask_per_instance.json", 'r') as f:
            json_data = json.load(f)
        self.description_based_coco = json_data

        self.data = {
            'refcoco': COCO('./data/segllm_data/annotation_folder/refcoco/instances.json'),
            'refcoco+': COCO('./data/segllm_data/annotation_folder/refcoco+/instances.json'),
            'refcocog': COCO('./data/segllm_data/annotation_folder/refcocog/instances.json'),
            'paco_lvis': COCO('./data/segllm_data/annotation_folder/paco_lvis_v1_val.json'),
            'lvis': COCO('./data/segllm_data/annotation_folder/lvis_v1_train.json'),
        }

    def get_bitmask(
        self,
        dataset,
        idx,
        is_eval=False,
        image_file=None,
        image_dim=None
    ):
        if dataset == "ade20k":
            anns_dir = './data/ade/ADEChallengeData2016/annotations/training'
            anns_file = image_file.replace(".jpg", ".png")
            anns_path = os.path.join(anns_dir, anns_file)
            anns_img = np.array(Image.open(anns_path))     # anns_img[x][y] = class_id

            anns_img[anns_img == 0] = 255
            anns_img -= 1
            anns_img[anns_img == 254] = 255

            class_id = idx                                 # for ade20k (semantic), mask_id will be class_id
            binary_mask = (anns_img == class_id).astype(np.uint8)

            mask = binary_mask.reshape(*binary_mask.shape, 1)
        elif dataset == "cocostuff":
            anns_dir = "./data/coco/cocostuff/train2017"
            anns_file = image_file.replace(".jpg", ".png")
            anns_path = os.path.join(anns_dir, anns_file)
            anns_img = np.array(Image.open(anns_path))    
            class_id = idx                                 
            binary_mask = (anns_img == class_id).astype(np.uint8)
            mask = binary_mask.reshape(*binary_mask.shape, 1)
        elif dataset == "visual_genome":
            mask = decode(self.visual_genome[idx])            # idx = mask_id
            mask = mask.reshape(*mask.shape, 1)
        elif dataset == "pascal":  
            raise NotImplementedError
        elif dataset == "description_based_coco":
            anno = self.description_based_coco[idx]
            seg=anno["segmentation"]
            if type(seg) == list:
                rles=frPyObjects(seg,anno["image_dim"][0],anno["image_dim"][1])
                rle=merge(rles)
            elif type(seg['counts']) == list:
                rle = frPyObjects(seg,anno["image_dim"][0],anno["image_dim"][1])
            else:
                rle=seg
            mask=decode(rle)
            mask = mask.reshape(*mask.shape,1)
        else:
            coco = self.data[dataset]
            ann = coco.loadAnns(ids=[idx])
            mask = coco.annToMask(ann[0])
            mask = mask.reshape(*mask.shape,1) # H W 1

        return mask

    def get_bitmask_bbox_encode(self, image_file_lst, image_root):
        image_file,dataset_name,mask_id = image_file_lst[0].split('|')

        # TODO: temp handle edge cases
        if mask_id == '' or mask_id == "'":         # this is the case for reason_seg sentences
            mask_id = None
        elif "_" in mask_id or "-" in mask_id:
            mask_id=mask_id
        else: 
            mask_id = int(mask_id)

        if 'VG_100K_2' in image_file:
            _image_file = os.path.basename(image_file)
            image_path = os.path.join(image_root, 'VG_100K_2', _image_file)
        elif 'VG_100K' in image_file:
            _image_file = os.path.basename(image_file)
            image_path = os.path.join(image_root, 'VG_100K', _image_file)
        else:
            _image_file = os.path.basename(image_file)
            image_path = os.path.join(image_root, _image_file)
        assert os.path.exists(image_path)

        # Mask image with GT mask
        image = Image.open(image_path)
        (w, h) = image.size
        image = np.array(image.convert('RGB'))
        gt_mask = self.get_bitmask(
            dataset_name,
            mask_id,
            is_eval=False,
            image_file=image_file.split("/")[-1], 
            image_dim=(h,w)
        )
        return gt_mask, mask_id
    
    def get_bitmask_decode(self, image_file_lst, image_root):
        #image_file,dataset_name,mask_id = image_file_lst[0].split('|')
        if ':' in image_file_lst[0]:
            # (reference mask decoding format)
            if len(re.findall(':', image_file_lst[0])) == 4:
                # 1 reference mask
                task_type,ref_mask_id,tgt_mask_id,image_file,dataset_name = image_file_lst[0].split(':')
            elif len(re.findall(':', image_file_lst[0])) == 5:
                # 2 reference masks
                task_type,ref_mask_id,ref_mask_id_2,tgt_mask_id,image_file,dataset_name = image_file_lst[0].split(':')
            else:
                raise ValueError("Base ref-mask decode format:", image_file_lst[0])
        elif '|' in image_file_lst[0]:
            # (no reference mask decoding format)
            image_file,dataset_name,tgt_mask_id = image_file_lst[0].split('|')
            task_type = 'none'
            ref_mask_id = None
        else:
            raise ValueError("Base decode format:", image_file_lst[0])
        
        # TODO: temp fix
        def process_mask_id(mask_id):
            if mask_id == '' or mask_id == "'" or mask_id == 'none' or mask_id == None:         # this is the case for reason_seg sentences
                mask_id = None
            elif "_" in mask_id or "-" in mask_id:
                mask_id=mask_id
            else: 
                mask_id = int(mask_id)
            return mask_id
        ref_mask_id = process_mask_id(ref_mask_id)
        tgt_mask_id = process_mask_id(tgt_mask_id)

        if 'VG_100K_2' in image_file:
            _image_file = os.path.basename(image_file)
            image_path = os.path.join(image_root, 'VG_100K_2', _image_file)
        elif 'VG_100K' in image_file:
            _image_file = os.path.basename(image_file)
            image_path = os.path.join(image_root, 'VG_100K', _image_file)
        else:
            _image_file = os.path.basename(image_file)
            image_path = os.path.join(image_root, _image_file)
        assert os.path.exists(image_path)

        image = Image.open(image_path)
        (w, h) = image.size
        image = np.array(image.convert('RGB'))
        tgt_mask = self.get_bitmask(
            dataset_name,
            tgt_mask_id,
            is_eval=False,                   # pass in is_eval flag, signals which seg anno file to use (train vs. eval)
            image_file=image_file.split("/")[-1], 
            image_dim=(h,w)
        ) 
        
        return tgt_mask, tgt_mask_id

    def build_query(self, x, image_root):
        data = torch.zeros(1,3,224,224)
        if image_file_lst := re.compile('IMAGE256:(.*)$').findall(x):
            image_file = image_file_lst[0]
            if 'VG_100K_2' in image_file:
                _image_file = os.path.basename(image_file)
                image_path = os.path.join(image_root, 'VG_100K_2', _image_file)
            elif 'VG_100K' in image_file:
                _image_file = os.path.basename(image_file)
                image_path = os.path.join(image_root, 'VG_100K', _image_file)
            else:
                _image_file = os.path.basename(image_file)
                image_path = os.path.join(image_root, _image_file)
            assert os.path.exists(image_path)
            return 'image-encode',image_path
        elif image_file_lst := re.compile('MASK-ENCODE:(.*)$').findall(x):
            masked_instance_processed, mask_id = self.get_bitmask_bbox_encode(image_file_lst, image_root)
            return 'mask-encode', (masked_instance_processed,mask_id)
        elif image_file_lst := re.compile('BOX-ENCODE:(.*)$').findall(x):
            # masked_instance_processed, bbox_coords_sam,mask_id = self.get_bitmask_bbox_encode(image_file_lst)
            return 'bbox-encode', None
        elif image_file_lst := re.compile('MASK-DECODE:(.*)$').findall(x):
            data, tgt_mask_id = self.get_bitmask_decode(image_file_lst, image_root)
            return 'mask-decode',(data,tgt_mask_id) # 1 1024
        else:
            raise NotImplementedError(x)

def main():

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

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

    train_file_root = "data/segllm_data/conversation_folder/all_data_mix_val"

    source_image_path = {
        'mr_refcoco_val': './data/coco/train2014',
        'mr_refcoco+_val': './data/coco/train2014',
        'mr_refcocog_val': './data/coco/train2014',
        'mr_paco_val': './data/coco/train2017',
    }

    multiround_dataset = MultiRoundDataset()

    skip_count = 0
    for source_name in source_image_path.keys():
        max_round = 10
        round_samples = {round_id+1: [] for round_id in range(max_round)}

        data_file = os.path.join(train_file_root, f"{source_name}.json")
        with open(data_file, 'r') as f:
            data_dict_list = json.load(f)

        for data_dict in tqdm.tqdm(data_dict_list):
            sources = copy.deepcopy(data_dict["conversations"])
            new_sources = []
            mask_dict = {}
            mask_id_2_unique_id = {}
            conversation_image_path = None
            skip_this_one = False
            for turn in sources:
                src = turn['from']
                val = turn['value']
            
                if src == 'human':
                    matches = find_brackets(val)
                    for prompt in matches:
                        prompt_clean = prompt[1:-1]
                        try:
                            placeholder_type, actual_value = multiround_dataset.build_query(prompt_clean, source_image_path[source_name])
                        except:
                            skip_this_one = True
                            break
                        if placeholder_type == 'image-encode':
                            conversation_image_path = actual_value
                            val = val.replace(prompt, '<image>\n', 1)
                        elif placeholder_type == 'mask-encode':
                            context_mask = actual_value[0]
                            if actual_value[1] in mask_id_2_unique_id:
                                mask_unique_uuid = mask_id_2_unique_id[actual_value[1]]
                            else:
                                # raise NotImplementedError
                                unique_uuid = uuid.uuid4()
                                mask_unique_uuid = str(unique_uuid)
                                mask_dict[mask_unique_uuid] = context_mask
                                mask_id_2_unique_id[actual_value[1]] = mask_unique_uuid
                            val = val.replace(prompt, mask_unique_uuid, 1)
                        elif placeholder_type == 'bbox-encode':
                            val = val.replace(prompt, '', 1)
                            continue
                        else:
                            raise NotImplementedError
                elif src == 'gpt':
                    matches = find_brackets(val)
                    for prompt in matches:
                        prompt_clean = prompt[1:-1]
                        try:
                            placeholder_type, actual_value = multiround_dataset.build_query(prompt_clean, source_image_path[source_name])
                        except:
                            skip_this_one = True
                            break
                        if placeholder_type == 'mask-decode':
                            result_mask = actual_value[0]
                            unique_uuid = uuid.uuid4()
                            mask_unique_uuid = str(unique_uuid)
                            mask_dict[mask_unique_uuid] = result_mask
                            val = val.replace(prompt, mask_unique_uuid, 1)
                            mask_id_2_unique_id[actual_value[1]] = mask_unique_uuid
                        else:
                            raise NotImplementedError        
                new_sources.append({'from': src, 'value': val})

            # print("========>raw conversations: ", sources)
            # print("========>current conversation: ", new_sources)

            if skip_this_one:
                print("skip this one...")
                skip_count += 1
                continue

            binary_masks = []
            mask_uuids = []
            for k, v in mask_dict.items():
                binary_masks.append(v[:, :, 0])
                mask_uuids.append(k)
            binary_masks = np.stack(binary_masks, axis=0)

            # image = Image.open(conversation_image_path).convert('RGB')
            # ori_width, ori_height = image.size
            # output_image = visualize(image, all_binary_masks, [tag[:4] for tag in list(mask_dict.keys())])
            # output_image.save('./segllm_ade20k.jpg')
            # exit(0)

            image = Image.open(conversation_image_path).convert('RGB')
            ori_width, ori_height = image.size

            sam2_image = np.array(image)
            sam2_image = sam2_image_processor.apply_image(sam2_image)
            sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
            sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

            masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])
            try:
                boxes = torchvision.ops.masks_to_boxes(masks)
            except:
                print("contain empty mask.")
                continue
            whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
            boxes = boxes / whwh
            boxes = boxes.to(vq_sam2.device)
            masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
            num_ins = len(masks)
            
            with torch.no_grad():
                try:
                    vq_sam2_output = vq_sam2(
                        sam2_pixel_values.repeat(num_ins, 1, 1, 1),
                        masks,
                        boxes,
                        reconstruct_mask=False,
                    )
                    quant_codes = vq_sam2_output.quant_codes
                except:
                    print("vq_sam2 error!!")
                    continue
            
            quant_codes = quant_codes.cpu().numpy().astype(np.int32).tolist()
            remap_quant_codes = []
            for _quant_codes in quant_codes:
                _quant_codes = _quant_codes[0]
                remap_quant_codes.append([depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes)])
            quant_codes = remap_quant_codes

            sam2_tokens_list = []
            for _quant_codes_ in quant_codes:
                sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes_]) + MT_END_TOKEN
                sam2_tokens_list.append(sam2_tokens)
            
            masktoken_dict = {}
            for k, v in zip(mask_uuids, sam2_tokens_list):
                masktoken_dict[k] = v
            
            conversations = []
            for turn in new_sources:
                src = turn['from']
                val = turn['value']

                for mask_uuid, masktoken in masktoken_dict.items():
                    if mask_uuid in val:
                        val = val.replace(mask_uuid, masktoken)
                
                conversations.append({'from': src, 'value': val})

            mask_uuid_2_rle = {}
            for mask_uuid, mask in mask_dict.items():
                rle = mask_utils.encode(np.array(mask, order="F", dtype="uint8"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")
                mask_uuid_2_rle[mask_uuid] = rle

            n_round = len(new_sources) // 2
            assert n_round * 2 == len(new_sources)
            for round_id in range(1, n_round+1):
                if round_id > max_round:
                    break
                history = conversations[:round_id*2-1]
                curr_answer = conversations[round_id*2-1]['value']
                raw_curr_answer = new_sources[round_id*2-1]['value']
                target_masks = []
                for mask_uuid, rle in mask_uuid_2_rle.items():
                    if mask_uuid in raw_curr_answer:
                        target_masks.append(rle)
                        assert masktoken_dict[mask_uuid] in curr_answer
               
                eval_item = {
                    "image": conversation_image_path,
                    "history": history,
                    "answer": curr_answer,
                    "target_masks": target_masks,
                    "round_id": round_id
                }
                round_samples[round_id].append(eval_item)
    
        with open(f"./data/segllm_data/conversation_folder/all_data_mix_val/{source_name}_sampled.json", 'w') as f:
            json.dump(round_samples, f, indent=4)
        print("Saved at ", f"./data/segllm_data/conversation_folder/all_data_mix_val/{source_name}_sampled.json")
        print("skip count: ", skip_count)
            
        
if __name__ == "__main__":
    main()

