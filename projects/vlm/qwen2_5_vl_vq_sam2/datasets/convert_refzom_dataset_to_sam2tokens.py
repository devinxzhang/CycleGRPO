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


import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import pdb
import copy
from random import choice
from textblob import TextBlob

from projects.vlm.qwen2_5_vl_vq_sam2.datasets.refzom_refer import REFER
import copy
import random
import torch
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import random


class Referzom_Dataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)
        self.dataset_type = args.dataset
        self.max_tokens = 20
        ref_ids = self.refer.getRefIds(split=self.split)
        self.img_ids = self.refer.getImgIds()

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in self.img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.input_ids_masked = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.eval_mode = eval_mode

        self.zero_sent_id_list = []
        self.one_sent_id_list = []
        self.all_sent_id_list = []
        self.sent_2_refid = {}
        for r in ref_ids:
            ref = self.refer.loadRefs(r)

            source_type = ref[0]['source']

            for sent_dict in ref[0]['sentences']:
                sent_id = sent_dict['sent_id']

                self.sent_2_refid[sent_id] = r
                self.all_sent_id_list.append(sent_id)
                if source_type=='zero':
                    self.zero_sent_id_list.append(sent_id)
                else:
                    self.one_sent_id_list.append(sent_id)

        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            sentences_for_ref_masked = []
            attentions_for_ref = []

            for i, el in enumerate(ref['sentences']):
                sentence_raw = el['raw']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens
                padded_input_ids_masked = [0] * self.max_tokens

                blob = TextBlob(sentence_raw.lower())
                chara_list = blob.tags
                mask_ops = []
                mask_ops1 = []
                for word_i, (word_now, chara) in enumerate(chara_list):
                    if (chara == 'NN' or chara == 'NNS') and word_i < 19 and word_now.lower():
                        mask_ops.append(word_i)
                        mask_ops1.append(word_now)
                mask_ops2 = self.get_adjacent_word(mask_ops)


                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)
                if len(mask_ops) == 0:
                    attention_remask = attention_mask
                    input_ids_masked = input_ids
                else:
                    could_mask = choice(mask_ops2)
                    input_ids_masked = copy.deepcopy(input_ids)
                    for i in could_mask:
                        input_ids_masked[i + 1] = 0
                padded_input_ids_masked[:len(input_ids_masked)] = input_ids_masked

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                sentences_for_ref_masked.append(torch.tensor(padded_input_ids_masked).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.extend(sentences_for_ref)
            self.input_ids_masked.extend(sentences_for_ref_masked)
            self.attention_masks.extend(attentions_for_ref)


    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.all_sent_id_list)
    
    def get_adjacent_word(self, mask_list):
        output_mask_list = []
        length = len(mask_list)
        i = 0
        while i < length:
            begin_pos = i
            while i+1 < length and mask_list[i+1] == mask_list[i] + 1:
                i += 1
            end_pos = i+1
            output_mask_list.append(mask_list[begin_pos:end_pos])
            i = end_pos

        return output_mask_list

    def __getitem__(self, index):
        
        sent_id = self.all_sent_id_list[index]
        this_ref_id = self.sent_2_refid[sent_id]

        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)
        if self.dataset_type == 'ref-zom':
            source_type = ref[0]['source']
        else:
            source_type = 'not_zero'

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])

        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")


        if self.image_transforms is not None:

            if self.split == 'train':
                img, target = self.image_transforms(img, annot)
            elif self.split == 'val':
                img, target = self.image_transforms(img, annot)
            else:
                img, target = self.image_transforms(img, annot)

        if self.eval_mode:
            embedding = []
            embedding_masked = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                # e1 = self.input_ids_masked[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                embedding_masked.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))
            
            tensor_embeddings = torch.cat(embedding, dim=-1)
            tensor_embeddings_masked = torch.cat(embedding_masked, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            tensor_embeddings_masked = self.input_ids_masked[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]

        return img, target, source_type, tensor_embeddings, tensor_embeddings_masked, attention_mask




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


def main():

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'


    refer = REFER('ref-zom', 'final')
    ref_ids = refer.getRefIds(split='train')
    img_ids = refer.getImgIds()
    all_imgs = refer.Imgs
    imgs = list(all_imgs[i] for i in img_ids)

    

    sent_id = self.all_sent_id_list[index]
    this_ref_id = self.sent_2_refid[sent_id]

    this_img_id = self.refer.getImgIds(this_ref_id)
    this_img = self.refer.Imgs[this_img_id[0]]

    img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

    ref = self.refer.loadRefs(this_ref_id)
    if self.dataset_type == 'ref-zom':
        source_type = ref[0]['source']
    else:
        source_type = 'not_zero'

    ref_mask = np.array(self.refer.getMask(ref[0])['mask'])

    annot = np.zeros(ref_mask.shape)
    annot[ref_mask == 1] = 1


   
    dataset = RefCocoDataset(
        data_root='./data/ref_seg/refzom',
        data_prefix=dict(img_path='coco2014/train2014/'),
        pipeline=None,
        ann_file='instances.json',
        split_file='Ref_ZOM.p',
    )
    temp_save_root = "./temp_data_256x4_0919/ref_seg/refzom"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    dataset_name = "refzom"

    sam2_config = SAM2Config(
        cfg_path="sam2.1_hiera_l.yaml",
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
    )

    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 4
    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=256,
    )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    pretrained_pth = "pretrained_weights/iter_17923_resampled_256x4.pth"
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
        
        turn_idx = 0
        for idx, inst in enumerate(instances):
            phrase = text[idx].lower()
            if '.' == phrase[-1]:
                phrase = phrase[:-1]
            

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
           
            output_image = visualize(image, binary_mask[np.newaxis, :, :], [""])
            output_image.save('./refzom_ins.jpg')
            print("===========>phrase: ", phrase)
            exit(0)
        
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

            remap_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(quant_codes)]
            quant_codes = remap_quant_codes
            
            # # verify the quality of the quant_codes
            # pred_masks = vq_sam2_output.pred_masks
            # pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
            # pred_masks = pred_masks > 0.5
            # pred_masks = pred_masks[0].cpu().numpy().astype(np.uint8)
            # target_mask = masks[0].cpu().numpy().astype(np.uint8)

            # iou = mask_iou(torch.from_numpy(target_mask), torch.from_numpy(pred_masks))
            # if iou[0][0].item() < 0.5:
            #     print('skip this one')
            #     continue
            
            question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
            if turn_idx == 0:
                question = "<image>\n" + question

            sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes]) + MT_END_TOKEN
            # answer = random.choice(ANSWER_LIST).format(SEG=sam2_tokens)
            answer = "```json\n[{mask_2d}]\n```"
            item_str = "{\"mask_2d\": " + sam2_tokens + ", \"label\": \"" + phrase + "\"}"
            answer = answer.format(mask_2d=item_str)

            conversation = []
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': answer})
            # turn_idx += 1

            rle = mask_utils.encode(np.array(binary_mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            ret_data_dict = {
                'image': image_path,
                'conversations': conversation,
                'segmentation': [rle],
                'segmentation_image_indices': [0],
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




