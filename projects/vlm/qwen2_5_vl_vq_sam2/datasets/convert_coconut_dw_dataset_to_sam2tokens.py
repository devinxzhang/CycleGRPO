import os
import json
import copy
import tqdm
import re

import torch
import torchvision
import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image

from xtuner.model.utils import guess_load_checkpoint

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize

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

def clear_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    import gc
    gc.collect()

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='COCONUT-DW')
    parser.add_argument('--task_id', '--task-id', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    temp_save_root = "./temp_data_256x2_0927/coconut_dw"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    dataset_name = "coconut_dw"

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

    with open('./data/tyfeld/coconut_dw.json', 'r') as f:
        all_data_dict = json.load(f)
    
    rows = len(all_data_dict)
    chunk_size = (rows+7) // 8
    _start_ = args.task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = rows if _end_ > rows else _end_

    count = 0
    shard_size = 1000
    shard_items = []
    shard_idx = 0

    for data_dict in tqdm.tqdm(all_data_dict[_start_:_end_]):
        image_path = data_dict['image']

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        mask_annotation = data_dict['mask_annotation']
        skip_this_case = False
        new_mask_annotation = {}
        for mask_id, mask_anno in mask_annotation.items():
            rle = mask_anno['rle']

            masks = decode_mask([rle], ori_height, ori_width)
            masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            try:
                boxes = torchvision.ops.masks_to_boxes(masks)
            except:
                print("Error at boxes = torchvision.ops.masks_to_boxes(masks)")
                skip_this_case = True
                break
            whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
            boxes = boxes / whwh
            boxes = boxes.to(vq_sam2.device)
            masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
            num_ins = len(masks)

            with torch.no_grad():
                vq_sam2_output = vq_sam2(
                    sam2_pixel_values,
                    masks,
                    boxes,
                    reconstruct_mask=False,
                )
                quant_codes = vq_sam2_output.quant_codes.detach().squeeze().cpu().numpy().astype(np.int32).tolist()
                remap_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(quant_codes)]
                quant_codes = remap_quant_codes
                sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes]) + MT_END_TOKEN
                new_mask_anno = copy.deepcopy(mask_anno)
                new_mask_anno.update({'mask_token': sam2_tokens})
                new_mask_annotation[mask_id] = new_mask_anno
        if skip_this_case:
            continue
        
        ret_data_dict = copy.deepcopy(data_dict)
        ret_data_dict.update({'mask_annotation': new_mask_annotation})

        shard_items.append(ret_data_dict)
        count += 1

        if count % shard_size == 0:
            shard_idx += 1
            out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-chunk{args.task_id}-{shard_idx:05d}.json")
            with open(out_path, "w") as f:
                json.dump(shard_items, f)
            shard_items.clear()
            print(f"[SAVE] {out_path} ({count} items)", flush=True)
    
    # 收尾
    if shard_items:
        shard_idx += 1
        out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{args.task_id}-{shard_idx:05d}.json")
        with open(out_path, "w") as f:
            json.dump(shard_items, f)
        shard_items.clear()
        print(f"[SAVE] {out_path} (final, total={count})", flush=True) 
            
if __name__ == "__main__":
    # main()

    all_data_dict = []
    for json_file in os.listdir("./temp_data_256x2_0927/coconut_dw"):
        if not json_file.endswith('.json'):
            continue
        json_path = os.path.join("./temp_data_256x2_0927/coconut_dw", json_file)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        all_data_dict.extend(json_data)
    
    with open(f"./data/coconut_dw_source.json", 'w') as f:
        json.dump(all_data_dict, f, indent=4)

