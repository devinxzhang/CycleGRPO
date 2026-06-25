import os
import argparse
import hydra
import json
import tqdm
from pycocotools import mask as mask_utils
from PIL import Image

import numpy as np
import torch
import torchvision

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.tokenmask.models import DirectResize

def parse_args():
    parser = argparse.ArgumentParser(description='GCG')
    parser.add_argument(
        '--vq_sam2_path',
        default="pretrained_weights/vq_sam2_2M/iter_44916.pth",
        help='vq-sam2 model path.')
    args = parser.parse_args()
    return args


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

def main():
    args = parse_args()

    # build vq-sam2 model
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 4
    with hydra.initialize(version_base=None, config_path="../../../../../projects/transformers/vq_sam2/sam2/sam2_configs"):
        sam2_config = SAM2Config(
            cfg_path="sam2.1_hiera_l.yaml",
            ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
        )
        
        vq_sam2_config = VQ_SAM2Config(
            sam2_config=sam2_config,
            codebook_size=CODEBOOK_SIZE,
            codebook_depth=CODEBOOK_DEPTH,
            shared_codebook=False,
            latent_dim=256,
        )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    state = torch.load(args.vq_sam2_path, map_location="cpu")
    vq_sam2.load_state_dict(state)

    sam2_image_processor = DirectResize(1024)

    with open("./data/entitysef_val_lr.json", 'r') as f:
        dataset = json.load(f)

    images = dataset['images']
    annotations = dataset['annotations']

    image_dict = {item['id']: item['file_name'] for item in images}

    all_iou = []
    for data_dict in tqdm.tqdm(annotations):
        segm = data_dict['segmentation']
        image_id = data_dict['image_id']
        image_file = image_dict[image_id]
        image_path = os.path.join("<PATH_TO_DATA>/segmentation_datasets/HQSeg/entity_lr", image_file)

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size
        binary_masks = decode_mask([segm], ori_height, ori_width)

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])
        boxes = torchvision.ops.masks_to_boxes(masks)
        whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
        boxes = boxes / whwh
        boxes = boxes.to(vq_sam2.device)
        masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
        
        with torch.no_grad():
            vq_sam2_output = vq_sam2(
                sam2_pixel_values,
                masks,
                boxes,
                reconstruct_mask=True,
            )
        
        pred_masks = vq_sam2_output.pred_masks
        pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
        pred_masks = pred_masks > 0.5
        pred_masks = pred_masks[0].cpu().numpy().astype(np.uint8)
        target_mask = masks[0].cpu().numpy().astype(np.uint8)
        iou = mask_iou(torch.from_numpy(target_mask), torch.from_numpy(pred_masks))
        all_iou.append(iou)
    
    print("rMaskIoU: ", sum(all_iou) / len(all_iou))
    
if __name__ == '__main__':
    main()