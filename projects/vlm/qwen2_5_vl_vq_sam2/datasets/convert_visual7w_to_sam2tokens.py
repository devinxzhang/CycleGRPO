import sys
import torch
import torchvision
import copy
from PIL import Image
import numpy as np
import os
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from projects.transformers.vq_sam2.sam2.build_sam import build_sam2_ori
from sam2.sam2_image_predictor import SAM2ImagePredictor

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize
from projects.vlm.vq_sam2.datasets import CoCoPanoSegValDataset, SA1BValDataset
from projects.vlm.vq_sam2.datasets.coco_category import COCO_CATEGORIES

from transformers import AutoProcessor

from xtuner.model.utils import guess_load_checkpoint


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
    
    alpha = 0.4

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

def xywh_to_x1y1x2y2(bboxes):
    converted = bboxes.copy()
    converted[:, 2] = bboxes[:, 0] + bboxes[:, 2] 
    converted[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return converted

def sort_mask_indices(boxes: torch.Tensor, mode: str = "ltr-ttb") -> np.ndarray:
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
    # boxes = torchvision.ops.masks_to_boxes(masks_t)  # [N,4] (x1,y1,x2,y2)
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


QUESTION_CLASS_LIST = [
    "Provide the mask(s) for all {CAT_INFO} in the image. If masks cannot be generated, supply the bounding box(es) for each {CAT_INFO} instead. Output results in JSON.",
    "Ground all {CAT_INFO} in the photo: prioritize mask output for each {CAT_INFO}. If masks are unavailable, return the bounding box for every {CAT_INFO}. Output results in JSON.",

]

QUESTION_DETAIL_LIST = [
    "For the object described as \"{CAT_INFO}\", first generate its mask. If mask creation is not feasible, provide the corresponding bounding box. Output results in JSON.",
    "Locate the object matching the description \"{CAT_INFO}\" — first produce its mask. If mask generation fails, provide the bounding box. Output results in JSON.",
]

QUESTION_BBOX_LIST = [

]

def main():
    sam2_checkpoint = "pretrained_weights/sam2.1_hiera_large.pt"
    model_cfg = "sam2.1_hiera_l.yaml"

    sam2_model = build_sam2_ori(model_cfg, sam2_checkpoint, device='cuda')

    predictor = SAM2ImagePredictor(sam2_model)

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    temp_save_root = "./temp_data_256x2_0927/ref_seg/visual7w"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    
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

    data_path = "<PATH_TO_DATA>/discrete_spatial_tokenizer/data/visual7w/dataset_v7w_pointing.json"
    dataset_name = 'visual7w'

    with open(data_path, 'r') as f:
        anno_data = json.load(f)

    id2box = {}
    for bbox_item in anno_data["boxes"]:
        x, y, height, width = bbox_item['x'], bbox_item['y'], bbox_item['height'], bbox_item['width']
        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height
        id2box[bbox_item['box_id']] = [x1, y1, x2, y2]

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    skip_count = 0

    for image_item in tqdm(anno_data['images']):
        image_file = os.path.join('./data/visual7w/images', image_item['filename'])
        image = Image.open(image_file).convert('RGB')
        ori_width, ori_height = image.size

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
        for qa_item in image_item['qa_pairs']:
            if count < skip_count:
                count += 1
                continue
            qa_id = qa_item['qa_id']
            question = qa_item['question']
            bbox_id = qa_item['answer']
            x1, y1, x2, y2 = id2box[bbox_id]

            x1 = x1 if x1 > 0 else 0
            y1 = y1 if y1 > 0 else 0
            x2 = x2 if x2 < ori_width else ori_width
            y2 = y2 if y2 < ori_height else ori_height

            bbox_x1y1x2y2_np = np.array([x1, y1, x2, y2])

            sam_image = np.array(image)
            predictor.set_image(sam_image)
            sam_masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox_x1y1x2y2_np,
                multimask_output=False,
            ) # (batch_size) x (num_predicted_masks_per_input) x H x W

            try:
                sam_masks = sam_masks[:, 0, :, :]
            except:
                # print("sam_masks.shape: ", sam_masks.shape)
                # output_image = visualize(image, sam_masks, ['']*len(sam_masks))
                # output_image.save("sam2_v3det_box2mask.jpg")
                sam_masks = sam_masks

            # output_image = visualize(image, sam_masks, [""])
            # output_image.save('visual7w_bbox2mask.jpg')
            # print("============>question: ", question)
            # exit(0)

            

            masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in sam_masks])
            try:
                boxes = torchvision.ops.masks_to_boxes(masks)
            except:
                continue
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

            question = "<image>\n" + question + ' Please respond with segmentation mask.'
            
            sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes]) + MT_END_TOKEN
            answer = sam2_tokens

            conversation = []
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': answer})
            # turn_idx += 1

            rle = mask_utils.encode(np.array(sam_masks[0, :, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            ret_data_dict = {
                'image': image_file,
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
