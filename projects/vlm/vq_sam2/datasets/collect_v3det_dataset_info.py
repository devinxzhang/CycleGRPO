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

def main(task_id):
    sam2_checkpoint = "pretrained_weights/sam2.1_hiera_large.pt"
    model_cfg = "sam2.1_hiera_l.yaml"

    sam2_model = build_sam2_ori(model_cfg, sam2_checkpoint, device='cuda')

    predictor = SAM2ImagePredictor(sam2_model)


    temp_save_root = "./any_other_seg_data"
    os.makedirs(temp_save_root, exist_ok=True)

    dataset_name = "v3det"

    with open('./data/v3det_bbox_info.json', 'r') as f:
        v3det_dataset = json.load(f)

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0
    
    chunk_idx = task_id
    n = len(v3det_dataset)
    chunk_size = (n+7) // 8
    start = chunk_idx * chunk_size
    end = min((chunk_idx + 1) * chunk_size, n)
    indices_list = list(range(len(v3det_dataset)))[start:end]
    for idx in tqdm(indices_list):
        data_dict = v3det_dataset[idx]
        image_file = data_dict['file_name']
        image_id = os.path.basename(image_file).split('.')[0]
        image = Image.open(image_file).convert('RGB')

        bbox_list = []
        class_id_list = []
        for anno in data_dict['annotations']:
            bbox_list.append(anno["bbox"])
            class_id_list.append(anno["category_id"])
        bbox_xywh_np = np.array(bbox_list)
        bbox_x1y1x2y2_np = xywh_to_x1y1x2y2(bbox_xywh_np)

        if len(bbox_x1y1x2y2_np) == 0:
            continue

        sam_image = np.array(image)
        predictor.set_image(sam_image)
        try:
            sam_masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox_x1y1x2y2_np,
                multimask_output=False,
            ) # (batch_size) x (num_predicted_masks_per_input) x H x W
        except torch.OutOfMemoryError:
            print("Encounter torch.OutOfMemoryError")
            continue

        try:
            sam_masks = sam_masks[:, 0, :, :]
            scores = scores[:, 0]
        except:
            # print("sam_masks.shape: ", sam_masks.shape)
            # output_image = visualize(image, sam_masks, ['']*len(sam_masks))
            # output_image.save("sam2_v3det_box2mask.jpg")
            pass
        
        for sam_mask, _score_ in zip(sam_masks, scores):
            if _score_ < 0.7:
                continue
            try:
                rle = encode_binary_mask(sam_mask.astype(np.bool))
                if rle is None:
                    # 空实例，跳过但记录
                    # print(f"[WARN] empty mask seg_id={seg_id} file={image_file}", flush=True)
                    continue

                shard_items.append({
                    "image_file": image_file,
                    "segmentation": rle,
                })
                count += 1

                if count % shard_size == 0:
                    shard_idx += 1
                    out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}-task{task_id}.json")
                    with open(out_path, "w") as f:
                        json.dump(shard_items, f)
                    shard_items.clear()
                    print(f"[SAVE] {out_path} ({count} items)", flush=True)

            except Exception as e:
                # 如果 pycocotools 在 C 层崩溃，这里是抓不到的；但大多数数据问题能在这儿被捕到
                print(f"[ERROR]..........", flush=True)
                continue
    
    # 收尾
    if shard_items:
        shard_idx += 1
        out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}-task{task_id}.json")
        with open(out_path, "w") as f:
            json.dump(shard_items, f)
        shard_items.clear()
        print(f"[SAVE] {out_path} (final, total={count})", flush=True) 

if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)
