import os
import json
import tqdm
from pycocotools import mask as mask_utils
from PIL import Image

import numpy as np

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
    if len(binary_masks) == 0:
        binary_masks.append(np.zeros((ori_height, ori_width), dtype=np.uint8))
    masks = np.stack(binary_masks, axis=0)
    return masks

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

def main():
    image_root = "<PATH_TO_DATA>/CityScapes/leftImg8bit/train"
    anno_root = "<PATH_TO_DATA>/CityScapes/gtFine/train"

    dataset_name = 'cityscape'

    temp_save_root = "./any_other_seg_data"
    os.makedirs(temp_save_root, exist_ok=True)

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    for split_name in os.listdir(image_root):
        for image_file in os.listdir(os.path.join(image_root, split_name)):
            anno_file = image_file.replace("_leftImg8bit.png", "_gtFine_polygons.json")
            image_path = os.path.join(image_root, split_name, image_file)
            anno_path = os.path.join(anno_root, split_name, anno_file)

            with open(anno_path, 'r') as f:
                anno_data = json.load(f)
            ori_height = anno_data["imgHeight"]
            ori_width = anno_data["imgWidth"]

            segms, class_names = [], []
            for obj in anno_data['objects']:
                if obj['label'] in ['building', 'sky', "out of roi"]:
                    continue
                polygon = np.array(obj['polygon'])
                if len(polygon.shape) == 2:
                    segms.append([polygon.flatten().tolist()])
                else:
                    segms.append([_polygon_.flatten().tolist() for _polygon_ in polygon])
                class_names.append(obj['label'])
            
            masks = decode_mask(segms, ori_height, ori_width)

            # image = Image.open(image_path).convert('RGB')

            # # output_image = visualize(image, masks, class_names)
            # # output_image.save('cityscape_visualize.jpg')
            # # exit(0)

            for bin_mask in masks:
                bin_mask = bin_mask.astype(np.bool)
                if np.sum(bin_mask) == 0:
                    continue

                try:
                    assert len(bin_mask.shape) == 2
                    rle = encode_binary_mask(bin_mask.astype(np.bool))
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
                    print(f"[ERROR] ...", flush=True)
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
    main()