import os
import json
import tqdm
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import io
from PIL import Image
import numpy as np

from pycocotools import mask as mask_utils

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


# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag

np.random.seed(42)

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

def main():
    from panopticapi.utils import rgb2id

    coconut_dw = "./data/tyfeld/coconut_dw"
    count = 0
    for parquet_file in os.listdir(coconut_dw):
        if not parquet_file.endswith('.parquet'):
            continue
        parquet_path = os.path.join(coconut_dw, parquet_file)
        parquet_f = pq.ParquetFile(parquet_path)
        data = parquet_f.read().to_pandas()
        rows = data.shape[0]

        subset_items = []
        
        for _, row in data.iterrows():
            # dict_keys(['mask', 'segments_info', 'image_info', 'image_caption', 'image'])
            row_dict = row.to_dict()
            image_info = row_dict['image_info']
            image_file = image_info['file_name']
            image_path = os.path.join("./data/coco/train2017", image_file)
            if not os.path.exists(image_path):
                image_path = os.path.join("./data/coco/unlabeled2017", image_file)
                if not os.path.exists(image_path):
                    print(image_path, "is not found!!!")
                    continue

            mask_img = Image.open(io.BytesIO(row_dict['mask']['bytes']))
            mask_img = _apply_exif_orientation(mask_img)
            pan_seg_gt = convert_PIL_to_numpy(mask_img, "RGB")

            segments_info = row_dict['segments_info']['segments_info'].tolist()
            pan_seg_gt = rgb2id(pan_seg_gt)

            mask_dict = {}
            for segment_info in segments_info:
                mask = pan_seg_gt == segment_info["id"]
                mask_id = segment_info['obj_render_id']
                rle = mask_utils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")
                mask_dict[mask_id] = {'rle': rle, 'category_name': segment_info['category_name'], 'caption': segment_info['caption']}
            
            ret_data_dict = {
                'image_id': image_info['id'],
                'image': image_path,
                'mask_annotation': mask_dict,
                'image_caption': row_dict['image_caption']
            }

            subset_items.append(ret_data_dict)
        
        with open(os.path.join("./data/tyfeld/coconut_dw_decoded", parquet_file.replace('.parquet', '.json')), 'w') as f:
            json.dump(subset_items, f)
    
    all_items = []
    for json_file in os.listdir("./data/tyfeld/coconut_dw_decoded"):
        if not json_file.endswith('.json'):
            continue
        with open(os.path.join("./data/tyfeld/coconut_dw_decoded", json_file), 'r') as f:
            subset = json.load(f)
            all_items.extend(subset)
    with open("./data/tyfeld/coconut_dw.json", 'w') as f:
        json.dump(all_items, f, indent=4)


if __name__ == "__main__":
    main()