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
import base64
from io import BytesIO

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize
from projects.vlm.vq_sam2.datasets import CoCoPanoSegValDataset, SA1BValDataset
from projects.vlm.vq_sam2.datasets.coco_category import COCO_CATEGORIES

from transformers import AutoProcessor

from xtuner.model.utils import guess_load_checkpoint

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

def image_to_base64_str(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_str = base64.b64encode(img_bytes).decode("utf-8")
    return img_str

def base64_str_to_image(img_str: str) -> Image.Image:
    img_bytes = base64.b64decode(img_str.encode("utf-8"))
    buffered = BytesIO(img_bytes)
    img = Image.open(buffered).convert("RGB")
    return img

def scandir_generator(path):
    with os.scandir(path) as entries:
        for entry in entries:
            yield entry.name  # 逐个返回文件名，不占用大量内存



def main(task_id):

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    temp_save_root = "./temp_data/visual_mask_text_mask_alignment/"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    
    sam2_config = SAM2Config(
        cfg_path="sam2.1_hiera_l.yaml",
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
    )

    CODEBOOK_SIZE = 1024
    CODEBOOK_DEPTH = 4
    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=1024,
        codebook_depth=4,
        shared_codebook=False,
        latent_dim=256,
    )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    pretrained_pth = "pretrained_weights/iter_17923.pth"
    pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

    pretrained_state_dict_new = {}
    for key in pretrained_state_dict.keys():
        new_key = copy.deepcopy(key)
        if key.startswith('hf_model.'):
            new_key = new_key[len('hf_model.'):]
        pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
    
    vq_sam2.load_state_dict(pretrained_state_dict_new)

    sam2_image_processor=dict(
        type=DirectResize,
        target_length=1024,
    )

    DATA_ROOT = ''
    sam_info_json = "./data/sam_info.json"

    val_dataset = SA1BValDataset(
        image_folder='',
        preprocessor=sam2_image_processor,
        multi_targets=False,
        repeats=1.0,
        fast_load=True,
        sam_info_json=sam_info_json,
        scan_record_folder='./left_sa1b_indices/vq_sam2_codebookx4depthx1024sizex256dimxunsharex1MT_datasetxsa1bx10xcoconutx10xentityx10xpixelwebx10/'
    )

    chunk_idx = task_id
    n = len(val_dataset)
    chunk_size = (n+31) // 32
    start = chunk_idx * chunk_size
    end = min((chunk_idx + 1) * chunk_size, n)
    indices_list = list(range(len(val_dataset)))[start:end]
    for idx in tqdm(indices_list):
        data = val_dataset[idx]
        image_file = data['image_file']
        image_name = os.path.basename(image_file).split('.jpg')[0]
        masks = data['masks']
        rles = data['segms']
        with Image.open(image_file) as image:
            width, height = image.size

        if os.path.exists(os.path.join(temp_save_root, f"{image_name}.json")):
            continue

        val_item = val_dataset.prepare_mask_input(image_file, masks)
        pixel_values = val_item['pixel_values']
        masks = val_item['masks']
        boxes = val_item['boxes'].to(vq_sam2.device)
        num_ins = len(boxes)

        pixel_values = pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device).repeat(num_ins, 1, 1, 1)
        masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]

        # try:
        #     with torch.no_grad():
        #         vq_sam2_output = vq_sam2(
        #             pixel_values,
        #             masks,
        #             boxes,
        #             reconstruct_mask=False,
        #         )
        # except torch.OutOfMemoryError:
        #     clear_gpu_memory()

        #     pixel_values = pixel_values[:10]
        #     masks = masks[:10]
        #     boxes = boxes[:10]

        #     rles = rles[:10]

        #     with torch.no_grad():
        #         vq_sam2_output = vq_sam2(
        #             pixel_values,
        #             masks,
        #             boxes,
        #             reconstruct_mask=False,
        #         )
        try:
            pixel_values = pixel_values[:10]
            masks = masks[:10]
            boxes = boxes[:10]

            rles = rles[:10]

            with torch.no_grad():
                vq_sam2_output = vq_sam2(
                    pixel_values,
                    masks,
                    boxes,
                    reconstruct_mask=False,
                )
        except Exception as e:
            print(f"Encounter exception: {e}")
            continue
        
        quant_codes = vq_sam2_output.quant_codes
        quant_codes = quant_codes.cpu().numpy().astype(np.int32).tolist()

        ret_data_dict = []
        for ins_idx, (rle, box, _quant_codes_) in enumerate(zip(rles, boxes, quant_codes)):
            ret_data_dict.append({
                'id': ins_idx,
                'segmentation': rle,
                'box': box.cpu().numpy().tolist(),
                'quant_codes': _quant_codes_[0],
                'image_file': image_file,
                'height': height,
                'width': width,
            })
        
        with open(os.path.join(temp_save_root, f"{image_name}.json"), 'w') as f:
            json.dump(ret_data_dict, f)

        clear_gpu_memory()


def collect_conversations():
    CODEBOOK_SIZE = 1024
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    save_root = "./temp_data/visual_mask_text_mask_alignment_conversation/"
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    temp_save_root = "./temp_data/visual_mask_text_mask_alignment/"
    # for json_file in tqdm(os.listdir(temp_save_root)):
    for json_file in scandir_generator(temp_save_root):
        json_path = os.path.join(temp_save_root, json_file)
        if os.path.exists(os.path.join(save_root, json_file)):
            continue
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            rles = [item['segmentation'] for item in json_data]
            quant_codes = [item['quant_codes'] for item in json_data]
            ids = [item['id'] for item in json_data]
            image_file = json_data[0]['image_file']

            height, width = rles[0]['size']
            masks = decode_mask(rles, height, width)
            image = Image.open(image_file).convert('RGB')
            output_image = visualize(image, masks, [str(_) for _ in ids])

            # uuid_str = uuid.uuid4()
            # output_image.save(f"{uuid_str}.jpg")
            # exit(0)
            img_str = image_to_base64_str(output_image)

            answer = "<segmentation>```json\n[{mask_2d}]\n```</segmentation>"
            mask_2d_str = ''
            for id, _quant_codes_ in zip(ids, quant_codes):
                _remap_quant_codes_ = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes_)]
                item_str = "{\"mask_2d\": [" + ', '.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _remap_quant_codes_]) + "], \"label\": \"" + str(id) + "\"}"
                mask_2d_str += item_str + ",\n"
            mask_2d_str = mask_2d_str[:-len(",\n")]
            answer = answer.format(mask_2d=mask_2d_str)

            conversation = []
            conversation.append({'from': 'human', 'value': '<image>\nMasks of the marked regions: '})
            conversation.append({'from': 'gpt', 'value': answer})
            ret_data_dict = {
                'image': img_str,
                'conversations': conversation,
            }

            with open(os.path.join(save_root, json_file), 'w') as f:
                json.dump(ret_data_dict, f)

if __name__ == "__main__":
    # task_id = sys.argv[1]
    # task_id = int(task_id)
    # main(task_id)

    collect_conversations()
