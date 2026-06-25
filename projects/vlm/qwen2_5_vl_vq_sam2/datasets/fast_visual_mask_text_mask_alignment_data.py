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
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
import time
from base64 import b64encode
import uuid
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

# 确保输出目录存在
Path("data").mkdir(parents=True, exist_ok=True)

def process_file(json_path):
    """处理单个文件，返回解析后的JSON数据或None"""

    CODEBOOK_SIZE = 1024
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    # save_root = "./temp_data/visual_mask_text_mask_alignment_conversation/"

    json_file = os.path.basename(json_path)
    ret_data_dict_list = []
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        for item in json_data:
            rle = item['segmentation']
            quant_codes = item['quant_codes']
            id = item['segmentation']
            image_file = item['image_file']
            answer = "<segmentation>```json\n[{mask_2d}]\n```</segmentation>"
            
            _remap_quant_codes_ = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes_)]
            item_str = "{\"mask_2d\": [" + ', '.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _remap_quant_codes_]) + "], \"label\": \"" + str(id) + "\"}"


        rles = [item['segmentation'] for item in json_data]
        quant_codes = [item['quant_codes'] for item in json_data]
        ids = [item['id'] for item in json_data]
        image_file = json_data[0]['image_file']



        # height, width = rles[0]['size']
        # masks = decode_mask(rles, height, width)
        # image = Image.open(image_file).convert('RGB')
        # output_image = visualize(image, masks, [str(_) for _ in ids])

        # uuid_str = uuid.uuid4()
        # output_image.save(f"{uuid_str}.jpg")
        # exit(0)
        # img_str = image_to_base64_str(output_image)

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
            'image': image_file,
            'conversations': conversation,
            'segmentation': rles,
            'ids': ids,
        }

        # with open(os.path.join(save_root, json_file), 'w') as f:
        #     json.dump(ret_data_dict, f)
        return ret_data_dict

def get_json_files(directory):
    """生成指定目录下所有JSON文件的路径"""
    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.endswith(".json"):
            yield entry.path

def main():
    input_dir = "./temp_data/visual_mask_text_mask_alignment/"
    batch_size = 10000  # 每1万个文件保存一次
    # Linux下可以使用比CPU核心数多一些的进程数提高IO密集型任务性能
    processes = min(multiprocessing.cpu_count() * 2, 64)  # 最多64个进程
    
    print(f"开始处理文件，使用 {processes} 个进程，每批处理 {batch_size} 个文件")
    start_time = time.time()
    
    json_files = get_json_files(input_dir)
    
    # Linux系统推荐使用fork启动方式，效率更高
    with Pool(processes=processes, initializer=lambda: os.nice(10)) as pool:
        batch_number = 0
        results = []
        total_processed = 0
        
        # 迭代处理所有文件，chunksize根据文件大小调整
        for result in pool.imap(process_file, json_files, chunksize=500):
            if result is not None:
                results.append(result)
                total_processed += 1
                
                # 达到批次大小则保存
                if len(results) >= batch_size:
                    batch_number += 1
                    output_file = f"data/visual_mask_text_mask_alignment_batch_{batch_number}.json"
                    with open(output_file, 'w') as f:
                        json.dump(results, f)
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed
                    print(f"已保存批次 {batch_number}，包含 {len(results)} 个文件，"
                          f"总处理: {total_processed} 个，速度: {rate:.2f} 个/秒")
                    results = []  # 清空当前批次数据
        
        # 处理剩余的文件
        if results:
            batch_number += 1
            output_file = f"data/visual_mask_text_mask_alignment_batch_{batch_number}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f)
            print(f"已保存最后批次 {batch_number}，包含 {len(results)} 个文件")
    
    total_time = time.time() - start_time
    print(f"所有文件处理完成，总耗时: {total_time:.2f} 秒，"
          f"平均速度: {total_processed/total_time:.2f} 个/秒")


if __name__ == "__main__":
    all_items = []
    for json_file in os.listdir('./data'):
        if not json_file.startswith('visual_mask_text_mask_alignment_batch'):
            continue
        with open(f'./data/{json_file}', 'r') as f:
            json_data = json.load(f)
            all_items.extend(json_data)
    num_samples = len(all_items) // 1000
    with open(f'./data/vq_sam2_data_x61/visual_mask_text_mask_alignment_{num_samples}k.json', 'w') as f:
        json.dump(all_items, f)
    exit(0)
    main()
