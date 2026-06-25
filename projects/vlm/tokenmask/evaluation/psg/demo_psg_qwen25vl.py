import argparse
import copy
import math
import os
import torch
import torchvision
import tqdm
from pycocotools import mask as mask_utils
import numpy as np
import random
import re
from PIL import Image
import json
import uuid
import matplotlib as mpl
import hydra

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import mmengine
from mmengine.dataset import BaseDataset
from mmdet.registry import DATASETS
from mmdet.datasets.coco_panoptic import COCOPanoptic, CocoPanopticDataset
from collections import defaultdict


from qwen_vl_utils import process_vision_info

from xtuner.model.utils import guess_load_checkpoint

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize
from projects.vlm.tokenmask.evaluation.psg.relation_utils import Result

import colorsys

def generate_distinct_bright_colors(count, saturation=0.7, value=0.9):
    """
    生成指定数量的明显不同的亮色RGB颜色
    
    参数:
        count: 要生成的颜色数量
        saturation: 饱和度 (0-1)，值越高颜色越鲜艳
        value: 明度 (0-1)，值越高颜色越明亮
        
    返回:
        包含RGB元组的列表，每个元组包含三个0-255的整数
    """
    colors = []
    
    # 均匀分布在色相环上，确保颜色差异明显
    hue_step = 1.0 / count
    
    for i in range(count):
        # 计算色相值，均匀分布在0-1之间
        hue = i * hue_step
        
        # 随机微调色相，增加多样性但保持区分度
        hue += random.uniform(-hue_step * 0.3, hue_step * 0.3)
        hue %= 1.0  # 确保在0-1范围内
        
        # 从HSV转换到RGB (HSV颜色模型更容易控制饱和度和明度)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # 转换到0-255范围
        # r = int(r * 255)
        # g = int(g * 255)
        # b = int(b * 255)
        
        colors.append((r, g, b))
    
    return colors


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

    # if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
    #     colors = (
    #         [self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes]
    #         if jittering
    #         else [
    #             tuple(mplc.to_rgb([x / 255 for x in self.metadata.thing_colors[c]]))
    #             for c in classes
    #         ]
    #     )

    #     alpha = 0.8
    # else:
    #     colors = None
    #     alpha = 0.5
    
    alpha = 0.5
    colors = generate_distinct_bright_colors(len(masks))

    self.overlay_instances(
        masks=masks,
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=colors,
        alpha=alpha,
    )
    return self.output

def draw_polygon_cache(self, segment, color, edge_color=None, alpha=0.5):
        """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted. If not provided, a darker shade
                of the polygon color will be used instead.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with polygon drawn.
        """
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=4,
        )
        self.output.ax.add_patch(polygon)
        return self.output


def draw_text_cache(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # # since the text background is dark, we don't want the text to be dark
        # color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        # color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

def _change_color_brightness_cache(self, color, brightness_factor):
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.

        Args:
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
                0 will correspond to no change, a factor in [-1.0, 0) range will result in
                a darker color and a factor in (0, 1.0] range will result in a lighter color.

        Returns:
            modified_color (tuple[double]): a tuple containing the RGB values of the
                modified color. Each value in the tuple is in the [0.0, 1.0] range.
        """
        # assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        # color = mplc.to_rgb(color)
        # polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        # modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        # modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        # modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        # modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        # return tuple(np.clip(modified_color, 0.0, 1.0))
        return color


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
    visualizer.draw_polygon = MethodType(draw_polygon_cache, visualizer)
    visualizer.draw_text = MethodType(draw_text_cache, visualizer)
    visualizer._change_color_brightness = MethodType(_change_color_brightness_cache, visualizer)
    vis_output = visualizer.draw_instance_predictions(labels=left_tags, np_masks=result_masks)
    output_image = vis_output.get_image()
    output_image = Image.fromarray(output_image)

    return output_image

def extract_mt_token_ids(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]

def find_first_index(arr, value):
    """
    在NumPy数组中找到第一个指定值的第一个出现的索引
    
    参数:
        arr: NumPy数组
        value: 要查找的值
        
    返回:
        第一个匹配值的索引，如果没有找到则返回-1
    """
    # 使用where找到所有匹配值的索引
    indices = np.where(arr == value)[0]
    
    # 返回第一个索引，如果没有找到则返回-1
    return indices[0] if len(indices) > 0 else -1

class COCOPanopticRelation(COCOPanoptic):
    def createIndex(self):
        # create index
        print('creating index...')
        # anns stores 'segment_id -> annotation'
        anns, cats, imgs = {}, {}, {}
        relations = {}

        segments_info = {}

        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann, img_info in zip(self.dataset['annotations'],
                                     self.dataset['images']):
                img_info['segm_file'] = ann['file_name']
                for seg_ann in ann['segments_info']:
                    # to match with instance.json
                    seg_ann['image_id'] = ann['image_id']
                    seg_ann['height'] = img_info['height']
                    seg_ann['width'] = img_info['width']
                    img_to_anns[ann['image_id']].append(seg_ann)
                    # segment_id is not unique in coco dataset orz...
                    if seg_ann['id'] in anns.keys():
                        anns[seg_ann['id']].append(seg_ann)
                    else:
                        anns[seg_ann['id']] = [seg_ann]

                relations[ann['image_id']] = ann['relations']
                segments_info[ann['image_id']] = ann['segments_info']

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    cat_to_imgs[seg_ann['category_id']].append(ann['image_id'])

        print('index created!')

        self.anns = anns
        self.imgToAnns = img_to_anns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats
        self.relations = relations
        self.segments_info = segments_info
        self.relations_categories = self.dataset['relations_categories']
        self.relationID2Categories = {item['id']: item['name'] for item in self.dataset['relations_categories']}

import json
import re
from typing import Any, List, Optional

FENCE_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def _clean_wrappers(s: str) -> str:
    """去掉常见包裹符、收尾空白等。"""
    s = s.replace("<|im_end|>", "")
    s = s.strip()
    # 去掉首尾引号（包括中英文引号）
    quotes = "'\"“”‘’"
    if len(s) >= 2 and s[0] in quotes and s[-1] in quotes:
        s = s[1:-1].strip()
    return s

def _extract_from_code_fence(text: str) -> List[str]:
    return [m.strip() for m in FENCE_JSON_RE.findall(text)]

def _extract_by_bracket_scan(text: str) -> List[str]:
    """
    在全文里用配对括号扫描，提取可能的 JSON 片段（对象或数组）。
    忽略字符串内的括号与转义。
    返回候选片段（可能有嵌套，调用方可按长度降序尝试 json.loads）。
    """
    candidates = []
    open_to_close = {"{": "}", "[": "]"}
    open_set = set(open_to_close.keys())
    close_set = set(open_to_close.values())

    n = len(text)
    for start in range(n):
        ch = text[start]
        if ch not in open_set:
            continue
        stack = [open_to_close[ch]]
        in_string = False
        escape = False
        for i in range(start + 1, n):
            c = text[i]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
                continue
            else:
                if c == '"':
                    in_string = True
                elif c in open_set:
                    stack.append(open_to_close[c])
                elif c in close_set:
                    if not stack or c != stack[-1]:
                        break  # 非法配对，放弃这个起点
                    stack.pop()
                    if not stack:
                        # 成功匹配一段
                        candidates.append(text[start : i + 1])
                        break
    # 去重 & 按长度降序（优先外层最大块）
    uniq = list(dict.fromkeys(candidates))
    uniq.sort(key=len, reverse=True)
    return uniq

def _try_parse_candidates(cands: List[str]) -> List[Any]:
    parsed = []
    for raw in cands:
        cand = _clean_wrappers(raw)
        try:
            parsed.append(json.loads(cand))
        except Exception:
            # 再尝试去掉再次包裹的三引号/反引号之类
            cand2 = cand.strip("`").strip()
            try:
                parsed.append(json.loads(cand2))
            except Exception:
                continue
    return parsed

def parse_first_json(text: str) -> Any:
    """
    提取并解析第一个可用 JSON（先看```json```代码块，再看正文扫描）。
    解析失败会抛出 ValueError。
    """
    text = _clean_wrappers(text)

    # 1) 代码块
    fence_cands = _extract_from_code_fence(text)
    parsed = _try_parse_candidates(fence_cands)
    if parsed:
        return parsed[0]

    # 2) 正文扫描（对象/数组）
    bracket_cands = _extract_by_bracket_scan(text)
    parsed = _try_parse_candidates(bracket_cands)
    if parsed:
        return parsed[0]

    raise ValueError("没有在文本中找到可解析的 JSON。")

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--vq_sam2_path',
        default="pretrained_weights/iter_175473.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--dataset',
        default='./data/PaDT-MLLM/RefCOCO/refcoco_val.json',
        help='Specify a ref dataset')
    parser.add_argument('--task_id', '--task-id', type=int, default=0)
    args = parser.parse_args()
    return args


from skimage.measure import label
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as ndi_label
def fill_holes_for_components(mask, hole_threshold=5, connectivity=1):
    """
    只对孔洞数量 > hole_threshold 的前景连通域填洞。
    - mask: 二值数组（0/1 或 bool），2D 或 3D 均可
    - connectivity: 2D时 1=4连通, 2=8连通；3D时 1=6连通, 2=18连通, 3=26连通
    """
    fg = mask.astype(bool)
    # 1) 标记前景连通域
    comp_label = label(fg, connectivity=connectivity)
    n_comp = comp_label.max()
    if n_comp == 0:
        return fg.astype(mask.dtype)
    # 2) 统计每个连通域内部孔洞数量
    # 对于每个连通域：只看该连通域的边界框区域，计算其孔洞数
    filled = fg.copy()
    for cid in range(1, n_comp + 1):
        comp_mask = (comp_label == cid)
        if not comp_mask.any():
            continue
        # 提取该连通域的紧致包围盒，减少计算
        coords = np.argwhere(comp_mask)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        slc = tuple(slice(a, b) for a, b in zip(mins, maxs))
        comp_sub = comp_mask[slc]
        # 背景区域（仅在该子块中）
        bg_sub = ~fg[slc]
        # 让“与子块边界相连的背景”为外部背景，其余背景连通块为孔洞
        # 做法：在子块上对背景做连通域标记，然后把粘到子块边缘的背景标签视为外部
        bg_labels, n_bg = ndi_label(bg_sub, structure=np.ones((3,)*bg_sub.ndim) if connectivity>1 else None)
        # 找出在子块边界上出现的背景标签
        border_tags = set()
        for axis in range(bg_sub.ndim):
            border_tags.update(np.unique(bg_labels.take(indices=0, axis=axis)))
            border_tags.update(np.unique(bg_labels.take(indices=-1, axis=axis)))
        # 孔洞标签 = 非0 且 不在边界的背景标签
        all_tags = set(np.unique(bg_labels))
        hole_tags = [t for t in all_tags if t != 0 and t not in border_tags]
        num_holes = len(hole_tags)
        # 3) 判断是否填洞
        if num_holes > hole_threshold:
            # 填洞：仅在该连通域内部进行 fill
            # 方式A：对 comp_sub 的前景做 binary_fill_holes
            comp_filled = binary_fill_holes(comp_sub)
            # 将填充结果写回，仅影响该连通域
            # 注意：只把原来是前景或新填补的孔洞位置置为True
            filled_region = comp_filled
            filled[slc] = np.where(comp_sub, True, filled[slc])  # 保持前景
            # 把 comp_sub 中由 fill 产生的新前景位置也置 True
            filled[slc] |= (filled_region & ~comp_sub)
    return filled.astype(mask.dtype)

def main():
    args = parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    # build vq-sam2 model
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
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

    coco = COCOPanopticRelation('./data/psg_data/psg_val.json')

    candidate_categories = [v['name'] for k, v in coco.cats.items()]
    candidate_predicates = [v['name'] for v in coco.relations_categories]
    candidate_categories_str = "{" + ", ".join(candidate_categories) + "}"
    candidate_predicates_str = "{" + ", ".join(candidate_predicates) + "}"

    # candidate_categories_str = "{" + ", ".join(['girl', 'totoro', 'cat bus', 'grass', 'house', 'farmland', 'sky']) + "}"
    # candidate_categories = [
    #     "sofa",
    #     "dog",
    #     "pillows",
    #     "blanket",
    #     "coffee table",
    #     "teapot",
    #     "mug",
    #     "rug",
    #     "floor lamp",
    #     "lampshade",
    #     "glass lamp base",
    #     "side table",
    #     "clock",
    #     "magazine rack",
    #     "wall art",
    #     "pendant light",
    #     "wall decor",
    #     "plants",
    #     "plant pots",
    #     "plant stand",
    #     "door",
    #     "bookshelf",
    #     "basket",
    #     "picture frames",
    #     "wall shelf",
    #     "floor"
    # ]
    # candidate_categories_str = "{" + ", ".join(candidate_categories) + "}"

    # image_path = "./CVPR2026/shutterstock_1723139395.jpg"
    # question_round1 = "Please carefully check the image and detect the following objects: " + candidate_categories_str
    image_path_list = os.listdir('./data/coco/val2014')
    count = 0
    for image_file in tqdm.tqdm(image_path_list):
        image_name = image_file.replace('.jpg', '')
        image_path = os.path.join('./data/coco/val2014', image_file)
        question_round1 = "Please carefully check the image and detect the following objects: " + candidate_categories_str
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": question_round1},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=2048,
            do_sample=False,  # 关闭采样，使用贪婪解码
            top_p=1.0,  # 配合do_sample=False使用
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        print("ROUND@1 Assistant: ", output_text[0])

        #=======================round 2
        candidate_masks_str = output_text[0].replace('<|im_end|>', '')
        question_round2 = "CANDIDATE PREDICATES: \n" + candidate_predicates_str + "\n" + "Create a scene graph by identifying triplets of <subject, predicate, object>. Focus on the most prominent interactions."
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": question_round1},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": candidate_masks_str},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question_round2},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=2048,
            do_sample=False,  # 关闭采样，使用贪婪解码
            top_p=1.0,  # 配合do_sample=False使用
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        print("ROUND@2 Assistant: ", output_text[0])

        quant_ids = extract_mt_token_ids(candidate_masks_str)
        batch_size = len(quant_ids) // CODEBOOK_DEPTH
        remap_quant_ids = []
        tags = []
        for bs_id in range(batch_size):
            chunk_quant_ids = quant_ids[bs_id*CODEBOOK_DEPTH:(bs_id+1)*CODEBOOK_DEPTH]
            tags.append(f"{chunk_quant_ids[0]}-{chunk_quant_ids[1]}")
            remap_chunk_quant_ids = [quant_id - book_id*CODEBOOK_SIZE for book_id, quant_id in enumerate(chunk_quant_ids)]
            remap_chunk_quant_ids_error_handle = [quant_id if quant_id < CODEBOOK_SIZE else -1 for quant_id in remap_chunk_quant_ids]
            remap_quant_ids.append(remap_chunk_quant_ids_error_handle)
            

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size
        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
        sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

        quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

        with torch.no_grad():
            try:
                _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
            except:
                continue
        _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
        _pred_masks = _pred_masks > 0.5
        _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)

        final_masks = []
        for _pred_mask in tqdm.tqdm(_pred_masks):
            _pred_mask = fill_holes_for_components(_pred_mask)
            final_masks.append(_pred_mask)
        _pred_masks = np.stack(final_masks, axis=0)

        count_str = f'{count}'.zfill(5)
        output_image = visualize(image, _pred_masks, tags)
        output_image.save(f'./CVPR2026/demo_psg/{count_str}_{image_name}.jpg')
        with open(f'./CVPR2026/demo_psg/{count_str}_{image_name}.txt', "w", encoding="utf-8") as ftxt:
            ftxt.write(output_text[0])

        # output_image_clean = visualize(image, _pred_masks, ['']*len(_pred_masks))
        # output_image_clean.save('./CVPR2026/demo_psg/clean.jpg')

        count += 1

if __name__ == '__main__':
    main()

