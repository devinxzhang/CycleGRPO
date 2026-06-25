import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print("use npu success!")
except:
    print("npu not enabled!")
import copy
from PIL import Image
import numpy as np
import os
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import matplotlib as mpl

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize
from projects.vlm.vq_sam2.datasets import CoCoPanoSegValDataset
from projects.vlm.vq_sam2.datasets.coco_category import COCO_CATEGORIES

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from xtuner.model.utils import guess_load_checkpoint

from qwen_vl_utils import process_vision_info


import random
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
    
    alpha = 0.0
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

def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union

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

    
if __name__ == "__main__":

    
    # all_results = []
    # for json_file in os.listdir('./reconstruction_eval_results/vq_sam2_cocopano_depth4_unshare_codebook_laten_dim_256_mask_token_1_thing'):
    #     if json_file.endswith('.json'):
    #         with open(f'./reconstruction_eval_results/vq_sam2_cocopano_depth4_unshare_codebook_laten_dim_256_mask_token_1_thing/{json_file}', 'r') as f:
    #             results = json.load(f)
    #             all_results.extend(results)
    # print("Mean Mask IoU: ", np.mean(all_results))
    # exit(0)

    # build qwen25vl model
    model_path = "./work_dirs/qwen25vl_vq_sam2_3b_1024x4_stage0/hf_ckpt"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(model_path)

    sam2_config = SAM2Config(
        cfg_path="sam2.1_hiera_l.yaml",
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
    )

    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=1024,
        codebook_depth=4,
        shared_codebook=False,
        latent_dim=256,
    )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    pretrained_pth = "./pretrained_weights/iter_17923.pth"
    pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

    pretrained_state_dict_new = {}
    for key in pretrained_state_dict.keys():
        new_key = copy.deepcopy(key)
        if key.startswith('hf_model.'):
            new_key = new_key[len('hf_model.'):]
        pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
    
    vq_sam2.load_state_dict(pretrained_state_dict_new)

    sam2_image_processor = DirectResize(1024)

    val_dataset = CoCoPanoSegValDataset(
        data_path="./data/coco/annotations/panoptic_val2017.json",
        image_folder="./data/coco/val2017",
        pano_gt_folder="./data/coco/annotations/panoptic_val2017",
        preprocessor=dict(
            type=DirectResize,
            target_length=1024,
        ),
    )

    eval_results_root = "./reconstruction_eval_results/qwen25vl_1024x4/"
    if not os.path.exists(eval_results_root):
        os.makedirs(eval_results_root)

    isthing_dict = {e['name']: e['isthing'] for e in COCO_CATEGORIES}

    all_iou = []
    max_ins = 1000
    for idx in tqdm(range(len(val_dataset))):
        if max_ins == 0:
            break
        data = val_dataset[idx]
        image_file = data['image_file']
        image_name = os.path.basename(image_file).split('.jpg')[0]
        masks = data['masks']
        class_names = data['class_names']
        image = Image.open(image_file)
        width, height = image.size
        all_quant_codes = []
        
        eval_result_path = os.path.join(eval_results_root, f'{image_name}.json')
        if os.path.exists(eval_result_path):
            continue

        this_file_results = []
        for mask, class_name in zip(masks, class_names):

            val_item = val_dataset.prepare_mask_input(image_file, mask, class_name)
            pixel_values = val_item['pixel_values']
            masks = val_item['masks']
            boxes = val_item['boxes'].to(vq_sam2.device)

            np_masks = masks.detach().cpu().numpy()
            contour_image = visualize(image, np_masks, [str(_) for _ in range(len(np_masks))])
            contour_image.save(f'./coco_val_recon/{image_name}_{max_ins}_gt.jpg')

            # question = text.replace('<image>\n', '').strip()
            question = "Masks of the marked regions: "
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": contour_image,
                        },
                        {"type": "text", "text": question},
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
                max_new_tokens=512,
                do_sample=False,  # 关闭采样，使用贪婪解码
                top_p=1.0,  # 配合do_sample=False使用
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            print("Assistant: ", output_text)

            quant_ids = extract_mt_token_ids(output_text[0])[:4]

            batch_size = 1
            remap_quant_ids = np.array([-1 for _ in range(4)])
            for quant_id in quant_ids:
                depth_idx = quant_id // 1024
                remap_quant_ids[depth_idx] = quant_id % 1024
            truncated_idx = find_first_index(remap_quant_ids, -1)
            if truncated_idx != -1:
                remap_quant_ids[truncated_idx:] = -1
            if remap_quant_ids[0] == -1:
                continue
            quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device).unsqueeze(0)

            pixel_values = pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
            pred_masks = vq_sam2.forward_with_codes(pixel_values, quant_ids)

            pred_masks = torch.nn.functional.interpolate(pred_masks, size=(height, width), mode='bilinear')
            pred_masks = pred_masks > 0.5
            pred_masks = pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)

            recon_image = visualize(image, pred_masks, [str(_) for _ in range(len(pred_masks))])
            recon_image.save(f'./coco_val_recon/{image_name}_{max_ins}_recon.jpg')
            max_ins -= 1
            exit(0)
