import os
import tensorflow as tf
import time
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

def read_tfrecord_files(file_pattern, feature_description=None, num_parallel_reads=4, batch_size=32,
                        shuffle_buffer_size=10000):
    """
    读取TFRecord文件并以字典形式返回数据

    参数:
        file_pattern: TFRecord文件路径模式，支持通配符
        feature_description: 特征解析字典，默认为None(自动检测简单特征)
        num_parallel_reads: 并行读取文件数
        batch_size: 批次大小
        shuffle_buffer_size: 打乱数据的缓冲区大小，0表示不打乱

    返回:
        tf.data.Dataset对象，每个元素是包含特征的字典
    """
    # 获取文件列表
    file_paths = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle_buffer_size > 0)

    # 自动检测特征描述(如果未提供)
    if feature_description is None:
        feature_description = _detect_feature_description(file_paths.take(1))

    # 创建数据集
    dataset = file_paths.interleave(
        tf.data.TFRecordDataset,
        cycle_length=num_parallel_reads,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 打乱数据(如果需要)
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)

    # 解析TFRecord
    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

    # 批处理
    dataset = dataset.batch(batch_size)

    # 预取数据
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def _detect_feature_description(file_paths):
    """自动检测TFRecord文件中的特征结构"""
    # 创建TFRecord读取器
    raw_dataset = tf.data.TFRecordDataset(file_paths)

    # 获取第一个样本
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

    # 构建特征描述
    feature_description = {}
    for key, feature in example.features.feature.items():
        kind = feature.WhichOneof('kind')
        if kind == 'bytes_list':
            feature_description[key] = tf.io.FixedLenFeature([], tf.string)
        elif kind == 'float_list':
            feature_description[key] = tf.io.FixedLenFeature([], tf.float32)
        elif kind == 'int64_list':
            feature_description[key] = tf.io.FixedLenFeature([], tf.int64)

    return feature_description

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

def main():
    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    QUESTION_TEMPLATE = "{content} A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
    ANSWER_TEMPLATE = "<think> {thinking} </think><answer> {answer} </answer>"

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

    dataset = read_tfrecord_files("./data/HarborYuan/VisualReasoningTracer/ver_training_*.tfrecord", shuffle_buffer_size=0)
    total_num = 0
    for batch in dataset:
        total_num += 1
    
    batch_idx = 0
    skip_count = 0
    for batch in tqdm.tqdm(dataset, total=total_num, desc="Processing"):
        if batch_idx < skip_count:
            batch_idx += 1
            continue
        # dict_keys(['image', 'json', 'key'])
        ann_tensor = batch['json']
        annotations = [json.loads(s.decode("utf-8")) for s in ann_tensor.numpy()]
        
        batch_data_dict = []
        for anno in annotations:
            # dict_keys(['image_name', 'annotation_file', 'original_annotation', 'generated_ver_data', 'timestamp', 'raw_response', 'meta_data'])
            image_name = anno['image_name']
            image_path = os.path.join('./data/sam_full', image_name)

            original_annotation = anno['original_annotation']
            generated_ver_data = anno['generated_ver_data']

            objects_anns = original_annotation['objects_anns']

            candidates_ver_samples = generated_ver_data['candidates']
            all_obj_ids = []
            for ver_sample in candidates_ver_samples:
                reasoning = ver_sample['reasoning']
                obj_ids = re.findall(r"<ver>(<obj\d+>)</ver>", reasoning)
                obj_ids.extend(ver_sample['answer_objects'])
                seen = set()
                unique_ids = [x for x in obj_ids if not (x in seen or seen.add(x))]
                all_obj_ids.extend(unique_ids)

            try:
                segms = [objects_anns[obj_id]['segmentation'] for obj_id in all_obj_ids]
            except:
                continue

            image = Image.open(image_path).convert('RGB')
            ori_width, ori_height = image.size

            masks = decode_mask(segms, ori_height, ori_width)
            assert len(masks) == len(all_obj_ids)
            if len(masks) == 0:
                continue

            # output_image = visualize(image, masks, all_obj_ids)
            # output_image.save('ver_sample.jpg')

            sam2_image = np.array(image)
            sam2_image = sam2_image_processor.apply_image(sam2_image)
            sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
            sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

            masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            try:
                boxes = torchvision.ops.masks_to_boxes(masks)
            except:
                print("Error at boxes = torchvision.ops.masks_to_boxes(masks)")
                continue

            whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
            boxes = boxes / whwh
            boxes = boxes.to(vq_sam2.device)
            masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
            num_ins = len(masks)

            skip_this_one = False
            all_quant_codes = []
            for mask_idx in range(num_ins):
                with torch.no_grad():
                    vq_sam2_output = vq_sam2(
                        sam2_pixel_values,
                        masks[mask_idx:mask_idx+1],
                        boxes[mask_idx:mask_idx+1],
                        reconstruct_mask=False,
                    )
                    quant_codes = vq_sam2_output.quant_codes.detach()
                    all_quant_codes.append(quant_codes)
            quant_codes = torch.cat(all_quant_codes, dim=0)

            quant_codes = quant_codes.cpu().numpy().astype(np.int32).tolist()
            remap_quant_codes = []
            for _quant_codes in quant_codes:
                _quant_codes = _quant_codes[0]
                remap_quant_codes.append([depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes)])
            quant_codes = remap_quant_codes

            mask_token_str_list = []
            for _quant_codes_ in quant_codes:
                sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes_]) + MT_END_TOKEN
                mask_token_str_list.append(sam2_tokens)
            
            obj_id_2_mask_token_str = {obj_id: mask_token_str for obj_id, mask_token_str in zip(all_obj_ids, mask_token_str_list)}

            for ver_sample in candidates_ver_samples:
                question = ver_sample['question']
                reasoning = ver_sample['reasoning']
                answer_caption = ver_sample['answer_caption']

                for obj_id, mask_token_str in obj_id_2_mask_token_str.items():
                    reasoning = reasoning.replace(f'<ver>{obj_id}</ver>', mask_token_str)
                    answer_caption = answer_caption.replace(f'<vea>{obj_id}</vea>', mask_token_str)
                
                question = QUESTION_TEMPLATE.format(content=question)
                answer = ANSWER_TEMPLATE.format(thinking=reasoning, answer=answer_caption)

                conversation = []
                conversation.append({'from': 'human', 'value': question})
                conversation.append({'from': 'gpt', 'value': answer})

                ret_data_dict = {
                    'image': image_path,
                    'conversations': conversation,
                }
                batch_data_dict.append(ret_data_dict)
            
                # print(ret_data_dict)
                # exit(0)
        
        with open(f'./temp_data_256x2_0927/ver/{batch_idx}.json', 'w') as f:
            json.dump(batch_data_dict, f)
        
        batch_idx += 1

if __name__ == "__main__":
    main()
