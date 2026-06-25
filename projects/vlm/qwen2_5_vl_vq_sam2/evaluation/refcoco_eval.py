import argparse
import copy
import math
import os
import torch
import torchvision
import tqdm
from pycocotools import mask as _mask
import numpy as np
import random
import re
from PIL import Image
import json
import uuid

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from dataset import RESDataset
from xtuner.model.utils import guess_load_checkpoint

from qwen_vl_utils import process_vision_info
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


def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--vq_sam2_path',
        default="pretrained_weights/iter_175473.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_ATTRIBUTES.keys(),
        default='refcoco',
        help='Specify a ref dataset')
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

DATASETS_ATTRIBUTES = {
    'refcoco': {'splitBy': "unc", 'dataset_name': 'refcoco'},
    'refcoco_plus': {'splitBy': "unc", 'dataset_name': 'refcoco_plus'},
    'refcocog': {'splitBy': "umd", 'dataset_name': 'refcocog'},
}

IMAGE_FOLDER = './data/glamm_data/images/coco2014/train2014/'
DATA_PATH = './data/ref_seg/'

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

def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union

def main():
    args = parse_args()

    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build qwen25vl model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    # build vq-sam2 model
    sam2_config = SAM2Config(
        cfg_path="sam2.1_hiera_l.yaml",
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
    )

    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=[256, 48],
        codebook_depth=2,
        shared_codebook=False,
        latent_dim=256,
    )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    pretrained_state_dict = guess_load_checkpoint(args.vq_sam2_path)

    pretrained_state_dict_new = {}
    for key in pretrained_state_dict.keys():
        new_key = copy.deepcopy(key)
        if key.startswith('hf_model.'):
            new_key = new_key[len('hf_model.'):]
        pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
    
    vq_sam2.load_state_dict(pretrained_state_dict_new)

    sam2_image_processor = DirectResize(1024)


    # dataset
    dataset_info = DATASETS_ATTRIBUTES[args.dataset]

    dataset = RESDataset(
        image_folder=IMAGE_FOLDER,
        dataset_name=dataset_info['dataset_name'],
        data_path=DATA_PATH,
        split=args.split,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for idx in tqdm.tqdm(per_rank_ids[:200]):
        data_batch = dataset[idx]
        prediction = {'img_id': data_batch['img_id'], 'gt_masks': data_batch['gt_masks']}
        target_masks = prediction['gt_masks'].cpu().numpy()
        prediction['gt_masks'] = mask_to_rle(prediction['gt_masks'].cpu().numpy())
        del data_batch['img_id'], data_batch['gt_masks']

        img_id = prediction['img_id']
        if os.path.exists(f"./temp_save/{args.dataset}/{img_id}.json"):
            print("file exists.............")
            continue

        texts = data_batch['text']
        del data_batch['text']
        pred_masks = []
        assert len(texts) == len(target_masks)
        for text_idx, text in enumerate(texts):
            _data_batch = copy.deepcopy(data_batch)
            _data_batch['text'] = text

            image_file= _data_batch['image_file']
            question = text.replace('<image>\n', '').strip()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_file,
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
            try:
                inputs = inputs.to("cuda")
            except Exception as e:
                print("inputs = inputs.to(\"cuda\") encounter error..........")
                # print("inputs.shape: ", inputs.shape)
                # print("inputs.dtype: ", inputs.dtype)
                # print("inputs.device: ", inputs.device)
                print("inputs: \n", inputs)
                print("Exception e: ", e)
                exit(0)
                pred_masks.append(None)

            # Inference: Generation of the output
            try:
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=64,
                    do_sample=False,  # 关闭采样，使用贪婪解码
                    top_p=1.0,  # 配合do_sample=False使用
                )
            except Exception as e:
                print("generated_ids = model.generate encounter error..............")
                print("Exception e: ", e)
                exit(0)
                pred_masks.append(None)
                continue
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            print("Assistant: ", output_text)

            quant_ids = extract_mt_token_ids(output_text[0])
            if len(quant_ids) == 0:
                pred_masks.append(None)
                continue
            # elif len(quant_ids) > 2:
            #     quant_ids = quant_ids[:2]
            
            # batch_size = len(quant_ids) // 4
            # remap_quant_ids = []
            # for bs_id in range(batch_size):
            #     chunk_quant_ids = quant_ids[bs_id*4:(bs_id+1)*4]
            #     remap_chunk_quant_ids = [quant_id - book_id*1024 for book_id, quant_id in enumerate(chunk_quant_ids)]
            #     remap_quant_ids.append(remap_chunk_quant_ids)
            batch_size = 1
            remap_quant_ids = np.array([-1 for _ in range(2)])
            for quant_id in quant_ids:
                depth_idx = quant_id // 256
                remap_quant_ids[depth_idx] = quant_id % 256
            truncated_idx = find_first_index(remap_quant_ids, -1)
            if truncated_idx != -1:
                remap_quant_ids[truncated_idx:] = -1
            if remap_quant_ids[0] == -1:
                pred_masks.append(None)
                continue
            quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device).unsqueeze(0)

            image = Image.open(image_file).convert('RGB')
            ori_width, ori_height = image.size
            sam2_image = np.array(image)
            sam2_image = sam2_image_processor.apply_image(sam2_image)
            sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
            sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
            sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

            _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)

            # try:
            #     _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
            # except Exception as e:
            #     # print("quant_ids: ", quant_ids)
            #     print("_pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)")
            #     print("Exception e: ", e)
            #     exit(0)
            #     pred_masks.append(None)
            #     continue
            _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
            _pred_masks = _pred_masks > 0.5

            try:
                _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
            except Exception as e:
                print("_pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)")
                print("Exception e: ", e)
                print("_pred_masks.shape: ", _pred_masks.shape)
                exit(0)
                pred_masks.append(None)
                continue
        
            # #==========VISUALIZE BAD CASE============
            # iou = mask_iou(torch.from_numpy(target_masks[text_idx:text_idx+1]), torch.from_numpy(_pred_masks))
            # if iou[0][0].item() < 1.1:
            #     uuid_str = str(uuid.uuid4())[:8]
            #     iou_str = "%.2f" % iou[0][0].item()

            #     quant_ids = quant_ids.squeeze(0).cpu().numpy().tolist()
            #     pred_quant_ids_str = '*'.join([str(_) for _ in quant_ids])

            #     with torch.no_grad():
            #         masks = [torch.from_numpy(m).unsqueeze(0).to(vq_sam2.device) for m in target_masks[text_idx:text_idx+1]]
            #         try:
            #             boxes = torchvision.ops.masks_to_boxes(torch.cat(masks))
            #         except:
            #             print("Error at boxes = torchvision.ops.masks_to_boxes(masks)")
            #             continue

            #         whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
            #         boxes = boxes / whwh.to(boxes.device)
            #         boxes = boxes.to(vq_sam2.device)
            #         vq_sam2_output = vq_sam2(
            #             sam2_pixel_values,
            #             masks,
            #             boxes,
            #         )
            #         gt_quant_codes = vq_sam2_output.quant_codes
            #         gt_quant_codes = gt_quant_codes.cpu().numpy().astype(np.int32).tolist()[0][0]
            #         reconstruct_masks = vq_sam2_output.pred_masks
            #         reconstruct_masks = torch.nn.functional.interpolate(reconstruct_masks, size=(ori_height, ori_width), mode='bilinear')
            #         reconstruct_masks = reconstruct_masks > 0.5
            #         reconstruct_masks = reconstruct_masks[0].cpu().numpy()
            #         gt_quant_codes_str = "*".join([str(_) for _ in gt_quant_codes])

            #         #======two token
            #         one_gt_quant_codes = [gt_quant_codes[0], -1]
            #         one_gt_quant_codes = torch.LongTensor(one_gt_quant_codes).to(vq_sam2.device).unsqueeze(0)
            #         one_token_pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, one_gt_quant_codes)
            #         one_token_pred_masks = torch.nn.functional.interpolate(one_token_pred_masks, size=(ori_height, ori_width), mode='bilinear')
            #         one_token_pred_masks = one_token_pred_masks > 0.5
            #         one_token_pred_masks = one_token_pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)

            #         #=======pred first token
            #         two_pred_quant_codes = [quant_ids[0], -1]
            #         two_pred_quant_codes = torch.LongTensor(two_pred_quant_codes).to(vq_sam2.device).unsqueeze(0)
            #         two_pred_token_masks = vq_sam2.forward_with_codes(sam2_pixel_values, two_pred_quant_codes)
            #         two_pred_token_masks = torch.nn.functional.interpolate(two_pred_token_masks, size=(ori_height, ori_width), mode='bilinear')
            #         two_pred_token_masks = two_pred_token_masks > 0.5
            #         two_pred_token_masks = two_pred_token_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)

            #         #=======pred second token
            #         second_pred_quant_codes = [-1, quant_ids[1]]
            #         second_pred_quant_codes = torch.LongTensor(second_pred_quant_codes).to(vq_sam2.device).unsqueeze(0)
            #         second_pred_quant_masks = vq_sam2.forward_with_codes(sam2_pixel_values, second_pred_quant_codes)
            #         second_pred_quant_masks = torch.nn.functional.interpolate(second_pred_quant_masks, size=(ori_height, ori_width), mode='bilinear')
            #         second_pred_quant_masks = second_pred_quant_masks > 0.5
            #         second_pred_quant_masks = second_pred_quant_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)

            #         #=======second token
            #         second_gt_quant_codes = [-1, gt_quant_codes[1]]
            #         second_gt_quant_codes = torch.LongTensor(second_gt_quant_codes).to(vq_sam2.device).unsqueeze(0)
            #         second_gt_quant_masks = vq_sam2.forward_with_codes(sam2_pixel_values, second_gt_quant_codes)
            #         second_gt_quant_masks = torch.nn.functional.interpolate(second_gt_quant_masks, size=(ori_height, ori_width), mode='bilinear')
            #         second_gt_quant_masks = second_gt_quant_masks > 0.5
            #         second_gt_quant_masks = second_gt_quant_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)

            #     output_image = visualize(image, _pred_masks, ['']*_pred_masks.shape[0])
            #     output_image.save(f'./coco_val_recon/{img_id}_{uuid_str}_{iou_str}_pred_{pred_quant_ids_str}.jpg')

            #     output_image = visualize(image, target_masks[text_idx:text_idx+1], [''])
            #     output_image.save(f'./coco_val_recon/{img_id}_{uuid_str}_{iou_str}_gt.jpg')

            #     output_image = visualize(image, reconstruct_masks, ['']*reconstruct_masks.shape[0])
            #     output_image.save(f'./coco_val_recon/{img_id}_{uuid_str}_{iou_str}_reco_{gt_quant_codes_str}.jpg')

            #     output_image = visualize(image, one_token_pred_masks, ['']*one_token_pred_masks.shape[0])
            #     output_image.save(f'./coco_val_recon/{img_id}_{uuid_str}_{iou_str}_gt_firsttoken.jpg')

            #     output_image = visualize(image, second_gt_quant_masks, ['']*second_gt_quant_masks.shape[0])
            #     output_image.save(f'./coco_val_recon/{img_id}_{uuid_str}_{iou_str}_gt_secondtoken.jpg')

            #     output_image = visualize(image, two_pred_token_masks, ['']*two_pred_token_masks.shape[0])
            #     output_image.save(f'./coco_val_recon/{img_id}_{uuid_str}_{iou_str}_pred_firsttoken.jpg')

            #     output_image = visualize(image, second_pred_quant_masks, ['']*second_pred_quant_masks.shape[0])
            #     output_image.save(f'./coco_val_recon/{img_id}_{uuid_str}_{iou_str}_pred_secondtoken.jpg')

            #     with open(f'./coco_val_recon/{img_id}_{uuid_str}_{iou_str}_gt.txt', 'w', encoding='utf-8') as file:
            #         file.write(question)
            
            _pred_masks = mask_to_rle(_pred_masks)
            pred_masks.append(_pred_masks)
            
        prediction.update({'prediction_masks': pred_masks})
        img_id = prediction['img_id']
        with open(f"./temp_save/{args.dataset}/{img_id}.json", "w") as f:
            json.dump(prediction, f)
        # results.append(prediction)
    # exit(0)
    results = []
    for json_file in os.listdir(f"./temp_save/{args.dataset}"):
        json_path = os.path.join(f"./temp_save/{args.dataset}", json_file)
        with open(json_path, 'r') as f:
            prediction = json.load(f)
            skip_this_one = False
            for pred_mask in prediction['prediction_masks']:
                if pred_mask is None:
                    skip_this_one = True
            if skip_this_one:
                continue
            else:
                results.append(prediction)
    print("=================, left items: ", len(results))
    tmpdir = './dist_test_temp_res_' + args.dataset + args.split + args.model_path.replace('/', '').replace('.', '')
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        metric = dataset.evaluate(results, './work_dirs')
        print(metric)

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

if __name__ == '__main__':
    main()