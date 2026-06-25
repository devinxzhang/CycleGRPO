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


# QUESTION_CLASS_LIST = [
#     "Provide the mask(s) for all {CAT_INFO} in the image. If masks cannot be generated, supply the bounding box(es) for each {CAT_INFO} instead. Output results in JSON.",
#     "Ground all {CAT_INFO} in the photo: prioritize mask output for each {CAT_INFO}. If masks are unavailable, return the bounding box for every {CAT_INFO}. Output results in JSON.",

# ]

# QUESTION_DETAIL_LIST = [
#     "For the object described as \"{CAT_INFO}\", first generate its mask. If mask creation is not feasible, provide the corresponding bounding box. Output results in JSON.",
#     "Locate the object matching the description \"{CAT_INFO}\" — first produce its mask. If mask generation fails, provide the bounding box. Output results in JSON.",
# ]

# QUESTION_BBOX_LIST = [

# ]

QUESTION_LIST = [
    "<image>\nSegment every instance that belongs to the following categories: {class_name}",
    "<image>\nLocate every instance that belongs to the following categories: {class_name}. Report segmentation masks in JSON format."
]

def main(task_id):
    sam2_checkpoint = "pretrained_weights/sam2.1_hiera_large.pt"
    model_cfg = "sam2.1_hiera_l.yaml"

    sam2_model = build_sam2_ori(model_cfg, sam2_checkpoint, device='cuda')

    predictor = SAM2ImagePredictor(sam2_model)

    # load
    from pycocotools.coco import COCO

    data_path = "./data/V3Det/v3det_2023_v1_train.json"
    image_folder = "./data/V3Det"

    coco_api = COCO(data_path)
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    id_2_class_name = {c["id"]: c["name"] for c in sorted(cats, key=lambda x: x["id"])}
    id_2_cat_info = {c["id"]: c["cat_info_gpt4v"] for c in sorted(cats, key=lambda x: x["id"])}

    # img_ids = sorted(coco_api.imgs.keys())
    # imgs = coco_api.loadImgs(img_ids)
    # anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    # imgs_anns = list(zip(imgs, anns))

    # dataset_dicts = []

    # ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"]

    # for img_dict, anno_dict_list in imgs_anns:
    #     record = {}
    #     record["file_name"] = os.path.join(image_folder, img_dict["file_name"])
    #     record["height"] = img_dict["height"]
    #     record["width"] = img_dict["width"]
    #     image_id = record["image_id"] = img_dict["id"]

    #     objs = []
    #     for anno in anno_dict_list:
    #         # Check that the image_id in this annotation is the same as
    #         # the image_id we're looking at.
    #         # This fails only when the data parsing logic or the annotation file is buggy.

    #         # The original COCO valminusminival2014 & minival2014 annotation files
    #         # actually contains bugs that, together with certain ways of using COCO API,
    #         # can trigger this assertion.
    #         assert anno["image_id"] == image_id

    #         assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

    #         obj = {key: anno[key] for key in ann_keys if key in anno}
    #         if "bbox" in obj and len(obj["bbox"]) == 0:
    #             raise ValueError(
    #                 f"One annotation of image {image_id} contains empty 'bbox' value! "
    #                 "This json does not have valid COCO format."
    #             )
    #         obj.update({
    #             'category_name': id_2_class_name[anno['category_id']],
    #             'cat_info_gpt4v': id_2_cat_info[anno['category_id']]
    #         })
            
    #         objs.append(obj)
    #     record["annotations"] = objs
    #     dataset_dicts.append(record)

    # with open('./data/v3det_bbox_info.json', 'w') as f:
    #     json.dump(dataset_dicts, f)
    # exit(0)


    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

    temp_save_root = "./temp_data_256x2_0927/v3det_grounding/"
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

    with open('./data/v3det_bbox_info.json', 'r') as f:
        v3det_dataset = json.load(f)
    
    chunk_idx = task_id
    n = len(v3det_dataset)
    chunk_size = (n+31) // 32
    start = chunk_idx * chunk_size
    end = min((chunk_idx + 1) * chunk_size, n)
    indices_list = list(range(len(v3det_dataset)))[start:end]
    for idx in tqdm(indices_list):
        data_dict = v3det_dataset[idx]
        image_file = data_dict['file_name']
        image_id = os.path.basename(image_file).split('.')[0]
        image = Image.open(image_file).convert('RGB')
        ori_width, ori_height = image.size

        if os.path.exists(os.path.join(temp_save_root, f"{image_id}.json")):
            continue

        bbox_list = []
        class_id_list = []
        for anno in data_dict['annotations']:
            bbox_list.append(anno["bbox"])
            class_id_list.append(anno["category_id"])
        bbox_xywh_np = np.array(bbox_list)
        bbox_x1y1x2y2_np = xywh_to_x1y1x2y2(bbox_xywh_np)
        whwh = np.array([[ori_width, ori_height, ori_width, ori_height]])
        bbox_x1y1x2y2_np_normalized = bbox_x1y1x2y2_np / whwh

        if len(bbox_x1y1x2y2_np) == 0:
            continue

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
        categories_name_to_masks = {}
        for class_id, mask, box in zip(class_id_list, sam_masks, bbox_x1y1x2y2_np_normalized):
            if class_id not in categories_name_to_masks:
                categories_name_to_masks[class_id] = []
            categories_name_to_masks[class_id].append((mask, box))
        
        conversation = []
        answer = "```json\n[{mask_2d}]\n```"
        mask_2d_str = ''
        class_names = []
        for category_id, category_masks_boxes in categories_name_to_masks.items():
            class_name = id_2_class_name[category_id].replace('/', '-')
            if os.path.exists(os.path.join(temp_save_root, f"{image_id}_{class_name}.json")):
                continue

            sam2_image = np.array(image)
            sam2_image = sam2_image_processor.apply_image(sam2_image)
            sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
            sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

            category_masks = [_mask_ for (_mask_, _box_) in category_masks_boxes]
            category_boxes = [_box_ for (_mask_, _box_) in category_masks_boxes]

            masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in category_masks])
            boxes = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in category_boxes])

            if len(masks) == 0:
                print("len(masks) == 0!!!")
                continue

            try:
                order = sort_mask_indices(boxes, mode="ltr-ttb")
            except:
                order = np.arange(masks.shape[0])
            
            masks = masks[torch.as_tensor(order, dtype=torch.long)]

            try:
                prompt_boxes = torchvision.ops.masks_to_boxes(masks)
            except:
                print("Error at boxes = torchvision.ops.masks_to_boxes(masks)")
                continue
            whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
            prompt_boxes = prompt_boxes / whwh
            prompt_boxes = prompt_boxes.to(vq_sam2.device)

            masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
            num_ins = len(masks)

            skip_this_one = False
            try:
                with torch.no_grad():
                    vq_sam2_output = vq_sam2(
                        sam2_pixel_values.repeat(num_ins, 1, 1, 1),
                        masks,
                        prompt_boxes,
                        reconstruct_mask=False,
                    )
                    quant_codes = vq_sam2_output.quant_codes
                    # pred_masks = vq_sam2_output.pred_masks
            except torch.OutOfMemoryError:
                print("num_ins is too large: ", num_ins, "; will be split into blocks (size 10)")
                NUM_BLOCKS = num_ins // 10
                if NUM_BLOCKS * 10 < num_ins:
                    NUM_BLOCKS += 1
                block_quant_codes = []
                block_pred_masks = []
                for block_idx in range(NUM_BLOCKS):
                    start_idx = block_idx * 10
                    end_idx = min(start_idx + 10, num_ins)
                    try:
                        with torch.no_grad():
                            vq_sam2_output = vq_sam2(
                                sam2_pixel_values[start_idx:end_idx],
                                masks[start_idx:end_idx],
                                prompt_boxes[start_idx:end_idx],
                                reconstruct_mask=False,
                            )
                    except torch.OutOfMemoryError:
                        skip_this_one = True
                        break
                    block_quant_codes.append(vq_sam2_output.quant_codes)
                    # block_pred_masks.append(vq_sam2_output.pred_masks)
                if skip_this_one:
                    continue
                quant_codes = torch.cat(block_quant_codes, dim=0)
                # pred_masks = torch.cat(block_pred_masks, dim=0)

                # print("num_ins is too large: ", num_ins)
                # exit(0)
            except Exception as e:
                # print("sam2_pixel_values.repeat(num_ins, 1, 1, 1).shape: ", sam2_pixel_values.repeat(num_ins, 1, 1, 1).shape)
                # import uuid
                # save_obj = {
                #     'image_file': image_file,
                #     'category_name': id_2_class_name[category_id],
                #     'cat_info_gpt4v': id_2_cat_info[category_id],
                #     "height": ori_height,
                #     "width": ori_width,
                # }

                # rles = []
                # for m in masks:
                #     m_np = m.detach().to("cpu").squeeze(0).numpy().astype(np.uint8)
                #     rle = mask_utils.encode(np.asfortranarray(m_np))
                #     if isinstance(rle["counts"], bytes):
                #         rle["counts"] = rle['counts'].decode("ascii")
                #     rles.append(rle)
                # save_obj['segmentation'] = rles

                # random_tag = uuid.uuid4().hex[:8]
                # _save_path = os.path.join("./temp_data/too_many_obj_cases", f"{image_id}_{random_tag}.json")
                # with open(_save_path, "w") as f:
                #     json.dump(save_obj, f)
                # print(f"[fallback saved] {_save_path}")
                continue
            
            if len(quant_codes) == 0:
                continue
            
            quant_codes = quant_codes.cpu().numpy().astype(np.int32).tolist()
            remap_quant_codes = []
            for _quant_codes in quant_codes:
                _quant_codes = _quant_codes[0]
                remap_quant_codes.append([depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(_quant_codes)])
            quant_codes = remap_quant_codes

            # if random.random() < 0.5:
            #     question = random.choice(QUESTION_CLASS_LIST).format(CAT_INFO=id_2_class_name[category_id])
            #     cat_info = id_2_class_name[category_id]
            # else:
            #     question = random.choice(QUESTION_DETAIL_LIST).format(CAT_INFO=id_2_cat_info[category_id])
            #     cat_info = id_2_cat_info[category_id]

            # # verify the quality of the quant_codes
            # # pred_masks = vq_sam2_output.pred_masks
            # pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
            # pred_masks = pred_masks > 0.5
            # skip_this_one = False
            # answer = "```json\n["
            # bbox_idx = 0
            # needed_bbox_dict = {}
            # for pred_mask, target_mask, target_box, item_quant_codes in zip(pred_masks, masks, boxes, quant_codes):
            #     iou = mask_iou(pred_mask, target_mask)
            #     if iou[0][0].item() < 0.5:
            #         item_str = "{\"mask_2d\": [" + ', '.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in item_quant_codes]) + "], \"self-inspection\": \"reject\"}" 
            #         answer += item_str + ",\n"
            #         bbox_idx_str = f"bbox_2d_" + str(bbox_idx).zfill(4)
            #         item_str = "{\"bbox_2d\": " + bbox_idx_str + ", \"self-inspection\": \"none\"}"
            #         answer += item_str + ",\n"
            #         needed_bbox_dict.update({bbox_idx_str: target_box.cpu().numpy().tolist()})
            #         bbox_idx += 1
            #     else:
            #         item_str = "{\"mask_2d\": [" + ', '.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in item_quant_codes]) + "], \"self-inspection\": \"accept\"}" 
            #         answer += item_str + ",\n"    
            # answer = answer[:-len(",\n")] + "]\n```"

            # conversation.append({'from': 'human', 'value': question})
            # if bbox_idx > 0:
            #     conversation.append({'from': 'gpt', 'value': answer, 'bbox_2d': needed_bbox_dict})
            # else:
            #     conversation.append({'from': 'gpt', 'value': answer})
            
            # ret_data_dict = {
            #     'image': image_file,
            #     'conversations': conversation,
            # }
            
            # with open(os.path.join(temp_save_root, f"{image_id}_{class_name}.json"), 'w') as f:
            #     json.dump(ret_data_dict, f)

            for _quant_codes_ in quant_codes:
                sam2_tokens = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in _quant_codes_]) + MT_END_TOKEN
                item_str = "{\"mask_2d\": " + sam2_tokens + ", \"label\": \"" + class_name + "\"}"
                mask_2d_str += item_str + ",\n"
                if class_name not in class_names:
                    class_names.append(class_name)
        
        if mask_2d_str == '':
            continue

        mask_2d_str = mask_2d_str[:-len(",\n")]
        answer = answer.format(mask_2d=mask_2d_str)

        category_name_str = ', '.join(class_names)
        question = random.choice(QUESTION_LIST).format(class_name=category_name_str)

        conversation.append({'from': 'human', 'value': question})
        conversation.append({'from': 'gpt', 'value': answer})
        
        ret_data_dict = {
            'image': image_file,
            'conversations': conversation,
        }
        
        with open(os.path.join(temp_save_root, f"{image_id}.json"), 'w') as f:
            json.dump(ret_data_dict, f)

if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)
