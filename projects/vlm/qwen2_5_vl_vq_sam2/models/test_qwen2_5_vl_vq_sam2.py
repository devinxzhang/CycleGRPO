import re
import copy
from PIL import Image
import numpy as np
from xtuner.model.utils import guess_load_checkpoint

import torch
import transformers

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize

import matplotlib.pyplot as plt
import cv2
import random

def visualize_masks_on_image(image, masks, save_path, alpha=0.5, draw_contour=True, contour_color=(255, 255, 255), contour_thickness=1):
    """
    将多个 mask 可视化叠加到图像上。

    Args:
        image (np.ndarray): 原始图像，shape=(H, W, 3) or (H, W)，值范围[0,255]。
        masks (np.ndarray): shape=(N, H, W)，每个 mask 是 np.uint8 类型的二值图。
        alpha (float): mask 的透明度（0~1）。
        draw_contour (bool): 是否绘制 mask 的轮廓。
        contour_color (tuple): 轮廓颜色 (B, G, R)。
        contour_thickness (int): 轮廓线宽。
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2BGR)

    image = image.copy()
    H, W = image.shape[:2]
    N = masks.shape[0]

    for i in range(N):
        mask = masks[i]
        color = [random.randint(0, 255) for _ in range(3)]
        colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]

        # 叠加mask
        image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

        if draw_contour:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, contour_color, contour_thickness)

    # 显示图像
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)



def extract_mt_token_ids(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x.lstrip("0")) for x in re.findall(pattern, text)]

if __name__ == "__main__":
    model_path = "./work_dirs/qwen2_5_vl_sft_on_llava_insseg335k_dam963k"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(model_path)

    # image_file = "<PATH_TO_DATA>/discrete_spatial_tokenizer/zhouyik/zt_any_visual_prompt/sa_9684593.jpg"
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": image_file,
    #             },
    #             {"type": "text", "text": "Segment the man wearing a white hat in the image."},
    #         ],
    #     }
    # ]

    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )

    # image_inputs, video_inputs = process_vision_info(messages)
    # inputs = processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    # inputs = inputs.to("cuda")

    # # Inference: Generation of the output
    # generated_ids = model.generate(
    #     **inputs, 
    #     max_new_tokens=128,
    # )
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    # )
    # print("Assistant: ", output_text)

    # quant_ids = extract_mt_token_ids(output_text[0])
    # if len(quant_ids) == 0:
    #     exit(0)
    # assert len(quant_ids) % 4 == 0
    # batch_size = len(quant_ids) // 4
    # remap_quant_ids = []
    # for bs_id in range(batch_size):
    #     chunk_quant_ids = quant_ids[bs_id*4:(bs_id+1)*4]
    #     remap_chunk_quant_ids = [quant_id - book_id*1024 for book_id, quant_id in enumerate(chunk_quant_ids)]
    #     remap_quant_ids.append(remap_chunk_quant_ids)


    # # vq-sam2
    # sam2_config = SAM2Config(
    #     cfg_path="sam2.1_hiera_l.yaml",
    #     ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
    # )

    # vq_sam2_config = VQ_SAM2Config(
    #     sam2_config=sam2_config,
    #     codebook_size=1024,
    #     codebook_depth=4,
    #     shared_codebook=False,
    #     latent_dim=256,
    # )

    # vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    # pretrained_pth = "pretrained_weights/vq_sam2_2M/iter_44916.pth"
    # pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

    # pretrained_state_dict_new = {}
    # for key in pretrained_state_dict.keys():
    #     new_key = copy.deepcopy(key)
    #     if key.startswith('hf_model.'):
    #         new_key = new_key[len('hf_model.'):]
    #     pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
    
    # vq_sam2.load_state_dict(pretrained_state_dict_new)

    # sam2_image_processor = DirectResize(1024)

    # image = Image.open(image_file).convert('RGB')
    # ori_width, ori_height = image.size
    # sam2_image = np.array(image)
    # sam2_image = sam2_image_processor.apply_image(sam2_image)
    # sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
    # sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
    # sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)
    
    # quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

    # with torch.no_grad():
    #     pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
    # pred_masks = torch.nn.functional.interpolate(pred_masks, size=(ori_height, ori_width), mode='bilinear')
    # pred_masks = pred_masks > 0.5

    # pred_masks = pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
    # visualize_image = cv2.imread(image_file)
    # visualize_masks_on_image(visualize_image, pred_masks, "./test_qwen25vl_vq_sam2.jpg")
    # exit(0)


    # test mask to text
    image_file = "<PATH_TO_DATA>/discrete_spatial_tokenizer/zhouyik/zt_any_visual_prompt/sa_9684593.jpg"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_file,
                },
                {"type": "text", "text": "<|mt_start|><|mt_0071|><|mt_1304|><|mt_2728|><|mt_3572|><|mt_end|>\nProvide a detailed description of the masked region."},
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

    # Inference: Generation of the output
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=128,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    print("Assistant: ", output_text)





