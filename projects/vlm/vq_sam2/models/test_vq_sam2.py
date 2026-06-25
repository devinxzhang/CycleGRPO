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

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from projects.vlm.vq_sam2.models import DirectResize
from projects.vlm.vq_sam2.datasets import CoCoPanoSegValDataset
from projects.vlm.vq_sam2.datasets.coco_category import COCO_CATEGORIES

from transformers import AutoProcessor

from xtuner.model.utils import guess_load_checkpoint

def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union

    
if __name__ == "__main__":

    # all_quant_codes = []
    # for json_file in os.listdir('./codebook_utilization_statistics'):
    #     if json_file.endswith('.json'):
    #         with open(f'./codebook_utilization_statistics/{json_file}', 'r') as f:
    #             quant_codes = json.load(f)
    #         quant_codes = torch.tensor(quant_codes)
    #         all_quant_codes.append(quant_codes)
    # all_quant_codes = torch.cat(all_quant_codes, dim=0) # (N, 4)
    # print(torch.unique(all_quant_codes))
    # exit(0)

    # fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # fig.suptitle('VQ-SAM2 Quantization Codes Histogram Analysis', fontsize=16)
    
    # # Analyze each column separately (depth0~3)
    # for depth in range(4):
    #     col_data = all_quant_codes[:, depth].numpy()
        
    #     # Calculate histogram
    #     hist, bins = np.histogram(col_data, bins=50, range=(0, 1024))
        
    #     # Plot histogram
    #     ax = axes[0, depth] if depth < 3 else axes[1, 0]
    #     ax.bar(bins[:-1], hist, width=bins[1]-bins[0], alpha=0.7, color=f'C{depth}')
    #     ax.set_title(f'Depth {depth} Histogram')
    #     ax.set_xlabel('Codebook Index')
    #     ax.set_ylabel('Frequency')
    #     ax.grid(True, alpha=0.3)
        
    #     # Add statistics
    #     unique_codes = len(np.unique(col_data))
    #     total_codes = len(col_data)
    #     utilization_rate = unique_codes / 1024 * 100
    #     ax.text(0.02, 0.98, f'Utilization: {utilization_rate:.1f}%\nUnique codes: {unique_codes}\nTotal codes: {total_codes}', 
    #             transform=ax.transAxes, verticalalignment='top',
    #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # # Analyze all elements together (overall)
    # overall_data = all_quant_codes.flatten().numpy()
    # hist, bins = np.histogram(overall_data, bins=50, range=(0, 1024))
    
    # ax = axes[1, 1]
    # ax.bar(bins[:-1], hist, width=bins[1]-bins[0], alpha=0.7, color='red')
    # ax.set_title('Overall Histogram')
    # ax.set_xlabel('Codebook Index')
    # ax.set_ylabel('Frequency')
    # ax.grid(True, alpha=0.3)
    
    # # Add statistics
    # unique_codes = len(np.unique(overall_data))
    # total_codes = len(overall_data)
    # utilization_rate = unique_codes / 1024 * 100
    # ax.text(0.02, 0.98, f'Utilization: {utilization_rate:.1f}%\nUnique codes: {unique_codes}\nTotal codes: {total_codes}', 
    #         transform=ax.transAxes, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # # Add detailed statistics table
    # ax = axes[1, 2]
    # ax.axis('off')
    
    # # Calculate statistics for each depth
    # stats_data = []
    # for depth in range(4):
    #     col_data = all_quant_codes[:, depth].numpy()
    #     unique_codes = len(np.unique(col_data))
    #     total_codes = len(col_data)
    #     utilization_rate = unique_codes / 1024 * 100
    #     mean_val = np.mean(col_data)
    #     std_val = np.std(col_data)
    #     stats_data.append([f'Depth {depth}', f'{utilization_rate:.1f}%', unique_codes, f'{mean_val:.1f}', f'{std_val:.1f}'])
    
    # # Add overall statistics
    # unique_codes = len(np.unique(overall_data))
    # total_codes = len(overall_data)
    # utilization_rate = unique_codes / 1024 * 100
    # mean_val = np.mean(overall_data)
    # std_val = np.std(overall_data)
    # stats_data.append(['Overall', f'{utilization_rate:.1f}%', unique_codes, f'{mean_val:.1f}', f'{std_val:.1f}'])
    
    # # Create table
    # table = ax.table(cellText=stats_data,
    #                 colLabels=['Dimension', 'Utilization', 'Unique Codes', 'Mean', 'Std'],
    #                 cellLoc='center',
    #                 loc='center')
    # table.auto_set_font_size(False)
    # table.set_fontsize(10)
    # table.scale(1, 2)
    
    # # Set table style
    # for i in range(len(stats_data) + 1):
    #     for j in range(5):
    #         if i == 0:  # Header
    #             table[(i, j)].set_facecolor('#4CAF50')
    #             table[(i, j)].set_text_props(weight='bold', color='white')
    #         else:
    #             table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    # plt.tight_layout()
    # plt.savefig('./quant_codes_histogram_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # # Print detailed statistics
    # print("\n=== Detailed Statistics ===")
    # print(f"{'Dimension':<12} {'Utilization':<12} {'Unique Codes':<12} {'Mean':<10} {'Std':<10}")
    # print("-" * 60)
    
    # for depth in range(4):
    #     col_data = all_quant_codes[:, depth].numpy()
    #     unique_codes = len(np.unique(col_data))
    #     total_codes = len(col_data)
    #     utilization_rate = unique_codes / 1024 * 100
    #     mean_val = np.mean(col_data)
    #     std_val = np.std(col_data)
    #     print(f"{f'Depth {depth}':<12} {f'{utilization_rate:.1f}%':<12} {unique_codes:<12} {f'{mean_val:.1f}':<10} {f'{std_val:.1f}':<10}")
    
    # unique_codes = len(np.unique(overall_data))
    # total_codes = len(overall_data)
    # utilization_rate = unique_codes / 1024 * 100
    # mean_val = np.mean(overall_data)
    # std_val = np.std(overall_data)
    # print(f"{'Overall':<12} {f'{utilization_rate:.1f}%':<12} {unique_codes:<12} {f'{mean_val:.1f}':<10} {f'{std_val:.1f}':<10}")

    # exit(0)

    # all_results = []
    # for json_file in os.listdir('./reconstruction_eval_results/vq_sam2_cocopano_depth4_unshare_codebook_laten_dim_256_mask_token_1_thing'):
    #     if json_file.endswith('.json'):
    #         with open(f'./reconstruction_eval_results/vq_sam2_cocopano_depth4_unshare_codebook_laten_dim_256_mask_token_1_thing/{json_file}', 'r') as f:
    #             results = json.load(f)
    #             all_results.extend(results)
    # print("Mean Mask IoU: ", np.mean(all_results))
    # exit(0)

    
    sam2_config = SAM2Config(
        cfg_path="sam2.1_hiera_l.yaml",
        ckpt_path="pretrained_weights/sam2.1_hiera_large.pt",
    )

    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=256,
        codebook_depth=4,
        shared_codebook=False,
        latent_dim=256,
    )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    pretrained_pth = "./pretrained_weights/iter_17923_resampled_256x4.pth"
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

    if not os.path.exists('./codebook_utilization_statistics/vq_sam2_1024x1'):
        os.makedirs('./codebook_utilization_statistics/vq_sam2_1024x1')
    
    eval_results_root = "./reconstruction_eval_results/vq_sam2_1024x1/"
    if not os.path.exists(eval_results_root):
        os.makedirs(eval_results_root)

    isthing_dict = {e['name']: e['isthing'] for e in COCO_CATEGORIES}

    all_iou = []
    max_ins = 1000
    for idx in tqdm(range(len(val_dataset))):
        # if max_ins == 0:
        #     break
        data = val_dataset[idx]
        image_file = data['image_file']
        image_name = os.path.basename(image_file).split('.jpg')[0]
        masks = data['masks']
        class_names = data['class_names']
        image = Image.open(image_file)
        width, height = image.size
        all_quant_codes = []
        if os.path.exists(f'./codebook_utilization_statistics/vq_sam2_1024x1/{image_name}.json'):
            continue

        
        # #===============Multiple Tragets==================
        # # 随机选择2到5个mask，如果masks数量少于5个，则使用所有masks
        # category_masks = {}
        # for mask, class_name in zip(masks, class_names):
        #     if class_name not in category_masks:
        #         category_masks[class_name] = []
        #     category_masks[class_name].append(mask)

        # # category_stuff_masks = {}
        # # for class_name, masks in category_masks.items():
        # #     if not isthing_dict[class_name]:
        # #         category_stuff_masks[class_name] = (np.sum(np.stack(masks, axis=0), axis=0) > 0).astype(np.uint8)
        
        # category_stuff_masks = {}
        # for class_name, masks in category_masks.items():
        #     if not isthing_dict[class_name] or len(masks) == 1:
        #         continue
        #     category_stuff_masks[class_name] = (np.sum(np.stack(masks, axis=0), axis=0) > 0).astype(np.uint8)
        
        # if len(category_stuff_masks) == 0:
        #     continue

        # class_names = list(category_stuff_masks.keys())
        # masks = list(category_stuff_masks.values())

        # select_idx = random.randint(0, len(category_stuff_masks) - 1)
        # class_name = class_names[select_idx]
        # mask = masks[select_idx]

        # val_item = val_dataset.prepare_mask_input(image_file, mask, class_name)
        # pixel_values = val_item['pixel_values']
        # masks = val_item['masks']
        # boxes = val_item['boxes'].to(vq_sam2.device)

        # pixel_values = pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
        # masks = [masks.to(vq_sam2.device)]

        # vq_sam2_output = vq_sam2(
        #     pixel_values,
        #     masks,
        #     boxes,
        # )
        # all_quant_codes.append(vq_sam2_output.quant_codes.squeeze(0))

        # pred_masks = vq_sam2_output.pred_masks
        # pred_masks = torch.nn.functional.interpolate(pred_masks, size=(height, width), mode='bilinear')
        # pred_masks = pred_masks > 0.5
        # pred_masks = pred_masks[0].cpu().numpy().astype(np.uint8)
        # tags = [''] * pred_masks.shape[0]

        # output_image = visualize(image, pred_masks, tags)
        # output_image.save(f'./coco_val_recon/test_vq_sam2_multi_targets_{max_ins}.jpg')

        # output_image = visualize(image, mask[np.newaxis, :, :], tags)
        # output_image.save(f'./coco_val_recon/test_vq_sam2_multi_targets_{max_ins}_gt.jpg')
        # max_ins -= 1
        # #===============Multiple Tragets==================


        eval_result_path = os.path.join(eval_results_root, f'{image_name}.json')
        if os.path.exists(f'./codebook_utilization_statistics/vq_sam2_1024x1/{image_name}.json') and os.path.exists(eval_result_path):
            continue

        this_file_results = []
        for mask, class_name in zip(masks, class_names):
            
            # if not isthing_dict[class_name]:
            #     continue

            val_item = val_dataset.prepare_mask_input(image_file, mask, class_name)
            pixel_values = val_item['pixel_values']
            masks = val_item['masks']
            boxes = val_item['boxes'].to(vq_sam2.device)

            pixel_values = pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
            masks = [masks.to(vq_sam2.device)]

            with torch.no_grad():
                vq_sam2_output = vq_sam2(
                    pixel_values,
                    masks,
                    boxes,
                )
            all_quant_codes.append(vq_sam2_output.quant_codes.squeeze(0))

            pred_masks = vq_sam2_output.pred_masks
            pred_masks = torch.nn.functional.interpolate(pred_masks, size=(height, width), mode='bilinear')
            pred_masks = pred_masks > 0.5
            pred_masks = pred_masks[0].cpu().numpy().astype(np.uint8)
            # tags = [''] * pred_masks.shape[0]

            # output_image = visualize(image, pred_masks, tags)
            # output_image.save(f'./coco_val_recon/test_vq_sam2_{max_ins}.jpg')

            # output_image = visualize(image, mask[np.newaxis, :, :], tags)
            # output_image.save(f'./coco_val_recon/test_vq_sam2_{max_ins}_gt.jpg')
            # max_ins -= 1

            target_mask = masks[0].cpu().numpy().astype(np.uint8)

            iou = mask_iou(torch.from_numpy(target_mask), torch.from_numpy(pred_masks))
            all_iou.append(iou[0][0].item())
            this_file_results.append(iou[0][0].item())
        
        if len(all_quant_codes) == 0:
            continue
        with open(eval_result_path, 'w') as f:
            json.dump(this_file_results, f)

        all_quant_codes = torch.cat(all_quant_codes, dim=0)
        all_quant_codes = all_quant_codes.cpu().numpy().tolist()

        with open(f'./codebook_utilization_statistics/vq_sam2_1024x1/{image_name}.json', 'w') as f:
            json.dump(all_quant_codes, f)

    print("Mean Mask IoU: ", np.mean(all_iou))
