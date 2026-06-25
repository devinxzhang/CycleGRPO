import os
import shutil
import tqdm


sft_file_dict = {}
sft_iou_dict = {}

for image_file in os.listdir('./CVPR2026/demo_gres'):
    if not image_file.endswith('.jpg'):
        continue
    image_path = os.path.join('./CVPR2026/demo_gres', image_file)
    image_name = image_file.replace('.jpg', '')

    prefix, iou_part = image_name.rsplit("_iou", 1)
    sft_file_dict[prefix] = image_path
    sft_iou_dict[prefix] = int(iou_part)

for image_file in tqdm.tqdm(os.listdir('./CVPR2026/demo_gres_rl')):
    if not image_file.endswith('.jpg'):
        continue
    image_path = os.path.join('./CVPR2026/demo_gres_rl', image_file)
    image_name = image_file.replace('.jpg', '')

    prefix, iou_part = image_name.rsplit("_iou", 1)

    if prefix in sft_iou_dict and int(iou_part) - sft_iou_dict[prefix] > 20:
        print("get one!!!")
        # sft image
        shutil.copy2(sft_file_dict[prefix], "./CVPR2026/demo_sft_vs_rl")
        # sft txt
        shutil.copy2(sft_file_dict[prefix].replace('.jpg', '.txt'), "./CVPR2026/demo_sft_vs_rl")
        # rl image
        shutil.copy2(image_path, "./CVPR2026/demo_sft_vs_rl")
        # rl txt
        shutil.copy2(image_path.replace('.jpg', '.txt'), "./CVPR2026/demo_sft_vs_rl")

