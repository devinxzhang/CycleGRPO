import os
import json
import uuid
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def decode_mask(object_masks, ori_height, ori_width):
    """object_masks: list[dict|list] (coco格式的segmentation，可为RLE或多边形)
       return: np.uint8, shape (N, H, W), 值为{0,1}
    """
    binary_masks = []
    for object_mask in object_masks:
        if isinstance(object_mask, dict):
            # RLE
            if isinstance(object_mask.get("counts", None), list):
                object_mask = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
            m = mask_utils.decode(object_mask)  # (H,W,1) or (H,W)
            m = m.astype(np.uint8).squeeze()
        elif object_mask:
            # 多边形列表
            rles = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
            rle = mask_utils.merge(rles)
            m = mask_utils.decode(rle).astype(np.uint8).squeeze()
        else:
            m = np.zeros((ori_height, ori_width), dtype=np.uint8)
        binary_masks.append(m)
    if len(binary_masks) == 0:
        binary_masks.append(np.zeros((ori_height, ori_width), dtype=np.uint8))
    masks = np.stack(binary_masks, axis=0)
    return masks


def overlay_masks(image: Image.Image, masks: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    在PIL图像上叠加(N,H,W)的二值mask；
    alpha: 叠加透明度[0,1]
    """
    # 保证RGBA
    base = image.convert("RGBA")
    H, W = base.height, base.width

    # 准备一个空的RGBA叠加层
    overlay = np.zeros((H, W, 4), dtype=np.uint8)

    # 调色板（可自行扩展/替换）
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 128, 255),
        (255, 128, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 255), (0, 255, 128), (255, 0, 128),
        (128, 128, 0), (0, 128, 128), (128, 0, 128),
    ]

    # 逐实例着色
    num_inst = masks.shape[0]
    for i in range(num_inst):
        color = palette[i % len(palette)]
        m = masks[i]  # (H,W) 0/1
        if m.max() == 0:
            continue
        # 上色
        overlay[m > 0, 0] = color[0]
        overlay[m > 0, 1] = color[1]
        overlay[m > 0, 2] = color[2]
        overlay[m > 0, 3] = int(alpha * 255)

    overlay_img = Image.fromarray(overlay, mode="RGBA")
    composed = Image.alpha_composite(base, overlay_img)
    return composed


def main():
    json_path = './data/coconut_segments.json'
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    os.makedirs('./', exist_ok=True)

    # 这里示例只处理前10个，可自行去掉[:10]以处理全部
    for data_dict in json_data[:10]:
        image_file = data_dict['image_file']
        # 打开图像并获取尺寸
        img = Image.open(image_file).convert("RGB")
        ori_width, ori_height = img.size

        # 有些数据是单个实例，有些是多个实例，统一转成 list
        # 你的原代码是 segms = [data_dict['segmentation']]
        # 如果 segmentation 本身就是一个 list(多实例)，直接用它；否则包一层
        seg = data_dict['segmentation']
        segms = [seg]

        masks = decode_mask(segms, ori_height, ori_width)  # (N,H,W)

        vis_img = overlay_masks(img, masks, alpha=0.5)

        # 随机文件名
        out_name = uuid.uuid4().hex + ".png"
        out_path = os.path.join("./", out_name)
        vis_img.save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
