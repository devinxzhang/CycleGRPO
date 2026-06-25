import os
import json
import numpy as np
from pycocotools import mask as mask_utils

def encode_binary_mask(bin_mask_bool):
    # 跳过空 mask，避免 encode 的边界行为
    if not np.any(bin_mask_bool):
        return None
    # pycocotools 期望的是 Fortran 连续的 0/1 uint8，形状 HxW
    m = np.asfortranarray(bin_mask_bool.astype(np.uint8, copy=False))
    rle = mask_utils.encode(m)
    # 某些版本返回的是{'counts': bytes, 'size': [H, W]}
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def main():
    pixelweb_root = "./data/pixelweb_100k/annotated"
    temp_save_root = "./temp_data"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0
    for split_name in os.listdir(pixelweb_root):
        split_path = os.path.join(pixelweb_root, split_name)
        for img_file in os.listdir(split_path):
            if '-screenshot.png' not in img_file:
                continue
            mask_file = img_file.replace('-screenshot.png', '-mask.json')
            class_file = img_file.replace('-screenshot.png', '-class.json')
            with open(os.path.join(split_path, mask_file), 'r') as f:
                mask_data = json.load(f)
                mask_data = np.array(mask_data).astype(np.long)
            
            with open(os.path.join(split_path, class_file), 'r') as f:
                class_data = json.load(f)
                num_elements = len(class_data)

            for ele_id in range(num_elements):
                mask = mask_data == ele_id+1
                class_name = class_data[ele_id]
                if class_name == 'none' or ele_id+1 == 1:
                    continue

                rle = encode_binary_mask(mask)
                if rle is None:
                    # 空实例，跳过但记录
                    # print(f"[WARN] empty mask seg_id={seg_id} file={image_file}", flush=True)
                    continue

                shard_items.append({
                    "image_file": os.path.join(pixelweb_root, split_name, img_file),
                    "segmentation": rle,
                })
                count += 1

                if count % shard_size == 0:
                    shard_idx += 1
                    out_path = os.path.join(temp_save_root, f"segment-{shard_idx:05d}.json")
                    with open(out_path, "w") as f:
                        json.dump(shard_items, f)
                    shard_items.clear()
                    print(f"[SAVE] {out_path} ({count} items)", flush=True)
            

if __name__ == "__main__":
    # 可选：降低原生库线程数，规避奇怪的并发崩溃
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()