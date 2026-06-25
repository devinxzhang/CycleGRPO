import os, io, json
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils
import pyarrow.parquet as pq
import tqdm

DATA_ROOT_COCO = "./data/coco"
TRAIN_DIR = os.path.join(DATA_ROOT_COCO, "train2017")
ULB_DIR   = os.path.join(DATA_ROOT_COCO, "unlabeled2017")

def load_mask_png(mask_bytes):
    # 强制成单通道整数阵列，避免调色板/多通道的坑
    with Image.open(io.BytesIO(mask_bytes)) as im:
        if im.mode not in ("L", "I", "P"):
            im = im.convert("L")
        else:
            # P 模式转成 L，I 模式保持即可
            if im.mode == "P":
                im = im.convert("L")
        arr = np.array(im)
    # 统一到 int32，后续比较稳定
    return arr.astype(np.int32, copy=False)

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
    image_root = "./data/object365/"
    pano_image_root = "./data/coconut_large/panoptic_object365"
    anno_file = "./data/coconut_large/panseg_object365_train_v2.json"
    with open(anno_file, 'r') as f:
        anno_data = json.load(f)

    temp_save_root = "./temp_data"
    os.makedirs(temp_save_root, exist_ok=True)

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    anno_info_list = anno_data['annotations']
    for anno_info in tqdm.tqdm(anno_info_list):
        segments_info = anno_info['segments_info']
        if 'object365_file_name' in anno_info:
            image_id = anno_info['object365_file_name']
        elif 'file_name' in anno_info:
            image_id = anno_info['file_name']
        else:
            ValueError(f"image_id not found in anno_info: {anno_info}")

        if '.png' in image_id:
            image_id = image_id.split('.png')[0]
        if '.jpg' in image_id:
            image_id = image_id.split('.jpg')[0]

        patch_list = ['patch17', 'patch23', 'patch25', 'patch28', 'patch32', 'patch35', 'patch38', 'patch40', 'patch42', 'patch44', 'patch50']
        image_path = None
        for patch_name in patch_list:
            if os.path.exists(os.path.join(image_root, patch_name, f"{image_id}.jpg")):
                image_path = os.path.join(image_root, patch_name, f"{image_id}.jpg")
                break
        if image_path is None:
            continue

        pano_file = os.path.join(pano_image_root, f"{image_id}.png")
        mask_image = Image.open(pano_file)
        mask_image_np = np.array(mask_image)[:, :, 0]

        for seg in segments_info:
            seg_id = seg.get('id', None)
            if seg_id is None:
                continue

            try:
                bin_mask = (mask_image_np == int(seg_id))  # bool HxW
                rle = encode_binary_mask(bin_mask)
                if rle is None:
                    # 空实例，跳过但记录
                    # print(f"[WARN] empty mask seg_id={seg_id} file={image_file}", flush=True)
                    continue

                shard_items.append({
                    "image_file": image_path,
                    "segmentation": rle,
                })
                count += 1

                if count % shard_size == 0:
                    shard_idx += 1
                    out_path = os.path.join(temp_save_root, f"large-segment-{shard_idx:05d}.json")
                    with open(out_path, "w") as f:
                        json.dump(shard_items, f)
                    shard_items.clear()
                    print(f"[SAVE] {out_path} ({count} items)", flush=True)

            except Exception as e:
                # 如果 pycocotools 在 C 层崩溃，这里是抓不到的；但大多数数据问题能在这儿被捕到
                print(f"[ERROR] encode failed seg_id={seg_id} file={image_path}: {e}", flush=True)
                continue
    # 收尾
    if shard_items:
        shard_idx += 1
        out_path = os.path.join(temp_save_root, f"large-segment-{shard_idx:05d}.json")
        with open(out_path, "w") as f:
            json.dump(shard_items, f)
        shard_items.clear()
        print(f"[SAVE] {out_path} (final, total={count})", flush=True)

if __name__ == "__main__":
    # 可选：降低原生库线程数，规避奇怪的并发崩溃
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
