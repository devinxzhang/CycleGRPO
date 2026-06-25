import os, io, json
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils
import pyarrow.parquet as pq

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
    sample_files = [
        "./data/coconut_base/train-00000-of-00004.parquet",
        "./data/coconut_base/train-00001-of-00004.parquet",
        "./data/coconut_base/train-00002-of-00004.parquet",
        "./data/coconut_base/train-00003-of-00004.parquet",
    ]

    temp_save_root = "./temp_data"
    os.makedirs(temp_save_root, exist_ok=True)

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0

    for sample_file in sample_files:
        parquet_file = pq.ParquetFile(sample_file)
        df = parquet_file.read().to_pandas()
        rows = len(df)
        for row_idx, row in enumerate(df.itertuples(index=False), start=1):
            print(f"===========================Processing row {row_idx} of {rows}===========================", flush=True)
            try:
                masks = getattr(row, 'mask')
                segments_info = getattr(row, 'segments_info')
                image_info = getattr(row, 'image_info')
                image_file = image_info['file_name']
            except Exception as e:
                print(f"[SKIP] meta parse failed at row {row_idx}: {e}", flush=True)
                continue

            # 找图片
            image_path = os.path.join(TRAIN_DIR, image_file)
            if not os.path.exists(image_path):
                image_path = os.path.join(ULB_DIR, image_file)
                if not os.path.exists(image_path):
                    print(f"[SKIP] image not found: {image_file}", flush=True)
                    continue

            # 解 mask
            try:
                mask_image_np = load_mask_png(masks['bytes'])  # HxW int32
            except Exception as e:
                print(f"[SKIP] mask decode failed {image_file}: {e}", flush=True)
                continue

            segs = segments_info.get('segments_info', [])
            for seg in segs:
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
                        out_path = os.path.join(temp_save_root, f"segment-{shard_idx:05d}.json")
                        with open(out_path, "w") as f:
                            json.dump(shard_items, f)
                        shard_items.clear()
                        print(f"[SAVE] {out_path} ({count} items)", flush=True)

                except Exception as e:
                    # 如果 pycocotools 在 C 层崩溃，这里是抓不到的；但大多数数据问题能在这儿被捕到
                    print(f"[ERROR] encode failed seg_id={seg_id} file={image_file}: {e}", flush=True)
                    continue

    # 收尾
    if shard_items:
        shard_idx += 1
        out_path = os.path.join(temp_save_root, f"segment-{shard_idx:05d}.json")
        with open(out_path, "w") as f:
            json.dump(shard_items, f)
        shard_items.clear()
        print(f"[SAVE] {out_path} (final, total={count})", flush=True)

if __name__ == "__main__":
    # 可选：降低原生库线程数，规避奇怪的并发崩溃
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
