import os, io, base64, mimetypes
import sys
import json
from PIL import Image
import pyarrow.parquet as pq
import random
import tqdm
import numpy as np
import pyarrow as pa
from urllib.parse import urlparse
from typing import Any, Dict, List, Tuple, Optional, Union

import io
import base64
import numpy as np
from typing import List, Union
from PIL import Image
from pathlib import Path

import uuid


def read_row_by_index(_pf_, idx):
    num_row_groups = _pf_.num_row_groups

    rows_so_far = 0
    for rg in range(num_row_groups):
        rg_meta = _pf_.metadata.row_group(rg)
        rg_rows = rg_meta.num_rows
        if idx < rows_so_far + rg_rows:
            local_idx = idx - rows_so_far
            table = _pf_.read_row_group(rg)
            return table.to_pandas().iloc[local_idx]
        rows_so_far += rg_rows
    
    return None

def read_row_by_indices(_pf_, indices):
    indices = sorted(indices)
    result = []

    rows_so_far = 0
    for rg in range(_pf_.num_row_groups):
        rg_rows = _pf_.metadata.row_group(rg).num_rows
        local_indices = [i - rows_so_far for i in indices if rows_so_far <= i < rows_so_far + rg_rows]
        if local_indices:
            table = _pf_.read_row_group(rg).to_pandas()
            for li in local_indices:
                result.append(table.iloc[li].to_dict())
        rows_so_far += rg_rows
    return result
            


def _bytes_to_image(data: bytes) -> Image.Image:
    """把原始图片 bytes 打开成 PIL.Image（自动转 RGB）。"""
    return Image.open(io.BytesIO(data)).convert("RGB")

def _maybe_b64_to_bytes(x: Union[bytes, str]) -> bytes:
    """
    既支持 bytes 也支持 str：
    - 如果看起来像 data:image 前缀 → 去头再 base64 解码
    - 如果像纯 base64 → 直接解码（validate=True 保守判定）
    - 否则当作原始图片 bytes 返回
    """
    if isinstance(x, bytes):
        # 先尝试当作 base64 文本
        try:
            s = x.decode("utf-8", errors="strict").strip()
            # 有 data:image 头
            if s.startswith("data:image"):
                s = s.split(",", 1)[1]
                return base64.b64decode(s)
            # 尝试严格 base64 解码
            return base64.b64decode(s, validate=True)
        except Exception:
            # 不是 utf-8 base64，就当作原始图片二进制
            return x

    elif isinstance(x, str):
        s = x.strip()
        if s.startswith("data:image"):
            s = s.split(",", 1)[1]
        try:
            return base64.b64decode(s, validate=True)
        except Exception:
            # 不是合法 base64，当作原始 bytes（需要再编码一下）
            return s.encode("latin1")

    else:
        raise TypeError(f"Unsupported type for image payload: {type(x)}")

def decode_images_field(images_field) -> List[Image.Image]:
    """
    images_field 可能是：
      - numpy.ndarray(dtype=object)，元素是 bytes 或 str
      - 单个 bytes/str
    返回：PIL.Image 列表
    """
    imgs = []
    if isinstance(images_field, np.ndarray):
        # numpy 数组（你的情况就是 (1,) 且元素是原始 PNG bytes）
        for item in images_field:
            payload = _maybe_b64_to_bytes(item)
            imgs.append(_bytes_to_image(payload))
    elif isinstance(images_field, (bytes, bytearray, str)):
        payload = _maybe_b64_to_bytes(images_field)
        imgs.append(_bytes_to_image(payload))
    else:
        raise TypeError(f"Unexpected images_field type: {type(images_field)}")
    return imgs

def save_images_field(images_field, out_dir: str, base_name: str) -> List[str]:
    """
    解码并保存到目录：
      out_dir/base_name_0.png, base_name_1.png, ...
    返回保存路径列表
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    imgs = decode_images_field(images_field)
    paths = []
    for i, img in enumerate(imgs):
        p = Path(out_dir) / f"{base_name}_{i}.png"
        img.save(p)
        paths.append(str(p))
    return paths


def main(task_id):

    image_save_root = "./data/internvl_sft_images/"
    if not os.path.exists(image_save_root):
        os.makedirs(image_save_root)
    parquet_file_root = "./data/sft_parquet"
    parquet_files = os.listdir(parquet_file_root)

    # with open('internvl_parquet_file_info.json', 'w') as f:
    #     json.dump(parquet_files, f)
    # exit(0)
    
    with open('internvl_parquet_file_info.json', 'r') as f:
        parquet_files = json.load(f)

    num_files = len(parquet_files)
    chunk_size = (num_files+4) // 5
    _start_ = task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = num_files if _end_ > num_files else _end_

    temp_save_root = "temp_data_internvl_data"
    if not os.path.exists(temp_save_root):
        os.makedirs(temp_save_root)
    
    grounding_temp_save_root = "temp_data_internvl_grounding_data"
    if not os.path.exists(grounding_temp_save_root):
        os.makedirs(grounding_temp_save_root)

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0
    for parquet_file in tqdm.tqdm(parquet_files[_start_:_end_][-17:-15]):
        if not parquet_file.endswith(".parquet"):
            continue

        pf = pq.ParquetFile(os.path.join("./data/sft_parquet", parquet_file))
        total_rows = pf.metadata.num_rows

        sample_size = int(total_rows * 0.2)
        random_indices = sorted(random.sample(range(total_rows), sample_size))
        
        sampled_rows = read_row_by_indices(pf, random_indices)

        for row in sampled_rows:
            source = row['source']

            conversations = row['conversations']
            id = row['id']
            image_files = []
            if "images" in row and row["images"] is not None:
                try:
                    imgs = decode_images_field(row["images"])
                except:
                    continue
                for img_id, img in enumerate(imgs):
                    random_name = uuid.uuid4().hex
                    image_name = f"id{id}_{img_id}th_{random_name}.jpg"
                    image_path = os.path.join(image_save_root, image_name)
                    image_files.append(image_path)
                    img.save(image_path)
            
            ret_data_dict = {
                'image': image_files,
                'conversations': conversations,
            }

            if source in ["ocr_vqa/coco/refcoco_grounding_aug_en.jsonl", "ocr_vqa/coco/refcoco_grounding_en.jsonl", "ocr_vqa/coco/tallyqa_coco_en.jsonl", "ocr_vqa/coco/toloka_grounding_aug_en.jsonl", "ocr_vqa/downstream_grounding/downstream_grounding_zh.jsonl"]:
                random_name = uuid.uuid4().hex
                with open(os.path.join(grounding_temp_save_root, f"id{id}_{random_name}.json"), 'w') as f:
                    json.dump(ret_data_dict, f)
            else:
                random_name = uuid.uuid4().hex
                with open(os.path.join(temp_save_root, f"id{id}_{random_name}.json"), 'w') as f:
                    json.dump(ret_data_dict, f)
            
if __name__ == "__main__":
    task_id = sys.argv[1]
    task_id = int(task_id)
    main(task_id)
        
        
            