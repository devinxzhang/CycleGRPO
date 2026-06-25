import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image
import tqdm


def load_json(p: str) -> Any:
    with open(p, "r") as f:
        return json.load(f)


def save_json(p: str, data: Any) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def even_sample_indices(indices: List[int], k: int) -> List[int]:
    """从可见帧索引均匀采样至多 k 个。"""
    if len(indices) <= k:
        return sorted(indices)
    pos = np.linspace(0, len(indices) - 1, num=k)
    chosen = sorted({indices[int(round(x))] for x in pos})
    return chosen[:k]


def ensure_compressed_rle(rle_like: Dict[str, Any]) -> Dict[str, Any]:
    """
    统一为 COCO 压缩 RLE（counts: str, size: [h,w]）
    - counts 为 list（未压缩）→ decode→encode
    - counts 为 bytes/str（压缩）→ 规整为 str
    """
    if not (isinstance(rle_like, dict) and "counts" in rle_like and "size" in rle_like):
        raise ValueError("Segmentation must be RLE dict with 'size' and 'counts'.")
    rle = rle_like
    if isinstance(rle["counts"], list):
        m = mask_utils.decode(rle).astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(m))
    counts = rle["counts"].decode("utf-8") if isinstance(rle["counts"], (bytes, bytearray)) else rle["counts"]
    return {"size": rle["size"], "counts": counts}


def find_pairs(sav_root: str, annotation_glob: str):
    """
    返回 (json_path, video_path, split_dir, video_id_guess)
    结构示例：
      <root>/sav_000/sav_train/sav_000/sav_000001_auto.json
      <root>/sav_000/sav_train/sav_000/sav_000001.mp4
    """
    root = Path(sav_root)
    json_files = sorted(root.glob(annotation_glob))
    # 兜底：如果用户的 glob 太严格，退化为递归找所有 *_auto.json
    if not json_files:
        json_files = sorted(root.rglob("*_auto.json"))

    pairs = []
    for jp in json_files:
        stem = jp.stem  # e.g. "sav_000001_auto"
        # 优先匹配去掉 "_auto" 的同名 mp4
        base = stem[:-5] if stem.endswith("_auto") else stem
        mp4p = jp.with_name(base + ".mp4")
        if not mp4p.exists():
            # 退一步：同名含 _auto 的 mp4（以防个别 split 命名特殊）
            alt = jp.with_suffix(".mp4")  # sav_000001_auto.mp4
            if alt.exists():
                mp4p = alt
            else:
                # 还不行就跳过，但保留一个提示
                # print(f"[Skip] mp4 not found for {jp}")
                continue

        parts = jp.relative_to(root).parts
        split_dir = "/".join(parts[:3]) if len(parts) >= 3 else "unknown_split"
        video_id_guess = base
        pairs.append((str(jp), str(mp4p), split_dir, video_id_guess))

    return pairs



def decode_and_save_frame_cached(cap: cv2.VideoCapture,
                                 f24: int,
                                 save_path: str,
                                 jpeg_quality: int = 95) -> Tuple[int, int]:
    """从已打开的 cap 读取指定帧并保存为 jpg。返回 (w, h)。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = min(max(f24, 0), max(0, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {idx}")
    h, w = frame.shape[:2]
    cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    return w, h


def process_video_worker(args: Tuple[str, str, str, str, str, int, int]) -> Tuple[str, int, Optional[str], List[Dict[str, Any]]]:
    """
    Worker 处理一个视频，返回：
      (json_path, item_count, error_message_if_any, items_batch)
    items_batch 为该视频所有抽样 items 的列表（为了降低 IPC 次数，按视频为批处理）。
    """
    (json_path, video_path, split_dir, frames_out_root, stride, max_samples_per_masklet, jpeg_quality) = args
    try:
        data = load_json(json_path)
        masklet = data["masklet"]            # List[frame][obj] -> RLE or None
        masklet_ids = data["masklet_id"]     # List[int]
        # 一些标注里 video_id 为字符串，这里只用于 frames 目录组织
        video_id = str(data.get("video_id", Path(video_path).stem.replace("_auto", "")))

        num_frames_annot = len(masklet)
        num_objects = len(masklet_ids) if num_frames_annot > 0 else 0

        # 收集每个物体可见帧（6fps）
        visible_idx_per_obj: List[List[int]] = [[] for _ in range(num_objects)]
        for f_idx in range(num_frames_annot):
            per_obj = masklet[f_idx]
            if per_obj is None:
                continue
            up_to = min(len(per_obj), num_objects)
            for j in range(up_to):
                rle = per_obj[j]
                if rle:
                    visible_idx_per_obj[j].append(f_idx)

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # 输出目录：<frames_out_root>/<split_dir>/<video_id>/
        frames_dir = Path(frames_out_root) / split_dir / video_id
        frames_dir.mkdir(parents=True, exist_ok=True)

        written_frames = set()
        items: List[Dict[str, Any]] = []

        for obj_idx, _ in enumerate(masklet_ids):
            vis = visible_idx_per_obj[obj_idx]
            if not vis:
                continue
            chosen = even_sample_indices(vis, max_samples_per_masklet)

            for f6 in chosen:
                rle_like = masklet[f6][obj_idx]
                if not rle_like:
                    continue

                f24 = f6 * stride
                jpg_path = str(frames_dir / f"{f24:06d}.jpg")

                if jpg_path not in written_frames:
                    _w, _h = decode_and_save_frame_cached(cap, f24, jpg_path, jpeg_quality=jpeg_quality)
                    written_frames.add(jpg_path)

                # 统一 RLE
                try:
                    rle = ensure_compressed_rle(rle_like)
                except Exception:
                    try:
                        m = mask_utils.decode(rle_like).astype(np.uint8)
                        rle_enc = mask_utils.encode(np.asfortranarray(m))
                        rle = {"size": rle_enc["size"], "counts": rle_enc["counts"].decode("utf-8")}
                    except Exception as e2:
                        # 跳过异常实例
                        continue

                items.append({"image_file": jpg_path, "segmentation": rle})

        cap.release()
        return (json_path, len(items), None, items)

    except Exception as e:
        return (json_path, 0, str(e), [])


def writer_process(queue: mp.Queue, out_prefix: str, shard_size: int):
    """
    单独写入进程：从队列接收 items（按视频批次 list），累积到 shard_size 写一次。
    收到 None 表示结束。
    """
    out_items: List[Dict[str, Any]] = []
    shard_id = 1

    def dump():
        nonlocal out_items, shard_id
        if not out_items:
            return
        out_path = f"{out_prefix}.part_{shard_id:06d}.json"
        save_json(out_path, out_items)
        print(f"[Write] {out_path} (items={len(out_items)})")
        out_items.clear()
        shard_id += 1

    while True:
        batch = queue.get()
        if batch is None:  # 结束信号
            break
        # batch 是一个 List[dict]
        out_items.extend(batch)
        if len(out_items) >= shard_size:
            dump()

    dump()
    print("[Writer] Done.")


def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    frames = []

    frame_id = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

        frame_id += 1

    cap.release()
    return frames

def decode_masklet(masklet):
    masks = []
    for _rle in masklet:
        mask = mask_utils.decode(_rle)
        masks.append(mask)
    return masks

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
    ap = argparse.ArgumentParser("Build SA_V items (multiprocessing, per-masklet ≤K frames, sharded JSON).")
    ap.add_argument("--sav-root", type=str, required=True,
                    help="Root dir of SA_V (contains sav_000/sav_train/sav_000, sav_001/...).")
    ap.add_argument("--annotation-glob", type=str, default="sav_*/sav_train/*/*_auto.json",
                    help="Glob (relative to --sav-root), e.g. 'sav_*/sav_train/*/*_auto.json'.")
    ap.add_argument("--out-prefix", type=str, required=True,
                    help="Output JSON prefix; shards -> <prefix>.part_xxxxxx.json")
    ap.add_argument("--frames-out-dir", type=str, required=True,
                    help="Directory to save extracted frames (jpg).")
    ap.add_argument("--max-samples-per-masklet", type=int, default=5,
                    help="Max frames per masklet.")
    ap.add_argument("--shard-size", type=int, default=10000,
                    help="Write a JSON every N items.")
    ap.add_argument("--stride", type=int, default=4,
                    help="6fps annotation -> 24fps frame index (default 4).")
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 48),
                    help="Number of worker processes (tune per disk bandwidth).")
    ap.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for saved frames.")
    ap.add_argument("--task-id", type=int, default=0, help="task_id.")
    args = ap.parse_args()

    # pairs = find_pairs(args.sav_root, args.annotation_glob)
    # if not pairs:
    #     raise SystemExit("No *_auto.json found. Check --sav-root and --annotation-glob.")
    
    # with open("./data/sam_v_video_info.json", 'w') as f:
    #     json.dump(pairs, f)
    # exit(0)

    dataset_name = 'sam_v'

    temp_save_root = "./any_other_seg_data"
    os.makedirs(temp_save_root, exist_ok=True)

    count = 0
    shard_size = 10000
    shard_items = []
    shard_idx = 0


    with open("./data/sam_v_video_info.json", 'r') as f:
        json_data = json.load(f)

    n = len(json_data)
    chunk_size = (n+7) // 8
    start = args.task_id * chunk_size
    end = args.task_id * chunk_size + chunk_size
    end = n if end > n else end
    
    for pair in tqdm.tqdm(json_data[start:end]):
        anno_path = pair[0]
        video_path = pair[1]
        split_name = pair[3]
        video_frames = get_video_frames(video_path)

        video_frames = video_frames[::4] # list, item.shape == h, w, 3

        # mask annotation
        with open(anno_path, 'r') as f:
            mask_data = json.load(f)
        masklets = decode_masklet(mask_data['masklet'])
        masklets = np.stack(masklets, axis=0)  # (n_frames, h, w, n_obj)

        if not os.path.exists(f"./data/sam_v_frames/{split_name}"):
            os.makedirs(f"./data/sam_v_frames/{split_name}")

        frame_indices = list(range(0, len(video_frames), 10))
        for frame_idx in frame_indices:
            frame = video_frames[frame_idx]
            frame = frame[:, :, ::-1]
            frame_image = Image.fromarray(frame).convert('RGB')
            frame_image.save(f"./data/sam_v_frames/{split_name}/frame_{frame_idx}.jpg")

            masks = masklets[frame_idx]
            for mask_idx in range(masks.shape[-1]):
                bin_mask = masks[:, :, mask_idx].astype(np.bool)
                if np.sum(bin_mask) == 0:
                    continue

                try:
                    assert len(bin_mask.shape) == 2
                    rle = encode_binary_mask(bin_mask.astype(np.bool))
                    if rle is None:
                        # 空实例，跳过但记录
                        # print(f"[WARN] empty mask seg_id={seg_id} file={image_file}", flush=True)
                        continue

                    shard_items.append({
                        "image_file": f"./data/sam_v_frames/{split_name}/frame_{frame_idx}.jpg",
                        "segmentation": rle,
                    })
                    count += 1

                    if count % shard_size == 0:
                        shard_idx += 1
                        out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}-split-{args.task_id}.json")
                        with open(out_path, "w") as f:
                            json.dump(shard_items, f)
                        shard_items.clear()
                        print(f"[SAVE] {out_path} ({count} items)", flush=True)

                except Exception as e:
                    # 如果 pycocotools 在 C 层崩溃，这里是抓不到的；但大多数数据问题能在这儿被捕到
                    print(f"[ERROR] ...", flush=True)
                    continue
    
    # 收尾
    if shard_items:
        shard_idx += 1
        out_path = os.path.join(temp_save_root, f"{dataset_name}-segment-{shard_idx:05d}-split-{args.task_id}.json")
        with open(out_path, "w") as f:
            json.dump(shard_items, f)
        shard_items.clear()
        print(f"[SAVE] {out_path} (final, total={count})", flush=True)     

if __name__ == "__main__":
    # Windows/某些集群环境/交互式下推荐显式使用 'spawn'
    mp.set_start_method("spawn", force=True)
    main()
