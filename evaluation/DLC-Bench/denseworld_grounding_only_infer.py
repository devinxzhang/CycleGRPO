import argparse
import io
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer captions from images + cap_problem and save to seg_problem."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="zhouyik/Qwen3-VL-4B-SAMTok-dam",
        help="HF model path",
    )
    parser.add_argument(
        "--input_parquet",
        type=str,
        default="./rl_dataset/denseworld_5k_img_22872_samples_train.parquet",
        help="Input parquet path",
    )
    parser.add_argument(
        "--output_parquet",
        type=str,
        default="./rl_dataset/denseworld_5k_img_22872_samples_train_grounding_only.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=".",
        help="Root prefix used to resolve relative image paths",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--overwrite_seg_problem",
        action="store_true",
        help="If set, overwrite existing seg_problem values; otherwise only fill empty ones.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start row index (inclusive).",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End row index (exclusive). Default: process to dataset end.",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default=None,
        help="Temporary directory for multi-GPU shard outputs. Default: sibling folder of output parquet.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=200,
        help="Save checkpoint every N successful updates per rank. Set <=0 to disable periodic save.",
    )
    return parser.parse_args()


def get_dist_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    return is_distributed, rank, local_rank, world_size


def setup_distributed(is_distributed: bool, local_rank: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    torch.cuda.set_device(local_rank)
    if is_distributed:
        dist.init_process_group(backend="nccl")


def cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def get_shard_indices(start: int, end: int, rank: int, world_size: int) -> list[int]:
    return list(range(start + rank, end, world_size))


def build_run_dir(args, start: int, end: int, world_size: int) -> str:
    output_dir = os.path.dirname(args.output_parquet)
    os.makedirs(output_dir, exist_ok=True)
    if args.tmp_dir is None:
        tmp_root = os.path.join(output_dir, "_grounding_only_tmp")
    else:
        tmp_root = args.tmp_dir
    os.makedirs(tmp_root, exist_ok=True)

    run_id = os.environ.get("RUN_ID")
    if run_id is None:
        out_name = os.path.splitext(os.path.basename(args.output_parquet))[0]
        run_id = f"{out_name}_s{start}_e{end}_w{world_size}"

    run_dir = os.path.join(tmp_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_shard(run_dir: str, rank: int, updates: dict[int, str]):
    shard_path = os.path.join(run_dir, f"shard_rank{rank:04d}.parquet")
    shard_df = pd.DataFrame({"idx": list(updates.keys()), "seg_problem": list(updates.values())})
    shard_df.to_parquet(shard_path, index=False)
    return shard_path, len(shard_df)


def merge_shards_to_output(input_parquet: str, output_parquet: str, run_dir: str, world_size: int):
    merged_df = pd.read_parquet(input_parquet)
    shard_files = [
        os.path.join(run_dir, f"shard_rank{r:04d}.parquet")
        for r in range(world_size)
        if os.path.exists(os.path.join(run_dir, f"shard_rank{r:04d}.parquet"))
    ]

    total_updates = 0
    for sf in shard_files:
        part = pd.read_parquet(sf)
        if len(part) == 0:
            continue
        total_updates += len(part)
        for row in part.itertuples(index=False):
            merged_df.at[int(row.idx), "seg_problem"] = row.seg_problem

    merged_df.to_parquet(output_parquet, index=False)
    return total_updates, len(shard_files)


def resolve_image_path(path: str, image_root: str) -> str:
    if os.path.isabs(path):
        return path
    if path.startswith("./"):
        return os.path.join(image_root, path[2:])
    return os.path.join(image_root, path)


def load_one_image(image_info, image_root: str) -> Image.Image:
    if isinstance(image_info, dict) and "bytes" in image_info:
        return Image.open(io.BytesIO(image_info["bytes"])).convert("RGB")
    if isinstance(image_info, bytes):
        return Image.open(io.BytesIO(image_info)).convert("RGB")

    if isinstance(image_info, str):
        image_path = resolve_image_path(image_info, image_root)
        return Image.open(image_path).convert("RGB")

    if isinstance(image_info, dict):
        image_path = image_info.get("path") or image_info.get("filename") or image_info.get("file")
        if image_path is not None:
            image_path = resolve_image_path(image_path, image_root)
            return Image.open(image_path).convert("RGB")

    raise ValueError(f"Unsupported image info format: {type(image_info)}")


def normalize_images(images_field) -> list:
    if images_field is None:
        return []
    if isinstance(images_field, (list, tuple)):
        return list(images_field)
    if isinstance(images_field, np.ndarray):
        return list(images_field.tolist())
    return [images_field]


def build_messages(cap_problem: str, pil_images: list[Image.Image]):
    content = []
    img_idx = 0

    segments = re.split(r"(<image>)", cap_problem)
    segments = [s for s in segments if s is not None and s != ""]

    for seg in segments:
        if seg == "<image>":
            if img_idx < len(pil_images):
                content.append({"type": "image", "image": pil_images[img_idx]})
                img_idx += 1
            elif len(pil_images) > 0:
                content.append({"type": "image", "image": pil_images[0]})
        else:
            content.append({"type": "text", "text": seg})

    if len(content) == 0 and len(pil_images) > 0:
        content = [{"type": "image", "image": pil_images[0]}, {"type": "text", "text": cap_problem}]

    return [{"role": "user", "content": content}]


def decode_output(processor, inputs, generated_ids):
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].replace("<|im_end|>", "").strip()


def run_infer_one(model, processor, cap_problem: str, images_field, image_root: str, args):
    image_items = normalize_images(images_field)
    pil_images = [load_one_image(item, image_root) for item in image_items]

    messages = build_messages(cap_problem, pil_images)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
        )

    return decode_output(processor, inputs, generated_ids)


def need_skip(existing_seg_problem, overwrite: bool) -> bool:
    if overwrite:
        return False
    if existing_seg_problem is None:
        return False
    if isinstance(existing_seg_problem, float) and pd.isna(existing_seg_problem):
        return False
    if isinstance(existing_seg_problem, str) and existing_seg_problem.strip() == "":
        return False
    return True


def main():
    args = parse_args()
    is_distributed, rank, local_rank, world_size = get_dist_info()
    setup_distributed(is_distributed, local_rank)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[rank {rank}] Loading input parquet: {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    total_rows = len(df)

    required_cols = {"images", "cap_problem", "seg_problem"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    start = max(args.start_index, 0)
    end = total_rows if args.end_index is None else min(args.end_index, total_rows)
    if start >= end:
        raise ValueError(f"Invalid range: start_index={start}, end_index={end}")

    shard_indices = get_shard_indices(start, end, rank, world_size)
    run_dir = build_run_dir(args, start, end, world_size)

    print(f"[rank {rank}] Loading model: {args.model_path} on cuda:{local_rank}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
    ).cuda().eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    print(
        f"[rank {rank}] Processing shard size={len(shard_indices)} "
        f"from rows [{start}, {end}) / {total_rows}, world_size={world_size}"
    )

    local_updates = {}
    successful_updates = 0
    iterator = tqdm(shard_indices, total=len(shard_indices), disable=(rank != 0))
    for idx in iterator:
        row = df.iloc[idx]

        if need_skip(row["seg_problem"], args.overwrite_seg_problem):
            continue

        cap_problem = row["cap_problem"]
        if cap_problem is None or (isinstance(cap_problem, float) and pd.isna(cap_problem)):
            continue

        try:
            pred_caption = run_infer_one(
                model=model,
                processor=processor,
                cap_problem=str(cap_problem),
                images_field=row["images"],
                image_root=args.image_root,
                args=args,
            )
            local_updates[idx] = pred_caption
            successful_updates += 1

            if args.save_every > 0 and (successful_updates % args.save_every == 0):
                shard_path, shard_count = write_shard(run_dir, rank, local_updates)
                print(f"[rank {rank}] Checkpoint shard saved: {shard_path} ({shard_count} rows)")

                if is_distributed:
                    dist.barrier()

                if rank == 0:
                    total_updates, shard_num = merge_shards_to_output(
                        input_parquet=args.input_parquet,
                        output_parquet=args.output_parquet,
                        run_dir=run_dir,
                        world_size=world_size,
                    )
                    print(
                        f"[rank 0] Periodic merge saved: {args.output_parquet} "
                        f"(updates={total_updates}, shard_files={shard_num})"
                    )

                if is_distributed:
                    dist.barrier()
        except Exception as e:
            print(f"[rank {rank}] [WARN] idx={idx} failed: {e}")

    shard_path, shard_count = write_shard(run_dir, rank, local_updates)
    print(f"[rank {rank}] Wrote shard results: {shard_path} ({shard_count} rows)")

    if is_distributed:
        dist.barrier()

    if rank == 0:
        total_updates, shard_num = merge_shards_to_output(
            input_parquet=args.input_parquet,
            output_parquet=args.output_parquet,
            run_dir=run_dir,
            world_size=world_size,
        )
        print(f"Saved output parquet to: {args.output_parquet}")
        print(f"Merged updates: {total_updates} rows from {shard_num} shard files")

    if is_distributed:
        dist.barrier()
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
