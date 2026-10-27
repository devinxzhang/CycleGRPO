"""Atomic conversion of verl FSDP actor shards to HF safetensors."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor


def _rank(path: Path) -> int:
    match = re.search(r"_rank_(\d+)\.pt$", path.name)
    if match is None:
        raise RuntimeError(f"cannot parse rank from {path}")
    return int(match.group(1))


def validate_hf_checkpoint(path: Path) -> tuple[int, int]:
    """Validate config, shard inventory, safetensor headers, and tensor keys."""

    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file() or not (path / "config.json").is_file():
        raise RuntimeError(f"missing HF index/config in {path}")
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map", {})
    if not weight_map:
        raise RuntimeError(f"empty weight map in {index_path}")
    expected_shards = set(weight_map.values())
    actual_shards = {item.name for item in path.glob("model-*-of-*.safetensors")}
    if actual_shards != expected_shards:
        raise RuntimeError(
            f"shard mismatch in {path}: expected={expected_shards}, actual={actual_shards}"
        )
    actual_keys: set[str] = set()
    total_size = 0
    for name in sorted(expected_shards):
        shard = path / name
        total_size += shard.stat().st_size
        with safe_open(shard, framework="pt", device="cpu") as handle:
            actual_keys.update(handle.keys())
    if actual_keys != set(weight_map):
        raise RuntimeError(f"tensor-key mismatch in {path}")
    return len(actual_keys), total_size


def merge_fsdp_actor_to_hf(
    actor_dir: str | Path,
    output_dir: str | Path,
    *,
    max_shard_gb: float = 4.0,
) -> tuple[int, int]:
    """Merge to a temporary sibling, validate it, then atomically publish it."""

    actor_dir = Path(actor_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        return validate_hf_checkpoint(output_dir)
    temporary = output_dir.with_name(f".{output_dir.name}.merge_tmp")
    if temporary.exists():
        shutil.rmtree(temporary)

    model_paths = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"), key=_rank)
    if not model_paths:
        raise FileNotFoundError(f"no model shards in {actor_dir}")
    world_size = len(model_paths)
    expected = [
        actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
    ]
    if model_paths != expected:
        raise RuntimeError(f"incomplete or inconsistent model shards in {actor_dir}")
    hf_dir = actor_dir / "huggingface"
    if not hf_dir.is_dir():
        raise FileNotFoundError(f"missing Hugging Face metadata in {hf_dir}")

    try:
        temporary.mkdir(parents=False)
        for source in hf_dir.iterdir():
            if source.is_file():
                shutil.copy2(source, temporary / source.name)

        states = [
            torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            for path in expected
        ]
        keys = list(states[0])
        if any(list(state) != keys for state in states[1:]):
            raise RuntimeError("rank state dict keys differ")

        limit = int(max_shard_gb * 1024**3)
        pending: dict[str, torch.Tensor] = {}
        pending_bytes = 0
        shards: list[tuple[str, list[str]]] = []
        weight_map: dict[str, str] = {}
        total_size = 0

        def flush() -> None:
            nonlocal pending, pending_bytes
            if not pending:
                return
            name = f"model-{len(shards) + 1:05d}.safetensors"
            save_file(pending, str(temporary / name))
            shards.append((name, list(pending)))
            pending = {}
            pending_bytes = 0

        for key in keys:
            values = [state[key] for state in states]
            first = values[0]
            if isinstance(first, DTensor):
                placements = first.placements
                if len(placements) != 1 or not placements[0].is_shard():
                    raise RuntimeError(f"unsupported placement for {key}: {placements}")
                dim = placements[0].dim
                tensor = torch.cat([value.to_local() for value in values], dim=dim)
                slices = [slice(None)] * tensor.ndim
                slices[dim] = slice(0, first.shape[dim])
                tensor = tensor[tuple(slices)]
            elif torch.is_tensor(first):
                if not all(torch.equal(first, value) for value in values[1:]):
                    raise RuntimeError(f"non-DTensor value differs across ranks: {key}")
                tensor = first
            else:
                raise TypeError(f"unsupported state value for {key}: {type(first)}")
            tensor = tensor.contiguous()
            size = tensor.numel() * tensor.element_size()
            if pending and pending_bytes + size > limit:
                flush()
            pending[key] = tensor
            pending_bytes += size
            total_size += size
        flush()

        total_shards = len(shards)
        for index, (old_name, shard_keys) in enumerate(shards, 1):
            final_name = f"model-{index:05d}-of-{total_shards:05d}.safetensors"
            (temporary / old_name).rename(temporary / final_name)
            for key in shard_keys:
                weight_map[key] = final_name
        with (temporary / "model.safetensors.index.json").open("w") as handle:
            json.dump(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                handle,
                indent=2,
            )
        result = validate_hf_checkpoint(temporary)
        temporary.rename(output_dir)
        return result
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge an FSDP actor checkpoint (global_step_*/actor) into a HF model.")
    parser.add_argument("actor_dir", help="e.g. checkpoints/<project>/<experiment>/global_step_190/actor")
    parser.add_argument("output_dir", help="new directory for the merged HF checkpoint, e.g. .../ckpt_190")
    parser.add_argument("--max_shard_gb", type=float, default=4.0)
    args = parser.parse_args()
    tensors, size = merge_fsdp_actor_to_hf(args.actor_dir, args.output_dir, max_shard_gb=args.max_shard_gb)
    print(f"Merged {tensors} tensors ({size / 1024**3:.2f} GiB) -> {args.output_dir}")
