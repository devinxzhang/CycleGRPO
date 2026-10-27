# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from .resolve_config import resolve_config, to_yaml


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.worker.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.worker.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")

        RemoteRewardManager = ray.remote(RewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    ppo_config = resolve_config(OmegaConf.from_cli())
    os.makedirs(ppo_config.trainer.save_checkpoint_path, exist_ok=True)
    with open(os.path.join(ppo_config.trainer.save_checkpoint_path, "resolved_config.yaml"), "w") as f:
        f.write(to_yaml(ppo_config))

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "CYCLEGRPO_INNER_ROLLOUT_N": os.environ.get("CYCLEGRPO_INNER_ROLLOUT_N", "4"),
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
                # Propagate the selected video backend to every Ray process.
                "FORCE_QWENVL_VIDEO_READER": os.getenv("FORCE_QWENVL_VIDEO_READER", "decord"),
                # Ray runtime_env is the authoritative cross-node environment;
                # shell exports on the head node alone do not reach all workers.
                "CYCLEGRPO_DECORD_NUM_THREADS": os.getenv("CYCLEGRPO_DECORD_NUM_THREADS", "2"),
                "CYCLEGRPO_VIDEO_DECODE_TIMEOUT_S": os.getenv(
                    "CYCLEGRPO_VIDEO_DECODE_TIMEOUT_S", "60"
                ),
                "CYCLEGRPO_INNER_ROLLOUT_TIMEOUT_S": os.getenv(
                    "CYCLEGRPO_INNER_ROLLOUT_TIMEOUT_S", "600"
                ),
                # Keep experiment-gated dense temporal shaping identical in
                # the driver and the remote reward actor.  A default of zero
                # preserves all historical runs.
                "CYCLEGRPO_TEMPORAL_DENSE_REWARD_WEIGHT": os.getenv(
                    "CYCLEGRPO_TEMPORAL_DENSE_REWARD_WEIGHT", "0"
                ),
            }
        }
        for name in ("CYCLEGRPO_VIDEO_MIN_FRAMES", "CYCLEGRPO_VIDEO_MAX_FRAMES",
                     "CYCLEGRPO_VIDEO_TOTAL_PIXELS", "CYCLEGRPO_FRAME_CACHE_ROOT",
                     "CYCLEGRPO_REQUIRE_FRAME_CACHE", "CYCLEGRPO_FRAME_CACHE_CHECK_SOURCE",
                     "WANDB_MODE", "USE_MODELSCOPE_HUB"):
            if name in os.environ:
                runtime_env["env_vars"][name] = os.environ[name]
        ray.init(runtime_env=runtime_env)

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))

    if ppo_config.trainer.ray_timeline is not None:
        # use `export RAY_PROFILING=1` to record the ray timeline
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
