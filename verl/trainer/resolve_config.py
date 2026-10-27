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
"""Resolve the trainer config from ``config=<yaml>`` plus CLI overrides, without Ray.

``python -m <runtime>.trainer.resolve_config config=... key=value ...`` prints the
fully merged config, so typos and type errors in overrides fail before training.
"""

from omegaconf import OmegaConf

from .config import PPOConfig


def resolve_config(cli_args) -> PPOConfig:
    default_config = OmegaConf.structured(PPOConfig())
    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        default_config = OmegaConf.merge(default_config, OmegaConf.load(config_path))

    ppo_config: PPOConfig = OmegaConf.to_object(OmegaConf.merge(default_config, cli_args))
    ppo_config.deep_post_init()
    return ppo_config


def to_yaml(ppo_config: PPOConfig) -> str:
    return OmegaConf.to_yaml(OmegaConf.structured(ppo_config))


if __name__ == "__main__":
    print(to_yaml(resolve_config(OmegaConf.from_cli())))
    print("Configuration resolved; not starting Ray.")
