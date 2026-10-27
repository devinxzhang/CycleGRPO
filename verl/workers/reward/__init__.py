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

from .config import RewardConfig


__all__ = ["BatchFunctionRewardManager", "FunctionRewardManager", "RewardConfig", "SequentialFunctionRewardManager"]


def __getattr__(name):
    # Reward managers import torch/transformers; load them only when used.
    if name in ("BatchFunctionRewardManager", "FunctionRewardManager", "SequentialFunctionRewardManager"):
        from . import function

        return getattr(function, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
