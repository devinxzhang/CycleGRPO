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
"""
Actor config
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    override_config: dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    freeze_vision_tower: bool = False

    def post_init(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        if self.model_path is not None and os.path.exists(self.model_path):  # ray job uses absolute path
            self.model_path = os.path.abspath(self.model_path)

        if self.tokenizer_path is not None and os.path.exists(self.tokenizer_path):
            self.tokenizer_path = os.path.abspath(self.tokenizer_path)


@dataclass
class OptimConfig:
    lr: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    strategy: str = "adamw"
    lr_warmup_ratio: float = 0.0
    lr_warmup_steps: Optional[int] = None
    lr_scheduler_type: str = "constant"
    min_lr_ratio: Optional[float] = None
    warmup_style: str = "constant"
    # below are auto keys
    training_steps: int = field(default=-1, init=False)


@dataclass
class FSDPConfig:
    enable_full_shard: bool = True
    enable_cpu_offload: bool = False
    enable_rank0_init: bool = True
    use_orig_params: bool = False
    torch_dtype: Optional[str] = None
    fsdp_size: int = -1
    mp_param_dtype: str = "bf16"
    mp_reduce_dtype: str = "fp32"
    mp_buffer_dtype: str = "fp32"


@dataclass
class OffloadConfig:
    offload_params: bool = False
    offload_optimizer: bool = False


@dataclass
class ActorConfig:
    strategy: str = "fsdp"
    global_batch_size: int = 256
    optimize_captioner: bool = True
    optimize_segmenter: bool = True
    # --- dispersion-gated dead-group GT repair (docs/design_dispersion_gated_repair.md) ---
    dead_repair_enable: bool = False
    """inject the GT-interval response into DEAD seg-GRPO groups (std<eps & mean<tau)"""
    dead_repair_eps: float = 0.05
    """group tIoU std below this = no gradient (collapsed or uniformly wrong)"""
    dead_repair_tau: float = 0.3
    """group tIoU mean below this = wrong; combined with eps → DEAD-WRONG"""
    dead_repair_max_frac: float = 0.5
    """max fraction of detected dead groups to repair per step (keeps RL dominant)"""
    dead_repair_n_replace: int = 1
    """responses replaced with the GT interval per repaired group (of K=8)"""
    # --- v3: readback judge (docs/design_v3_readback_judge.md) ---
    judge_enable: bool = False
    """likelihood-readback judge: caption-side window-QA bonus + seg-side gate/injection"""
    judge_tau: float = 3.0
    """M3 total-logprob margin threshold: below = caption doesn't pin its window"""
    judge_bonus_weight: float = 2.0
    """max caption bonus (vs 10*mtiou primary); bonus = w * min(M3, cap)/cap, argmax==GT only"""
    judge_m3_cap: float = 10.0
    """margin value that earns the full bonus"""
    judge_max_cands: int = 4
    """max candidate windows per caption (GT + up to 3 clustered strays)"""
    judge_max_cap_tokens: int = 350
    """caption tokens scored per candidate (truncation bound)"""
    # --- v2: segment-level advantage for the inner grounding GRPO ---
    seg_adv_level: str = "caption"
    """advantage normalization pool for the seg side: 'caption' (per-caption group,
    vanilla) or 'segment' (all K*n samples of a segment share one baseline -> dead
    groups get contrast from sibling wordings; wrong-frozen modes receive negative
    advantage instead of zero gradient)"""
    """number of samples per minibatch for updating actor"""
    micro_batch_size_per_device_for_update: int = 4
    """number of samples per forward pass for updating actor"""
    micro_batch_size_per_device_for_experience: int = 16
    """number of samples per forward pass for computing log probs"""
    max_grad_norm: float = 1.0
    """number to clip grad norm"""
    clip_ratio_low: float = 0.2
    """clip ratio in PPO & DAPO"""
    clip_ratio_high: float = 0.3
    """clip ratio in PPO & DAPO"""
    clip_ratio_dual: float = 3.0
    """constant C in dual-clip PPO, clips when advantage < -C"""
    loss_avg_mode: str = "token"
    """loss average mode: `token`, `seq`"""
    loss_type: str = "default"
    """loss type: `default`, `gspo`, `cispo`"""
    ppo_epochs: int = 1
    """number of ppo epochs for each rollout batch"""
    padding_free: bool = True
    """use padding-free training"""
    dynamic_batching: bool = True
    """enable dynamic batching"""
    selective_response_logits: bool = False
    """compute vocabulary logits only at positions that predict response tokens.
    This is exact for PPO log-prob/loss computation and avoids materializing
    full-vocabulary logits for the (usually much longer) multimodal prompt."""
    ulysses_size: int = 1
    """ulysses sequence parallel size"""
    use_torch_compile: bool = True
    """enable torch compile"""
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    # below are auto keys
    global_batch_size_per_device: int = field(default=-1, init=False)
    disable_kl: bool = field(default=False, init=False)
    use_kl_loss: bool = field(default=False, init=False)
    kl_penalty: str = field(default="kl", init=False)
    kl_coef: float = field(default=0.0, init=False)


@dataclass
class RefConfig:
    strategy: str = "fsdp"
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    # below are auto keys
    micro_batch_size_per_device_for_experience: int = field(default=-1, init=False)
    padding_free: bool = field(default=False, init=False)
    dynamic_batching: bool = field(default=False, init=False)
    selective_response_logits: bool = False
    """same response-only logits fast path as the actor; opt-in for compatible models"""
    ulysses_size: int = field(default=1, init=False)
    use_torch_compile: bool = field(default=True, init=False)
