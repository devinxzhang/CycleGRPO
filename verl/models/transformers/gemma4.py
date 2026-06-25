# Copyright 2024 The Gemma team and the HuggingFace Inc. team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Based on:
# https://github.com/huggingface/transformers/blob/d379ac18db61a4e194f0d53c0c57105d08183c59/src/transformers/models/gemma4/modular_gemma4.py
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

from typing import Optional

import torch
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4CausalLMOutputWithPast,
    Gemma4ForConditionalGeneration,
    Gemma4Model,
    Gemma4ModelOutputWithPast,
)


def _resolve_pad_token_id(model: "Gemma4Model") -> int:
    """Multimodal token IDs (image/video/audio) often live outside the LLM vocab range and would
    blow up ``embed_tokens`` lookup. The upstream forward avoids that by overwriting them with the
    text pad id before embedding. Fall back to ``0`` if the config has no pad token configured.
    """
    pad_token_id = getattr(model.config.text_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    return int(pad_token_id)


def _get_input_embeds(
    model: "Gemma4Model",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    input_features_mask: Optional[torch.Tensor] = None,
    image_position_ids: Optional[torch.LongTensor] = None,
    video_position_ids: Optional[torch.LongTensor] = None,
):
    image_token_id = model.config.image_token_id
    video_token_id = model.config.video_token_id
    audio_token_id = model.config.audio_token_id

    image_mask = input_ids == image_token_id if image_token_id is not None else torch.zeros_like(input_ids, dtype=torch.bool)
    video_mask = input_ids == video_token_id if video_token_id is not None else torch.zeros_like(input_ids, dtype=torch.bool)
    audio_mask = input_ids == audio_token_id if audio_token_id is not None else torch.zeros_like(input_ids, dtype=torch.bool)
    multimodal_mask = image_mask | video_mask | audio_mask

    pad_token_id = _resolve_pad_token_id(model)
    llm_input_ids = input_ids.clone()
    llm_input_ids[multimodal_mask] = pad_token_id
    inputs_embeds = model.get_input_embeddings()(llm_input_ids)

    # Gemma's per-layer skip-embedding path — only present when ``hidden_size_per_layer_input``
    # is set on the text config. Mirrors the upstream forward.
    per_layer_inputs = None
    text_config = model.config.get_text_config()
    if getattr(text_config, "hidden_size_per_layer_input", None):
        pad_embedding = model.language_model.embed_tokens.weight[pad_token_id, :]
        llm_inputs_embeds = torch.where(
            multimodal_mask[..., None], pad_embedding.view(1, 1, -1), inputs_embeds
        )
        per_layer_inputs = model.language_model.get_per_layer_inputs(llm_input_ids, llm_inputs_embeds)

    # ----- image -----
    if pixel_values is not None:
        image_features = model.get_image_features(
            pixel_values, image_position_ids, return_dict=True
        ).pooler_output
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        n_image_tokens = image_mask.sum().item()
        n_image_features = image_features.numel() // image_features.shape[-1]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, image_features)

    # ----- video -----
    if pixel_values_videos is not None:
        video_features = model.get_video_features(
            pixel_values_videos, video_position_ids, return_dict=True
        ).pooler_output
        video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
        n_video_tokens = video_mask.sum().item()
        n_video_features = video_features.numel() // video_features.shape[-1]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )
        video_mask_expanded = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask_expanded, video_features)

    # ----- audio -----
    if input_features is not None and input_features_mask is not None:
        audio_output = model.get_audio_features(input_features, input_features_mask, return_dict=True)
        audio_features = audio_output.pooler_output
        audio_attn_mask = audio_output.attention_mask  # True = valid soft tokens
        # Strip padded audio soft tokens, mirroring the upstream forward.
        audio_features = audio_features[audio_attn_mask]
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        n_audio_tokens = audio_mask.sum().item()
        n_audio_features = audio_features.shape[0]
        if n_audio_tokens != n_audio_features:
            raise ValueError(
                f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
            )
        audio_mask_expanded = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask_expanded, audio_features)

    # Dummy vision/audio passes to keep gradients flowing on text-only batches when training
    # adapters on the multimodal towers (e.g. LoRA over vision_tower / audio_tower).
    if (
        pixel_values is None
        and pixel_values_videos is None
        and input_features is None
        and getattr(model, "vision_tower", None) is not None
    ):
        try:
            vc = model.config.vision_config
            patch_dim = vc.num_channels * vc.patch_size * vc.patch_size
            dummy_pixels = torch.zeros((1, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            dummy_pos = torch.zeros((1, 1, 2), dtype=torch.long, device=inputs_embeds.device)
            dummy_out = model.get_image_features(dummy_pixels, dummy_pos, return_dict=True)
            inputs_embeds = inputs_embeds + 0.0 * dummy_out.pooler_output.mean()
        except Exception:
            # Vision tower input shape varies across model variants — skip the dummy if it would error.
            pass

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "per_layer_inputs": per_layer_inputs,
    }


def gemma4_base_forward(
    self: "Gemma4Model",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    input_features_mask: Optional[torch.Tensor] = None,
    image_position_ids: Optional[torch.LongTensor] = None,
    video_position_ids: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.LongTensor] = None,
    **kwargs,
):
    input_kwargs = _get_input_embeds(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_values_videos,
        input_features,
        input_features_mask,
        image_position_ids,
        video_position_ids,
    )
    kwargs.update(input_kwargs)  # avoid lora module to have multiple keyword arguments
    # mm_token_type_ids is consumed only by the upstream bidirectional vision-attention mask
    # (create_causal_mask_mapping). Under Ulysses + flash-attn we rely on packed position_ids,
    # so we drop it here.
    outputs = self.language_model(input_ids=None, **kwargs)
    return Gemma4ModelOutputWithPast(last_hidden_state=outputs.last_hidden_state)


def gemma4_model_forward(
    self: "Gemma4ForConditionalGeneration",
    input_ids: torch.LongTensor,
    labels: Optional[torch.LongTensor] = None,
    **kwargs,
) -> "Gemma4CausalLMOutputWithPast":
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    # Gemma's final-logit softcap (kept here because it's not applied inside the language_model).
    text_config = self.config.get_text_config()
    final_logit_softcapping = getattr(text_config, "final_logit_softcapping", None)
    if final_logit_softcapping is not None:
        logits = logits / final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * final_logit_softcapping

    return Gemma4CausalLMOutputWithPast(logits=logits)
