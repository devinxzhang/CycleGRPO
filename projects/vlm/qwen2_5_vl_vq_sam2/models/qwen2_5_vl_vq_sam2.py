import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModel
from mmengine.config import Config, ConfigDict
from xtuner.registry import BUILDER

from peft import prepare_model_for_kbit_training, get_peft_model

import os.path as osp
from typing import List, Optional
from types import MethodType
import copy

from transformers import AutoConfig
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

class Qwen2_5_VL_VQ_SAM2(BaseModel):
    def __init__(
        self, 
        vq_sam2_config,
        base_mllm_path,
        tokenizer=None,
        preprocessor=None,
        freeze_llm=False,
        freeze_visual_encoder=False,
        data_flatten=True,
        mllm_path=None,
        llm_lora_config=None,
    ):
        super(Qwen2_5_VL_VQ_SAM2, self).__init__()

        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder

        if mllm_path is None:
            assert base_mllm_path is not None, "mllm_path or base_mllm_path must be provided"

            # sam2_config = BUILDER.build(vq_sam2_config['sam2_config'])
            # vq_sam2_config.update({'sam2_config': sam2_config})
            # vq_sam2_config = BUILDER.build(vq_sam2_config)

            # add special tokens
            self.tokenizer = BUILDER.build(tokenizer)
            print("ORI TOKENIZER LENGTH: ", len(self.tokenizer))
            MT_START_TOKEN = '<|mt_start|>'
            MT_END_TOKEN = '<|mt_end|>'
            MT_CONTEXT_TOKEN = '<|mt_{}|>'
            num_codebooks = 1 if vq_sam2_config["shared_codebook"] else vq_sam2_config["codebook_depth"]
            if isinstance(vq_sam2_config["codebook_size"], list):
                num_spatial_tokens = sum(vq_sam2_config["codebook_size"]) + 2
            else:
                num_spatial_tokens = vq_sam2_config["codebook_size"] * num_codebooks + 2 # +2 for the start and end tokens
            special_tokens = [MT_START_TOKEN] + [MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in range(num_spatial_tokens-2)] + [MT_END_TOKEN]
            # special_tokens = [MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in range(num_spatial_tokens)]
            add_tokens = dict(additional_special_tokens=special_tokens)
            self.special_tokens = special_tokens

            #replace unk_token to '<|endoftext|>'
            old_unk_token = self.tokenizer.unk_token
            print("old_unk_token: ", old_unk_token)
            self.tokenizer.unk_token = '<|endoftext|>'

            add_len = self.tokenizer.add_special_tokens(
                add_tokens,
                replace_additional_special_tokens=False
            )

            #recover unk_token
            self.tokenizer.unk_token = old_unk_token
            print("recover unk_token: ", self.tokenizer.unk_token)
            print(f"Added {add_len} special tokens")

            # mt_start_token_id = self.tokenizer.convert_tokens_to_ids(MT_START_TOKEN)
            # mt_end_token_id = self.tokenizer.convert_tokens_to_ids(MT_END_TOKEN)

            print("AFTER TOKENIZER LENGTH: ", len(self.tokenizer))
            
            self.preprocessor = BUILDER.build(preprocessor)

            base_mllm_config = AutoConfig.from_pretrained(base_mllm_path)
            base_mllm_config_dict = base_mllm_config.to_dict()
            
            ori_vocab_size = copy.deepcopy(base_mllm_config_dict['vocab_size'])
            # base_mllm_config_dict.update({'vocab_size': ori_vocab_size + num_spatial_tokens, 'mt_start_token_id': mt_start_token_id, 'mt_end_token_id': mt_end_token_id})
            base_mllm_config_dict.update({'vocab_size': ori_vocab_size + num_spatial_tokens})
            # print(base_mllm_config_dict)
            # exit(0)
            # base_mllm_config_dict['text_config'].update({'vocab_size': ori_vocab_size + num_spatial_tokens})

            mllm_config = Qwen2_5_VLConfig(**base_mllm_config_dict)
            self.hf_model = Qwen2_5_VLForConditionalGeneration(mllm_config)

            for key, value in base_mllm_config_dict.items():
                print(f"{key}=============>", value)

            print("ori_vocab_size: ", ori_vocab_size)
            print("num_spatial_tokens: ", num_spatial_tokens)
            print("self.hf_model.get_input_embeddings().shape: ", self.hf_model.get_input_embeddings().weight.shape)
            print("self.hf_model.get_output_embeddings().shape: ", self.hf_model.get_output_embeddings().weight.shape)

            # load pretrained MLLM weights
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_mllm_path)
            model.resize_token_embeddings(ori_vocab_size + num_spatial_tokens)

            print("model.get_input_embeddings().shape:", model.get_input_embeddings().weight.shape)
            print("model.get_output_embeddings().shape: ", model.get_output_embeddings().weight.shape)

            self._load_pretrained_weights_strict(model)

            print("after self.hf_model.get_input_embeddings().shape: ", self.hf_model.get_input_embeddings().weight.shape)
            print("after self.hf_model.get_output_embeddings().shape: ", self.hf_model.get_output_embeddings().weight.shape)
        else:
            self.hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(mllm_path, attn_implementation="flash_attention_2", torch_dtype="bfloat16", use_cache=False)
        
        # gradient settings
        if self.freeze_llm:
            self.hf_model.model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.hf_model.visual.requires_grad_(False)
        
        if hasattr(self.hf_model, "enable_input_require_grads"):
            self.hf_model.enable_input_require_grads()
        else:
            
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self.hf_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        self.hf_model.gradient_checkpointing_enable()

        if llm_lora_config is not None:
            self.hf_model.model.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_cache, self.hf_model.model.language_model)
            self._prepare_llm_for_lora(llm_lora_config)
            self.hf_model.model.get_input_embeddings().requires_grad_(True)

        if data_flatten:
            replace_qwen2_vl_attention_class()
    
    def _parse_lora_config(self, lora_config):
        if (
            isinstance(lora_config, dict)
            or isinstance(lora_config, Config)
            or isinstance(lora_config, ConfigDict)
        ):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.hf_model.model.language_model = prepare_model_for_kbit_training(
            self.hf_model.model.language_model, use_activation_checkpointing
        )
        
        self.hf_model.model.language_model = get_peft_model(
            self.hf_model.model.language_model, lora_config
        )

    def _load_pretrained_weights_strict(self, pretrained_model):
        
        pretrained_state_dict = {}
        for key, value in pretrained_model.state_dict().items():
            pretrained_state_dict[key] = value.detach().cpu()
        
        current_state_dict = {}
        for key, value in self.hf_model.state_dict().items():
            current_state_dict[key] = value.detach().cpu()
        
        missing_keys = []
        unexpected_keys = []
        shape_mismatch_keys = []
        
        for key in current_state_dict.keys():
            if key not in pretrained_state_dict:
                missing_keys.append(key)
            elif current_state_dict[key].shape != pretrained_state_dict[key].shape:
                shape_mismatch_keys.append(f"{key}: {current_state_dict[key].shape} vs {pretrained_state_dict[key].shape}")
        
        for key in pretrained_state_dict.keys():
            if key not in current_state_dict:
                unexpected_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Warning=>Error: Missing keys in pretrained model: {missing_keys}")
        if unexpected_keys:
            raise ValueError(f"Warning=>Error: Unexpected keys in pretrained model: {unexpected_keys}")
        if shape_mismatch_keys:
            raise ValueError(f"Shape mismatch between current model and pretrained model:\n" + "\n".join(shape_mismatch_keys))
     
        new_state_dict = {}
        for key in current_state_dict.keys():
            if key in pretrained_state_dict:
                new_state_dict[key] = pretrained_state_dict[key]
        
        load_result = self.hf_model.load_state_dict(new_state_dict, strict=False)
        
        if load_result.missing_keys:
            raise ValueError(f"Warning=>Error: Some keys were not loaded: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            raise ValueError(f"Warning=>Error: Some unexpected keys were ignored: {load_result.unexpected_keys}")
        
        del pretrained_model
        del pretrained_state_dict
        del current_state_dict
        del new_state_dict
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("Pretrained weights loaded successfully!")
    
    def forward(self, data, data_samples=None, mode='loss'):
        pixel_values = copy.deepcopy(data.pop('pixel_values', None))
        image_grid_thw = copy.deepcopy(data.pop('image_grid_thw', None))

        input_ids = copy.deepcopy(data.pop('input_ids', None))
        # position_ids = copy.deepcopy(data.pop('position_ids', None))
        position_ids = None
        attention_mask = copy.deepcopy(data.pop('attention_mask', None))
        labels = copy.deepcopy(data.pop('labels', None))

        assert torch.max(input_ids) < self.hf_model.vocab_size, f"{torch.max(input_ids)} < {self.hf_model.vocab_size}"
        
        # try:
        outputs = self.hf_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        return {'llm_loss': outputs.loss}
        # except:
        #     print(f"pixel_values.device={pixel_values.device}, image_grid_thw.device={image_grid_thw.device}, input_ids.device={input_ids.device}, attention_mask.device={attention_mask.device}, labels.device={labels.device}")
        #     sum_loss = 0.0
        #     for n, p in self.hf_model.named_parameters():
        #         if p.requires_grad:
        #             sum_loss += p.sum() * 0.0
        #     return {'llm_loss': sum_loss}
    


def prepare_inputs_for_generation_cache(
        self,
        input_ids,  ## yes
        past_key_values=None,
        attention_mask=None, ## yes
        inputs_embeds=None,
        cache_position=None,
        position_ids=None, 
        use_cache=True, 
        pixel_values=None, ## yes
        pixel_values_videos=None,
        image_grid_thw=None, ## yes
        video_grid_thw=None,
        second_per_grid_ts=None,
        visual_prompt_embeds_out=None, ## chenshihao
        img_context_token_idx=None, ## chenshihao
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        vit_embeds = self.visual.foward_add_vp_mutil_single(pixel_values, grid_thw=image_grid_thw, visual_prompt_embeds=visual_prompt_embeds_out)
        B, N = input_ids.shape
        input_embeds = self.model.get_input_embeddings()(input_ids).clone()

        B, N, C  = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)

        skip_this_case = False
        selected = (input_ids == img_context_token_idx)
        true_count = selected.sum().item()
        # print(f"Number of True elements: {true_count}")
        # print("vit_embeds:",vit_embeds.shape)
        input_ids = input_ids.reshape(B, N)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f"warning: {e}, input_embeds[selected].shape="
                  f"{input_embeds[selected].shape}, "
                  f"vit_embeds.shape={vit_embeds.shape}")
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
        input_embeds = input_embeds.reshape(B, N, C)


        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
                "second_per_grid_ts": second_per_grid_ts,
            }
        )
        return model_inputs


if __name__ == "__main__":
    # import json

    # name = "qwen2_5_vl_vq_sam2_7b_256x4_910b"

    # cfg = Config.fromfile("projects/vlm/qwen2_5_vl_vq_sam2/configs/qwen2_5_vl_vq_sam2_7b_convert.py")
    # model = BUILDER.build(cfg.model)
    # model.tokenizer.save_pretrained(f"./pretrained_weights/{name}")
    # model.hf_model.save_pretrained(f"./pretrained_weights/{name}")
    # model.preprocessor.save_pretrained(f"./pretrained_weights/{name}")

    # with open(f'./pretrained_weights/{name}/added_tokens.json', 'r') as f:
    #     added_tokens = json.load(f)
    #     for special_token in model.special_tokens:
    #         added_tokens.update({special_token: model.tokenizer.convert_tokens_to_ids(special_token)})
    # with open(f'./pretrained_weights/{name}/added_tokens.json', 'w') as f:
    #     json.dump(added_tokens, f)

    # with open(f'./pretrained_weights/{name}/tokenizer_config.json', 'r') as f:
    #     tokenizer_config = json.load(f)
    #     for special_token in model.special_tokens:
    #         tokenizer_config['added_tokens_decoder'].update({
    #             f"{model.tokenizer.convert_tokens_to_ids(special_token)}": {
    #                 "content": special_token,
    #                 "lstrip": False,
    #                 "normalized": False,
    #                 "rstrip": False,
    #                 "single_word": False,
    #                 "special": True
    #             }
    #         })
    
    # with open(f'./pretrained_weights/{name}/tokenizer_config.json', 'w') as f:
    #     json.dump(tokenizer_config, f)

    # with open(f'./pretrained_weights/{name}/tokenizer.json', 'r') as f:
    #     tokenizer = json.load(f)
    #     for special_token in model.special_tokens:
    #         tokenizer["added_tokens"].append({
    #             "id": model.tokenizer.convert_tokens_to_ids(special_token),
    #             "content": special_token,
    #             "single_word": False,
    #             "lstrip": False,
    #             "rstrip": False,
    #             "normalized": False,
    #             "special": True
    #         })
    # with open(f'./pretrained_weights/{name}/tokenizer.json', 'w') as f:
    #     json.dump(tokenizer, f)


    from transformers import AutoProcessor, AutoModelForCausalLM
    from tokenizers import AddedToken
    import torch, os

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    save_dir = "Qwen/Qwen2.5-VL-3B-MT-256+128"

    # 1) 扩 tokenizer
    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'
    new_tokens = [MT_START_TOKEN] + [MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in range(256+128)] + [MT_END_TOKEN]
    # new_tokens = ['[SEG]']

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    added = tokenizer.add_tokens(
        [AddedToken(t, lstrip=False, rstrip=False, single_word=False, normalized=False) for t in new_tokens],
        special_tokens=False,
    )
    print("added:", added)
    os.makedirs(save_dir, exist_ok=True)
    processor.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # 2) 扩模型词表并初始化新增行
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        old_vocab = emb.shape[0] - added
        mu = emb[:old_vocab].mean(0, keepdim=True)
        std = emb[:old_vocab].std(0, keepdim=True).clamp_min(1e-3)
        emb[old_vocab:].copy_(mu + 0.02 * torch.randn_like(emb[old_vocab:]) * std)

    model.save_pretrained(save_dir)
    print("Saved to", save_dir)


