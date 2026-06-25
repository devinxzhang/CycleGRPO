import copy
import torch
from types import MethodType
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print("use npu success!")
except:
    print("npu not enabled!")
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModel
from mmengine.config import Config, ConfigDict
from xtuner.registry import BUILDER
from xtuner.model.utils import find_all_linear_names
from peft import get_peft_model, prepare_model_for_kbit_training
# from xtuner.model.utils import guess_load_checkpoint

import os.path as osp
from typing import List, Optional

from projects.transformers.vq_sam2.losses import CrossEntropyLoss, DiceLoss, point_sample, get_uncertain_point_coords_with_randomness


def prepare_inputs_for_generation_cache(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen2-5-VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs


def guess_load_checkpoint(pth_model):
    if osp.isfile(pth_model):
        state_dict = torch.load(pth_model, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    elif osp.isdir(pth_model):
        try:
            from xtuner.utils.zero_to_any_dtype import (
                get_state_dict_from_zero_checkpoint,
            )
        except ImportError:
            raise ImportError(
                "The provided PTH model appears to be a DeepSpeed checkpoint. "
                "However, DeepSpeed library is not detected in current "
                "environment. This suggests that DeepSpeed may not be "
                "installed or is incorrectly configured. Please verify your "
                "setup."
            )
        state_dict = get_state_dict_from_zero_checkpoint(
            osp.dirname(pth_model), osp.basename(pth_model)
        )
    else:
        raise FileNotFoundError(f"Cannot find {pth_model}")
    return state_dict

class QWEN25VL_VQSAM2Model(BaseModel):
    def __init__(
        self,
        qwen25vl_hf_model,
        vqsam2_hf_model,
        tokenizer=None,
        preprocessor=None,
        llm_lora=None,
        vqsam2_pretrained_weights=None,
        pretrained_pth=None,
        freeze_sam2_decoder=False,
        loss_sample_points=False,
    ):
        super(QWEN25VL_VQSAM2Model, self).__init__()

        #==================VQ-SAM2===================
        vq_sam2_config = vqsam2_hf_model['config']
        sam2_config = BUILDER.build(vq_sam2_config['sam2_config'])
        vq_sam2_config.update({'sam2_config': sam2_config})
        vq_sam2_config = BUILDER.build(vq_sam2_config)
        vqsam2_hf_model.update({'config': vq_sam2_config})
        
        self.vqsam2_model = BUILDER.build(vqsam2_hf_model)

        #==================QWEN25VL===================
        self.qwen25vl_model = BUILDER.build(qwen25vl_hf_model)
        self.tokenizer = BUILDER.build(tokenizer)
        self.preprocessor = BUILDER.build(preprocessor)
        self.mt_start_token_id = self.tokenizer.convert_tokens_to_ids('<|mt_start|>')
        self.mt_end_token_id = self.tokenizer.convert_tokens_to_ids('<|mt_end|>')

        # #==================bridge layer===============
        # llm_dim = self.qwen25vl_model.config.hidden_size
        # sam_dim = vq_sam2_config.latent_dim
        # self.llm_to_sam = nn.Sequential(
        #     nn.LayerNorm(llm_dim),
        #     nn.Linear(llm_dim, sam_dim),
        #     nn.GELU(),
        #     nn.Linear(sam_dim, sam_dim)
        # )
        
        #==================freeze parameters================
        self.vqsam2_model.model.requires_grad_(False)
        if not freeze_sam2_decoder:
            self.vqsam2_model.model.sam2_model.sam_mask_decoder.requires_grad_(True)
            self.vqsam2_model.model.sam2_model.sam_mask_decoder.pred_obj_score_head.requires_grad_(False)
            self.vqsam2_model.model.sam2_model.sam_mask_decoder.iou_prediction_head.requires_grad_(False)
        
        self.qwen25vl_model.model.visual.requires_grad_(False)
        if llm_lora is not None:
            self.qwen25vl_model.model.requires_grad_(False)

        if hasattr(self.qwen25vl_model, "enable_input_require_grads"):
            self.qwen25vl_model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self.qwen25vl_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        self.gradient_checkpointing_enable()

        #================support lora====================
        if llm_lora is not None:
            self.qwen25vl_model.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_cache, self.qwen25vl_model.language_model)
            self._prepare_llm_for_lora(llm_lora)
            self.qwen25vl_model.get_input_embeddings().requires_grad_(True)
            self.qwen25vl_model.get_output_embeddings().weight.requires_grad_(True)
            self.qwen25vl_model.tie_weights()

        #==================load weights===============
        if vqsam2_pretrained_weights is not None:
            pretrained_state_dict = guess_load_checkpoint(vqsam2_pretrained_weights)
            pretrained_state_dict_new = {}
            for key in pretrained_state_dict.keys():
                new_key = copy.deepcopy(key)
                if key.startswith('hf_model.'):
                    new_key = new_key[len('hf_model.'):]
                pretrained_state_dict_new[new_key] = pretrained_state_dict[key]
            self.vqsam2_model.load_state_dict(pretrained_state_dict_new)
        
        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained_pth}")

        #================loss=================
        self.loss_sample_points = loss_sample_points
        self.num_points = 12544
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75
        self.loss_mask = CrossEntropyLoss(use_sigmoid=True, reduction='mean', loss_weight=2.0)
        self.loss_dice = DiceLoss(use_sigmoid=True, activate=True, reduction='mean', naive_dice=True, eps=1.0, loss_weight=0.5)


    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.qwen25vl_model.language_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.qwen25vl_model.language_model.gradient_checkpointing_disable()

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
            lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.qwen25vl_model.model.language_model = prepare_model_for_kbit_training(self.qwen25vl_model.model.language_model, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.qwen25vl_model.model.language_model)
            lora_config.target_modules = modules
        
        self.qwen25vl_model.model.language_model = get_peft_model(self.qwen25vl_model.model.language_model, lora_config)
    
    def _merge_lora(self):
        try:
            self.qwen25vl_model.model.language_model = self.qwen25vl_model.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        return
    
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        from collections import OrderedDict

        to_return = OrderedDict()

        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'language_model.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'llm_to_sam.' in k})

        return to_return
        
    def init_weights(self):
        pass

    def forward(self, data, data_samples=None, mode='loss'):
        # for n, p in self.named_parameters():
        #     if p.requires_grad:
        #         print(n)
        # exit(0)

        sam2_pixel_values = data.pop('sam2_pixel_values', None) #[2, 3, 1024, 1024]
        gt_masks = data.pop('masks', None) #list each has shape like this [1, 480, 640]

        qwen25vl_output = self.qwen25vl_model(**data, use_cache=False, output_hidden_states=True)

        # hidden_states = qwen25vl_output.hidden_states
        # hidden_states = self.llm_to_sam(hidden_states[-1]) # hidden_states[-1].shape: [2, 485, 2048]

        # labels = data['labels']

        # shift_labels = labels[..., 1:].contiguous()
        # shift_hidden_states = hidden_states[:, :-1, :]
        # C = shift_hidden_states.shape[-1]

        # mt_token_mask = torch.logical_and(shift_labels >= self.mt_start_token_id, shift_labels <= self.mt_end_token_id)
        
        # _zero = hidden_states.mean() * 0.0
        # loss_dice, loss_mask = 0.0, 0.0
        # if gt_masks is None:
        #     loss_mask += _zero
        #     loss_dice += _zero
        # else:
        #     seg_valid = True
        #     pred_embeddings = shift_hidden_states[mt_token_mask] + _zero        
        #     pred_embeddings = pred_embeddings.reshape(-1, 6, C)
            
        #     MAX_MASKS_IN_BATCH = 20
        #     for depth_idx in range(1, 5):
        #         sam2_pixel_values_per_depth = sam2_pixel_values[:MAX_MASKS_IN_BATCH]
        #         pred_mt_embeds_per_depth = torch.sum(pred_embeddings[:MAX_MASKS_IN_BATCH, :depth_idx, :], dim=1)
        #         gt_masks_per_depth = [gt_mask if len(gt_mask.shape) == 3 else gt_mask.unsqueeze(0) for gt_mask in gt_masks[:MAX_MASKS_IN_BATCH]]
        #         pred_masks = self.vqsam2_model.forward_with_embeds(sam2_pixel_values_per_depth, pred_mt_embeds_per_depth)
                
        #         gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks.shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_per_depth]
        #         gt_masks = torch.cat(gt_masks, dim=0)
        #         if len(pred_masks) != len(gt_masks):
        #             # drop this data
        #             print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
        #             min_num = min(len(pred_masks), len(gt_masks))
        #             pred_masks = pred_masks[:min_num]
        #             gt_masks = gt_masks[:min_num]
        #             seg_valid = False

        #         valid_flag = torch.sum(gt_masks, dim=(-2, -1)) > 0
        #         pred_masks = pred_masks.flatten(0, 1)[valid_flag]
        #         gt_masks = gt_masks[valid_flag]

        #         if self.loss_sample_points:
        #             sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
        #             loss_dice_per_depth = self.loss_dice(
        #                 sampled_pred_mask,
        #                 sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
        #             loss_mask_per_depth = self.loss_mask(
        #                 sampled_pred_mask.reshape(-1),
        #                 sampled_gt_mask.reshape(-1),
        #                 avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
        #         else:
        #             loss_mask_per_depth = self.loss_mask(pred_masks, gt_masks)
        #             loss_dice_per_depth = self.loss_dice(pred_masks, gt_masks)

        #         if not seg_valid:
        #             loss_dice += loss_dice_per_depth * 0.0
        #             loss_mask += loss_mask_per_depth * 0.0
        #         else:
        #             loss_dice += loss_dice_per_depth
        #             loss_mask += loss_mask_per_depth
        
        loss_dict = {
            # 'loss_mask': loss_mask,
            # 'loss_dice': loss_dice,
            'llm_loss': qwen25vl_output.loss,
        }

        return loss_dict
                
    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        # (N, 1, h, w)

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.config.num_points,
                self.config.oversample_ratio, self.config.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)



if __name__ == "__main__":
    save_path = "./work_dirs/qwen25vl_3b_t2m_m2t_v4_refcoco_unfreeze_visionencoder/hf_ckpt25k"
    cfg = Config.fromfile('projects/vlm/vq_sam2/configs/a100_qwen25vl/qwen25vl-3b_vqsam2_t2m_m2t_refcoco.py')
   
    model = QWEN25VL_VQSAM2Model(
        qwen25vl_hf_model=cfg.model.qwen25vl_hf_model,
        vqsam2_hf_model=cfg.model.vqsam2_hf_model,
        tokenizer=cfg.model.tokenizer,
        preprocessor=cfg.model.preprocessor,
        llm_lora=cfg.model.llm_lora,
        vqsam2_pretrained_weights=cfg.model.vqsam2_pretrained_weights,
        pretrained_pth=cfg.model.pretrained_pth,
        freeze_sam2_decoder=cfg.model.freeze_sam2_decoder,
        loss_sample_points=False
    )

    # for k, v in model.named_parameters():
    #     print(k, v.dtype)

    # model_state_dict = model.state_dict()
    # for k, v in model_state_dict.items():
    #     if k.startswith('qwen25vl_model.'):
    #         print(k)
    # exit(0)

    # qwen25vl_model.model.language_model.base_model.model

    # qwen25vl_model.model.language_model

    pth_path = "work_dirs/qwen25vl_3b_t2m_m2t_v4_refcoco_unfreeze_visionencoder/iter_25000.pth"
    state = torch.load(pth_path, map_location="cpu", weights_only=False)
    model_sd = (state.get("state_dict")
                or state.get("model")
                or state.get("module")
                or state)
    if any(k.startswith("module.") for k in model_sd.keys()):
        model_sd = {k.replace("module.", "", 1): v for k, v in model_sd.items()}

    # if 'qwen25vl_model.lm_head.weight' not in model_sd:
    #     model_sd['qwen25vl_model.lm_head.weight'] = copy.deepcopy(model_sd['qwen25vl_model.model.language_model.base_model.model.embed_tokens.weight'])

    # qwen25vl_model.model.language_model.base_model.model.embed_tokens.weight

    # for k, v in model_sd.items():
    #     print(k, v.dtype)
    # exit(0)




    # new_state_dict = {}
    # for k, v in model_sd.items():
    #     if k.startswith('qwen25vl_model.model.language_model'):
    #         new_k = copy.deepcopy(k).replace('qwen25vl_model.model.language_model.', 'qwen25vl_model.model.language_model.base_model.model.')
    #         new_state_dict[new_k] = copy.deepcopy(v)
    #     elif k.startswith('llm_to_sam'):
    #         new_k = copy.deepcopy(k)
    #         new_state_dict[new_k] = copy.deepcopy(v)

   
    model.load_state_dict(model_sd, strict=False)
    print(f'Load PTH model from {pth_path}')

    model._merge_lora()
    model.qwen25vl_model.tie_weights()

    model.qwen25vl_model.save_pretrained(save_path)
    model.tokenizer.save_pretrained(save_path)
    model.preprocessor.save_pretrained(save_path)
    print("Done!")

