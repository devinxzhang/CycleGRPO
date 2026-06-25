import os
import torch
from os import PathLike
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from .configuration_vq_sam2 import SAM2Config

from hydra.utils import instantiate
# from .sam2.utils.load_checkpoint import load_checkpoint_with_prefix, load_state_dict_to_model


class SAM2Model(PreTrainedModel):
    config_class = SAM2Config
    base_model_prefix = "sam2"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)

        self.sam2_model = instantiate(config.cfg.model, _recursive_=True)
        
        self.hidden_dim = self.sam2_model.hidden_dim
        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
    #     from sam2.build_sam import build_sam2_hf

    #     cfg_path, ckpt_path = build_sam2_hf(pretrained_model_name_or_path)
    #     config = SAM2Config(cfg_path, ckpt_path)

    #     model = cls(config)

    #     state_dict = load_checkpoint_with_prefix(ckpt_path)
    #     load_state_dict_to_model(model.sam2_model, state_dict)

    #     return model
    
    # def load_pretrained_weights(self, ckpt_path, strict=True):
    #     state_dict = load_checkpoint_with_prefix(ckpt_path)
    #     load_state_dict_to_model(self.sam2_model, state_dict, strict=strict)
        
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image / 255.
        img_mean = torch.tensor(self.img_mean, dtype=image.dtype, device=image.device)[:, None, None]
        img_std = torch.tensor(self.img_std, dtype=image.dtype, device=image.device)[:, None, None]
        image -= img_mean
        image /= img_std
        return image
    
    def encode_mask_box_input(self, sam_states, mask_input, box_input_normalized, sam2_resolution=1024):
        if box_input_normalized is not None:
            box_input_normalized = box_input_normalized.reshape(-1, 2, 2)
            box_input_normalized = box_input_normalized * sam2_resolution
            box_labels = torch.tensor([[2,3]], dtype=torch.int, device=box_input_normalized.device)
            box_labels = box_labels.repeat(box_input_normalized.shape[0], 1)
            concat_points = (box_input_normalized, box_labels)
        else:
            concat_points = None
        
        sam_mask_prompt = [torch.nn.functional.interpolate(
            one_mask.unsqueeze(0).float(),
            size=self.sam2_model.sam_prompt_encoder.mask_input_size,
            align_corners=False,
            mode="bilinear",
            antialias=True).squeeze(0) for one_mask in mask_input]
        sam_mask_prompt = torch.cat(sam_mask_prompt, dim=0).unsqueeze(1)

        sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=sam_mask_prompt,
        )

        B = sam_states['current_vision_feats'][-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = sam_states['feat_sizes'][-1]

        if self.sam2_model.directly_add_no_mem_embed:
            # directly add no-mem embedding (instead of using the transformer encoder)
            pix_feat_with_mem = sam_states['current_vision_feats'][-1] + self.sam2_model.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        else:
            raise NotImplementedError("directly add no memory embedding is not implemented")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mask_tokens = self.sam2_model.sam_mask_encoder(
                image_embeddings=pix_feat_with_mem,
                image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                repeat_image=False,
            )

        return mask_tokens

    def inject_language_embd(self, sam_states, language_embed, nf_nobj=None):
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(sam_states['current_vision_feats'][:-1], sam_states['feat_sizes'][:-1])
        ]

        B = sam_states['current_vision_feats'][-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = sam_states['feat_sizes'][-1]

        if self.sam2_model.directly_add_no_mem_embed:
            # directly add no-mem embedding (instead of using the transformer encoder)
            pix_feat_with_mem = sam_states['current_vision_feats'][-1] + self.sam2_model.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        else:
            raise NotImplementedError("directly add no memory embedding is not implemented")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, _, _, low_res_masks, high_res_masks, obj_ptr, _, = self.sam2_model._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=None,
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=self.sam2_model._use_multimask(is_init_cond_frame=True, point_inputs=None),
                # Inject language Embed if possible
                language_embed=language_embed,
            )

        if nf_nobj is not None:
            pred_masks = low_res_masks.squeeze(1)
            pred_masks = pred_masks.unflatten(0, nf_nobj)
        else:
            pred_masks = low_res_masks
        return pred_masks
    
    def get_sam2_embeddings(self, images, expand_size=1):
        # Step 1: inference the backbone with the images
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feats = self.sam2_model.forward_image(images)

        if expand_size > 1:
            # feats['vision_features'] = feats['vision_features'][:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
            for i, feat in enumerate(feats["backbone_fpn"]):
                feats["backbone_fpn"][i] = feat[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
            for i, pos in enumerate(feats["vision_pos_enc"]):
                pos = pos[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
                feats["vision_pos_enc"][i] = pos

        # Step 2: Process the features to output
        _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self.sam2_model._prepare_backbone_features(feats)

        return {
            "current_vision_feats": current_vision_feats,
            "current_vision_pos_embeds": current_vision_pos_embeds,
            "feat_sizes": feat_sizes,
        }

    def forward(self, pixel_values):
        raise NotImplementedError