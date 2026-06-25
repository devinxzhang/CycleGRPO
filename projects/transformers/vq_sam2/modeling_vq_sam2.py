import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.distributed as dist

from typing import Any, Callable, Optional, Union, Iterable
from dataclasses import dataclass
import numpy as np

from transformers import PreTrainedModel
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.utils import can_return_tuple, ModelOutput
from transformers.activations import ACT2FN

from .configuration_vq_sam2 import VQ_SAM2Config
from .modeling_sam2 import SAM2Model
from .losses import CrossEntropyLoss, DiceLoss, point_sample, get_uncertain_point_coords_with_randomness

        
class VQEmebedding(nn.Embedding):
    """VQ embedding module with ema update."""

    def __init__(
        self, 
        codebook_size: int, 
        embedding_dim: int, 
        ema: bool=True, 
        decay: float=0.99,
        restart_unused_codes: bool=True,
        eps: float=1e-5,
    ):
        super().__init__(num_embeddings=codebook_size+1, embedding_dim=embedding_dim, padding_idx=codebook_size)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.codebook_size = codebook_size

        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]

            # padding index is not updated by EMA
            self.register_buffer('cluster_size_ema', torch.zeros(codebook_size))
            self.register_buffer('embed_ema', self.weight[:-1, :].detach().clone())
    
    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.weight[:-1, :].t().contiguous()

        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim).contiguous()

        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1).contiguous()
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)
        embed_idxs = distances.argmin(dim=-1)

        return embed_idxs

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B -1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x    
    
    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        
        n_embed, embed_dim = self.weight.shape[0]-1, self.weight.shape[-1]

        vectors = vectors.reshape(-1, embed_dim).contiguous()
        idxs = idxs.reshape(-1).contiguous()

        n_vectors = vectors.shape[0]
        n_total_embed = n_embed

        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(dim=0,
                              index=idxs.unsqueeze(0),
                              src=vectors.new_ones(1, n_vectors)
                              )
        
        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        assert dist.is_initialized()
        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)
        
        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)

        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][:n_embed]
            
            assert dist.is_initialized()
            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)
            
            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1-usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1-usage).view(-1))
    
    @torch.no_grad()
    def _update_embedding(self):

        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1).contiguous()

    def forward(self, inputs, freeze_codebook=False):
        embed_idxs = self.find_nearest_embedding(inputs)
        if self.training and self.ema and not freeze_codebook:
            self._update_buffers(inputs, embed_idxs)
        
        embeds = self.embed(embed_idxs)

        if self.ema and self.training and not freeze_codebook:
            print("================>here: self._update_embedding()")
            # exit(0)
            self._update_embedding()
        # print("================>self.ema and self.training and not freeze_codebook: ", self.ema and self.training and not freeze_codebook)
        
        return embeds, embed_idxs
            
    def embed(self, idxs):
        embeds = super().forward(idxs)
        return embeds

class ResidualQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        latent_dim: int,
        codebook_depth: int,
        decay: float = 0.99,
        shared_codebook: bool = False,
        restart_unused_codes: bool = True,
        commitment_loss: str = 'cumsum'
    ):
        super().__init__()

        self.shared_codebook = shared_codebook
        if self.shared_codebook:
            if isinstance(codebook_size, Iterable) or isinstance(decay, Iterable):
                raise ValueError("Shared codebooks are incompatible with list types of momentums or sizes: Change it into int")
        
        self.restart_unused_codes = restart_unused_codes
        self.codebook_size = codebook_size if isinstance(codebook_size, Iterable) else [codebook_size for _ in range(codebook_depth)]
        self.decay = decay if isinstance(decay, Iterable) else [decay for _ in range(codebook_depth)]
        self.codebook_depth = codebook_depth

        if self.shared_codebook:
            codebook0 = VQEmebedding(codebook_size=self.codebook_size[0], 
                embedding_dim=latent_dim, decay=self.decay[0], restart_unused_codes=restart_unused_codes,)
            self.codebooks = nn.ModuleList([codebook0 for _ in range(codebook_depth)])
        else:
            codebooks = [VQEmebedding(self.codebook_size[idx],
                                      latent_dim,
                                      decay=self.decay[idx],
                                      restart_unused_codes=restart_unused_codes,)
                                      for idx in range(codebook_depth)]
            self.codebooks = nn.ModuleList(codebooks)
        
        self.commitment_loss = commitment_loss

    def quantize(self, x, freeze_codebook=False):
        B, L, C = x.shape

        residual_feature = x.detach().clone()

        quant_list = []
        code_list = []
        aggregated_quants = torch.zeros_like(x)
        for i in range(self.codebook_depth):
            quant, code = self.codebooks[i](residual_feature, freeze_codebook)

            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)

            quant_list.append(aggregated_quants.clone())
            code_list.append(code.unsqueeze(-1))
        
        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes
    
    def compute_commitment_loss(self, x, quant_list):
        r"""
        Compute the commitment loss for the residual quantization.
        The loss is iteratively computed by aggregating quantized features.
        """
        loss_list = []

        for idx, quant in enumerate(quant_list):
            partial_loss = (x - quant.detach()).pow(2.0).mean()
            loss_list.append(partial_loss)
        
        commitment_loss = torch.mean(torch.stack(loss_list))
        return commitment_loss
    
    @torch.no_grad()
    def embed_code(self, code):
        # N, 4

        fake_code = code
        fake_code[code == -1] = 0
        code_slices = torch.chunk(fake_code, chunks=self.codebook_depth, dim=-1)

        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        
        embeds = torch.cat(embeds, dim=-2)
        sum_embeds = []
        for _embeds_, _code_ in zip(embeds, code):
            valid_mask = _code_ != -1
            sum_embeds.append(_embeds_[valid_mask].sum(0))
        
        return torch.stack(sum_embeds, dim=0)

        # embeds = torch.cat(embeds, dim=-2).sum(-2)
        
        # return embeds

    def forward(self, x, freeze_codebook=False):
        quant_list, codes = self.quantize(x, freeze_codebook)

        commitment_loss = self.compute_commitment_loss(x, quant_list)
        quants_trunc = quant_list[-1]
        quants_trunc = x + (quants_trunc - x).detach()

        return quants_trunc, commitment_loss, codes
    

@dataclass
class VQ_SAM2ModelOutput(ModelOutput):
    """
    Base class for VQ_SAM2's output

    """
    loss: Optional[torch.FloatTensor] = None
    loss_recon: Optional[torch.FloatTensor] = None
    loss_quant: Optional[torch.FloatTensor] = None
    pred_masks: Optional[torch.FloatTensor] = None
    continues_mask_embeds: Optional[torch.FloatTensor] = None
    quant_mask_embeds: Optional[torch.FloatTensor] = None
    quant_codes: Optional[torch.LongTensor] = None


    
class VQ_SAM2(PreTrainedModel):
    base_model_prefix = ""
    config_class = VQ_SAM2Config
    _no_split_modules = ["MultiScaleBlock", "TwoWayAttentionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.model = SAM2Model._from_config(config.sam2_config)

        sam_hidden_dim = config.sam2_config.cfg.model.memory_attention.d_model
        self.num_mask_tokens = int(os.environ.get("MASK_TOKENIZER_NUM_MASK_TOKEN", 1))
        if self.num_mask_tokens > 1:
            self.concate_mask_embeds = nn.Sequential(
                nn.LayerNorm(sam_hidden_dim * self.num_mask_tokens),
                nn.Linear(sam_hidden_dim * self.num_mask_tokens, config.latent_dim),
                nn.GELU(),
                nn.Linear(config.latent_dim, config.latent_dim)
            )
            self.deconcate_quant_embed = nn.Sequential(
                nn.LayerNorm(config.latent_dim),
                nn.Linear(config.latent_dim, sam_hidden_dim * self.num_mask_tokens),
                nn.GELU(),
                nn.Linear(sam_hidden_dim * self.num_mask_tokens, sam_hidden_dim * self.num_mask_tokens)
            )
        else:
            self.concate_mask_embeds = nn.Identity()
            self.deconcate_quant_embed = nn.Identity()

        self.quantizer = ResidualQuantizer(
            codebook_size=config.codebook_size,
            latent_dim=config.latent_dim,
            codebook_depth=config.codebook_depth,
            shared_codebook=config.shared_codebook,
            restart_unused_codes=True,
        )

        self.loss_mask = CrossEntropyLoss(use_sigmoid=True, reduction='mean', loss_weight=2.0)
        self.loss_dice = DiceLoss(use_sigmoid=True, activate=True, reduction='mean', naive_dice=True, eps=1.0, loss_weight=0.5)
        
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
    
    def forward_with_codes(self, pixel_values, quant_codes):
        batch_size = len(quant_codes)
        pixel_values = torch.stack([
            self.model.preprocess_image(pixel) for pixel in pixel_values
        ])
        sam2_states = self.model.get_sam2_embeddings(pixel_values, expand_size=1)

        quant_mask_embeds = self.quantizer.embed_code(quant_codes)
        quant_mask_embeds = quant_mask_embeds.unsqueeze(1)
        quant_mask_embeds = self.deconcate_quant_embed(quant_mask_embeds)
        quant_mask_embeds = quant_mask_embeds.reshape(batch_size, self.num_mask_tokens, -1).contiguous()

        pred_masks = self.model.inject_language_embd(sam2_states, quant_mask_embeds, nf_nobj=(batch_size, 1))

        return pred_masks
    
    def forward_with_embeds(self, pixel_values, embeds):
        batch_size = len(embeds)
        pixel_values = torch.stack([
            self.model.preprocess_image(pixel) for pixel in pixel_values
        ])
        sam2_states = self.model.get_sam2_embeddings(pixel_values, expand_size=1)
        embeds = embeds.unsqueeze(1)

        pred_masks = self.model.inject_language_embd(sam2_states, embeds, nf_nobj=(batch_size, 1))

        return pred_masks


    @can_return_tuple
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        gt_masks: Optional[list[torch.Tensor]] = None,
        gt_boxes: Optional[torch.Tensor] = None,
        reconstruct_mask = True,
        freeze_codebook = False,
    ) -> VQ_SAM2ModelOutput:
        """
        Args:
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*).

        """
        assert gt_boxes is not None, "Tokenizer works better given bbox prompt"

        batch_size = len(pixel_values)
        pixel_values = torch.stack([
            self.model.preprocess_image(pixel) for pixel in pixel_values
        ])
        sam2_states = self.model.get_sam2_embeddings(pixel_values, expand_size=1)

        mask_embeds = self.model.encode_mask_box_input(sam2_states, gt_masks, gt_boxes)

        mask_embeds = mask_embeds.reshape(batch_size, 1, -1).contiguous()
        mask_embeds = self.concate_mask_embeds(mask_embeds)
        quant_mask_embeds, quant_loss, code = self.quantizer(mask_embeds, freeze_codebook)
        if not reconstruct_mask:
            return VQ_SAM2ModelOutput(
                quant_codes=code,
            )

        quant_mask_embeds = self.deconcate_quant_embed(quant_mask_embeds)
        quant_mask_embeds = quant_mask_embeds.reshape(batch_size, self.num_mask_tokens, -1).contiguous()

        pred_masks = self.model.inject_language_embd(sam2_states, quant_mask_embeds, nf_nobj=(batch_size, 1))

        loss = quant_loss * self.config.vq_loss_weight
        if self.training and gt_masks is not None:
            gt_masks = [F.interpolate(gt_mask.unsqueeze(0).to(pred_masks.dtype), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks]
            gt_masks = torch.cat(gt_masks, dim=0)
            pred_masks = pred_masks.flatten(0, 1)

            if self.config.loss_sample_points:
                sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
                loss_dice = self.loss_dice(
                    sampled_pred_mask,
                    sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
                loss_mask = self.loss_mask(
                    sampled_pred_mask.reshape(-1),
                    sampled_gt_mask.reshape(-1),
                    avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
            else:
                loss_mask = self.loss_mask(pred_masks, gt_masks)
                loss_dice = self.loss_dice(pred_masks, gt_masks)
            loss += loss_mask + loss_dice
            
            return VQ_SAM2ModelOutput(
                loss=loss,
                loss_recon=loss_mask+loss_dice,
                loss_quant=quant_loss*self.config.vq_loss_weight,
                pred_masks=pred_masks,
                continues_mask_embeds=mask_embeds,
                quant_mask_embeds=quant_mask_embeds,
                quant_codes=code,
            )
        else:
            return VQ_SAM2ModelOutput(
                pred_masks=pred_masks,
                continues_mask_embeds=mask_embeds,
                quant_mask_embeds=quant_mask_embeds,
                quant_codes=code,
            )

    # @can_return_tuple
    # def forward(
    #     self,
    #     pixel_values: Optional[torch.Tensor] = None,
    #     mask_embeds: Optional[torch.Tensor] = None,
    #     gt_masks: Optional[list[torch.Tensor]] = None,
    #     reconstruct_mask = True,
    #     freeze_codebook = False,
    # ) -> VQ_SAM2ModelOutput:
    #     """
    #     Args:
    #         mask_embeds: (batch_size, 1, hidden_dim)

    #     """
    #     batch_size = len(pixel_values)
    #     pixel_values = torch.stack([
    #         self.model.preprocess_image(pixel) for pixel in pixel_values
    #     ])
    #     sam2_states = self.model.get_sam2_embeddings(pixel_values, expand_size=1)

    #     mask_embeds = self.concate_mask_embeds(mask_embeds)
    #     quant_mask_embeds, quant_loss, code = self.quantizer(mask_embeds, freeze_codebook)
    #     if not reconstruct_mask:
    #         return VQ_SAM2ModelOutput(
    #             quant_codes=code,
    #         )

    #     quant_mask_embeds = self.deconcate_quant_embed(quant_mask_embeds)
    #     quant_mask_embeds = quant_mask_embeds.reshape(batch_size, self.num_mask_tokens, -1).contiguous()

    #     pred_masks = self.model.inject_language_embd(sam2_states, quant_mask_embeds, nf_nobj=(batch_size, 1))

    #     loss = quant_loss * self.config.vq_loss_weight
    #     if self.training and gt_masks is not None:
    #         gt_masks = [F.interpolate(gt_mask.unsqueeze(0).to(pred_masks.dtype), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks]
    #         gt_masks = torch.cat(gt_masks, dim=0)
    #         pred_masks = pred_masks.flatten(0, 1)

    #         if self.config.loss_sample_points:
    #             sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
    #             loss_dice = self.loss_dice(
    #                 sampled_pred_mask,
    #                 sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
    #             loss_mask = self.loss_mask(
    #                 sampled_pred_mask.reshape(-1),
    #                 sampled_gt_mask.reshape(-1),
    #                 avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
    #         else:
    #             loss_mask = self.loss_mask(pred_masks, gt_masks)
    #             loss_dice = self.loss_dice(pred_masks, gt_masks)
    #         loss += loss_mask + loss_dice
            
    #         return VQ_SAM2ModelOutput(
    #             loss=loss,
    #             loss_recon=loss_mask+loss_dice,
    #             loss_quant=quant_loss*self.config.vq_loss_weight,
    #             pred_masks=pred_masks,
    #             continues_mask_embeds=mask_embeds,
    #             quant_mask_embeds=quant_mask_embeds,
    #             quant_codes=code,
    #         )
    #     else:
    #         return VQ_SAM2ModelOutput(
    #             pred_masks=pred_masks,
    #             continues_mask_embeds=mask_embeds,
    #             quant_mask_embeds=quant_mask_embeds,
    #             quant_codes=code,
    #         )

