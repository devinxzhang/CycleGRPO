from transformers import PretrainedConfig
from hydra import compose
from omegaconf import OmegaConf


class SAM2Config(PretrainedConfig):
    model_type = "sam2"
    base_config_key = "sam2_config"

    def __init__(
        self,
        cfg_path: str = "sam2.1_hiera_l.yaml",
        ckpt_path: str = "sam2.1_hiera_large.pt",
        hydra_overrides_extra = None,
        apply_postprocessing = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path

        if hydra_overrides_extra is None:
            hydra_overrides_extra = []
        hydra_overrides = [
            ## Extension: LLM prompt
            "++model._target_=projects.transformers.vq_sam2.sam2_base.SAM2Base",
        ]

        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                # dynamically fall back to multi-mask if the single mask is not stable
                # "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                # "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                # "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
                # "++model.binarize_mask_from_pts_for_mem_enc=true",
                # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
                # "++model.fill_hole_area=8",
            ]
        hydra_overrides.extend(hydra_overrides_extra)
        
        # Read config and init model
        cfg = compose(config_name=cfg_path, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        self.cfg = cfg
    
    def to_dict(self):
        """重写 to_dict 方法以处理 OmegaConf 对象"""
        output = super().to_dict()
        
        # 处理 cfg 中的 OmegaConf 对象
        if hasattr(self, 'cfg') and self.cfg is not None:
            if hasattr(self.cfg, '_content') and hasattr(self.cfg, 'to_container'):
                output['cfg'] = OmegaConf.to_container(self.cfg, resolve=True)
            else:
                output['cfg'] = self.cfg
        
        return output


class VQ_SAM2Config(PretrainedConfig):
    model_type = "vq_sam2"
    sub_configs = {
        "sam2_config": SAM2Config,
    }

    def __init__(
        self,
        sam2_config: SAM2Config = None,
        codebook_size: int = 1024,
        codebook_depth: int = 4,
        shared_codebook: bool = False,
        latent_dim: int = 256,
        # mask loss
        loss_sample_points: bool = False,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        # vq loss
        vq_loss_weight: float = 0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sam2_config = sam2_config
        self.codebook_size = codebook_size
        self.codebook_depth = codebook_depth
        self.shared_codebook = shared_codebook
        self.latent_dim = latent_dim

        # mask loss
        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        # vq loss
        self.vq_loss_weight = vq_loss_weight


    
    def to_dict(self):
        """重写 to_dict 方法以处理 OmegaConf 对象"""
        output = super().to_dict()
        
        # 处理 sam2_config 中的 OmegaConf 对象
        if hasattr(self, 'sam2_config') and self.sam2_config is not None:
            sam2_dict = {}
            for key, value in self.sam2_config.__dict__.items():
                if hasattr(value, '_content') and hasattr(value, 'to_container'):
                    # 这是 OmegaConf 对象
                    sam2_dict[key] = OmegaConf.to_container(value, resolve=True)
                else:
                    sam2_dict[key] = value
            output['sam2_config'] = sam2_dict
        
        return output