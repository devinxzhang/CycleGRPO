from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from projects.transformers.vq_sam2.configuration_vq_sam2 import VQ_SAM2Config
from omegaconf import OmegaConf

class Qwen2_5_VL_VQ_SAM2Config(Qwen2_5_VLConfig):

    def __init__(
        self,
        codebook_depth=4,
        codebook_size=1024,
        shared_codebook=False,
        codebook_latent_dim=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.codebook_depth = codebook_depth
        self.codebook_size = codebook_size
        self.shared_codebook = shared_codebook
        self.codebook_latent_dim = codebook_latent_dim
   
        
__all__ = ["Qwen2_5_VL_VQ_SAM2Config"]