import numpy as np
from torchvision.transforms.functional import resize, to_pil_image

from .vq_sam2 import VQ_SAM2Model
from .qwen25vl_vq_sam2 import QWEN25VL_VQSAM2Model
from .qwen25vl_tokenmask import QWEN25VL_TokenMask
from .sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))
    
