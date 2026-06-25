from .perceptionlm import PerceptionLM_TokenMask
from .qwen25vl import QWEN25VL_VQSAM2Model
from .qwen3vl import QWEN3VL_VQSAM2Model
from .processing_perception_lm import PerceptionLMProcessor

import numpy as np
from torchvision.transforms.functional import resize, to_pil_image
class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))