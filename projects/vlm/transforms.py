import numpy as np
from torchvision.transforms.functional import to_pil_image


class DirectResize:
    """Resize an image to a fixed square resolution, ignoring aspect ratio.

    Matches the preprocessing the SAM2 mask tokenizer was trained with.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """Expects a numpy array with shape HxWxC in uint8 format."""
        img = to_pil_image(image, mode="RGB")
        return np.array(img.resize((self.target_length, self.target_length)))
