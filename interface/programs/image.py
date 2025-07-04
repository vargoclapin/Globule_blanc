import numpy as np
import torch

from PIL import Image

class Image_Normalizer_VGG16():
    """Class used to normalise an image to VGG16
    input standards."""

    def __init__(self, file):
        self._image = Image.open(file).convert('RGB')

    def normalise(self):
        
        image_np = np.array(self._image).astype(np.float32) / 255.0  # Normalize to [0,1]
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        # Normalize using ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor