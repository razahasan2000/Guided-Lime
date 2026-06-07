"""
gradcam_utils.py
================================
Grad-CAM implementation for Swin Transformer (swin_t).

Swin Transformer has no standard convolutional feature maps.
We hook the output of the last Swin stage (model.layers[-1])
which produces patch tokens of shape (B, H*W, C).
These are reshaped to (H, W, C) and the class activation map is
computed as the weighted sum: CAM = ReLU(sum_c(alpha_c * A_c))

Usage:
    from gradcam_utils import SwinGradCAM
    gcam = SwinGradCAM(model, device)
    cam  = gcam.generate(image_tensor, class_idx)   # returns H x W numpy array
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms


class SwinGradCAM:
    """
    Grad-CAM for Swin Transformer via hooks on model.layers[-1].
    """

    def __init__(self, model, device):
        self.model  = model
        self.device = device
        self._features = None
        self._gradients = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        if hasattr(self.model, 'layers'):
            target_layer = self.model.layers[-1]
        elif hasattr(self.model, 'features'):
            target_layer = self.model.features[-1]
        else:
            raise AttributeError("Model has neither 'layers' nor 'features' attributes.")

        def forward_hook(module, input, output):
            # output shape: (B, num_patches, C)  or (B, H, W, C) depending on version
            self._features = output

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0]

        self._hook_handles.append(
            target_layer.register_forward_hook(forward_hook)
        )
        self._hook_handles.append(
            target_layer.register_full_backward_hook(backward_hook)
        )

    def _to_spatial(self, tensor):
        """
        Convert patch token tensor to spatial (H, W, C).
        Handles both (B, num_patches, C) and (B, H, W, C) shapes.
        """
        if tensor.dim() == 4:
            # (B, H, W, C) - newer torchvision
            return tensor.squeeze(0)  # (H, W, C)
        elif tensor.dim() == 3:
            # (B, num_patches, C)
            B, N, C = tensor.shape
            # Assume square spatial layout
            H = W = int(N ** 0.5)
            if H * W != N:
                # Non-square: use 1D fallback
                return tensor.squeeze(0).unsqueeze(0)  # (1, N, C) - limited
            return tensor.squeeze(0).reshape(H, W, C)
        return tensor

    def generate(self, image_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            image_tensor: (1, 3, H, W) tensor, already on correct device
            class_idx: Target class index. If None, uses predicted class.

        Returns:
            cam_224: numpy array of shape (224, 224), values in [0, 1]
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device).requires_grad_(False)
        image_tensor.requires_grad = False

        # Forward pass
        output = self.model(image_tensor)
        probs  = torch.softmax(output, dim=1)

        if class_idx is None:
            class_idx = probs.argmax(dim=1).item()

        # Zero grads, backward on target class score
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # features: (H, W, C), gradients: same shape
        features  = self._to_spatial(self._features.detach())   # (H, W, C)
        gradients = self._to_spatial(self._gradients.detach())  # (H, W, C)

        # Pool gradients over spatial dims → channel weights (C,)
        if features.dim() == 3:
            alpha = gradients.mean(dim=(0, 1))                  # (C,)
            cam   = torch.relu((features * alpha).sum(dim=-1))  # (H, W)
        else:
            # Fallback for edge shapes
            cam = gradients.abs().mean(dim=-1).squeeze()

        cam = cam.cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        # Upsample to 224×224
        from PIL import Image as PILImage
        cam_pil = PILImage.fromarray((cam * 255).astype(np.uint8))
        cam_224 = np.array(cam_pil.resize((224, 224), PILImage.BILINEAR)) / 255.0

        return cam_224, class_idx, probs.detach().cpu().numpy()[0]

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def load_image(path):
    """Load PIL image and return (pil_image, tensor (1,3,224,224))."""
    pil = Image.open(path).convert('RGB')
    tensor = get_transform()(pil).unsqueeze(0)
    return pil, tensor
