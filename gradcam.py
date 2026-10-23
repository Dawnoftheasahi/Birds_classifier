"""Grad-CAM (Selvaraju et al., 2017) without extra dependencies.

The class score is differentiated with respect to the last convolutional feature map; each
channel is weighted by its average gradient and the weighted sum (after ReLU) is the heat map.
"""
import numpy as np
import torch
import torch.nn.functional as F

CAM_RGB = np.array([0xEB, 0x68, 0x34]) / 255.0      # orange tint used in the report figures too


def grad_cam(model, x):
    """Return (cam [B, h, w] in [0, 1], logits [B, C]) for the predicted class of each image."""
    with torch.no_grad():
        feats = model.forward_features(x)                          # B x C x h x w
    feats = feats.detach().float().requires_grad_(True)
    with torch.enable_grad():
        logits = model.forward_head(feats)
        cls = logits.argmax(1)
        logits.gather(1, cls[:, None]).sum().backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)            # channel importance
    cam = F.relu((weights * feats.detach()).sum(1))
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam, logits.detach()


def overlay_cam(img, cam):
    """Spotlight overlay on a float RGB image (H, W, 3): dim ignored regions, tint important ones."""
    cam = F.interpolate(cam[None, None], size=img.shape[:2], mode='bilinear', align_corners=False)[0, 0].numpy()
    cam = np.clip(cam, 0, 1)[..., None]
    out = img * (0.25 + 0.75 * cam) * (1 - 0.35 * cam) + CAM_RGB * 0.35 * cam
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
