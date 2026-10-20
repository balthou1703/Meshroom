"""
RAFT optical flow (torchvision's raft_large, weights C_T_SKHT_V2): sharper motion edges and larger motions than the
Lucas-Kanade flow of flow.py. Same convention as flow.computeFlow: (h, w, 2) over the bbox of the first image, u to
the right, v up, image(x) ~ target(x + flow(x)).
"""
from __future__ import annotations

import numpy as np

from . import NeuralUnavailable, displayRGB, requireTorch, torchDevice
from .models import cachedModel, importPackage, loadError, useTorchHub

# RAFT works at 1/8 of the resolution with 4 correlation levels: sizes multiple of 8, at least 128 pixels
_MULTIPLE = 8
_MINIMUM = 128


def _model(device):
    torch = requireTorch()
    importPackage("raft")

    def load():
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        useTorchHub()
        try:
            return raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False).eval().to(device)
        except Exception as e:
            raise loadError("raft", e) from e
    with torch.no_grad():
        return cachedModel(("raft", str(device)), load)


def _padded(tensor, height: int, width: int):
    """ A (1, 3, h, w) tensor padded (edges repeated) to `height` x `width`. """
    import torch.nn.functional as F
    h, w = tensor.shape[-2:]
    return F.pad(tensor, (0, width - w, 0, height - h), mode="replicate")


def computeFlow(image, target, factor: int, device, cancel=None) -> np.ndarray:
    """
    The flow (h, w, 2) over the bbox of `image` moving it onto `target` (Nuke units), computed by RAFT on copies reduced
    by `factor`, on `device` (None: the CPU, much slower). Raises NeuralUnavailable when RAFT cannot be used.
    """
    torch = requireTorch()
    from ..flow import _downsample, _resize
    device = torchDevice(device)
    bbox = image.bbox
    h, w = bbox.height, bbox.width
    a = _downsample(displayRGB(image, bbox), factor)
    b = _downsample(displayRGB(target, bbox), factor)
    sh, sw = a.shape[:2]
    if min(sh, sw) < 2:
        return np.zeros((h, w, 2), np.float32)
    height = max(_MINIMUM, -(-sh // _MULTIPLE) * _MULTIPLE)
    width = max(_MINIMUM, -(-sw // _MULTIPLE) * _MULTIPLE)
    model = _model(device)
    if cancel:
        cancel()
    try:
        with torch.no_grad():
            # RAFT takes images in [-1, 1]
            first = _padded(torch.from_numpy(a).permute(2, 0, 1)[None].to(device) * 2.0 - 1.0, height, width)
            second = _padded(torch.from_numpy(b).permute(2, 0, 1)[None].to(device) * 2.0 - 1.0, height, width)
            flows = model(first, second, num_flow_updates=12)
            # The last refinement, (1, 2, H, W): x to the right, y down, in pixels
            small = flows[-1][0, :, :sh, :sw].permute(1, 2, 0).to("cpu").numpy().astype(np.float32)
    except NeuralUnavailable:
        raise
    except RuntimeError as e:
        # E.g. out of memory on the GPU: the classic flow is used
        raise NeuralUnavailable(f"RAFT failed ({e}); lower the resolution of the flow") from e
    flow = _resize(small, h, w) * float(factor) if factor > 1 else small
    # Rows go down, v goes up
    return np.stack([flow[:, :, 0], -flow[:, :, 1]], axis=2).astype(np.float32)
