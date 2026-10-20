"""
Neural networks used by the compositing nodes, with PyTorch (optional):

- RAFT (raft.py): dense optical flow (OpticalFlow, Retime / TimeWarp "motion", Roto flow, Resection geometry tracking).
- BootsTAPIR (TAPIR) and CoTracker (pointTracking.py): many points tracked together through a sequence, occlusions
  included (Tracker, pins of the Resection).
- LightGlue with ALIKED features (matching.py): feature matching between two images, even far apart (CameraTracker
  autotrack, Resection "Align").
- MegaPose (megapose.py): the pose of a known object in an image from its mesh (Resection "Align").

Their weights are read from a local folder (MESHROOM_MODELS_PATH), filled once by `python -m
meshroom.live.neural.download` (models.py lists them, with their licenses). When PyTorch, the package of a network or
its weights are missing, NeuralUnavailable is raised and the node uses its classic method (logged once).
"""
from __future__ import annotations

import logging
import threading

_warned = set()
_lock = threading.Lock()


class NeuralUnavailable(RuntimeError):
    """ A network cannot be used: PyTorch, its package or its weights are missing (the message says which). """


def warnUnavailable(what: str, error: Exception):
    """ Log once that a network is not available and the classic method is used instead. """
    with _lock:
        key = (what, str(error))
        if key in _warned:
            return
        _warned.add(key)
    logging.warning(f"[Neural] {what} is not available, the classic method is used: {error}")


def requireTorch():
    """ The torch module, or NeuralUnavailable. """
    try:
        import torch
        return torch
    except Exception as e:
        raise NeuralUnavailable(f"PyTorch is not installed ({e})") from e


def torchDevice(device):
    """ The torch device to run a network on: `device` (the GPU, see meshroom/live/gpu.py), else the CPU. """
    torch = requireTorch()
    return device if device is not None else torch.device("cpu")


def displayRGB(image, bbox=None):
    """
    The RGB of an image over `bbox` (its bbox by default) as the networks expect it: display referred values in [0, 1]
    (the linear values clipped and encoded with a 2.2 gamma), rows from the top. (h, w, 3) float32.
    """
    import numpy as np
    from ..ops import alignToBBox
    bbox = bbox or image.bbox
    rgb = alignToBBox(image.rgba(), image.bbox, bbox)[:, :, :3]
    return np.power(np.clip(rgb, 0.0, 1.0), 1.0 / 2.2).astype(np.float32)
