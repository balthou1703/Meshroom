"""
Dense optical flow (OpticalFlow node), and its uses: frame interpolation of the time nodes
(Retime / TimeWarp "motion" filter) and shapes of the Roto moved along the flow.

A flow field is an array (h, w, 2) over the bbox of an image, in Nuke conventions: (u, v) in pixels,
u to the right, v up. The forward flow of frame f moves its pixels to frame f + 1:
image_f(x) ~ image_f+1(x + forward(x)); the backward flow moves them to frame f - 1. They are stored
in the "forward" and "backward" layers of an image (channels "u", "v"), as Nuke's motion vectors.

The flow is a pyramidal Lucas-Kanade (coarse to fine, a few iterations per level, a local window
that sets the smoothness), computed on a reduced copy of the images for large ones.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from .imageBuffer import BBox

FORWARD = "forward"
BACKWARD = "backward"
FLOW_CHANNELS = ("u", "v")


# Resampling

def _boxFilter(values: np.ndarray, radius: int) -> np.ndarray:
    """ Mean over (2 radius + 1)^2 windows of a 2D array (edges: mean of the part inside). """
    if radius <= 0:
        return values
    size = 2 * radius + 1
    padded = np.pad(values, ((radius + 1, radius), (radius + 1, radius)), mode="edge")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    total = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    return (total / float(size * size)).astype(np.float32)


def _downsample(values: np.ndarray, factor: int) -> np.ndarray:
    """ Average of blocks of factor x factor pixels of a (h, w) or (h, w, c) array (edges repeated). """
    if factor <= 1:
        return values
    h, w = values.shape[:2]
    height, width = -(-h // factor), -(-w // factor)
    pad = ((0, height * factor - h), (0, width * factor - w)) + ((0, 0),) * (values.ndim - 2)
    padded = np.pad(values, pad, mode="edge")
    shape = (height, factor, width, factor) + values.shape[2:]
    return padded.reshape(shape).mean(axis=(1, 3)).astype(np.float32)


def _resize(values: np.ndarray, height: int, width: int) -> np.ndarray:
    """ Bilinear resize of a (h, w) or (h, w, c) array, pixel centers aligned. """
    h, w = values.shape[:2]
    rows = np.clip((np.arange(height) + 0.5) * h / height - 0.5, 0, h - 1)
    columns = np.clip((np.arange(width) + 0.5) * w / width - 0.5, 0, w - 1)
    r0, c0 = np.floor(rows).astype(int), np.floor(columns).astype(int)
    r1, c1 = np.minimum(r0 + 1, h - 1), np.minimum(c0 + 1, w - 1)
    fr, fc = (rows - r0).astype(np.float32), (columns - c0).astype(np.float32)
    extra = (np.newaxis,) * (values.ndim - 2)
    fr = fr[(slice(None), np.newaxis) + extra]
    fc = fc[(np.newaxis, slice(None)) + extra]
    top = values[r0][:, c0] * (1 - fc) + values[r0][:, c1] * fc
    bottom = values[r1][:, c0] * (1 - fc) + values[r1][:, c1] * fc
    return (top * (1 - fr) + bottom * fr).astype(np.float32)


def sampleRows(values: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """
    Bilinear sampling of a (h, w) or (h, w, c) array at fractional (rows, columns) of pixel centers
    (0 is the center of the first pixel); clamped to the edges.
    """
    h, w = values.shape[:2]
    rows = np.clip(rows, 0, h - 1)
    columns = np.clip(columns, 0, w - 1)
    r0 = np.minimum(np.floor(rows).astype(np.int64), h - 2) if h > 1 else np.zeros(rows.shape, np.int64)
    c0 = np.minimum(np.floor(columns).astype(np.int64), w - 2) if w > 1 else np.zeros(columns.shape, np.int64)
    fr = (rows - r0).astype(np.float32)
    fc = (columns - c0).astype(np.float32)
    # Gathers on the flattened array (faster than 2D fancy indexing)
    flat = values.reshape(h * w, -1) if values.ndim == 3 else values.reshape(h * w)
    index = r0 * w + c0
    stepColumn = 1 if w > 1 else 0
    stepRow = w if h > 1 else 0
    if values.ndim == 3:
        fr, fc = fr[..., np.newaxis], fc[..., np.newaxis]
    top = np.take(flat, index, axis=0) * (1 - fc) + np.take(flat, index + stepColumn, axis=0) * fc
    bottom = np.take(flat, index + stepRow, axis=0) * (1 - fc) + np.take(flat, index + stepRow + stepColumn, axis=0) * fc
    return (top * (1 - fr) + bottom * fr).astype(np.float32)


def _shifted(values: np.ndarray, flowColumns: np.ndarray, flowRows: np.ndarray) -> np.ndarray:
    """ values sampled at (row + flowRows, column + flowColumns) of each pixel. """
    h, w = values.shape[:2]
    rows, columns = np.mgrid[0:h, 0:w].astype(np.float32)
    return sampleRows(values, rows + flowRows, columns + flowColumns)


# Optical flow

def _lucasKanade(a: np.ndarray, b: np.ndarray, window: int, iterations: int, cancel=None) -> np.ndarray:
    """
    Flow (h, w, 2) in pixel (column, row) units from the luminance `a` to `b` (same shape):
    a(p) ~ b(p + flow(p)), pyramidal Lucas-Kanade.
    """
    levels = [(a, b)]
    while min(levels[-1][0].shape) >= 32 and len(levels) < 8:
        levels.append((_downsample(levels[-1][0], 2), _downsample(levels[-1][1], 2)))
    radius = max(1, window // 2)
    flow = np.zeros(levels[-1][0].shape + (2,), dtype=np.float32)
    for index in range(len(levels) - 1, -1, -1):
        la, lb = levels[index]
        if flow.shape[:2] != la.shape:
            flow = _resize(flow, la.shape[0], la.shape[1]) * 2.0
        gradRows, gradColumns = np.gradient(la)
        ixx = _boxFilter(gradColumns * gradColumns, radius)
        iyy = _boxFilter(gradRows * gradRows, radius)
        ixy = _boxFilter(gradColumns * gradRows, radius)
        # Regularized: flat areas (no gradient) keep the flow of the coarser level
        det = ixx * iyy - ixy * ixy
        epsilon = 1e-4 + 1e-3 * float(np.mean(ixx + iyy))
        det = det + epsilon * (ixx + iyy) + epsilon * epsilon
        for _ in range(iterations):
            if cancel:
                cancel()
            warped = _shifted(lb, flow[:, :, 0], flow[:, :, 1])
            difference = warped - la
            ixt = _boxFilter(gradColumns * difference, radius)
            iyt = _boxFilter(gradRows * difference, radius)
            du = -((iyy + epsilon) * ixt - ixy * iyt) / det
            dv = -((ixx + epsilon) * iyt - ixy * ixt) / det
            flow[:, :, 0] += np.clip(du, -2.0, 2.0)
            flow[:, :, 1] += np.clip(dv, -2.0, 2.0)
        # A little smoothing between the levels
        flow[:, :, 0] = _boxFilter(flow[:, :, 0], 1)
        flow[:, :, 1] = _boxFilter(flow[:, :, 1], 1)
    return flow


def _luminance(image, bbox: BBox) -> np.ndarray:
    from .ops import alignToBBox
    rgba = alignToBBox(image.rgba(), image.bbox, bbox)
    return (0.2126 * rgba[:, :, 0] + 0.7152 * rgba[:, :, 1] + 0.0722 * rgba[:, :, 2]).astype(np.float32)


METHODS = ("lucasKanade", "raft")


def computeFlow(image, target, maxSize: int = 960, window: int = 11, iterations: int = 3,
                cancel: Optional[Callable[[], None]] = None, useGpu: bool = False,
                method: str = "lucasKanade") -> np.ndarray:
    """
    The flow (h, w, 2) over the bbox of `image` moving it onto `target` (Nuke units: u right, v up):
    image(x) ~ target(x + flow(x)). Computed on a copy reduced to `maxSize` pixels at most.

    Args:
        useGpu: compute on the GPU when there is one (meshroom/live/gpu.py), else on the CPU.
        method: "lucasKanade" (pyramidal Lucas-Kanade, `window` and `iterations`), or "raft" (the RAFT network,
                meshroom/live/neural: sharper motion edges and large motions; PyTorch and its weights needed, the
                Lucas-Kanade flow when they are missing).
    """
    from . import gpu
    bbox = image.bbox
    h, w = bbox.height, bbox.width
    if h == 0 or w == 0:
        return np.zeros((h, w, 2), np.float32)
    factor = max(1, int(math.ceil(max(h, w) / float(max(64, maxSize)))))
    if method == "raft":
        from .neural import NeuralUnavailable, warnUnavailable
        try:
            from .neural import raft
            return raft.computeFlow(image, target, factor, gpu.device(useGpu), cancel)
        except NeuralUnavailable as e:
            warnUnavailable("RAFT", e)
    a = _luminance(image, bbox)
    b = _luminance(target, bbox)
    a, b = _downsample(a, factor), _downsample(b, factor)
    if min(a.shape) < 2:
        return np.zeros((h, w, 2), np.float32)  # Too small to measure a motion

    def onGpu(device):
        from .gpuOps import lucasKanade
        return lucasKanade(a, b, window, iterations, device, cancel)
    small = gpu.run(useGpu, onGpu, lambda: _lucasKanade(a, b, window, iterations, cancel), "Optical flow")
    flow = _resize(small, h, w) * float(factor) if factor > 1 else small
    # Rows go down, v goes up
    return np.stack([flow[:, :, 0], -flow[:, :, 1]], axis=2).astype(np.float32)


def flowLayer(image, name: str) -> Optional[np.ndarray]:
    """ The flow stored in a layer ("forward" / "backward") of an image, (h, w, 2) over its bbox, or None. """
    if image is None or not image.hasLayer(name):
        return None
    channels = image.channels(name)
    pixels = image.layer(name)
    if "u" in channels and "v" in channels:
        return np.stack([pixels[:, :, channels.index("u")], pixels[:, :, channels.index("v")]], axis=2)
    return pixels[:, :, :2] if pixels.shape[2] >= 2 else None


# Uses of the flow

def interpolate(image0, image1, forward0: np.ndarray, backward1: np.ndarray, amount: float):
    """
    The image between image0 (amount 0) and image1 (amount 1) along the motion: image0 moved by
    amount x its forward flow, image1 moved back by (1 - amount) x its backward flow, mixed.
    `forward0` is over the bbox of image0, `backward1` over the bbox of image1. All the layers of
    image0 are moved; the result has the bbox of image0.
    """
    from collections import OrderedDict
    from .imageBuffer import RGBA, RGBA_CHANNELS
    from .ops import alignToBBox
    bbox = image0.bbox
    h, w = bbox.height, bbox.width
    rows, columns = np.mgrid[0:h, 0:w].astype(np.float32)
    backward = alignToBBox(backward1, image1.bbox, bbox)
    # Sample image0 at x - amount F0(x), image1 at x - (1 - amount) B1(x) (rows go down: v is negated)
    rows0 = rows + amount * forward0[:, :, 1]
    columns0 = columns - amount * forward0[:, :, 0]
    rows1 = rows + (1.0 - amount) * backward[:, :, 1]
    columns1 = columns - (1.0 - amount) * backward[:, :, 0]
    from .ops import parallelBands
    source0, source1 = image0.rgba(), alignToBBox(image1.rgba(), image1.bbox, bbox)
    rgba = np.empty((h, w, 4), np.float32)

    def band(start, end):
        first = sampleRows(source0, rows0[start:end], columns0[start:end])
        second = sampleRows(source1, rows1[start:end], columns1[start:end])
        rgba[start:end] = first * (1.0 - amount) + second * amount
    parallelBands(band, h)
    # The main layer becomes RGBA (an image without alpha gets the interpolated, zero, alpha)
    layers = OrderedDict([(RGBA, (RGBA_CHANNELS, rgba))])
    for name, channels, pixels in image0.layers():
        if name != RGBA:
            layers[name] = (channels, sampleRows(pixels, rows0, columns0))
    return image0.withPixels(layers, bbox)


def flowAt(flow: np.ndarray, bbox: BBox, x: float, y: float):
    """ The flow (u, v) at a point (Nuke coordinates) of a field over `bbox` (bilinear, clamped to the edges). """
    row = bbox.y1 - y - 0.5
    column = x - bbox.x0 - 0.5
    value = sampleRows(flow, np.array([row], np.float32), np.array([column], np.float32))[0]
    return float(value[0]), float(value[1])
