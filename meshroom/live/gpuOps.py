"""
GPU versions (PyTorch) of the heavy operations of the compositing: blur, erode / dilate, median, FFT convolution
(ZDefocus), pyramidal Lucas-Kanade optical flow, and the pattern matching of the trackers (many patterns at once).

Each function takes numpy arrays and a torch device and returns numpy arrays, computing exactly what its CPU version
does (meshroom/live/ops.py, FilterNodes.py, defocus.py, flow.py, tracking.py), within floating point precision. They
are called through gpu.run, which falls back on the CPU version. They also run on the "cpu" torch device (the tests
compare them with the CPU versions wherever PyTorch is installed).
"""
from __future__ import annotations

import math
import numpy as np


def _torch():
    import torch
    import torch.nn.functional as F
    return torch, F


def _tensor(values: np.ndarray, device):
    torch, _ = _torch()
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).to(device)


def _numpy(tensor) -> np.ndarray:
    return tensor.detach().to("cpu").numpy().astype(np.float32, copy=False)


# Blur (ops.blur): separable gaussian, three box blurs for the large sizes; the result grows by the blur radius

def _blurAxis(x, size: float):
    """ Blur of a (n, 1, length) tensor along its last axis, extended by the blur radius on both sides. """
    from . import ops
    _, F = _torch()
    sigma = ops.blurSigma(size)
    if sigma < 1e-3:
        return x
    if sigma < 2.0:
        kernel = _tensor(ops.gaussianKernel(size), x.device).view(1, 1, -1)
        radius = kernel.shape[-1] // 2
        if radius == 0:
            return x
        # Full convolution: padded by twice the radius, as ops._kernelAxis
        return F.conv1d(F.pad(x, (2 * radius, 2 * radius)), kernel)
    for radius in ops.boxRadii(sigma):
        if radius <= 0:
            continue
        # Black around (as ops._boxBlurAxis): padded by the radius, then a box of 2r + 1 counting the padding
        x = F.avg_pool1d(F.pad(x, (radius, radius)), 2 * radius + 1, stride=1, padding=radius, count_include_pad=True)
    return x


def _alongAxis(tensor, axis: int, function):
    """ Apply `function` to the rows of a (c, h, w) tensor along `axis` (1: vertical, 2: horizontal). """
    moved = tensor.movedim(axis, -1)
    shape = moved.shape
    result = function(moved.reshape(-1, 1, shape[-1]))
    return result.reshape(shape[:-1] + (result.shape[-1],)).movedim(-1, axis)


def blur(pixels: np.ndarray, sizeX: float, sizeY: float, device) -> np.ndarray:
    """ ops.blur on the GPU: (h, w, c) -> (h + 2 ry, w + 2 rx, c). """
    tensor = _tensor(pixels, device).permute(2, 0, 1)
    tensor = _alongAxis(tensor, 2, lambda x: _blurAxis(x, sizeX))
    tensor = _alongAxis(tensor, 1, lambda x: _blurAxis(x, sizeY))
    return _numpy(tensor.permute(1, 2, 0).contiguous())


# Erode / dilate (FilterNodes.Erode): minimum / maximum over a square, black outside

def extremum(channels: np.ndarray, radius: int, useMax: bool, device) -> np.ndarray:
    """ Minimum (or maximum) of each channel of a (h, w, c) array over a square of 2r + 1 pixels (black outside). """
    _, F = _torch()
    if radius <= 0:
        return channels

    def axis(x):
        x = F.pad(x, (radius, radius), value=0.0)
        if useMax:
            return F.max_pool1d(x, 2 * radius + 1, stride=1)
        return -F.max_pool1d(-x, 2 * radius + 1, stride=1)
    tensor = _tensor(channels, device).permute(2, 0, 1)
    tensor = _alongAxis(tensor, 2, axis)
    tensor = _alongAxis(tensor, 1, axis)
    return _numpy(tensor.permute(1, 2, 0).contiguous())


# Median (FilterNodes.Median): median of each channel over a square, black outside

def median(pixels: np.ndarray, width: int, device, maxBytes: int = 256 * 1024 * 1024) -> np.ndarray:
    """ Median of each channel of a (h, w, c) array over a square of `width` (odd) pixels, black outside, same size. """
    torch, F = _torch()
    radius = width // 2
    h, w, c = pixels.shape
    if radius <= 0:
        return pixels
    tensor = F.pad(_tensor(pixels, device).permute(2, 0, 1).unsqueeze(0), (radius, radius, radius, radius)).squeeze(0)
    result = torch.empty((c, h, w), dtype=torch.float32, device=device)
    # In bands of rows: the windows of a band take c x rows x w x width^2 values
    rows = max(1, int(maxBytes // max(1, c * w * width * width * 4)))
    for start in range(0, h, rows):
        end = min(h, start + rows)
        windows = tensor[:, start:end + 2 * radius].unfold(1, width, 1).unfold(2, width, 1)
        result[:, start:end] = windows.reshape(c, end - start, w, width * width).median(dim=-1).values
    return _numpy(result.permute(1, 2, 0).contiguous())


# FFT convolution (defocus.fftConvolve)

def fftConvolve(values: np.ndarray, kernel: np.ndarray, device) -> np.ndarray:
    """ Convolution of each channel of a (h, w, c) array by a 2D kernel (odd sizes), black outside, same size. """
    torch, _ = _torch()
    from .defocus import _fastLength
    h, w, _ = values.shape
    kh, kw = kernel.shape
    if kh == 1 and kw == 1:
        return values * kernel[0, 0]
    fh, fw = _fastLength(h + kh - 1), _fastLength(w + kw - 1)
    data = _tensor(values, device).permute(2, 0, 1)
    kernelSpectrum = torch.fft.rfft2(_tensor(kernel, device), s=(fh, fw))
    spectrum = torch.fft.rfft2(data, s=(fh, fw))
    result = torch.fft.irfft2(spectrum * kernelSpectrum.unsqueeze(0), s=(fh, fw))
    oy, ox = kh // 2, kw // 2
    return _numpy(result[:, oy:oy + h, ox:ox + w].permute(1, 2, 0).contiguous())


# Optical flow (flow._lucasKanade)

def _gradient(x, dim: int):
    """ numpy.gradient along a dimension: central differences inside, one-sided ones on the edges. """
    torch, _ = _torch()
    n = x.shape[dim]
    if n < 2:
        return torch.zeros_like(x)
    inner = (x.narrow(dim, 2, n - 2) - x.narrow(dim, 0, n - 2)) * 0.5
    first = x.narrow(dim, 1, 1) - x.narrow(dim, 0, 1)
    last = x.narrow(dim, n - 1, 1) - x.narrow(dim, n - 2, 1)
    return torch.cat([first, inner, last], dim=dim)


def _boxFilter(x, radius: int):
    """ flow._boxFilter: mean over (2r + 1)^2 windows of a (h, w) tensor, the edges repeated. """
    _, F = _torch()
    if radius <= 0:
        return x
    padded = F.pad(x[None, None], (radius, radius, radius, radius), mode="replicate")
    return F.avg_pool2d(padded, 2 * radius + 1, stride=1)[0, 0]


def _downsample(x):
    """ flow._downsample by 2: average of blocks of 2 x 2 pixels of a (h, w) tensor (edges repeated). """
    _, F = _torch()
    h, w = x.shape
    padded = F.pad(x[None, None], (0, w % 2, 0, h % 2), mode="replicate")
    return F.avg_pool2d(padded, 2)[0, 0]


def _resizeFlow(flow, height: int, width: int):
    """ flow._resize of a (h, w, 2) tensor (bilinear, pixel centers aligned). """
    _, F = _torch()
    resized = F.interpolate(flow.permute(2, 0, 1)[None], size=(height, width), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0)


def _shifted(values, flowColumns, flowRows):
    """ flow._shifted: a (h, w) tensor sampled (bilinear, clamped to the edges) at each pixel moved by the flow. """
    torch, F = _torch()
    h, w = values.shape
    rows = torch.arange(h, dtype=torch.float32, device=values.device)[:, None] + flowRows
    columns = torch.arange(w, dtype=torch.float32, device=values.device)[None, :] + flowColumns
    grid = torch.stack([columns * (2.0 / max(1, w - 1)) - 1.0, rows * (2.0 / max(1, h - 1)) - 1.0], dim=-1)
    return F.grid_sample(values[None, None], grid[None], mode="bilinear", padding_mode="border", align_corners=True)[0, 0]


def lucasKanade(a: np.ndarray, b: np.ndarray, window: int, iterations: int, device, cancel=None) -> np.ndarray:
    """ flow._lucasKanade on the GPU: the flow (h, w, 2) in pixels (column, row) from luminance `a` to `b`. """
    torch, _ = _torch()
    levels = [(_tensor(a, device), _tensor(b, device))]
    while min(levels[-1][0].shape) >= 32 and len(levels) < 8:
        levels.append((_downsample(levels[-1][0]), _downsample(levels[-1][1])))
    radius = max(1, window // 2)
    flow = torch.zeros(tuple(levels[-1][0].shape) + (2,), dtype=torch.float32, device=device)
    for index in range(len(levels) - 1, -1, -1):
        la, lb = levels[index]
        if tuple(flow.shape[:2]) != tuple(la.shape):
            flow = _resizeFlow(flow, la.shape[0], la.shape[1]) * 2.0
        gradRows, gradColumns = _gradient(la, 0), _gradient(la, 1)
        ixx = _boxFilter(gradColumns * gradColumns, radius)
        iyy = _boxFilter(gradRows * gradRows, radius)
        ixy = _boxFilter(gradColumns * gradRows, radius)
        # Regularized: flat areas (no gradient) keep the flow of the coarser level
        epsilon = 1e-4 + 1e-3 * float((ixx + iyy).mean())
        det = ixx * iyy - ixy * ixy + epsilon * (ixx + iyy) + epsilon * epsilon
        for _ in range(iterations):
            if cancel:
                cancel()
            difference = _shifted(lb, flow[:, :, 0], flow[:, :, 1]) - la
            ixt = _boxFilter(gradColumns * difference, radius)
            iyt = _boxFilter(gradRows * difference, radius)
            du = -((iyy + epsilon) * ixt - ixy * iyt) / det
            dv = -((ixx + epsilon) * iyt - ixy * ixt) / det
            flow = flow + torch.stack([du.clamp(-2.0, 2.0), dv.clamp(-2.0, 2.0)], dim=-1)
        # A little smoothing between the levels
        flow = torch.stack([_boxFilter(flow[:, :, 0], 1), _boxFilter(flow[:, :, 1], 1)], dim=-1)
    return _numpy(flow.contiguous())


# Pattern matching (tracking.matchPattern), many patterns at once

def matchPatterns(patterns: np.ndarray, lum: np.ndarray, bbox, centers: np.ndarray, searchSize: int,
                  device) -> np.ndarray:
    """
    tracking.matchPattern for n patterns (n, size, size) at once, each searched in the area of `lum` centered on its
    center (n, 2) (Nuke coordinates).

    Returns:
        (n, 3): the position (x, y) of each pattern center and its normalized cross-correlation; NaN where the search
        area leaves the image.
    """
    torch, F = _torch()
    count, size = patterns.shape[0], patterns.shape[1]
    searchSize = max(searchSize, size + 2)
    half = searchSize // 2
    result = np.full((count, 3), np.nan)
    if count == 0:
        return result
    # The search areas (tracking._window), gathered on the CPU
    columns = np.floor(centers[:, 0]).astype(np.int64) - bbox.x0
    rows = bbox.y1 - 1 - np.floor(centers[:, 1]).astype(np.int64)
    tops, lefts = rows - half, columns - half
    valid = (tops >= 0) & (lefts >= 0) & (tops + searchSize <= lum.shape[0]) & (lefts + searchSize <= lum.shape[1])
    indices = np.flatnonzero(valid)
    if not len(indices):
        return result
    areas = np.stack([lum[tops[i]:tops[i] + searchSize, lefts[i]:lefts[i] + searchSize] for i in indices]).astype(np.float64)
    # Centered (the normalized correlation does not change): the variances of bright, flat areas stay precise in float32
    areas -= areas.mean(axis=(1, 2), keepdims=True)
    windows = _tensor(areas, device).unsqueeze(0)                       # (1, n, S, S)
    p = _tensor(patterns[indices], device)
    p = p - p.mean(dim=(1, 2), keepdim=True)
    pNorm = torch.sqrt((p * p).sum(dim=(1, 2))) + 1e-6                    # (n,)
    # Correlation with the centered pattern (its sum is 0: the window mean does not matter)
    numerator = F.conv2d(windows, p.unsqueeze(1), groups=len(indices))[0]   # (n, S - size + 1, S - size + 1)
    means = F.avg_pool2d(windows, size, stride=1)[0]
    squares = F.avg_pool2d(windows * windows, size, stride=1)[0]
    norms = torch.sqrt(torch.clamp(squares - means * means, min=0.0) * float(size * size)) + 1e-6
    scores = numerator / (norms * pNorm[:, None, None])
    scoreValues = _numpy(scores)
    span = scoreValues.shape[1]
    best = scoreValues.reshape(len(indices), -1).argmax(axis=1)
    from .tracking import _subpixel
    center = (searchSize - size) // 2
    for k, i in enumerate(indices):
        row, column = divmod(int(best[k]), span)
        dy, dx = _subpixel(scoreValues[k], row, column)
        result[i] = (math.floor(centers[i, 0]) + 0.5 + column + dx - center,
                     math.floor(centers[i, 1]) + 0.5 - (row + dy - center), float(scoreValues[k, row, column]))
    return result
