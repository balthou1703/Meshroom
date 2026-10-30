"""
Depth of field (ZDefocus node): a blur whose size depends on the depth of each pixel, with the shape
of a lens aperture (bokeh).

The image is cut into depth slices by the signed size of the blur of its pixels (negative: in front
of the focal plane, positive: behind). Each slice (premultiplied, with its coverage) is convolved with
the bokeh kernel of its size, and the slices are composited back to front (the far ones first), so
that the blurred foreground spreads over what is behind it. Large kernels are applied on a reduced
copy of the slice (the blur hides the lost resolution), which keeps the cost almost independent of
the blur size.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

# Kernel radius (in pixels) above which a slice is convolved on a reduced copy
_REDUCED_RADIUS = 4.0


# Bokeh kernels

def bokehKernel(radius: float, blades: int = 0, rotation: float = 0.0, aspect: float = 1.0,
                filterImage: Optional[np.ndarray] = None) -> np.ndarray:
    """
    The normalized (sum 1) kernel of a defocus blur of `radius` pixels: a disc (blades < 3), a regular
    polygon with `blades` sides rotated by `rotation` degrees, or the shape of `filterImage` (h, w)
    (its luminance). `aspect` > 1 stretches it horizontally (anamorphic lens). Antialiased edges.
    """
    if radius < 0.5:
        return np.ones((1, 1), dtype=np.float32)
    aspect = max(0.1, float(aspect))
    rx, ry = radius * aspect, radius
    halfX, halfY = int(math.ceil(rx)) + 1, int(math.ceil(ry)) + 1
    if filterImage is not None and filterImage.size > 0:
        shape = _resizeCentered(filterImage[:, :, np.newaxis].astype(np.float32), 2 * halfY + 1, 2 * halfX + 1)[:, :, 0]
        kernel = np.maximum(shape, 0.0)
    else:
        ys, xs = np.mgrid[-halfY:halfY + 1, -halfX:halfX + 1].astype(np.float64)
        # Normalized coordinates (unit shape), image rows going down: y up
        u, v = xs / rx, -ys / ry
        if blades >= 3:
            angle = math.radians(rotation)
            # Distance to the center in units of the apothem of a polygon whose vertices are at 1
            edge = np.full(u.shape, -np.inf)
            for i in range(blades):
                normal = angle + (i + 0.5) * 2.0 * math.pi / blades
                edge = np.maximum(edge, u * math.cos(normal) + v * math.sin(normal))
            distance = edge / math.cos(math.pi / blades)
        else:
            distance = np.hypot(u, v)
        # Antialiased edge: about one pixel wide
        kernel = np.clip((1.0 - distance) * radius + 0.5, 0.0, 1.0)
    total = kernel.sum()
    if total <= 0:
        return np.ones((1, 1), dtype=np.float32)
    return (kernel / total).astype(np.float32)


# Resampling

def _resizeCentered(values: np.ndarray, height: int, width: int) -> np.ndarray:
    """ Bilinear resize of a (h, w, c) array, pixel centers aligned (no shift). """
    h, w = values.shape[:2]
    rows = np.clip((np.arange(height) + 0.5) * h / height - 0.5, 0, h - 1)
    columns = np.clip((np.arange(width) + 0.5) * w / width - 0.5, 0, w - 1)
    r0, c0 = np.floor(rows).astype(int), np.floor(columns).astype(int)
    r1, c1 = np.minimum(r0 + 1, h - 1), np.minimum(c0 + 1, w - 1)
    fr = (rows - r0).astype(np.float32)[:, np.newaxis, np.newaxis]
    fc = (columns - c0).astype(np.float32)[np.newaxis, :, np.newaxis]
    values = np.asarray(values, dtype=np.float32)
    # Separable: the columns first (on the few rows of the small image), then the rows
    across = values[:, c0] * (1 - fc) + values[:, c1] * fc
    return across[r0] * (1 - fr) + across[r1] * fr


def _downsample(values: np.ndarray, factor: int) -> np.ndarray:
    """ Average of blocks of factor x factor pixels (the last blocks padded with the edge). """
    h, w, c = values.shape
    height, width = -(-h // factor), -(-w // factor)
    padded = np.pad(values, ((0, height * factor - h), (0, width * factor - w), (0, 0)), mode="edge")
    return padded.reshape(height, factor, width, factor, c).mean(axis=(1, 3), dtype=np.float32)


# Convolution

def _fastLength(n: int) -> int:
    """ The smallest length >= n whose factors are 2, 3 and 5 (fast FFT). """
    best = 1 << max(0, (n - 1).bit_length())
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p = p35
            while p < n:
                p *= 2
            best = min(best, p)
            p35 *= 3
        p5 *= 5
    return best


def fftConvolve(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """ Convolution of each channel of a (h, w, c) array by a 2D kernel (odd sizes), black outside, same size. """
    h, w, _ = values.shape
    kh, kw = kernel.shape
    if kh == 1 and kw == 1:
        return values * kernel[0, 0]
    fh, fw = _fastLength(h + kh - 1), _fastLength(w + kw - 1)
    kernelSpectrum = np.fft.rfft2(kernel, s=(fh, fw))
    spectrum = np.fft.rfft2(values, s=(fh, fw), axes=(0, 1))
    result = np.fft.irfft2(spectrum * kernelSpectrum[:, :, np.newaxis], s=(fh, fw), axes=(0, 1))
    oy, ox = kh // 2, kw // 2
    return result[oy:oy + h, ox:ox + w].astype(np.float32)


def _blurSlice(data: np.ndarray, radius: float, kernelFor: Callable[[float], np.ndarray],
               convolve: Callable = fftConvolve) -> np.ndarray:
    """ Convolve a slice (h, w, c) by the kernel of `radius`, on a reduced copy for the large ones. """
    if radius < 0.5:
        return data
    factor = max(1, int(radius / _REDUCED_RADIUS))
    if factor == 1:
        return convolve(data, kernelFor(radius))
    h, w = data.shape[:2]
    small = convolve(_downsample(data, factor), kernelFor(radius / factor))
    return _resizeCentered(small, -(-h // factor) * factor, -(-w // factor) * factor)[:h, :w]


# Depth of field

def defocus(rgba: np.ndarray, signedRadius: np.ndarray, maxRadius: float, slices: int,
            kernelFor: Callable[[float], np.ndarray], cancel: Optional[Callable[[], None]] = None,
            convolve: Callable = fftConvolve) -> np.ndarray:
    """
    Depth of field of a premultiplied (h, w, 4) image: `signedRadius` (h, w) is the blur radius of each
    pixel in pixels, negative in front of the focal plane, positive behind; `slices` depth slices on each
    side; `kernelFor(radius)` gives the bokeh kernel; `convolve(values, kernel)` convolves a slice (fftConvolve, or
    its GPU version). Returns the (h, w, 4) result.
    """
    h, w = signedRadius.shape
    # The slices span the blur sizes actually present (not just the allowed maximum)
    maxRadius = min(float(maxRadius), float(np.abs(signedRadius).max()) if signedRadius.size else 0.0)
    if maxRadius < 0.5:
        return rgba
    slices = max(1, int(slices))
    # Position of each pixel between the slices: -slices (nearest) .. 0 (in focus) .. slices (farthest)
    position = np.clip(signedRadius / maxRadius, -1.0, 1.0) * slices
    color = np.zeros((h, w, 4), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    # Back to front: the far slices, the focal plane, then the near ones (the nearest last)
    order = list(range(slices, 0, -1)) + [0] + list(range(-1, -slices - 1, -1))
    for index in order:
        if cancel is not None:
            cancel()
        # Coverage of the slice: linear between two slices (no visible steps)
        membership = np.clip(1.0 - np.abs(position - index), 0.0, 1.0).astype(np.float32)
        rows = np.flatnonzero(membership.any(axis=1))
        if rows.size == 0:
            continue
        columns = np.flatnonzero(membership.any(axis=0))
        radius = abs(index) / slices * maxRadius
        kernel = kernelFor(radius) if radius >= 0.5 else None
        marginY = kernel.shape[0] // 2 if kernel is not None else 0
        marginX = kernel.shape[1] // 2 if kernel is not None else 0
        y0, y1 = max(0, rows[0] - marginY), min(h, rows[-1] + 1 + marginY)
        x0, x1 = max(0, columns[0] - marginX), min(w, columns[-1] + 1 + marginX)
        cover = membership[y0:y1, x0:x1, np.newaxis]
        data = np.concatenate([rgba[y0:y1, x0:x1] * cover, cover], axis=2)
        blurred = _blurSlice(data, radius, lambda r, k=kernel, r0=radius: k if r == r0 else kernelFor(r), convolve)
        alpha = np.clip(blurred[:, :, 4:5], 0.0, 1.0)
        # "over": the slice on top of the farther ones; the weight follows the same compositing,
        # so that the partial coverages (between two slices, image edges) are normalized at the end
        color[y0:y1, x0:x1] = blurred[:, :, :4] + color[y0:y1, x0:x1] * (1.0 - alpha)
        weight[y0:y1, x0:x1] = alpha + weight[y0:y1, x0:x1] * (1.0 - alpha)
    return np.where(weight > 1e-4, color / np.maximum(weight, 1e-4), 0.0).astype(np.float32)


def blurRadius(depth: np.ndarray, math_: str, focalPlane: float, depthOfField: float, size: float,
               maxSize: float) -> np.ndarray:
    """
    The signed blur radius (pixels) of each pixel from its depth value:
      - "depth": the distance to the camera (Z pass): size * |z - focal| / z (thin lens, size at infinity);
      - "far = 0": 1 / distance (normalized depth, 0 at infinity), converted to a distance;
      - "linear": size * |v - focal| (e.g. a gradient drawn with a Roto or a Ramp, 0 to 1);
      - "direct": size * v, the blur amount itself (always behind the focal plane).
    Values within depthOfField / 2 of the focal plane stay sharp. Clamped to maxSize.
    """
    depth = depth.astype(np.float32)
    half = max(0.0, depthOfField) / 2.0
    if math_ == "direct":
        return np.clip(size * np.maximum(depth, 0.0), 0.0, maxSize).astype(np.float32)
    if math_ == "far = 0":
        depth = 1.0 / np.maximum(depth, 1e-6)
        if focalPlane > 0:
            focalPlane = 1.0 / focalPlane
    elif math_ == "depth":
        # No depth (0, e.g. the background of a CG render): at infinity
        depth = np.where(depth > 0, depth, np.float32(1e30))
    difference = depth - focalPlane
    distance = np.maximum(np.abs(difference) - half, 0.0)
    if math_ == "linear":
        radius = size * distance
    else:
        radius = size * distance / np.maximum(depth, 1e-6)
    return (np.sign(difference) * np.clip(radius, 0.0, maxSize)).astype(np.float32)
