"""
Pixel operations shared by the compositing nodes (numpy, float32).

Coordinates follow Nuke conventions so that scripts can be reproduced: the origin is the
bottom-left corner of the format and y goes up, whereas numpy row 0 is the top of a bbox.

Heavy operations split the image in bands of rows processed by a thread pool (numpy releases
the GIL), and call an optional `cancel` function between bands so that an outdated render
stops quickly.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Sequence

import numpy as np

from .imageBuffer import BBox


# Parallelism

_pool: Optional[ThreadPoolExecutor] = None


def threadCount() -> int:
    """ Number of threads used by the pixel operations (MESHROOM_LIVE_THREADS, all cores by default). """
    try:
        requested = int(os.environ.get("MESHROOM_LIVE_THREADS", "0"))
    except ValueError:
        requested = 0
    return requested if requested > 0 else max(1, os.cpu_count() or 1)


def _getPool() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=threadCount(), thread_name_prefix="LiveOps")
    return _pool


def parallelBands(func: Callable[[int, int], None], length: int, cancel: Optional[Callable[[], None]] = None,
                  minBand: int = 16):
    """
    Call `func(start, end)` on bands covering [0, length), concurrently.

    Bands are smaller than the number of threads requires, so that `cancel` (which raises to
    stop) is called often. Must not be nested: `func` must not call parallelBands itself.
    """
    if length <= 0:
        return
    threads = threadCount()
    nbBands = max(1, min(length // minBand, threads * 4))
    if nbBands == 1:
        if cancel:
            cancel()
        func(0, length)
        return
    bounds = np.linspace(0, length, nbBands + 1).astype(int)

    def run(start, end):
        if cancel:
            cancel()
        func(start, end)

    futures = [_getPool().submit(run, int(bounds[i]), int(bounds[i + 1])) for i in range(nbBands)]
    error = None
    for future in futures:
        try:
            future.result()
        except BaseException as e:  # Wait for every band before raising
            error = error or e
    if error is not None:
        raise error


def mapRows(func: Callable, arrays: Sequence, outChannels: int, cancel=None) -> np.ndarray:
    """
    Apply `func(*bands) -> band` over bands of rows of several arrays of the same height and
    return the concatenated result, of shape (h, w, outChannels).
    Scalars and None in `arrays` are passed through unchanged.
    """
    reference = next(a for a in arrays if isinstance(a, np.ndarray))
    height, width = reference.shape[:2]
    out = np.empty((height, width, outChannels), dtype=np.float32)

    def band(start, end):
        parts = [a[start:end] if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[0] == height else a
                 for a in arrays]
        out[start:end] = func(*parts)

    parallelBands(band, height, cancel)
    return out


# BBox alignment

def alignToBBox(pixels: np.ndarray, src: BBox, dst: BBox) -> np.ndarray:
    """
    Return the pixels of `src` placed in the `dst` rectangle: cropped where `dst` is smaller,
    black where it is larger. Returns `pixels` itself when the rectangles are equal.
    """
    if tuple(src) == tuple(dst):
        return pixels
    out = np.zeros((dst.height, dst.width, pixels.shape[2]), dtype=np.float32)
    common = src.intersection(dst)
    if common.isEmpty:
        return out
    # Row 0 is the top (y1 - 1) of each rectangle
    out[dst.y1 - common.y1:dst.y1 - common.y0, common.x0 - dst.x0:common.x1 - dst.x0] = \
        pixels[src.y1 - common.y1:src.y1 - common.y0, common.x0 - src.x0:common.x1 - src.x0]
    return out


def fitToFormat(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    """ Crop or pad (with black) pixels whose bbox is (0, 0, w, h) to another format. """
    h, w = pixels.shape[:2]
    return alignToBBox(pixels, BBox(0, 0, w, h), BBox(0, 0, int(width), int(height)))


# Masks

MASK_CHANNELS = ("alpha", "red", "green", "blue", "luminance")


def maskValues(image, channel: str, invert: bool, bbox: BBox) -> np.ndarray:
    """
    Return the mask of an image over `bbox`, shape (h, w, 1): one channel of its "rgba" layer
    ("alpha", "red", "green", "blue") or its luminance. Outside the mask bbox the mask is 0
    (1 when inverted).
    """
    rgba = image.rgba()
    if channel == "luminance":
        values = 0.2126 * rgba[:, :, 0:1] + 0.7152 * rgba[:, :, 1:2] + 0.0722 * rgba[:, :, 2:3]
    else:
        index = {"red": 0, "green": 1, "blue": 2}.get(channel, 3)
        values = rgba[:, :, index:index + 1]
    values = alignToBBox(np.ascontiguousarray(values, dtype=np.float32), image.bbox, bbox)
    if invert:
        values = 1.0 - values
    return values


# Color

def grade(pixels: np.ndarray, blackpoint=0.0, whitepoint=1.0, lift=0.0, gain=1.0, multiply=1.0,
          offset=0.0, gamma=1.0, blackClamp=True, whiteClamp=False) -> np.ndarray:
    """
    Nuke Grade formula:
        A = multiply * (gain - lift) / (whitepoint - blackpoint)
        B = offset + lift - A * blackpoint
        out = (A * in + B) ^ (1 / gamma)    (gamma applied to positive values only)
    Each parameter is a scalar or a per-channel sequence.
    """
    def vec(value):
        return np.asarray(value, dtype=np.float32)
    blackpoint, whitepoint, lift, gain = vec(blackpoint), vec(whitepoint), vec(lift), vec(gain)
    multiply, offset, gamma = vec(multiply), vec(offset), vec(gamma)
    span = whitepoint - blackpoint
    span = np.where(span == 0, np.float32(1e-6), span)
    a = multiply * (gain - lift) / span
    b = offset + lift - a * blackpoint
    out = pixels * a + b
    if np.any(gamma != 1.0):
        invGamma = np.float32(1.0) / np.where(gamma <= 0, np.float32(1e-6), gamma)
        out = np.where(out > 0, np.power(np.maximum(out, 0), invGamma), out)
    if blackClamp:
        out = np.maximum(out, 0)
    if whiteClamp:
        out = np.minimum(out, 1)
    return out.astype(np.float32, copy=False)


def mix(original: np.ndarray, processed: np.ndarray, amount) -> np.ndarray:
    """ Linear interpolation between the original and processed pixels (amount: scalar or (h, w, 1) mask). """
    if np.isscalar(amount):
        if amount >= 1.0:
            return processed
        if amount <= 0.0:
            return original
    return (original + (processed - original) * amount).astype(np.float32, copy=False)


# Merge

MERGE_OPERATIONS = (
    "over", "under", "plus", "minus", "from", "multiply", "screen", "max", "min",
    "in", "out", "mask", "stencil", "atop", "xor", "difference", "average", "copy",
)

# Operations that return B unchanged when A is black and transparent
MERGE_IDENTITY_WITHOUT_A = ("over", "under", "plus", "from", "screen", "atop", "xor", "stencil")


def merge(a: np.ndarray, b: np.ndarray, operation: str) -> np.ndarray:
    """
    Merge two premultiplied RGBA images (h, w, 4) with a Nuke Merge operation, applied to
    all channels including alpha ("A" is the foreground, "B" the background).
    """
    alphaA = a[:, :, 3:4]
    alphaB = b[:, :, 3:4]
    if operation == "over":
        out = a + b * (1 - alphaA)
    elif operation == "under":
        out = a * (1 - alphaB) + b
    elif operation == "plus":
        out = a + b
    elif operation == "minus":
        out = a - b
    elif operation == "from":
        out = b - a
    elif operation == "multiply":
        out = a * b
    elif operation == "screen":
        out = a + b - a * b
    elif operation == "max":
        out = np.maximum(a, b)
    elif operation == "min":
        out = np.minimum(a, b)
    elif operation == "in":
        out = a * alphaB
    elif operation == "out":
        out = a * (1 - alphaB)
    elif operation == "mask":
        out = b * alphaA
    elif operation == "stencil":
        out = b * (1 - alphaA)
    elif operation == "atop":
        out = a * alphaB + b * (1 - alphaA)
    elif operation == "xor":
        out = a * (1 - alphaB) + b * (1 - alphaA)
    elif operation == "difference":
        out = np.abs(a - b)
    elif operation == "average":
        out = (a + b) * 0.5
    elif operation == "copy":
        out = a.copy()
    else:
        raise ValueError(f"Unknown merge operation: {operation}")
    return out.astype(np.float32, copy=False)


# Geometry

def transformMatrix(translate=(0.0, 0.0), rotate=0.0, scale=(1.0, 1.0), skewX=0.0, center=(0.0, 0.0)) -> np.ndarray:
    """
    Return the 3x3 matrix of a Nuke-like Transform (in pixels, bottom-left origin):
    translate * center * rotate * skew * scale * -center. Rotation in degrees, counter-clockwise.
    """
    cx, cy = center
    angle = math.radians(rotate)
    toCenter = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    fromCenter = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    skew = np.array([[1, math.tan(math.radians(skewX)), 0], [0, 1, 0], [0, 0, 1]])
    scaling = np.array([[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]])
    translation = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]])
    return translation @ toCenter @ rotation @ skew @ scaling @ fromCenter


def integerTranslation(matrix: np.ndarray) -> Optional[tuple]:
    """ Return (dx, dy) if the matrix is a translation by whole pixels, None otherwise. """
    if not np.allclose(matrix[:2, :2], np.eye(2), atol=1e-9) or not np.allclose(matrix[2], [0, 0, 1], atol=1e-12):
        return None
    dx, dy = matrix[0, 2], matrix[1, 2]
    if abs(dx - round(dx)) > 1e-6 or abs(dy - round(dy)) > 1e-6:
        return None
    return int(round(dx)), int(round(dy))


def transformedBBox(bbox: BBox, matrix: np.ndarray) -> BBox:
    """ Bounding box of a transformed rectangle (rounded outwards). """
    corners = np.array([[bbox.x0, bbox.y0, 1], [bbox.x1, bbox.y0, 1], [bbox.x0, bbox.y1, 1], [bbox.x1, bbox.y1, 1]],
                       dtype=np.float64).T
    points = matrix @ corners
    if not (matrix[2, 0] == 0 and matrix[2, 1] == 0):
        # Projective: a corner behind the horizon makes the image unbounded (clamped by the caller)
        if np.any(points[2] <= 1e-9):
            return BBox(-10 ** 6, -10 ** 6, 10 ** 6, 10 ** 6)
        points = points / points[2]
    return BBox(int(math.floor(points[0].min() + 1e-6)), int(math.floor(points[1].min() + 1e-6)),
                int(math.ceil(points[0].max() - 1e-6)), int(math.ceil(points[1].max() - 1e-6)))


def warpAffine(pixels: np.ndarray, srcBBox: BBox, matrix: np.ndarray, dstBBox: BBox, filter: str = "bilinear",
               cancel=None) -> np.ndarray:
    """
    Apply an affine or projective transform (3x3 matrix, Nuke coordinates) to the pixels of
    `srcBBox` and return the pixels of `dstBBox`. Each output pixel samples the input at the
    inverse-transformed position of its center; outside of the input is black.

    Args:
        filter: "impulse" (nearest pixel) or "bilinear".
    """
    h, w, c = pixels.shape
    out = np.zeros((dstBBox.height, dstBBox.width, c), dtype=np.float32)
    if out.size == 0 or pixels.size == 0:
        return out
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return out
    xs = dstBBox.x0 + np.arange(dstBBox.width, dtype=np.float64) + 0.5

    def band(start, end):
        # Output pixel centers in Nuke coordinates (y up), rows start..end of the output
        ys = dstBBox.y1 - (np.arange(start, end, dtype=np.float64) + 0.5)
        gridX, gridY = np.meshgrid(xs, ys)
        srcX = inverse[0, 0] * gridX + inverse[0, 1] * gridY + inverse[0, 2]
        srcY = inverse[1, 0] * gridX + inverse[1, 1] * gridY + inverse[1, 2]
        if not (inverse[2, 0] == 0 and inverse[2, 1] == 0):
            # Projective transform (CornerPin): homogeneous division; behind the horizon is outside
            weight = inverse[2, 0] * gridX + inverse[2, 1] * gridY + inverse[2, 2]
            behind = weight <= 1e-12
            weight = np.where(behind, 1.0, weight)
            srcX = np.where(behind, -1e9, srcX / weight)
            srcY = np.where(behind, -1e9, srcY / weight)
        # Continuous array coordinates (column, row) of the input, pixel centers at integers
        col = srcX - srcBBox.x0 - 0.5
        row = (srcBBox.y1 - srcY) - 0.5
        if filter == "impulse":
            ci = np.floor(col + 0.5).astype(np.int64)
            ri = np.floor(row + 0.5).astype(np.int64)
            valid = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
            result = np.zeros((end - start, dstBBox.width, c), dtype=np.float32)
            result[valid] = pixels[ri[valid], ci[valid]]
            out[start:end] = result
            return
        c0 = np.floor(col).astype(np.int64)
        r0 = np.floor(row).astype(np.int64)
        fc = (col - c0).astype(np.float32)[:, :, np.newaxis]
        fr = (row - r0).astype(np.float32)[:, :, np.newaxis]

        def sample(r, cc):
            valid = (cc >= 0) & (cc < w) & (r >= 0) & (r < h)
            values = pixels[np.clip(r, 0, h - 1), np.clip(cc, 0, w - 1)]
            return values * valid[:, :, np.newaxis]

        top = sample(r0, c0) * (1 - fc) + sample(r0, c0 + 1) * fc
        bottom = sample(r0 + 1, c0) * (1 - fc) + sample(r0 + 1, c0 + 1) * fc
        out[start:end] = top * (1 - fr) + bottom * fr

    parallelBands(band, dstBBox.height, cancel)
    return out


# Blur

def blurSigma(size: float) -> float:
    """ Gaussian standard deviation for a Nuke-like blur "size" (the width of the filter in pixels). """
    return max(size, 0.0) / 2.0 / 1.6


def gaussianKernel(size: float) -> np.ndarray:
    """ 1D gaussian kernel for a blur "size". """
    sigma = blurSigma(size)
    if sigma < 1e-3:
        return np.ones(1, dtype=np.float32)
    radius = max(1, int(math.ceil(sigma * 3)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return (kernel / kernel.sum()).astype(np.float32)


def boxRadii(sigma: float, passes: int = 3) -> list:
    """
    Radii of `passes` successive box blurs approximating a gaussian of standard deviation `sigma`
    (the variance of a box of width 2r+1 is ((2r+1)^2 - 1) / 12).
    """
    idealWidth = math.sqrt(12.0 * sigma * sigma / passes + 1.0)
    lower = int(math.floor(idealWidth))
    if lower % 2 == 0:
        lower -= 1
    upper = lower + 2
    nbLower = round((12 * sigma * sigma - passes * lower * lower - 4 * passes * lower - 3 * passes) / (-4 * lower - 4))
    widths = [lower if i < nbLower else upper for i in range(passes)]
    return [max(0, (width - 1) // 2) for width in widths]


def blurRadius(size: float) -> int:
    """ Number of pixels by which a blur extends the bbox, on each side. """
    sigma = blurSigma(size)
    if sigma < 1e-3:
        return 0
    if sigma < 2.0:
        return len(gaussianKernel(size)) // 2
    return sum(boxRadii(sigma))


try:
    from scipy.ndimage import uniform_filter1d as _uniformFilter1d
except ImportError:  # pragma: no cover - depends on the environment
    _uniformFilter1d = None


def _boxBlurAxis(pixels: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """ Box blur of width 2r+1 along an axis, extending the result by r pixels on both sides. """
    if radius <= 0:
        return pixels
    width = 2 * radius + 1
    if _uniformFilter1d is not None:
        # Running box filter in C (scipy), on the input padded by the radius
        padWidth = [(0, 0)] * pixels.ndim
        padWidth[axis] = (radius, radius)
        padded = np.pad(pixels, padWidth, mode="constant")
        return _uniformFilter1d(padded, width, axis=axis, mode="constant", cval=0.0)
    # Running sum in float64: precise even over long rows of HDR values
    length = pixels.shape[axis]
    sums = np.cumsum(pixels, axis=axis, dtype=np.float64)
    zeroShape = list(sums.shape)
    zeroShape[axis] = 1
    sums = np.concatenate([np.zeros(zeroShape), sums], axis=axis)
    # Output pixel i averages the input pixels i - 2r .. i
    index = np.arange(length + 2 * radius)
    upper = np.take(sums, np.minimum(index + 1, length), axis=axis)
    lower = np.take(sums, np.clip(index - 2 * radius, 0, length), axis=axis)
    return ((upper - lower) * (1.0 / width)).astype(np.float32)


def _kernelAxis(pixels: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """ Convolution with a small kernel along an axis, extending the result by its radius. """
    radius = len(kernel) // 2
    if radius == 0:
        return pixels
    padWidth = [(0, 0)] * pixels.ndim
    padWidth[axis] = (2 * radius, 2 * radius)
    padded = np.pad(pixels, padWidth, mode="constant")
    length = pixels.shape[axis] + 2 * radius
    out = np.zeros(tuple(length if i == axis else s for i, s in enumerate(pixels.shape)), dtype=np.float32)
    for i, weight in enumerate(kernel):
        out += weight * np.take(padded, range(i, i + length), axis=axis)
    return out


def _blurAxis(pixels: np.ndarray, size: float, axis: int, cancel=None) -> np.ndarray:
    """ Blur along one axis (0: vertical, 1: horizontal), in parallel bands of the other axis. """
    sigma = blurSigma(size)
    if sigma < 1e-3:
        return pixels
    if sigma < 2.0:
        def process(band):
            return _kernelAxis(band, gaussianKernel(size), axis)
        radius = len(gaussianKernel(size)) // 2
    else:
        radii = boxRadii(sigma)
        radius = sum(radii)

        def process(band):
            for r in radii:
                band = _boxBlurAxis(band, r, axis)
            return band

    outShape = list(pixels.shape)
    outShape[axis] += 2 * radius
    out = np.empty(outShape, dtype=np.float32)
    other = 1 - axis

    def band(start, end):
        if other == 0:
            out[start:end] = process(pixels[start:end])
        else:
            out[:, start:end] = process(pixels[:, start:end])

    parallelBands(band, pixels.shape[other], cancel)
    return out


def blur(pixels: np.ndarray, sizeX: float, sizeY: float, cancel=None) -> np.ndarray:
    """
    Separable gaussian blur (three box blurs for large sizes, which costs the same whatever the
    size). The result is larger than the input by `blurRadius` on each side, as in Nuke.
    """
    out = _blurAxis(pixels, sizeX, 1, cancel)
    return _blurAxis(out, sizeY, 0, cancel)


def channelMask(channels: Sequence[str], selected: str) -> np.ndarray:
    """
    Return a boolean mask over `channels` of the channels affected by a "channels" knob value:
    "rgb", "rgba", "alpha" or "all".
    """
    names = {"rgb": ("R", "G", "B"), "rgba": ("R", "G", "B", "A"), "alpha": ("A",)}.get(selected)
    if names is None:
        return np.ones(len(channels), dtype=bool)
    return np.array([c in names for c in channels], dtype=bool)


def transformImage(image, matrix: np.ndarray, filter: str = "bilinear", cancel=None):
    """
    Return an ImageBuffer transformed by a 3x3 matrix (Nuke coordinates), all its layers. Whole-pixel
    translations only move the bbox; the overscan is limited to the format extended by its size.
    """
    shift = integerTranslation(matrix)
    if shift is not None:
        return image.mapLayers(lambda pixels: pixels, bbox=image.bbox.translated(*shift))
    limit = BBox(-image.width, -image.height, 2 * image.width, 2 * image.height)
    bbox = transformedBBox(image.bbox, matrix).clampedTo(limit)
    return image.mapLayers(lambda pixels: warpAffine(pixels, image.bbox, matrix, bbox, filter, cancel), bbox=bbox)


# Motion blur

def shiftAccumulate(image, offsets, cancel=None):
    """
    Average of an image shifted by `offsets` [(dx, dy), ...] (pixels, Nuke coordinates, y up), all its
    layers: a fast motion blur along a translation. Each shift is rounded to whole pixels (the samples
    are about one pixel apart), identical shifts are counted once with their weight.
    """
    from collections import Counter, OrderedDict
    counts = Counter((int(round(dx)), int(round(dy))) for dx, dy in offsets)
    total = float(sum(counts.values()))
    bbox = image.bbox
    out = bbox
    for dx, dy in counts:
        out = out.union(bbox.translated(dx, dy))
    layers = OrderedDict()
    for name, channels, pixels in image.layers():
        result = np.zeros((out.height, out.width, pixels.shape[2]), dtype=np.float32)

        def band(start, end, pixels=pixels, result=result):
            # Output rows [start, end): each shifted copy adds its overlapping rows
            for (dx, dy), count in counts.items():
                weight = np.float32(count / total)
                # Row of the output where the shifted input starts (rows go down, y goes up)
                top = (out.y1 - (bbox.y1 + dy))
                left = (bbox.x0 + dx) - out.x0
                r0, r1 = max(start, top), min(end, top + bbox.height)
                if r0 >= r1:
                    continue
                result[r0:r1, left:left + bbox.width] += pixels[r0 - top:r1 - top] * weight

        parallelBands(band, out.height, cancel)
        layers[name] = (channels, result)
    return image.withPixels(layers, out)


def resizeBilinear(values: np.ndarray, height: int, width: int) -> np.ndarray:
    """ Bilinear resize of a (h, w, c) array, the corners aligned. """
    h, w = values.shape[:2]
    rows = np.linspace(0, h - 1, height)
    columns = np.linspace(0, w - 1, width)
    r0, c0 = np.floor(rows).astype(int), np.floor(columns).astype(int)
    r1, c1 = np.minimum(r0 + 1, h - 1), np.minimum(c0 + 1, w - 1)
    fr, fc = (rows - r0)[:, np.newaxis, np.newaxis], (columns - c0)[np.newaxis, :, np.newaxis]
    top = values[r0][:, c0] * (1 - fc) + values[r0][:, c1] * fc
    bottom = values[r1][:, c0] * (1 - fc) + values[r1][:, c1] * fc
    return (top * (1 - fr) + bottom * fr).astype(np.float32)


def vectorBlur(image, field: np.ndarray, fieldBBox, times, cancel=None):
    """
    Motion blur along a per-pixel motion: `field` (h, w, 2) is the displacement (dx, dy) per unit of
    time of each pixel of `fieldBBox`; the result averages the image sampled at x - field * t for
    each t of `times` (bilinear), all its layers.
    """
    from collections import OrderedDict
    out = fieldBBox
    xs = out.x0 + np.arange(out.width, dtype=np.float64) + 0.5
    src = image.bbox
    layers = OrderedDict()
    stacked = [(name, channels, pixels) for name, channels, pixels in image.layers()]
    for name, channels, pixels in stacked:
        h, w, c = pixels.shape
        result = np.zeros((out.height, out.width, c), dtype=np.float32)

        def band(start, end, pixels=pixels, result=result, h=h, w=w, c=c):
            ys = out.y1 - (np.arange(start, end, dtype=np.float64) + 0.5)
            gridX, gridY = np.meshgrid(xs, ys)
            motion = field[start:end]
            accum = np.zeros((end - start, out.width, c), dtype=np.float32)
            for t in times:
                col = gridX - motion[:, :, 0] * t - src.x0 - 0.5
                row = (src.y1 - (gridY - motion[:, :, 1] * t)) - 0.5
                c0, r0 = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
                fc = (col - c0).astype(np.float32)[:, :, np.newaxis]
                fr = (row - r0).astype(np.float32)[:, :, np.newaxis]

                def sample(rr, cc):
                    valid = (cc >= 0) & (cc < w) & (rr >= 0) & (rr < h)
                    return pixels[np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)] * valid[:, :, np.newaxis]

                accum += (sample(r0, c0) * (1 - fc) + sample(r0, c0 + 1) * fc) * (1 - fr) + \
                         (sample(r0 + 1, c0) * (1 - fc) + sample(r0 + 1, c0 + 1) * fc) * fr
            result[start:end] = accum / np.float32(len(times))

        parallelBands(band, out.height, cancel)
        layers[name] = (channels, result)
    return image.withPixels(layers, out)


def downscaleImage(image, factor: int):
    """
    An image (all its layers) reduced `factor` times (average of factor x factor blocks), for the
    proxy renders: its format is divided by the factor (rounded up), its bbox follows.
    """
    import math
    from collections import OrderedDict
    factor = int(factor)
    if factor <= 1:
        return image
    bbox = image.bbox
    out = BBox(int(math.floor(bbox.x0 / factor)), int(math.floor(bbox.y0 / factor)),
               int(math.ceil(bbox.x1 / factor)), int(math.ceil(bbox.y1 / factor)))
    full = BBox(out.x0 * factor, out.y0 * factor, out.x1 * factor, out.y1 * factor)
    layers = OrderedDict()
    for name, channels, pixels in image.layers():
        aligned = alignToBBox(pixels, bbox, full)
        h, w, c = aligned.shape
        layers[name] = (channels, aligned.reshape(h // factor, factor, w // factor, factor, c)
                        .mean(axis=(1, 3), dtype=np.float32))
    width, height = image.format
    return image.withPixels(layers, out, (int(math.ceil(width / factor)), int(math.ceil(height / factor))))
