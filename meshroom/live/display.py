"""
Conversion of the rendered images to 8-bit pixels for the viewer: layer and channel choice, exposure,
gamma and display transform (meshroom.live.color). Qt-free, used by meshroom.ui.live.
"""
from typing import Optional

import numpy as np

from .color import linearToSrgb
from .imageBuffer import BBox, ImageBuffer
from .ops import parallelBands


# Display lookup table: linear values in [0, 1] sampled on this many steps
_DISPLAY_LUT_SIZE = 16384
_displayLuts = {}


def _displayLut(gamma: float, transform=None) -> np.ndarray:
    """
    Table converting linear values in [0, 1] to 8-bit display values, with a display gamma and the
    curve of a display transform (sRGB without transform).
    """
    curve = linearToSrgb if transform is None else transform.curve
    key = (round(gamma, 4), id(curve))
    lut = _displayLuts.get(key)
    if lut is None:
        values = np.linspace(0.0, 1.0, _DISPLAY_LUT_SIZE, dtype=np.float64)
        if gamma > 0 and gamma != 1.0:
            values = np.power(values, 1.0 / gamma)
        if curve is not None:
            values = curve(values)
        lut = (np.clip(values, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        if len(_displayLuts) > 32:
            _displayLuts.clear()
        _displayLuts[key] = lut
    return lut


def displayPixels(image: ImageBuffer, layer: str, channelMode: str, exposure: float, gamma: float,
                  stride: int = 1, region: Optional[BBox] = None, transform=None) -> np.ndarray:
    """
    Convert a layer of an image to 8-bit display pixels, shape (h, w, 4), opaque, covering the image bbox.

    Args:
        layer: the layer to display; its first three channels are shown as RGB, a single channel as grey.
        channelMode: "rgb", "r", "g", "b", "a" (one channel shown as grey) or "luminance".
        exposure: exposure in f-stops (the pixels are multiplied by 2^exposure).
        gamma: display gamma applied after the exposure.
        stride: keep one pixel out of `stride` in each direction (when the viewer is zoomed out).
        region: the part of the bbox to convert (the visible area), the whole bbox by default.
        transform: the display transform (meshroom.live.color.DisplayTransform), sRGB by default.
    """
    if not image.hasLayer(layer):
        layer = image.layerNames[0]
    channels = image.channels(layer)
    bbox = image.bbox
    region = bbox if region is None else region.intersection(bbox)
    pixels = image.layer(layer)[bbox.y1 - region.y1:bbox.y1 - region.y0:stride,
                                region.x0 - bbox.x0:region.x1 - bbox.x0:stride]
    height, width = pixels.shape[:2]
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:, :, 3] = 255
    if out.size == 0:
        return out

    def index(name, fallbackIndex=None):
        if name in channels:
            return channels.index(name)
        if fallbackIndex is not None and fallbackIndex < len(channels):
            return fallbackIndex
        return None

    if len(channels) == 1:
        sources = [0, 0, 0]
    else:
        sources = [index("R", 0), index("G", 1), index("B", 2)]
    if channelMode in ("r", "g", "b"):
        sources = [sources["rgb".index(channelMode)]] * 3
    elif channelMode == "a":
        sources = [index("A")] * 3
    luminance = channelMode == "luminance"

    if transform is not None and transform.processor is not None:
        return _processedDisplayPixels(pixels, out, sources, luminance, exposure, gamma, transform.processor)
    lut = _displayLut(gamma, transform)
    gain = np.float32((2.0 ** exposure) * (_DISPLAY_LUT_SIZE - 1))
    # Primaries of the display (applied on RGB before the curve)
    matrix = transform.matrix.T.astype(np.float32) if transform is not None and transform.matrix is not None else None

    def toDisplay(values):
        values = values * gain
        np.nan_to_num(values, copy=False, nan=0.0, posinf=_DISPLAY_LUT_SIZE - 1, neginf=0.0)
        np.clip(values, 0, _DISPLAY_LUT_SIZE - 1, out=values)
        return np.take(lut, values.astype(np.int32))

    def band(start, end):
        rows = pixels[start:end]
        if luminance:
            r, g, b = (rows[:, :, i] if i is not None else 0.0 for i in sources)
            grey = toDisplay(0.2126 * r + 0.7152 * g + 0.0722 * b)
            out[start:end, :, 0] = grey
            out[start:end, :, 1] = grey
            out[start:end, :, 2] = grey
            return
        if matrix is not None and channelMode == "rgb" and None not in sources:
            rgb = np.stack([rows[:, :, i] for i in sources], axis=2) @ matrix
            for channel in range(3):
                out[start:end, :, channel] = toDisplay(rgb[:, :, channel])
            return
        converted = {}
        for channel, source in enumerate(sources):
            if source is None:
                out[start:end, :, channel] = 0
                continue
            if source not in converted:
                converted[source] = toDisplay(rows[:, :, source])
            out[start:end, :, channel] = converted[source]

    parallelBands(band, height)
    return out


def _processedDisplayPixels(pixels, out, sources, luminance, exposure, gamma, processor):
    """ displayPixels through a display transform that is not a curve per channel (OCIO processor). """
    gain = np.float32(2.0 ** exposure)

    def band(start, end):
        rows = pixels[start:end]
        channels = [rows[:, :, i] if i is not None else np.zeros(rows.shape[:2], np.float32) for i in sources]
        if luminance:
            grey = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
            channels = [grey, grey, grey]
        rgb = np.stack(channels, axis=2) * gain
        if gamma > 0 and gamma != 1.0:
            rgb = np.sign(rgb) * np.abs(rgb) ** (1.0 / gamma)
        rgb = np.nan_to_num(processor(rgb), nan=0.0, posinf=1.0, neginf=0.0)
        out[start:end, :, :3] = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    parallelBands(band, pixels.shape[0])
    return out
