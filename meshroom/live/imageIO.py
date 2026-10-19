"""
Image file input/output for the live compositing engine.

OpenImageIO (`pip install OpenImageIO`) is used when available: it reads every channel and
every part of multi-layer EXR files (AOVs). Without it, a reduced fallback based on
tifffile / imageio reads and writes the main color layer only.
"""
from __future__ import annotations

import glob
import os
import re
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np

from .color import linearToSrgb, srgbToLinear  # noqa: F401 (used by the viewer and the tests)
from .imageBuffer import BBox, ImageBuffer, RGBA, RGBA_CHANNELS, splitChannelNames
from .ops import alignToBBox

try:
    import OpenImageIO as oiio
except ImportError:  # pragma: no cover - depends on the environment
    oiio = None

# Matches the frame number placeholders of a sequence path: "####", "@@@@", "%04d", "%d"
_FRAME_PATTERN = re.compile(r"(#+|@+|%0?(\d*)d)")


def hasOpenImageIO() -> bool:
    return oiio is not None


# Sequences

def isSequencePath(path: str) -> bool:
    return bool(_FRAME_PATTERN.search(os.path.basename(path)))


def framePath(path: str, frame: int) -> str:
    """ Return the path of a frame of a sequence ("img.####.exr", 12 -> "img.0012.exr"). """
    def replace(match):
        token = match.group(1)
        if token[0] in "#@":
            padding = len(token)
        else:
            padding = int(match.group(2) or 0)
        return f"{frame:0{padding}d}" if frame >= 0 else f"-{-frame:0{padding}d}"
    directory, name = os.path.split(path)
    return os.path.join(directory, _FRAME_PATTERN.sub(replace, name, count=1))


def detectFrameRange(path: str) -> Optional[Tuple[int, int]]:
    """
    Return the (first, last) frames found on disk for a sequence path, or None if no frame exists.
    """
    directory, name = os.path.split(path)
    match = _FRAME_PATTERN.search(name)
    if not match:
        return None
    prefix, suffix = name[:match.start()], name[match.end():]
    regex = re.compile(re.escape(prefix) + r"(-?\d+)" + re.escape(suffix) + r"$")
    frames = []
    globPattern = os.path.join(glob.escape(directory) if directory else "", glob.escape(prefix) + "*" + glob.escape(suffix))
    for candidate in glob.glob(globPattern):
        found = regex.match(os.path.basename(candidate))
        if found:
            frames.append(int(found.group(1)))
    if not frames:
        return None
    return min(frames), max(frames)


# Color

_integerLuts = {}


def _integerLut(bits: int, conversion) -> np.ndarray:
    """
    Table converting every 8 or 16-bit integer value to float, through a colorspace conversion.

    Args:
        conversion: None / False (no conversion), True (sRGB to linear) or a (source, target) pair of
                    colorspaces converted by a curve of the color config.
    """
    from . import color
    key = (bits, conversion)
    if key not in _integerLuts:
        values = np.arange(1 << bits, dtype=np.float64) / ((1 << bits) - 1)
        if conversion is True:
            values = srgbToLinear(values)
        elif conversion:
            values = color.config().curve(*conversion)(values)
        if len(_integerLuts) > 64:
            _integerLuts.clear()
        _integerLuts[key] = np.asarray(values, dtype=np.float32)
    return _integerLuts[key]


def integerToFloat(raw: np.ndarray, conversions) -> np.ndarray:
    """
    Convert 8 or 16-bit integer pixels (h, w, c) to float32 through lookup tables (much faster than
    computing the colorspace curve on every pixel).

    Args:
        conversions: for each channel, its conversion (see _integerLut): True converts from sRGB to linear.
    """
    from .ops import parallelBands
    bits = 8 if raw.dtype == np.uint8 else 16
    raw = raw.astype(np.uint16, copy=False) if bits == 16 else raw
    luts = [_integerLut(bits, conversion if conversion else None) for conversion in conversions]
    out = np.empty(raw.shape, dtype=np.float32)

    def band(start, end):
        for channel, lut in enumerate(luts):
            out[start:end, :, channel] = np.take(lut, raw[start:end, :, channel])

    parallelBands(band, raw.shape[0])
    return out


def _colorChannels(channelNames):
    """ Mask of the channels holding color (R, G, B of the main layer), the ones colorspaces apply to. """
    if len(channelNames) == 1:
        # A grey image: the colorspace applies, except to an alpha or a depth channel
        return [channelNames[0].upper() not in ("A", "Z", "DEPTH")]
    return [name.upper() in ("R", "G", "B") for name in channelNames]


def convertColor(pixels: np.ndarray, colorMask, source: str, target: str, inPlace: bool = False) -> np.ndarray:
    """
    Convert the color channels (`colorMask`) of float pixels (h, w, c) between two colorspaces, in
    parallel bands of rows. A single color channel is converted as a grey (r = g = b).
    """
    from . import color
    from .ops import parallelBands
    config = color.config()
    indices = [i for i, isColor in enumerate(colorMask) if isColor]
    if not indices or config.isIdentity(source, target):
        return pixels
    out = pixels if inPlace else pixels.copy()
    curve = config.curve(source, target)
    processor = None if curve is not None else config.processor(source, target)

    def band(start, end):
        values = out[start:end, :, indices]
        if curve is not None:
            out[start:end, :, indices] = curve(values)
        elif len(indices) >= 3:
            out[start:end, :, indices[:3]] = processor(values[:, :, :3])
        else:
            grey = processor(np.repeat(values[:, :, :1], 3, axis=2))
            out[start:end, :, indices[0]] = grey[:, :, 1]

    parallelBands(band, out.shape[0])
    return out


def _toWorkingSpace(raw: np.ndarray, names, colorspace: str) -> np.ndarray:
    """ Float pixels of a file in the working space: the color channels converted from `colorspace`. """
    from . import color
    config = color.config()
    isIntegerFile = raw.dtype in (np.uint8, np.uint16)
    source = config.resolveForFile(colorspace, isIntegerFile)
    target = config.workingSpace()
    colorMask = _colorChannels(names)
    if isIntegerFile:
        # A curve is baked in the integer lookup tables
        curve = None if config.isIdentity(source, target) else config.curve(source, target)
        conversion = (source, target) if curve is not None else None
        pixels = integerToFloat(raw, [conversion if isColor else None for isColor in colorMask])
        if curve is not None or config.isIdentity(source, target):
            return pixels
        return convertColor(pixels, colorMask, source, target, inPlace=True)
    return convertColor(raw.astype(np.float32, copy=False), colorMask, source, target)


# Reading

def readImageSize(path: str) -> Tuple[int, int]:
    """ Return the (width, height) format of an image file, reading only its header when possible. """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if oiio is not None:
        inp = oiio.ImageInput.open(path)
        if not inp:
            raise IOError(f"Cannot open '{path}': {oiio.geterror()}")
        try:
            spec = inp.spec()
            return spec.full_width or spec.width, spec.full_height or spec.height
        finally:
            inp.close()
    image = _readFallback(path, "linear")
    return image.width, image.height


def readImage(path: str, colorspace: str = "auto") -> ImageBuffer:
    """
    Read an image file into an ImageBuffer with all its layers.

    The format is the display window of the file and the bbox its data window (overscan is kept).

    Args:
        path: the file path.
        colorspace: colorspace of the file (see meshroom.live.color), or "auto": sRGB for 8/16-bit integer
                    files, the working space otherwise. The color channels of the main layer are
                    converted to the working space.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if oiio is not None:
        return _readWithOIIO(path, colorspace)
    return _readFallback(path, colorspace)


def _readWithOIIO(path: str, colorspace: str) -> ImageBuffer:
    inp = oiio.ImageInput.open(path)
    if not inp:
        raise IOError(f"Cannot open '{path}': {oiio.geterror()}")
    try:
        parts = []
        metadata = {}
        format = None
        subimage = 0
        while inp.seek_subimage(subimage, 0):
            spec = inp.spec()
            fullWidth, fullHeight = spec.full_width or spec.width, spec.full_height or spec.height
            if subimage == 0:
                format = (fullWidth, fullHeight)
                metadata = {attr.name: attr.value for attr in spec.extra_attribs}
            names = list(spec.channelnames)
            # Read integer files in their own type: the conversion to float goes through a lookup table
            basetype = spec.format.basetype
            if basetype == oiio.UINT8:
                raw = inp.read_image(subimage, 0, 0, spec.nchannels, "uint8")
            elif basetype == oiio.UINT16:
                raw = inp.read_image(subimage, 0, 0, spec.nchannels, "uint16")
            else:
                raw = inp.read_image(subimage, 0, 0, spec.nchannels, "float")
            if raw is None:
                raise IOError(f"Cannot read '{path}': {inp.geterror()}")
            raw = np.asarray(raw).reshape(spec.height, spec.width, spec.nchannels)
            pixels = _toWorkingSpace(raw, names, colorspace)
            # Data window in Nuke coordinates (y up, origin at the bottom-left of the display window)
            x0 = spec.x - spec.full_x
            y1 = (spec.full_y + fullHeight) - spec.y
            bbox = BBox(x0, y1 - spec.height, x0 + spec.width, y1)
            # Multi-part EXR: channels of a part may be unprefixed, the part name is then the layer name
            partName = (spec.getattribute("oiio:subimagename") or spec.getattribute("name")) if subimage > 0 else None
            if partName and not any("." in n for n in names):
                names = [f"{partName}.{n}" for n in names]
            parts.append((names, pixels, bbox))
            subimage += 1
        # The failed seek that ends the loop leaves an error to retrieve
        inp.geterror()
    finally:
        inp.close()

    # Parts may have different data windows: use their union
    bbox = parts[0][2]
    for _, _, partBBox in parts[1:]:
        bbox = bbox.union(partBBox)
    layers = OrderedDict()
    for names, pixels, partBBox in parts:
        pixels = alignToBBox(pixels, partBBox, bbox)
        groups = splitChannelNames(names)
        for layerName, channels in groups.items():
            indices = [index for index, _ in channels]
            layerName = _uniqueName(layerName, layers)
            # Avoid copying the pixels when the layer is the whole part
            layerPixels = pixels if indices == list(range(pixels.shape[2])) else pixels[:, :, indices]
            layers[layerName] = (tuple(c for _, c in channels), layerPixels)
    return ImageBuffer(layers, format, bbox, metadata)


def _uniqueName(name, existing):
    if name not in existing:
        return name
    index = 2
    while f"{name}{index}" in existing:
        index += 1
    return f"{name}{index}"


def _readFallback(path: str, colorspace: str) -> ImageBuffer:
    """ Read the main color layer without OpenImageIO. """
    extension = os.path.splitext(path)[1].lower()
    if extension in (".tif", ".tiff"):
        import tifffile
        raw = tifffile.imread(path)
    else:
        try:
            import imageio.v3 as iio
        except ImportError:
            raise RuntimeError(f"Cannot read '{path}': install OpenImageIO (pip install OpenImageIO).")
        raw = iio.imread(path)
    if raw.ndim == 2:
        raw = raw[:, :, np.newaxis]
    if raw.shape[2] > 4:
        raw = raw[:, :, :4]
    names = RGBA_CHANNELS[:raw.shape[2]]
    pixels = _toWorkingSpace(raw, names, colorspace)
    return ImageBuffer.fromRGBA(pixels, (pixels.shape[1], pixels.shape[0]))


# Writing

def writeImage(path: str, image: ImageBuffer, dataType: str = "half", colorspace: str = "auto"):
    """
    Write an ImageBuffer to a file.

    With OpenImageIO, every layer is written to formats supporting arbitrary channels (EXR),
    named "layer.channel" (main layer channels are written unprefixed: "R", "G", "B", "A").

    Args:
        path: the output file path; its extension selects the format.
        dataType: "half", "float", "uint8" or "uint16".
        colorspace: colorspace of the file (see meshroom.live.color), or "auto": sRGB for integer data
                    types, the working space otherwise.
    """
    from . import color
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    extension = os.path.splitext(path)[1].lower()
    multiLayer = extension == ".exr"
    config = color.config()
    target = config.resolveForFile(colorspace, dataType in ("uint8", "uint16"))
    source = config.workingSpace()

    # EXR keeps the data window (overscan); other formats are cropped to the format
    formatBBox = BBox.fromFormat(image.width, image.height)
    bbox = image.bbox if multiLayer else formatBBox
    names, arrays = [], []
    for layerName, channels, pixels in image.layers():
        if layerName != RGBA and not multiLayer:
            continue
        pixels = alignToBBox(pixels, image.bbox, bbox)
        if layerName == RGBA:
            pixels = convertColor(pixels, [c in RGBA_CHANNELS[:3] for c in channels], source, target)
        for i, channel in enumerate(channels):
            names.append(channel if layerName == RGBA else f"{layerName}.{channel}")
            arrays.append(pixels[:, :, i])
    if not arrays:
        raise ValueError("The image has no 'rgba' layer to write in this file format.")
    data = np.ascontiguousarray(np.stack(arrays, axis=2))

    if oiio is not None:
        _writeWithOIIO(path, data, names, dataType, bbox, (image.width, image.height))
    else:
        _writeFallback(path, data, dataType)


def _writeWithOIIO(path, data, names, dataType, bbox, format):
    typeDesc = {"half": oiio.HALF, "float": oiio.FLOAT, "uint8": oiio.UINT8, "uint16": oiio.UINT16}[dataType]
    spec = oiio.ImageSpec(data.shape[1], data.shape[0], data.shape[2], oiio.TypeDesc(typeDesc))
    # Data window (bbox) inside the display window (format), OpenImageIO coordinates are y down
    spec.x = bbox.x0
    spec.y = format[1] - bbox.y1
    spec.full_x = 0
    spec.full_y = 0
    spec.full_width = format[0]
    spec.full_height = format[1]
    spec.channelnames = tuple(names)
    if "A" in names:
        spec.alpha_channel = names.index("A")
    if path.lower().endswith(".exr"):
        spec.attribute("compression", "zip")
    out = oiio.ImageOutput.create(path)
    if not out:
        raise IOError(f"Cannot create '{path}': {oiio.geterror()}")
    try:
        if not out.open(path, spec):
            raise IOError(f"Cannot open '{path}' for writing: {out.geterror()}")
        if not out.write_image(data):
            raise IOError(f"Cannot write '{path}': {out.geterror()}")
    finally:
        out.close()


def _writeFallback(path, data, dataType):
    extension = os.path.splitext(path)[1].lower()
    if dataType == "uint8":
        data = (np.clip(data, 0, 1) * 255 + 0.5).astype(np.uint8)
    elif dataType == "uint16":
        data = (np.clip(data, 0, 1) * 65535 + 0.5).astype(np.uint16)
    elif dataType == "half":
        data = data.astype(np.float16)
    if extension in (".tif", ".tiff"):
        import tifffile
        tifffile.imwrite(path, data, photometric="rgb" if data.shape[2] >= 3 else "minisblack")
        return
    try:
        import imageio.v3 as iio
    except ImportError:
        raise RuntimeError(f"Cannot write '{path}': install OpenImageIO (pip install OpenImageIO).")
    iio.imwrite(path, data)
