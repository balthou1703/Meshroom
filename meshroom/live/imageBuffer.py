"""
In-memory image representation used by the live compositing engine.

Coordinates follow Nuke: pixels, origin at the bottom-left corner of the format, y up.
An image has a *format* (display window, `width` x `height`) and a *bounding box* (data window,
`bbox`) that holds its pixels. The bbox can be larger than the format (overscan) or smaller.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

# Name of the main color layer, as in Nuke
RGBA = "rgba"
RGBA_CHANNELS = ("R", "G", "B", "A")


class BBox(NamedTuple):
    """ Pixel rectangle [x0, x1) x [y0, y1), in Nuke coordinates (y up). """
    x0: int
    y0: int
    x1: int
    y1: int

    @classmethod
    def fromFormat(cls, width: int, height: int) -> "BBox":
        return cls(0, 0, int(width), int(height))

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def isEmpty(self) -> bool:
        return self.width == 0 or self.height == 0

    def union(self, other: "BBox") -> "BBox":
        if self.isEmpty:
            return other
        if other.isEmpty:
            return self
        return BBox(min(self.x0, other.x0), min(self.y0, other.y0), max(self.x1, other.x1), max(self.y1, other.y1))

    def intersection(self, other: "BBox") -> "BBox":
        result = BBox(max(self.x0, other.x0), max(self.y0, other.y0), min(self.x1, other.x1), min(self.y1, other.y1))
        return result if not result.isEmpty else BBox(0, 0, 0, 0)

    def expanded(self, dx: int, dy: int) -> "BBox":
        return BBox(self.x0 - dx, self.y0 - dy, self.x1 + dx, self.y1 + dy)

    def translated(self, dx: int, dy: int) -> "BBox":
        return BBox(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def clampedTo(self, limit: "BBox") -> "BBox":
        return self.intersection(limit)


class ImageInfo:
    """
    Information about a node output that is known without rendering pixels.

    Attributes:
        width, height: the image format (display window) in pixels.
        firstFrame, lastFrame: the frame range of the image stream, or None for a still image
                               that is the same at every frame.
    """

    def __init__(self, width: int, height: int, firstFrame: Optional[int] = None, lastFrame: Optional[int] = None):
        self.width = int(width)
        self.height = int(height)
        self.firstFrame = firstFrame
        self.lastFrame = lastFrame

    @property
    def hasFrameRange(self) -> bool:
        return self.firstFrame is not None and self.lastFrame is not None

    def withFormat(self, width: int, height: int) -> "ImageInfo":
        return ImageInfo(width, height, self.firstFrame, self.lastFrame)

    def __eq__(self, other):
        return isinstance(other, ImageInfo) and \
            (self.width, self.height, self.firstFrame, self.lastFrame) == \
            (other.width, other.height, other.firstFrame, other.lastFrame)

    def __repr__(self):
        return f"ImageInfo({self.width}x{self.height}, frames={self.firstFrame}-{self.lastFrame})"


def _freeze(pixels) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.ndim == 2:
        pixels = pixels[:, :, np.newaxis]
    if pixels.flags.writeable:
        # A later write raises an error instead of silently altering a cached image
        pixels.flags.writeable = False
    return pixels


class ImageBuffer:
    """
    An image made of named layers (Nuke "layers", EXR "AOVs") sharing the same format and bbox.

    Each layer is a float32 numpy array of shape (bbox height, bbox width, nbChannels), row 0
    being the top of the bbox, associated with its channel names (e.g. "rgba" -> ("R", "G", "B", "A"),
    "depth" -> ("Z",), "diffuse" -> ("R", "G", "B")).

    An ImageBuffer is immutable: its arrays are read-only since cached buffers are shared
    between nodes. Operations create new buffers (`withLayer`, `withRGBA`, `withPixels`, ...),
    which share the arrays of the layers they do not change.
    """

    def __init__(self, layers: "OrderedDict[str, Tuple[Sequence[str], np.ndarray]]",
                 format: Optional[Tuple[int, int]] = None, bbox: Optional[BBox] = None,
                 metadata: Optional[dict] = None):
        if not layers:
            raise ValueError("An ImageBuffer needs at least one layer.")
        self._layers: "OrderedDict[str, Tuple[Tuple[str, ...], np.ndarray]]" = OrderedDict()
        shape = None
        for name, (channels, pixels) in layers.items():
            pixels = _freeze(pixels)
            if pixels.ndim != 3 or pixels.shape[2] != len(channels):
                raise ValueError(f"Layer '{name}': {len(channels)} channel names for pixels of shape {pixels.shape}.")
            if shape is None:
                shape = pixels.shape[:2]
            elif pixels.shape[:2] != shape:
                raise ValueError(f"Layer '{name}' has size {pixels.shape[:2]}, expected {shape}.")
            self._layers[name] = (tuple(channels), pixels)
        if bbox is None:
            bbox = BBox(0, 0, shape[1], shape[0])
        bbox = BBox(*(int(v) for v in bbox))
        if (bbox.height, bbox.width) != tuple(shape):
            raise ValueError(f"The bbox {tuple(bbox)} does not match the pixels size {shape}.")
        self._bbox = bbox
        self._format = (int(format[0]), int(format[1])) if format is not None else (bbox.x1, bbox.y1)
        self.metadata = dict(metadata or {})

    # Construction helpers

    @classmethod
    def fromRGBA(cls, pixels: np.ndarray, format: Optional[Tuple[int, int]] = None, bbox: Optional[BBox] = None,
                 metadata: Optional[dict] = None) -> "ImageBuffer":
        """ Create an image with a single "rgba" layer from an array of shape (h, w, 1 to 4). """
        pixels = np.asarray(pixels, dtype=np.float32)
        if pixels.ndim == 2:
            pixels = pixels[:, :, np.newaxis]
        return cls(OrderedDict([(RGBA, (RGBA_CHANNELS[:pixels.shape[2]], pixels))]), format, bbox, metadata)

    @classmethod
    def constant(cls, width: int, height: int, color: Sequence[float]) -> "ImageBuffer":
        """ Create an image filled with a constant RGBA color, its bbox being its format. """
        pixels = np.empty((int(height), int(width), 4), dtype=np.float32)
        pixels[:, :] = np.asarray(color, dtype=np.float32)[:4]
        return cls.fromRGBA(pixels, (width, height))

    # Accessors

    @property
    def width(self) -> int:
        """ Format width. """
        return self._format[0]

    @property
    def height(self) -> int:
        """ Format height. """
        return self._format[1]

    @property
    def format(self) -> Tuple[int, int]:
        return self._format

    @property
    def bbox(self) -> BBox:
        return self._bbox

    @property
    def layerNames(self) -> List[str]:
        return list(self._layers.keys())

    def hasLayer(self, name: str) -> bool:
        return name in self._layers

    def channels(self, layer: str) -> Tuple[str, ...]:
        return self._layers[layer][0]

    def layer(self, name: str) -> np.ndarray:
        """ Return the (read-only) pixels of a layer, of shape (bbox height, bbox width, nbChannels). """
        return self._layers[name][1]

    def layers(self) -> Iterable[Tuple[str, Tuple[str, ...], np.ndarray]]:
        for name, (channels, pixels) in self._layers.items():
            yield name, channels, pixels

    def arrays(self) -> List[np.ndarray]:
        """ The pixel arrays of the layers (used to count the memory shared between images). """
        return [pixels for _, pixels in self._layers.values()]

    def rgba(self) -> np.ndarray:
        """
        Return the main layer as a new (h, w, 4) array over the bbox.
        Missing channels are 0 (like in Nuke: an image without alpha has a zero alpha).
        A single-channel main layer is returned as grey.
        """
        out = np.zeros((self._bbox.height, self._bbox.width, 4), dtype=np.float32)
        if RGBA in self._layers:
            channels, pixels = self._layers[RGBA]
            if channels == RGBA_CHANNELS:
                out[:] = pixels
                return out
            for i, channel in enumerate(channels):
                if channel in RGBA_CHANNELS:
                    out[:, :, RGBA_CHANNELS.index(channel)] = pixels[:, :, i]
            if channels == ("R",):
                out[:, :, 1] = out[:, :, 0]
                out[:, :, 2] = out[:, :, 0]
        return out

    @property
    def nbytes(self) -> int:
        return sum(pixels.nbytes for _, pixels in self._layers.values())

    @property
    def info(self) -> ImageInfo:
        return ImageInfo(self.width, self.height)

    # Derivation helpers

    def withLayer(self, name: str, pixels: np.ndarray, channels: Optional[Sequence[str]] = None) -> "ImageBuffer":
        """ Return a new image where the layer `name` is replaced (or added); same bbox. """
        if channels is None:
            channels = self._layers[name][0] if name in self._layers else RGBA_CHANNELS[:np.shape(pixels)[-1]]
        layers = OrderedDict(self._layers)
        layers[name] = (tuple(channels), pixels)
        return ImageBuffer(layers, self._format, self._bbox, self.metadata)

    def withRGBA(self, pixels: np.ndarray) -> "ImageBuffer":
        """ Return a new image whose main layer is replaced by `pixels` of shape (h, w, 4); same bbox. """
        return self.withLayer(RGBA, pixels, RGBA_CHANNELS)

    def withPixels(self, layers: "OrderedDict[str, Tuple[Sequence[str], np.ndarray]]", bbox: BBox,
                   format: Optional[Tuple[int, int]] = None) -> "ImageBuffer":
        """ Return a new image with other layers and bbox, keeping the format (unless given) and metadata. """
        return ImageBuffer(layers, format or self._format, bbox, self.metadata)

    def mapLayers(self, func, layerNames: Optional[Iterable[str]] = None, bbox: Optional[BBox] = None) -> "ImageBuffer":
        """
        Return a new image where `func(pixels) -> pixels` is applied to the given layers
        (all layers by default).

        Args:
            bbox: the new bbox when `func` changes the size of the pixels; then every layer
                  must be processed.
        """
        selected = set(self._layers.keys() if layerNames is None else layerNames)
        if bbox is not None and selected != set(self._layers.keys()):
            raise ValueError("Changing the bbox requires processing every layer.")
        layers = OrderedDict()
        for name, (channels, pixels) in self._layers.items():
            layers[name] = (channels, func(pixels) if name in selected else pixels)
        return ImageBuffer(layers, self._format, bbox or self._bbox, self.metadata)

    def __repr__(self):
        return f"ImageBuffer(format={self.width}x{self.height}, bbox={tuple(self._bbox)}, layers={self.layerNames})"


def splitChannelNames(channelNames: Sequence[str]) -> "OrderedDict[str, List[Tuple[int, str]]]":
    """
    Group file channel names into layers, following the EXR / OpenImageIO conventions:
    "R", "G", "B", "A" -> "rgba" ; "Z" -> "depth" ; "diffuse.R" -> "diffuse" ; other names -> "other".

    Returns:
        OrderedDict {layer name: [(channel index in the file, channel name in the layer), ...]}
    """
    groups: "OrderedDict[str, List[Tuple[int, str]]]" = OrderedDict()
    for index, fullName in enumerate(channelNames):
        if "." in fullName:
            layer, _, channel = fullName.rpartition(".")
        elif fullName.upper() in RGBA_CHANNELS:
            layer, channel = RGBA, fullName.upper()
        elif fullName.upper() == "Z":
            layer, channel = "depth", "Z"
        else:
            layer, channel = "other", fullName
        groups.setdefault(layer, []).append((index, channel))
    return groups
