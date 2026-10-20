"""
Filter nodes (Nuke "Filter" menu): Sharpen, Erode (FilterErode), Median.
"""
__version__ = "0.1"

from collections import OrderedDict

import numpy as np

from meshroom.core import desc
from meshroom.live.gpu import gpuParam
from meshroom.live.params import colorOpInputs, imageOutput, mixParam

try:
    from scipy import ndimage as _ndimage
except ImportError:  # pragma: no cover - depends on the environment
    _ndimage = None


def _float(name, label, description, value, range):
    return desc.FloatParam(name=name, label=label, description=description, value=value, range=range)


def _extremumAxis(pixels, radius, axis, useMax):
    """ Minimum or maximum over a window of 2r+1 pixels along an axis (black outside). """
    if radius <= 0:
        return pixels
    if _ndimage is not None:
        function = _ndimage.maximum_filter1d if useMax else _ndimage.minimum_filter1d
        return function(pixels, 2 * radius + 1, axis=axis, mode="constant", cval=0.0)
    padWidth = [(0, 0)] * pixels.ndim
    padWidth[axis] = (radius, radius)
    padded = np.pad(pixels, padWidth, mode="constant")
    length = pixels.shape[axis]
    out = np.take(padded, range(0, length), axis=axis).copy()
    reduce = np.maximum if useMax else np.minimum
    for offset in range(1, 2 * radius + 1):
        reduce(out, np.take(padded, range(offset, offset + length), axis=axis), out=out)
    return out


def _blendSelected(ctx, inputs, image, rgba, result):
    """ Keep `result` on the selected channels, blended with `rgba` by the mix and the mask. """
    from meshroom.live import RGBA_CHANNELS, ops
    from meshroom.live.params import maskAmount
    selected = ops.channelMask(RGBA_CHANNELS, ctx.value("channels"))
    out = rgba.copy()
    out[:, :, selected] = result[:, :, selected]
    amount = maskAmount(ctx, inputs, image.bbox, ctx.value("mix"))
    return image.withRGBA(ops.mapRows(ops.mix, [rgba, out, amount], 4, ctx.checkCancelled))


class Sharpen(desc.ImageOpNode):
    category = "Comp Filter"
    # Parameters in pixels, scaled in the proxy renders
    proxyScaled = ('size',)
    color = "#CC804E"
    documentation = """
Unsharp mask: adds the difference between the image and its blur, multiplied by "Amount".
"""
    inputs = colorOpInputs() + [
        _float("amount", "Amount", "Strength of the sharpening.", 0.0, (0.0, 4.0, 0.01)),
        _float("size", "Size", "Size of the blur the image is compared to, in pixels.", 3.0, (0.0, 50.0, 0.1)),
        mixParam(),
        gpuParam(),
    ]
    outputs = imageOutput("Sharpened image.")

    def inputRegions(self, ctx, region):
        """ Region of interest: the input around the region, by the radius of the blur compared to. """
        from meshroom.live import ops
        margin = ops.blurRadius(ctx.value("size")) + 1
        return {"input": region.expanded(margin, margin), "mask": region}

    def isIdentity(self, ctx):
        if ctx.value("amount") == 0.0 or ctx.value("size") <= 0.0 or ctx.value("mix") <= 0.0:
            return "input"
        return None

    def render(self, ctx, inputs):
        from meshroom.live import gpu, ops
        image = inputs["input"]
        amount, size = np.float32(ctx.value("amount")), ctx.value("size")
        rgba = image.rgba()
        radius = ops.blurRadius(size)
        blurred = ops.blurOnDevice(rgba, size, size, gpu.enabled(ctx), ctx.checkCancelled)
        # Back to the input bbox
        blurred = blurred[radius:radius + rgba.shape[0], radius:radius + rgba.shape[1]]
        return _blendSelected(ctx, inputs, image, rgba, rgba + amount * (rgba - blurred))


class Erode(desc.ImageOpNode):
    category = "Comp Filter"
    # Parameters in pixels, scaled in the proxy renders
    proxyScaled = ('size',)
    color = "#CC804E"
    documentation = """
Erodes (positive size) or dilates (negative size) the selected channels with a square of
2 x |size| + 1 pixels, like the Nuke FilterErode (box). A dilation grows the bbox.
"""
    inputs = colorOpInputs(defaultChannels="alpha", channelValues=("alpha", "rgba", "rgb")) + [
        _float("size", "Size", "Positive: erode (shrink), negative: dilate (grow), in pixels.", 0.0,
               (-50.0, 50.0, 1.0)),
        mixParam(),
        gpuParam(),
    ]
    outputs = imageOutput("Eroded or dilated image.")

    def inputRegions(self, ctx, region):
        """ Region of interest: the input around the region, by the size of the filter. """
        margin = int(round(abs(ctx.value("size")))) + 1
        return {"input": region.expanded(margin, margin), "mask": region}

    def isIdentity(self, ctx):
        if int(round(abs(ctx.value("size")))) == 0 or ctx.value("mix") <= 0.0:
            return "input"
        return None

    def render(self, ctx, inputs):
        from meshroom.live import RGBA_CHANNELS, ops
        from meshroom.live.params import maskAmount
        image = inputs["input"]
        size = ctx.value("size")
        radius = int(round(abs(size)))
        dilate = size < 0
        bbox = image.bbox.expanded(radius, radius) if dilate else image.bbox
        rgba = ops.alignToBBox(image.rgba(), image.bbox, bbox)
        selected = ops.channelMask(RGBA_CHANNELS, ctx.value("channels"))

        from meshroom.live import gpu, gpuOps
        filtered = rgba.copy()
        channels = np.ascontiguousarray(rgba[:, :, selected])

        def onCpu():
            def horizontal(start, end):
                channels[start:end] = _extremumAxis(channels[start:end], radius, 1, dilate)

            def vertical(start, end):
                channels[:, start:end] = _extremumAxis(channels[:, start:end], radius, 0, dilate)

            ops.parallelBands(horizontal, channels.shape[0], ctx.checkCancelled)
            ops.parallelBands(vertical, channels.shape[1], ctx.checkCancelled)
            return channels
        filtered[:, :, selected] = gpu.run(gpu.enabled(ctx), lambda device: gpuOps.extremum(channels, radius, dilate, device),
                                           onCpu, "Erode")
        amount = maskAmount(ctx, inputs, bbox, ctx.value("mix"))
        result = ops.mapRows(ops.mix, [rgba, filtered, amount], 4, ctx.checkCancelled)
        # The other layers follow the bbox
        if bbox == image.bbox:
            return image.withRGBA(result)
        layers = OrderedDict((name, (layerChannels, ops.alignToBBox(pixels, image.bbox, bbox)))
                             for name, layerChannels, pixels in image.layers())
        layers["rgba"] = (RGBA_CHANNELS, result)
        return image.withPixels(layers, bbox)


class Median(desc.ImageOpNode):
    category = "Comp Filter"
    # Parameters in pixels, scaled in the proxy renders
    proxyScaled = ('size',)
    color = "#CC804E"
    documentation = """
Median filter over a square of 2 x size + 1 pixels: removes noise and isolated pixels while keeping
the edges. Requires scipy. Large sizes are slow on large images.
"""
    inputs = colorOpInputs(defaultChannels="rgba") + [
        desc.IntParam(name="size", label="Size", description="Radius of the filter in pixels.", value=0,
                      range=(0, 10, 1)),
        mixParam(),
        gpuParam(),
    ]
    outputs = imageOutput("Filtered image.")

    def inputRegions(self, ctx, region):
        """ Region of interest: the input around the region, by the size of the filter. """
        margin = int(ctx.value("size")) + 1
        return {"input": region.expanded(margin, margin), "mask": region}

    def isIdentity(self, ctx):
        return "input" if ctx.value("size") <= 0 or ctx.value("mix") <= 0.0 else None

    def render(self, ctx, inputs):
        from meshroom.live import gpu, gpuOps, ops
        width = 2 * ctx.value("size") + 1
        image = inputs["input"]
        rgba = image.rgba()
        overlap = width // 2

        def onCpu():
            if _ndimage is None:
                raise RuntimeError("The Median node requires scipy (or a GPU).")
            result = np.empty_like(rgba)

            def band(start, end):
                # Each band is filtered with the rows around it, then cropped
                low, high = max(0, start - overlap), min(rgba.shape[0], end + overlap)
                filtered = _ndimage.median_filter(rgba[low:high], size=(width, width, 1), mode="constant", cval=0.0)
                result[start:end] = filtered[start - low:start - low + (end - start)]

            ops.parallelBands(band, rgba.shape[0], ctx.checkCancelled)
            return result
        result = gpu.run(gpu.enabled(ctx), lambda device: gpuOps.median(rgba, width, device), onCpu, "Median")
        return _blendSelected(ctx, inputs, image, rgba, result)
