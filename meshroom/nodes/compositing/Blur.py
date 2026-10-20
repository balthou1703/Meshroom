__version__ = "0.3"

from collections import OrderedDict

import numpy as np

from meshroom.core import Version, desc
from meshroom.live.gpu import gpuParam
from meshroom.live.params import linkedVec2, maskParams


class Blur(desc.ImageOpNode):
    category = "Comp Filter"
    # Parameters in pixels, scaled in the proxy renders
    proxyScaled = ('size',)
    color = "#CC804E"
    documentation = """
Gaussian blur. "Size" is the width of the filter in pixels (approximation of the Nuke Blur size).
Large sizes use three successive box blurs, whose cost does not depend on the size.
The bbox grows by the blur radius, as in Nuke. A size of 0 passes the input through.
"""

    inputs = [
        desc.Image(
            name="input",
            label="Input",
            description="Image to blur.",
        ),
        desc.Image(
            name="mask",
            label="Mask",
            description="Optional mask: its 'Mask Channel' limits where the blur applies.",
            optional=True,
        ),
        *maskParams(),
        desc.ChoiceParam(
            name="channels",
            label="Channels",
            description="Layers to blur: the 'rgba' layer only, or every layer.",
            value="rgba",
            values=["rgba", "all"],
        ),
        linkedVec2("size", "Size", "Blur size in pixels (the '+' button edits x and y separately).", 0.0,
                   (0.0, 100.0, 0.1)),
        desc.FloatParam(
            name="mix",
            label="Mix",
            description="Blend between the input (0) and the blurred image (1).",
            value=1.0,
            range=(0.0, 1.0, 0.01),
        ),
        gpuParam(),
    ]

    outputs = [
        desc.Image(
            name="output",
            label="Output",
            description="Blurred image.",
        ),
    ]

    def inputRegions(self, ctx, region):
        """ Region of interest: the input around the region, by the radius of the blur. """
        from meshroom.live import ops
        sizeX, sizeY = self._sizes(ctx)
        return {"input": region.expanded(ops.blurRadius(sizeX), ops.blurRadius(sizeY)), "mask": region}

    def upgradeAttributeValues(self, attrValues, fromVersion):
        # Before 0.3, "size" was a float and "sizeY" a float (negative: same as size)
        if fromVersion < Version("0.3"):
            size = attrValues.get("size")
            if isinstance(size, (int, float)):
                sizeY = attrValues.pop("sizeY", -1.0)
                attrValues["size"] = {"x": float(size), "y": float(size if sizeY is None or sizeY < 0 else sizeY)}
        return attrValues

    def _sizes(self, ctx):
        size = ctx.value("size")
        return max(0.0, size["x"]), max(0.0, size["y"])

    def isIdentity(self, ctx):
        sizeX, sizeY = self._sizes(ctx)
        if (sizeX <= 0 and sizeY <= 0) or ctx.value("mix") <= 0.0:
            return "input"
        return None

    def render(self, ctx, inputs):
        from meshroom.live import RGBA, gpu, ops
        from meshroom.live.params import maskAmount
        image = inputs["input"]
        sizeX, sizeY = self._sizes(ctx)
        useGpu = gpu.enabled(ctx)
        allLayers = ctx.value("channels") == "all"
        bbox = image.bbox.expanded(ops.blurRadius(sizeX), ops.blurRadius(sizeY))
        amount = maskAmount(ctx, inputs, bbox, ctx.value("mix"))

        def apply(pixels):
            blurred = ops.blurOnDevice(pixels, sizeX, sizeY, useGpu, ctx.checkCancelled)
            if not (np.isscalar(amount) and amount >= 1.0):
                original = ops.alignToBBox(pixels, image.bbox, bbox)
                blurred = ops.mapRows(ops.mix, [original, blurred, amount], pixels.shape[2], ctx.checkCancelled)
            return blurred

        def extend(pixels):
            return ops.alignToBBox(pixels, image.bbox, bbox)

        # The bbox grows for every layer: the layers that are not blurred are extended with black
        layers = []
        for name, channels, pixels in image.layers():
            layers.append((name, channels, apply(pixels) if allLayers or name == RGBA else extend(pixels)))
        return image.withPixels(OrderedDict((name, (channels, pixels)) for name, channels, pixels in layers), bbox)
