__version__ = "0.2"

from collections import OrderedDict

import numpy as np

from meshroom.core import desc
from meshroom.live import ImageInfo, RGBA, RGBA_CHANNELS
from meshroom.live.imageBuffer import deferred
from meshroom.live.ops import MERGE_IDENTITY_WITHOUT_A, MERGE_OPERATIONS
from meshroom.live.params import maskParams


class Merge(desc.ImageOpNode):
    category = "Comp Merge"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#4B5EC6"
    documentation = """
Combines the "rgba" layers of two premultiplied images with a Nuke Merge operation
(`over`: A + B * (1 - a), ...). "A" is the foreground, "B" the background.

The output has the format and the other layers of B (of A when B is not connected).
Its bbox is the union of the A and B bboxes by default.
A mix of 0, or A not connected with an operation leaving B unchanged, passes B through.
"""

    inputs = [
        desc.Image(
            name="B",
            label="B",
            description="Background image.",
            optional=True,
        ),
        desc.Image(
            name="A",
            label="A",
            description="Foreground image.",
            optional=True,
        ),
        desc.Image(
            name="mask",
            label="Mask",
            description="Optional mask: its 'Mask Channel' limits where the merge applies.",
            optional=True,
        ),
        *maskParams(),
        desc.ChoiceParam(
            name="operation",
            label="Operation",
            description="Merge operation.",
            value="over",
            values=list(MERGE_OPERATIONS),
        ),
        desc.ChoiceParam(
            name="bbox",
            label="Set BBox To",
            description="Bounding box of the result: union of A and B, B, A, or their intersection.",
            value="union",
            values=["union", "B", "A", "intersection"],
        ),
        desc.FloatParam(
            name="mix",
            label="Mix",
            description="Blend between B (0) and the merged result (1).",
            value=1.0,
            range=(0.0, 1.0, 0.01),
        ),
    ]

    outputs = [
        desc.Image(
            name="output",
            label="Output",
            description="Merged image.",
        ),
    ]

    def getInfo(self, ctx, inputsInfo):
        info = inputsInfo.get("B") or inputsInfo.get("A")
        if info is None:
            raise RuntimeError("Connect at least one of the A and B inputs.")
        other = inputsInfo.get("A") if inputsInfo.get("B") else None
        if other is None or not other.hasFrameRange:
            return info
        if not info.hasFrameRange:
            return ImageInfo(info.width, info.height, other.firstFrame, other.lastFrame)
        # The merged stream spans both frame ranges
        return ImageInfo(info.width, info.height, min(info.firstFrame, other.firstFrame),
                         max(info.lastFrame, other.lastFrame))

    def isIdentity(self, ctx):
        hasA = ctx.planNode.inputs.get("A") is not None
        hasB = ctx.planNode.inputs.get("B") is not None
        operation = ctx.value("operation")
        if hasB and (ctx.value("mix") <= 0.0 or (not hasA and operation in MERGE_IDENTITY_WITHOUT_A)):
            return "B"
        if hasA and not hasB and operation in ("over", "under", "plus", "minus", "screen", "xor", "copy"):
            return "A"
        return None

    def _outputBBox(self, ctx, imageA, imageB):
        if imageA is None:
            return imageB.bbox
        if imageB is None:
            return imageA.bbox
        mode = ctx.value("bbox")
        if mode == "B":
            return imageB.bbox
        if mode == "A":
            return imageA.bbox
        if mode == "intersection":
            return imageA.bbox.intersection(imageB.bbox)
        return imageA.bbox.union(imageB.bbox)

    def render(self, ctx, inputs):
        from meshroom.live import ops
        from meshroom.live.params import maskAmount
        imageB, imageA = inputs.get("B"), inputs.get("A")
        if imageB is None and imageA is None:
            raise RuntimeError("Connect at least one of the A and B inputs.")
        base = imageB if imageB is not None else imageA
        bbox = self._outputBBox(ctx, imageA, imageB)
        shape = (bbox.height, bbox.width, 4)
        b = ops.alignToBBox(imageB.rgba(), imageB.bbox, bbox) if imageB is not None else np.zeros(shape, dtype=np.float32)
        a = ops.alignToBBox(imageA.rgba(), imageA.bbox, bbox) if imageA is not None else np.zeros(shape, dtype=np.float32)
        operation = ctx.value("operation")
        amount = maskAmount(ctx, inputs, bbox, ctx.value("mix"))

        def band(aBand, bBand, amountBand):
            return ops.mix(bBand, ops.merge(aBand, bBand, operation), amountBand)
        merged = ops.mapRows(band, [a, b, amount], 4, ctx.checkCancelled)

        # The other layers of the base image follow the new bbox (when they are used: deferred)
        layers = OrderedDict()
        for name, channels, pixels in base.rawLayers():
            if name == RGBA:
                # Its place kept (replaced below)
                layers[name] = None
                continue
            layers[name] = (channels, pixels if tuple(base.bbox) == tuple(bbox) else
                            deferred(lambda p, src=base.bbox: ops.alignToBBox(p, src, bbox), pixels,
                                     (bbox.height, bbox.width, len(channels))))
        layers[RGBA] = (RGBA_CHANNELS, merged)
        return base.withPixels(layers, bbox)
