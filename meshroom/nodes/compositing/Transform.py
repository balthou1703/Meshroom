__version__ = "0.2"

from meshroom.core import desc
from meshroom.live.params import linkedVec2


class Transform(desc.ImageOpNode):
    category = "Comp Transform"
    # Parameters in pixels, scaled in the proxy renders
    proxyScaled = ('translate', 'center')
    color = "#A57AAA"
    documentation = """
2D affine transform of every layer: translate, rotate, scale and skew around a center.
Coordinates are in pixels from the bottom-left corner, y up, as in Nuke.
The output keeps the input format; its bbox follows the transformed image (overscan is kept,
within the format extended by its own size on each side). A translation by whole pixels only
moves the bbox, without resampling.
"""

    inputs = [
        desc.Image(
            name="input",
            label="Input",
            description="Image to transform.",
        ),
        desc.Vec2d(
            name="translate",
            label="Translate",
            description="Translation in pixels.",
            x=0.0,
            y=0.0,
            xRange=(-2000.0, 2000.0, 1.0),
            yRange=(-2000.0, 2000.0, 1.0),
        ),
        desc.FloatParam(
            name="rotate",
            label="Rotate",
            description="Rotation in degrees, counter-clockwise.",
            value=0.0,
            range=(-180.0, 180.0, 0.1),
        ),
        linkedVec2("scale", "Scale", "Scale factor (the '+' button edits x and y separately).", 1.0, (0.0, 4.0, 0.01)),
        desc.FloatParam(
            name="skewX",
            label="Skew X",
            description="Horizontal skew angle in degrees.",
            value=0.0,
            range=(-60.0, 60.0, 0.1),
        ),
        desc.BoolParam(
            name="centerOnImage",
            label="Center On Image",
            description="Rotate and scale around the image center. If disabled, use 'Center'.",
            value=True,
        ),
        desc.Vec2d(
            name="center",
            label="Center",
            description="Center of rotation and scale, in pixels.",
            x=0.0,
            y=0.0,
            xRange=(-2000.0, 4000.0, 1.0),
            yRange=(-2000.0, 4000.0, 1.0),
            enabled=lambda node: not node.centerOnImage.value,
        ),
        desc.BoolParam(
            name="invert",
            label="Invert",
            description="Apply the inverse transform.",
            value=False,
        ),
        desc.ChoiceParam(
            name="filter",
            label="Filter",
            description="Pixel filter: 'impulse' (nearest pixel) or 'bilinear'.",
            value="bilinear",
            values=["impulse", "bilinear"],
        ),
    ]

    outputs = [
        desc.Image(
            name="output",
            label="Output",
            description="Transformed image.",
        ),
    ]

    def inputRegions(self, ctx, region):
        """ Region of interest: the part of the input the region comes from (inverse transform, plus the filter). """
        import numpy as np
        from meshroom.live import ops
        info = ctx.inputInfo("input")
        if info is None:
            return None
        try:
            inverse = np.linalg.inv(self._matrix(ctx, info))
        except np.linalg.LinAlgError:
            return None
        return {"input": ops.transformedBBox(region, inverse).expanded(2, 2)}

    def _matrix(self, ctx, info):
        import numpy as np
        from meshroom.live import ops
        translate, scale = ctx.value("translate"), ctx.value("scale")
        if ctx.value("centerOnImage"):
            center = (info.width / 2.0, info.height / 2.0)
        else:
            center = (ctx.value("center")["x"], ctx.value("center")["y"])
        matrix = ops.transformMatrix(translate=(translate["x"], translate["y"]), rotate=ctx.value("rotate"),
                                     scale=(scale["x"], scale["y"]), skewX=ctx.value("skewX"), center=center)
        if ctx.value("invert"):
            matrix = np.linalg.inv(matrix)
        return matrix

    def isIdentity(self, ctx):
        import numpy as np
        info = ctx.inputInfo("input")
        if info is None:
            return None
        return "input" if np.allclose(self._matrix(ctx, info), np.eye(3)) else None

    def render(self, ctx, inputs):
        from meshroom.live import BBox, ops
        image = inputs["input"]
        matrix = self._matrix(ctx, image)
        shift = ops.integerTranslation(matrix)
        if shift is not None:
            # Whole pixels: the layers are shared, only the bbox moves
            return image.mapLayers(lambda pixels: pixels, bbox=image.bbox.translated(*shift))
        # Limit the overscan to the format extended by its size on each side
        limit = BBox(-image.width, -image.height, 2 * image.width, 2 * image.height)
        bbox = ops.transformedBBox(image.bbox, matrix).clampedTo(limit)
        return image.mapLayers(
            lambda pixels: ops.warpAffine(pixels, image.bbox, matrix, bbox, ctx.value("filter"), ctx.checkCancelled),
            bbox=bbox)
