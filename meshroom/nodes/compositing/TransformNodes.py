"""
Transformation nodes (Nuke "Transform" menu): Reformat, Crop, Mirror.
"""
__version__ = "0.2"

from collections import OrderedDict

import numpy as np

from meshroom.core import desc
from meshroom.live import BBox, ImageInfo
from meshroom.live.params import imageInput, imageOutput, linkedVec2


def _float(name, label, description, value, range, enabled=True):
    return desc.FloatParam(name=name, label=label, description=description, value=value, range=range,
                           enabled=enabled)


class Reformat(desc.ImageOpNode):
    category = "Comp Transform"
    color = "#A57AAA"
    documentation = """
Resizes the image to another format, or scales it.
- To Format: the image is resized to Width x Height, or to the format of the optional "Format" input
  when it is connected; "Resize" chooses how its aspect ratio is handled (distort, fit inside, fill);
- Scale: the format and the image are multiplied by "Scale".
Downscaling applies a blur first to avoid aliasing. "Crop" removes the overscan: the image is cut to
the output format.
"""
    inputs = [
        imageInput(),
        desc.Image(name="formatInput", label="Format", optional=True, infoOnly=True,
                   description="Optional: 'To Format' resizes to the format of this image (its pixels are not used)."),
        desc.ChoiceParam(name="reformatType", label="Type", description="Resize to a format, or scale.", value="to format",
                         values=["to format", "scale"]),
        desc.IntParam(name="width", label="Width", description="Output format width.", value=1920,
                      range=(1, 16384, 1), enabled=lambda node: node.reformatType.value == "to format"),
        desc.IntParam(name="height", label="Height", description="Output format height.", value=1080,
                      range=(1, 16384, 1), enabled=lambda node: node.reformatType.value == "to format"),
        desc.ChoiceParam(name="resize", label="Resize", value="fit", values=["distort", "fit", "fill"],
                         description="distort: stretch to the format; fit: whole image inside; fill: cover the format.",
                         enabled=lambda node: node.reformatType.value == "to format"),
        linkedVec2("scale", "Scale", "Scale factor.", 1.0, (0.01, 4.0, 0.01)),
        desc.ChoiceParam(name="filter", label="Filter", description="Pixel filter.", value="bilinear",
                         values=["impulse", "bilinear"]),
        desc.BoolParam(name="crop", label="Crop", description="Cut the image to the output format (no overscan).",
                       value=False),
    ]
    outputs = imageOutput("Resized image.")

    def _geometry(self, ctx, info):
        """ Return (output width, output height, scale x, scale y, offset x, offset y). """
        if ctx.value("reformatType") == "scale":
            scale = ctx.value("scale")
            sx, sy = max(scale["x"], 1e-4), max(scale["y"], 1e-4)
            return max(1, int(round(info.width * sx))), max(1, int(round(info.height * sy))), sx, sy, 0.0, 0.0
        formatInfo = ctx.inputInfo("formatInput")
        if formatInfo is not None:
            width, height = formatInfo.width, formatInfo.height
        else:
            width, height = ctx.value("width"), ctx.value("height")
        sx, sy = width / info.width, height / info.height
        resize = ctx.value("resize")
        if resize == "fit":
            sx = sy = min(sx, sy)
        elif resize == "fill":
            sx = sy = max(sx, sy)
        # Centered in the output format
        return width, height, sx, sy, (width - info.width * sx) / 2.0, (height - info.height * sy) / 2.0

    def getInfo(self, ctx, inputsInfo):
        info = inputsInfo["input"]
        width, height = self._geometry(ctx, info)[:2]
        return ImageInfo(width, height, info.firstFrame, info.lastFrame)

    def isIdentity(self, ctx):
        info = ctx.inputInfo("input")
        if info is None:
            return None
        width, height, sx, sy, ox, oy = self._geometry(ctx, info)
        if ctx.value("crop"):
            return None
        return "input" if (width, height) == (info.width, info.height) and sx == sy == 1 and ox == oy == 0 else None

    def render(self, ctx, inputs):
        from meshroom.live import ops
        image = inputs["input"]
        width, height, sx, sy, ox, oy = self._geometry(ctx, image)
        matrix = np.array([[sx, 0, ox], [0, sy, oy], [0, 0, 1]], dtype=np.float64)
        limit = BBox(0, 0, width, height) if ctx.value("crop") else BBox(-width, -height, 2 * width, 2 * height)
        bbox = ops.transformedBBox(image.bbox, matrix).clampedTo(limit)
        filter = ctx.value("filter")

        def resize(pixels):
            if sx < 1.0 or sy < 1.0:
                # Low-pass before downscaling, then back to the input bbox
                blurX, blurY = (1.0 / sx if sx < 1 else 0.0), (1.0 / sy if sy < 1 else 0.0)
                rx, ry = ops.blurRadius(blurX), ops.blurRadius(blurY)
                pixels = ops.blur(pixels, blurX, blurY, ctx.checkCancelled)[ry:ry + image.bbox.height,
                                                                           rx:rx + image.bbox.width]
            return ops.warpAffine(pixels, image.bbox, matrix, bbox, filter, ctx.checkCancelled)

        layers = OrderedDict((name, (channels, resize(pixels))) for name, channels, pixels in image.layers())
        return image.withPixels(layers, bbox, format=(width, height))


class Crop(desc.ImageOpNode):
    category = "Comp Transform"
    # Region of interest: the same region of the inputs
    pointwise = True
    # Parameters in pixels, scaled in the proxy renders
    proxyScaled = ('left', 'bottom', 'right', 'top', 'softness')
    color = "#A57AAA"
    documentation = """
Removes everything outside of a box (pixels, from the bottom-left corner). "Right" and "Top" at 0
use the format width and height. "Softness" fades the edges. With "Reformat", the output format
becomes the box.
"""
    inputs = [
        imageInput(),
        _float("left", "Left", "Left edge of the box.", 0.0, (-1000.0, 8000.0, 1.0)),
        _float("bottom", "Bottom", "Bottom edge of the box.", 0.0, (-1000.0, 8000.0, 1.0)),
        _float("right", "Right", "Right edge of the box (0: format width).", 0.0, (0.0, 8000.0, 1.0)),
        _float("top", "Top", "Top edge of the box (0: format height).", 0.0, (0.0, 8000.0, 1.0)),
        _float("softness", "Softness", "Width of the fade at the edges, in pixels.", 0.0, (0.0, 200.0, 0.5)),
        desc.BoolParam(name="reformat", label="Reformat", description="The output format becomes the box.",
                       value=False),
    ]
    outputs = imageOutput("Cropped image.")

    def inputRegions(self, ctx, region):
        """ Region of interest: the same region of the input (moved back when the crop box becomes the format). """
        if ctx.value("reformat"):
            left, bottom = int(np.floor(ctx.value("left"))), int(np.floor(ctx.value("bottom")))
            return {"input": region.translated(left, bottom)}
        return {"input": region}

    def _box(self, ctx, info):
        right = ctx.value("right") if ctx.value("right") > 0 else info.width
        top = ctx.value("top") if ctx.value("top") > 0 else info.height
        return ctx.value("left"), ctx.value("bottom"), right, top

    def getInfo(self, ctx, inputsInfo):
        info = inputsInfo["input"]
        if not ctx.value("reformat"):
            return info
        left, bottom, right, top = self._box(ctx, info)
        return ImageInfo(max(1, int(round(right - left))), max(1, int(round(top - bottom))),
                         info.firstFrame, info.lastFrame)

    def render(self, ctx, inputs):
        from meshroom.live import ops
        image = inputs["input"]
        left, bottom, right, top = self._box(ctx, image)
        box = BBox(int(np.floor(left)), int(np.floor(bottom)), int(np.ceil(right)), int(np.ceil(top)))
        bbox = image.bbox.intersection(box)
        softness = ctx.value("softness")

        # Fade of each pixel center: 1 inside, linear over "softness" pixels inside the box edges
        xs = bbox.x0 + np.arange(bbox.width, dtype=np.float32) + 0.5
        ys = bbox.y1 - (np.arange(bbox.height, dtype=np.float32) + 0.5)
        if softness > 0:
            fadeX = np.clip(np.minimum(xs - left, right - xs) / softness, 0, 1)
            fadeY = np.clip(np.minimum(ys - bottom, top - ys) / softness, 0, 1)
        else:
            fadeX = ((xs >= left) & (xs < right)).astype(np.float32)
            fadeY = ((ys >= bottom) & (ys < top)).astype(np.float32)
        fade = (fadeY[:, np.newaxis] * fadeX[np.newaxis, :])[:, :, np.newaxis]

        layers = OrderedDict()
        for name, channels, pixels in image.layers():
            layers[name] = (channels, ops.alignToBBox(pixels, image.bbox, bbox) * fade)
        if ctx.value("reformat"):
            # The box becomes the format: move its bottom-left corner to the origin
            shift = (-int(np.floor(left)), -int(np.floor(bottom)))
            return image.withPixels(layers, bbox.translated(*shift),
                                    format=(max(1, int(round(right - left))), max(1, int(round(top - bottom)))))
        return image.withPixels(layers, bbox)


class Mirror(desc.ImageOpNode):
    category = "Comp Transform"
    color = "#A57AAA"
    documentation = """
Mirrors the image horizontally (flop) and / or vertically (flip) inside its format.
"""
    inputs = [
        imageInput(),
        desc.BoolParam(name="horizontal", label="Horizontal", description="Mirror left / right.", value=False),
        desc.BoolParam(name="vertical", label="Vertical", description="Mirror top / bottom.", value=False),
    ]
    outputs = imageOutput("Mirrored image.")

    def isIdentity(self, ctx):
        return "input" if not ctx.value("horizontal") and not ctx.value("vertical") else None

    def render(self, ctx, inputs):
        image = inputs["input"]
        horizontal, vertical = ctx.value("horizontal"), ctx.value("vertical")
        x0, y0, x1, y1 = image.bbox
        if horizontal:
            x0, x1 = image.width - x1, image.width - x0
        if vertical:
            y0, y1 = image.height - y1, image.height - y0

        def mirror(pixels):
            if horizontal:
                pixels = pixels[:, ::-1]
            if vertical:
                pixels = pixels[::-1]
            return np.ascontiguousarray(pixels)

        return image.mapLayers(mirror, bbox=BBox(x0, y0, x1, y1))


def _corner(name, label, description, x, y):
    return desc.Vec2d(name=name, label=label, description=description, x=x, y=y,
                      xRange=(-10000.0, 20000.0, 1.0), yRange=(-10000.0, 20000.0, 1.0))


class CornerPin(desc.ImageOpNode):
    category = "Comp Transform"
    # Parameters in pixels, scaled in the proxy renders
    proxyScaled = ('to1', 'to2', 'to3', 'to4', 'from1', 'from2', 'from3', 'from4')
    color = "#A57AAA"
    documentation = """
Perspective transform moving the 4 "From" corners onto the 4 "To" corners (as Nuke's CornerPin2D):
1 bottom-left, 2 bottom-right, 3 top-right, 4 top-left, in pixels.

Driven by tracks: connect a Tracker to the "Tracker" input and name 4 of its tracks (created by the
"CornerPin" button of the Tracker toolbar with 4 tracks selected): "From" is then the position of
the tracks at the reference frame and "To" their position at the current frame (match-move of a
plane). "Invert" applies the opposite transform (e.g. to flatten a plane).
"""
    inputs = [
        imageInput(),
        desc.Image(name="tracker", label="Tracker", optional=True, infoOnly=True,
                   description="Optional Tracker whose 4 tracks drive the corners."),
        _corner("to1", "To 1", "Bottom-left corner of the output.", 0.0, 0.0),
        _corner("to2", "To 2", "Bottom-right corner of the output.", 1920.0, 0.0),
        _corner("to3", "To 3", "Top-right corner of the output.", 1920.0, 1080.0),
        _corner("to4", "To 4", "Top-left corner of the output.", 0.0, 1080.0),
        _corner("from1", "From 1", "Bottom-left corner of the input.", 0.0, 0.0),
        _corner("from2", "From 2", "Bottom-right corner of the input.", 1920.0, 0.0),
        _corner("from3", "From 3", "Top-right corner of the input.", 1920.0, 1080.0),
        _corner("from4", "From 4", "Top-left corner of the input.", 0.0, 1080.0),
        desc.StringParam(name="tracks", label="Tracks",
                         description="Names of the 4 tracks driving the corners, comma separated (empty: the corners above).",
                         value=""),
        desc.IntParam(name="referenceFrame", label="Reference Frame",
                      description="Frame where the tracks give the 'From' corners.", value=1, range=(-1000, 10000, 1)),
        desc.BoolParam(name="invert", label="Invert", description="Apply the inverse transform.", value=False),
        desc.ChoiceParam(name="filter", label="Filter", description="Pixel filter.", value="bilinear",
                         values=["impulse", "bilinear"]),
    ]
    outputs = imageOutput("Transformed image.")
    valuesOnlyInputs = ("tracker",)

    def infoOnlyInputs(self, ctx):
        return ["tracker"]

    def _corners(self, ctx):
        """ ("from" corners, "to" corners), each (4, 2). """
        from meshroom.live import tracking
        names = tracking.trackNames(ctx.value("tracks"))
        if names:
            if len(names) != 4:
                raise RuntimeError("Name 4 tracks (comma separated).")
            try:
                tracks = tracking.parseTracks((ctx.inputValues("tracker") or {}).get("tracks", ""))
                info = ctx.inputInfo("input")
                if info is not None:
                    tracks = tracking.scaleTracks(tracks, info.width, info.height)
            except ValueError:
                tracks = []
            src, dst = [], []
            for name in names:
                track = tracking.findTrack(tracks, name)
                a = tracking.trackPosition(track, ctx.value("referenceFrame")) if track else None
                b = tracking.trackPosition(track, ctx.frame) if track else None
                if a is None or b is None:
                    raise RuntimeError(f"Track not found (or without key) in the connected Tracker: {name}.")
                src.append(a)
                dst.append(b)
            return np.array(src), np.array(dst)
        src = np.array([[ctx.value(f"from{i}")["x"], ctx.value(f"from{i}")["y"]] for i in range(1, 5)])
        dst = np.array([[ctx.value(f"to{i}")["x"], ctx.value(f"to{i}")["y"]] for i in range(1, 5)])
        return src, dst

    def _matrix(self, ctx):
        from meshroom.live import geometry
        src, dst = self._corners(ctx)
        matrix = geometry.homography(src, dst)
        if matrix is None:
            raise RuntimeError("Three corners are aligned: no perspective transform.")
        return np.linalg.inv(matrix) if ctx.value("invert") else matrix

    def isIdentity(self, ctx):
        try:
            return "input" if np.allclose(self._matrix(ctx), np.eye(3), atol=1e-9) else None
        except Exception:
            return None

    def render(self, ctx, inputs):
        from meshroom.live import ops
        return ops.transformImage(inputs["input"], self._matrix(ctx), ctx.value("filter"), ctx.checkCancelled)
