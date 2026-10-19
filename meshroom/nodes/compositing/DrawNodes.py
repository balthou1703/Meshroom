"""
Image generators and drawing nodes (Nuke "Image" and "Draw" menus): CheckerBoard, ColorWheel, Shape, Ramp.

Generators take their format from their optional "Format" input when it is connected. Drawing nodes
draw over their optional input, or only take its format ("Input Usage").
"""
__version__ = "0.3"

import math
from collections import OrderedDict

import numpy as np

from meshroom.core import Version, desc
from meshroom.live import BBox, ImageBuffer
from meshroom.live.params import (colorParam, colorValue, drawInfoOnlyInputs, formatInput, formatParams,
                                  imageInput, imageOutput, inputUsageParam, linkedVec2, outputInfo)


def drawOver(ctx, out, bbox, format, background):
    """
    Return the drawing `out` (RGBA pixels over `bbox`) merged over the `background` image, or alone
    (over transparent black, in `format`) when there is no background. The other layers of the
    background are kept.
    """
    from meshroom.live import ops
    if background is None:
        return ImageBuffer.fromRGBA(out, format, bbox)
    under = ops.alignToBBox(background.rgba(), background.bbox, bbox)
    merged = ops.mapRows(lambda a, b: ops.merge(a, b, "over"), [out, under], 4, ctx.checkCancelled)
    layers = OrderedDict((name, (channels, ops.alignToBBox(pixels, background.bbox, bbox)))
                         for name, channels, pixels in background.layers())
    layers["rgba"] = (("R", "G", "B", "A"), merged)
    return background.withPixels(layers, bbox)


def _drawBBox(width, height, background):
    """ The bbox to draw in: the format, extended to the bbox of the background. """
    bbox = BBox.fromFormat(width, height)
    return bbox.union(background.bbox) if background is not None else bbox


def _pixelCenters(bbox):
    """ Coordinates (x, y) of the pixel centers of a bbox, in Nuke coordinates (y up), shape (h, w). """
    xs = bbox.x0 + np.arange(bbox.width, dtype=np.float32) + 0.5
    ys = bbox.y1 - (np.arange(bbox.height, dtype=np.float32) + 0.5)
    return np.meshgrid(xs, ys)


class CheckerBoard(desc.ImageOpNode):
    category = "Comp Image"
    # Parameters in pixels, scaled in the proxy renders
    proxyScaled = ('boxSize', 'lineWidth')
    color = "#7B82A8"
    documentation = """
A checkerboard of two colors, with optional center lines: useful to check transforms, filters and lens distortions.
"""
    inputs = [formatInput()] + formatParams() + [
        desc.FloatParam(name="boxSize", label="Box Size", description="Size of the boxes in pixels.", value=64.0,
                        range=(2.0, 1024.0, 1.0)),
        colorParam("color0", "Color 1", "Color of the first boxes.", (0.1, 0.1, 0.1, 1.0)),
        colorParam("color1", "Color 2", "Color of the other boxes.", (0.5, 0.5, 0.5, 1.0)),
        colorParam("lineColor", "Line Color", "Color of the center lines.", (1.0, 1.0, 0.0, 1.0)),
        desc.FloatParam(name="lineWidth", label="Line Width", description="Width of the center lines (0: none).",
                        value=0.0, range=(0.0, 20.0, 0.5)),
    ]
    outputs = imageOutput("Checkerboard.")

    def getInfo(self, ctx, inputsInfo):
        return outputInfo(ctx, inputsInfo)

    def render(self, ctx, inputs):
        width, height = ctx.info().width, ctx.info().height
        size = max(ctx.value("boxSize"), 1.0)
        x, y = _pixelCenters(BBox(0, 0, width, height))
        # Boxes counted from the center of the image, as in Nuke
        parity = (np.floor((x - width / 2.0) / size) + np.floor((y - height / 2.0) / size)) % 2
        colors = np.array([colorValue(ctx, "color0"), colorValue(ctx, "color1")], dtype=np.float32)
        out = colors[parity.astype(np.int64)]
        lineWidth = ctx.value("lineWidth")
        if lineWidth > 0:
            onLine = (np.abs(x - width / 2.0) < lineWidth / 2.0) | (np.abs(y - height / 2.0) < lineWidth / 2.0)
            out[onLine] = np.array(colorValue(ctx, "lineColor"), dtype=np.float32)
        return ImageBuffer.fromRGBA(out, (width, height))


def _hsvToRgb(hue, saturation, value):
    """ Vectorized HSV (hue in [0, 1)) to RGB. """
    h6 = (hue % 1.0) * 6.0
    sector = np.floor(h6).astype(np.int64) % 6
    fraction = h6 - np.floor(h6)
    p = value * (1 - saturation)
    q = value * (1 - saturation * fraction)
    t = value * (1 - saturation * (1 - fraction))
    choices = [(value, t, p), (q, value, p), (p, value, t), (p, q, value), (t, p, value), (value, p, q)]
    rgb = np.empty(hue.shape + (3,), dtype=np.float32)
    for channel in range(3):
        rgb[..., channel] = np.choose(sector, [choice[channel] for choice in choices])
    return rgb


class ColorWheel(desc.ImageOpNode):
    category = "Comp Image"
    color = "#7B82A8"
    documentation = """
A color wheel: the hue turns around the center (red on the right), the saturation goes from the
center (0) to the edge (1). Outside of the wheel the image is transparent.
"""
    inputs = [formatInput()] + formatParams() + [
        desc.FloatParam(name="value", label="Value", description="Brightness of the wheel.", value=1.0,
                        range=(0.0, 4.0, 0.01)),
        desc.FloatParam(name="gamma", label="Gamma", description="Gamma applied to the saturation ramp.",
                        value=1.0, range=(0.2, 5.0, 0.01)),
        desc.FloatParam(name="rotate", label="Rotate", description="Rotation of the hues in degrees.", value=0.0,
                        range=(-180.0, 180.0, 1.0)),
    ]
    outputs = imageOutput("Color wheel.")

    def getInfo(self, ctx, inputsInfo):
        return outputInfo(ctx, inputsInfo)

    def render(self, ctx, inputs):
        width, height = ctx.info().width, ctx.info().height
        x, y = _pixelCenters(BBox(0, 0, width, height))
        radius = min(width, height) / 2.0
        dx, dy = x - width / 2.0, y - height / 2.0
        distance = np.sqrt(dx * dx + dy * dy) / radius
        hue = (np.arctan2(dy, dx) / (2 * math.pi) - ctx.value("rotate") / 360.0) % 1.0
        saturation = np.power(np.clip(distance, 0, 1), 1.0 / max(ctx.value("gamma"), 1e-3))
        # Antialiased edge, one pixel wide
        alpha = np.clip((1.0 - distance) * radius + 0.5, 0, 1).astype(np.float32)
        rgb = _hsvToRgb(hue, saturation, np.float32(ctx.value("value")))
        out = np.empty((height, width, 4), dtype=np.float32)
        out[:, :, :3] = rgb * alpha[:, :, np.newaxis]
        out[:, :, 3] = alpha
        return ImageBuffer.fromRGBA(out, (width, height))


def _shapeDistance(shape, x, y, halfX, halfY):
    """ Signed distance in pixels to the edge of a shape centered at the origin (negative inside). """
    if shape == "circle":
        # Ellipse: distance approximated by scaling the unit circle distance
        rx, ry = max(halfX, 1e-3), max(halfY, 1e-3)
        normalized = np.sqrt((x / rx) ** 2 + (y / ry) ** 2)
        return (normalized - 1.0) * min(rx, ry)
    if shape in ("rectangle", "square"):
        if shape == "square":
            halfY = halfX
        qx, qy = np.abs(x) - halfX, np.abs(y) - halfY
        outside = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
        return outside + np.minimum(np.maximum(qx, qy), 0)
    # Equilateral triangle pointing up, inscribed in a circle of radius halfX
    radius = max(halfX, 1e-3)
    k = math.sqrt(3.0)
    side = radius * k / 2.0  # Half of the side length
    px = np.abs(x) - side
    py = y + radius / 2.0
    flip = px + k * py > 0
    px, py = np.where(flip, (px - k * py) / 2.0, px), np.where(flip, (-k * px - py) / 2.0, py)
    px = px - np.clip(px, -2.0 * side, 0.0)
    return -np.sqrt(px * px + py * py) * np.sign(py)


class Shape(desc.ImageOpNode):
    category = "Comp Draw"
    # Region of interest: the same region of the inputs
    pointwise = True
    # Parameters in pixels, scaled in the proxy renders
    proxyScaled = ('center', 'shapeSize', 'feather')
    color = "#7B82A8"
    documentation = """
Draws a simple shape (circle / ellipse, rectangle, square, triangle) with a feathered edge, in a
color, over the optional input (or over transparent black of the given format).
The color is premultiplied by the shape: the alpha holds the shape, ready to be used as a mask.
"""
    inputs = [
        imageInput(description="Optional background (or format, see 'Input Usage').", optional=True),
        inputUsageParam(),
        desc.ChoiceParam(name="shape", label="Shape", description="Type of shape.", value="circle",
                         values=["circle", "rectangle", "square", "triangle"]),
        desc.BoolParam(name="centerOnImage", label="Center On Image", description="Center the shape on the image.",
                       value=True),
        desc.Vec2d(name="center", label="Center", description="Center of the shape in pixels.", x=960.0, y=540.0,
                   xRange=(-2000.0, 8000.0, 1.0), yRange=(-2000.0, 8000.0, 1.0),
                   enabled=lambda node: not node.centerOnImage.value),
        linkedVec2("shapeSize", "Size", "Width and height of the shape in pixels.", 400.0, (1.0, 4000.0, 1.0)),
        desc.FloatParam(name="rotate", label="Rotate", description="Rotation in degrees, counter-clockwise.",
                        value=0.0, range=(-180.0, 180.0, 0.5)),
        desc.FloatParam(name="feather", label="Feather", description="Width of the soft edge in pixels.",
                        value=0.0, range=(0.0, 500.0, 0.5)),
        desc.ChoiceParam(name="featherMode", label="Feather Mode", value="outside",
                         values=["outside", "centered", "inside"],
                         description="Where the soft edge is, relative to the shape outline."),
        colorParam("shapeColor", "Color", "Color of the shape.", (1.0, 1.0, 1.0, 1.0)),
        desc.BoolParam(name="invert", label="Invert", description="Fill outside of the shape.", value=False),
        *formatParams(),
    ]
    outputs = imageOutput("Image with the shape.")

    def infoOnlyInputs(self, ctx):
        return drawInfoOnlyInputs(ctx)

    def getInfo(self, ctx, inputsInfo):
        return outputInfo(ctx, inputsInfo)

    def render(self, ctx, inputs):
        from meshroom.live import ops
        # None when the input is not connected, or only gives the format
        background = inputs.get("input")
        width, height = ctx.info().width, ctx.info().height
        bbox = _drawBBox(width, height, background)

        if ctx.value("centerOnImage"):
            cx, cy = width / 2.0, height / 2.0
        else:
            cx, cy = ctx.value("center")["x"], ctx.value("center")["y"]
        size = ctx.value("shapeSize")
        halfX, halfY = size["x"] / 2.0, size["y"] / 2.0
        angle = math.radians(ctx.value("rotate"))
        feather, mode = ctx.value("feather"), ctx.value("featherMode")
        shape, invert = ctx.value("shape"), ctx.value("invert")
        color = np.array(colorValue(ctx, "shapeColor"), dtype=np.float32)
        cos, sin = math.cos(angle), math.sin(angle)

        def band(start, end):
            xs = bbox.x0 + np.arange(bbox.width, dtype=np.float32) + 0.5
            ys = bbox.y1 - (np.arange(start, end, dtype=np.float32) + 0.5)
            x, y = np.meshgrid(xs - cx, ys - cy)
            # Into the shape frame (inverse rotation)
            u, v = cos * x + sin * y, -sin * x + cos * y
            distance = _shapeDistance(shape, u, v, halfX, halfY)
            if feather > 0:
                shift = {"outside": 0.0, "centered": feather / 2.0, "inside": feather}[mode]
                alpha = np.clip(1.0 - (distance + shift) / feather, 0, 1)
            else:
                alpha = np.clip(0.5 - distance, 0, 1)  # One pixel of antialiasing
            if invert:
                alpha = 1.0 - alpha
            out[start:end] = alpha[:, :, np.newaxis] * color

        out = np.empty((bbox.height, bbox.width, 4), dtype=np.float32)
        ops.parallelBands(band, bbox.height, ctx.checkCancelled)
        return drawOver(ctx, out, bbox, (width, height), background)


def _rampShape(t, rampType):
    """ Shape of the ramp profile, `t` in [0, 1]. """
    if rampType == "smooth":
        return t * t * (3.0 - 2.0 * t)
    if rampType == "smooth0":
        # Smooth at point 0 only
        return 1.0 - (1.0 - t) * (1.0 - t)
    if rampType == "smooth1":
        # Smooth at point 1 only
        return t * t
    return t


class Ramp(desc.ImageOpNode):
    category = "Comp Draw"
    # Region of interest: the same region of the inputs
    pointwise = True
    # Parameters in pixels, scaled in the proxy renders
    proxyScaled = ('point0', 'point1')
    color = "#7B82A8"
    documentation = """
A gradient from "Color 0" at point 0 to "Color 1" at point 1 (constant beyond the points), drawn
over the optional input or alone. The two points can be moved in the viewer: they set the
orientation and the length of the fade.
"""
    inputs = [
        imageInput(description="Optional background (or format, see 'Input Usage').", optional=True),
        inputUsageParam(),
        desc.Vec2d(name="point0", label="Point 0", description="Start of the ramp, in pixels.", x=480.0, y=540.0,
                   xRange=(-2000.0, 8000.0, 1.0), yRange=(-2000.0, 8000.0, 1.0)),
        desc.Vec2d(name="point1", label="Point 1", description="End of the ramp, in pixels.", x=1440.0, y=540.0,
                   xRange=(-2000.0, 8000.0, 1.0), yRange=(-2000.0, 8000.0, 1.0)),
        colorParam("color0", "Color 0", "Color at point 0.", (0.0, 0.0, 0.0, 0.0)),
        colorParam("color1", "Color 1", "Color at point 1.", (1.0, 1.0, 1.0, 1.0)),
        desc.ChoiceParam(name="rampType", label="Type", value="linear", values=["linear", "smooth", "smooth0", "smooth1"],
                         description="Profile of the fade: linear, smooth at both ends, or at point 0 / point 1 only."),
        *formatParams(),
    ]
    outputs = imageOutput("Image with the ramp.")

    def infoOnlyInputs(self, ctx):
        return drawInfoOnlyInputs(ctx)

    def getInfo(self, ctx, inputsInfo):
        return outputInfo(ctx, inputsInfo)

    def render(self, ctx, inputs):
        from meshroom.live import ops
        background = inputs.get("input")
        width, height = ctx.info().width, ctx.info().height
        bbox = _drawBBox(width, height, background)
        p0, p1 = ctx.value("point0"), ctx.value("point1")
        x0, y0 = p0["x"], p0["y"]
        dx, dy = p1["x"] - x0, p1["y"] - y0
        length2 = dx * dx + dy * dy
        color0 = np.array(colorValue(ctx, "color0"), dtype=np.float32)
        color1 = np.array(colorValue(ctx, "color1"), dtype=np.float32)
        rampType = ctx.value("rampType")

        def band(start, end):
            xs = bbox.x0 + np.arange(bbox.width, dtype=np.float32) + 0.5 - x0
            ys = bbox.y1 - (np.arange(start, end, dtype=np.float32) + 0.5) - y0
            if length2 < 1e-9:
                # Both points at the same place: a hard edge
                t = np.ones((end - start, bbox.width), dtype=np.float32)
            else:
                t = np.clip((xs[np.newaxis, :] * dx + ys[:, np.newaxis] * dy) / length2, 0.0, 1.0)
            t = _rampShape(t, rampType).astype(np.float32)[:, :, np.newaxis]
            out[start:end] = color0 + (color1 - color0) * t

        out = np.empty((bbox.height, bbox.width, 4), dtype=np.float32)
        ops.parallelBands(band, bbox.height, ctx.checkCancelled)
        return drawOver(ctx, out, bbox, (width, height), background)


class Roto(desc.ImageOpNode):
    category = "Comp Draw"
    # Region of interest: the same region of the inputs
    pointwise = True
    # Parameters in pixels, scaled in the proxy renders
    proxyScaled = ('feather',)
    color = "#7B82A8"
    documentation = """
Rotoscoping shapes: closed Bézier curves (with tangents), rectangles and ellipses, drawn in the
viewer when the node is selected (toolbar on the left of the viewer):
- Select: click or drag a selection box to select points and shapes; drag points, tangents
  (Shift: break the tangents) or a curve; double-click a point to make it smooth / a corner;
  Delete removes the selected points (or shapes);
- Bézier: click to add corner points, click and drag to add smooth points; the shape is closed
  from the start; click the first point, press Enter or change tool to finish;
- Rectangle / Ellipse: drag a box.

Animation: with "Auto Key" (toolbar), a new shape gets a key at the current frame and every
change at another frame adds a key there; the points are interpolated between the keys.

Tracking: connect up to 3 Trackers to the "Tracker" inputs, then add links to a shape in the shape
list: the whole shape, or one of its points, follows a track from the link frame. A point with its
own link follows it instead of the links of the whole shape.

Optical flow: connect an OpticalFlow node (on the plate) to the "Vectors" input (or only the plate to
the input: the flow is then computed on the fly), then "Track Shapes Forward / Backward" moves the
points of the shapes along the flow from the current frame to the end (or the start) of the frame
range, with a key at each frame: the shapes whose "flow" button is on in the shape list, or all of
them when none is (not the shapes linked to tracks).

Each shape has its own feather, opacity and invert; "Feather" adds a feather to all of them. A shape
can be closed (filled) or an open curve (a stroke of its width).
"Output": over (the color over the input), replaceAlpha (the matte becomes the alpha of the
input), addAlpha (the matte is added to the alpha of the input), cutOut (the input multiplied by
the matte).
"""
    inputs = [
        imageInput(description="Optional background (or format, see 'Input Usage').", optional=True),
        desc.Image(name="tracker", label="Tracker 1", optional=True, infoOnly=True,
                   description="Optional Tracker (or CameraTracker): its tracks can drive the shapes (see the shape list)."),
        desc.Image(name="tracker2", label="Tracker 2", optional=True, infoOnly=True,
                   description="Optional second Tracker for the shape links."),
        desc.Image(name="tracker3", label="Tracker 3", optional=True, infoOnly=True,
                   description="Optional third Tracker for the shape links."),
        desc.Image(name="vectors", label="Vectors", optional=True, infoOnly=True,
                   description="Optional motion vectors (an OpticalFlow node) to move the shapes along the flow."),
        inputUsageParam(),
        desc.StringParam(name="shapes", label="Shapes", description="Shapes as JSON (edited in the viewer).",
                         value="[]", semantic="rotoShapes"),
        desc.PushButtonParam(name="flowForward", label="Track Shapes Forward", semantic="liveAction",
                             description="Move the shapes along the optical flow from the current frame to the end of "
                                         "the frame range, with a key at each frame."),
        desc.PushButtonParam(name="flowBackward", label="Track Shapes Backward", semantic="liveAction",
                             description="Move the shapes along the optical flow from the current frame to the start of "
                                         "the frame range, with a key at each frame."),
        desc.ChoiceParam(name="outputMode", label="Output", value="over",
                         values=["over", "replaceAlpha", "addAlpha", "cutOut"],
                         description="over: the color, premultiplied by the matte, over the input. replaceAlpha: the "
                                     "matte replaces the alpha of the input. addAlpha: the matte is added to the alpha "
                                     "of the input. cutOut: the input multiplied by the matte."),
        colorParam("shapeColor", "Color", "Color of the shapes ('over' output).", (1.0, 1.0, 1.0, 1.0)),
        desc.FloatParam(name="feather", label="Feather", description="Feather added to every shape, in pixels.",
                        value=0.0, range=(0.0, 200.0, 0.5)),
        desc.FloatParam(name="opacity", label="Opacity", description="Opacity of the whole matte.", value=1.0,
                        range=(0.0, 1.0, 0.01)),
        *formatParams(),
    ]
    outputs = imageOutput("Image with the shapes.")
    valuesOnlyInputs = ("tracker", "tracker2", "tracker3")

    def upgradeAttributeValues(self, attrValues, fromVersion):
        # Before 0.3, "alpha" was the only alpha output
        if fromVersion < Version("0.3") and attrValues.get("outputMode") == "alpha":
            attrValues["outputMode"] = "replaceAlpha"
        return attrValues

    def infoOnlyInputs(self, ctx):
        return ["tracker", "tracker2", "tracker3", "vectors"] + drawInfoOnlyInputs(ctx)

    def runAction(self, name, context):
        """ Track the shapes along the optical flow (see the documentation): the new shapes. """
        import json
        from meshroom.live import flow, roto
        if name not in ("flowForward", "flowBackward"):
            raise RuntimeError(f"Unknown action: {name}")
        useVectors = context.hasInput("vectors")
        if not useVectors and not context.hasInput("input"):
            raise RuntimeError("Connect an OpticalFlow node to 'Vectors', or the plate to the input.")
        shapes = roto.parseShapes(context.value("shapes"))
        if not shapes:
            raise RuntimeError("No shape to track.")
        indices = [i for i, shape in enumerate(shapes) if shape.get("flow")] or list(range(len(shapes)))
        # Shapes following tracks move with them: not along the flow too
        indices = [i for i in indices if not roto.shapeLinks(shapes[i])]
        if not indices:
            raise RuntimeError("The shapes to track are linked to tracks: remove their links to track them with the flow.")
        forward = name == "flowForward"
        start = context.currentFrame
        allFrames = context.frames if context.hasInput("input") else context.inputFrames("vectors")
        frames = [f for f in allFrames if (f >= start if forward else f <= start)]
        frames = frames if forward else frames[::-1]
        if len(frames) < 2:
            raise RuntimeError("No frame to track to from the current frame.")
        for step, (frame, nextFrame) in enumerate(zip(frames, frames[1:])):
            context.checkCancelled()
            if useVectors:
                vectors = context.renderInput("vectors", frame)
                field = flow.flowLayer(vectors, flow.FORWARD if forward else flow.BACKWARD)
                if field is None:
                    raise RuntimeError("The 'Vectors' input has no forward / backward layers (OpticalFlow node).")
                bbox = vectors.bbox
            else:
                image = context.render(frame)
                field = flow.computeFlow(image, context.render(nextFrame), cancel=context.checkCancelled)
                bbox = image.bbox
            for index in indices:
                shapes[index] = roto.advectShape(shapes[index], frame, nextFrame,
                                                 lambda x, y: flow.flowAt(field, bbox, x, y))
            # Shown live: the shapes move in the viewer, frame after frame
            context.publish({"shapes": json.dumps(shapes)}, nextFrame)
            context.progress((step + 1) / float(len(frames) - 1), "Track shapes")
        return {"shapes": json.dumps(shapes)}

    def getInfo(self, ctx, inputsInfo):
        return outputInfo(ctx, inputsInfo)

    def render(self, ctx, inputs):
        from meshroom.live import ops, roto, tracking
        try:
            shapes = roto.parseShapes(ctx.value("shapes"))
        except ValueError as e:
            raise RuntimeError(f"Invalid shapes: {e}") from e
        if ctx.proxy != 1.0:
            # Shapes drawn in the pixels of the full resolution
            shapes = roto.scaleShapes(shapes, ctx.proxy)
        # Tracks of the connected Trackers, by input, in the pixels of the Roto format
        tracks = {}
        info = ctx.info()
        for name in self.valuesOnlyInputs:
            try:
                tracks[name] = tracking.scaleTracks(tracking.parseTracks((ctx.inputValues(name) or {}).get("tracks", "")),
                                                    info.width, info.height)
            except ValueError:
                tracks[name] = []
        background = inputs.get("input")
        width, height = ctx.info().width, ctx.info().height
        bbox = _drawBBox(width, height, background)
        matte = roto.shapesMatte(shapes, bbox, ctx.checkCancelled, frame=ctx.frame, tracks=tracks,
                                 feather=ctx.value("feather"))
        matte = (matte * np.float32(ctx.value("opacity")))[:, :, np.newaxis]
        mode = ctx.value("outputMode")
        if mode == "over":
            out = matte * np.array(colorValue(ctx, "shapeColor"), dtype=np.float32)
            return drawOver(ctx, out.astype(np.float32), bbox, (width, height), background)
        if background is None:
            if mode == "cutOut":
                return ImageBuffer.fromRGBA(np.zeros((bbox.height, bbox.width, 4), np.float32), (width, height), bbox)
            return ImageBuffer.fromRGBA(np.repeat(matte, 4, axis=2), (width, height), bbox)
        rgba = np.array(ops.alignToBBox(background.rgba(), background.bbox, bbox))
        if mode == "replaceAlpha":
            rgba[:, :, 3] = matte[:, :, 0]
        elif mode == "addAlpha":
            rgba[:, :, 3] = rgba[:, :, 3] + matte[:, :, 0] * (1.0 - rgba[:, :, 3])
        else:  # cutOut
            rgba *= matte
        layers = OrderedDict((name, (channels, ops.alignToBBox(pixels, background.bbox, bbox)))
                             for name, channels, pixels in background.layers())
        layers["rgba"] = (("R", "G", "B", "A"), rgba)
        return background.withPixels(layers, bbox)
