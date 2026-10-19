"""
Time nodes (Nuke "Time" menu): TimeOffset, FrameHold, Retime, TimeWarp, OpticalFlow.

They render other frames of their input (`ctx.renderInput`), so they are `timeBased`: the
evaluator does not render their input at the current frame beforehand.
"""
__version__ = "0.4"

import math
from collections import OrderedDict

import numpy as np

from meshroom.core import Version, desc
from meshroom.live import ImageInfo
from meshroom.live.params import imageInput, imageOutput


_FILTERS = ["nearest", "blend", "motion"]
_FILTER_DESCRIPTION = ("nearest: the closest input frame. blend: mix of the two input frames around. motion: the two "
                       "frames moved along the optical flow, then mixed (see 'Motion Vectors').")


def _vectorsInput():
    return desc.Image(name="vectors", label="Vectors", optional=True,
                      description="Optional motion vectors (an OpticalFlow node on the same input), used by the 'motion' "
                                  "filter when 'Motion Vectors' is 'vectors input'.")


def _vectorsSourceParam():
    return desc.ChoiceParam(name="motionVectors", label="Motion Vectors", value="compute",
                            values=["compute", "vectors input"],
                            description="'motion' filter: compute the optical flow on the fly (no other node needed), or "
                                        "read it from the 'Vectors' input (an OpticalFlow node, faster when it is reused).",
                            enabled=lambda node: node.filter.value == "motion")


def _flowBetween(ctx, image0, image1, frame0, frame1):
    """ (forward flow of image0, backward flow of image1): from the 'vectors' input, or computed. """
    from meshroom.live import flow
    forward = backward = None
    if ctx.value("motionVectors") == "vectors input":
        if ctx.inputInfo("vectors") is None:
            raise RuntimeError("'Motion Vectors' is 'vectors input': connect an OpticalFlow node to 'Vectors'.")
        forward = flow.flowLayer(ctx.renderInput("vectors", frame0), flow.FORWARD)
        backward = flow.flowLayer(ctx.renderInput("vectors", frame1), flow.BACKWARD)
        if forward is not None and forward.shape[:2] != (image0.bbox.height, image0.bbox.width):
            forward = None
        if backward is not None and backward.shape[:2] != (image1.bbox.height, image1.bbox.width):
            backward = None
    if forward is None:
        forward = flow.computeFlow(image0, image1, cancel=ctx.checkCancelled)
    if backward is None:
        backward = flow.computeFlow(image1, image0, cancel=ctx.checkCancelled)
    return forward, backward


def _upgradeMotionVectors(attrValues, fromVersion):
    """ Before 0.4, the 'motion' filter used the 'Vectors' input whenever it was connected. """
    if fromVersion < Version("0.4") and isinstance(attrValues.get("vectors"), str) and "{" in attrValues["vectors"]:
        attrValues["motionVectors"] = "vectors input"
    return attrValues


def _retimeRender(ctx, source):
    """ The input at a fractional frame `source`, with the "filter" of the node (nearest, blend, motion). """
    from meshroom.live import flow, ops
    if ctx.value("filter") == "nearest" or abs(source - round(source)) < 1e-4:
        return ctx.renderInput("input", int(math.floor(source + 0.5)))
    before, after = int(math.floor(source)), int(math.floor(source)) + 1
    amount = source - before
    imageBefore = ctx.renderInput("input", before)
    imageAfter = ctx.renderInput("input", after)
    if ctx.value("filter") == "motion":
        forward, backward = _flowBetween(ctx, imageBefore, imageAfter, before, after)
        return flow.interpolate(imageBefore, imageAfter, forward, backward, amount)
    bbox = imageBefore.bbox.union(imageAfter.bbox)
    blended = ops.mapRows(
        lambda a, b: ops.mix(a, b, amount),
        [ops.alignToBBox(imageBefore.rgba(), imageBefore.bbox, bbox),
         ops.alignToBBox(imageAfter.rgba(), imageAfter.bbox, bbox)], 4, ctx.checkCancelled)
    layers = OrderedDict((name, (channels, ops.alignToBBox(pixels, imageBefore.bbox, bbox)))
                         for name, channels, pixels in imageBefore.layers())
    layers["rgba"] = (("R", "G", "B", "A"), blended)
    return imageBefore.withPixels(layers, bbox)


def _shiftedInfo(info, offset):
    if not info.hasFrameRange:
        return info
    return ImageInfo(info.width, info.height, info.firstFrame + offset, info.lastFrame + offset)


class TimeOffset(desc.ImageOpNode):
    category = "Comp Time"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#AB9B5C"
    timeBased = True
    documentation = """
Moves the input in time: with an offset of +10, the input frame 1 is shown at frame 11.
"""
    inputs = [
        imageInput(),
        desc.IntParam(name="timeOffset", label="Time Offset", description="Offset in frames (negative: earlier).",
                      value=0, range=(-1000, 1000, 1)),
        desc.BoolParam(name="reverse", label="Reverse", description="Play the input frame range backwards.",
                       value=False),
    ]
    outputs = imageOutput("Image moved in time.")

    def isIdentity(self, ctx):
        return "input" if ctx.value("timeOffset") == 0 and not ctx.value("reverse") else None

    def getInfo(self, ctx, inputsInfo):
        return _shiftedInfo(inputsInfo["input"], ctx.value("timeOffset"))

    def sourceFrame(self, ctx, frame):
        """ The input frame shown at `frame`. """
        source = frame - ctx.value("timeOffset")
        info = ctx.inputInfo("input")
        if ctx.value("reverse") and info is not None and info.hasFrameRange:
            source = info.firstFrame + info.lastFrame - source
        return source

    def render(self, ctx, inputs):
        return ctx.renderInput("input", self.sourceFrame(ctx, ctx.frame))


class FrameHold(desc.ImageOpNode):
    category = "Comp Time"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#AB9B5C"
    timeBased = True
    documentation = """
Holds a frame of the input. With an increment above 0, a new frame is held every "Increment" frames
from "First Frame" (e.g. 2: frames 1, 1, 3, 3, 5...).
"""
    inputs = [
        imageInput(),
        desc.IntParam(name="firstFrame", label="First Frame", description="Frame held.", value=1,
                      range=(-1000, 10000, 1)),
        desc.IntParam(name="increment", label="Increment", value=0, range=(0, 100, 1),
                      description="0: hold a single frame. N: hold a frame every N frames."),
    ]
    outputs = imageOutput("Held frame.")

    def getInfo(self, ctx, inputsInfo):
        return inputsInfo["input"]

    def sourceFrame(self, ctx, frame):
        first, increment = ctx.value("firstFrame"), ctx.value("increment")
        if increment <= 0:
            return first
        return first + ((frame - first) // increment) * increment

    def render(self, ctx, inputs):
        return ctx.renderInput("input", self.sourceFrame(ctx, ctx.frame))


class Retime(desc.ImageOpNode):
    category = "Comp Time"
    color = "#AB9B5C"
    timeBased = True
    documentation = """
Remaps time, as Nuke's Retime: the input range [Input First, Input Last] is stretched onto the
output range [Output First, Output Last]. Each bound is taken from the input frame range unless
its checkbox is on. Without an output last frame, the output length comes from "Speed"
(2: twice as fast). "Reverse" plays the input backwards; the "blend" filter mixes the two input
frames around a fractional frame.
"""
    inputs = [
        imageInput(),
        desc.BoolParam(name="useInputFirst", label="Input First", description="Set the first input frame.", value=False),
        desc.IntParam(name="inputFirst", label="Input First Frame", description="First frame of the input range.",
                      value=1, range=(-1000, 10000, 1), enabled=lambda node: node.useInputFirst.value),
        desc.BoolParam(name="useInputLast", label="Input Last", description="Set the last input frame.", value=False),
        desc.IntParam(name="inputLast", label="Input Last Frame", description="Last frame of the input range.",
                      value=100, range=(-1000, 10000, 1), enabled=lambda node: node.useInputLast.value),
        desc.BoolParam(name="useOutputFirst", label="Output First",
                       description="Set the first output frame (otherwise: the first input frame).", value=False),
        desc.IntParam(name="outputFirst", label="Output First Frame", description="First frame of the output.",
                      value=1, range=(-1000, 10000, 1), enabled=lambda node: node.useOutputFirst.value),
        desc.BoolParam(name="useOutputLast", label="Output Last",
                       description="Set the last output frame (otherwise: from the speed).", value=False),
        desc.IntParam(name="outputLast", label="Output Last Frame", description="Last frame of the output.",
                      value=100, range=(-1000, 10000, 1), enabled=lambda node: node.useOutputLast.value),
        desc.FloatParam(name="speed", label="Speed", description="Playback speed of the input (without output last frame).",
                        value=1.0, range=(0.05, 10.0, 0.05), enabled=lambda node: not node.useOutputLast.value),
        desc.BoolParam(name="reverse", label="Reverse", description="Play backwards.", value=False),
        desc.ChoiceParam(name="filter", label="Filter", value="nearest", values=_FILTERS, description=_FILTER_DESCRIPTION),
        _vectorsSourceParam(),
        _vectorsInput(),
    ]
    outputs = imageOutput("Retimed image.")

    def inputRegions(self, ctx, region):
        """ Region of interest: the same region of the input frames (the whole frames for the optical flow). """
        if ctx.value("filter") == "motion":
            return None
        return {"input": region}

    def upgradeAttributeValues(self, attrValues, fromVersion):
        _upgradeMotionVectors(attrValues, fromVersion)
        # Before 0.2, "keepFirstFrame" / "outputFirstFrame" set the first output frame
        if fromVersion < Version("0.2"):
            keepFirst = attrValues.pop("keepFirstFrame", True)
            outputFirst = attrValues.pop("outputFirstFrame", None)
            if keepFirst is False and outputFirst is not None:
                attrValues["useOutputFirst"] = True
                attrValues["outputFirst"] = outputFirst
        return attrValues

    def isIdentity(self, ctx):
        neutral = not any(ctx.value(name) for name in ("useInputFirst", "useInputLast", "useOutputFirst",
                                                       "useOutputLast", "reverse"))
        return "input" if neutral and ctx.value("speed") == 1.0 else None

    def _ranges(self, ctx, info):
        """ ((input first, input last), (output first, output last)), or None without range. """
        hasRange = info is not None and info.hasFrameRange
        inFirst = ctx.value("inputFirst") if ctx.value("useInputFirst") else (info.firstFrame if hasRange else None)
        inLast = ctx.value("inputLast") if ctx.value("useInputLast") else (info.lastFrame if hasRange else None)
        if inFirst is None or inLast is None:
            return None
        outFirst = ctx.value("outputFirst") if ctx.value("useOutputFirst") else inFirst
        if ctx.value("useOutputLast"):
            outLast = ctx.value("outputLast")
        else:
            # Exact (fractional) end: the mapping keeps the speed, the frame range is rounded down
            speed = max(ctx.value("speed"), 1e-3)
            outLast = outFirst + abs(inLast - inFirst) / speed
        return (inFirst, inLast), (outFirst, outLast)

    def getInfo(self, ctx, inputsInfo):
        info = inputsInfo["input"]
        ranges = self._ranges(ctx, info)
        if ranges is None:
            return info
        outFirst, outLast = ranges[1]
        outLast = int(math.floor(outLast + 1e-6))
        return ImageInfo(info.width, info.height, min(outFirst, outLast), max(outFirst, outLast))

    def sourceFrame(self, ctx, frame):
        """ The (fractional) input frame shown at `frame`. """
        ranges = self._ranges(ctx, ctx.inputInfo("input"))
        if ranges is None:
            return float(frame)
        (inFirst, inLast), (outFirst, outLast) = ranges
        if ctx.value("reverse"):
            inFirst, inLast = inLast, inFirst
        if outLast == outFirst:
            return float(inFirst)
        t = (frame - outFirst) / float(outLast - outFirst)
        t = min(max(t, 0.0), 1.0)
        return inFirst + t * (inLast - inFirst)

    def render(self, ctx, inputs):
        return _retimeRender(ctx, self.sourceFrame(ctx, ctx.frame))


class TimeWarp(desc.ImageOpNode):
    category = "Comp Time"
    color = "#AB9B5C"
    timeBased = True
    documentation = """
Remaps time with a curve, as Nuke's TimeWarp: for each output frame (x), the curve gives the input
frame shown (y, fractional frames with the "blend" or "motion" filter). Edit the curve in the attribute editor:
click to add a point, drag the points, double-click one to remove it. Without point the time is
unchanged; the output frame range is the x range of the curve when it has two points or more.
"""
    inputs = [
        imageInput(),
        desc.StringParam(name="lookup", label="Lookup", value='{"time": []}', semantic="timeCurve",
                         description="Curve (JSON): output frame -> input frame."),
        desc.ChoiceParam(name="filter", label="Filter", value="nearest", values=_FILTERS, description=_FILTER_DESCRIPTION),
        _vectorsSourceParam(),
        _vectorsInput(),
    ]
    outputs = imageOutput("Image with its time remapped.")

    def inputRegions(self, ctx, region):
        """ Region of interest: the same region of the input frames (the whole frames for the optical flow). """
        if ctx.value("filter") == "motion":
            return None
        return {"input": region}

    def _points(self, ctx):
        from meshroom.live import curves
        try:
            return curves.parseCurves(ctx.value("lookup")).get("time", [])
        except ValueError as e:
            raise RuntimeError(f"Invalid curve: {e}") from e

    def isIdentity(self, ctx):
        from meshroom.live import curves
        try:
            return "input" if curves.isIdentity(self._points(ctx)) else None
        except RuntimeError:
            return None

    def upgradeAttributeValues(self, attrValues, fromVersion):
        return _upgradeMotionVectors(attrValues, fromVersion)

    def getInfo(self, ctx, inputsInfo):
        from meshroom.live import curves
        info = inputsInfo["input"]
        bounds = curves.frameRange(self._points(ctx))
        if bounds is None:
            return info
        return ImageInfo(info.width, info.height, int(math.ceil(bounds[0])), int(math.floor(bounds[1])))

    def sourceFrame(self, ctx, frame):
        from meshroom.live import curves
        return float(curves.evaluate(self._points(ctx), [frame])[0])

    def render(self, ctx, inputs):
        return _retimeRender(ctx, self.sourceFrame(ctx, ctx.frame))


class OpticalFlow(desc.ImageOpNode):
    category = "Comp Time"
    color = "#AB9B5C"
    timeBased = True
    documentation = """
Motion vectors of the input (as Nuke's VectorGenerator): the input with its "forward" (motion to
the next frame) and "backward" (motion to the previous frame) layers, channels u (right) and v (up)
in pixels. Connect it to the "Vectors" input of a Retime / TimeWarp ("motion" filter) to
interpolate frames, or of a Roto to move shapes along the flow ("Track Shapes With Flow").

The flow is computed on a copy of the images reduced to "Resolution" pixels at most (faster, less
detailed); "Smoothness" is the size of the window the motion is measured on.
"""
    inputs = [
        imageInput(),
        desc.ChoiceParam(name="vectors", label="Vectors", value="forward and backward",
                         values=["forward and backward", "forward", "backward"],
                         description="Layers computed (each is a computation of the flow between two frames)."),
        desc.IntParam(name="resolution", label="Resolution", value=960, range=(128, 4096, 32),
                      description="Largest size (pixels) of the images the flow is computed on."),
        desc.IntParam(name="smoothness", label="Smoothness", value=11, range=(3, 41, 2),
                      description="Size (pixels, at the computation resolution) of the window the motion is measured on."),
        desc.IntParam(name="iterations", label="Iterations", value=3, range=(1, 10, 1),
                      description="Refinements per level of the pyramid."),
    ]
    outputs = imageOutput("The input with its 'forward' and 'backward' motion vector layers.")

    def render(self, ctx, inputs):
        from meshroom.live import flow
        frame = ctx.frame
        image = ctx.renderInput("input", frame)
        info = ctx.inputInfo("input")
        hasRange = info is not None and info.hasFrameRange
        zero = np.zeros((image.bbox.height, image.bbox.width, 2), np.float32)
        options = dict(maxSize=ctx.value("resolution"), window=ctx.value("smoothness"),
                       iterations=ctx.value("iterations"), cancel=ctx.checkCancelled)
        result = image
        vectors = ctx.value("vectors")
        if vectors != "backward":
            # No next frame after the last one: no motion
            forward = flow.computeFlow(image, ctx.renderInput("input", frame + 1), **options) \
                if not hasRange or frame < info.lastFrame else zero
            result = result.withLayer(flow.FORWARD, forward, flow.FLOW_CHANNELS)
        if vectors != "forward":
            backward = flow.computeFlow(image, ctx.renderInput("input", frame - 1), **options) \
                if not hasRange or frame > info.firstFrame else zero
            result = result.withLayer(flow.BACKWARD, backward, flow.FLOW_CHANNELS)
        return result
