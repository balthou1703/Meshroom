"""
Color correction nodes (Nuke "Color" menu): ColorCorrect, Saturation, HueShift, Exposure,
Multiply, Add, Clamp, Invert, ColorLookup (curves), ColorLookupHSV and OCIOColorSpace.

All work on the "rgba" layer, on the channels selected by "Channels", limited by the optional
mask and blended by "Mix". With neutral values they pass their input through.
"""
__version__ = "0.3"

import numpy as np

from meshroom.core import Version, desc
from meshroom.live.params import (
    colorOpInputs, colorspaceParam, imageOutput, linkedColor, linkedColorValues, mixParam, processRGBA, upgradeLinkedColors,
)

# Rec. 709 luminance weights
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _luminance(rgba):
    return (rgba[:, :, :3] @ _LUMA)[:, :, np.newaxis]


def _float(name, label, description, value, range):
    return desc.FloatParam(name=name, label=label, description=description, value=value, range=range)


def _neutral(ctx, values):
    """ True when every parameter has its neutral value, or the mix is 0. """
    if ctx.value("mix") <= 0.0:
        return True
    return all(ctx.value(name) == value for name, value in values.items())


class ColorCorrect(desc.ImageOpNode):
    category = "Comp Color"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#7AA9FF"
    documentation = """
Master color correction, applied in this order: saturation, contrast (around 0.18), gamma,
gain, offset.

Contrast, gamma, gain and offset are a single value, or a value per channel when "split" is
checked (r, g, b, a).
"""
    _COLORS = ("contrast", "gamma", "gain", "offset")
    _NEUTRAL = {"contrast": 1.0, "gamma": 1.0, "gain": 1.0, "offset": 0.0}

    inputs = colorOpInputs() + [
        _float("saturation", "Saturation", "0: grey, 1: unchanged.", 1.0, (0.0, 4.0, 0.01)),
        linkedColor("contrast", "Contrast", "Contrast around the 0.18 grey.", 1.0, (0.0, 4.0, 0.01)),
        linkedColor("gamma", "Gamma", "Gamma (applied to positive values).", 1.0, (0.2, 5.0, 0.01)),
        linkedColor("gain", "Gain", "Multiplier.", 1.0, (0.0, 4.0, 0.01)),
        linkedColor("offset", "Offset", "Value added at the end.", 0.0, (-1.0, 1.0, 0.01)),
        mixParam(),
    ]
    outputs = imageOutput("Corrected image.")

    def upgradeAttributeValues(self, attrValues, fromVersion):
        # Before 0.3, contrast, gamma, gain and offset were single floats
        if fromVersion < Version("0.3"):
            upgradeLinkedColors(attrValues, self._COLORS)
        return attrValues

    def _values(self, ctx):
        return {name: linkedColorValues(ctx, name) for name in self._COLORS}

    def isIdentity(self, ctx):
        if ctx.value("mix") <= 0.0:
            return "input"
        values = self._values(ctx)
        # Contrast and gamma only apply to the color channels (their alpha component is not used)
        used = {"contrast": slice(0, 3), "gamma": slice(0, 3), "gain": slice(0, 4), "offset": slice(0, 4)}
        neutral = ctx.value("saturation") == 1.0 and all((values[name][used[name]] == neutral).all()
                                                          for name, neutral in self._NEUTRAL.items())
        return "input" if neutral else None

    def render(self, ctx, inputs):
        saturation = ctx.value("saturation")
        values = self._values(ctx)
        contrast, gamma, gain, offset = (values[name] for name in self._COLORS)

        def correct(rgba):
            out = rgba.copy()
            rgb = out[:, :, :3]
            if saturation != 1.0:
                luma = _luminance(rgba)
                rgb[:] = luma + (rgb - luma) * saturation
            # Contrast and gamma: on the color channels (per channel)
            if (contrast[:3] != 1.0).any():
                pivot = np.float32(0.18)
                positive = np.maximum(rgb, 0) / pivot
                rgb[:] = np.where(rgb > 0, np.power(positive, contrast[:3]) * pivot, rgb)
            if (gamma[:3] != 1.0).any():
                rgb[:] = np.where(rgb > 0, np.power(np.maximum(rgb, 0), 1.0 / np.maximum(gamma[:3], 1e-6)), rgb)
            out *= gain
            out += offset
            return out

        return processRGBA(ctx, inputs, correct)


class Saturation(desc.ImageOpNode):
    category = "Comp Color"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#7AA9FF"
    documentation = """
Saturation around the Rec. 709 luminance: 0 makes the image grey, values above 1 increase the saturation.
"""
    inputs = colorOpInputs() + [
        _float("saturation", "Saturation", "0: grey, 1: unchanged.", 1.0, (0.0, 4.0, 0.01)),
        mixParam(),
    ]
    outputs = imageOutput("Image with its saturation changed.")

    def isIdentity(self, ctx):
        return "input" if _neutral(ctx, {"saturation": 1.0}) else None

    def render(self, ctx, inputs):
        saturation = np.float32(ctx.value("saturation"))

        def saturate(rgba):
            out = rgba.copy()
            luma = _luminance(rgba)
            out[:, :, :3] = luma + (rgba[:, :, :3] - luma) * saturation
            return out

        return processRGBA(ctx, inputs, saturate)


class HueShift(desc.ImageOpNode):
    category = "Comp Color"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#7AA9FF"
    documentation = """
Rotates the hue (in degrees) while keeping the luminance, and scales the saturation.
"""
    inputs = colorOpInputs() + [
        _float("hueRotation", "Hue Rotation", "Hue rotation in degrees.", 0.0, (-180.0, 180.0, 0.5)),
        _float("saturation", "Saturation", "Saturation multiplier.", 1.0, (0.0, 4.0, 0.01)),
        mixParam(),
    ]
    outputs = imageOutput("Image with its hue shifted.")

    def isIdentity(self, ctx):
        return "input" if _neutral(ctx, {"hueRotation": 0.0, "saturation": 1.0}) else None

    def render(self, ctx, inputs):
        # Rotation in the YIQ space: Y (luminance) is kept, the IQ chroma plane is rotated and scaled
        angle = np.radians(ctx.value("hueRotation"))
        saturation = ctx.value("saturation")
        toYiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
        rotation = np.array([[1, 0, 0],
                             [0, np.cos(angle) * saturation, -np.sin(angle) * saturation],
                             [0, np.sin(angle) * saturation, np.cos(angle) * saturation]])
        matrix = (np.linalg.inv(toYiq) @ rotation @ toYiq).T.astype(np.float32)

        def shift(rgba):
            out = rgba.copy()
            out[:, :, :3] = rgba[:, :, :3] @ matrix
            return out

        return processRGBA(ctx, inputs, shift)


class Exposure(desc.ImageOpNode):
    category = "Comp Color"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#7AA9FF"
    documentation = """
Exposure change in stops: the color is multiplied by 2^stops.
"""
    inputs = colorOpInputs() + [
        _float("stops", "Stops", "Exposure change in stops (f-stops).", 0.0, (-6.0, 6.0, 0.1)),
        mixParam(),
    ]
    outputs = imageOutput("Exposed image.")

    def isIdentity(self, ctx):
        return "input" if _neutral(ctx, {"stops": 0.0}) else None

    def render(self, ctx, inputs):
        factor = np.float32(2.0 ** ctx.value("stops"))
        return processRGBA(ctx, inputs, lambda rgba: rgba * factor)


class Multiply(desc.ImageOpNode):
    category = "Comp Color"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#7AA9FF"
    documentation = """
Multiplies the selected channels by a value.
"""
    inputs = colorOpInputs() + [
        _float("value", "Value", "Multiplier.", 1.0, (0.0, 4.0, 0.01)),
        mixParam(),
    ]
    outputs = imageOutput("Multiplied image.")

    def isIdentity(self, ctx):
        return "input" if _neutral(ctx, {"value": 1.0}) else None

    def render(self, ctx, inputs):
        value = np.float32(ctx.value("value"))
        return processRGBA(ctx, inputs, lambda rgba: rgba * value)


class Add(desc.ImageOpNode):
    category = "Comp Color"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#7AA9FF"
    documentation = """
Adds a value to the selected channels.
"""
    inputs = colorOpInputs() + [
        _float("value", "Value", "Value to add.", 0.0, (-1.0, 1.0, 0.01)),
        mixParam(),
    ]
    outputs = imageOutput("Image with the value added.")

    def isIdentity(self, ctx):
        return "input" if _neutral(ctx, {"value": 0.0}) else None

    def render(self, ctx, inputs):
        value = np.float32(ctx.value("value"))
        return processRGBA(ctx, inputs, lambda rgba: rgba + value)


class Clamp(desc.ImageOpNode):
    category = "Comp Color"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#7AA9FF"
    documentation = """
Clamps the selected channels between a minimum and a maximum.
"""
    inputs = colorOpInputs(defaultChannels="rgba") + [
        _float("minimum", "Minimum", "Lower limit.", 0.0, (-1.0, 1.0, 0.01)),
        desc.BoolParam(name="clampMinimum", label="Clamp Minimum", description="Apply the lower limit.", value=True),
        _float("maximum", "Maximum", "Upper limit.", 1.0, (0.0, 4.0, 0.01)),
        desc.BoolParam(name="clampMaximum", label="Clamp Maximum", description="Apply the upper limit.", value=True),
        mixParam(),
    ]
    outputs = imageOutput("Clamped image.")

    def isIdentity(self, ctx):
        if ctx.value("mix") <= 0.0 or not (ctx.value("clampMinimum") or ctx.value("clampMaximum")):
            return "input"
        return None

    def render(self, ctx, inputs):
        low = ctx.value("minimum") if ctx.value("clampMinimum") else None
        high = ctx.value("maximum") if ctx.value("clampMaximum") else None
        return processRGBA(ctx, inputs, lambda rgba: np.clip(rgba, low, high))


class Invert(desc.ImageOpNode):
    category = "Comp Color"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#7AA9FF"
    documentation = """
Inverts the selected channels: 1 - value.
"""
    inputs = colorOpInputs() + [mixParam()]
    outputs = imageOutput("Inverted image.")

    def isIdentity(self, ctx):
        return "input" if ctx.value("mix") <= 0.0 else None

    def render(self, ctx, inputs):
        return processRGBA(ctx, inputs, lambda rgba: 1.0 - rgba)


_IDENTITY = [[0.0, 0.0], [1.0, 1.0]]


def _lookupInputs(curveNames, semanticDescription):
    """ Inputs shared by the lookup nodes: curves, pickers, color matching. """
    import json
    from meshroom.live.params import colorParam
    return colorOpInputs(defaultChannels="rgba", channelValues=("rgba", "rgb", "alpha")) + [
        desc.StringParam(name="curves", label="Curves", semantic="colorCurves",
                         value=json.dumps({name: _IDENTITY for name in curveNames}),
                         description=semanticDescription),
        colorParam("sourceColor", "Source Color", "Color to change, picked in the viewer.", (0.18, 0.18, 0.18, 1.0)),
        colorParam("targetColor", "Target Color", "Color it should become, picked in the viewer.", (0.18, 0.18, 0.18, 1.0)),
        desc.BoolParam(name="matchSource", label="Match Source",
                       description="Show the 'Match Target' input: the 'Match' button of the curves makes the colors "
                                   "of the input match those of this second image.", value=False),
        desc.Image(name="matchTarget", label="Match Target", optional=True, infoOnly=True,
                   description="Image whose colors are matched (with the 'Match' button of the curves).",
                   enabled=lambda node: node.matchSource.value),
        mixParam(),
    ]


def _curves(ctx, names):
    from meshroom.live import curves
    try:
        parsed = curves.parseCurves(ctx.value("curves"))
    except ValueError as e:
        raise RuntimeError(f"Invalid curves: {e}") from e
    return {name: parsed.get(name, _IDENTITY) for name in names}


class ColorLookup(desc.ImageOpNode):
    category = "Comp Color"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#7AA9FF"
    documentation = """
Color correction by curves, as Nuke's ColorLookup: the "master" curve applies to red, green and blue,
then each channel has its own curve (red, green, blue, alpha). Edit the curves in the attribute
editor: click to add a point, drag the points, double-click one to remove it.

- Pickers: pick a "Source Color" and a "Target Color" in the viewer, then "Add Point" puts on the red,
  green and blue curves the points moving the source to the target. Repeat to refine.
- Match: check "Match Source", connect a second image to "Match Target", then "Match" sets the
  curves so that the colors of the input match those of the second image (histogram matching).
"""
    _NAMES = ("master", "red", "green", "blue", "alpha")
    inputs = _lookupInputs(_NAMES, "Curves (JSON): master, red, green, blue, alpha.")
    outputs = imageOutput("Corrected image.")
    valuesOnlyInputs = ("matchTarget",)

    def isIdentity(self, ctx):
        from meshroom.live import curves
        if ctx.value("mix") <= 0.0:
            return "input"
        try:
            values = _curves(ctx, self._NAMES)
        except RuntimeError:
            return None
        return "input" if all(curves.isIdentity(points) for points in values.values()) else None

    def render(self, ctx, inputs):
        from meshroom.live import curves
        values = _curves(ctx, self._NAMES)

        def lookup(rgba):
            out = rgba.copy()
            for index, name in enumerate(("red", "green", "blue")):
                master = curves.evaluate(values["master"], rgba[:, :, index])
                out[:, :, index] = curves.evaluate(values[name], master)
            out[:, :, 3] = curves.evaluate(values["alpha"], rgba[:, :, 3])
            return out

        return processRGBA(ctx, inputs, lookup)


class ColorLookupHSV(desc.ImageOpNode):
    category = "Comp Color"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#7AA9FF"
    documentation = """
Color correction by curves in HSV: the "hue" curve remaps the hue (0-1 around the color wheel, e.g.
a point (0.33, 0.4) turns greens towards cyan), the "saturation" and "value" curves remap the
saturation and the value (brightness).

The pickers ("Add Point" from a source to a target color) and the color matching ("Match Source")
work as in ColorLookup, on the hue, saturation and value curves.
"""
    _NAMES = ("hue", "saturation", "value")
    inputs = _lookupInputs(_NAMES, "Curves (JSON): hue, saturation, value.")
    outputs = imageOutput("Corrected image.")
    valuesOnlyInputs = ("matchTarget",)

    def isIdentity(self, ctx):
        from meshroom.live import curves
        if ctx.value("mix") <= 0.0:
            return "input"
        try:
            values = _curves(ctx, self._NAMES)
        except RuntimeError:
            return None
        return "input" if all(curves.isIdentity(points) for points in values.values()) else None

    def render(self, ctx, inputs):
        from meshroom.live import curves
        values = _curves(ctx, self._NAMES)

        def lookup(rgba):
            hsv = curves.rgbToHsv(rgba[:, :, :3].astype(np.float64))
            hsv[..., 0] = curves.evaluate(values["hue"], hsv[..., 0]) % 1.0
            hsv[..., 1] = np.maximum(curves.evaluate(values["saturation"], hsv[..., 1]), 0.0)
            hsv[..., 2] = curves.evaluate(values["value"], hsv[..., 2])
            out = rgba.copy()
            out[:, :, :3] = curves.hsvToRgb(hsv)
            return out

        return processRGBA(ctx, inputs, lookup)


class OCIOColorSpace(desc.ImageOpNode):
    category = "Comp Color"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#7AA9FF"
    documentation = """
Converts the colors from a colorspace to another, with the color config: the OpenColorIO config of the
"OCIO" environment variable (when PyOpenColorIO is installed), else the builtin colorspaces (linear,
sRGB, rec709, Gamma, Cineon, AlexaV3LogC, ACES...).

The images are processed in the working space ("linear", or the "scene_linear" role of the OCIO
config): e.g. "linear" to "Cineon" to grade in log, then "Cineon" to "linear" to come back.
"""
    inputs = colorOpInputs() + [
        colorspaceParam("inColorspace", "In", "Colorspace of the input.", "linear"),
        colorspaceParam("outColorspace", "Out", "Colorspace of the output.", "sRGB"),
        mixParam(),
    ]
    outputs = imageOutput("Converted image.")

    def isIdentity(self, ctx):
        from meshroom.live import color
        if ctx.value("mix") <= 0.0:
            return "input"
        try:
            if color.config().isIdentity(ctx.value("inColorspace"), ctx.value("outColorspace")):
                return "input"
        except Exception:
            # An unknown colorspace: the render reports it
            pass
        return None

    def render(self, ctx, inputs):
        from meshroom.live import color
        processor = color.config().processor(ctx.value("inColorspace"), ctx.value("outColorspace"))

        def convert(rgba):
            out = rgba.copy()
            out[:, :, :3] = processor(rgba[:, :, :3])
            return out
        return processRGBA(ctx, inputs, convert)
