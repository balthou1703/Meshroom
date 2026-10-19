"""
Keying nodes (Nuke "Keyer" menu): Keyer (luminance, channel or saturation key with a range),
ChromaKeyer (green / blue screen color difference key with despill) and Keylight (screen color
picked in the viewer, garbage / core mattes, matte views).
"""
__version__ = "0.1"

import numpy as np

from meshroom.core import desc
from meshroom.live.params import colorParam, colorValue, imageInput, imageOutput

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _float(name, label, description, value, range):
    return desc.FloatParam(name=name, label=label, description=description, value=value, range=range)


class Keyer(desc.ImageOpNode):
    category = "Comp Keyer"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#6CC77B"
    documentation = """
Pulls a key into the alpha from a value of each pixel, like the Nuke Keyer: the key is 0 below A,
rises to 1 between A and B, stays 1 between B and C, falls to 0 between C and D.
The output can be premultiplied by the key.
"""
    inputs = [
        imageInput(),
        desc.ChoiceParam(name="operation", label="Operation", description="Value the key is computed from.",
                         value="luminance", values=["luminance", "red", "green", "blue", "alpha", "saturation", "max"]),
        _float("a", "A", "Start of the rising ramp.", 0.0, (0.0, 1.0, 0.01)),
        _float("b", "B", "End of the rising ramp.", 1.0, (0.0, 1.0, 0.01)),
        _float("c", "C", "Start of the falling ramp.", 1.0, (0.0, 2.0, 0.01)),
        _float("d", "D", "End of the falling ramp.", 1.0, (0.0, 2.0, 0.01)),
        desc.BoolParam(name="invert", label="Invert", description="Invert the key.", value=False),
        desc.ChoiceParam(name="outputMode", label="Output", value="alpha", values=["alpha", "premultiplied"],
                         description="Write the key into the alpha, or also premultiply the color by it."),
    ]
    outputs = imageOutput("Image with the key in its alpha.")

    def render(self, ctx, inputs):
        from meshroom.live import ops
        image = inputs["input"]
        operation = ctx.value("operation")
        a, b, c, d = (ctx.value(k) for k in "abcd")
        invert, premultiply = ctx.value("invert"), ctx.value("outputMode") == "premultiplied"

        def band(rgba):
            if operation == "luminance":
                value = rgba[:, :, :3] @ _LUMA
            elif operation == "saturation":
                high, low = rgba[:, :, :3].max(axis=2), rgba[:, :, :3].min(axis=2)
                value = np.where(high > 0, (high - low) / np.maximum(high, 1e-6), 0)
            elif operation == "max":
                value = rgba[:, :, :3].max(axis=2)
            else:
                value = rgba[:, :, {"red": 0, "green": 1, "blue": 2, "alpha": 3}[operation]]
            rising = np.clip((value - a) / max(b - a, 1e-6), 0, 1) if b > a else (value >= a).astype(np.float32)
            falling = np.clip((d - value) / max(d - c, 1e-6), 0, 1) if d > c else (value <= d).astype(np.float32)
            key = np.minimum(rising, falling)
            if invert:
                key = 1 - key
            out = rgba.copy()
            out[:, :, 3] = key
            if premultiply:
                out[:, :, :3] *= key[:, :, np.newaxis]
            return out

        return image.withRGBA(ops.mapRows(band, [image.rgba()], 4, ctx.checkCancelled))


class ChromaKeyer(desc.ImageOpNode):
    category = "Comp Keyer"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#6CC77B"
    documentation = """
Green or blue screen key by color difference: the screen channel minus a mix of the two others
(balanced by "Balance") gives the screen amount, normalized by the screen color.
- Clip Black / Clip White tighten the matte;
- Despill limits the screen channel to the mix of the two others;
- the output is premultiplied by the matte.
"""
    inputs = [
        imageInput(),
        desc.ChoiceParam(name="screen", label="Screen", description="Color of the screen.", value="green",
                         values=["green", "blue"]),
        colorParam("screenColor", "Screen Color", "Sampled color of the screen (0 to use a pure green / blue).",
                   (0.0, 0.0, 0.0, 1.0)),
        _float("balance", "Balance", "Weight of red (vs the other channel) in the color difference.", 0.5,
               (0.0, 1.0, 0.01)),
        _float("gain", "Gain", "Strength of the key.", 1.0, (0.0, 4.0, 0.01)),
        _float("clipBlack", "Clip Black", "Matte values below this become 0.", 0.0, (0.0, 1.0, 0.01)),
        _float("clipWhite", "Clip White", "Matte values above this become 1.", 1.0, (0.0, 1.0, 0.01)),
        desc.BoolParam(name="despill", label="Despill", description="Remove the screen color spill.", value=True),
        desc.BoolParam(name="premultiply", label="Premultiply", description="Premultiply the color by the matte.",
                       value=True),
    ]
    outputs = imageOutput("Keyed image.")

    def render(self, ctx, inputs):
        from meshroom.live import ops
        image = inputs["input"]
        screenIndex = 1 if ctx.value("screen") == "green" else 2
        otherIndex = 2 if screenIndex == 1 else 1
        balance, gain = ctx.value("balance"), ctx.value("gain")
        clipBlack, clipWhite = ctx.value("clipBlack"), ctx.value("clipWhite")
        despill, premultiply = ctx.value("despill"), ctx.value("premultiply")
        screenColor = colorValue(ctx, "screenColor")[:3]
        if max(screenColor) <= 0:
            screenColor = (0.0, 1.0, 0.0) if screenIndex == 1 else (0.0, 0.0, 1.0)
        # Color difference of the screen color itself: the normalization of the key
        screenDifference = screenColor[screenIndex] - (balance * screenColor[0] + (1 - balance) * screenColor[otherIndex])
        screenDifference = max(screenDifference, 1e-4)

        def band(rgba):
            others = balance * rgba[:, :, 0] + (1 - balance) * rgba[:, :, otherIndex]
            difference = rgba[:, :, screenIndex] - others
            matte = 1.0 - np.clip(difference * gain / screenDifference, 0, 1)
            matte = np.clip((matte - clipBlack) / max(clipWhite - clipBlack, 1e-6), 0, 1)
            out = rgba.copy()
            if despill:
                out[:, :, screenIndex] = np.minimum(rgba[:, :, screenIndex], others)
            out[:, :, 3] = matte
            if premultiply:
                out[:, :, :3] *= matte[:, :, np.newaxis]
            return out

        return image.withRGBA(ops.mapRows(band, [image.rgba()], 4, ctx.checkCancelled))


class Keylight(desc.ImageOpNode):
    category = "Comp Keyer"
    color = "#6CC77B"
    documentation = """
A screen keyer in the spirit of Nuke's Keylight. Pick the "Screen Color" in the viewer with the
eyedropper: its dominant channel (green or blue) is the screen channel.

- the screen amount of a pixel is the screen channel minus a mix of the two others ("Screen Balance"),
  normalized by the same difference on the screen color, and multiplied by "Screen Gain";
- "Clip Black" / "Clip White" tighten the matte;
- "Despill" removes the screen color from the foreground (limit of the screen channel);
- the optional "Garbage" matte (its alpha) is removed from the matte, the "Core" matte is added to it;
- "View": the final premultiplied result, the screen matte, the status (black: background,
  white: foreground, grey: transparency) or the despilled foreground before premultiplication.
"""
    inputs = [
        imageInput(description="Image to key."),
        desc.Image(name="garbage", label="Garbage", description="Optional garbage matte (alpha): always background.",
                   optional=True),
        desc.Image(name="core", label="Core", description="Optional core matte (alpha): always foreground.",
                   optional=True),
        desc.ChoiceParam(name="view", label="View", value="final",
                         values=["final", "screenMatte", "status", "intermediate"],
                         description="final: premultiplied result. screenMatte: the matte. status: black / grey / white "
                                     "matte areas. intermediate: despilled foreground, not premultiplied."),
        colorParam("screenColor", "Screen Color", "Color of the screen, picked in the viewer.", (0.1, 0.6, 0.15, 1.0)),
        _float("screenGain", "Screen Gain", "Strength of the key.", 1.05, (0.0, 4.0, 0.01)),
        _float("screenBalance", "Screen Balance",
               "Weight of the first of the two other channels in the color difference.", 0.5, (0.0, 1.0, 0.01)),
        _float("clipBlack", "Clip Black", "Matte values below this become 0.", 0.0, (0.0, 1.0, 0.01)),
        _float("clipWhite", "Clip White", "Matte values above this become 1.", 1.0, (0.0, 1.0, 0.01)),
        _float("despill", "Despill", "Amount of spill removed from the foreground.", 1.0, (0.0, 1.0, 0.01)),
    ]
    outputs = imageOutput("Keyed image.")

    @staticmethod
    def _channels(screenColor):
        """ (screen channel, first other, second other) indices from the dominant channel of the screen color. """
        screen = int(np.argmax(screenColor[:3]))
        others = [i for i in range(3) if i != screen]
        return screen, others[0], others[1]

    def render(self, ctx, inputs):
        from meshroom.live import ops
        image = inputs["input"]
        screenColor = colorValue(ctx, "screenColor")[:3]
        if max(screenColor) <= 0:
            raise RuntimeError("Pick a screen color.")
        screen, other0, other1 = self._channels(screenColor)
        balance, gain = ctx.value("screenBalance"), ctx.value("screenGain")
        clipBlack, clipWhite = ctx.value("clipBlack"), ctx.value("clipWhite")
        despill, view = ctx.value("despill"), ctx.value("view")
        reference = screenColor[screen] - (balance * screenColor[other0] + (1 - balance) * screenColor[other1])
        if reference <= 1e-4:
            raise RuntimeError("The screen color must have a dominant channel (green or blue screen).")
        garbage = ops.maskValues(inputs["garbage"], "alpha", False, image.bbox) if inputs.get("garbage") is not None else None
        core = ops.maskValues(inputs["core"], "alpha", False, image.bbox) if inputs.get("core") is not None else None

        def band(rgba, garbageBand, coreBand):
            others = balance * rgba[:, :, other0] + (1 - balance) * rgba[:, :, other1]
            difference = rgba[:, :, screen] - others
            matte = 1.0 - np.clip(difference * gain / reference, 0, 1)
            matte = np.clip((matte - clipBlack) / max(clipWhite - clipBlack, 1e-6), 0, 1)
            if garbageBand is not None:
                matte = matte * (1.0 - garbageBand[:, :, 0])
            if coreBand is not None:
                matte = np.maximum(matte, coreBand[:, :, 0])
            matte = matte.astype(np.float32)
            out = rgba.copy()
            if despill > 0:
                spill = np.maximum(rgba[:, :, screen] - others, 0)
                out[:, :, screen] = rgba[:, :, screen] - spill * despill
            if view == "screenMatte":
                out[:, :, :3] = matte[:, :, np.newaxis]
            elif view == "status":
                # 0: background, 1: foreground, 0.5: anything in between
                status = np.where(matte <= 1e-3, 0.0, np.where(matte >= 1 - 1e-3, 1.0, 0.5)).astype(np.float32)
                out[:, :, :3] = status[:, :, np.newaxis]
            elif view == "final":
                out[:, :, :3] *= matte[:, :, np.newaxis]
            out[:, :, 3] = matte if view != "status" else 1.0
            return out

        return image.withRGBA(ops.mapRows(band, [image.rgba(), garbage, core], 4, ctx.checkCancelled))
