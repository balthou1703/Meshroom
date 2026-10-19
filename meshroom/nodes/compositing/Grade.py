__version__ = "0.3"

from meshroom.core import Version, desc
from meshroom.live.params import linkedColor, linkedColorValues, maskParams, upgradeLinkedColors


def _gradeParam(name, label, description, value, range=(-1.0, 4.0, 0.01)):
    """ A value of the grade: a single number, or one per channel (r, g, b, a) when split. """
    return linkedColor(name, label, description, value, range)


# Component (r, g, b, a) of the values used for a channel of a layer, by its name (else its position)
_COMPONENTS = {"R": 0, "X": 0, "U": 0, "G": 1, "Y": 1, "V": 1, "B": 2, "Z": 2, "W": 2, "A": 3}


class Grade(desc.ImageOpNode):
    category = "Comp Color"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#7AA9FF"
    documentation = """
Color correction with the Nuke Grade formula:
`A = multiply * (gain - lift) / (whitepoint - blackpoint)`, `B = offset + lift - A * blackpoint`,
`out = (A * in + B) ^ (1 / gamma)`.

Each value is a single number, or one per channel (r, g, b, a) when "split" is checked.
It applies to the channels of the "rgba" layer selected by "Channels" ("all" grades every layer).
With its default values the node passes its input through without computing anything
(the black clamp only applies once a value is changed).
"""

    inputs = [
        desc.Image(
            name="input",
            label="Input",
            description="Image to grade.",
        ),
        desc.Image(
            name="mask",
            label="Mask",
            description="Optional mask: its 'Mask Channel' limits where the grade applies.",
            optional=True,
        ),
        *maskParams(),
        desc.ChoiceParam(
            name="channels",
            label="Channels",
            description="Channels to grade.",
            value="rgb",
            values=["rgb", "rgba", "alpha", "all"],
        ),
        _gradeParam("blackpoint", "Blackpoint", "Input value mapped to the lift.", 0.0),
        _gradeParam("whitepoint", "Whitepoint", "Input value mapped to the gain.", 1.0),
        _gradeParam("lift", "Lift", "Output value of the blackpoint.", 0.0),
        _gradeParam("gain", "Gain", "Output value of the whitepoint.", 1.0),
        _gradeParam("multiply", "Multiply", "Multiplier applied after the lift/gain mapping.", 1.0),
        _gradeParam("offset", "Offset", "Value added after the multiply.", 0.0),
        _gradeParam("gamma", "Gamma", "Gamma applied last (to positive values).", 1.0, (0.2, 5.0, 0.01)),
        desc.BoolParam(
            name="blackClamp",
            label="Black Clamp",
            description="Clamp negative output values to 0.",
            value=True,
        ),
        desc.BoolParam(
            name="whiteClamp",
            label="White Clamp",
            description="Clamp output values above 1 to 1.",
            value=False,
        ),
        desc.FloatParam(
            name="mix",
            label="Mix",
            description="Blend between the input (0) and the graded image (1).",
            value=1.0,
            range=(0.0, 1.0, 0.01),
        ),
    ]

    outputs = [
        desc.Image(
            name="output",
            label="Output",
            description="Graded image.",
        ),
    ]

    _NEUTRAL = {"blackpoint": 0.0, "whitepoint": 1.0, "lift": 0.0, "gain": 1.0, "multiply": 1.0, "offset": 0.0, "gamma": 1.0}

    def upgradeAttributeValues(self, attrValues, fromVersion):
        # Before 0.3, the values were single floats
        if fromVersion < Version("0.3"):
            upgradeLinkedColors(attrValues, self._NEUTRAL)
        return attrValues

    def isIdentity(self, ctx):
        if ctx.value("mix") <= 0.0:
            return "input"
        if ctx.value("whiteClamp"):
            return None
        if all((linkedColorValues(ctx, name) == value).all() for name, value in self._NEUTRAL.items()):
            return "input"
        return None

    def render(self, ctx, inputs):
        import numpy as np
        from meshroom.live import RGBA, ops
        from meshroom.live.params import maskAmount
        image = inputs["input"]
        values = {name: linkedColorValues(ctx, name) for name in self._NEUTRAL}
        clamps = {name: ctx.value(name) for name in ("blackClamp", "whiteClamp")}
        selection = ctx.value("channels")
        amount = maskAmount(ctx, inputs, image.bbox, ctx.value("mix"))

        def apply(pixels, channels):
            selected = ops.channelMask(channels, selection)
            if not selected.any():
                return pixels
            # The values of each selected channel
            components = [_COMPONENTS.get(name.upper(), index % 4) for index, name in enumerate(channels)]
            components = np.array(components)[selected]
            params = {name: value[components] for name, value in values.items()}

            def band(pixelsBand, amountBand):
                out = pixelsBand.copy()
                out[:, :, selected] = ops.grade(pixelsBand[:, :, selected], **params, **clamps)
                return ops.mix(pixelsBand, out, amountBand)
            return ops.mapRows(band, [pixels, amount], pixels.shape[2], ctx.checkCancelled)

        result = image
        for name, channels, pixels in image.layers():
            if selection == "all" or name == RGBA:
                result = result.withLayer(name, apply(pixels, channels), channels)
        return result
