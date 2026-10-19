__version__ = "0.1"

from meshroom.core import desc
from meshroom.live.params import colorParam, formatInput, formatParams, outputInfo


class Constant(desc.ImageOpNode):
    category = "Comp Image"
    color = "#7B82A8"
    documentation = """
Generates an image of a constant color, the same at every frame. Its format comes from its
optional "Format" input when connected (the pixels of the input are not used).
"""

    inputs = [
        formatInput(),
        # Not "color": this name is taken by the node color in the graph editor
        colorParam("rgba", "Color", "RGBA color (linear, premultiplied).", (0.0, 0.0, 0.0, 0.0)),
        *formatParams(),
    ]

    outputs = [
        desc.Image(
            name="output",
            label="Output",
            description="Constant image.",
        ),
    ]

    def getInfo(self, ctx, inputsInfo):
        return outputInfo(ctx, inputsInfo)

    def render(self, ctx, inputs):
        from meshroom.live import ImageBuffer
        color = ctx.value("rgba")
        info = ctx.info()
        return ImageBuffer.constant(info.width, info.height,
                                    (color["r"], color["g"], color["b"], color["a"]))
