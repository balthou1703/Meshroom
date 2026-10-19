"""
Channel nodes (Nuke "Channel" and "Merge" menus): Shuffle, Copy, Premult, Unpremult.
"""
__version__ = "0.1"

import numpy as np

from meshroom.core import desc
from meshroom.live.params import imageInput, imageOutput

_SOURCES = ["red", "green", "blue", "alpha", "black", "white"]
_INDEX = {"red": 0, "green": 1, "blue": 2, "alpha": 3}


def _sourceChannel(name, label, value):
    return desc.ChoiceParam(name=name, label=label, description=f"Source of the output {label.lower()} channel.",
                            value=value, values=_SOURCES)


class Shuffle(desc.ImageOpNode):
    category = "Comp Channel"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#9E3C63"
    documentation = """
Rearranges the channels: each output channel of the "rgba" layer takes a channel of the input layer
(by default "rgba"; any layer of the image, e.g. an AOV, can be used), or black / white.
"""
    inputs = [
        imageInput(),
        desc.StringParam(name="inputLayer", label="Input Layer", value="rgba", semantic="imageLayer",
                         description="Layer read by the shuffle (e.g. rgba, diffuse, depth): the list shows the "
                                     "layers of the input image once it has been rendered."),
        _sourceChannel("red", "Red", "red"),
        _sourceChannel("green", "Green", "green"),
        _sourceChannel("blue", "Blue", "blue"),
        _sourceChannel("alpha", "Alpha", "alpha"),
    ]
    outputs = imageOutput("Shuffled image.")

    def isIdentity(self, ctx):
        if ctx.value("inputLayer") == "rgba" and \
                [ctx.value(c) for c in ("red", "green", "blue", "alpha")] == ["red", "green", "blue", "alpha"]:
            return "input"
        return None

    def render(self, ctx, inputs):
        image = inputs["input"]
        layerName = ctx.value("inputLayer") or "rgba"
        if not image.hasLayer(layerName):
            raise RuntimeError(f"No layer '{layerName}' in the input (layers: {', '.join(image.layerNames)}).")
        channels = image.channels(layerName)
        pixels = image.layer(layerName)
        names = {"red": ("R", "X"), "green": ("G", "Y"), "blue": ("B", "Z"), "alpha": ("A",)}

        def source(choice):
            if choice == "black":
                return 0.0
            if choice == "white":
                return 1.0
            for candidate in names[choice]:
                if candidate in channels:
                    return pixels[:, :, channels.index(candidate)]
            # A single-channel layer (e.g. depth) feeds every channel
            if len(channels) == 1:
                return pixels[:, :, 0]
            index = _INDEX[choice]
            return pixels[:, :, index] if index < len(channels) else 0.0

        out = np.empty(pixels.shape[:2] + (4,), dtype=np.float32)
        for i, channel in enumerate(("red", "green", "blue", "alpha")):
            out[:, :, i] = source(ctx.value(channel))
        return image.withRGBA(out)


class Copy(desc.ImageOpNode):
    category = "Comp Channel"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#9E3C63"
    documentation = """
Copies a channel of A into a channel of B (by default the alpha of A into the alpha of B).
The result has the format and bbox of B.
"""
    inputs = [
        imageInput("B", "B", "Image receiving the channel."),
        imageInput("A", "A", "Image the channel is copied from."),
        desc.ChoiceParam(name="fromChannel", label="From", description="Channel of A.", value="alpha",
                         values=["red", "green", "blue", "alpha"]),
        desc.ChoiceParam(name="toChannel", label="To", description="Channel of B.", value="alpha",
                         values=["red", "green", "blue", "alpha"]),
    ]
    outputs = imageOutput("B with the copied channel.")

    def render(self, ctx, inputs):
        from meshroom.live import ops
        imageB, imageA = inputs["B"], inputs["A"]
        source = imageA.rgba()[:, :, _INDEX[ctx.value("fromChannel")]:_INDEX[ctx.value("fromChannel")] + 1]
        source = ops.alignToBBox(np.ascontiguousarray(source), imageA.bbox, imageB.bbox)
        out = imageB.rgba()
        out[:, :, _INDEX[ctx.value("toChannel")]] = source[:, :, 0]
        return imageB.withRGBA(out)


class Premult(desc.ImageOpNode):
    category = "Comp Merge"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#4B5EC6"
    documentation = """
Multiplies the color by the alpha.
"""
    inputs = [imageInput()]
    outputs = imageOutput("Premultiplied image.")

    def render(self, ctx, inputs):
        from meshroom.live import ops
        image = inputs["input"]

        def band(rgba):
            out = rgba.copy()
            out[:, :, :3] *= rgba[:, :, 3:4]
            return out
        return image.withRGBA(ops.mapRows(band, [image.rgba()], 4, ctx.checkCancelled))


class Unpremult(desc.ImageOpNode):
    category = "Comp Merge"
    # Region of interest: the same region of the inputs
    pointwise = True
    color = "#4B5EC6"
    documentation = """
Divides the color by the alpha (where the alpha is not 0).
"""
    inputs = [imageInput()]
    outputs = imageOutput("Unpremultiplied image.")

    def render(self, ctx, inputs):
        from meshroom.live import ops
        image = inputs["input"]

        def band(rgba):
            out = rgba.copy()
            alpha = rgba[:, :, 3:4]
            np.divide(rgba[:, :, :3], alpha, out=out[:, :, :3], where=alpha != 0)
            return out
        return image.withRGBA(ops.mapRows(band, [image.rgba()], 4, ctx.checkCancelled))


class Dot(desc.ImageOpNode):
    category = "Comp Merge"
    color = "#9A9A9A"
    # Nodes reading the values of their input (ctx.inputValues) look through it
    passThroughValues = True
    documentation = """
Passes its input through: used to route the links of the graph.
"""
    inputs = [imageInput()]
    outputs = imageOutput("The input, unchanged.")

    def isIdentity(self, ctx):
        return "input"

    def render(self, ctx, inputs):
        return inputs["input"]


class Clone(desc.ImageOpNode):
    category = "Comp Merge"
    color = "#9A9A9A"
    passThroughValues = True
    # Created from the node menu ("Clone"): its hidden input cannot be connected by hand
    hiddenFromMenu = True
    documentation = """
A clone of another node: it looks disconnected (its link to the cloned node is hidden), but it
outputs the result of the cloned node, computed with the whole graph behind it. Use it to reuse
a result far away in the graph without long links. Create it with the "Clone" item of the node
menu; its name shows the cloned node.
"""
    inputs = [
        desc.Image(name="source", label="Source", description="The cloned node (hidden link).", visible=False),
    ]
    outputs = imageOutput("Result of the cloned node.")

    def isIdentity(self, ctx):
        return "source"
