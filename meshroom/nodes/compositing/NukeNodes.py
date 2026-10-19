"""
Nodes of an imported Nuke script that have no equivalent (meshroom/live/nuke.py).
"""
__version__ = "0.1"

from meshroom.core import desc
from meshroom.live.params import imageInput, imageOutput


class NukeUnsupported(desc.ImageOpNode):
    category = "Comp Other"
    color = "#8A8A8A"
    documentation = """
A node of an imported Nuke script without an equivalent here: it passes its first input through and
keeps the class and the knobs of the Nuke node (to rebuild it by hand, or for a later import).
"""
    inputs = [
        imageInput(optional=True),
        imageInput("input2", "Input 2", optional=True),
        imageInput("input3", "Input 3", optional=True),
        imageInput("input4", "Input 4", optional=True),
        desc.StringParam(name="nukeClass", label="Nuke Class", value="", invalidate=False,
                         description="Class of the Nuke node."),
        desc.StringParam(name="knobs", label="Knobs", value="", semantic="multiline", invalidate=False,
                         description="Knobs of the Nuke node, as written in the script."),
    ]
    outputs = imageOutput("The first input, unchanged.")

    def isIdentity(self, ctx):
        return "input" if ctx.inputInfo("input") is not None else None

    def getInfo(self, ctx, inputsInfo):
        from meshroom.live import ImageInfo
        info = inputsInfo.get("input")
        return info if info is not None else ImageInfo(1920, 1080)

    def render(self, ctx, inputs):
        from meshroom.live import ImageBuffer
        import numpy as np
        info = ctx.info()
        # No input: transparent black
        return ImageBuffer.fromRGBA(np.zeros((info.height, info.width, 4), np.float32), (info.width, info.height))
