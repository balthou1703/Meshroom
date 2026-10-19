__version__ = "0.3"

import os

from meshroom.core import desc
from meshroom.live.params import colorspaceParam


class Read(desc.ImageOpNode):
    category = "Comp Image"
    color = "#7B82A8"
    documentation = """
Reads an image or an image sequence, with all its layers (EXR AOVs need OpenImageIO).

Sequences use "#" (one per digit), "@" or printf ("%04d") placeholders, e.g. `render.####.exr`.
"Frame Offset" moves the sequence in time (e.g. +10: file frame 1001 is shown at frame 1011).
Outside of the frame range, the first or last frame is held, or the image is black.
"""

    inputs = [
        desc.File(
            name="file",
            label="File",
            description="Image file or sequence (e.g. render.####.exr).",
            value="",
            semantic="imageSequence",
            exposed=True,
        ),
        colorspaceParam(
            "colorspace", "Colorspace",
            "Colorspace of the file (of the color config: OCIO or builtin). 'auto' reads 8/16-bit integer files as sRGB "
            "and float files as the working space (linear)."),
        desc.ChoiceParam(
            name="missingFrames",
            label="Missing Frames",
            description="Behavior when a frame of the sequence is missing inside the frame range.",
            value="error",
            values=["error", "black"],
        ),
        desc.IntParam(
            name="frameOffset",
            label="Frame Offset",
            description="Moves the sequence in time: frames read from the file are shown this many frames later "
                        "(negative: earlier).",
            value=0,
            range=(-1000, 1000, 1),
        ),
        desc.ChoiceParam(
            name="outsideRange",
            label="Outside Range",
            description="Image before the first frame and after the last frame of the sequence.",
            value="hold",
            values=["hold", "black"],
        ),
    ]

    outputs = [
        desc.Image(
            name="output",
            label="Output",
            description="Image read from the file.",
        ),
    ]

    def _frameRange(self, path):
        from meshroom.live import imageIO
        if not imageIO.isSequencePath(path):
            return None
        frameRange = imageIO.detectFrameRange(path)
        if frameRange is None:
            raise RuntimeError(f"No frame found for the sequence: {path}")
        return frameRange

    def getInfo(self, ctx, inputsInfo):
        from meshroom.live import ImageInfo, imageIO
        path = ctx.value("file")
        if not path:
            raise RuntimeError("No file set.")
        frameRange = self._frameRange(path)
        if frameRange is None:
            width, height = imageIO.readImageSize(path)
            return self._proxyInfo(ctx, ImageInfo(width, height))
        width, height = imageIO.readImageSize(imageIO.framePath(path, frameRange[0]))
        offset = ctx.value("frameOffset")
        return self._proxyInfo(ctx, ImageInfo(width, height, frameRange[0] + offset, frameRange[1] + offset))

    @staticmethod
    def _proxyFactor(ctx):
        """ The reduction of the images at the proxy scale of the render (1: full resolution). """
        return max(1, int(round(1.0 / ctx.proxy))) if ctx.proxy < 1.0 else 1

    def _proxyInfo(self, ctx, info):
        import math
        factor = self._proxyFactor(ctx)
        if factor == 1:
            return info
        return info.withFormat(int(math.ceil(info.width / factor)), int(math.ceil(info.height / factor)))

    def render(self, ctx, inputs):
        from meshroom.live import ImageBuffer, imageIO
        path = ctx.value("file")
        if not path:
            raise RuntimeError("No file set.")
        info = ctx.info()
        if info.hasFrameRange:
            if not info.firstFrame <= ctx.frame <= info.lastFrame and ctx.value("outsideRange") == "black":
                return ImageBuffer.constant(info.width, info.height, (0, 0, 0, 0))
            # Hold the first / last frame outside of the frame range
            frame = min(max(ctx.frame, info.firstFrame), info.lastFrame)
            path = imageIO.framePath(path, frame - ctx.value("frameOffset"))
            if not os.path.isfile(path) and ctx.value("missingFrames") == "black":
                return ImageBuffer.constant(info.width, info.height, (0, 0, 0, 0))
        from meshroom.live import ops
        return ops.downscaleImage(imageIO.readImage(path, ctx.value("colorspace")), self._proxyFactor(ctx))
