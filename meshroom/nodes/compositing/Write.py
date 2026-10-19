__version__ = "0.2"

import logging

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
from meshroom.live.params import colorspaceParam


def _connectedImageNode(node):
    """ Return the live node connected to the "image" input of a Write node, or None. """
    attr = node.attribute("image")
    if not attr.isLink:
        return None
    return attr.inputRootLink.node


def writeFrameRange(node):
    """
    Return the (first, last) frames to write: the custom range, or the frame range of the
    connected image (a single frame, 1, for still images).
    """
    if node.frameRangeMode.value == "custom":
        return node.firstFrame.value, node.lastFrame.value
    source = _connectedImageNode(node)
    if source is None:
        return 1, 1
    from meshroom.live import Evaluator, buildPlan
    info = Evaluator().info(buildPlan(source))
    if not info.hasFrameRange:
        return 1, 1
    return info.firstFrame, info.lastFrame


def writeSize(node):
    """ Number of frames to write, used to split the computation into chunks. """
    try:
        first, last = writeFrameRange(node)
    except Exception as e:
        logging.debug(f"[Write] Cannot determine the frame range of '{node.name}': {e}")
        return 1
    return max(1, last - first + 1)


class Write(desc.Node):
    size = writeSize
    parallelization = desc.Parallelization(blockSize=20)
    category = "Comp Image"
    color = "#B85C5C"
    documentation = """
Renders the connected compositing branch to image files.

Unlike the other compositing nodes, Write is a regular Meshroom node: it is computed with
"Compute" or submitted to a render farm, and split into chunks of frames. Each chunk evaluates
the branch in memory for its frames and writes them.

With OpenImageIO, EXR files contain every layer of the image.
"""

    inputs = [
        desc.Image(
            name="image",
            label="Image",
            description="Image to write.",
        ),
        desc.File(
            name="file",
            label="File",
            description="Output file; use '####' for a sequence (e.g. /path/render.####.exr).",
            value="",
            exposed=True,
        ),
        desc.ChoiceParam(
            name="dataType",
            label="Data Type",
            description="Pixel data type of the file.",
            value="half",
            values=["half", "float", "uint8", "uint16"],
        ),
        colorspaceParam(
            "colorspace", "Colorspace",
            "Colorspace of the file (of the color config: OCIO or builtin). 'auto' writes 8/16-bit integer files as sRGB "
            "and float files as the working space (linear)."),
        desc.ChoiceParam(
            name="frameRangeMode",
            label="Frame Range",
            description="Write the frame range of the input, or a custom one.",
            value="input",
            values=["input", "custom"],
        ),
        desc.IntParam(
            name="firstFrame",
            label="First Frame",
            description="First frame to write.",
            value=1,
            range=None,
            enabled=lambda node: node.frameRangeMode.value == "custom",
        ),
        desc.IntParam(
            name="lastFrame",
            label="Last Frame",
            description="Last frame to write.",
            value=1,
            range=None,
            enabled=lambda node: node.frameRangeMode.value == "custom",
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Output",
            description="Written file or sequence.",
            value="{fileValue}",
        ),
    ]

    def processChunk(self, chunk):
        from meshroom.live import Evaluator, ImageCache, buildPlan, imageIO
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            node = chunk.node
            path = node.file.value
            if not path:
                raise RuntimeError("No output file set.")
            source = _connectedImageNode(node)
            if source is None:
                raise RuntimeError("The 'image' input is not connected.")
            first, last = writeFrameRange(node)
            frames = list(range(first, last + 1))
            chunkFrames = frames[chunk.range.start:min(chunk.range.end, len(frames))]
            isSequence = imageIO.isSequencePath(path)
            if len(frames) > 1 and not isSequence:
                raise RuntimeError(f"Several frames to write but no '####' in the file name: {path}")

            # A small cache: frames are rendered once, only the upstream stills benefit from it
            evaluator = Evaluator(ImageCache(1024 ** 3))
            plan = buildPlan(source)
            for frame in chunkFrames:
                image = evaluator.render(plan, frame)
                framePath = imageIO.framePath(path, frame) if isSequence else path
                imageIO.writeImage(framePath, image, node.dataType.value, node.colorspace.value)
                chunk.logger.info(f"Frame {frame} written: {framePath}")
        finally:
            chunk.logManager.end()
