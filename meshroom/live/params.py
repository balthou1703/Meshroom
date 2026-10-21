"""
Parameter descriptions shared by the compositing nodes.
"""
from meshroom.core import desc

from .ops import MASK_CHANNELS


def maskParams():
    """ Parameters choosing the channel of the "mask" input used to limit the effect of a node. """
    return [
        desc.ChoiceParam(
            name="maskChannel",
            label="Mask Channel",
            description="Channel of the 'mask' input that limits the effect: alpha, red, green, blue or luminance.",
            value="alpha",
            values=list(MASK_CHANNELS),
        ),
        desc.BoolParam(
            name="invertMask",
            label="Invert Mask",
            description="Invert the mask.",
            value=False,
        ),
    ]


def maskAmount(ctx, inputs, bbox, amount):
    """
    Return the blend amount of a node over `bbox`: `amount` (its "mix") multiplied by the mask
    channel of the "mask" input when it is connected.
    """
    from .ops import maskValues
    mask = inputs.get("mask")
    if mask is None:
        return amount
    return maskValues(mask, ctx.value("maskChannel"), ctx.value("invertMask"), bbox) * amount


def linkedVec2(name, label, description, value, range, advanced=False):
    """
    A 2D value edited as a single number by default, which can be split into x and y with the "+"
    button of the attribute editor (like a Nuke knob with "2" channels). Its value is {"x": ..., "y": ...}.
    """
    return desc.Vec2d(name=name, label=label, description=description, x=value, y=value, xRange=range,
                      yRange=range, semantic="linkedVec2", advanced=advanced)


def imageInput(name="input", label="Input", description="Input image.", optional=False):
    return desc.Image(name=name, label=label, description=description, optional=optional)


def imageOutput(description="Result."):
    return [desc.Image(name="output", label="Output", description=description)]


def colorOpInputs(defaultChannels="rgb", channelValues=("rgb", "rgba", "alpha")):
    """ Inputs shared by the per-pixel color nodes: input, mask, mask channel, channels. """
    return [
        imageInput(),
        desc.Image(name="mask", label="Mask", description="Optional mask: its 'Mask Channel' limits the effect.",
                   optional=True),
        *maskParams(),
        desc.ChoiceParam(name="channels", label="Channels", description="Channels of the 'rgba' layer to process.",
                         value=defaultChannels, values=list(channelValues)),
    ]


def mixParam():
    return desc.FloatParam(name="mix", label="Mix", description="Blend between the input (0) and the result (1).",
                           value=1.0, range=(0.0, 1.0, 0.01))


def colorParam(name, label, description, rgba):
    """
    An RGBA color, as a group of 4 floats (r, g, b, a), edited with a color field (swatch, values and
    an eyedropper picking the color in the viewer).
    """
    return desc.GroupAttribute(
        name=name, label=label, description=description, joinChar=",", semantic="rgbaColor",
        items=[desc.FloatParam(name=channel, label=channel.upper(), description=f"{channel.upper()} component.",
                               value=float(value), range=(0.0, 1.0, 0.01))
               for channel, value in zip("rgba", rgba)])


def linkedColor(name, label, description, value, range):
    """
    A value per channel (r, g, b, a) edited as a single number by default, which can be split into
    a color with the "split" checkbox of the attribute editor (like a Nuke knob with 4 channels).
    Its value is {"r": ..., "g": ..., "b": ..., "a": ...}.
    """
    return desc.GroupAttribute(
        name=name, label=label, description=description, joinChar=",", semantic="linkedColor",
        items=[desc.FloatParam(name=channel, label=channel.upper(), description=f"{label} of the {channel.upper()} channel.",
                               value=float(value), range=range)
               for channel in "rgba"])


def linkedColorValues(ctx, name):
    """ The 4 values (r, g, b, a) of a linkedColor as a float32 array (a single number is used for all). """
    import numpy as np
    value = ctx.value(name)
    if isinstance(value, (int, float)):
        return np.full(4, value, dtype=np.float32)
    return np.array([float(value[channel]) for channel in "rgba"], dtype=np.float32)


def upgradeLinkedColors(attrValues, names):
    """ Values of parameters that were floats before becoming linkedColors (node upgrade). """
    for name in names:
        value = attrValues.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            attrValues[name] = {channel: float(value) for channel in "rgba"}
    return attrValues


def colorValue(ctx, name):
    """ The value of a colorParam as a tuple (r, g, b, a). """
    color = ctx.value(name)
    return tuple(float(color[channel]) for channel in "rgba")


def processRGBA(ctx, inputs, func, selection=None, useMix=True, inputName="input"):
    """
    Apply `func(rgba) -> rgba` to bands of the main layer of the input (as RGBA), keep the result on
    the selected channels only ("channels" parameter by default), and blend it with the input by the
    mix and the mask. Runs in parallel bands of rows.
    """
    from .imageBuffer import RGBA_CHANNELS
    from .ops import channelMask, mapRows, mix
    image = inputs[inputName]
    rgba = image.rgba()
    selected = channelMask(RGBA_CHANNELS, selection or ctx.value("channels"))
    amount = maskAmount(ctx, inputs, image.bbox, ctx.value("mix")) if useMix else 1.0

    def band(pixels, amountBand):
        result = func(pixels)
        out = pixels.copy()
        out[:, :, selected] = result[:, :, selected]
        return mix(pixels, out, amountBand)

    return image.withRGBA(mapRows(band, [rgba, amount], 4, ctx.checkCancelled))


def _inputConnected(node, name="input"):
    """ Whether the image input `name` of a node is connected (used by "enabled" lambdas). """
    try:
        return node.attribute(name).isLink
    except Exception:
        return False


def formatParams(width=1920, height=1080):
    """ Width and height of a generated image, disabled when the "input" gives the format. """
    return [
        desc.IntParam(name="width", label="Width", description="Image width in pixels (when no input gives the format).",
                      value=width, range=(1, 8192, 1), enabled=lambda node: not _inputConnected(node)),
        desc.IntParam(name="height", label="Height", description="Image height in pixels (when no input gives the format).",
                      value=height, range=(1, 8192, 1), enabled=lambda node: not _inputConnected(node)),
    ]


def formatInput():
    """ Optional input of a generator: only its format (and frame range) is used, never its pixels. """
    return desc.Image(name="input", label="Format",
                      description="Optional: the image takes the format and frame range of this input "
                                  "(its pixels are not used).",
                      optional=True, infoOnly=True)


def inputUsageParam():
    """ How a drawing node uses its optional input: as a background, or only for its format. """
    return desc.ChoiceParam(
        name="inputUsage", label="Input Usage", value="background", values=["background", "format"],
        description="background: draw over the input. format: only use the format of the input (transparent background).",
        enabled=lambda node: _inputConnected(node))


def drawInfoOnlyInputs(ctx):
    """ infoOnlyInputs() of a drawing node with an inputUsageParam. """
    return ["input"] if ctx.value("inputUsage") == "format" else []


def outputInfo(ctx, inputsInfo=None):
    """
    ImageInfo of a generator: the format and frame range of its "input" when connected, its width
    and height parameters otherwise.
    """
    from .imageBuffer import ImageInfo
    info = inputsInfo.get("input") if inputsInfo is not None else ctx.inputInfo("input")
    if info is not None:
        return ImageInfo(info.width, info.height, info.firstFrame, info.lastFrame)
    return ImageInfo(ctx.value("width"), ctx.value("height"))


def colorspaceParam(name, label, description, value="auto"):
    """
    A colorspace of the color config (meshroom.live.color), edited with a menu of its colorspaces
    (a name can also be typed, e.g. for a colorspace of another config).
    """
    return desc.StringParam(name=name, label=label, description=description, value=value, semantic="colorspace")


def pointTrackerParam():
    """ How the tracks are tracked in the viewer (Tracker, pins of the Resection). """
    return desc.ChoiceParam(
        name="pointTracker", label="Point Tracker", value="patches", values=["patches", "tapir", "cotracker"],
        invalidate=False,
        description="patches: the pattern of each track searched from frame to frame. tapir: the BootsTAPIR network "
                    "(sees many frames at once: fast motions, blur; a hidden track stops). cotracker: CoTracker3 "
                    "(the same, non-commercial license only). The networks need PyTorch and their weights "
                    "(MESHROOM_MODELS_PATH); without them, patches.")


def trackRangeParams():
    """
    The frame range of the tracking actions of a node (Track buttons, autotrack, Track Face, Track Geometry...):
    "Limit Track Range" off: the whole input; on: from "Track First" to "Track Last" only (see trackRange).
    """
    return [
        desc.BoolParam(name="limitTrackRange", label="Limit Track Range", value=False, invalidate=False,
                       description="Track only between 'Track First' and 'Track Last' (not over the whole input)."),
        desc.IntParam(name="trackFirst", label="Track First", value=1001, range=(0, 10000, 1), invalidate=False,
                      enabled=lambda node: node.limitTrackRange.value,
                      description="First frame tracked when 'Limit Track Range' is on."),
        desc.IntParam(name="trackLast", label="Track Last", value=1100, range=(0, 10000, 1), invalidate=False,
                      enabled=lambda node: node.limitTrackRange.value,
                      description="Last frame tracked when 'Limit Track Range' is on."),
    ]


def trackRange(values, frames):
    """ `frames` (any order) limited to the track range of a node's values ({name: value}); unchanged without it. """
    if not values.get("limitTrackRange"):
        return list(frames)
    first, last = sorted((int(values.get("trackFirst", 0)), int(values.get("trackLast", 0))))
    return [frame for frame in frames if first <= frame <= last]
