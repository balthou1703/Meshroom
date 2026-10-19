__version__ = "0.1"

import numpy as np

from meshroom.core import desc
from meshroom.live.params import maskParams


class ZDefocus(desc.ImageOpNode):
    category = "Comp Filter"
    # Parameters in pixels, scaled in the proxy renders
    proxyScaled = ('size', 'maxSize')
    color = "#CC804E"
    documentation = """
Depth of field: a blur whose size depends on the depth of each pixel, with the shape of the lens
aperture (bokeh: disc, polygon with blades, or the image of the 'Filter' input).

The depth comes from the 'Depth' input when it is connected (a CG Z pass, or a gradient drawn with a
Roto, a Ramp...), else from a layer of the input (e.g. the "depth" layer of an EXR):
- Math "depth": the distance to the camera (Z pass); 0 is at infinity;
- Math "far = 0": 1 / distance (normalized depth);
- Math "linear": the blur grows linearly with the difference to the focal plane (gradients, 0 to 1);
- Math "direct": the value is the blur amount (0: sharp, 1: 'Size').

"Focal Plane Setup" shows the focus: red in front of the focal plane, green in focus, blue behind.
The image is cut into depth slices (Depth Slices on each side of the focal plane), blurred and
composited back to front, so that the blurred foreground spreads over the background.
"""

    inputs = [
        desc.Image(name="input", label="Input", description="Image to defocus."),
        desc.Image(name="depthImage", label="Depth", optional=True,
                   description="Optional depth image (CG Z pass, roto or ramp gradient). Not connected: a layer of the input."),
        desc.Image(name="filter", label="Filter", optional=True,
                   description="Optional image of the bokeh shape (its luminance), used with the 'filter image' shape."),
        desc.Image(name="mask", label="Mask", optional=True,
                   description="Optional mask: its 'Mask Channel' limits where the defocus applies."),
        *maskParams(),
        desc.StringParam(name="depthLayer", label="Depth Layer", value="depth", semantic="imageLayer:depthImage,input",
                         description="Layer holding the depth: of the 'Depth' input if it is connected, else of the input "
                                     "(e.g. depth, Z, rgba)."),
        desc.ChoiceParam(name="depthChannel", label="Depth Channel", value="auto",
                         values=["auto", "red", "green", "blue", "alpha", "luminance"],
                         description="Channel of the depth layer. auto: Z if the layer has it, else its first channel "
                                     "(the luminance for an 'rgba' layer)."),
        desc.ChoiceParam(name="math", label="Math", value="depth", values=["depth", "far = 0", "linear", "direct"],
                         description="How the depth values give the blur size (see the documentation)."),
        desc.ChoiceParam(name="outputMode", label="Output", value="result", values=["result", "focal plane setup"],
                         description="The defocused image, or the focus shown in color (red: near, green: in focus, blue: far)."),
        desc.FloatParam(name="focalPlane", label="Focal Plane", value=1.0, range=(0.0, 100.0, 0.01),
                        description="Depth value in focus."),
        desc.FloatParam(name="depthOfField", label="Depth Of Field", value=0.0, range=(0.0, 10.0, 0.01),
                        description="Depth range around the focal plane that stays sharp."),
        desc.FloatParam(name="size", label="Size", value=10.0, range=(0.0, 100.0, 0.1),
                        description="Blur radius in pixels (at infinity for 'depth', at a difference of 1 for 'linear')."),
        desc.FloatParam(name="maxSize", label="Maximum Size", value=50.0, range=(0.0, 200.0, 0.5),
                        description="Largest blur radius in pixels."),
        desc.ChoiceParam(name="shape", label="Bokeh Shape", value="disc", values=["disc", "blades", "filter image"],
                         description="Shape of the aperture: a disc, a polygon (Blades), or the 'Filter' input."),
        desc.IntParam(name="blades", label="Blades", value=6, range=(3, 16, 1),
                      description="Number of blades of the aperture (sides of the polygon).",
                      enabled=lambda node: node.shape.value == "blades"),
        desc.FloatParam(name="rotation", label="Rotation", value=0.0, range=(-180.0, 180.0, 1.0),
                        description="Rotation of the polygon in degrees.",
                        enabled=lambda node: node.shape.value == "blades"),
        desc.FloatParam(name="aspectRatio", label="Aspect Ratio", value=1.0, range=(0.25, 4.0, 0.01),
                        description="Width / height of the bokeh (> 1: anamorphic, wider)."),
        desc.IntParam(name="slices", label="Depth Slices", value=8, range=(1, 32, 1),
                      description="Depth slices on each side of the focal plane: more is smoother and slower."),
        desc.FloatParam(name="mix", label="Mix", value=1.0, range=(0.0, 1.0, 0.01),
                        description="Blend between the input (0) and the defocused image (1)."),
    ]

    outputs = [desc.Image(name="output", label="Output", description="Defocused image.")]

    def isIdentity(self, ctx):
        if ctx.value("outputMode") == "result" and (ctx.value("mix") <= 0.0 or ctx.value("size") <= 0.0 or
                                                ctx.value("maxSize") < 0.5):
            return "input"
        return None

    def _depth(self, ctx, inputs, image):
        """ The depth values over the bbox of `image`, (h, w). """
        from meshroom.live import RGBA, ops
        source = inputs.get("depthImage") if inputs.get("depthImage") is not None else image
        layerName = ctx.value("depthLayer") or "depth"
        if not source.hasLayer(layerName) and source is not image and source.hasLayer(RGBA):
            # A depth image without that layer (e.g. a roto gradient): its "rgba" layer
            layerName = RGBA
        if not source.hasLayer(layerName):
            which = "the 'Depth' input" if inputs.get("depthImage") is not None else "the input"
            raise RuntimeError(f"No layer '{layerName}' in {which} (layers: {', '.join(source.layerNames)}).")
        channels = source.channels(layerName)
        pixels = ops.alignToBBox(source.layer(layerName), source.bbox, image.bbox)
        choice = ctx.value("depthChannel")
        names = {"red": ("R", "X"), "green": ("G", "Y"), "blue": ("B", "Z"), "alpha": ("A",)}
        if choice == "luminance" or (choice == "auto" and layerName == RGBA and "Z" not in channels):
            if len(channels) >= 3:
                return pixels[:, :, :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            return pixels[:, :, 0]
        if choice == "auto":
            return pixels[:, :, channels.index("Z") if "Z" in channels else 0]
        for candidate in names[choice]:
            if candidate in channels:
                return pixels[:, :, channels.index(candidate)]
        index = ("red", "green", "blue", "alpha").index(choice)
        return pixels[:, :, index] if index < len(channels) else pixels[:, :, 0]

    def _kernelFor(self, ctx, inputs):
        from meshroom.live import defocus
        shape = ctx.value("shape")
        blades = ctx.value("blades") if shape == "blades" else 0
        rotation = ctx.value("rotation")
        aspect = ctx.value("aspectRatio")
        filterImage = None
        if shape == "filter image" and inputs.get("filter") is not None:
            rgba = inputs["filter"].rgba()
            filterImage = rgba[:, :, :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        cache = {}

        def kernelFor(radius):
            key = round(radius, 3)
            if key not in cache:
                cache[key] = defocus.bokehKernel(radius, blades, rotation, aspect, filterImage)
            return cache[key]
        return kernelFor

    def render(self, ctx, inputs):
        from meshroom.live import defocus, ops
        from meshroom.live.params import maskAmount
        image = inputs["input"]
        rgba = image.rgba()
        radius = defocus.blurRadius(self._depth(ctx, inputs, image), ctx.value("math"), ctx.value("focalPlane"),
                                    ctx.value("depthOfField"), ctx.value("size"), ctx.value("maxSize"))
        if ctx.value("outputMode") == "focal plane setup":
            # Grey image tinted by the focus: red in front, green in focus, blue behind
            luma = (rgba[:, :, :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))[:, :, np.newaxis]
            near = (radius < -0.5)[:, :, np.newaxis]
            far = (radius > 0.5)[:, :, np.newaxis]
            tint = np.where(near, [1.0, 0.25, 0.25], np.where(far, [0.25, 0.4, 1.0], [0.3, 1.0, 0.3])).astype(np.float32)
            out = rgba.copy()
            out[:, :, :3] = 0.25 + 0.75 * luma * tint
            out[:, :, 3] = 1.0
            return image.withRGBA(out)
        blurred = defocus.defocus(rgba, radius, ctx.value("maxSize"), ctx.value("slices"),
                                  self._kernelFor(ctx, inputs), ctx.checkCancelled)
        amount = maskAmount(ctx, inputs, image.bbox, ctx.value("mix"))
        if not (np.isscalar(amount) and amount >= 1.0):
            blurred = ops.mix(rgba, blurred, amount)
        return image.withRGBA(blurred)
