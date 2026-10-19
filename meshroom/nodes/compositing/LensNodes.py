"""
Lens distortion (3DEqualizer models, compatible with the 3DE4 "LD" nodes of Nuke).
"""
__version__ = "0.2"

from collections import OrderedDict

import numpy as np

from meshroom.core import desc
from meshroom.live import lens
from meshroom.live.params import imageInput, imageOutput

_DIRECTIONS = ["undistort", "distort", "STMap undistort", "STMap distort"]


def _modelParams():
    """ The parameters of the models: a parameter shared by several models (same 3DE name) is created once. """
    models = OrderedDict()
    for model in lens.MODELS:
        for name, label, value in lens.MODEL_PARAMETERS[model]:
            models.setdefault((name, label, value), []).append(model)
    params = []
    for (name, label, value), usedBy in models.items():
        rangeFor = (-180.0, 180.0, 0.1) if name in ("phiCylindricDirection", "lensRotation") else \
            (0.1, 4.0, 0.001) if name in ("anamorphicSqueeze", "squeezeX", "squeezeY", "rescale") else (-0.5, 0.5, 0.0001)
        params.append(desc.FloatParam(
            name=name, label=label, value=value, range=rangeFor,
            description=f"{label} ({', '.join(usedBy)}).",
            enabled=lambda node, usedBy=tuple(usedBy): node.model.value in usedBy))
    return params


class LensDistortion(desc.ImageOpNode):
    category = "Comp Transform"
    color = "#A57AAA"
    documentation = """
Lens distortion with the models of 3DEqualizer, as the 3DE4 "LD" nodes of Nuke (same parameters, in
the units of 3DE): "3DE4 Classic LD Model", "3DE4 Radial - Standard, Degree 4" (spherical lenses),
"3DE4 Anamorphic - Standard, Degree 4" and "Degree 6" (anamorphic lenses, with the pixel aspect of the
plate), "3DE4 Anamorphic - Rescaled, Degree 4" (with a Rescale factor of the radius, to check against
3DE) and "3DE4 Radial - Fisheye, Degree 8" (equisolid-angle fisheye: it needs the focal length and
the filmback).

- Direction: "undistort" removes the distortion of the plate, "distort" applies it to an undistorted
  image (e.g. CG); the "STMap" directions output the map instead (red / green: position of the
  source pixel divided by the width / height), for an STMap node.
- Filmback and lens center offset in cm, as in 3DE (filmback 0: from the proportions of the image and
  its pixel aspect, without lens center offset).
- "Import 3DE Node" reads a 3DE4 LD node from a Nuke script (.nk, e.g. exported by 3DE) in "3DE File":
  model, filmback, lens center, pixel aspect, distortion parameters and direction.
The output has the format of the input.
"""
    inputs = [
        imageInput(),
        desc.ChoiceParam(name="model", label="Model", value=lens.RADIAL4, values=list(lens.MODELS),
                         description="3DE lens distortion model."),
        desc.ChoiceParam(name="direction", label="Direction", value="undistort", values=_DIRECTIONS,
                         description="undistort the plate, distort an undistorted image, or output the STMap of either."),
        desc.FloatParam(name="filmbackWidth", label="Filmback Width (cm)", value=0.0, range=(0.0, 10.0, 0.001),
                        description="tde4_filmback_width_cm (0: from the image)."),
        desc.FloatParam(name="filmbackHeight", label="Filmback Height (cm)", value=0.0, range=(0.0, 10.0, 0.001),
                        description="tde4_filmback_height_cm (0: from the image)."),
        desc.FloatParam(name="lensCenterOffsetX", label="Lens Center Offset X (cm)", value=0.0, range=(-1.0, 1.0, 0.0001),
                        description="tde4_lens_center_offset_x_cm."),
        desc.FloatParam(name="lensCenterOffsetY", label="Lens Center Offset Y (cm)", value=0.0, range=(-1.0, 1.0, 0.0001),
                        description="tde4_lens_center_offset_y_cm."),
        desc.FloatParam(name="pixelAspect", label="Pixel Aspect", value=1.0, range=(0.25, 4.0, 0.01),
                        description="tde4_pixel_aspect (2 for a 2x anamorphic plate that is not desqueezed)."),
        desc.FloatParam(name="focalLength", label="Focal Length (cm)", value=0.0, range=(0.0, 30.0, 0.01),
                        description="tde4_focal_length_cm: used by the fisheye model (with the filmback).",
                        enabled=lambda node: node.model.value == lens.FISHEYE8),
        *_modelParams(),
        desc.File(name="lensFile", label="3DE File", value="", exposed=False,
                  description="Nuke script (.nk) with a 3DE4 LD node, to import with 'Import 3DE Node'."),
        desc.PushButtonParam(name="import3de", label="Import 3DE Node", semantic="liveAction",
                             description="Read the model and the parameters of the 3DE4 LD node of '3DE File'."),
    ]
    outputs = imageOutput("Undistorted (or distorted) image, or its STMap.")

    def _model(self, ctx):
        model = ctx.value("model")
        return model, [ctx.value(name) for name, _, _ in lens.MODEL_PARAMETERS[model]]

    def _camera(self, ctx):
        return {name: ctx.value(name) for name, _, _ in lens.CAMERA_PARAMETERS}

    def isIdentity(self, ctx):
        if ctx.value("direction").startswith("STMap"):
            return None
        model, params = self._model(ctx)
        neutral = [value for _, _, value in lens.MODEL_PARAMETERS[model]]
        return "input" if params == neutral else None

    def render(self, ctx, inputs):
        from meshroom.live import BBox, flow, ops
        image = inputs["input"]
        width, height = image.format
        model, params = self._model(ctx)
        direction = ctx.value("direction")
        mode = "distort" if direction in ("distort", "STMap distort") else "undistort"
        positions = lens.cachedSourcePositions(model, params, self._camera(ctx), width, height, mode)
        out = BBox(0, 0, width, height)
        if direction.startswith("STMap"):
            rgba = np.zeros((height, width, 4), np.float32)
            rgba[:, :, 0] = positions[:, :, 0] / float(width)
            rgba[:, :, 1] = positions[:, :, 1] / float(height)
            rgba[:, :, 3] = 1.0
            return image.withPixels(OrderedDict([("rgba", (("R", "G", "B", "A"), rgba))]), out)
        # Sample every layer of the input (black outside its bbox: a black border is added around it)
        padded = image.bbox.expanded(1, 1)
        rows = padded.y1 - positions[:, :, 1] - 0.5
        columns = positions[:, :, 0] - padded.x0 - 0.5
        layers = OrderedDict()
        for name, channels, pixels in image.layers():
            ctx.checkCancelled()
            source = ops.alignToBBox(pixels, image.bbox, padded)
            layers[name] = (channels, flow.sampleRows(source, rows, columns))
        return image.withPixels(layers, out)

    def runAction(self, name, context):
        """ Import a 3DE4 LD node of a Nuke script (see the documentation): the new values. """
        if name != "import3de":
            raise RuntimeError(f"Unknown action: {name}")
        path = context.value("lensFile")
        if not path:
            raise RuntimeError("Set '3DE File' (a Nuke script with a 3DE4 LD node).")
        from meshroom.live import animation
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                values = lens.parseNukeLensNode(f.read())
        except (OSError, ValueError) as e:
            raise RuntimeError(str(e)) from e
        # Animated knobs (e.g. a zoom): keys of the parameters; set last, replacing the previous animation
        curves = values.pop("curves", {})
        for name in curves:
            values.pop(name, None)
        values["animation"] = animation.dumpAnimation(curves)
        return values
