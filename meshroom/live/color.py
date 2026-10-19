"""
Color management: the colorspaces of the files (Read / Write, OCIOColorSpace node) and the display
transforms of the viewer.

The images are processed in a linear working space. Two sources of colorspaces:
- an OpenColorIO config, when the PyOpenColorIO module is installed and the "OCIO" environment variable
  points to a config file (the working space is then its "scene_linear" role);
- otherwise a builtin set, close to the default colorspaces of Nuke: the working space is "linear"
  (Rec.709 / sRGB primaries), the others are transfer curves and primaries computed with numpy.
"""

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# Primaries and white points (CIE xy chromaticities)

_D65 = (0.3127, 0.3290)
_ACES_WHITE = (0.32168, 0.33767)

PRIMARIES = {
    "rec709": ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06), _D65),
    "rec2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), _D65),
    "p3d65": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), _D65),
    "ap0": ((0.7347, 0.2653), (0.0, 1.0), (0.0001, -0.0770), _ACES_WHITE),
    "ap1": ((0.713, 0.293), (0.165, 0.830), (0.128, 0.044), _ACES_WHITE),
    "awg3": ((0.6840, 0.3130), (0.2210, 0.8480), (0.0861, -0.1020), _D65),
}

_BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                      [-0.7502, 1.7135, 0.0367],
                      [0.0389, -0.0685, 1.0296]])


def _xyz(xy) -> np.ndarray:
    x, y = xy
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def rgbToXyzMatrix(primaries: str) -> np.ndarray:
    """ Matrix converting linear RGB of the given primaries to CIE XYZ (of its own white point). """
    red, green, blue, white = PRIMARIES[primaries]
    columns = np.stack([_xyz(red), _xyz(green), _xyz(blue)], axis=1)
    scale = np.linalg.solve(columns, _xyz(white))
    return columns * scale


def _adaptation(sourceWhite, targetWhite) -> np.ndarray:
    """ Bradford chromatic adaptation matrix between two white points (XYZ to XYZ). """
    if sourceWhite == targetWhite:
        return np.eye(3)
    source = _BRADFORD @ _xyz(sourceWhite)
    target = _BRADFORD @ _xyz(targetWhite)
    return np.linalg.inv(_BRADFORD) @ np.diag(target / source) @ _BRADFORD


def primariesMatrix(source: str, target: str) -> np.ndarray:
    """ Matrix converting linear RGB from the `source` primaries to the `target` primaries (Bradford). """
    if source == target:
        return np.eye(3)
    adaptation = _adaptation(PRIMARIES[source][3], PRIMARIES[target][3])
    return np.linalg.inv(rgbToXyzMatrix(target)) @ adaptation @ rgbToXyzMatrix(source)


# Transfer curves: (encoded -> linear, linear -> encoded)

def srgbToLinear(values: np.ndarray) -> np.ndarray:
    power = ((np.maximum(values, 0.04045) + 0.055) / 1.055) ** 2.4
    return np.where(values <= 0.04045, values / 12.92, power).astype(np.float32)


def linearToSrgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, None)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * np.power(values, 1.0 / 2.4) - 0.055).astype(np.float32)


def _rec709ToLinear(values):
    return np.where(values < 0.081, values / 4.5, ((np.maximum(values, 0.081) + 0.099) / 1.099) ** (1.0 / 0.45))


def _linearToRec709(values):
    values = np.clip(values, 0.0, None)
    return np.where(values < 0.018, values * 4.5, 1.099 * np.power(values, 0.45) - 0.099)


def _gamma(exponent):
    """ Pure power curve, mirrored for the negative values. """
    def toLinear(values):
        return np.sign(values) * np.abs(values) ** exponent

    def fromLinear(values):
        return np.sign(values) * np.abs(values) ** (1.0 / exponent)
    return toLinear, fromLinear


# Cineon (Kodak printing density, as in Nuke: black 95, white 685)
_CINEON_OFFSET = 10.0 ** ((95.0 - 685.0) / 300.0)


def _cineonToLinear(values):
    return (10.0 ** ((1023.0 * values - 685.0) / 300.0) - _CINEON_OFFSET) / (1.0 - _CINEON_OFFSET)


def _linearToCineon(values):
    values = np.maximum(values * (1.0 - _CINEON_OFFSET) + _CINEON_OFFSET, 1e-10)
    return (685.0 + 300.0 * np.log10(values)) / 1023.0


# ARRI LogC v3 (EI 800)
_LOGC = dict(cut=0.010591, a=5.555556, b=0.052272, c=0.247190, d=0.385537, e=5.367655, f=0.092809)


def _logCToLinear(values):
    p = _LOGC
    logPart = (10.0 ** ((values - p["d"]) / p["c"]) - p["b"]) / p["a"]
    return np.where(values > p["e"] * p["cut"] + p["f"], logPart, (values - p["f"]) / p["e"])


def _linearToLogC(values):
    p = _LOGC
    logPart = p["c"] * np.log10(np.maximum(p["a"] * values + p["b"], 1e-10)) + p["d"]
    return np.where(values > p["cut"], logPart, p["e"] * values + p["f"])


# ACEScct and ACEScc (on AP1 primaries)
_ACES_MAX = (np.log2(65504.0) + 9.72) / 17.52


def _acescctToLinear(values):
    log = 2.0 ** (np.minimum(values, _ACES_MAX) * 17.52 - 9.72)
    return np.where(values <= 0.155251141552511, (values - 0.0729055341958355) / 10.5402377416545, log)


def _linearToAcescct(values):
    log = (np.log2(np.maximum(values, 1e-10)) + 9.72) / 17.52
    return np.where(values <= 0.0078125, 10.5402377416545 * values + 0.0729055341958355, log)


def _acesccToLinear(values):
    values = np.minimum(values, _ACES_MAX)
    power = 2.0 ** (values * 17.52 - 9.72)
    return np.where(values < (9.72 - 15.0) / 17.52, (power - 2.0 ** -16) * 2.0, power)


def _linearToAcescc(values):
    small = (np.log2(2.0 ** -16 + np.maximum(values, 0.0) * 0.5) + 9.72) / 17.52
    large = (np.log2(np.maximum(values, 2.0 ** -15)) + 9.72) / 17.52
    return np.where(values < 2.0 ** -15, small, large)


_IDENTITY_CURVE = (None, None)

# name: (transfer curve (to linear, from linear), primaries, family, description)
_BUILTIN = {
    "linear": (_IDENTITY_CURVE, "rec709", "Linear", "Scene linear, Rec.709 / sRGB primaries (working space)."),
    "sRGB": ((srgbToLinear, linearToSrgb), "rec709", "Display", "sRGB curve, Rec.709 primaries."),
    "rec709": ((_rec709ToLinear, _linearToRec709), "rec709", "Display", "Rec.709 (BT.709) camera curve."),
    "Gamma1.8": (_gamma(1.8), "rec709", "Display", "Pure power 1.8 curve."),
    "Gamma2.2": (_gamma(2.2), "rec709", "Display", "Pure power 2.2 curve."),
    "Gamma2.4": (_gamma(2.4), "rec709", "Display", "Pure power 2.4 curve (Rec.1886 display)."),
    "Cineon": ((_cineonToLinear, _linearToCineon), "rec709", "Log", "Kodak Cineon printing density (black 95, white 685)."),
    "AlexaV3LogC": ((_logCToLinear, _linearToLogC), "awg3", "Log", "ARRI LogC v3 (EI 800), ARRI Wide Gamut 3."),
    "ACES2065-1": (_IDENTITY_CURVE, "ap0", "ACES", "ACES linear, AP0 primaries (archive / interchange)."),
    "ACEScg": (_IDENTITY_CURVE, "ap1", "ACES", "ACES linear, AP1 primaries (CG rendering)."),
    "ACEScct": ((_acescctToLinear, _linearToAcescct), "ap1", "ACES", "ACES log with a toe, AP1 primaries (grading)."),
    "ACEScc": ((_acesccToLinear, _linearToAcescc), "ap1", "ACES", "ACES pure log, AP1 primaries (grading)."),
    "rec2020": (_IDENTITY_CURVE, "rec2020", "Linear", "Linear, Rec.2020 primaries."),
    "Display P3": ((srgbToLinear, linearToSrgb), "p3d65", "Display", "sRGB curve, P3 D65 primaries (Apple displays)."),
    "raw": (_IDENTITY_CURVE, "rec709", "Utility", "No conversion (data: normals, depth, motion vectors...)."),
}

# Names used by other configs and by Nuke scripts
_ALIASES = {
    "scene_linear": "linear", "lin": "linear", "linear_rec709": "linear", "lin_rec709": "linear",
    "srgb": "sRGB", "srgb_texture": "sRGB", "srgb - texture": "sRGB", "srgb_rec709": "sRGB",
    "rec.709": "rec709", "bt.709": "rec709", "gamma 1.8": "Gamma1.8", "gamma 2.2": "Gamma2.2", "gamma 2.4": "Gamma2.4",
    "cineon": "Cineon", "logc": "AlexaV3LogC", "alexav3logc": "AlexaV3LogC", "arri logc": "AlexaV3LogC",
    "aces2065-1": "ACES2065-1", "aces - aces2065-1": "ACES2065-1", "aces": "ACES2065-1",
    "acescg": "ACEScg", "aces - acescg": "ACEScg", "acescct": "ACEScct", "aces - acescct": "ACEScct",
    "acescc": "ACEScc", "aces - acescc": "ACEScc", "rec2020": "rec2020", "linear_rec2020": "rec2020",
    "display p3": "Display P3", "p3-d65": "Display P3", "raw": "raw", "data": "raw", "utility - raw": "raw",
}

# Display transforms of the viewer: display -> {view: (colorspace of the encoding, or None for no transform)}
_BUILTIN_DISPLAYS = {
    "sRGB": {"Standard": "sRGB", "Raw": None},
    "Rec.1886": {"Standard": "Gamma2.4", "Raw": None},
    "Rec.709": {"Standard": "rec709", "Raw": None},
    "Display P3": {"Standard": "Display P3", "Raw": None},
}


def _ocioModule():
    try:
        import PyOpenColorIO
        return PyOpenColorIO
    except ImportError:
        return None


class ColorConfig:
    """
    The colorspaces available, and the conversions between them.

    Conversions take and return float32 RGB arrays (..., 3). `processor(source, target)` returns a
    function, cached, to convert many images.
    """

    def __init__(self, ocioConfig=None, path: str = ""):
        self._ocio = ocioConfig
        self.path = path
        self._processors = {}

    @property
    def isOcio(self) -> bool:
        return self._ocio is not None

    @property
    def description(self) -> str:
        if self._ocio is not None:
            return f"OCIO config: {self.path}"
        return "Builtin colorspaces (no OCIO config)"

    # Colorspaces

    def colorspaceNames(self) -> List[str]:
        if self._ocio is not None:
            return list(self._ocio.getColorSpaceNames())
        return list(_BUILTIN.keys())

    def workingSpace(self) -> str:
        """ The linear space the images are processed in. """
        if self._ocio is not None:
            return self.resolve("scene_linear") or self.colorspaceNames()[0]
        return "linear"

    def resolve(self, name: str) -> Optional[str]:
        """ The colorspace of a name, role or alias, None if unknown. """
        if not name:
            return None
        if self._ocio is not None:
            colorspace = self._ocio.getColorSpace(name)
            return colorspace.getName() if colorspace is not None else None
        if name in _BUILTIN:
            return name
        return _ALIASES.get(name.strip().lower())

    def defaultSpace(self, isIntegerFile: bool) -> str:
        """ The colorspace of "auto": sRGB-like for 8/16-bit integer files, the working space for float files. """
        if not isIntegerFile:
            return self.workingSpace()
        if self._ocio is not None:
            for name in ("matte_paint", "texture_paint", "color_picking", "sRGB", "srgb_texture",
                         "Utility - sRGB - Texture", "sRGB - Texture"):
                found = self.resolve(name)
                if found:
                    return found
            return self.workingSpace()
        return "sRGB"

    def resolveForFile(self, name: str, isIntegerFile: bool) -> str:
        """ The colorspace of a Read / Write: "auto" or a name; raises for an unknown name. """
        if not name or name == "auto":
            return self.defaultSpace(isIntegerFile)
        found = self.resolve(name)
        if found is None:
            raise ValueError(f"Unknown colorspace '{name}' ({self.description}).")
        return found

    # Conversions

    def isIdentity(self, source: str, target: str) -> bool:
        source, target = self.resolve(source), self.resolve(target)
        if source == target:
            return True
        if self._ocio is None:
            return {source, target} <= {"linear", "raw"}
        return False

    def curve(self, source: str, target: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        """
        The conversion as a function of one channel when it is the same curve on each channel (no change
        of primaries): it can then be baked in a lookup table. None otherwise.
        """
        if self._ocio is not None:
            return None
        source, target = self.resolve(source), self.resolve(target)
        (toLinear, _), sourcePrimaries = _BUILTIN[source][0], _BUILTIN[source][1]
        (_, fromLinear), targetPrimaries = _BUILTIN[target][0], _BUILTIN[target][1]
        if sourcePrimaries != targetPrimaries and "raw" not in (source, target):
            return None

        def convert(values):
            values = np.asarray(values, dtype=np.float64)
            if toLinear is not None:
                values = toLinear(values)
            if fromLinear is not None:
                values = fromLinear(values)
            return values.astype(np.float32)
        return convert

    def processor(self, source: str, target: str) -> Callable[[np.ndarray], np.ndarray]:
        """ A function converting RGB pixels (..., 3) from `source` to `target`. """
        resolvedSource, resolvedTarget = self.resolve(source), self.resolve(target)
        for name, resolved in ((source, resolvedSource), (target, resolvedTarget)):
            if resolved is None:
                raise ValueError(f"Unknown colorspace '{name}' ({self.description}).")
        key = (resolvedSource, resolvedTarget)
        if key not in self._processors:
            if self.isIdentity(*key):
                self._processors[key] = lambda rgb: np.asarray(rgb, dtype=np.float32)
            elif self._ocio is not None:
                self._processors[key] = _ocioApply(self._ocio.getProcessor(*key))
            else:
                self._processors[key] = self._builtinProcessor(*key)
        return self._processors[key]

    def convert(self, rgb: np.ndarray, source: str, target: str) -> np.ndarray:
        return self.processor(source, target)(rgb)

    @staticmethod
    def _builtinProcessor(source: str, target: str):
        if "raw" in (source, target):
            return lambda rgb: np.asarray(rgb, dtype=np.float32)
        (toLinear, _), sourcePrimaries = _BUILTIN[source][0], _BUILTIN[source][1]
        (_, fromLinear), targetPrimaries = _BUILTIN[target][0], _BUILTIN[target][1]
        matrix = primariesMatrix(sourcePrimaries, targetPrimaries).astype(np.float32).T

        def convert(rgb):
            values = np.asarray(rgb, dtype=np.float32)
            if toLinear is not None:
                values = toLinear(values.astype(np.float64)).astype(np.float32)
            if sourcePrimaries != targetPrimaries:
                values = values @ matrix
            if fromLinear is not None:
                values = fromLinear(values.astype(np.float64)).astype(np.float32)
            return values
        return convert

    # Displays of the viewer

    def displays(self) -> List[str]:
        if self._ocio is not None:
            return list(self._ocio.getDisplays())
        return list(_BUILTIN_DISPLAYS.keys())

    def views(self, display: str) -> List[str]:
        if self._ocio is not None:
            return list(self._ocio.getViews(display))
        return list(_BUILTIN_DISPLAYS.get(display, {}).keys())

    def defaultDisplayView(self) -> Tuple[str, str]:
        if self._ocio is not None:
            display = self._ocio.getDefaultDisplay()
            return display, self._ocio.getDefaultView(display)
        return "sRGB", "Standard"

    def displayTransforms(self) -> List[str]:
        """ The display transforms of the viewer, as "display / view" names. """
        return [f"{display} / {view}" for display in self.displays() for view in self.views(display)]

    def defaultDisplayTransform(self) -> str:
        return "{} / {}".format(*self.defaultDisplayView())

    def displayTransform(self, name: str) -> "DisplayTransform":
        """ The viewer transform of a "display / view" name (the default one when unknown). """
        display, _, view = (name or "").partition(" / ")
        if display not in self.displays() or view not in self.views(display):
            display, view = self.defaultDisplayView()
        key = ("display", display, view)
        if key not in self._processors:
            if self._ocio is not None:
                ocio = _ocioModule()
                transform = ocio.DisplayViewTransform(src=self.workingSpace(), display=display, view=view)
                self._processors[key] = DisplayTransform(processor=_ocioApply(self._ocio.getProcessor(transform)))
            else:
                target = _BUILTIN_DISPLAYS[display][view]
                if target is None:
                    self._processors[key] = DisplayTransform()
                else:
                    curve = self.curve("linear", target)
                    if curve is not None:
                        self._processors[key] = DisplayTransform(curve=curve)
                    else:
                        matrix = primariesMatrix("rec709", _BUILTIN[target][1]).astype(np.float32)
                        self._processors[key] = DisplayTransform(curve=_BUILTIN[target][0][1], matrix=matrix)
        return self._processors[key]


class DisplayTransform:
    """
    Transform of the viewer from the working space to the display, applied after the exposure and the
    viewer gamma:
    - `curve`: a function applied on each channel (baked in a lookup table), with an optional 3x3
      `matrix` applied before it on RGB;
    - or `processor`: a function of RGB pixels (OCIO), for transforms that are not a curve per channel.
    No curve and no processor: the values are shown as they are (raw).
    """

    def __init__(self, curve=None, matrix: Optional[np.ndarray] = None, processor=None):
        self.curve = curve
        self.matrix = matrix
        self.processor = processor


def _ocioApply(processor):
    """ A function converting RGB pixels with an OCIO processor (on a contiguous float32 copy). """
    cpu = processor.getDefaultCPUProcessor()

    def convert(rgb):
        rgb = np.asarray(rgb, dtype=np.float32)
        shape = rgb.shape
        buffer = np.ascontiguousarray(rgb.reshape(-1, 3), dtype=np.float32).copy()
        cpu.applyRGB(buffer)
        return buffer.reshape(shape)
    return convert


_config: Optional[ColorConfig] = None


def loadConfig(path: Optional[str] = None) -> ColorConfig:
    """
    Load the color config: the OCIO config file `path` (by default the "OCIO" environment variable) when
    PyOpenColorIO is installed, else the builtin colorspaces.
    """
    global _config
    path = path if path is not None else os.environ.get("OCIO", "")
    ocio = _ocioModule() if path else None
    if path and ocio is None:
        logging.warning(f"OCIO config '{path}' ignored: the PyOpenColorIO module is not installed. Builtin colorspaces used.")
    config = None
    if ocio is not None:
        try:
            config = ColorConfig(ocio.Config.CreateFromFile(path), path)
        except Exception as error:
            logging.warning(f"Cannot load the OCIO config '{path}': {error}. Builtin colorspaces used.")
    _config = config or ColorConfig()
    return _config


def config() -> ColorConfig:
    """ The current color config (loaded on first use). """
    return _config if _config is not None else loadConfig()


def setConfig(newConfig: Optional[ColorConfig]):
    """ Replace the current color config (None: load it again on next use). """
    global _config
    _config = newConfig


def colorspaceNames() -> List[str]:
    return config().colorspaceNames()


def builtinColorspaces() -> Dict[str, str]:
    """ The builtin colorspaces and their descriptions. """
    return {name: values[3] for name, values in _BUILTIN.items()}
