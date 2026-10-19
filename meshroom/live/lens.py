"""
Lens distortion models of 3DEqualizer (LensDistortion node), compatible with the 3DE4 "LD" nodes
of Nuke: same models, same parameters (in cm and in the units of the 3DE user interface).

Coordinates follow the 3DE lens distortion plugin kit (LDPK):
- unit coordinates: (0, 0) at the lower left corner of the image, (1, 1) at the upper right one;
- "diagonally normalized" coordinates (dn): centered on the lens center (the filmback center moved
  by the lens center offset), in units of half the filmback diagonal.

Each model is a function from distorted to undistorted dn coordinates (as in 3DE: the closed form is
the undistortion); distorting inverts it numerically (Newton).

Models (parameters in the order of 3DE):
- "3DE4 Classic LD Model": Distortion, Anamorphic Squeeze, Curvature X, Curvature Y, Quartic Distortion;
- "3DE4 Radial - Standard, Degree 4": Distortion - Degree 2, U - Degree 2, V - Degree 2, Quartic
  Distortion - Degree 4, U - Degree 4, V - Degree 4, Phi - Cylindric Direction (degrees),
  B - Cylindric Bending;
- "3DE4 Anamorphic - Standard, Degree 4": Cx02, Cy02, Cx22, Cy22 (degree 2), Cx04, Cy04, Cx24, Cy24,
  Cx44, Cy44 (degree 4), Lens Rotation (degrees), Squeeze-X, Squeeze-Y; it also uses the pixel aspect;
- "3DE4 Anamorphic - Standard, Degree 6": the same with the degree 6 coefficients (Cx06 ... Cy66);
- "3DE4 Anamorphic - Rescaled, Degree 4": the degree 4 model with a Rescale factor of the radius
  (the polynomial is evaluated on the radius times Rescale; formula to check against 3DE);
- "3DE4 Radial - Fisheye, Degree 8": an equisolid-angle fisheye projection (it needs the focal length
  and the filmback), then a radial polynomial of degree 8: Distortion - Degree 2, Quartic Distortion
  - Degree 4, Degree 6, Degree 8.
(Formulas as in AliceVision's Undistortion3DEClassicLD / Undistortion3DERadial4 / Undistortion3DEAnamorphic4,
and in the Lens Distortion Plugin Kit of 3DE for the fisheye and the anamorphic polynomial of degree 6.)
"""
from __future__ import annotations

import math
import re
import threading
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np

CLASSIC = "3DE4 Classic LD Model"
RADIAL4 = "3DE4 Radial - Standard, Degree 4"
ANAMORPHIC4 = "3DE4 Anamorphic - Standard, Degree 4"
ANAMORPHIC6 = "3DE4 Anamorphic - Standard, Degree 6"
RESCALED4 = "3DE4 Anamorphic - Rescaled, Degree 4"
FISHEYE8 = "3DE4 Radial - Fisheye, Degree 8"
MODELS = [CLASSIC, RADIAL4, ANAMORPHIC4, ANAMORPHIC6, RESCALED4, FISHEYE8]
# Classes of the 3DE LD nodes of Nuke
NUKE_CLASSES = {
    "LD_3DE_Classic_LD_Model": CLASSIC,
    "LD_3DE4_Classic_LD_Model": CLASSIC,
    "LD_3DE4_Radial_Standard_Degree_4": RADIAL4,
    "LD_3DE4_Anamorphic_Standard_Degree_4": ANAMORPHIC4,
    "LD_3DE4_Anamorphic_Standard_Degree_6": ANAMORPHIC6,
    "LD_3DE4_Anamorphic_Degree_6": ANAMORPHIC6,
    "LD_3DE4_Anamorphic_Rescaled_Degree_4": RESCALED4,
    "LD_3DE4_Radial_Fisheye_Degree_8": FISHEYE8,
}

# Coefficients of the anamorphic polynomials: (cos power, r power) in the order of 3DE
_ANAMORPHIC_TERMS = {4: [(0, 2), (2, 2), (0, 4), (2, 4), (4, 4)],
                     6: [(0, 2), (2, 2), (0, 4), (2, 4), (4, 4), (0, 6), (2, 6), (4, 6), (6, 6)]}


def _anamorphicParameters(degree):
    params = []
    for phi, r in _ANAMORPHIC_TERMS[degree]:
        for axis in ("x", "y"):
            params.append((f"c{axis}{phi}{r}Degree{r}", f"C{axis}{phi}{r} - Degree {r}", 0.0))
    return params + [("lensRotation", "Lens Rotation", 0.0), ("squeezeX", "Squeeze-X", 1.0), ("squeezeY", "Squeeze-Y", 1.0)]


# Parameters of each model: (parameter name, 3DE label, default value)
MODEL_PARAMETERS: Dict[str, List[Tuple[str, str, float]]] = {
    CLASSIC: [
        ("distortion", "Distortion", 0.0),
        ("anamorphicSqueeze", "Anamorphic Squeeze", 1.0),
        ("curvatureX", "Curvature X", 0.0),
        ("curvatureY", "Curvature Y", 0.0),
        ("quarticDistortion", "Quartic Distortion", 0.0),
    ],
    RADIAL4: [
        ("distortionDegree2", "Distortion - Degree 2", 0.0),
        ("uDegree2", "U - Degree 2", 0.0),
        ("vDegree2", "V - Degree 2", 0.0),
        ("quarticDistortionDegree4", "Quartic Distortion - Degree 4", 0.0),
        ("uDegree4", "U - Degree 4", 0.0),
        ("vDegree4", "V - Degree 4", 0.0),
        ("phiCylindricDirection", "Phi - Cylindric Direction", 0.0),
        ("bCylindricBending", "B - Cylindric Bending", 0.0),
    ],
    ANAMORPHIC4: _anamorphicParameters(4),
    ANAMORPHIC6: _anamorphicParameters(6),
    RESCALED4: _anamorphicParameters(4) + [("rescale", "Rescale", 1.0)],
    FISHEYE8: [
        ("distortionDegree2", "Distortion - Degree 2", 0.0),
        ("quarticDistortionDegree4", "Quartic Distortion - Degree 4", 0.0),
        ("degree6", "Degree 6", 0.0),
        ("degree8", "Degree 8", 0.0),
    ],
}

# Camera parameters (3DE "tde4_..." knobs of the Nuke LD nodes)
CAMERA_PARAMETERS = [
    ("filmbackWidth", "tde4_filmback_width_cm", 0.0),
    ("filmbackHeight", "tde4_filmback_height_cm", 0.0),
    ("lensCenterOffsetX", "tde4_lens_center_offset_x_cm", 0.0),
    ("lensCenterOffsetY", "tde4_lens_center_offset_y_cm", 0.0),
    ("pixelAspect", "tde4_pixel_aspect", 1.0),
    ("focalLength", "tde4_focal_length_cm", 0.0),
]


# Models: distorted -> undistorted, in dn coordinates (arrays)

def _classic(x, y, p):
    c2, sq, cx, cy, c4 = p
    sq = sq if abs(sq) > 1e-12 else 1.0
    x2, y2 = x * x, y * y
    x4, y4, x2y2 = x2 * x2, y2 * y2, x2 * y2
    xu = x * (1.0 + c2 / sq * x2 + (c2 + cx) / sq * y2 + c4 / sq * x4 + 2.0 * c4 / sq * x2y2 + c4 / sq * y4)
    yu = y * (1.0 + (c2 + cy) * x2 + c2 * y2 + c4 * x4 + 2.0 * c4 * x2y2 + c4 * y4)
    return xu, yu


def _radial4(x, y, p):
    c2, u1, v1, c4, u3, v3, phi, b = p
    x2, y2, xy = x * x, y * y, x * y
    r2 = x2 + y2
    r4 = r2 * r2
    rx = x * (1.0 + c2 * r2 + c4 * r4) + (r2 + 2.0 * x2) * (u1 + u3 * r2) + 2.0 * xy * (v1 + v3 * r2)
    ry = y * (1.0 + c2 * r2 + c4 * r4) + (r2 + 2.0 * y2) * (v1 + v3 * r2) + 2.0 * xy * (u1 + u3 * r2)
    # Cylindric extender
    q = math.sqrt(max(1.0 + b, 1e-12))
    c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    m11 = c * c * q + s * s / q
    m12 = (q - 1.0 / q) * c * s
    m22 = c * c / q + s * s * q
    return m11 * rx + m12 * ry, m12 * rx + m22 * ry


def _anamorphic(x, y, p, pixelAspect, degree, rescale=1.0):
    """ Anamorphic polynomial of `degree` (4 or 6), with lens rotation, squeezes and pixel aspect. """
    terms = _ANAMORPHIC_TERMS[degree]
    coefficients = p[:2 * len(terms)]
    rotation, sqx, sqy = p[2 * len(terms):2 * len(terms) + 3]
    pa = pixelAspect if abs(pixelAspect) > 1e-12 else 1.0
    c, s = math.cos(math.radians(rotation)), math.sin(math.radians(rotation))
    xu, yu = x / pa, y
    rx, ry = c * xu - s * yu, s * xu + c * yu
    r2 = (rx * rx + ry * ry) * (rescale * rescale)
    theta = np.arctan2(ry, rx)
    cosines = {phi: (np.cos(phi * theta) if phi else 1.0) for phi in (0, 2, 4, 6)}
    qx, qy = 1.0, 1.0
    for index, (phi, power) in enumerate(terms):
        term = cosines[phi] * r2 ** (power // 2)
        qx = qx + coefficients[2 * index] * term
        qy = qy + coefficients[2 * index + 1] * term
    px, py = rx * qx, ry * qy
    sx, sy = px * pa * sqx, py * sqy
    return c * sx + s * sy, -s * sx + c * sy


def _fisheye8(x, y, p, focalDn):
    """ Equisolid-angle fisheye (radius 2 f sin(angle / 2)) to a plain projection, then a radial polynomial. """
    c2, c4, c6, c8 = p
    if focalDn <= 0:
        raise ValueError("The fisheye model needs the focal length and the filmback (in cm).")
    radius = np.sqrt(x * x + y * y)
    angle = 2.0 * np.arcsin(np.minimum(radius / (2.0 * focalDn), 0.9999))
    angle = np.minimum(angle, math.pi / 2.0 - 1e-5)
    plain = np.minimum(focalDn * np.tan(angle), 5.0)
    scale = np.where(radius > 0, plain / np.where(radius > 0, radius, 1.0), 1.0)
    px, py = x * scale, y * scale
    r2 = px * px + py * py
    q = 1.0 + c2 * r2 + c4 * r2 ** 2 + c6 * r2 ** 3 + c8 * r2 ** 4
    return px * q, py * q


def _plainToEquisolid(x, y, focalDn):
    """ The equisolid-angle fisheye projection of plain dn coordinates (the inverse of the fisheye mapping). """
    radius = np.sqrt(x * x + y * y)
    fisheye = 2.0 * focalDn * np.sin(np.arctan2(radius, focalDn) / 2.0)
    scale = np.where(radius > 0, fisheye / np.where(radius > 0, radius, 1.0), 1.0)
    return x * scale, y * scale


def undistortDn(model: str, params, x, y, pixelAspect: float = 1.0, focalDn: float = 0.0):
    """
    Undistorted dn coordinates of distorted ones (arrays), with the parameters of the model in 3DE order.
    `focalDn`: the focal length divided by half the filmback diagonal (fisheye model).
    """
    if model == CLASSIC:
        return _classic(x, y, params)
    if model == RADIAL4:
        return _radial4(x, y, params)
    if model == ANAMORPHIC4:
        return _anamorphic(x, y, params, pixelAspect, 4)
    if model == ANAMORPHIC6:
        return _anamorphic(x, y, params, pixelAspect, 6)
    if model == RESCALED4:
        rescale = params[-1] if abs(params[-1]) > 1e-12 else 1.0
        ux, uy = _anamorphic(x, y, params[:-1], pixelAspect, 4, rescale)
        return ux, uy
    if model == FISHEYE8:
        return _fisheye8(x, y, params, focalDn)
    raise ValueError(f"Unknown lens model: {model}")


def distortDn(model: str, params, x, y, pixelAspect: float = 1.0, iterations: int = 12, focalDn: float = 0.0):
    """ Distorted dn coordinates of undistorted ones: the inverse of undistortDn (Newton iterations). """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if model == FISHEYE8 and focalDn > 0:
        # Start from the fisheye projection of the point (a plain point is far outside of the image)
        dx, dy = _plainToEquisolid(x, y, focalDn)
    else:
        dx, dy = x.copy(), y.copy()
    eps = 1e-6
    for _ in range(iterations):
        fx, fy = undistortDn(model, params, dx, dy, pixelAspect, focalDn)
        ex, ey = fx - x, fy - y
        ax, ay = undistortDn(model, params, dx + eps, dy, pixelAspect, focalDn)
        bx, by = undistortDn(model, params, dx, dy + eps, pixelAspect, focalDn)
        j11, j21 = (ax - fx) / eps, (ay - fy) / eps
        j12, j22 = (bx - fx) / eps, (by - fy) / eps
        det = j11 * j22 - j12 * j21
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        stepX = (j22 * ex - j12 * ey) / det
        stepY = (-j21 * ex + j11 * ey) / det
        dx -= stepX
        dy -= stepY
        if float(np.max(np.abs(stepX))) < 1e-10 and float(np.max(np.abs(stepY))) < 1e-10:
            break
    # No solution (strong distortion folding over, or another branch): far outside (black)
    fx, fy = undistortDn(model, params, dx, dy, pixelAspect, focalDn)
    invalid = ~(np.isfinite(dx) & np.isfinite(dy)) | ~(np.abs(fx - x) + np.abs(fy - y) < 1e-6)
    dx[invalid] = 1e6
    dy[invalid] = 1e6
    return dx, dy


# Unit <-> dn coordinates

def filmback(width: int, height: int, camera: dict) -> Tuple[float, float, float, float]:
    """
    (filmback width, height, lens center offset x, y) of a camera: the given filmback in cm, or, when
    it is not set (0), the proportions of the image times its pixel aspect (the offsets are then ignored).
    """
    fbw, fbh = float(camera.get("filmbackWidth") or 0.0), float(camera.get("filmbackHeight") or 0.0)
    if fbw > 0 and fbh > 0:
        return fbw, fbh, float(camera.get("lensCenterOffsetX") or 0.0), float(camera.get("lensCenterOffsetY") or 0.0)
    pa = float(camera.get("pixelAspect") or 1.0)
    return float(width) * pa, float(height), 0.0, 0.0


def focalDn(camera: dict, fb) -> float:
    """ The focal length in dn units (divided by half the filmback diagonal); 0 when it is not set. """
    focal = float(camera.get("focalLength") or 0.0)
    if focal <= 0 or not (float(camera.get("filmbackWidth") or 0.0) > 0 and float(camera.get("filmbackHeight") or 0.0) > 0):
        return 0.0
    return focal / (0.5 * math.hypot(fb[0], fb[1]))


def unitToDn(u, v, fb):
    fbw, fbh, lcx, lcy = fb
    radius = 0.5 * math.hypot(fbw, fbh)
    return ((u - 0.5) * fbw - lcx) / radius, ((v - 0.5) * fbh - lcy) / radius


def dnToUnit(x, y, fb):
    fbw, fbh, lcx, lcy = fb
    radius = 0.5 * math.hypot(fbw, fbh)
    return (x * radius + lcx) / fbw + 0.5, (y * radius + lcy) / fbh + 0.5


def sourcePositions(model: str, params, camera: dict, width: int, height: int, direction: str,
                    step: int = 4) -> np.ndarray:
    """
    For each pixel of the output (rows from the top), the position (Nuke pixel coordinates x, y) of the
    input pixel it shows, (height, width, 2):
    - "undistort": the output is undistorted, the input distorted (the source of an undistorted
      position is its distorted position);
    - "distort": the output is distorted, the input undistorted.
    Computed every `step` pixels and interpolated (the mapping is smooth).
    """
    from .flow import sampleRows
    fb = filmback(width, height, camera)
    pa = float(camera.get("pixelAspect") or 1.0)
    gridW, gridH = int(math.ceil(width / float(step))) + 1, int(math.ceil(height / float(step))) + 1
    # Grid of pixel centers (Nuke y up: rows from the top)
    xs = np.linspace(0.5, width - 0.5, gridW)
    ys = np.linspace(height - 0.5, 0.5, gridH)
    gx, gy = np.meshgrid(xs, ys)
    x, y = unitToDn(gx / width, gy / height, fb)
    focal = focalDn(camera, fb)
    if direction == "undistort":
        sx, sy = distortDn(model, params, x, y, pa, focalDn=focal)
    else:
        sx, sy = undistortDn(model, params, x, y, pa, focal)
    su, sv = dnToUnit(sx, sy, fb)
    grid = np.stack([su * width, sv * height], axis=2)
    grid = np.where(np.isfinite(grid), np.clip(grid, -1e7, 1e7), -1e7).astype(np.float32)
    if gridW == width and gridH == height:
        return grid
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
    rows = rows * (gridH - 1) / max(1.0, height - 1.0)
    columns = columns * (gridW - 1) / max(1.0, width - 1.0)
    return sampleRows(grid, rows, columns)


# Maps of the last parameter sets (the same map serves every frame of a sequence)
_positionsCache: "OrderedDict" = OrderedDict()
_positionsLock = threading.Lock()


def cachedSourcePositions(model: str, params, camera: dict, width: int, height: int, direction: str) -> np.ndarray:
    """ sourcePositions, kept for the last parameter sets (read-only array). """
    key = (model, tuple(float(p) for p in params), tuple(sorted((k, float(v or 0.0)) for k, v in camera.items())),
           int(width), int(height), direction)
    with _positionsLock:
        if key in _positionsCache:
            _positionsCache.move_to_end(key)
            return _positionsCache[key]
    positions = sourcePositions(model, params, camera, width, height, direction)
    positions.setflags(write=False)
    with _positionsLock:
        _positionsCache[key] = positions
        while len(_positionsCache) > 3:
            _positionsCache.popitem(last=False)
    return positions


# 3DE / Nuke files

def _normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def parseNukeLensNode(text: str) -> dict:
    """
    The model and the values of a 3DE4 LD node of a Nuke script (e.g. "LD_3DE4_Anamorphic_Standard_Degree_4 {...}",
    as exported by 3DE): {"model": ..., parameter name: value, ...}. Raises ValueError if there is none.
    """
    match = re.search(r"\b(LD_3DE\w*)\s*\{", text)
    if match is None:
        raise ValueError("No 3DE lens distortion node (LD_3DE...) in the file.")
    className = match.group(1)
    model = NUKE_CLASSES.get(className)
    if model is None:
        raise ValueError(f"Unsupported 3DE lens model: {className} (supported: {', '.join(sorted(NUKE_CLASSES))}).")
    # The body of the node, up to its closing brace
    depth, start, end = 0, match.end() - 1, len(text)
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    body = text[start + 1:end]
    names = {}
    for name, label, _ in MODEL_PARAMETERS[model]:
        names[_normalized(name)] = name
        names[_normalized(label)] = name
    for name, knob, _ in CAMERA_PARAMETERS:
        names[_normalized(knob)] = name
        names[_normalized(name)] = name
    # Knobs at their default value are not written in the script: start from the defaults
    result = {"model": model}
    result.update({name: value for name, _, value in MODEL_PARAMETERS[model]})
    result.update({name: value for name, _, value in CAMERA_PARAMETERS})
    curves = {}
    number = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    for line in body.splitlines():
        line = line.strip()
        knobMatch = re.match(r'^("([^"]+)"|(\S+))\s+(.*)$', line)
        if not knobMatch:
            continue
        knob = knobMatch.group(2) or knobMatch.group(3)
        value = knobMatch.group(4)
        if knob == "direction":
            result["direction"] = "distort" if "distort" == value.strip().strip('"').lower() else "undistort"
            continue
        name = names.get(_normalized(knob))
        if name is None:
            continue
        value = value.strip()
        if value.startswith("{"):
            # Animated knob ({{curve x1001 0.1 0.2 ...}}): keys; an expression is not supported
            keys = _parseCurve(value)
            if not keys:
                raise ValueError(f"Unsupported value of '{knob}': {value}")
            curves[name] = keys
            result[name] = keys[0][1]
            continue
        found = re.match(number, value)
        if found is not None:
            result[name] = float(found.group(0))
    if curves:
        result["curves"] = curves
    return result


def _parseCurve(value: str) -> List[list]:
    """ Keys [[frame, value], ...] of a Nuke curve ("{{curve x1001 0.1 0.2 x1010 0.3}}"), [] if it is not one. """
    match = re.search(r"\bcurve\b([^}]*)", value)
    if match is None:
        return []
    keys = []
    frame = None
    for token in match.group(1).split():
        if re.fullmatch(r"x[-+]?\d+(\.\d*)?", token):
            frame = float(token[1:])
            continue
        try:
            number = float(token)
        except ValueError:
            continue  # Interpolation flags (K, L, s0...)
        if frame is None:
            frame = 1.0
        keys.append([frame, number])
        frame += 1.0
    return keys
