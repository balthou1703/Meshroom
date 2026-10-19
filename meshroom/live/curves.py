"""
Curves of the lookup nodes (ColorLookup, ColorLookupHSV, TimeWarp) and color matching.

A set of curves is stored as JSON: {"curveName": [[x, y], ...], ...}. A curve goes smoothly through
its points (monotone cubic interpolation: no overshoot between two points) and continues in
straight lines beyond its first and last points. A curve without point is the identity, a curve
with one point is constant.
"""
from __future__ import annotations

import colorsys
import json
from typing import Dict, List, Optional, Sequence

import numpy as np


def parseCurves(text: str) -> Dict[str, List[List[float]]]:
    """ Return the curves stored in a JSON string. Raises ValueError if it is not valid. """
    if not text or not text.strip():
        return {}
    curves = json.loads(text)
    if not isinstance(curves, dict):
        raise ValueError("The curves must be a JSON object {name: [[x, y], ...]}.")
    return {name: sortedPoints(points) for name, points in curves.items()}


def sortedPoints(points) -> List[List[float]]:
    """ The valid points of a curve, sorted by x, one point per x. """
    result = {}
    for point in points or []:
        try:
            result[float(point[0])] = float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
    return [[x, result[x]] for x in sorted(result)]


def _slopes(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """ Tangents of a monotone cubic interpolation (Fritsch-Carlson). """
    deltas = np.diff(ys) / np.diff(xs)
    slopes = np.empty_like(ys)
    slopes[0], slopes[-1] = deltas[0], deltas[-1]
    for i in range(1, len(xs) - 1):
        if deltas[i - 1] * deltas[i] <= 0:
            slopes[i] = 0.0
        else:
            # Weighted harmonic mean: keeps the curve monotone between monotone points
            w1 = 2 * (xs[i + 1] - xs[i]) + (xs[i] - xs[i - 1])
            w2 = (xs[i + 1] - xs[i]) + 2 * (xs[i] - xs[i - 1])
            slopes[i] = (w1 + w2) / (w1 / deltas[i - 1] + w2 / deltas[i])
    return slopes


def evaluate(points: Sequence[Sequence[float]], values) -> np.ndarray:
    """ Evaluate a curve at `values` (any shape). """
    values = np.asarray(values, dtype=np.float64)
    points = sortedPoints(points)
    if not points:
        return values.copy()
    if len(points) == 1:
        return np.full_like(values, points[0][1])
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    slopes = _slopes(xs, ys)
    result = np.empty_like(values)
    # Straight lines beyond the ends
    below, above = values < xs[0], values > xs[-1]
    result[below] = ys[0] + (values[below] - xs[0]) * slopes[0]
    result[above] = ys[-1] + (values[above] - xs[-1]) * slopes[-1]
    inside = ~(below | above)
    v = values[inside]
    index = np.clip(np.searchsorted(xs, v, side="right") - 1, 0, len(xs) - 2)
    h = xs[index + 1] - xs[index]
    t = (v - xs[index]) / h
    t2, t3 = t * t, t * t * t
    result[inside] = ((2 * t3 - 3 * t2 + 1) * ys[index] + (t3 - 2 * t2 + t) * h * slopes[index]
                      + (-2 * t3 + 3 * t2) * ys[index + 1] + (t3 - t2) * h * slopes[index + 1])
    return result


def isIdentity(points) -> bool:
    """ Whether a curve leaves its values unchanged. """
    points = sortedPoints(points)
    if not points:
        return True
    if len(points) == 1:
        return False
    return all(abs(x - y) < 1e-9 for x, y in points)


def addPoint(points, x: float, y: float, tolerance: float = 1e-3) -> List[List[float]]:
    """ A copy of a curve with the point (x, y), replacing a point at the same x. """
    kept = [p for p in sortedPoints(points) if abs(p[0] - x) > tolerance]
    return sortedPoints(kept + [[x, y]])


# HSV (vectorized, hue in [0, 1))

def rgbToHsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maximum = np.max(rgb[..., :3], axis=-1)
    minimum = np.min(rgb[..., :3], axis=-1)
    delta = maximum - minimum
    safe = np.where(delta > 1e-12, delta, 1.0)
    hue = np.where(maximum == r, (g - b) / safe, np.where(maximum == g, 2.0 + (b - r) / safe, 4.0 + (r - g) / safe))
    hue = np.where(delta > 1e-12, (hue / 6.0) % 1.0, 0.0)
    saturation = np.where(maximum > 1e-12, delta / np.where(maximum > 1e-12, maximum, 1.0), 0.0)
    return np.stack([hue, saturation, maximum], axis=-1)


def hsvToRgb(hsv: np.ndarray) -> np.ndarray:
    hue, saturation, value = hsv[..., 0] % 1.0, hsv[..., 1], hsv[..., 2]
    h6 = hue * 6.0
    sector = np.floor(h6).astype(np.int64) % 6
    fraction = h6 - np.floor(h6)
    p = value * (1 - saturation)
    q = value * (1 - saturation * fraction)
    t = value * (1 - saturation * (1 - fraction))
    choices = [(value, t, p), (q, value, p), (p, value, t), (p, q, value), (t, p, value), (value, p, q)]
    rgb = np.empty(hue.shape + (3,), dtype=np.float64)
    for channel in range(3):
        rgb[..., channel] = np.choose(sector, [choice[channel] for choice in choices])
    return rgb


def colorToHsv(color: Sequence[float]):
    return colorsys.rgb_to_hsv(*[max(0.0, float(c)) for c in color[:3]])


# Color matching

_QUANTILES = (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98)


def _quantilePoints(source: np.ndarray, target: np.ndarray, quantiles=_QUANTILES) -> List[List[float]]:
    """ Points mapping the quantiles of `source` onto those of `target` (histogram matching). """
    xs = np.quantile(source, quantiles)
    ys = np.quantile(target, quantiles)
    points = []
    for x, y in zip(xs, ys):
        if points and x - points[-1][0] < 1e-4:
            continue  # Flat distributions: one point per value
        points.append([round(float(x), 5), round(float(y), 5)])
    return points


def _sample(pixels: np.ndarray, maxCount: int = 200000) -> np.ndarray:
    """ Up to `maxCount` pixels (rgb) of an image, regularly sampled. """
    flat = pixels.reshape(-1, pixels.shape[-1])[:, :3]
    step = max(1, len(flat) // maxCount)
    return flat[::step].astype(np.float64)


def matchCurvesRGB(source: np.ndarray, target: np.ndarray) -> Dict[str, List[List[float]]]:
    """ Red, green and blue curves making the colors of `source` (rgba pixels) match those of `target`. """
    src, tgt = _sample(source), _sample(target)
    curves = {"master": [[0.0, 0.0], [1.0, 1.0]]}
    for index, name in enumerate(("red", "green", "blue")):
        curves[name] = _quantilePoints(src[:, index], tgt[:, index])
    return curves


def matchCurvesHSV(source: np.ndarray, target: np.ndarray) -> Dict[str, List[List[float]]]:
    """
    Hue, saturation and value curves making `source` match `target`: the saturations and values are
    matched by histogram, the hue is shifted by the difference of the mean hues (weighted by the
    saturation, on the circle of hues).
    """
    src, tgt = rgbToHsv(_sample(source)), rgbToHsv(_sample(target))

    def meanHue(hsv):
        angles = hsv[:, 0] * 2 * np.pi
        weights = hsv[:, 1]
        return np.arctan2((np.sin(angles) * weights).sum(), (np.cos(angles) * weights).sum()) / (2 * np.pi)

    shift = (meanHue(tgt) - meanHue(src) + 0.5) % 1.0 - 0.5
    return {
        "hue": [[0.0, round(float(shift), 5)], [1.0, round(float(1.0 + shift), 5)]],
        "saturation": _quantilePoints(src[:, 1], tgt[:, 1]),
        "value": _quantilePoints(src[:, 2], tgt[:, 2]),
    }


def lookupPoints(curves: Dict[str, list], mode: str, source: Sequence[float], target: Sequence[float]):
    """
    A copy of `curves` with the points moving the color `source` to `target` (picked in the viewer):
    on the red, green and blue curves ("rgb" mode), or the hue, saturation and value curves ("hsv").
    """
    curves = {name: sortedPoints(points) for name, points in curves.items()}
    if mode == "hsv":
        srcHsv, tgtHsv = colorToHsv(source), colorToHsv(target)
        # The nearest target hue (around the circle)
        hueTarget = srcHsv[0] + ((tgtHsv[0] - srcHsv[0] + 0.5) % 1.0 - 0.5)
        for name, x, y in (("hue", srcHsv[0], hueTarget), ("saturation", srcHsv[1], tgtHsv[1]),
                           ("value", srcHsv[2], tgtHsv[2])):
            curves[name] = addPoint(curves.get(name) or [[0.0, 0.0], [1.0, 1.0]], round(x, 5), round(y, 5))
        return curves
    for index, name in enumerate(("red", "green", "blue")):
        curves[name] = addPoint(curves.get(name) or [[0.0, 0.0], [1.0, 1.0]],
                                round(float(source[index]), 5), round(float(target[index]), 5))
    return curves


def frameRange(points) -> Optional[tuple]:
    """ The (first, last) x of a curve with at least two points, None otherwise. """
    points = sortedPoints(points)
    if len(points) < 2:
        return None
    return points[0][0], points[-1][0]
