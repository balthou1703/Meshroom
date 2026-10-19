"""
Roto shapes: closed Bézier curves rasterized into antialiased mattes, and open Bézier curves
drawn as antialiased strokes.

A shape is a dict (stored as JSON in the "shapes" parameter of the Roto node):
    {
        "name": "Bezier1",
        "closed": true,          # false: an open curve, drawn as a stroke of "width" pixels
        "width": 4.0,            # width of the stroke of an open curve
        "visible": true,
        "invert": false,
        "opacity": 1.0,
        "feather": 0.0,          # width of the soft edge in pixels (a blur of the shape matte)
        "points": [{"x": .., "y": .., "inX": .., "inY": .., "outX": .., "outY": ..}, ...],
        "keys": [{"frame": 1001, "points": [...]}, ...],   # optional: animated points
        "links": [{"input": "tracker", "track": "track1", "point": -1, "frame": 1001}, ...]  # optional
    }
Point coordinates are in pixels, in Nuke coordinates (y up). The tangents ("in" before the point,
"out" after it) are relative to the point: (0, 0) tangents give a corner.

An animated shape has keys: its points are interpolated linearly between two keys (all keys have
the same number of points), held before the first key and after the last one.

A linked shape follows 2D tracks (see meshroom/live/tracking.py), read from the Tracker connected to
the image input "input" of the Roto: a link moves the whole shape ("point": -1) or one point by the
offset of its track between the link frame and the current frame. A point with its own link follows
its track instead of the links of the whole shape (e.g. the shape follows track1 and one of its
points follows track2). The former single "link" is read as a link to the "tracker" input.
"""
from __future__ import annotations

import json
import math
from typing import List, Optional

import numpy as np

from .imageBuffer import BBox

# Vertical supersampling of the scanlines (horizontal coverage is computed exactly)
SUBSAMPLES = 4


def parseShapes(text: str) -> List[dict]:
    """ Return the shapes stored in a JSON string ([] if empty). Raises ValueError if it is not valid. """
    if not text or not text.strip():
        return []
    shapes = json.loads(text)
    if not isinstance(shapes, list):
        raise ValueError("The shapes must be a JSON list.")
    return shapes


_POINT_KEYS = ("x", "y", "inX", "inY", "outX", "outY")


def _lerpPoints(pointsA, pointsB, t):
    return [{key: (1 - t) * a.get(key, 0.0) + t * b.get(key, 0.0) for key in _POINT_KEYS}
            for a, b in zip(pointsA, pointsB)]


def pointsAtFrame(shape: dict, frame: Optional[float]) -> List[dict]:
    """ The points of a shape at a frame: interpolated between its keys when it is animated. """
    keys = sorted((key for key in shape.get("keys") or [] if key.get("points")), key=lambda key: key["frame"])
    if not keys or frame is None:
        return shape.get("points", [])
    if frame <= keys[0]["frame"]:
        return keys[0]["points"]
    if frame >= keys[-1]["frame"]:
        return keys[-1]["points"]
    for before, after in zip(keys, keys[1:]):
        if before["frame"] <= frame <= after["frame"]:
            if len(before["points"]) != len(after["points"]):
                return before["points"]
            span = float(after["frame"] - before["frame"])
            return _lerpPoints(before["points"], after["points"], (frame - before["frame"]) / span if span else 0.0)
    return keys[-1]["points"]


def shapeLinks(shape: dict) -> List[dict]:
    """ The track links of a shape (the former single "link" is a link to the "tracker" input). """
    links = list(shape.get("links") or [])
    if shape.get("link"):
        links.append(dict(shape["link"], input=shape["link"].get("input", "tracker")))
    return [link for link in links if isinstance(link, dict) and link.get("track")]


def shapeAtFrame(shape: dict, frame: Optional[float], tracks=None) -> dict:
    """
    A copy of a shape with its points at `frame`: animation keys, then the tracks it is linked to.

    Args:
        tracks: {input name: list of tracks} of the connected Trackers (a list: the "tracker" input).
    """
    from .tracking import trackOffset
    result = dict(shape)
    points = [dict(point) for point in pointsAtFrame(shape, frame)]
    if isinstance(tracks, list):
        tracks = {"tracker": tracks}
    links = shapeLinks(shape)
    if links and tracks and frame is not None:
        shapeOffset = [0.0, 0.0]
        pointOffsets = {}
        for link in links:
            offset = trackOffset(tracks.get(link.get("input", "tracker")) or [], link["track"],
                                 link.get("frame", frame), frame)
            if offset is None:
                continue
            index = link.get("point", -1)
            if index is None or index < 0:
                shapeOffset[0] += offset[0]
                shapeOffset[1] += offset[1]
            else:
                current = pointOffsets.setdefault(index, [0.0, 0.0])
                current[0] += offset[0]
                current[1] += offset[1]
        for i, point in enumerate(points):
            # A point with its own link follows it instead of the links of the whole shape
            dx, dy = pointOffsets.get(i, shapeOffset)
            point["x"] = point.get("x", 0.0) + dx
            point["y"] = point.get("y", 0.0) + dy
    result["points"] = points
    return result


def scaleShapes(shapes: List[dict], scale: float) -> List[dict]:
    """ The shapes in the pixels of an image `scale` times smaller or larger (proxy renders): points, keys, widths. """
    def scalePoints(points):
        return [{key: (value * scale if key in _POINT_KEYS else value) for key, value in point.items()}
                for point in points or []]
    result = []
    for shape in shapes:
        shape = dict(shape)
        shape["points"] = scalePoints(shape.get("points"))
        if shape.get("keys"):
            shape["keys"] = [dict(key, points=scalePoints(key.get("points"))) for key in shape["keys"]]
        for key in ("feather", "width"):
            if isinstance(shape.get(key), (int, float)):
                shape[key] = shape[key] * scale
        result.append(shape)
    return result


def setKey(shape: dict, frame: float, points: List[dict]) -> dict:
    """ A copy of a shape with a key of `points` at `frame` (replacing the key of that frame). """
    result = dict(shape)
    keys = [key for key in shape.get("keys") or [] if key.get("frame") != frame]
    keys.append({"frame": frame, "points": [dict(point) for point in points]})
    result["keys"] = sorted(keys, key=lambda key: key["frame"])
    return result


def advectShape(shape: dict, frame: float, nextFrame: float, flowAt) -> dict:
    """
    A copy of a shape with a key at `nextFrame`: its points at `frame` moved by the flow, `flowAt(x, y)`
    giving the motion (u, v) of a point (its tangent handles are moved with it, so the curve deforms
    with the image). Without keys, the shape gets a key at `frame` too.
    """
    points = [dict(point) for point in pointsAtFrame(shape, frame)]
    # A key where the tracking starts: the shape does not change at that frame
    hasKey = any(key.get("frame") == frame and key.get("points") for key in shape.get("keys") or [])
    result = shape if hasKey else setKey(shape, frame, points)
    moved = []
    for point in points:
        x, y = point.get("x", 0.0), point.get("y", 0.0)
        u, v = flowAt(x, y)
        newPoint = dict(point, x=x + u, y=y + v)
        for prefix in ("in", "out"):
            hx, hy = x + point.get(prefix + "X", 0.0), y + point.get(prefix + "Y", 0.0)
            if (hx, hy) != (x, y):
                hu, hv = flowAt(hx, hy)
                newPoint[prefix + "X"] = hx + hu - newPoint["x"]
                newPoint[prefix + "Y"] = hy + hv - newPoint["y"]
        moved.append(newPoint)
    return setKey(result, nextFrame, moved)


def _cubic(p0, p1, p2, p3, count):
    """ `count` points of a cubic Bézier segment, its start included, its end excluded. """
    t = np.linspace(0.0, 1.0, count, endpoint=False)[:, np.newaxis]
    u = 1.0 - t
    return u ** 3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t ** 3 * p3


def polygon(shape: dict, tolerance: float = 1.0, closed: bool = True) -> np.ndarray:
    """
    Flatten a shape into a polygon, shape (n, 2), about `tolerance` pixels between two points along
    the curves. Not `closed`: the polyline of an open curve (no segment from the last point to the
    first one, the last point included).
    """
    points = shape.get("points", [])
    if len(points) < 2:
        return np.zeros((0, 2))
    vertices = []
    count = len(points)
    for index in range(count if closed else count - 1):
        a, b = points[index], points[(index + 1) % count]
        p0 = np.array([a["x"], a["y"]], dtype=np.float64)
        p3 = np.array([b["x"], b["y"]], dtype=np.float64)
        p1 = p0 + np.array([a.get("outX", 0.0), a.get("outY", 0.0)])
        p2 = p3 + np.array([b.get("inX", 0.0), b.get("inY", 0.0)])
        if np.allclose(p1, p0) and np.allclose(p2, p3):
            vertices.append(p0[np.newaxis, :])  # Straight segment
            continue
        # Length of the control polygon: an upper bound of the curve length
        length = np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)
        steps = int(min(max(math.ceil(length / max(tolerance, 0.1)), 4), 400))
        vertices.append(_cubic(p0, p1, p2, p3, steps))
    if not closed:
        vertices.append(np.array([[points[-1]["x"], points[-1]["y"]]], dtype=np.float64))
    return np.concatenate(vertices)


def _orientedLoop(loop: np.ndarray) -> np.ndarray:
    """ The loop counterclockwise (positive area): all the loops of a union have the same orientation. """
    x, y = loop[:, 0], loop[:, 1]
    area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return loop if area >= 0 else loop[::-1]


def strokeLoops(vertices: np.ndarray, width: float, joinAngle: float = 0.15) -> List[np.ndarray]:
    """
    Polygons whose union (non-zero winding) is the stroke of a polyline of `width` pixels: a quad per
    segment, and a disc at both ends and where the direction turns by more than `joinAngle` radians
    (round caps and joins).
    """
    half = max(float(width), 0.0) / 2.0
    if half <= 0 or len(vertices) == 0:
        return []
    sides = max(8, min(32, int(math.ceil(half * 2))))
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1) * half
    loops = []
    # Drop the repeated points
    keep = np.r_[True, np.any(np.abs(np.diff(vertices, axis=0)) > 1e-9, axis=1)]
    vertices = vertices[keep]
    if len(vertices) == 1:
        return [vertices[0] + circle]
    directions = np.diff(vertices, axis=0)
    lengths = np.linalg.norm(directions, axis=1)
    normals = np.stack([-directions[:, 1], directions[:, 0]], axis=1) / lengths[:, np.newaxis] * half
    for a, b, n in zip(vertices[:-1], vertices[1:], normals):
        loops.append(_orientedLoop(np.array([a - n, b - n, b + n, a + n])))
    headings = np.arctan2(directions[:, 1], directions[:, 0])
    turns = np.abs((np.diff(headings) + np.pi) % (2.0 * np.pi) - np.pi)
    joints = [0, len(vertices) - 1] + [i + 1 for i in np.flatnonzero(turns > joinAngle)]
    for i in joints:
        loops.append(vertices[i] + circle)
    return loops


def polygonBBox(vertices: np.ndarray) -> Optional[BBox]:
    if len(vertices) == 0:
        return None
    x0, y0 = np.floor(vertices.min(axis=0)).astype(int)
    x1, y1 = np.ceil(vertices.max(axis=0)).astype(int)
    return BBox(int(x0), int(y0), int(x1) + 1, int(y1) + 1)


def rasterize(vertices, bbox: BBox, cancel=None) -> np.ndarray:
    """
    Antialiased coverage (0..1) of a closed polygon (non-zero winding rule) over `bbox`,
    shape (h, w), rows from the top (Nuke y1) to the bottom. `vertices` can also be a list of
    polygons: the coverage of their union (when they have the same orientation).

    Each pixel row is sampled by SUBSAMPLES scanlines; along a scanline, the coverage of the spans
    inside the polygon is exact (accumulated as area differences, then summed).
    """
    height, width = bbox.height, bbox.width
    coverage = np.zeros((height, width), dtype=np.float32)
    loops = [loop for loop in (vertices if isinstance(vertices, list) else [vertices]) if len(loop) >= 3]
    if not loops or height <= 0 or width <= 0:
        return coverage
    start = np.concatenate(loops)
    end = np.concatenate([np.roll(loop, -1, axis=0) for loop in loops])
    # Ignore the horizontal edges
    keep = start[:, 1] != end[:, 1]
    start, end = start[keep], end[keep]
    if len(start) == 0:
        return coverage
    # Edges oriented upward, with their winding direction
    direction = np.where(end[:, 1] > start[:, 1], 1, -1)
    low = np.where(direction[:, np.newaxis] > 0, start, end)
    high = np.where(direction[:, np.newaxis] > 0, end, start)
    slope = (high[:, 0] - low[:, 0]) / (high[:, 1] - low[:, 1])

    # Process bands of rows to bound the memory (edges x scanlines)
    bandRows = max(1, int(2_000_000 // max(1, len(start) * SUBSAMPLES)))
    for top in range(0, height, bandRows):
        if cancel:
            cancel()
        bottom = min(height, top + bandRows)
        # Scanline heights (Nuke y) of the rows [top, bottom), from the top of the bbox
        rows = np.arange(top, bottom)
        offsets = (np.arange(SUBSAMPLES) + 0.5) / SUBSAMPLES
        scanY = (bbox.y1 - rows[:, np.newaxis] - 1 + (1.0 - offsets[np.newaxis, :])).ravel()
        # Crossings of each edge with each scanline: [low.y, high.y) avoids counting vertices twice
        crosses = (scanY[np.newaxis, :] >= low[:, 1:2]) & (scanY[np.newaxis, :] < high[:, 1:2])
        edgeIndex, lineIndex = np.nonzero(crosses)
        if len(edgeIndex) == 0:
            continue
        xs = low[edgeIndex, 0] + (scanY[lineIndex] - low[edgeIndex, 1]) * slope[edgeIndex] - bbox.x0
        winds = direction[edgeIndex]
        order = np.lexsort((xs, lineIndex))
        xs, winds, lineIndex = xs[order], winds[order], lineIndex[order]
        # Winding number after each crossing, per scanline (it is back to 0 at the end of each scanline)
        after = np.cumsum(winds)
        lineStarts = np.r_[0, np.nonzero(np.diff(lineIndex))[0] + 1]
        offsetsPerLine = np.repeat(np.r_[0, after[lineStarts[1:] - 1]], np.diff(np.r_[lineStarts, len(after)]))
        after = after - offsetsPerLine
        before = after - winds
        # Span edges: entering (0 -> non-zero) or leaving (non-zero -> 0) the polygon
        enter = (before == 0) & (after != 0)
        leave = (before != 0) & (after == 0)
        events = enter | leave
        weight = np.where(enter, 1.0, -1.0)[events]
        positions = np.clip(xs[events], 0.0, float(width))
        lines = lineIndex[events]
        pixel = np.floor(positions).astype(np.int64)
        fraction = positions - pixel
        # Area accumulation: a span starting at x covers 1 - frac(x) of its first pixel, then whole pixels
        accumulation = np.zeros((len(scanY), width + 2), dtype=np.float64)
        np.add.at(accumulation, (lines, pixel), weight * (1.0 - fraction))
        np.add.at(accumulation, (lines, pixel + 1), weight * fraction)
        lineCoverage = np.cumsum(accumulation, axis=1)[:, :width]
        coverage[top:bottom] = lineCoverage.reshape(bottom - top, SUBSAMPLES, width).mean(axis=1)
    np.clip(coverage, 0.0, 1.0, out=coverage)
    return coverage


def shapesMatte(shapes: List[dict], bbox: BBox, cancel=None, frame: Optional[float] = None,
                tracks: Optional[List[dict]] = None, feather: float = 0.0) -> np.ndarray:
    """
    Matte (h, w) of the visible shapes over `bbox` at `frame`: each shape is rasterized (filled if
    closed, a stroke of its width if open), feathered (its own feather plus `feather`), inverted,
    scaled by its opacity, and combined with the previous ones with "over" (a + b (1 - a)).
    """
    from .ops import blur, blurRadius
    matte = np.zeros((bbox.height, bbox.width), dtype=np.float32)
    for shape in shapes:
        if not shape.get("visible", True):
            continue
        closed = shape.get("closed", True) is not False
        shape = shapeAtFrame(shape, frame, tracks)
        if closed:
            vertices = polygon(shape)
            shapeBBox = polygonBBox(vertices)
        else:
            vertices = strokeLoops(polygon(shape, closed=False), float(shape.get("width", 4.0) or 0.0))
            shapeBBox = polygonBBox(np.concatenate(vertices)) if vertices else None
        if shapeBBox is None:
            continue
        shapeFeather = max(float(shape.get("feather") or 0.0) + float(feather or 0.0), 0.0)
        margin = blurRadius(shapeFeather) + 1
        # Only rasterize around the shape (and its feather)
        region = shapeBBox.expanded(margin, margin).intersection(bbox)
        coverage = np.zeros((bbox.height, bbox.width), dtype=np.float32)
        if not region.isEmpty:
            local = rasterize(vertices, region, cancel)
            if shapeFeather > 0:
                blurred = blur(local[:, :, np.newaxis], shapeFeather, shapeFeather, cancel)[:, :, 0]
                radius = (blurred.shape[0] - local.shape[0]) // 2
                local = blurred[radius:radius + local.shape[0], radius:radius + local.shape[1]]
            coverage[bbox.y1 - region.y1:bbox.y1 - region.y0, region.x0 - bbox.x0:region.x1 - bbox.x0] = local
        if shape.get("invert", False):
            coverage = 1.0 - coverage
        opacity = shape.get("opacity")
        coverage *= float(1.0 if opacity is None else opacity)
        matte = matte + coverage * (1.0 - matte)
    return matte
