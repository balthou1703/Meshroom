"""
2D point tracking (Tracker node) and the use of tracks by other nodes (Roto shapes linked to a track).

A track is a dict (stored as JSON in the "tracks" parameter of the Tracker node):
    {
        "name": "track1",
        "enabled": true,
        "pattern": 21,          # size in pixels of the pattern matched from frame to frame
        "search": 61,           # size in pixels of the area searched around the previous position
        "keys": {"1001": [x, y], "1002": [x, y], ...}   # positions (Nuke coordinates, y up)
    }
Between two keys the position is interpolated linearly; it is held before the first key and after
the last one.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np


def parseTracks(text: str) -> List[dict]:
    """ Return the tracks stored in a JSON string ([] if empty). Raises ValueError if it is not valid. """
    if not text or not text.strip():
        return []
    tracks = json.loads(text)
    if not isinstance(tracks, list):
        raise ValueError("The tracks must be a JSON list.")
    return tracks


def trackKeys(track: dict) -> List[Tuple[int, float, float]]:
    """ The keys of a track as sorted (frame, x, y). """
    keys = []
    for frame, position in (track.get("keys") or {}).items():
        try:
            keys.append((int(frame), float(position[0]), float(position[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return sorted(keys)


def trackPosition(track: dict, frame: float) -> Optional[Tuple[float, float]]:
    """ Position of a track at a frame (interpolated between its keys), or None if it has no key. """
    keys = trackKeys(track)
    if not keys:
        return None
    if frame <= keys[0][0]:
        return keys[0][1], keys[0][2]
    if frame >= keys[-1][0]:
        return keys[-1][1], keys[-1][2]
    for (f0, x0, y0), (f1, x1, y1) in zip(keys, keys[1:]):
        if f0 <= frame <= f1:
            t = (frame - f0) / float(f1 - f0)
            return x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    return keys[-1][1], keys[-1][2]


def findTrack(tracks: List[dict], name: str) -> Optional[dict]:
    return next((track for track in tracks if track.get("name") == name), None)


def trackOffset(tracks: List[dict], name: str, referenceFrame: float, frame: float) -> Optional[Tuple[float, float]]:
    """ How much a track moved between `referenceFrame` and `frame`, or None (unknown or disabled track). """
    track = findTrack(tracks, name)
    if track is None or not track.get("enabled", True):
        return None
    start, current = trackPosition(track, referenceFrame), trackPosition(track, frame)
    if start is None or current is None:
        return None
    return current[0] - start[0], current[1] - start[1]


def trackCovers(track: dict, frameA: float, frameB: float) -> bool:
    """ Whether a track has keys around both frames (not held before its first key or after its last one). """
    keys = trackKeys(track)
    return bool(keys) and keys[0][0] <= min(frameA, frameB) + 1e-9 and keys[-1][0] >= max(frameA, frameB) - 1e-9


def trackPoints(tracks: List[dict], names, frameA: float, frameB: float, covering: bool = False):
    """
    Positions of the enabled tracks (all of them, or those in `names`) at two frames, as two arrays
    (n, 2) of corresponding points. Tracks without key are skipped. `covering`: only the tracks with
    keys around both frames (e.g. the autotrack of a CameraTracker, whose features each live on a part
    of the sequence: a track held after its last key would count as a still point); all of them when
    none covers both frames.
    """
    if covering:
        selected = [track for track in tracks if track.get("enabled", True) and (not names or track.get("name") in names)]
        covered = [track for track in selected if trackCovers(track, frameA, frameB)]
        if covered:
            return trackPoints(covered, None, frameA, frameB)
    pointsA, pointsB = [], []
    for track in tracks:
        if not track.get("enabled", True) or (names and track.get("name") not in names):
            continue
        a, b = trackPosition(track, frameA), trackPosition(track, frameB)
        if a is not None and b is not None:
            pointsA.append(a)
            pointsB.append(b)
    return np.array(pointsA, dtype=np.float64).reshape(-1, 2), np.array(pointsB, dtype=np.float64).reshape(-1, 2)


def scaleTracks(tracks: List[dict], width: int, height: int) -> List[dict]:
    """
    The tracks in the pixels of a `width` x `height` image: a track records the format of the image it
    was tracked on ("format": [w, h]), its keys are scaled to the format of the image using it (e.g.
    tracked on a 1K proxy, used on the 4K plate). Tracks without format are returned as they are.
    """
    result = []
    for track in tracks:
        trackFormat = track.get("format")
        try:
            valid = len(trackFormat) == 2 and float(trackFormat[0]) > 0 and float(trackFormat[1]) > 0
        except (TypeError, ValueError):
            valid = False
        # No (or an invalid) format, or the same one, or no target format: as it is
        if not valid or width <= 0 or height <= 0 or (trackFormat[0] == width and trackFormat[1] == height):
            result.append(track)
            continue
        sx, sy = width / float(trackFormat[0]), height / float(trackFormat[1])
        scaled = dict(track)
        scaled["keys"] = {frame: [position[0] * sx, position[1] * sy] for frame, position in (track.get("keys") or {}).items()}
        scaled["format"] = [width, height]
        scaled["pattern"] = max(5, int(round(track.get("pattern", 21) * sx)))
        scaled["search"] = max(7, int(round(track.get("search", 61) * sx)))
        result.append(scaled)
    return result


def trackNames(text: str) -> List[str]:
    """ Track names from a comma separated list ("" -> []: all the tracks). """
    return [name.strip() for name in (text or "").split(",") if name.strip()]


# Tracking

def luminance(image) -> tuple:
    """ Luminance of the "rgba" layer of an image (h, w), with its bbox. """
    rgba = image.rgba()
    return (0.2126 * rgba[:, :, 0] + 0.7152 * rgba[:, :, 1] + 0.0722 * rgba[:, :, 2]).astype(np.float32), image.bbox


def _window(lum: np.ndarray, bbox, cx: float, cy: float, size: int) -> Optional[np.ndarray]:
    """ The square (size x size) of a luminance image centered on the pixel containing (cx, cy), or None if outside. """
    half = size // 2
    column = int(np.floor(cx)) - bbox.x0
    row = bbox.y1 - 1 - int(np.floor(cy))
    top, left = row - half, column - half
    if top < 0 or left < 0 or top + size > lum.shape[0] or left + size > lum.shape[1]:
        return None
    return lum[top:top + size, left:left + size]


def _patch(lum: np.ndarray, bbox, cx: float, cy: float, size: int) -> Optional[np.ndarray]:
    """ A (size x size) patch of a luminance image sampled (bilinear) exactly centered on (cx, cy), or None if outside. """
    half = size // 2
    # Continuous array coordinates of the patch pixel centers (column = x - x0 - 0.5, row = y1 - y - 0.5)
    columns = (cx - bbox.x0 - 0.5) + np.arange(-half, half + 1)
    rows = (bbox.y1 - cy - 0.5) + np.arange(-half, half + 1)
    if columns[0] < 0 or rows[0] < 0 or columns[-1] > lum.shape[1] - 1 or rows[-1] > lum.shape[0] - 1:
        return None
    c0, r0 = np.floor(columns).astype(int), np.floor(rows).astype(int)
    c1, r1 = np.minimum(c0 + 1, lum.shape[1] - 1), np.minimum(r0 + 1, lum.shape[0] - 1)
    fc, fr = (columns - c0)[np.newaxis, :], (rows - r0)[:, np.newaxis]
    top = lum[np.ix_(r0, c0)] * (1 - fc) + lum[np.ix_(r0, c1)] * fc
    bottom = lum[np.ix_(r1, c0)] * (1 - fc) + lum[np.ix_(r1, c1)] * fc
    return (top * (1 - fr) + bottom * fr).astype(np.float32)


def _subpixel(scores: np.ndarray, row: int, column: int) -> Tuple[float, float]:
    """ Parabolic refinement of the best score position (row, column offsets in [-0.5, 0.5]). """
    def refine(left, center, right):
        denominator = left - 2 * center + right
        if abs(denominator) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    dy = refine(scores[row - 1, column], scores[row, column], scores[row + 1, column]) \
        if 0 < row < scores.shape[0] - 1 else 0.0
    dx = refine(scores[row, column - 1], scores[row, column], scores[row, column + 1]) \
        if 0 < column < scores.shape[1] - 1 else 0.0
    return dy, dx


def matchPattern(pattern: np.ndarray, lum: np.ndarray, bbox, cx: float, cy: float,
                 searchSize: int) -> Optional[Tuple[float, float, float]]:
    """
    Find `pattern` (a luminance patch) in the search area of `lum` centered on (cx, cy).

    Returns:
        (x, y, score): the new position of the pattern center (Nuke coordinates) and its normalized
        cross-correlation (1: perfect match), or None if the search area leaves the image.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    size = pattern.shape[0]
    searchSize = max(searchSize, size + 2)
    area = _window(lum, bbox, cx, cy, searchSize)
    if area is None:
        return None
    windows = sliding_window_view(area, (size, size))
    # Normalized cross-correlation: robust to brightness and contrast changes
    p = pattern - pattern.mean()
    pNorm = np.sqrt((p * p).sum()) + 1e-6
    means = windows.mean(axis=(2, 3), keepdims=True)
    centered = windows - means
    norms = np.sqrt((centered * centered).sum(axis=(2, 3))) + 1e-6
    scores = np.einsum("ijkl,kl->ij", centered, p) / (norms * pNorm)
    row, column = np.unravel_index(int(np.argmax(scores)), scores.shape)
    dy, dx = _subpixel(scores, row, column)
    # Offset of the best window from the centered one, in pixels (rows go down, y goes up), from the
    # center of the pixel containing (cx, cy)
    center = (searchSize - size) // 2
    offsetX = column + dx - center
    offsetY = -(row + dy - center)
    return np.floor(cx) + 0.5 + offsetX, np.floor(cy) + 0.5 + offsetY, float(scores[row, column])


def refine(pattern: np.ndarray, lum: np.ndarray, bbox, x: float, y: float, iterations: int = 4) -> Tuple[float, float]:
    """
    Sub-pixel refinement of a match (Lucas-Kanade): move (x, y) so that the patch of `lum` sampled
    there best fits `pattern` (brightness and contrast normalized).
    """
    def normalized(patch):
        patch = patch - patch.mean()
        return patch / (patch.std() + 1e-6)
    target = normalized(pattern)
    for _ in range(iterations):
        patch = _patch(lum, bbox, x, y, pattern.shape[0])
        if patch is None:
            break
        current = normalized(patch)
        # Gradients of the sampled patch: along columns (x) and rows (y goes down)
        gradRows, gradColumns = np.gradient(current)
        gx, gy = gradColumns[1:-1, 1:-1], -gradRows[1:-1, 1:-1]
        error = (target - current)[1:-1, 1:-1]
        a = np.array([[(gx * gx).sum(), (gx * gy).sum()], [(gx * gy).sum(), (gy * gy).sum()]])
        b = np.array([(gx * error).sum(), (gy * error).sum()])
        if abs(np.linalg.det(a)) < 1e-9:
            break
        dx, dy = np.linalg.solve(a, b)
        if abs(dx) > 1 or abs(dy) > 1:
            break
        x, y = x + dx, y + dy
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            break
    return x, y


def trackFrames(track: dict, frames: List[int], imageAt, cancel=None, minScore: float = 0.5,
                progress=None, onKey=None, scores: Optional[Dict[str, float]] = None) -> Dict[str, List[float]]:
    """
    Track a point through `frames` (consecutive, forward or backward), starting from its position at
    the first of them. The pattern is taken at the previous frame each time (it follows slow changes
    of appearance).

    Args:
        imageAt: function(frame) -> ImageBuffer, the image to track in.
        cancel: function raising to stop.
        minScore: the tracking stops when the match is worse than this.
        progress: function(index, count) called after each frame.
        onKey: function(frame, x, y, score) called for each tracked frame (live display).
        scores: filled with the match score of each tracked frame {"frame": score} (1: perfect).

    Returns:
        The new keys {"frame": [x, y]} (the start frame excluded). When stopped (`cancel` or `imageAt`
        raising EvaluationCancelled) or when a frame cannot be rendered, the keys found so far.
    """
    from .evaluator import EvaluationCancelled, EvaluationError
    size = int(track.get("pattern", 21)) | 1  # Odd sizes: the pattern has a center pixel
    searchSize = int(track.get("search", 61)) | 1
    start = trackPosition(track, frames[0])
    if start is None:
        return {}
    x, y = start
    keys = {}
    try:
        previous = luminance(imageAt(frames[0]))
    except (EvaluationCancelled, EvaluationError):
        return keys
    for index, frame in enumerate(frames[1:], start=1):
        pattern = _patch(previous[0], previous[1], x, y, size)
        if pattern is None:
            break
        try:
            if cancel:
                cancel()
            current = luminance(imageAt(frame))
        except (EvaluationCancelled, EvaluationError):
            break
        match = matchPattern(pattern, current[0], current[1], x, y, searchSize)
        if match is None or match[2] < minScore:
            break
        x, y = refine(pattern, current[0], current[1], match[0], match[1])
        keys[str(frame)] = [round(float(x), 3), round(float(y), 3)]
        if scores is not None:
            scores[str(frame)] = round(float(match[2]), 4)
        if onKey:
            onKey(frame, keys[str(frame)][0], keys[str(frame)][1], float(match[2]))
        previous = current
        if progress:
            progress(index, len(frames) - 1)
    return keys
