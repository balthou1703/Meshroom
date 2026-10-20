"""
Camera tracking (CameraTracker node): automatic tracking of features through a sequence, solve of
the camera motion and of the 3D positions of the features (structure from motion), export to USD.

Conventions:
- tracks are those of meshroom/live/tracking.py (Nuke pixel coordinates, y up);
- the solve works in a "computer vision" frame: image x right / y down, camera looking along +z,
  a camera is (R, t) with x_camera = R x_world + t; the world is the first solved camera;
- the USD export converts it to a Y-up world with cameras looking along -z.

The solve is a classic incremental reconstruction: an initial pair of frames with enough parallax
(essential matrix, RANSAC), triangulation, then each other frame by resection (PnP, RANSAC) with new
points triangulated as they come, and a final bundle adjustment when scipy is available.
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import tracking


# Feature detection and automatic tracking

def _boxSum(values: np.ndarray, radius: int) -> np.ndarray:
    """ Sum of `values` over (2 radius + 1)^2 windows (same shape, zero padded). """
    padded = np.pad(values, radius + 1)
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    return (integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size])[
        :values.shape[0], :values.shape[1]]


def detectFeatures(lum: np.ndarray, bbox, count: int, minDistance: float = 20.0, margin: int = 30,
                   existing: Optional[List[Tuple[float, float]]] = None,
                   mask: Optional[np.ndarray] = None) -> List[Tuple[float, float]]:
    """
    Corners good to track (Shi-Tomasi: largest minimum eigenvalue of the structure tensor), spread
    over the image (one per cell of `minDistance` pixels), away from the `existing` points, and
    where `mask` (same shape as `lum`, if given) is above 0.5.

    Returns:
        Up to `count` positions (x, y) in Nuke coordinates.
    """
    if lum.shape[0] <= 2 * margin or lum.shape[1] <= 2 * margin:
        return []
    # HDR highlights compressed: a bright light must not hide the rest (no gamma: it would lift the grain)
    lum = np.maximum(lum.astype(np.float64), 0.0)
    lum = lum / (1.0 + lum)
    gradRows, gradColumns = np.gradient(lum)
    xx = _boxSum(gradColumns * gradColumns, 2)
    yy = _boxSum(gradRows * gradRows, 2)
    xy = _boxSum(gradColumns * gradRows, 2)
    score = (xx + yy) / 2 - np.sqrt(((xx - yy) / 2) ** 2 + xy ** 2)
    score[:margin, :] = 0
    score[-margin:, :] = 0
    score[:, :margin] = 0
    score[:, -margin:] = 0
    if mask is not None:
        score[mask < 0.5] = 0
    cell = max(4, int(minDistance))
    rows, columns = score.shape[0] // cell, score.shape[1] // cell
    cropped = score[:rows * cell, :columns * cell].reshape(rows, cell, columns, cell).transpose(0, 2, 1, 3)
    flat = cropped.reshape(rows, columns, cell * cell)
    best = flat.argmax(axis=2)
    values = flat.max(axis=2)
    # Relative to the typical strong corners (not the strongest one)
    positive = values[values > 0]
    threshold = max(1e-10, 0.02 * float(np.percentile(positive, 90))) if positive.size else 1e-10
    candidates = []
    for r in range(rows):
        for c in range(columns):
            if values[r, c] < threshold:
                continue
            row = r * cell + best[r, c] // cell
            column = c * cell + best[r, c] % cell
            candidates.append((values[r, c], bbox.x0 + column + 0.5, bbox.y1 - row - 0.5))
    candidates.sort(reverse=True)
    existing = list(existing or [])
    result = []
    for _, x, y in candidates:
        if len(result) >= count:
            break
        if any((x - ex) ** 2 + (y - ey) ** 2 < minDistance ** 2 for ex, ey in existing + result):
            continue
        result.append((x, y))
    return result


def _inMask(mask: Optional[np.ndarray], bbox, x: float, y: float) -> bool:
    """ Whether the point (Nuke coordinates) is where the mask (over `bbox`) is above 0.5 (True without mask). """
    if mask is None:
        return True
    row, column = int(math.floor(bbox.y1 - y)), int(math.floor(x - bbox.x0))
    if row < 0 or column < 0 or row >= mask.shape[0] or column >= mask.shape[1]:
        return False
    return bool(mask[row, column] >= 0.5)


def autoTrack(imageAt: Callable, frames: List[int], count: int = 150, pattern: int = 15, search: int = 41,
              minScore: float = 0.7, checkCancelled=None, progress=None, maskAt: Optional[Callable] = None,
              onFrame: Optional[Callable] = None, device=None) -> List[dict]:
    """
    Detect and track features through `frames` (consecutive): features lost along the way are
    replaced by new ones, so that about `count` are tracked at each frame. With `maskAt(frame, bbox)`
    (the mask values (h, w) over the bbox of the image), features are only detected and kept where
    the mask is above 0.5 (e.g. to leave out a moving object). `onFrame(frame, tracks)` is called after
    each frame with the tracks so far (those with 2 keys at least). With a GPU `device` (meshroom/live/gpu.py), the
    patterns of all the features are matched at once on it.

    Returns:
        Tracks ({"name", "keys", "scores", "pattern", "search", "auto": True}).
    """
    pattern, search = int(pattern) | 1, int(search) | 1
    tracks: List[dict] = []
    active: List[Tuple[int, float, float]] = []

    def addFeatures(lum, bbox, frame, size, mask):
        positions = detectFeatures(lum, bbox, count - len(active), minDistance=max(10.0, pattern * 1.5),
                                   margin=search // 2 + 2, existing=[(x, y) for _, x, y in active], mask=mask)
        for x, y in positions:
            tracks.append({"name": f"auto{len(tracks) + 1}", "enabled": True, "pattern": pattern, "search": search,
                           "auto": True, "keys": {str(frame): [round(x, 3), round(y, 3)]}, "scores": {},
                           "format": list(size)})
            active.append((len(tracks) - 1, x, y))

    first = imageAt(frames[0])
    size = first.format
    previous = tracking.luminance(first)
    addFeatures(previous[0], previous[1], frames[0], size, maskAt(frames[0], previous[1]) if maskAt else None)
    for index, frame in enumerate(frames[1:], start=1):
        if checkCancelled:
            checkCancelled()
        current = tracking.luminance(imageAt(frame))
        mask = maskAt(frame, current[1]) if maskAt else None
        stillActive = []
        patches = [(trackIndex, x, y, tracking._patch(previous[0], previous[1], x, y, pattern)) for trackIndex, x, y in active]
        patches = [item for item in patches if item[3] is not None]
        matches = tracking.matchMany(np.array([item[3] for item in patches]).reshape(-1, pattern, pattern),
                                     current[0], current[1], np.array([item[1:3] for item in patches]).reshape(-1, 2),
                                     search, device)
        for (trackIndex, x, y, patch), match in zip(patches, matches):
            if not np.isfinite(match[2]) or match[2] < minScore:
                continue
            nx, ny = tracking.refine(patch, current[0], current[1], match[0], match[1])
            if not _inMask(mask, current[1], nx, ny):
                continue  # Entered the masked out area
            tracks[trackIndex]["keys"][str(frame)] = [round(float(nx), 3), round(float(ny), 3)]
            tracks[trackIndex]["scores"][str(frame)] = round(float(match[2]), 4)
            stillActive.append((trackIndex, nx, ny))
        active = stillActive
        # New features where tracks were lost
        if len(active) < 0.75 * count:
            addFeatures(current[0], current[1], frame, size, mask)
        previous = current
        if progress:
            progress(index / float(max(1, len(frames) - 1)))
        if onFrame:
            onFrame(frame, [track for track in tracks if len(track["keys"]) >= 2])
    # A feature seen on one frame only is useless
    return [track for track in tracks if len(track["keys"]) >= 2]


def autoTrackMatched(imageAt: Callable, frames: List[int], matcher, count: int = 150, minScore: float = 0.3,
                     checkCancelled=None, progress=None, maskAt: Optional[Callable] = None,
                     onFrame: Optional[Callable] = None, pattern: int = 15, search: int = 41) -> List[dict]:
    """
    autoTrack with a feature matcher (LightGlue, meshroom/live/neural/matching.py) instead of patterns: the features of
    each frame are matched with those of the next one; a feature matched again from there extends its track (the
    features of a frame are detected once, so a feature keeps exactly its position). New tracks start from the best
    new matches while fewer than `count` are tracked. Same arguments and result as autoTrack (`minScore`: the
    least match score of LightGlue, in [0, 1]).
    """
    tracks: List[dict] = []
    # Tracks reaching the previous frame, by the rounded position of their feature there
    active: Dict[Tuple[int, int], int] = {}
    first = imageAt(frames[0])
    size = first.format
    previous, previousFeatures = first, matcher.features(first)

    def key(x, y):
        return int(round(x * 100)), int(round(y * 100))
    for index, frame in enumerate(frames[1:], start=1):
        if checkCancelled:
            checkCancelled()
        current = imageAt(frame)
        currentFeatures = matcher.features(current)
        pointsA, pointsB, scores = matcher.match(previous, current, previousFeatures, currentFeatures)
        mask = maskAt(frame, current.bbox) if maskAt else None
        order = np.argsort(-scores)
        stillActive: Dict[Tuple[int, int], int] = {}
        starts = []
        for i in order:
            (xa, ya), (xb, yb), score = pointsA[i], pointsB[i], scores[i]
            if score < minScore or not _inMask(mask, current.bbox, xb, yb):
                continue
            trackIndex = active.get(key(xa, ya))
            if trackIndex is None:
                starts.append(i)
                continue
            tracks[trackIndex]["keys"][str(frame)] = [round(float(xb), 3), round(float(yb), 3)]
            tracks[trackIndex]["scores"][str(frame)] = round(float(score), 4)
            stillActive[key(xb, yb)] = trackIndex
        for i in starts:
            if len(stillActive) >= count:
                break
            (xa, ya), (xb, yb) = pointsA[i], pointsB[i]
            tracks.append({"name": f"auto{len(tracks) + 1}", "enabled": True, "pattern": pattern, "search": search,
                           "auto": True, "format": list(size),
                           "keys": {str(frames[index - 1]): [round(float(xa), 3), round(float(ya), 3)],
                                    str(frame): [round(float(xb), 3), round(float(yb), 3)]},
                           "scores": {str(frame): round(float(scores[i]), 4)}})
            stillActive[key(xb, yb)] = len(tracks) - 1
        active = stillActive
        previous, previousFeatures = current, currentFeatures
        if progress:
            progress(index / float(max(1, len(frames) - 1)))
        if onFrame:
            onFrame(frame, list(tracks))
    return tracks


def previewFeatures(image, nextImage, count: int, pattern: int, search: int, minScore: float,
                    mask: Optional[np.ndarray] = None) -> List[dict]:
    """
    What the autotrack would do at a frame with these parameters (to choose them): the features
    detected in `image`, each matched in `nextImage` (the next frame, or None).

    Returns:
        [{"x", "y", "score": correlation or None (lost / no next frame), "nx", "ny": position in the
        next frame or None}, ...]
    """
    pattern, search = int(pattern) | 1, int(search) | 1
    lum, bbox = tracking.luminance(image)
    positions = detectFeatures(lum, bbox, count, minDistance=max(10.0, pattern * 1.5), margin=search // 2 + 2,
                               mask=mask)
    following = tracking.luminance(nextImage) if nextImage is not None else None
    result = []
    for x, y in positions:
        feature = {"x": x, "y": y, "score": None, "nx": None, "ny": None}
        if following is not None:
            patch = tracking._patch(lum, bbox, x, y, pattern)
            match = tracking.matchPattern(patch, following[0], following[1], x, y, search) if patch is not None else None
            if match is not None:
                feature["score"] = float(match[2])
                feature["nx"], feature["ny"] = float(match[0]), float(match[1])
        result.append(feature)
    return result


def drawPreview(rgba: np.ndarray, bbox, features: List[dict], pattern: int, search: int, minScore: float,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    The image (h, w, 4 over `bbox`) with the preview of the features drawn over it: the pattern
    squares (green: matched in the next frame above the minimum correlation, red: lost, yellow: no
    next frame), the search areas (faint), the motion to the next frame, and the masked out areas darker.
    """
    out = rgba.copy()
    if mask is not None:
        out[:, :, :3] *= np.where(mask[:, :, np.newaxis] >= 0.5, 1.0, 0.35).astype(np.float32)
    height, width = out.shape[:2]
    # Lines about 2 pixels wide at 1080p (visible when the image is shown smaller)
    thickness = max(1, int(round(min(height, width) / 540.0)))
    offsets = [(dx, dy) for dx in range(thickness) for dy in range(thickness)]

    def plot(xs, ys, color, alpha=1.0):
        xs = np.concatenate([np.asarray(xs, np.float64) + dx - thickness // 2 for dx, _ in offsets])
        ys = np.concatenate([np.asarray(ys, np.float64) + dy - thickness // 2 for _, dy in offsets])
        columns = np.floor(xs - bbox.x0).astype(int)
        rows = np.floor(bbox.y1 - ys).astype(int)
        keep = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
        rows, columns = rows[keep], columns[keep]
        out[rows, columns, :3] = out[rows, columns, :3] * (1 - alpha) + np.asarray(color, np.float32) * alpha
        out[rows, columns, 3] = np.maximum(out[rows, columns, 3], alpha)

    def square(x, y, half, color, alpha=1.0):
        steps = np.arange(-half, half + 1, dtype=np.float64)
        ones = np.ones_like(steps)
        plot(np.r_[x + steps, x + steps, (x - half) * ones, (x + half) * ones],
             np.r_[(y - half) * ones, (y + half) * ones, y + steps, y + steps], color, alpha)

    def line(x0, y0, x1, y1, color):
        count = int(max(abs(x1 - x0), abs(y1 - y0))) + 2
        t = np.linspace(0.0, 1.0, count)
        plot(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, color)

    for feature in features:
        x, y, score = feature["x"], feature["y"], feature["score"]
        if feature["nx"] is None and score is None:
            color = (1.0, 0.85, 0.2) if all(f["score"] is None for f in features) else (1.0, 0.2, 0.2)
        else:
            color = (0.3, 1.0, 0.3) if score is not None and score >= minScore else (1.0, 0.2, 0.2)
        square(x, y, search // 2, color, 0.3)
        square(x, y, pattern // 2, color)
        if feature["nx"] is not None:
            line(x, y, feature["nx"], feature["ny"], color)
    return out


# Geometry

def rodrigues(vector: np.ndarray) -> np.ndarray:
    """ Rotation matrix of a rotation vector. """
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.eye(3)
    k = vector / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * kx @ kx


def rodriguesMany(vectors: np.ndarray) -> np.ndarray:
    """ Rotation matrices (n, 3, 3) of rotation vectors (n, 3). """
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(vectors, axis=1)
    safe = np.where(theta < 1e-12, 1.0, theta)
    k = vectors / safe[:, np.newaxis]
    kx = np.zeros((len(vectors), 3, 3))
    kx[:, 0, 1], kx[:, 0, 2] = -k[:, 2], k[:, 1]
    kx[:, 1, 0], kx[:, 1, 2] = k[:, 2], -k[:, 0]
    kx[:, 2, 0], kx[:, 2, 1] = -k[:, 1], k[:, 0]
    sin, cos = np.sin(theta)[:, np.newaxis, np.newaxis], np.cos(theta)[:, np.newaxis, np.newaxis]
    result = np.eye(3)[np.newaxis] + sin * kx + (1 - cos) * kx @ kx
    result[theta < 1e-12] = np.eye(3)
    return result


def rotationVector(matrix: np.ndarray) -> np.ndarray:
    """ Rotation vector of a rotation matrix. """
    angle = math.acos(max(-1.0, min(1.0, (np.trace(matrix) - 1) / 2)))
    if angle < 1e-12:
        return np.zeros(3)
    if abs(angle - math.pi) < 1e-6:
        # 180 degrees: the axis is the eigenvector of the eigenvalue 1
        values, vectors = np.linalg.eig(matrix)
        axis = np.real(vectors[:, np.argmin(np.abs(values - 1))])
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    return axis / (2 * math.sin(angle)) * angle


def triangulate(poses: List[Tuple[np.ndarray, np.ndarray]], observations: List[np.ndarray]) -> np.ndarray:
    """ Point seen at the normalized image positions `observations` from the cameras `poses` (linear DLT). """
    rows = []
    for (rotation, translation), (x, y) in zip(poses, observations):
        projection = np.c_[rotation, translation]
        rows.append(x * projection[2] - projection[0])
        rows.append(y * projection[2] - projection[1])
    _, _, vt = np.linalg.svd(np.array(rows))
    point = vt[-1]
    return point[:3] / point[3] if abs(point[3]) > 1e-12 else np.full(3, np.nan)


def _depth(pose, point) -> float:
    return float((pose[0] @ point + pose[1])[2])


def essentialMatrix(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """ Essential matrix from 8 or more normalized correspondences (x_b^T E x_a = 0). """
    if len(a) < 8:
        return None
    rows = np.c_[b[:, 0:1] * a, b[:, 0:1], b[:, 1:2] * a, b[:, 1:2], a, np.ones((len(a), 1))]
    _, _, vt = np.linalg.svd(rows)
    e = vt[-1].reshape(3, 3)
    u, _, vt2 = np.linalg.svd(e)
    return u @ np.diag([1.0, 1.0, 0.0]) @ vt2


def _sampson(e: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ha, hb = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
    ea, etb = ha @ e.T, hb @ e
    numerator = (hb * ea).sum(axis=1) ** 2
    return numerator / (ea[:, 0] ** 2 + ea[:, 1] ** 2 + etb[:, 0] ** 2 + etb[:, 1] ** 2 + 1e-18)


def relativePose(a: np.ndarray, b: np.ndarray, threshold: float, rng: np.random.Generator, iterations: int = 500):
    """
    Pose (R, t) of camera b relatively to camera a (|t| = 1) from normalized correspondences, by
    RANSAC on the essential matrix. Returns (R, t, inlier mask) or None.
    """
    best, bestCount = None, 0
    for _ in range(iterations):
        sample = rng.choice(len(a), 8, replace=False)
        e = essentialMatrix(a[sample], b[sample])
        if e is None:
            continue
        inliers = _sampson(e, a, b) < threshold ** 2
        if inliers.sum() > bestCount:
            best, bestCount = e, inliers.sum()
    if best is None or bestCount < 8:
        return None
    inliers = _sampson(best, a, b) < threshold ** 2
    e = essentialMatrix(a[inliers], b[inliers])
    inliers = _sampson(e, a, b) < threshold ** 2
    u, _, vt = np.linalg.svd(e)
    if np.linalg.det(u) < 0:
        u = -u
    if np.linalg.det(vt) < 0:
        vt = -vt
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    # The 4 solutions: the right one has the points in front of both cameras
    best = None
    for rotation in (u @ w @ vt, u @ w.T @ vt):
        for translation in (u[:, 2], -u[:, 2]):
            poses = [(np.eye(3), np.zeros(3)), (rotation, translation)]
            front = 0
            for i in np.nonzero(inliers)[0][:200]:
                point = triangulate(poses, [a[i], b[i]])
                if np.all(np.isfinite(point)) and _depth(poses[0], point) > 0 and _depth(poses[1], point) > 0:
                    front += 1
            if best is None or front > best[0]:
                best = (front, rotation, translation)
    return best[1], best[2], inliers


def _projectError(pose, points, observations) -> np.ndarray:
    camera = points @ pose[0].T + pose[1]
    depth = np.where(np.abs(camera[:, 2]) < 1e-12, 1e-12, camera[:, 2])
    return np.linalg.norm(camera[:, :2] / depth[:, np.newaxis] - observations, axis=1)


def _poseDLT(points: np.ndarray, observations: np.ndarray):
    """ Camera pose from 6 or more 3D points and their normalized image positions (DLT). """
    rows = []
    for (x, y, z), (u, v) in zip(points, observations):
        rows.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        rows.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
    _, _, vt = np.linalg.svd(np.array(rows))
    projection = vt[-1].reshape(3, 4)
    left = projection[:, :3]
    u, singular, vt2 = np.linalg.svd(left)
    scale = singular.mean()
    rotation = u @ vt2
    if np.linalg.det(rotation) < 0:
        rotation, scale = -rotation, -scale
    translation = projection[:, 3] / scale
    # Points in front of the camera
    if np.median((points @ rotation.T + translation)[:, 2]) < 0:
        return None
    return rotation, translation


def _refinePose(pose, points, observations, iterations: int = 8):
    """ Gauss-Newton refinement of a pose on the reprojection errors (numerical Jacobian). """
    params = np.r_[rotationVector(pose[0]), pose[1]]

    def residuals(p):
        camera = points @ rodrigues(p[:3]).T + p[3:]
        depth = np.where(np.abs(camera[:, 2]) < 1e-9, 1e-9, camera[:, 2])
        return (camera[:, :2] / depth[:, np.newaxis] - observations).ravel()

    for _ in range(iterations):
        r = residuals(params)
        jacobian = np.empty((len(r), 6))
        for k in range(6):
            step = np.zeros(6)
            step[k] = 1e-6
            jacobian[:, k] = (residuals(params + step) - r) / 1e-6
        try:
            delta = np.linalg.lstsq(jacobian, -r, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        params = params + delta
        if np.linalg.norm(delta) < 1e-10:
            break
    return rodrigues(params[:3]), params[3:]


def resection(points: np.ndarray, observations: np.ndarray, threshold: float, rng: np.random.Generator,
              iterations: int = 200):
    """ Camera pose from 3D points and their normalized image positions, by RANSAC (PnP). (pose, inliers) or None. """
    if len(points) < 6:
        return None
    best, bestInliers = None, None
    for _ in range(iterations):
        sample = rng.choice(len(points), 6, replace=False)
        pose = _poseDLT(points[sample], observations[sample])
        if pose is None:
            continue
        inliers = _projectError(pose, points, observations) < threshold
        if bestInliers is None or inliers.sum() > bestInliers.sum():
            best, bestInliers = pose, inliers
    if best is None or bestInliers.sum() < 6:
        return None
    pose = _refinePose(best, points[bestInliers], observations[bestInliers])
    return pose, _projectError(pose, points, observations) < threshold


def bundleAdjust(poses: Dict[int, tuple], points: Dict[int, np.ndarray], observations, fixed: int,
                 maxIterations: int = 60):
    """
    Refine poses and points together on all the reprojection errors (scipy least squares, Huber
    loss). The pose of the frame `fixed` does not move. Without scipy, nothing is done.

    Args:
        observations: list of (frame, point id, normalized (x, y)).
    """
    try:
        from scipy.optimize import least_squares
        from scipy.sparse import coo_matrix
    except ImportError:
        logging.info("[CameraTracker] scipy is not available: no bundle adjustment.")
        return poses, points
    frames = [frame for frame in sorted(poses) if frame != fixed]
    pointIds = sorted(points)
    frameIndex = {frame: i for i, frame in enumerate(frames)}
    pointIndex = {pid: i for i, pid in enumerate(pointIds)}
    obs = [(frame, pid, xy) for frame, pid, xy in observations if frame in poses and pid in points]
    if not obs or not frames:
        return poses, points
    x0 = np.r_[np.concatenate([np.r_[rotationVector(poses[f][0]), poses[f][1]] for f in frames]),
               np.concatenate([points[pid] for pid in pointIds])]
    nCam = 6 * len(frames)
    obsFrames = np.array([frameIndex.get(frame, -1) for frame, _, _ in obs])
    obsPoints = np.array([pointIndex[pid] for _, pid, _ in obs])
    obsXY = np.array([xy for _, _, xy in obs])
    fixedPose = poses[fixed]

    def residuals(params):
        cams = params[:nCam].reshape(-1, 6)
        pts = params[nCam:].reshape(-1, 3)
        # The fixed camera last (index -1 of the observations)
        rotations = np.concatenate([rodriguesMany(cams[:, :3]), fixedPose[0][np.newaxis]])
        translations = np.concatenate([cams[:, 3:], fixedPose[1][np.newaxis]])
        camera = np.einsum("nij,nj->ni", rotations[obsFrames], pts[obsPoints]) + translations[obsFrames]
        z = np.where(np.abs(camera[:, 2]) > 1e-9, camera[:, 2], 1e-9)
        return (camera[:, :2] / z[:, np.newaxis] - obsXY).ravel()

    # Jacobian sparsity: each residual (2 per observation) depends on its camera (6) and its point (3)
    rows2 = np.arange(2 * len(obs)).reshape(-1, 2)
    withCamera = obsFrames >= 0
    cameraRows = np.repeat(rows2[withCamera], 6, axis=1).ravel()
    cameraCols = np.tile((6 * obsFrames[withCamera])[:, np.newaxis] + np.arange(6), (1, 2)).ravel()
    pointRows = np.repeat(rows2, 3, axis=1).ravel()
    pointCols = np.tile((nCam + 3 * obsPoints)[:, np.newaxis] + np.arange(3), (1, 2)).ravel()
    sparsity = coo_matrix((np.ones(len(cameraRows) + len(pointRows), dtype=int),
                           (np.r_[cameraRows, pointRows], np.r_[cameraCols, pointCols])),
                          shape=(2 * len(obs), len(x0))).tocsr()
    solution = least_squares(residuals, x0, jac_sparsity=sparsity, loss="huber", f_scale=2e-3,
                             x_scale="jac", max_nfev=maxIterations, method="trf")
    params = solution.x
    newPoses = dict(poses)
    for frame, i in frameIndex.items():
        cam = params[6 * i:6 * i + 6]
        newPoses[frame] = (rodrigues(cam[:3]), cam[3:])
    newPoints = {pid: params[nCam + 3 * i:nCam + 3 * i + 3] for pid, i in pointIndex.items()}
    return newPoses, newPoints


def solveCamera(tracks: List[dict], width: int, height: int, focalPixels: float, checkCancelled=None,
                progress=None, seed: int = 0, quick: bool = False) -> dict:
    """
    Solve the camera of a tracked sequence.

    Args:
        tracks: tracks (keys in Nuke pixel coordinates).
        focalPixels: focal length in pixels (focal length / film back width x image width).
        quick: fewer RANSAC iterations and adjustments (to compare focal lengths, see estimateFocal).

    Returns:
        {"width", "height", "focalPixels", "cameras": {frame: {"R": 3x3, "t": 3}}, "points": {track name: xyz},
         "error": RMS reprojection error in pixels, "frames": [solved frames],
         "interpolated": [frames of the tracks whose camera could not be solved: interpolated between the
         solved ones (held before the first / after the last)]}

    Raises:
        RuntimeError: if the tracks do not allow a solve (too few, no parallax...).
    """
    rng = np.random.default_rng(seed)
    cx, cy = width / 2.0, height / 2.0
    # Normalized observations per frame: {frame: {track index: (x, y)}}, image y down
    observations: Dict[int, Dict[int, np.ndarray]] = {}
    for index, track in enumerate(tracks):
        if not track.get("enabled", True):
            continue
        for frame, (x, y) in (track.get("keys") or {}).items():
            observations.setdefault(int(frame), {})[index] = np.array([(x - cx) / focalPixels,
                                                                       ((height - y) - cy) / focalPixels])
    frames = sorted(observations)
    if len(frames) < 2:
        raise RuntimeError("Track the sequence first (Autotrack): at least 2 frames are needed.")
    # Inlier threshold: 2 pixels, more for big images (the tracking precision is in their pixels)
    threshold = max(2.0, width / 1000.0) / focalPixels

    # Initial pair: the first frame with enough common tracks, and a later frame with enough parallax
    first = max(frames[:max(1, len(frames) // 4)], key=lambda f: len(observations[f]))
    best = None
    for frame in frames:
        if frame == first:
            continue
        common = sorted(set(observations[first]) & set(observations[frame]))
        if len(common) < 20:
            continue
        a = np.array([observations[first][i] for i in common])
        b = np.array([observations[frame][i] for i in common])
        parallax = float(np.median(np.linalg.norm(a - b, axis=1)))
        score = parallax * min(len(common), 200)
        if best is None or score > best[0]:
            best = (score, frame, common, a, b, parallax)
        if parallax > 0.08:
            break
    if best is None:
        raise RuntimeError("Not enough tracks shared between frames (20 at least): track more features.")
    _, second, common, a, b, parallax = best
    if parallax < 0.005:
        raise RuntimeError("Not enough camera motion (parallax) to solve: a nodal pan or a still camera.")
    relative = relativePose(a, b, threshold, rng, iterations=200 if quick else 500)
    if relative is None:
        raise RuntimeError("No consistent camera motion found between the frames (check the focal length).")
    rotation, translation, inliers = relative
    poses = {first: (np.eye(3), np.zeros(3)), second: (rotation, translation)}
    points: Dict[int, np.ndarray] = {}
    for k, trackIndex in enumerate(common):
        if not inliers[k]:
            continue
        point = triangulate([poses[first], poses[second]], [a[k], b[k]])
        if np.all(np.isfinite(point)) and _depth(poses[first], point) > 0 and _depth(poses[second], point) > 0:
            points[trackIndex] = point

    def triangulateNew(frame):
        """ Points seen in `frame` and in the solved frame the farthest in time, not yet in 3D. """
        for trackIndex, xy in observations[frame].items():
            if trackIndex in points:
                continue
            others = [f for f in poses if f != frame and trackIndex in observations[f]]
            if not others:
                continue
            other = max(others, key=lambda f: abs(f - frame))
            pair = [poses[other], poses[frame]]
            point = triangulate(pair, [observations[other][trackIndex], xy])
            if not np.all(np.isfinite(point)) or _depth(pair[0], point) <= 0 or _depth(pair[1], point) <= 0:
                continue
            # Enough angle between the two views, and a good reprojection
            rays = [point - (-p[0].T @ p[1]) for p in pair]
            cosine = np.dot(rays[0], rays[1]) / (np.linalg.norm(rays[0]) * np.linalg.norm(rays[1]) + 1e-12)
            if cosine > math.cos(math.radians(1.0)):
                continue
            errors = [_projectError(p, point[np.newaxis], o[np.newaxis])[0]
                      for p, o in zip(pair, [observations[other][trackIndex], xy])]
            if max(errors) < threshold:
                points[trackIndex] = point

    # Resection of the other frames, outward from the solved ones; several passes: the frames that had too
    # few known points get a new chance once the points of their neighbours are triangulated
    remaining = [f for f in frames if f not in poses]
    for passIndex in range(4):
        unsolved = [f for f in remaining if f not in poses]
        if not unsolved:
            break
        unsolved.sort(key=lambda f: min(abs(f - solvedFrame) for solvedFrame in poses))
        added = 0
        for frame in unsolved:
            if checkCancelled:
                checkCancelled()
            known = [i for i in observations[frame] if i in points]
            if len(known) >= 6:
                result = resection(np.array([points[i] for i in known]),
                                   np.array([observations[frame][i] for i in known]), threshold * 1.5, rng,
                                   iterations=60 if quick else 200)
                if result is not None:
                    poses[frame] = result[0]
                    triangulateNew(frame)
                    added += 1
            if progress:
                progress(0.8 * len(poses) / len(frames))
        if not added:
            break
        # Points seen by the newly solved frames, then a quick adjustment against the drift
        for frame in list(poses):
            triangulateNew(frame)
        if not quick and any(f not in poses for f in remaining):
            adjusted = [(f, i, xy) for f in poses for i, xy in observations[f].items() if i in points]
            poses, points = bundleAdjust(poses, points, adjusted, fixed=first, maxIterations=20)
    if len(poses) < 2 or len(points) < 8:
        raise RuntimeError("The solve failed: too few frames or points could be reconstructed.")

    # Bundle adjustment of everything
    allObservations = [(frame, i, xy) for frame in poses for i, xy in observations[frame].items() if i in points]
    poses, points = bundleAdjust(poses, points, allObservations, fixed=first, maxIterations=20 if quick else 60)
    if progress:
        progress(1.0)

    errors = []
    for frame, i, xy in allObservations:
        errors.append(_projectError(poses[frame], points[i][np.newaxis], xy[np.newaxis])[0] * focalPixels)
    errors = np.array(errors)
    # Points of the outliers (more than 5 pixels) are dropped
    bad = {i for (frame, i, _), e in zip(allObservations, errors) if e > 5.0}
    points = {i: p for i, p in points.items() if i not in bad}
    good = errors[errors <= 5.0] if np.any(errors <= 5.0) else errors
    # The frames of the tracks without a solved camera: interpolated, so that the camera covers them all
    solvedFrames = sorted(poses)
    cameras = dict(poses)
    interpolated = [f for f in range(frames[0], frames[-1] + 1) if f not in poses]
    for frame in interpolated:
        cameras[frame] = interpolatePose(poses, frame)
    return {
        "width": width, "height": height, "focalPixels": focalPixels,
        "cameras": {str(frame): {"R": np.round(pose[0], 9).tolist(), "t": np.round(pose[1], 9).tolist()}
                    for frame, pose in sorted(cameras.items())},
        "points": {tracks[i].get("name", str(i)): np.round(p, 6).tolist() for i, p in points.items()},
        "error": round(float(np.sqrt(np.mean(good ** 2))), 4),
        "frames": solvedFrames,
        "interpolated": interpolated,
    }


def interpolatePose(poses: Dict[int, tuple], frame: int) -> tuple:
    """
    The pose (R, t) at a frame between solved poses: the camera center interpolated linearly and the
    rotation along the shortest arc; held before the first and after the last solved frame.
    """
    solved = sorted(poses)
    before = [f for f in solved if f <= frame]
    after = [f for f in solved if f >= frame]
    if not before:
        return poses[after[0]]
    if not after:
        return poses[before[-1]]
    a, b = before[-1], after[0]
    if a == b:
        return poses[a]
    weight = (frame - a) / float(b - a)
    (ra, ta), (rb, tb) = poses[a], poses[b]
    centerA, centerB = -ra.T @ ta, -rb.T @ tb
    delta = rotationVector(rb @ ra.T)
    rotation = rodrigues(delta * weight) @ ra
    center = centerA + (centerB - centerA) * weight
    return rotation, -rotation @ center


def estimateFocal(tracks: List[dict], width: int, height: int, candidates: List[float], checkCancelled=None,
                  progress=None, maxFrames: int = 40) -> float:
    """
    Estimate the focal length in pixels: the candidate (focal lengths in pixels) whose solve reprojects
    the tracks the best (on `maxFrames` frames spread over the sequence), refined between its neighbours.
    The frames that cannot be solved count as errors. Raises RuntimeError if no candidate solves.
    """
    allFrames = sorted({int(f) for track in tracks for f in (track.get("keys") or {})})
    step = max(1, int(math.ceil(len(allFrames) / float(maxFrames))))
    kept = {str(f) for f in allFrames[::step]}
    subset = [dict(track, keys={f: xy for f, xy in (track.get("keys") or {}).items() if f in kept}) for track in tracks]
    subset = [track for track in subset if len(track["keys"]) >= 2]
    scores = {}

    def score(focal):
        key = round(focal, 1)
        if key not in scores:
            if checkCancelled:
                checkCancelled()
            try:
                solve = solveCamera(subset, width, height, focal, quick=True)
                missing = 1.0 - len(solve["frames"]) / float(max(1, len(kept)))
                scores[key] = solve["error"] * (1.0 + 4.0 * missing)
            except RuntimeError:
                scores[key] = float("inf")
        return scores[key]

    candidates = sorted(candidates)
    for index, focal in enumerate(candidates):
        score(focal)
        if progress:
            progress(0.7 * (index + 1) / len(candidates))
    best = min(candidates, key=score)
    if not math.isfinite(score(best)):
        raise RuntimeError("The focal length could not be estimated: no candidate solves the tracks.")
    # Golden section search between the neighbours of the best candidate
    index = candidates.index(best)
    low = candidates[max(0, index - 1)]
    high = candidates[min(len(candidates) - 1, index + 1)]
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    for iteration in range(6):
        a = high - ratio * (high - low)
        b = low + ratio * (high - low)
        if score(a) < score(b):
            high = b
        else:
            low = a
        if progress:
            progress(0.7 + 0.3 * (iteration + 1) / 6.0)
    return min([best, (low + high) / 2.0], key=score)


# USD export

_CV_TO_USD = np.diag([1.0, -1.0, -1.0])  # x right, y down, z forward -> x right, y up, z backward


def _usdMatrix(rotation: np.ndarray, position: np.ndarray) -> str:
    """ USD matrix4d (row vectors: the rows are the local axes, then the position). """
    rows = [list(rotation[:, i]) + [0.0] for i in range(3)] + [list(position) + [1.0]]
    return "(" + ", ".join("(" + ", ".join(f"{v:.9g}" for v in row) + ")" for row in rows) + ")"


def exportCameraUsd(solve: dict, path: str, focalLength: float, filmBackWidth: float, fps: float = 24.0,
                    scale: float = 1.0):
    """ Write the solved camera as an animated USD camera (ASCII .usda), in a Y-up world. """
    width, height = solve["width"], solve["height"]
    frames = sorted(int(f) for f in solve["cameras"])
    samples = []
    for frame in frames:
        camera = solve["cameras"][str(frame)]
        rotation, translation = np.array(camera["R"]), np.array(camera["t"])
        # Camera to world in the CV frame, then in the USD world (Y up) with a USD camera (looking along -z)
        position = _CV_TO_USD @ (-rotation.T @ translation) * scale
        cameraToWorld = _CV_TO_USD @ rotation.T @ _CV_TO_USD
        samples.append(f"            {frame}: {_usdMatrix(cameraToWorld, position)},")
    verticalAperture = filmBackWidth * height / float(width)
    text = f'''#usda 1.0
(
    defaultPrim = "Camera"
    startTimeCode = {frames[0]}
    endTimeCode = {frames[-1]}
    timeCodesPerSecond = {fps:g}
    upAxis = "Y"
    metersPerUnit = 1
    doc = "Camera solved by the Meshroom CameraTracker"
)

def Camera "Camera"
{{
    float focalLength = {focalLength:g}
    float horizontalAperture = {filmBackWidth:g}
    float verticalAperture = {verticalAperture:.6g}
    float2 clippingRange = (0.01, 100000)
    matrix4d xformOp:transform.timeSamples = {{
{chr(10).join(samples)}
    }}
    uniform token[] xformOpOrder = ["xformOp:transform"]
}}
'''
    with open(path, "w") as file:
        file.write(text)


def exportPointsUsd(solve: dict, path: str, scale: float = 1.0):
    """ Write the solved 3D points as a USD point cloud (ASCII .usda), in the same Y-up world as the camera. """
    names = sorted(solve["points"])
    positions = np.array([solve["points"][name] for name in names], dtype=np.float64).reshape(-1, 3)
    positions = (positions @ _CV_TO_USD.T) * scale
    extent = float(np.ptp(positions, axis=0).max()) if len(positions) else 1.0
    width = max(extent * 0.004, 1e-4)
    points = ", ".join(f"({x:.6g}, {y:.6g}, {z:.6g})" for x, y, z in positions)
    text = f'''#usda 1.0
(
    defaultPrim = "PointCloud"
    upAxis = "Y"
    metersPerUnit = 1
    doc = "Points solved by the Meshroom CameraTracker ({len(names)} points)"
)

def Points "PointCloud"
{{
    point3f[] points = [{points}]
    float[] widths = [{", ".join(f"{width:.6g}" for _ in names)}]
    color3f[] primvars:displayColor = [(0.9, 0.75, 0.3)] (
        interpolation = "constant"
    )
}}
'''
    with open(path, "w") as file:
        file.write(text)
