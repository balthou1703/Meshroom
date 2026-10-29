"""
Tracking of a rigid geometry in a plate (the Resection node, like GeoTracker): the pose of the camera (the geometry
staying still) follows the image, driven by many points of the surface instead of a few pins.

- Surface samples: points spread on the faces of the geometry (or its vertices for a point cloud), with their normals.
- At a frame whose pose is known, the samples seen by the camera (in front, facing it, not hidden: a coarse depth
  buffer) on a textured part of the image (Shi-Tomasi corner score) are taken.
- Their motion to the next frame is measured, by the dense optical flow of the image (meshroom/live/flow.py) or by
  matching a patch around each of them (meshroom/live/tracking.py), searched where the pose moving at constant speed
  puts them; the pose of the next frame is solved from them, rejecting the points that do not fit (robustPose).
- `trackPose` repeats it frame after frame; `alignPose` finds the pose of a frame against a reference frame whose pose
  is known (the patches of the reference searched around where the current pose puts them, coarse to fine): "the right
  position"; `blendPoses` spreads the drift between two keyframes (tracked from both ends, weighted by the distance).

Positions are in Nuke pixels of the plate format (y up); cameras are camera-to-world matrices (looking along -Z).
"""
from __future__ import annotations

import weakref
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import scene3d, tracking

# The masks of the images (setMask): where the geometry may be followed
_MASKS = weakref.WeakKeyDictionary()


def setMask(image, mask: Optional[np.ndarray]):
    """
    Give an image its mask (h, w) over its bbox: the geometry is only followed where it is above 0.5 (e.g. an
    occluding object or a moving part left out, a roto inverted). None removes it. Returns the image.
    """
    if mask is None:
        _MASKS.pop(image, None)
    else:
        _MASKS[image] = np.asarray(mask, dtype=np.float32).reshape(image.bbox.height, image.bbox.width)
    return image


def maskedOut(image, positions: np.ndarray) -> np.ndarray:
    """ Whether each image position (n, 2), Nuke pixels, is masked out (False everywhere without mask). """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    mask = _MASKS.get(image) if image is not None else None
    if mask is None or not len(positions):
        return np.zeros(len(positions), dtype=bool)
    bbox = image.bbox
    rows = np.floor(bbox.y1 - positions[:, 1]).astype(np.int64)
    columns = np.floor(positions[:, 0] - bbox.x0).astype(np.int64)
    inside = (rows >= 0) & (rows < mask.shape[0]) & (columns >= 0) & (columns < mask.shape[1])
    out = np.ones(len(positions), dtype=bool)
    out[inside] = mask[rows[inside], columns[inside]] < 0.5
    return out

# flow: Lucas-Kanade optical flow (flow.py); raft: RAFT optical flow (neural/raft.py, Lucas-Kanade without it);
# patches: the patch of each point searched in the next frame
METHODS = ("flow", "raft", "patches")


def projectPoints(matrix: np.ndarray, points: np.ndarray, width: int, height: int, focalPixels: float):
    """ Image positions (n, 2) of world points seen by a camera (camera-to-world `matrix`), and their depth (n,). """
    view = np.linalg.inv(matrix)
    camera = points @ view[:3, :3].T + view[:3, 3]
    depth = -camera[:, 2]
    safe = np.where(np.abs(depth) < 1e-12, 1e-12, depth)
    xy = np.c_[width / 2.0 + focalPixels * camera[:, 0] / safe, height / 2.0 + focalPixels * camera[:, 1] / safe]
    return xy, depth


def robustPose(points: np.ndarray, positions: np.ndarray, width: int, height: int, focalPixels: float,
               initial: np.ndarray, rounds: int = 3):
    """
    The camera pose putting the 3D `points` at the image `positions`, ignoring the points that do not fit (wrong
    matches, moving objects, occlusions): solved, then the points much farther than the typical error are dropped,
    and solved again. With few points (pins), one wrong point pulls the pose toward it: the worst point is dropped one at
    a time while it is far from the others.

    Returns:
        (camera-to-world matrix, inliers (n,) bool, RMS error of the inliers in pixels).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    inliers = np.ones(len(points), dtype=bool)
    matrix = scene3d.toMatrix4(initial)
    if len(points) == 0:
        return matrix, inliers, 0.0

    def solve(start):
        solved, _ = scene3d.solvePose(points[inliers], positions[inliers], width, height, focalPixels, start, iterations=20)
        xy, depth = projectPoints(solved, points, width, height, focalPixels)
        return solved, np.where(depth > 0, np.hypot(*(xy - positions).T), np.inf)

    def threshold(errors):
        # 3 robust standard deviations, at least 2 pixels (noise of the matches)
        return max(2.0, 3.0 * 1.4826 * float(np.median(errors))) if len(errors) else np.inf

    if len(points) <= 30:
        matrix, errors = solve(matrix)
        # Keep at least 4 points (a determined pose) and two thirds of them
        while inliers.sum() > max(4, (2 * len(points)) // 3):
            candidates = np.flatnonzero(inliers)
            worst = candidates[np.argmax(errors[candidates])]
            others = errors[candidates[candidates != worst]]
            if errors[worst] <= threshold(others):
                break
            inliers[worst] = False
            matrix, errors = solve(matrix)
    else:
        for _ in range(rounds):
            if inliers.sum() < 4:
                break
            matrix, errors = solve(matrix)
            updated = errors < threshold(errors[inliers])
            if np.array_equal(updated, inliers):
                break
            inliers = updated
        else:
            # The last points dropped: solved without them
            if inliers.sum() >= 4:
                matrix, _ = solve(matrix)
    xy, _ = projectPoints(matrix, points, width, height, focalPixels)
    error = float(np.sqrt(np.mean(np.sum((xy[inliers] - positions[inliers]) ** 2, axis=1)))) if inliers.any() else 0.0
    return matrix, inliers, error


class SurfaceSamples:
    """ Points spread on the surface of a geometry (by area), with their normals (None for a point cloud). """

    def __init__(self, geo: scene3d.GeoData, count: int = 20000, seed: int = 0):
        rng = np.random.default_rng(seed)
        triangles = geo.triangles() if geo is not None and not geo.isEmpty else np.zeros((0, 3), dtype=np.int64)
        if not len(triangles):
            vertices = geo.vertices if geo is not None else np.zeros((0, 3))
            if len(vertices) > count:
                vertices = vertices[rng.choice(len(vertices), count, replace=False)]
            self.points, self.normals = np.asarray(vertices, dtype=np.float64), None
            return
        v0, v1, v2 = (geo.vertices[triangles[:, i]] for i in range(3))
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=1)
        valid = areas > 1e-12
        if not valid.any():
            self.points, self.normals = np.asarray(geo.vertices, dtype=np.float64), None
            return
        probabilities = np.where(valid, areas, 0.0) / areas[valid].sum()
        chosen = rng.choice(len(triangles), count, p=probabilities)
        # Uniform barycentric coordinates
        a, b = rng.random(count), rng.random(count)
        flip = a + b > 1.0
        a, b = np.where(flip, 1.0 - a, a), np.where(flip, 1.0 - b, b)
        self.points = v0[chosen] + a[:, np.newaxis] * (v1 - v0)[chosen] + b[:, np.newaxis] * (v2 - v0)[chosen]
        self.normals = cross[chosen] / areas[chosen][:, np.newaxis]

    def __len__(self):
        return len(self.points)

    def visible(self, matrix: np.ndarray, width: int, height: int, focalPixels: float, margin: float = 16.0,
                cell: int = 6):
        """
        The samples seen by a camera: in front of it, inside the image (away from its edges by `margin`), facing it
        (a surface seen from both sides when its faces do not agree), and not hidden by a nearer part of the surface
        (a depth buffer of `cell` pixels).

        Returns:
            (indices of the visible samples, their image positions (n, 2)).
        """
        xy, depth = projectPoints(matrix, self.points, width, height, focalPixels)
        keep = (depth > 1e-9) & (xy[:, 0] >= margin) & (xy[:, 0] < width - margin) & \
               (xy[:, 1] >= margin) & (xy[:, 1] < height - margin)
        if self.normals is not None:
            facing = ((matrix[:3, 3] - self.points) * self.normals).sum(axis=1) > 0
            # An open surface whose normals all look away: seen from behind, it is still the surface seen
            if (facing & keep).sum() >= 0.1 * keep.sum():
                keep &= facing
        indices = np.flatnonzero(keep)
        if not len(indices):
            return indices, np.zeros((0, 2))
        columns = (xy[indices, 0] // cell).astype(np.int64)
        rows = (xy[indices, 1] // cell).astype(np.int64)
        cells = rows * (int(width) // cell + 2) + columns
        nearest = np.full(cells.max() + 1, np.inf)
        np.minimum.at(nearest, cells, depth[indices])
        # The surface nearest to the camera in each cell (a tolerance for the slope inside a cell)
        front = depth[indices] <= nearest[cells] * 1.02 + 1e-9
        indices = indices[front]
        return indices, xy[indices]


def cornerScores(lum: np.ndarray, bbox, positions: np.ndarray) -> np.ndarray:
    """ The Shi-Tomasi corner score of a luminance image (h, w) at image positions (n, 2): high on textured spots. """
    from .camera import _boxSum
    values = np.maximum(lum.astype(np.float64), 0.0)
    values = values / (1.0 + values)
    gradRows, gradColumns = np.gradient(values)
    xx = _boxSum(gradColumns * gradColumns, 2)
    yy = _boxSum(gradRows * gradRows, 2)
    xy = _boxSum(gradColumns * gradRows, 2)
    score = (xx + yy) / 2 - np.sqrt(((xx - yy) / 2) ** 2 + xy ** 2)
    columns = np.clip((positions[:, 0] - bbox.x0).astype(np.int64), 0, lum.shape[1] - 1)
    rows = np.clip((bbox.y1 - 1 - np.floor(positions[:, 1])).astype(np.int64), 0, lum.shape[0] - 1)
    return score[rows, columns]


def selectFeatures(lum: np.ndarray, bbox, positions: np.ndarray, count: int, minDistance: float) -> np.ndarray:
    """
    Up to `count` of the `positions` on textured parts of the image, spread over it (the best one per cell of
    `minDistance` pixels, the best cells first). Returns their indices in `positions`.
    """
    if not len(positions):
        return np.zeros(0, dtype=np.int64)
    scores = cornerScores(lum, bbox, positions)
    positive = scores[scores > 0]
    if not positive.size:
        return np.zeros(0, dtype=np.int64)
    threshold = max(1e-10, 0.02 * float(np.percentile(positive, 90)))
    cell = max(4.0, float(minDistance))
    keys = np.floor(positions[:, 1] / cell).astype(np.int64) * 100000 + np.floor(positions[:, 0] / cell).astype(np.int64)
    order = np.lexsort((-scores, keys))
    first = np.r_[True, keys[order][1:] != keys[order][:-1]]
    best = order[first]
    best = best[scores[best] >= threshold]
    best = best[np.argsort(-scores[best])]
    return best[:count]


class TrackSettings:
    """ How the geometry is tracked. """

    def __init__(self, method: str = "flow", features: int = 150, pattern: int = 15, search: int = 41,
                 minScore: float = 0.7, minInliers: int = 8, anchored: bool = True, device=None):
        self.method = method if method in METHODS else "flow"
        self.features = int(features)
        self.pattern = int(pattern) | 1
        self.search = int(search) | 1
        self.minScore = float(minScore)
        self.minInliers = int(minInliers)
        # Each pose corrected against an anchor frame (less drift, slower)
        self.anchored = bool(anchored)
        # The GPU to compute on (meshroom/live/gpu.py), None: the CPU
        self.device = device


class TrackingLost(RuntimeError):
    """ Not enough points of the geometry could be followed to the next frame. """


def _measureMotion(settings: TrackSettings, lumA: tuple, lumB: tuple, imageA, imageB, positionsA: np.ndarray,
                   centersB: np.ndarray, cancel=None):
    """
    Where the points at `positionsA` in image A are in image B: (positions (n, 2), found (n,) bool). With the optical
    flow, the dense flow of A to B at each point; with patches, the patch of A around each point searched in B around
    its expected position `centersB`.
    """
    if settings.method in ("flow", "raft"):
        from .flow import computeFlow, sampleRows
        field = computeFlow(imageA, imageB, cancel=cancel, useGpu=settings.device is not None,
                            method="raft" if settings.method == "raft" else "lucasKanade")
        bbox = imageA.bbox
        motion = sampleRows(field, (bbox.y1 - positionsA[:, 1] - 0.5).astype(np.float32),
                            (positionsA[:, 0] - bbox.x0 - 0.5).astype(np.float32))
        return positionsA + motion.astype(np.float64), np.ones(len(positionsA), dtype=bool)
    positions = np.array(centersB, dtype=np.float64)
    found = np.zeros(len(positionsA), dtype=bool)
    if cancel:
        cancel()
    patterns = [tracking._patch(lumA[0], lumA[1], x, y, settings.pattern) for x, y in positionsA]
    indices = [index for index, pattern in enumerate(patterns) if pattern is not None]
    if not indices:
        return positions, found
    # All the patterns at once (on the GPU when there is one)
    matches = tracking.matchMany(np.stack([patterns[index] for index in indices]), lumB[0], lumB[1],
                                 np.asarray(centersB, dtype=np.float64)[indices], settings.search, settings.device)
    for index, match in zip(indices, matches):
        if not np.isfinite(match[2]) or match[2] < settings.minScore:
            continue
        positions[index] = tracking.refine(patterns[index], lumB[0], lumB[1], match[0], match[1])
        found[index] = True
    return positions, found


def stepPose(samples: SurfaceSamples, settings: TrackSettings, matrixA: np.ndarray, imageA, imageB,
             predictedB: np.ndarray, width: int, height: int, focalPixels: float, cancel=None,
             report: Optional[dict] = None):
    """
    The pose of frame B from the known pose of frame A: the visible textured samples of A followed into B.
    `report` (a dict) receives the "positions" (k, 2) in frame B (Nuke pixels) of the points followed and whether
    each one fits the pose ("inliers", k booleans).

    Returns:
        (camera-to-world matrix of B, number of points fitting it, their RMS error in pixels). Raises TrackingLost when
        too few points can be followed.
    """
    lumA, lumB = tracking.luminance(imageA), tracking.luminance(imageB)
    indices, positionsA = samples.visible(matrixA, width, height, focalPixels, margin=settings.search / 2.0 + 2)
    keep = ~maskedOut(imageA, positionsA)
    indices, positionsA = indices[keep], positionsA[keep]
    minDistance = max(8.0, np.sqrt(width * height / max(1.0, settings.features * 2.0)))
    chosen = selectFeatures(lumA[0], lumA[1], positionsA, settings.features, minDistance)
    if len(chosen) < settings.minInliers:
        raise TrackingLost(f"only {len(chosen)} textured points of the geometry are visible")
    points = samples.points[indices[chosen]]
    positionsA = positionsA[chosen]
    centersB, _ = projectPoints(predictedB, points, width, height, focalPixels)
    positionsB, found = _measureMotion(settings, lumA, lumB, imageA, imageB, positionsA, centersB, cancel)
    found &= ~maskedOut(imageB, positionsB)
    if found.sum() < settings.minInliers:
        raise TrackingLost(f"only {int(found.sum())} points of the geometry were found in the next frame")
    matrix, inliers, error = robustPose(points[found], positionsB[found], width, height, focalPixels, predictedB)
    if report is not None:
        report["positions"], report["inliers"] = positionsB[found], np.asarray(inliers, dtype=bool)
    if inliers.sum() < settings.minInliers:
        raise TrackingLost(f"only {int(inliers.sum())} points of the geometry move with it")
    return matrix, int(inliers.sum()), error


def _predicted(previous: Optional[np.ndarray], current: np.ndarray) -> np.ndarray:
    """ The next pose of a camera moving at constant speed (the motion from `previous` to `current` applied again). """
    if previous is None:
        return current
    return current @ np.linalg.inv(previous) @ current


def trackPose(samples: SurfaceSamples, settings: TrackSettings, frames: List[int], imageAt: Callable,
              startMatrix: np.ndarray, width: int, height: int, focalPixels: float, cancel=None,
              progress: Optional[Callable[[int, int], None]] = None,
              onPose: Optional[Callable[[int, np.ndarray, int, float], None]] = None) -> Tuple[Dict[int, np.ndarray], str]:
    """
    Track the pose of the camera through `frames` (consecutive, forward or backward) from its pose at the first one.

    Args:
        imageAt: function(frame) -> ImageBuffer.
        cancel: function raising to stop.
        progress: function(index, count) after each frame.
        onPose: function(frame, matrix, inliers, error) for each tracked frame (live display).

    Returns:
        ({frame: camera-to-world matrix} of the tracked frames (the first one excluded), why it stopped before the last
        frame ("" if it reached it)). Cancelling (EvaluationCancelled) stops it with the poses found so far.
    """
    from .evaluator import EvaluationCancelled, EvaluationError
    poses = {}
    if len(frames) < 2:
        return poses, ""
    try:
        imageA = imageAt(frames[0])
    except (EvaluationCancelled, EvaluationError) as e:
        return poses, getattr(e, "message", "") or "cancelled"
    previous, current = None, scene3d.toMatrix4(startMatrix)
    # The frame the poses are anchored to (the first one, then the last one where the anchoring held)
    anchor = (current, imageA)
    for index, frame in enumerate(frames[1:], start=1):
        try:
            if cancel:
                cancel()
            imageB = imageAt(frame)
            matrix, inliers, error = stepPose(samples, settings, current, imageA, imageB, _predicted(previous, current),
                                              width, height, focalPixels, cancel)
            # From frame to frame, the small errors add up (drift): the pose is corrected against the anchor frame
            # (its patches searched close to where the pose puts them); when its look changed too much, this frame
            # becomes the anchor
            try:
                if not settings.anchored:
                    raise TrackingLost("")
                anchored, anchoredInliers, anchoredError = alignPose(samples, settings, anchor[0], anchor[1], matrix,
                                                                     imageB, width, height, focalPixels,
                                                                     searches=(settings.pattern + 6,), cancel=cancel)
                if anchoredInliers >= max(settings.minInliers, inliers // 2) and anchoredError <= max(1.5, 2.0 * error):
                    matrix, inliers, error = anchored, anchoredInliers, anchoredError
                else:
                    anchor = (matrix, imageB)
            except TrackingLost:
                anchor = (matrix, imageB)
        except EvaluationCancelled:
            return poses, "cancelled"
        except EvaluationError as e:
            return poses, f"frame {frame}: {e.message}"
        except TrackingLost as e:
            return poses, f"lost at frame {frame}: {e}"
        poses[frame] = matrix
        if onPose:
            onPose(frame, matrix, inliers, error)
        if progress:
            progress(index, len(frames) - 1)
        previous, current, imageA = current, matrix, imageB
    return poses, ""


def alignPose(samples: SurfaceSamples, settings: TrackSettings, referenceMatrix: np.ndarray, referenceImage,
              matrix: np.ndarray, image, width: int, height: int, focalPixels: float, searches=(81, 41, 21),
              cancel=None):
    """
    "Find the right position": the pose of a frame (from `matrix`, roughly right) making the geometry fit its image
    as it fits the reference image with the reference pose. The textured points of the geometry in the reference image
    are searched in the image (their patches) around where the current pose puts them, the pose solved from them, and
    again with smaller search areas.

    Returns:
        (camera-to-world matrix, number of points fitting it, their RMS error in pixels). Raises TrackingLost when too
        few points are found.
    """
    lumReference, lumImage = tracking.luminance(referenceImage), tracking.luminance(image)
    indices, positions = samples.visible(referenceMatrix, width, height, focalPixels, margin=settings.pattern)
    keep = ~maskedOut(referenceImage, positions)
    indices, positions = indices[keep], positions[keep]
    minDistance = max(8.0, np.sqrt(width * height / max(1.0, settings.features * 2.0)))
    chosen = selectFeatures(lumReference[0], lumReference[1], positions, settings.features, minDistance)
    if len(chosen) < settings.minInliers:
        raise TrackingLost(f"only {len(chosen)} textured points of the geometry are visible at the reference frame")
    points, referencePositions = samples.points[indices[chosen]], positions[chosen]
    patchSettings = TrackSettings("patches", settings.features, settings.pattern, settings.search, settings.minScore,
                                  settings.minInliers, device=settings.device)
    current, inliers, error = scene3d.toMatrix4(matrix), 0, 0.0
    for search in searches:
        patchSettings.search = int(search) | 1
        centers, _ = projectPoints(current, points, width, height, focalPixels)
        found, ok = _measureMotion(patchSettings, lumReference, lumImage, None, None, referencePositions, centers, cancel)
        ok &= ~maskedOut(image, found)
        if ok.sum() < settings.minInliers:
            if inliers:
                break
            raise TrackingLost(f"only {int(ok.sum())} points of the geometry were found in the image")
        current, fitting, error = robustPose(points[ok], found[ok], width, height, focalPixels, current)
        inliers = int(fitting.sum())
    if inliers < settings.minInliers:
        raise TrackingLost(f"only {inliers} points of the geometry fit")
    return current, inliers, error


def alignWithMatches(samples: SurfaceSamples, referenceMatrix: np.ndarray, referenceImage, matrix: np.ndarray, image,
                     width: int, height: int, focalPixels: float, matcher, minScore: float = 0.3, minInliers: int = 8):
    """
    The pose of a frame from feature matches with a reference frame whose pose is known (LightGlue,
    meshroom/live/neural/matching.py): the matched points of the reference are put on the geometry (the visible sample
    of the surface seen there, within 3 pixels), and the pose solved from them. Works far from the right position and
    from another angle, where alignPose needs a rough pose.

    Returns:
        (camera-to-world matrix, number of points fitting it, their RMS error in pixels). Raises TrackingLost when too
        few matches land on the geometry.
    """
    pointsReference, pointsImage, scores = matcher.match(referenceImage, image)
    keep = (scores >= minScore) & ~maskedOut(referenceImage, pointsReference) & ~maskedOut(image, pointsImage)
    pointsReference, pointsImage = pointsReference[keep], pointsImage[keep]
    indices, positions = samples.visible(referenceMatrix, width, height, focalPixels, margin=0.0)
    if not len(indices) or not len(pointsReference):
        raise TrackingLost("no match on the geometry")
    # The nearest visible sample of each matched point, through a grid of 3 pixel cells
    cell = 3.0
    grid = {}
    for k, (x, y) in enumerate(positions):
        grid.setdefault((int(x // cell), int(y // cell)), []).append(k)
    points, found = [], []
    for (x, y), target in zip(pointsReference, pointsImage):
        cx, cy = int(x // cell), int(y // cell)
        near = [k for dx in (-1, 0, 1) for dy in (-1, 0, 1) for k in grid.get((cx + dx, cy + dy), ())]
        if not near:
            continue
        distances = np.hypot(positions[near, 0] - x, positions[near, 1] - y)
        best = int(np.argmin(distances))
        if distances[best] <= cell:
            points.append(samples.points[indices[near[best]]])
            found.append(target)
    if len(points) < minInliers:
        raise TrackingLost(f"only {len(points)} matches land on the geometry")
    points, found = np.array(points), np.array(found)
    # From the rough pose, or the reference one when it explains the matches better
    starts = []
    for start in (scene3d.toMatrix4(matrix), scene3d.toMatrix4(referenceMatrix)):
        xy, depth = projectPoints(start, points, width, height, focalPixels)
        starts.append((float(np.median(np.where(depth > 0, np.hypot(*(xy - found).T), np.inf))), start))
    start = min(starts, key=lambda item: item[0])[1]
    pose, inliers, error = robustPose(points, found, width, height, focalPixels, start)
    if inliers.sum() < minInliers:
        raise TrackingLost(f"only {int(inliers.sum())} matches fit one pose")
    return pose, int(inliers.sum()), error


def _luminance(image, bbox):
    """ The luminance (h, w) of an image over `bbox`, compressed like the display (bright areas do not hide the rest). """
    from .ops import alignToBBox
    rgba = alignToBBox(image.rgba(), image.bbox, bbox)
    lum = 0.2126 * rgba[:, :, 0] + 0.7152 * rgba[:, :, 1] + 0.0722 * rgba[:, :, 2]
    return np.power(np.clip(lum.astype(np.float64), 0.0, None), 1.0 / 2.2)


def _silhouetteSamples(geo, matrix: np.ndarray, width: int, height: int, focalPixels: float, bbox, creaseDegrees: float,
                       spacing: float, maxSamples: int):
    """
    Points along the visible outline of a geometry seen by a camera (its silhouettes and open edges, and its creases
    sharper than `creaseDegrees` when it is below 180), every `spacing` pixels, the hidden parts removed.

    Returns:
        (points (k, 3), positions (k, 2) in the buffer over `bbox` (u: columns, v: rows from the top), normals (k, 2):
        the unit direction across the edge in the image, pointing out of the geometry for a silhouette).
    """
    from . import geoRender
    empty = np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 2))
    camera = scene3d.CameraData(matrix, focalLength=focalPixels / float(width) * 36.0, horizontalAperture=36.0)
    shape = (bbox.height, bbox.width)
    u, v, z, inFront = geoRender.screenPositions(geo, camera, width, height, bbox)
    zbuffer, _ = geoRender.rasterize(u, v, z, geo.triangles(), shape)
    edges = geoRender.featureEdges(geo, camera.position, min(float(creaseDegrees), 180.0))
    if len(edges):
        edges = edges[inFront[edges[:, 0]] & inFront[edges[:, 1]]]
    if not len(edges):
        return empty
    a, b = edges[:, 0], edges[:, 1]
    lengths = np.hypot(u[b] - u[a], v[b] - v[a])
    counts = np.minimum(np.maximum(1, np.ceil(lengths / spacing).astype(np.int64)), 4 * max(shape))
    segment = np.repeat(np.arange(len(a)), counts)
    firsts = np.r_[0, np.cumsum(counts)[:-1]]
    t = ((np.arange(len(segment)) - firsts[segment]) + 0.5) / counts[segment]
    points = geo.vertices[a][segment] * (1 - t)[:, None] + geo.vertices[b][segment] * t[:, None]
    # Projected exactly (perspective), in the buffer
    xy, depth = projectPoints(matrix, points, width, height, focalPixels)
    su, sv = xy[:, 0] - bbox.x0, bbox.y1 - xy[:, 1]
    direction = np.c_[(u[b] - u[a])[segment], (v[b] - v[a])[segment]]
    normals = np.c_[-direction[:, 1], direction[:, 0]] / np.maximum(lengths[segment], 1e-9)[:, None]
    h, w = shape
    inside = (su >= 0) & (su < w) & (sv >= 0) & (sv < h) & (depth > 0) & (lengths[segment] > 1e-6)
    columns = np.clip(np.floor(su).astype(np.int64), 0, w - 1)
    rows = np.clip(np.floor(sv).astype(np.int64), 0, h - 1)
    visible = inside & (depth <= zbuffer[rows, columns] * 1.004 + 1e-9)
    # The sides of the edge: the normal is turned toward the side without geometry (a silhouette); the samples with
    # geometry on both sides (creases) keep their normal
    covered = np.isfinite(zbuffer)

    def coveredAt(offset):
        r = np.clip(np.floor(sv + offset * normals[:, 1]).astype(np.int64), 0, h - 1)
        c = np.clip(np.floor(su + offset * normals[:, 0]).astype(np.int64), 0, w - 1)
        return covered[r, c]
    flip = coveredAt(2.0) & ~coveredAt(-2.0)
    normals[flip] *= -1
    chosen = np.flatnonzero(visible)
    if len(chosen) > maxSamples:
        chosen = chosen[np.linspace(0, len(chosen) - 1, maxSamples).astype(np.int64)]
    return points[chosen], np.c_[su, sv][chosen], normals[chosen]


def _edgeSearch(gradients, positions: np.ndarray, normals: np.ndarray, reach: float):
    """
    The edge of the image across each sample: the strongest gradient along its normal within `reach` pixels
    (sub-pixel), as in RAPiD / moving edges trackers. `gradients`: (d/drow, d/dcolumn) of the luminance.

    Returns:
        (offsets (k,) along the normals in pixels, strengths (k,): the gradient across the edge found).
    """
    from scipy import ndimage
    gradRows, gradColumns = gradients
    steps = np.arange(-np.ceil(reach), np.ceil(reach) + 1.0)
    u = positions[:, 0][:, None] + steps[None, :] * normals[:, 0][:, None]
    v = positions[:, 1][:, None] + steps[None, :] * normals[:, 1][:, None]
    # Pixel (r, c) is centered on (c + 0.5, r + 0.5)
    coordinates = np.stack([v.ravel() - 0.5, u.ravel() - 0.5])
    gr = ndimage.map_coordinates(gradRows, coordinates, order=1, mode="nearest").reshape(u.shape)
    gc = ndimage.map_coordinates(gradColumns, coordinates, order=1, mode="nearest").reshape(u.shape)
    across = np.abs(gc * normals[:, 0][:, None] + gr * normals[:, 1][:, None])
    # A little preference for the nearest edges (between two edges as strong, the nearest)
    score = across * (1.0 - 0.25 * np.abs(steps)[None, :] / max(reach, 1.0))
    best = np.argmax(score, axis=1)
    k = np.arange(len(best))
    offsets = steps[best].astype(np.float64)
    # Sub-pixel: the parabola through the scores around the best
    inner = (best > 0) & (best < len(steps) - 1)
    left, middle, right = score[k[inner], best[inner] - 1], score[k[inner], best[inner]], score[k[inner], best[inner] + 1]
    curvature = left - 2 * middle + right
    offsets[inner] += np.where(curvature < -1e-12, 0.5 * (left - right) / np.where(curvature < -1e-12, curvature, -1.0), 0.0)
    return offsets, across[k, best]


def _solveAlongNormals(points: np.ndarray, targets: np.ndarray, normals: np.ndarray, width: int, height: int,
                       focalPixels: float, bbox, initial: np.ndarray, iterations: int = 10):
    """
    The camera pose putting the 3D `points` on the image edges through `targets` across `normals` (buffer over `bbox`):
    only the distance along the normal counts (an edge does not tell where along it a point is). Robust (Tukey weights:
    the points far from the others' fit are ignored), Levenberg-Marquardt.

    Returns:
        (camera-to-world matrix, inliers (k,) bool, RMS distance of the inliers in pixels).
    """
    from .camera import rodrigues
    rotation0, translation0 = scene3d.cvPose(scene3d.toMatrix4(initial))
    # The targets in normalized camera coordinates (x right, y down)
    tx = (targets[:, 0] + bbox.x0 - width / 2.0) / focalPixels
    ty = (targets[:, 1] - bbox.y1 + height / 2.0) / focalPixels
    nx, ny = normals[:, 0], normals[:, 1]

    def distances(params):
        camera = points @ (rodrigues(params[:3]) @ rotation0).T + translation0 + params[3:]
        z = np.where(camera[:, 2] < 1e-9, 1e-9, camera[:, 2])
        return focalPixels * (nx * (camera[:, 0] / z - tx) + ny * (camera[:, 1] / z - ty))

    params = np.zeros(6)
    weights = np.ones(len(points))
    for _ in range(iterations):
        current = distances(params)
        # Tukey weights, 4.7 robust standard deviations (at least 1 pixel)
        spread = max(1.0, 4.685 * 1.4826 * float(np.median(np.abs(current))))
        weights = np.where(np.abs(current) < spread, (1 - (current / spread) ** 2) ** 2, 0.0)
        if weights.sum() < 6:
            break
        jacobian = np.empty((len(points), 6))
        for k in range(6):
            step = np.zeros(6)
            step[k] = 1e-6
            jacobian[:, k] = (distances(params + step) - current) / 1e-6
        normal = jacobian.T @ (jacobian * weights[:, None])
        gradient = jacobian.T @ (weights * current)
        cost = np.sum(weights * current ** 2)
        delta = None
        # Levenberg-Marquardt: the step shortened until it lowers the cost
        for damping in (1e-6, 1e-3, 1e-1, 1.0, 10.0, 100.0):
            try:
                step = -np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-9), gradient)
            except np.linalg.LinAlgError:
                continue
            if np.sum(weights * distances(params + step) ** 2) < cost:
                delta = step
                break
        if delta is None:
            break
        params = params + delta
        if np.linalg.norm(delta) < 1e-9:
            break
    final = distances(params)
    inliers = weights > 0
    error = float(np.sqrt(np.mean(final[inliers] ** 2))) if inliers.any() else 0.0
    return scene3d.cameraMatrixFromCvPose(rodrigues(params[:3]) @ rotation0, translation0 + params[3:]), inliers, error


def alignToSilhouette(geo, matrix: np.ndarray, image, width: int, height: int, focalPixels: float,
                      maxDistance: float = 40.0, creaseDegrees: float = 180.0, iterations: int = 6, cancel=None):
    """
    "Find the right position" without keyframe nor network: the outline of the geometry (its silhouettes, seen from
    the current pose) snapped onto the edges of the image. Each point along the outline searches the strongest edge
    of the image across the outline (along its normal, as RAPiD / moving edges trackers do: the texture inside the
    object, along the outline, does not attract it), and the pose is solved from these matches (robustly), again and
    again; coarse to fine: on the image blurred (the edges found from far, the details ignored), then sharper, with a
    shorter search. Needs a rough pose (the geometry roughly on its place, within `maxDistance` pixels) and an object
    whose outline stands out of its background. `creaseDegrees` < 180 also snaps the creases of the geometry.

    Returns:
        (camera-to-world matrix, number of points fitting it, their RMS error in pixels). Raises TrackingLost when the
        geometry shows no outline in the image or no edge of the image is near it.
    """
    from scipy import ndimage
    bbox = image.bbox
    lum = _luminance(image, bbox)
    start = current = scene3d.toMatrix4(matrix)
    inliers, error = 0, 0.0
    # Coarse to fine: blur (pixels) and reach of the search
    levels = []
    sigma, reach = max(1.0, maxDistance / 8.0), float(maxDistance)
    while True:
        levels.append((sigma, reach))
        if sigma <= 1.0:
            break
        sigma, reach = max(1.0, sigma / 2.0), max(3.0, reach / 2.0)
    for sigma, reach in levels:
        blurred = ndimage.gaussian_filter(lum, sigma)
        gradients = np.gradient(blurred)
        for _ in range(iterations):
            if cancel:
                cancel()
            points, positions, normals = _silhouetteSamples(geo, current, width, height, focalPixels, bbox,
                                                             creaseDegrees, 2.0, 2000)
            if len(points) < 8:
                raise TrackingLost("the geometry shows no outline in the image: place it roughly first")
            offsets, strengths = _edgeSearch(gradients, positions, normals, reach)
            # The weak edges (flat areas: nothing found) are not matches, nor the masked out ones (an occluding object)
            strong = strengths > 0.3 * float(np.percentile(strengths, 90))
            found = positions + offsets[:, None] * normals
            strong &= ~maskedOut(image, np.c_[found[:, 0] + bbox.x0, bbox.y1 - found[:, 1]])
            strong &= ~maskedOut(image, np.c_[positions[:, 0] + bbox.x0, bbox.y1 - positions[:, 1]])
            if strong.sum() < 8:
                raise TrackingLost("no edge of the image near the outline of the geometry: place it roughly first")
            targets = positions[strong] + offsets[strong, None] * normals[strong]
            previous = current
            current, fitting, error = _solveAlongNormals(points[strong], targets, normals[strong], width, height,
                                                         focalPixels, bbox, current)
            inliers = int(fitting.sum())
            moved, _ = projectPoints(current, points, width, height, focalPixels)
            before, _ = projectPoints(previous, points, width, height, focalPixels)
            if float(np.abs(moved - before).max()) < 0.05:
                break
    # Snapped onto edges much farther than searched: another outline (a wrong solution), not a correction
    outline, _, _ = _silhouetteSamples(geo, start, width, height, focalPixels, bbox, creaseDegrees, 4.0, 500)
    if len(outline):
        a, _ = projectPoints(start, outline, width, height, focalPixels)
        b, _ = projectPoints(current, outline, width, height, focalPixels)
        if float(np.median(np.hypot(*(b - a).T))) > 2.0 * maxDistance:
            raise TrackingLost("the outline snapped far from where the geometry was: place it closer first")
    return current, inliers, error


def blendPoses(forward: Dict[int, np.ndarray], backward: Dict[int, np.ndarray], first: int, last: int) -> Dict[int, np.ndarray]:
    """
    The poses between two keyframes from a track forward from `first` and a track backward from `last`: weighted
    toward the nearest keyframe (its track drifted the least). A frame tracked one way only keeps that pose.
    """
    result = {}
    for frame in range(first + 1, last):
        a, b = forward.get(frame), backward.get(frame)
        if a is not None and b is not None:
            result[frame] = scene3d.interpolateMatrices(a, b, (frame - first) / float(last - first))
        elif a is not None or b is not None:
            result[frame] = a if a is not None else b
    return result


# Stored poses of a Resection ("geoTrack" parameter): {"poses": {"frame": 16 numbers}, "keyframes": [frames]}

def parsePoses(text: str) -> Tuple[Dict[int, np.ndarray], List[int]]:
    """ The poses ({frame: camera-to-world matrix}) and the keyframes stored in a JSON string. """
    import json
    if not text or not text.strip():
        return {}, []
    try:
        data = json.loads(text)
        poses = {int(frame): np.array(values, dtype=np.float64).reshape(4, 4)
                 for frame, values in (data.get("poses") or {}).items()}
        keyframes = sorted({int(frame) for frame in data.get("keyframes") or []})
    except (TypeError, ValueError, AttributeError) as e:
        raise ValueError(f"Invalid geometry track: {e}") from e
    return poses, keyframes


def dumpPoses(poses: Dict[int, np.ndarray], keyframes) -> str:
    """ The JSON string of poses and keyframes ("" when empty). """
    import json
    if not poses and not keyframes:
        return ""
    return json.dumps({"poses": {str(frame): [round(float(v), 9) for v in np.asarray(matrix).ravel()]
                                 for frame, matrix in sorted(poses.items())},
                       "keyframes": sorted({int(frame) for frame in keyframes})})


def deletePoses(text: str, frame: int, mode: str) -> Tuple[str, int]:
    """
    tracking.deleteKeys for the poses tracked with "Track Geometry" (JSON of parsePoses): forward / backward of `frame`
    (the pose at `frame` kept), or the tracked ones (the keyframes kept). Returns (the new JSON, the number removed).
    """
    poses, keyframes = parsePoses(text)
    if mode == "forward":
        removed = [f for f in poses if f > frame]
        keyframes = [f for f in keyframes if f <= frame]
    elif mode == "backward":
        removed = [f for f in poses if f < frame]
        keyframes = [f for f in keyframes if f >= frame]
    else:
        removed = [f for f in poses if f not in keyframes]
    for f in removed:
        del poses[f]
    return dumpPoses(poses, keyframes), len(removed)


# Deformation of the geometry by the optical flow ("Deform With Flow" of a Resection): control vertices of the geometry
# follow the flow of the plate, frame after frame; their offsets (pixels, from where the rigid pose puts them) move every
# vertex in the image, at its depth. Stored as {"indices": [control vertices], "radius": r, "frames": {"f": [dx, dy...]}}

def deformControls(geo, count: int = 400) -> np.ndarray:
    """ About `count` vertices spread over a geometry (the first one of each cell of a grid): the control vertices. """
    vertices = np.asarray(geo.vertices, dtype=np.float64)
    if len(vertices) <= count:
        return np.arange(len(vertices))
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    extent = np.maximum(high - low, 1e-9)
    # Cells of a grid sized for about `count` occupied cells (a surface fills about the square of the cells per axis)
    cell = float(np.sqrt(np.prod(np.sort(extent)[1:]) / count)) if np.sort(extent)[1] > 1e-6 else float(extent.max() / count)
    for _ in range(12):
        keys = np.floor((vertices - low) / max(cell, 1e-12)).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        if len(first) > 1.5 * count:
            cell *= 1.3
        elif len(first) < 0.6 * count:
            cell /= 1.3
        else:
            break
    return np.sort(first)


def deformRadius(geo, indices: np.ndarray) -> float:
    """ The reach of each control vertex: twice the typical distance between neighboring controls. """
    from scipy.spatial import cKDTree
    points = np.asarray(geo.vertices, dtype=np.float64)[indices]
    if len(points) < 2:
        return 1.0
    distances, _ = cKDTree(points).query(points, k=2)
    return max(2.0 * float(np.median(distances[:, 1])), 1e-6)


class DeformWeights:
    """ How each vertex follows the control vertices: gaussian weights of its nearest controls (in the rest shape). """

    def __init__(self, geo, indices: np.ndarray, radius: float, neighbors: int = 8):
        from scipy.spatial import cKDTree
        vertices = np.asarray(geo.vertices, dtype=np.float64)
        controls = vertices[np.asarray(indices, dtype=np.int64)]
        k = min(neighbors, len(controls))
        distances, self.nearest = cKDTree(controls).query(vertices, k=k)
        distances, self.nearest = distances.reshape(len(vertices), k), self.nearest.reshape(len(vertices), k)
        weights = np.exp(-(distances / max(radius, 1e-9)) ** 2)
        # Normalized (a vertex far from every control follows the nearest ones)
        weights[:, 0] = np.maximum(weights[:, 0], 1e-12)
        self.weights = weights / weights.sum(axis=1, keepdims=True)

    def shifts(self, offsets: np.ndarray) -> np.ndarray:
        """ The shift (n, 2) of every vertex for the offsets (k, 2) of the controls. """
        return (np.asarray(offsets, dtype=np.float64)[self.nearest] * self.weights[:, :, None]).sum(axis=1)


def deformVertices(vertices: np.ndarray, matrix: np.ndarray, width: int, height: int, focalPixels: float,
                   shifts: np.ndarray) -> np.ndarray:
    """ World vertices moved in the image of the camera (camera-to-world `matrix`) by `shifts` (n, 2) pixels, each
    at its depth. """
    view = np.linalg.inv(matrix)
    inCamera = vertices @ view[:3, :3].T + view[:3, 3]
    depth = -inCamera[:, 2]
    front = depth > 1e-9
    xy = np.c_[focalPixels * inCamera[:, 0] / np.where(front, depth, 1.0),
               focalPixels * inCamera[:, 1] / np.where(front, depth, 1.0)] + shifts
    moved = inCamera.copy()
    moved[front] = np.c_[xy[front] / focalPixels, -np.ones(int(front.sum()))] * depth[front, None]
    return moved @ matrix[:3, :3].T + matrix[:3, 3]


def parseDeform(text: str) -> dict:
    """ {"indices": array, "radius": float, "frames": {frame: offsets (k, 2)}} of a stored deformation ({} if none). """
    import json
    if not text or not text.strip():
        return {}
    try:
        data = json.loads(text)
        indices = np.array(data["indices"], dtype=np.int64)
        frames = {int(frame): np.array(values, dtype=np.float64).reshape(-1, 2)
                  for frame, values in (data.get("frames") or {}).items()}
        return {"indices": indices, "radius": float(data.get("radius", 1.0)), "frames": frames}
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        raise ValueError(f"Invalid geometry deformation: {e}") from e


def dumpDeform(deform: dict) -> str:
    import json
    if not deform or not deform.get("frames"):
        return ""
    return json.dumps({"indices": [int(i) for i in deform["indices"]], "radius": round(float(deform["radius"]), 9),
                       "frames": {str(frame): [round(float(v), 2) for v in np.asarray(offsets).ravel()]
                                  for frame, offsets in sorted(deform["frames"].items())}})


def deformOffsetsAt(deform: dict, frame: int) -> Optional[np.ndarray]:
    """ The offsets of the controls at a frame: its own, else held from the nearest deformed frame; None without. """
    frames = deform.get("frames") or {}
    if not frames:
        return None
    if frame in frames:
        return frames[frame]
    return frames[min(frames, key=lambda f: (abs(f - frame), f))]


def trackDeformation(geo, settings: TrackSettings, frames: List[int], imageAt: Callable, poseAt: Callable,
                     width: int, height: int, focalPixels: float, deform: dict, cancel=None,
                     progress: Optional[Callable[[int, int], None]] = None,
                     onFrame: Optional[Callable[[int, np.ndarray], None]] = None) -> Tuple[dict, str]:
    """
    Deform the geometry with the optical flow of the plate through `frames` (consecutive, forward or backward), from
    its deformation at the first one (none: the rigid geometry). Each visible control vertex (not hidden by the
    geometry nor masked out) moves with the flow (Lucas-Kanade or RAFT: `settings.method`); a hidden one keeps its
    offset. `poseAt(frame)`: the camera-to-world matrix (the rigid pose) at a frame.

    Returns:
        (the deformation {"indices", "radius", "frames"} with the new frames, why it stopped ("" at the end)).
    """
    from . import geoRender
    from .evaluator import EvaluationCancelled, EvaluationError
    from .flow import computeFlow, sampleRows
    if not deform:
        indices = deformControls(geo)
        deform = {"indices": indices, "radius": deformRadius(geo, indices), "frames": {}}
    indices = deform["indices"]
    controls = np.asarray(geo.vertices, dtype=np.float64)[indices]
    weights = DeformWeights(geo, indices, deform["radius"])
    offsets = deformOffsetsAt(deform, frames[0]) if frames[0] in deform["frames"] else np.zeros((len(indices), 2))
    deform["frames"][frames[0]] = offsets
    method = "raft" if settings.method == "raft" else "lucasKanade"
    try:
        imageA = imageAt(frames[0])
    except (EvaluationCancelled, EvaluationError) as e:
        return deform, getattr(e, "message", "") or "cancelled"
    for index, frame in enumerate(frames[1:], start=1):
        try:
            if cancel:
                cancel()
            imageB = imageAt(frame)
            matrixA, matrixB = poseAt(frames[index - 1]), poseAt(frame)
            rigidA, depthA = projectPoints(matrixA, controls, width, height, focalPixels)
            positionsA = rigidA + offsets
            # Visible at A: in front, in the image, not hidden by the (deformed) geometry, not masked out
            vertices = deformVertices(geo.vertices, matrixA, width, height, focalPixels, weights.shifts(offsets))
            camera = scene3d.CameraData(matrixA, focalLength=focalPixels / float(width) * 36.0, horizontalAperture=36.0)
            bbox = imageA.bbox
            u, v, z, _ = geoRender.screenPositions(scene3d.GeoData(vertices, topology=geo.topology), camera, width,
                                                   height, bbox)
            zbuffer, _ = geoRender.rasterize(u, v, z, geo.triangles(), (bbox.height, bbox.width))
            columns, rows = np.floor(u[indices]).astype(np.int64), np.floor(v[indices]).astype(np.int64)
            inside = (columns >= 0) & (columns < bbox.width) & (rows >= 0) & (rows < bbox.height) & (depthA > 0)
            visible = inside.copy()
            visible[inside] = z[indices][inside] <= zbuffer[rows[inside], columns[inside]] * 1.01 + 1e-9
            visible &= ~maskedOut(imageA, positionsA)
            field = computeFlow(imageA, imageB, cancel=cancel, useGpu=settings.device is not None, method=method)
            motion = sampleRows(field, (bbox.y1 - positionsA[:, 1] - 0.5).astype(np.float32),
                                (positionsA[:, 0] - bbox.x0 - 0.5).astype(np.float32)).astype(np.float64)
            rigidB, _ = projectPoints(matrixB, controls, width, height, focalPixels)
            # Moved with the flow (visible), else kept where the rigid motion and their offset put them
            updated = offsets.copy()
            updated[visible] = positionsA[visible] + motion[visible] - rigidB[visible]
            visibleB = ~maskedOut(imageB, rigidB + updated)
            updated[~visibleB] = offsets[~visibleB]
            offsets = updated
        except EvaluationCancelled:
            return deform, "cancelled"
        except EvaluationError as e:
            return deform, f"frame {frame}: {e.message}"
        deform["frames"][frame] = offsets
        if onFrame:
            onFrame(frame, offsets)
        if progress:
            progress(index, len(frames) - 1)
        imageA = imageB
    return deform, ""


def deleteDeform(text: str, frame: int, mode: str) -> Tuple[str, int]:
    """ The deformation (dumpDeform JSON) without its frames after ("forward") or before ("backward") `frame`, and the
    number removed. """
    deform = parseDeform(text)
    if not deform:
        return text, 0
    removed = [f for f in deform["frames"] if (f > frame if mode == "forward" else f < frame)]
    for f in removed:
        del deform["frames"][f]
    return dumpDeform(deform), len(removed)
