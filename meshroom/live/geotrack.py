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

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import scene3d, tracking

METHODS = ("flow", "patches")


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
                 minScore: float = 0.7, minInliers: int = 8, anchored: bool = True):
        self.method = method if method in METHODS else "flow"
        self.features = int(features)
        self.pattern = int(pattern) | 1
        self.search = int(search) | 1
        self.minScore = float(minScore)
        self.minInliers = int(minInliers)
        # Each pose corrected against an anchor frame (less drift, slower)
        self.anchored = bool(anchored)


class TrackingLost(RuntimeError):
    """ Not enough points of the geometry could be followed to the next frame. """


def _measureMotion(settings: TrackSettings, lumA: tuple, lumB: tuple, imageA, imageB, positionsA: np.ndarray,
                   centersB: np.ndarray, cancel=None):
    """
    Where the points at `positionsA` in image A are in image B: (positions (n, 2), found (n,) bool). With the optical
    flow, the dense flow of A to B at each point; with patches, the patch of A around each point searched in B around
    its expected position `centersB`.
    """
    if settings.method == "flow":
        from .flow import computeFlow, sampleRows
        field = computeFlow(imageA, imageB, cancel=cancel)
        bbox = imageA.bbox
        motion = sampleRows(field, (bbox.y1 - positionsA[:, 1] - 0.5).astype(np.float32),
                            (positionsA[:, 0] - bbox.x0 - 0.5).astype(np.float32))
        return positionsA + motion.astype(np.float64), np.ones(len(positionsA), dtype=bool)
    positions = np.array(centersB, dtype=np.float64)
    found = np.zeros(len(positionsA), dtype=bool)
    for index, ((x, y), (cx, cy)) in enumerate(zip(positionsA, centersB)):
        if cancel and index % 32 == 0:
            cancel()
        pattern = tracking._patch(lumA[0], lumA[1], x, y, settings.pattern)
        if pattern is None:
            continue
        match = tracking.matchPattern(pattern, lumB[0], lumB[1], cx, cy, settings.search)
        if match is None or match[2] < settings.minScore:
            continue
        positions[index] = tracking.refine(pattern, lumB[0], lumB[1], match[0], match[1])
        found[index] = True
    return positions, found


def stepPose(samples: SurfaceSamples, settings: TrackSettings, matrixA: np.ndarray, imageA, imageB,
             predictedB: np.ndarray, width: int, height: int, focalPixels: float, cancel=None):
    """
    The pose of frame B from the known pose of frame A: the visible textured samples of A followed into B.

    Returns:
        (camera-to-world matrix of B, number of points fitting it, their RMS error in pixels). Raises TrackingLost when
        too few points can be followed.
    """
    lumA, lumB = tracking.luminance(imageA), tracking.luminance(imageB)
    indices, positionsA = samples.visible(matrixA, width, height, focalPixels, margin=settings.search / 2.0 + 2)
    minDistance = max(8.0, np.sqrt(width * height / max(1.0, settings.features * 2.0)))
    chosen = selectFeatures(lumA[0], lumA[1], positionsA, settings.features, minDistance)
    if len(chosen) < settings.minInliers:
        raise TrackingLost(f"only {len(chosen)} textured points of the geometry are visible")
    points = samples.points[indices[chosen]]
    positionsA = positionsA[chosen]
    centersB, _ = projectPoints(predictedB, points, width, height, focalPixels)
    positionsB, found = _measureMotion(settings, lumA, lumB, imageA, imageB, positionsA, centersB, cancel)
    if found.sum() < settings.minInliers:
        raise TrackingLost(f"only {int(found.sum())} points of the geometry were found in the next frame")
    matrix, inliers, error = robustPose(points[found], positionsB[found], width, height, focalPixels, predictedB)
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
    minDistance = max(8.0, np.sqrt(width * height / max(1.0, settings.features * 2.0)))
    chosen = selectFeatures(lumReference[0], lumReference[1], positions, settings.features, minDistance)
    if len(chosen) < settings.minInliers:
        raise TrackingLost(f"only {len(chosen)} textured points of the geometry are visible at the reference frame")
    points, referencePositions = samples.points[indices[chosen]], positions[chosen]
    patchSettings = TrackSettings("patches", settings.features, settings.pattern, settings.search, settings.minScore,
                                  settings.minInliers)
    current, inliers, error = scene3d.toMatrix4(matrix), 0, 0.0
    for search in searches:
        patchSettings.search = int(search) | 1
        centers, _ = projectPoints(current, points, width, height, focalPixels)
        found, ok = _measureMotion(patchSettings, lumReference, lumImage, None, None, referencePositions, centers, cancel)
        if ok.sum() < settings.minInliers:
            if inliers:
                break
            raise TrackingLost(f"only {int(ok.sum())} points of the geometry were found in the image")
        current, fitting, error = robustPose(points[ok], found[ok], width, height, focalPixels, current)
        inliers = int(fitting.sum())
    if inliers < settings.minInliers:
        raise TrackingLost(f"only {inliers} points of the geometry fit")
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
