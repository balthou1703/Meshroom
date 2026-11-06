"""
"Solve" of the FaceTracker: the head poses of a range of frames optimized together (a layer over the tracking, which
it keeps). The keyframes are not truths but soft constraints, each as strong as the confidence given to it: from the
poses tracked, the most coherent sequence is found, a keyframe less sure than the others being moved to agree with
them. What ties the frames together: the moves of the landmarks from one frame to the next (relative: they do not
drift as their absolute places do), their places (weakly), the rigid pins of the keyframes, and a smoothness of the
pose (its accelerations).
"""
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve as sparseSolve


def _rotation(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def _moved(head: np.ndarray, theta: np.ndarray, center: np.ndarray) -> np.ndarray:
    """ The head matrix (face space to world) with a move of the face (rotation vector theta[:3] around `center`, then
    translation theta[3:], in face space) applied before it. """
    move = np.eye(4)
    rotation = _rotation(theta[:3])
    move[:3, :3] = rotation
    move[:3, 3] = center - rotation @ center + theta[3:6]
    return head @ move


def _project(camera, head: np.ndarray, points: np.ndarray, width: int, height: int) -> np.ndarray:
    world = points @ head[:3, :3].T + head[:3, 3]
    xy, _, _ = camera.project(world, width, height)
    return xy


def confidenceWeight(confidence: float) -> float:
    """ The weight of the pins of a keyframe from its confidence in [0, 1]: 1 at 0.5, 0 at 0, ~100 at 1 (held). """
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return confidence / (1.0 - confidence + 0.01)


def solvePoses(frames, cameras: Dict[int, object], landmarks: Dict[int, np.ndarray], hidden: Dict[int, set],
               heads: Dict[int, np.ndarray], rigidPoints: np.ndarray, rigidIndices, width: int, height: int,
               keys: Optional[Dict[int, Tuple[np.ndarray, np.ndarray, float]]] = None, smoothness: float = 1.0,
               relativeWeight: float = 1.0, absoluteWeight: float = 0.5, keyWeight: float = 1.0, huber: float = 3.0,
               iterations: int = 8, cancel=None, progress=None) -> dict:
    """
    The head poses of `frames` (consecutive) solved together.

    Args:
        cameras, landmarks (n, 2 Nuke pixels), hidden (landmark indices), heads (the poses tracked: face space to
            world, where the solve starts): per frame.
        rigidPoints: (k, 3) the points of the rigid landmarks `rigidIndices` on the face (face space).
        keys: {keyframe: (rigid pin points (m, 3) face space, their positions (m, 2) Nuke pixels, confidence in [0, 1])}.
        smoothness: the weight of the accelerations of the pose (1: one pixel of acceleration of the face against one
            pixel of error of the landmarks); relativeWeight / absoluteWeight: the moves of the landmarks from frame to
            frame / their places; keyWeight: the pins of a keyframe of confidence 0.5 (scaled so that its pins
            together weigh as much as all the rigid landmarks of a frame times keyWeight; above, confidenceWeight: a
            keyframe sure at 100% is held); huber: robust, in pixels (the landmarks only).

    Returns:
        {"heads": {frame: head matrix}, "keys": {keyframe: (pins RMS error before, after) in pixels}, "error": (before,
        after) the RMS of the residuals}.
    """
    frames = [f for f in frames if f in heads and f in landmarks]
    if len(frames) < 2:
        raise ValueError("Solve needs at least two tracked frames")
    keys = {f: key for f, key in (keys or {}).items() if f in frames and len(key[0])}
    rigidIndices = np.asarray(list(rigidIndices), dtype=np.int64)
    center = rigidPoints.mean(0)
    index = {f: i for i, f in enumerate(frames)}
    count = 6 * len(frames)
    thetas = np.zeros((len(frames), 6))
    # Pixels per centimeter of the face (its size on the image), for the smoothness and the rotation scale
    sizes = []
    for f in frames:
        xy = _project(cameras[f], heads[f], rigidPoints, width, height)
        extent = float(np.linalg.norm(rigidPoints.max(0) - rigidPoints.min(0)))
        sizes.append(np.ptp(xy, axis=0).max() / max(extent, 1e-6))
    pixelsPerUnit = float(np.median(sizes))
    radius = float(np.linalg.norm(rigidPoints - center, axis=1).mean())

    seenAt = {f: np.array([int(i) not in set(hidden.get(f) or ()) and i < len(landmarks[f]) for i in rigidIndices],
                          dtype=bool) for f in frames}

    def residuals(thetas):
        """ The residual blocks (before the robust weights), and the positions in `frames` of the frames each depends
        on (one to three consecutive ones). """
        rows, owners = [], []
        moved = {f: _moved(heads[f], thetas[index[f]], center) for f in frames}
        projected = {f: _project(cameras[f], moved[f], rigidPoints, width, height) for f in frames}
        for f in frames:
            seen = seenAt[f]
            observed = landmarks[f][rigidIndices[seen]]
            rows.append(absoluteWeight * (projected[f][seen] - observed).ravel())
            owners.append((index[f],))
        for f, g in zip(frames, frames[1:]):
            if g != f + 1:
                continue
            seen = seenAt[f] & seenAt[g]
            observed = landmarks[g][rigidIndices[seen]] - landmarks[f][rigidIndices[seen]]
            rows.append(relativeWeight * ((projected[g][seen] - projected[f][seen]) - observed).ravel())
            owners.append((index[f], index[g]))
        for f, (points, positions, confidence) in keys.items():
            weight = keyWeight * confidenceWeight(confidence) * np.sqrt(len(rigidIndices) / float(len(points)))
            rows.append(weight * (_project(cameras[f], moved[f], points, width, height) - positions).ravel())
            owners.append((index[f], None))
        # Smoothness: the accelerations of the face (rotation at its radius, translation), in pixels
        scale = np.r_[np.full(3, radius), np.ones(3)] * pixelsPerUnit
        for a, b, c in zip(frames, frames[1:], frames[2:]):
            if c != a + 2:
                continue
            rows.append(smoothness * scale * (thetas[index[a]] - 2.0 * thetas[index[b]] + thetas[index[c]]))
            owners.append((index[a], index[b], index[c]))
        return rows, owners

    def robust(rows, owners):
        """ The landmark blocks reweighted (Huber of `huber` pixels per point); the pins and the smoothness kept. """
        weights = []
        for block, owner in zip(rows, owners):
            if None in owner or len(owner) == 3:
                weights.append(np.ones(len(block)))
                continue
            points = block.reshape(-1, 2) if len(block) % 2 == 0 else block[:, None]
            norm = np.linalg.norm(points, axis=1)
            weight = np.sqrt(np.minimum(1.0, huber / np.maximum(norm, 1e-12)))
            weights.append(np.repeat(weight, points.shape[1]))
        return weights

    def keyErrors(thetas):
        errors = {}
        for f, (points, positions, _) in keys.items():
            xy = _project(cameras[f], _moved(heads[f], thetas[index[f]], center), points, width, height)
            errors[f] = float(np.sqrt(((xy - positions) ** 2).sum(1).mean()))
        return errors
    keysBefore = keyErrors(thetas)
    rows, _ = residuals(thetas)
    before = float(np.sqrt(np.mean(np.concatenate(rows) ** 2)))
    step = 1e-6
    for iteration in range(iterations):
        if cancel:
            cancel()
        rows, owners = residuals(thetas)
        weights = robust(rows, owners)
        current = np.concatenate([w * r for w, r in zip(weights, rows)])
        offsets = np.cumsum([0] + [len(r) for r in rows])
        # The jacobian numerically, sparse: each block depends on up to three consecutive frames, so a parameter of
        # every third frame is moved at once (18 evaluations whatever the length of the shot)
        entries, rowIds, columnIds = [], [], []
        for color in range(3):
            for parameter in range(6):
                moved = thetas.copy()
                moved[color::3, parameter] += step
                shifted, _ = residuals(moved)
                for block, (w, r, owner) in enumerate(zip(weights, shifted, owners)):
                    which = [i for i in owner if i is not None and i % 3 == color]
                    if not which:
                        continue
                    derivative = (w * r - current[offsets[block]:offsets[block + 1]]) / step
                    nonzero = np.flatnonzero(derivative)
                    entries.append(derivative[nonzero])
                    rowIds.append(offsets[block] + nonzero)
                    columnIds.append(np.full(len(nonzero), 6 * which[0] + parameter))
        jacobian = sparse.csr_matrix((np.concatenate(entries), (np.concatenate(rowIds), np.concatenate(columnIds))),
                                     shape=(len(current), count))
        normal = (jacobian.T @ jacobian).tocsc()
        # (a little damping: the gauge of the sequence held where nothing tells)
        damping = 1e-6 * normal.diagonal().sum() / count + 1e-9
        delta = sparseSolve(normal + damping * sparse.identity(count, format="csc"), -(jacobian.T @ current))
        thetas += delta.reshape(-1, 6)
        if progress:
            progress((iteration + 1) / float(iterations))
        if np.abs(delta).max() < 1e-7:
            break
    rows, _ = residuals(thetas)
    after = float(np.sqrt(np.mean(np.concatenate(rows) ** 2)))
    keysAfter = keyErrors(thetas)
    return {"heads": {f: _moved(heads[f], thetas[index[f]], center) for f in frames},
            "keys": {f: (keysBefore[f], keysAfter[f]) for f in keys}, "error": (before, after)}
