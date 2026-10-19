"""
2D transforms fitted on point correspondences (Stabilize, MotionBlur, CornerPin) and 3x3 matrices
in Nuke coordinates (pixels, y up).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

# Transform models of the Stabilize node, with the number of points they need
MODELS = {
    "translate": 1,
    "rotate": 2,
    "scale": 2,
    "translate rotate": 2,
    "translate rotate scale": 2,
    "perspective": 4,
}


def homography(src: Sequence, dst: Sequence) -> Optional[np.ndarray]:
    """
    The 3x3 projective transform mapping the points `src` onto `dst` (4 points or more, least
    squares by normalized DLT), or None if they are degenerate.
    """
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    if len(src) < 4 or len(src) != len(dst):
        return None

    def normalization(points):
        center = points.mean(axis=0)
        scale = math.sqrt(2) / max(np.sqrt(((points - center) ** 2).sum(axis=1)).mean(), 1e-12)
        return np.array([[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1]])

    ts, td = normalization(src), normalization(dst)
    s = (ts @ np.c_[src, np.ones(len(src))].T).T
    d = (td @ np.c_[dst, np.ones(len(dst))].T).T
    rows = []
    for (x, y, _), (u, v, _) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, singular, vt = np.linalg.svd(np.array(rows))
    if singular[-2] < 1e-10:
        return None
    h = vt[-1].reshape(3, 3)
    matrix = np.linalg.inv(td) @ h @ ts
    # Sign and scale: a weight of 1 at the center of the source points (they are in front, whatever
    # the position of the horizon relatively to the origin)
    weight = matrix[2] @ np.r_[src.mean(axis=0), 1.0]
    if abs(weight) < 1e-12:
        return None
    return matrix / weight


def similarity(src: np.ndarray, dst: np.ndarray, rotate=True, scale=True, translate=True) -> np.ndarray:
    """
    The 3x3 transform mapping `src` onto `dst` by least squares, made of the allowed components
    (Umeyama): rotation and uniform scale around the centroid of `src`, translation.
    """
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    srcCenter, dstCenter = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - srcCenter, dst - dstCenter
    angle, factor = 0.0, 1.0
    if len(src) >= 2 and (rotate or scale):
        # Complex numbers: dst ~ s e^(i theta) src
        za, zb = a[:, 0] + 1j * a[:, 1], b[:, 0] + 1j * b[:, 1]
        ratio = (np.conj(za) * zb).sum() / max((np.abs(za) ** 2).sum(), 1e-12)
        if rotate:
            angle = float(np.angle(ratio))
        if scale:
            factor = float(np.abs(ratio)) if rotate else float(max((a * b).sum() / max((a * a).sum(), 1e-12), 1e-6))
    cos, sin = math.cos(angle) * factor, math.sin(angle) * factor
    linear = np.array([[cos, -sin], [sin, cos]])
    # Without translation, rotation and scale happen around the centroid of the source points
    offset = (dstCenter if translate else srcCenter) - linear @ srcCenter
    return np.array([[cos, -sin, offset[0]], [sin, cos, offset[1]], [0, 0, 1]])


def fitTransform(model: str, src: Sequence, dst: Sequence) -> np.ndarray:
    """
    The transform of a model (see MODELS) mapping the points `src` onto `dst`.

    Raises:
        ValueError: if there are not enough points for the model, or they are degenerate.
    """
    src, dst = np.asarray(src, dtype=np.float64).reshape(-1, 2), np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    needed = MODELS.get(model)
    if needed is None:
        raise ValueError(f"Unknown transform: {model}.")
    if len(src) < needed:
        raise ValueError(f"'{model}' needs {needed} tracks or more ({len(src)} available).")
    if model == "perspective":
        matrix = homography(src, dst)
        if matrix is None:
            raise ValueError("The tracks are aligned: no perspective transform.")
        return matrix
    return similarity(src, dst, rotate="rotate" in model, scale="scale" in model, translate="translate" in model)
