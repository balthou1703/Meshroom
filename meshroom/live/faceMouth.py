"""
The inside of the mouth of the FaceTracker (after FaceCapture, Technicolor: teeth placed behind the lips): two arches
of teeth on the rest face, the upper one fixed to the skull, the lower one carried by the jaw (its rigid move fitted
at each frame on the landmarks of the chin and the jaw line).
"""
from typing import Tuple

import numpy as np

from . import scene3d

# MediaPipe landmarks: mouth corners (right, left), inner upper and lower lips (middle), the jaw line and the chin
CORNERS = (61, 291)
INNER_UPPER, INNER_LOWER = 13, 14
JAW = (152, 148, 176, 149, 150, 377, 400, 378, 379, 175, 199, 200, 171, 396)
# The arch against the mouth width: its width, its depth (the molars behind the incisors), the height of the teeth
ARCH_WIDTH, ARCH_DEPTH, TOOTH_HEIGHT, BEHIND_LIPS = 0.85, 0.9, 0.22, 0.12


def _arch(front: np.ndarray, width: float, depth: float, height: float, up: float, segments: int = 16,
          thickness: float = 0.25):
    """ (vertices, counts, indices) of an arch of teeth (a band: its outer and inner faces and its edges): its front
    middle at `front` (its biting edge), opening toward -Z, the teeth going `up` (+1: the upper ones, -1: the lower). """
    t = np.linspace(-1.0, 1.0, segments + 1)
    x = 0.5 * width * np.sin(0.5 * np.pi * t)
    z = -depth * (1.0 - np.cos(0.5 * np.pi * t))
    outline = np.stack([x, np.zeros_like(x), z], axis=1)
    # Inward (toward the inside of the arch), horizontal
    tangent = np.gradient(outline, axis=0)
    inward = np.cross([0.0, 1.0, 0.0], tangent)
    inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-12)
    rows = [outline, outline + [0.0, up * height, 0.0], outline + [0.0, up * height, 0.0] + thickness * inward,
            outline + thickness * inward]
    vertices = np.concatenate(rows) + front
    count = segments + 1
    indices = []
    for r in range(4):
        a, b = r * count, ((r + 1) % 4) * count
        for s in range(segments):
            indices += [a + s, a + s + 1, b + s + 1, b + s]
    return vertices, np.full(len(indices) // 4, 4), np.array(indices)


def teeth(rest: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """ ((vertices, counts, indices) of the upper teeth, of the lower teeth) of a rest face (face space, looking along
    +Z): behind the inner lips, as wide as the mouth. """
    mouth = float(np.linalg.norm(rest[CORNERS[0]] - rest[CORNERS[1]]))
    width, depth, height = ARCH_WIDTH * mouth, ARCH_DEPTH * mouth, TOOTH_HEIGHT * mouth
    behind = np.array([0.0, 0.0, -BEHIND_LIPS * mouth])
    middle = 0.5 * (rest[INNER_UPPER] + rest[INNER_LOWER])
    upper = _arch(middle + behind + [0.0, 0.02 * mouth, 0.0], width, depth, height, 1.0)
    lower = _arch(middle + behind + [0.0, -0.02 * mouth, -0.03 * mouth], 0.92 * width, 0.92 * depth, 0.9 * height, -1.0)
    return upper, lower


def jawMove(rest: np.ndarray, current: np.ndarray) -> np.ndarray:
    """ The 4x4 rigid move (face space) of the jaw from the rest face to a fitted one (`current`, face space): its chin
    and jaw line landmarks (Kabsch). """
    a, b = rest[list(JAW)], current[list(JAW)]
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((b - cb).T @ (a - ca))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ d @ vt
    move = np.eye(4)
    move[:3, :3] = rotation
    move[:3, 3] = cb - rotation @ ca
    return move


def teethAt(rest: np.ndarray, current: np.ndarray, head: np.ndarray, name: str = "teeth") -> scene3d.GeoData:
    """ The teeth at a frame, in the world: the upper ones on the skull (the head matrix), the lower ones moved by the
    jaw (jawMove of the fitted face `current`, face space) then by the head. """
    (upperVertices, upperCounts, upperIndices), (lowerVertices, lowerCounts, lowerIndices) = teeth(rest)
    jaw = jawMove(rest, current)
    lowerVertices = lowerVertices @ jaw[:3, :3].T + jaw[:3, 3]
    vertices = np.concatenate([upperVertices, lowerVertices]) @ head[:3, :3].T + head[:3, 3]
    return scene3d.GeoData(vertices, np.concatenate([upperCounts, lowerCounts]),
                           np.concatenate([upperIndices, lowerIndices + len(upperVertices)]), name=name)
