"""
The eyes of the FaceTracker (after FaceCapture, Technicolor: an eyeball sphere per eye, its gaze fitted on the image):
each eyeball placed behind the opening of its eyelids on the rest face, turned at each frame so that its iris center
goes through MediaPipe's iris landmark (its ray from the camera on the sphere).
"""
from typing import Dict, Optional, Tuple

import numpy as np

from . import scene3d

# MediaPipe landmarks of each eye (the subject's right and left): corners (outer, inner), upper and lower lids, the
# center of the iris (478 landmarks)
EYES = {"right": {"corners": (33, 133), "lids": (159, 145), "iris": 468},
        "left": {"corners": (263, 362), "lids": (386, 374), "iris": 473}}
# The eyeball radius against the width of the eye opening (about 1.2 cm for 2.9 cm), the iris radius against it
RADIUS_OF_WIDTH = 0.42
IRIS_OF_RADIUS = 0.5


def eyeballs(rest: np.ndarray, forward=(0.0, 0.0, 1.0)) -> Dict[str, Tuple[np.ndarray, float]]:
    """ {side: (center (3,), radius)} of the eyeballs of a rest face (face space, the face looking along `forward`):
    behind the middle of the eye opening (corners and lids) by most of their radius. """
    forward = np.asarray(forward, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    result = {}
    for side, eye in EYES.items():
        a, b = rest[eye["corners"][0]], rest[eye["corners"][1]]
        opening = (a + b + rest[eye["lids"][0]] + rest[eye["lids"][1]]) / 4.0
        radius = RADIUS_OF_WIDTH * float(np.linalg.norm(a - b))
        result[side] = (opening - forward * 0.8 * radius, radius)
    return result


def gazeThrough(center: np.ndarray, radius: float, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """ The unit direction from an eyeball center to where a ray (the iris seen) meets its sphere (the side toward the
    ray's origin; the point of the sphere nearest to the ray when it misses it). """
    direction = direction / np.linalg.norm(direction)
    toCenter = center - origin
    along = float(toCenter @ direction)
    closest = origin + along * direction
    miss = float(np.linalg.norm(closest - center))
    if miss < radius:
        point = origin + (along - np.sqrt(radius ** 2 - miss ** 2)) * direction
    else:
        point = closest
    gaze = point - center
    return gaze / max(np.linalg.norm(gaze), 1e-12)


def eyeGeometry(eyes, rings: int = 10, segments: int = 16, name: str = "eyes") -> scene3d.GeoData:
    """ The eyeballs [(center, radius, gaze)] as one mesh: each a sphere whose pole is along its gaze, its iris (the cap
    of IRIS_OF_RADIUS) with a ring of its own (the edge of the iris drawn). """
    vertices, counts, indices = [], [], []
    irisAngle = np.arcsin(IRIS_OF_RADIUS)
    # Latitudes from the front pole: the iris edge, then evenly to the back pole
    latitudes = np.r_[irisAngle * np.array([0.5, 1.0]), np.linspace(irisAngle, np.pi, rings + 1)[1:]]
    for center, radius, gaze in eyes:
        gaze = np.asarray(gaze, dtype=np.float64) / np.linalg.norm(gaze)
        helper = np.array([0.0, 1.0, 0.0]) if abs(gaze[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(helper, gaze)
        u /= np.linalg.norm(u)
        v = np.cross(gaze, u)
        base = len(vertices)
        vertices.append(center + radius * gaze)
        angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        for latitude in latitudes:
            for angle in angles:
                direction = np.cos(latitude) * gaze + np.sin(latitude) * (np.cos(angle) * u + np.sin(angle) * v)
                vertices.append(center + radius * direction)
        ring = lambda k, s: base + 1 + k * segments + s % segments  # noqa: E731
        for s in range(segments):
            counts.append(3)
            indices += [base, ring(0, s), ring(0, s + 1)]
        for k in range(len(latitudes) - 1):
            for s in range(segments):
                counts.append(4)
                indices += [ring(k, s), ring(k + 1, s), ring(k + 1, s + 1), ring(k, s + 1)]
    return scene3d.GeoData(np.array(vertices, dtype=np.float64).reshape(-1, 3), np.array(counts, dtype=np.int64),
                           np.array(indices, dtype=np.int64), name=name)


def eyesAt(rest: np.ndarray, head: np.ndarray, pixels: np.ndarray, camera, width: int, height: int,
           forward=(0.0, 0.0, 1.0)) -> Optional[list]:
    """ [(center, radius, gaze)] of both eyes in the world at a frame: the eyeballs of the rest face moved by the head
    matrix (face space to world), each turned toward the ray of its iris landmark (`pixels`: the 478 landmarks of the
    frame, Nuke pixels); None without the irises. """
    if pixels is None or len(pixels) < 478:
        return None
    linear = head[:3, :3]
    scale = float(np.cbrt(abs(np.linalg.det(linear))))
    eyes = []
    for side, (center, radius) in eyeballs(rest, forward).items():
        worldCenter = linear @ center + head[:3, 3]
        origin, direction = camera.ray(float(pixels[EYES[side]["iris"], 0]), float(pixels[EYES[side]["iris"], 1]),
                                       width, height)
        eyes.append((worldCenter, radius * scale, gazeThrough(worldCenter, radius * scale, origin, direction)))
    return eyes


def gazeAngles(gaze: np.ndarray, head: np.ndarray) -> Tuple[float, float]:
    """ (yaw, pitch) in degrees of a gaze (world) in the face space of the head (0, 0: straight ahead, +Z; yaw toward
    +X, pitch toward +Y). """
    local = np.linalg.solve(head[:3, :3], gaze)
    local /= np.linalg.norm(local)
    return float(np.degrees(np.arctan2(local[0], local[2]))), float(np.degrees(np.arcsin(np.clip(local[1], -1, 1))))
