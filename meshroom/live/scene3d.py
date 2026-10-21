"""
3D data of the live compositing nodes: cameras, geometries and transformation matrices, carried by the
camera, geo and matrix streams (desc.Camera, desc.Geo, desc.Matrix), and the 3D math they need.

Conventions (as USD, Nuke and Maya):
- a right-handed world, Y up; 4x4 matrices act on column vectors (p' = M @ [x, y, z, 1]);
- a transform is translate @ rotate @ scale (scale first), the rotation of Euler angles in degrees applied
  in the order of their rotation order ("ZXY": Z first, then X, then Y, as Nuke's default);
- a camera is its camera-to-world matrix, looking along -Z with Y up; its lens is a focal length and a film
  back (horizontal / vertical aperture) in millimeters, the horizontal aperture fitting the image width;
- image positions are in pixels from the bottom-left corner, y up (Nuke coordinates);
- the 3x3 transforms of 2D images (Transform, CornerPin...) are embedded in 4x4 matrices (embed2D).
"""
from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# Matrices

def identity() -> np.ndarray:
    return np.eye(4)


def embed2D(matrix: np.ndarray) -> np.ndarray:
    """ A 3x3 transform of the plane (x, y, w) as a 4x4 matrix leaving z alone (Nuke's 4x4 matrix of a Transform). """
    m = np.asarray(matrix, dtype=np.float64)
    result = np.eye(4)
    result[np.ix_([0, 1, 3], [0, 1, 3])] = m
    return result


def extract2D(matrix: np.ndarray) -> np.ndarray:
    """ The 3x3 transform of the plane in a 4x4 matrix (inverse of embed2D). """
    m = np.asarray(matrix, dtype=np.float64)
    return m[np.ix_([0, 1, 3], [0, 1, 3])].copy()


def toMatrix4(matrix) -> np.ndarray:
    """ A 4x4 matrix from None (identity), a 3x3 2D transform (embedded) or a 4x4 matrix. """
    if matrix is None:
        return identity()
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape == (3, 3):
        return embed2D(m)
    if m.shape != (4, 4):
        raise ValueError(f"Not a transformation matrix: shape {m.shape}.")
    return m.copy()


_AXES = {"X": 0, "Y": 1, "Z": 2}
ROTATION_ORDERS = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]


def axisRotation(axis: int, degrees: float) -> np.ndarray:
    """ 3x3 rotation of `degrees` around the axis 0 (X), 1 (Y) or 2 (Z). """
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    rotation = np.eye(3)
    rotation[i, i], rotation[i, j], rotation[j, i], rotation[j, j] = c, -s, s, c
    return rotation


def eulerMatrix(rotate: Sequence[float], order: str = "ZXY") -> np.ndarray:
    """ 3x3 rotation of Euler angles (x, y, z) in degrees, applied in `order` (first letter first). """
    rotation = np.eye(3)
    for letter in order.upper():
        axis = _AXES[letter]
        rotation = axisRotation(axis, rotate[axis]) @ rotation
    return rotation


def matrixToEuler(rotation: np.ndarray, order: str = "ZXY") -> Tuple[float, float, float]:
    """ Euler angles (x, y, z) in degrees of a 3x3 rotation, for a rotation order (inverse of eulerMatrix). """
    i, j, k = (_AXES[letter] for letter in order.upper())
    # Parity of the permutation (i, j, k) of (0, 1, 2)
    sign = 1.0 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0
    r = np.asarray(rotation, dtype=np.float64)
    sinJ = max(-1.0, min(1.0, -sign * r[k, i]))
    angles = [0.0, 0.0, 0.0]
    angles[j] = math.asin(sinJ)
    if abs(sinJ) < 1.0 - 1e-9:
        angles[i] = math.atan2(sign * r[k, j], r[k, k])
        angles[k] = math.atan2(sign * r[j, i], r[i, i])
    else:
        # Gimbal lock: the first and last rotations turn around the same axis, all of it put on the first one
        angles[k] = 0.0
        angles[i] = math.atan2(-sign * r[j, k], r[j, j])
    return tuple(math.degrees(a) for a in angles)


def composeTRS(translate=(0, 0, 0), rotate=(0, 0, 0), scale=(1, 1, 1), order: str = "ZXY",
               pivot=(0, 0, 0)) -> np.ndarray:
    """ 4x4 matrix translate @ rotate @ scale, rotating and scaling around `pivot`. """
    m = np.eye(4)
    m[:3, :3] = eulerMatrix(rotate, order) @ np.diag([float(v) for v in scale])
    pivot = np.asarray(pivot, dtype=np.float64)
    m[:3, 3] = np.asarray(translate, dtype=np.float64) + pivot - m[:3, :3] @ pivot
    return m


def decomposeTRS(matrix: np.ndarray, order: str = "ZXY"):
    """ (translate, rotate in degrees, scale) of a 4x4 matrix without shear (inverse of composeTRS). """
    m = np.asarray(matrix, dtype=np.float64)
    translate = m[:3, 3].copy()
    linear = m[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    rotation = linear / scale
    if np.linalg.det(rotation) < 0:
        scale[0], rotation[:, 0] = -scale[0], -rotation[:, 0]
    # Orthonormalize (the closest rotation)
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    return tuple(translate), matrixToEuler(rotation, order), tuple(scale)


def decompose2D(matrix: np.ndarray, center=(0.0, 0.0)):
    """
    (translate (x, y), rotate in degrees, scale (x, y), skewX in degrees) of a 2D affine transform (3x3) around
    `center`, as the parameters of the Transform node (meshroom.live.ops.transformMatrix).
    """
    m = np.asarray(matrix, dtype=np.float64)
    m = m / m[2, 2]
    linear = m[:2, :2]
    # linear = rotation @ [[sx, tan(skew) sy], [0, sy]]
    rotate = math.degrees(math.atan2(linear[1, 0], linear[0, 0]))
    angle = math.radians(rotate)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    upper = rotation.T @ linear
    scaleX, scaleY = float(upper[0, 0]), float(upper[1, 1])
    skewX = math.degrees(math.atan(upper[0, 1] / scaleY)) if abs(scaleY) > 1e-12 else 0.0
    centerArray = np.asarray(center, dtype=np.float64)
    translate = m[:2, 2] - centerArray + linear @ centerArray
    return (float(translate[0]), float(translate[1])), rotate, (scaleX, scaleY), skewX


def interpolateMatrices(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    """ A matrix between a and b: translation, rotation (shortest arc) and scale interpolated. """
    from .camera import rodrigues, rotationVector
    ta, ra, sa = a[:3, 3], a[:3, :3], np.linalg.norm(a[:3, :3], axis=0)
    tb, rb, sb = b[:3, 3], b[:3, :3], np.linalg.norm(b[:3, :3], axis=0)
    ra, rb = ra / np.where(sa < 1e-12, 1, sa), rb / np.where(sb < 1e-12, 1, sb)
    rotation = rodrigues(rotationVector(rb @ ra.T) * weight) @ ra
    result = np.eye(4)
    result[:3, :3] = rotation @ np.diag(sa + (sb - sa) * weight)
    result[:3, 3] = ta + (tb - ta) * weight
    return result


def transformPoints(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


# Cameras

class CameraData:
    """
    A camera at one frame: its camera-to-world matrix (looking along -Z, Y up) and its lens (millimeters).
    """

    def __init__(self, matrix=None, focalLength: float = 35.0, horizontalAperture: float = 36.0,
                 verticalAperture: float = 24.0, near: float = 0.1, far: float = 10000.0, name: str = ""):
        self.matrix = toMatrix4(matrix)
        self.focalLength = float(focalLength)
        self.horizontalAperture = float(horizontalAperture)
        self.verticalAperture = float(verticalAperture)
        self.near = float(near)
        self.far = float(far)
        self.name = name

    def copy(self, **changes) -> "CameraData":
        values = dict(matrix=self.matrix, focalLength=self.focalLength, horizontalAperture=self.horizontalAperture,
                      verticalAperture=self.verticalAperture, near=self.near, far=self.far, name=self.name)
        values.update(changes)
        return CameraData(**values)

    @property
    def position(self) -> np.ndarray:
        return self.matrix[:3, 3].copy()

    @property
    def viewMatrix(self) -> np.ndarray:
        """ World to camera. """
        return np.linalg.inv(self.matrix)

    def focalPixels(self, width: int) -> float:
        """ Focal length in pixels for an image `width` pixels wide (the horizontal aperture fits the width). """
        return self.focalLength / max(self.horizontalAperture, 1e-9) * width

    def project(self, points: np.ndarray, width: int, height: int):
        """
        Image positions (Nuke pixels) of world points, their depth (distance along the view axis) and whether
        they are in front of the camera: (xy (n, 2), depth (n,), inFront (n,)).
        """
        camera = transformPoints(self.viewMatrix, points)
        depth = -camera[:, 2]
        inFront = depth > max(self.near, 1e-9)
        safe = np.where(np.abs(depth) < 1e-12, 1e-12, depth)
        f = self.focalPixels(width)
        xy = np.c_[width / 2.0 + f * camera[:, 0] / safe, height / 2.0 + f * camera[:, 1] / safe]
        return xy, depth, inFront

    def ray(self, x: float, y: float, width: int, height: int):
        """ (origin, unit direction) in the world of the ray through the image position (x, y) (Nuke pixels). """
        f = self.focalPixels(width)
        direction = self.matrix[:3, :3] @ np.array([(x - width / 2.0) / f, (y - height / 2.0) / f, -1.0])
        return self.position, direction / np.linalg.norm(direction)

    def asDict(self) -> dict:
        return {"matrix": np.round(self.matrix, 9).tolist(), "focalLength": self.focalLength,
                "horizontalAperture": self.horizontalAperture, "verticalAperture": self.verticalAperture,
                "near": self.near, "far": self.far}

    def __repr__(self):
        return (f"CameraData(position={np.round(self.position, 4).tolist()}, focal={self.focalLength:g}mm, "
                f"aperture={self.horizontalAperture:g}x{self.verticalAperture:g}mm)")


# Pose of a camera in the frame of computer vision (x right, y down, z forward), used by the solvers
_GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def cvPose(cameraMatrix: np.ndarray):
    """ (R, t) with x_cv = R X + t of a camera-to-world matrix (camera looking along -Z). """
    view = np.linalg.inv(cameraMatrix)
    return _GL_TO_CV @ view[:3, :3], _GL_TO_CV @ view[:3, 3]


def cameraMatrixFromCvPose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    view = np.eye(4)
    view[:3, :3] = _GL_TO_CV @ rotation
    view[:3, 3] = _GL_TO_CV @ translation
    return np.linalg.inv(view)


def lookAt(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """ Camera-to-world matrix of a camera at `eye` looking at `target`. """
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    back = eye - target
    back /= max(np.linalg.norm(back), 1e-12)
    right = np.cross(up, back)
    if np.linalg.norm(right) < 1e-9:
        right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    trueUp = np.cross(back, right)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, trueUp, back, eye
    return m


# Geometries

class Topology:
    """
    The faces of a geometry, shared by its GeoData at every frame (only the vertices move): the triangles, the edges
    and the triangles on each side of an edge are computed once, not at each frame.
    """

    def __init__(self, faceCounts=None, faceIndices=None, vertexCount: Optional[int] = None):
        self.faceCounts = np.asarray(faceCounts if faceCounts is not None else [], dtype=np.int64).ravel()
        self.faceIndices = np.asarray(faceIndices if faceIndices is not None else [], dtype=np.int64).ravel()
        if self.faceCounts.sum() != len(self.faceIndices):
            raise ValueError("The face vertex counts do not match the face vertex indices.")
        if len(self.faceIndices) and (self.faceIndices.min() < 0 or
                                      (vertexCount is not None and self.faceIndices.max() >= vertexCount)):
            raise ValueError("Face vertex index out of range.")
        self.maxIndex = int(self.faceIndices.max()) if len(self.faceIndices) else -1
        # Reentrant: a cached value is computed from another one (the triangle edges from the triangles)
        self._lock = threading.RLock()
        self._cache = {}

    def _cached(self, key, compute):
        with self._lock:
            if key not in self._cache:
                self._cache[key] = compute()
            return self._cache[key]

    def _faceStarts(self) -> np.ndarray:
        return np.r_[0, np.cumsum(self.faceCounts)[:-1]].astype(np.int64)

    def triangles(self) -> np.ndarray:
        """ Vertex indices (m, 3) of the triangles of the faces (fan triangulation). """
        def compute():
            starts, result = self._faceStarts(), []
            for count in np.unique(self.faceCounts):
                if count < 3:
                    continue
                faceStarts = starts[self.faceCounts == count]
                for k in range(1, count - 1):
                    result.append(np.c_[self.faceIndices[faceStarts], self.faceIndices[faceStarts + k],
                                        self.faceIndices[faceStarts + k + 1]])
            return np.concatenate(result) if result else np.zeros((0, 3), dtype=np.int64)
        return self._cached("triangles", compute)

    def edges(self) -> np.ndarray:
        """ Unique vertex index pairs (m, 2) of the edges of the faces. """
        def compute():
            starts, result = self._faceStarts(), []
            for count in np.unique(self.faceCounts):
                if count < 2:
                    continue
                faceStarts = starts[self.faceCounts == count]
                for k in range(count):
                    result.append(np.c_[self.faceIndices[faceStarts + k], self.faceIndices[faceStarts + (k + 1) % count]])
            if not result:
                return np.zeros((0, 2), dtype=np.int64)
            return np.unique(np.sort(np.concatenate(result), axis=1), axis=0)
        return self._cached("edges", compute)

    def triangleEdges(self):
        """
        The edges of the triangles (m, 2) and the triangles on their two sides (m, 2) (-1: an open edge; a third
        triangle on an edge is ignored): to find the silhouettes and the creases.
        """
        def compute():
            triangles = self.triangles()
            if not len(triangles):
                return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
            pairs = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
            faces = np.tile(np.arange(len(triangles)), 3)
            edges, inverse = np.unique(pairs, axis=0, return_inverse=True)
            inverse = inverse.ravel()
            order = np.argsort(inverse, kind="stable")
            sortedEdges = inverse[order]
            first = np.r_[True, sortedEdges[1:] != sortedEdges[:-1]]
            sides = np.full((len(edges), 2), -1, dtype=np.int64)
            sides[sortedEdges[first], 0] = faces[order][first]
            # The second triangle of an edge: the entry right after its first one
            secondIndex = np.flatnonzero(~first)
            secondIndex = secondIndex[first[secondIndex - 1]]
            sides[sortedEdges[secondIndex], 1] = faces[order][secondIndex]
            return edges, sides
        return self._cached("triangleEdges", compute)


_MERGED_TOPOLOGIES: "OrderedDict[tuple, tuple]" = OrderedDict()
_MERGED_LOCK = threading.Lock()


def _mergedTopology(geos) -> "Topology":
    """ The topology of geometries merged together, the same object while they keep their topologies. """
    topologies = tuple(geo.topology for geo in geos)
    key = tuple(id(topology) for topology in topologies) + tuple(len(geo.vertices) for geo in geos)
    with _MERGED_LOCK:
        cached = _MERGED_TOPOLOGIES.get(key)
        if cached is not None and all(a is b for a, b in zip(cached[0], topologies)):
            _MERGED_TOPOLOGIES.move_to_end(key)
            return cached[1]
    offsets = np.cumsum([0] + [len(geo.vertices) for geo in geos[:-1]])
    topology = Topology(np.concatenate([geo.faceCounts for geo in geos]),
                        np.concatenate([geo.faceIndices + offset for geo, offset in zip(geos, offsets)]))
    with _MERGED_LOCK:
        _MERGED_TOPOLOGIES[key] = (topologies, topology)
        while len(_MERGED_TOPOLOGIES) > 32:
            _MERGED_TOPOLOGIES.popitem(last=False)
    return topology


class GeoData:
    """
    Geometry at one frame, in world space: vertices (n, 3) and polygons (`faceCounts`: number of vertices of
    each face, `faceIndices`: their vertex indices, concatenated). Without faces: a point cloud. The faces are a
    Topology, shared with the same geometry at the other frames.
    """

    def __init__(self, vertices=None, faceCounts=None, faceIndices=None, name: str = "",
                 topology: Optional[Topology] = None):
        self.vertices = np.asarray(vertices if vertices is not None else np.zeros((0, 3)), dtype=np.float64).reshape(-1, 3)
        if topology is None:
            topology = Topology(faceCounts, faceIndices, len(self.vertices))
        elif topology.maxIndex >= len(self.vertices):
            raise ValueError("Face vertex index out of range.")
        self.topology = topology
        self.name = name

    @property
    def faceCounts(self) -> np.ndarray:
        return self.topology.faceCounts

    @property
    def faceIndices(self) -> np.ndarray:
        return self.topology.faceIndices

    @property
    def isEmpty(self) -> bool:
        return len(self.vertices) == 0

    def transformed(self, matrix: np.ndarray) -> "GeoData":
        if matrix is None or np.allclose(matrix, np.eye(4)):
            return self
        return GeoData(transformPoints(matrix, self.vertices), name=self.name, topology=self.topology)

    @staticmethod
    def merged(geos: Sequence["GeoData"]) -> "GeoData":
        geos = [geo for geo in geos if geo is not None]
        if not geos:
            return GeoData()
        if len(geos) == 1:
            return geos[0]
        return GeoData(np.concatenate([geo.vertices for geo in geos]), name="+".join(geo.name for geo in geos if geo.name),
                       topology=_mergedTopology(geos))

    def triangles(self) -> np.ndarray:
        """ Vertex indices (m, 3) of the triangles of the faces (fan triangulation). """
        return self.topology.triangles()

    def edges(self) -> np.ndarray:
        """ Unique vertex index pairs (m, 2) of the edges of the faces. """
        return self.topology.edges()

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.isEmpty:
            return np.zeros(3), np.zeros(3)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def __repr__(self):
        return f"GeoData({len(self.vertices)} vertices, {len(self.faceCounts)} faces)"


def framingCamera(geo: Optional[GeoData], focalLength=35.0, horizontalAperture=36.0, verticalAperture=24.0) -> CameraData:
    """ A camera looking at a geometry from +Z, far enough to see all of it (at the origin without geometry). """
    camera = CameraData(focalLength=focalLength, horizontalAperture=horizontalAperture, verticalAperture=verticalAperture)
    if geo is None or geo.isEmpty:
        camera.matrix = lookAt((0.0, 0.0, 10.0), (0.0, 0.0, 0.0))
        return camera
    low, high = geo.bounds()
    center = (low + high) / 2.0
    radius = max(float(np.linalg.norm(high - low)) / 2.0, 1e-6)
    halfAngle = math.atan(min(horizontalAperture, verticalAperture) / 2.0 / focalLength)
    distance = radius / math.sin(halfAngle) * 1.1
    camera.matrix = lookAt(center + np.array([0.0, 0.0, distance]), center)
    camera.near = max(distance - radius * 2, distance * 1e-3)
    camera.far = distance + radius * 4
    return camera


def centeredCamera(camera: CameraData, geo: Optional[GeoData], aspect: float = 16.0 / 9.0) -> CameraData:
    """
    The camera moved along its view axis (same orientation and lens) to have the whole geometry centered in its view:
    e.g. a camera inside a large geometry stepped back out of it. `aspect`: width / height of the image.
    """
    if geo is None or geo.isEmpty:
        return camera
    low, high = geo.bounds()
    center = (low + high) / 2.0
    radius = max(float(np.linalg.norm(high - low)) / 2.0, 1e-6)
    halfWidth = math.atan(camera.horizontalAperture / 2.0 / camera.focalLength)
    halfHeight = math.atan(camera.horizontalAperture / max(aspect, 1e-6) / 2.0 / camera.focalLength)
    distance = radius / math.sin(min(halfWidth, halfHeight)) * 1.1
    matrix = camera.matrix.copy()
    back = matrix[:3, 2] / max(np.linalg.norm(matrix[:3, 2]), 1e-12)
    matrix[:3, 3] = center + back * distance
    return camera.copy(matrix=matrix, near=min(camera.near, max(distance - radius * 2, distance * 1e-3)),
                       far=max(camera.far, distance + radius * 4))


def raycast(geo: GeoData, origin: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
    """ The nearest point of the faces of a geometry hit by a ray (Moller-Trumbore), or None. """
    triangles = geo.triangles()
    if not len(triangles):
        return None
    v0, v1, v2 = (geo.vertices[triangles[:, i]] for i in range(3))
    edge1, edge2 = v1 - v0, v2 - v0
    p = np.cross(direction, edge2)
    determinant = (edge1 * p).sum(axis=1)
    valid = np.abs(determinant) > 1e-12
    inverse = np.where(valid, 1.0 / np.where(valid, determinant, 1.0), 0.0)
    s = origin - v0
    u = (s * p).sum(axis=1) * inverse
    q = np.cross(s, edge1)
    v = (q * direction).sum(axis=1) * inverse
    t = (edge2 * q).sum(axis=1) * inverse
    # A small tolerance: a ray through a vertex or along an edge hits (rounding would miss every triangle around it)
    hit = valid & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7) & (t > 1e-9)
    if not hit.any():
        return None
    nearest = np.argmin(np.where(hit, t, np.inf))
    return origin + direction * t[nearest]


def nearestVertexOnScreen(geo: GeoData, camera: CameraData, x: float, y: float, width: int, height: int,
                          radius: float) -> Optional[np.ndarray]:
    """ The vertex (of a point cloud) projected the nearest to (x, y), within `radius` pixels, or None. """
    if geo.isEmpty:
        return None
    xy, _, inFront = camera.project(geo.vertices, width, height)
    distance = np.where(inFront, np.hypot(xy[:, 0] - x, xy[:, 1] - y), np.inf)
    index = int(np.argmin(distance))
    return geo.vertices[index].copy() if distance[index] <= radius else None


# Pose of a camera from 2D / 3D correspondences (the pins of the Resection node)

def solvePose(points: np.ndarray, positions: np.ndarray, width: int, height: int, focalPixels: float,
              initial: np.ndarray, iterations: int = 30):
    """
    The camera-to-world matrix whose projection puts the 3D `points` at the image `positions` (Nuke pixels), and
    the RMS error in pixels. With fewer than 4 points the pose is not determined: it moves as little as possible
    from `initial` (1 point: the camera moves parallel to the image; 2: and turns, gets closer; 3: almost free).
    Gauss-Newton with a damping toward `initial` (prior weighted down as points are added).
    """
    from .camera import _poseDLT, rodrigues
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    count = len(points)
    if count == 0:
        return toMatrix4(initial), 0.0
    observations = np.c_[(positions[:, 0] - width / 2.0) / focalPixels, -(positions[:, 1] - height / 2.0) / focalPixels]
    r0, t0 = cvPose(toMatrix4(initial))
    depth = np.median((points @ r0.T + t0)[:, 2])
    scale = max(abs(float(depth)), 1e-6)
    # Weights of the prior (in pixels per radian / per unit of depth)
    factor = {1: 1.0, 2: 0.3, 3: 0.05}.get(count, 1e-3)
    weights = focalPixels * factor * np.array([0.5, 0.5, 0.5, 0.1 / scale, 0.1 / scale, 0.3 / scale])

    def residuals(params, rotation0, translation0):
        rotation = rodrigues(params[:3]) @ rotation0
        translation = translation0 + params[3:]
        camera = points @ rotation.T + translation
        z = np.where(np.abs(camera[:, 2]) < 1e-9, 1e-9, camera[:, 2])
        reprojection = (camera[:, :2] / z[:, np.newaxis] - observations).ravel() * focalPixels
        # Points behind the camera are pushed in front
        behind = np.minimum(camera[:, 2], 0.0) * focalPixels / scale * 10.0
        return np.r_[reprojection, behind, weights * params]

    def solveFrom(rotation0, translation0):
        params = np.zeros(6)
        damping = 1e-3
        current = residuals(params, rotation0, translation0)
        for _ in range(iterations):
            jacobian = np.empty((len(current), 6))
            for k in range(6):
                step = np.zeros(6)
                step[k] = 1e-6
                jacobian[:, k] = (residuals(params + step, rotation0, translation0) - current) / 1e-6
            normal = jacobian.T @ jacobian
            gradient = jacobian.T @ current
            improved = False
            for _ in range(8):
                try:
                    delta = -np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-9), gradient)
                except np.linalg.LinAlgError:
                    break
                candidate = residuals(params + delta, rotation0, translation0)
                if candidate @ candidate < current @ current:
                    params, current = params + delta, candidate
                    damping = max(damping / 3.0, 1e-9)
                    improved = True
                    break
                damping *= 4.0
            if not improved or np.linalg.norm(delta) < 1e-10:
                break
        rotation = rodrigues(params[:3]) @ rotation0
        return rotation, translation0 + params[3:]

    candidates = [(r0, t0)]
    if count >= 6:
        pose = _poseDLT(points, observations)
        if pose is not None:
            candidates.append(pose)
    best, bestError = None, None
    for rotation0, translation0 in candidates:
        rotation, translation = solveFrom(rotation0, translation0)
        camera = points @ rotation.T + translation
        z = np.where(np.abs(camera[:, 2]) < 1e-9, 1e-9, camera[:, 2])
        error = float(np.sqrt(np.mean(np.sum((camera[:, :2] / z[:, np.newaxis] - observations) ** 2, axis=1)))) * focalPixels
        if np.any(camera[:, 2] <= 0):
            error += 1e6
        if bestError is None or error < bestError:
            best, bestError = (rotation, translation), error
    return cameraMatrixFromCvPose(*best), bestError


# Drawing

def drawSegments(rgba: np.ndarray, bbox, starts: np.ndarray, ends: np.ndarray, color, alpha: float = 1.0,
                 thickness: int = 1, maxSamples: int = 20_000_000) -> np.ndarray:
    """ Draw segments (Nuke pixels) over an RGBA array covering `bbox` (row 0 at the top), in place. """
    starts, ends = np.asarray(starts, np.float64).reshape(-1, 2), np.asarray(ends, np.float64).reshape(-1, 2)
    if not len(starts):
        return rgba
    height, width = rgba.shape[:2]
    lengths = np.ceil(np.abs(ends - starts).max(axis=1)).astype(np.int64) + 1
    lengths = np.minimum(lengths, 4 * max(width, height))
    total = int(lengths.sum())
    if total > maxSamples:
        keep = np.linspace(0, len(starts) - 1, max(1, int(len(starts) * maxSamples / total))).astype(np.int64)
        starts, ends, lengths = starts[keep], ends[keep], lengths[keep]
    segment = np.repeat(np.arange(len(starts)), lengths)
    first = np.r_[0, np.cumsum(lengths)[:-1]]
    t = (np.arange(len(segment)) - first[segment]) / np.maximum(lengths[segment] - 1, 1)
    xs = starts[segment, 0] + (ends[segment, 0] - starts[segment, 0]) * t
    ys = starts[segment, 1] + (ends[segment, 1] - starts[segment, 1]) * t
    _plot(rgba, bbox, xs, ys, color, alpha, thickness)
    return rgba


def drawPoints(rgba: np.ndarray, bbox, points: np.ndarray, color, alpha: float = 1.0, size: int = 1) -> np.ndarray:
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if len(points):
        _plot(rgba, bbox, points[:, 0], points[:, 1], color, alpha, size)
    return rgba


def _plot(rgba, bbox, xs, ys, color, alpha, thickness):
    height, width = rgba.shape[:2]
    offsets = range(-(thickness // 2), thickness - thickness // 2)
    columns0 = np.floor(xs - bbox.x0).astype(np.int64)
    rows0 = np.floor(bbox.y1 - ys).astype(np.int64)
    color = np.asarray(color, np.float32)[:3]
    for dx in offsets:
        for dy in offsets:
            columns, rows = columns0 + dx, rows0 + dy
            keep = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
            rows, columns = rows[keep], columns[keep]
            rgba[rows, columns, :3] = rgba[rows, columns, :3] * (1 - alpha) + color * alpha
            rgba[rows, columns, 3] = np.maximum(rgba[rows, columns, 3], alpha)


def drawGeo(rgba: np.ndarray, bbox, geo: GeoData, camera: CameraData, width: int, height: int, color,
            alpha: float = 0.8, maxEdges: int = 200_000, pointSize: int = 2) -> np.ndarray:
    """ Draw the wireframe of a geometry (its points without faces) seen through a camera, in place. """
    if geo is None or geo.isEmpty:
        return rgba
    xy, _, inFront = camera.project(geo.vertices, width, height)
    if len(geo.faceCounts):
        edges = geo.edges()
        edges = edges[inFront[edges[:, 0]] & inFront[edges[:, 1]]]
        if len(edges) > maxEdges:
            edges = edges[np.linspace(0, len(edges) - 1, maxEdges).astype(np.int64)]
        # Edges far outside the image are skipped
        a, b = xy[edges[:, 0]], xy[edges[:, 1]]
        margin = max(width, height)
        inside = ~(((a[:, 0] < -margin) & (b[:, 0] < -margin)) | ((a[:, 0] > width + margin) & (b[:, 0] > width + margin)) |
                   ((a[:, 1] < -margin) & (b[:, 1] < -margin)) | ((a[:, 1] > height + margin) & (b[:, 1] > height + margin)))
        drawSegments(rgba, bbox, a[inside], b[inside], color, alpha)
    else:
        drawPoints(rgba, bbox, xy[inFront], color, alpha, pointSize)
    return rgba


# Files: geometries and cameras (OBJ and ASCII USD read here; USD, Alembic and FBX through their modules if installed)

GEO_EXTENSIONS = (".obj", ".usd", ".usda", ".usdc", ".usdz", ".abc", ".fbx", ".ply")
CAMERA_EXTENSIONS = (".usd", ".usda", ".usdc", ".usdz", ".abc", ".fbx")


class SceneFile:
    """
    The content of a 3D file: meshes / point clouds and cameras, each at any frame (sampled or static).
    `geo(frame)` merges all the geometries (or the one named), `camera(frame, name)` gives one camera.
    """

    def __init__(self):
        # name: {"vertices": samples, "faceCounts", "faceIndices", "matrix": samples}
        self.meshes: "OrderedDict[str, dict]" = OrderedDict()
        # name: {"matrix": samples, "focalLength": samples, "horizontalAperture"..., "near", "far"}
        self.cameras: "OrderedDict[str, dict]" = OrderedDict()

    def geo(self, frame: float, name: str = "") -> GeoData:
        names = [n for n in self.meshes if not name or _namesMatch(n, name)]
        if name and not names:
            raise RuntimeError(f"No geometry '{name}' in the file (geometries: {', '.join(self.meshes) or 'none'}).")
        geos = []
        for meshName in names:
            mesh = self.meshes[meshName]
            vertices = sampleAt(mesh["vertices"], frame)
            topology = mesh.get("_topology")
            if topology is None:
                # The faces do not move: one Topology for every frame
                topology = mesh.setdefault("_topology", Topology(mesh.get("faceCounts"), mesh.get("faceIndices")))
            geo = GeoData(vertices, name=meshName, topology=topology)
            geos.append(geo.transformed(sampleAt(mesh.get("matrix"), frame)))
        return GeoData.merged(geos)

    def camera(self, frame: float, name: str = "") -> CameraData:
        names = [n for n in self.cameras if not name or _namesMatch(n, name)]
        if not names:
            raise RuntimeError(f"No camera '{name}' in the file (cameras: {', '.join(self.cameras) or 'none'})."
                               if name else "No camera in the file.")
        camera = self.cameras[names[0]]
        values = {key: sampleAt(camera.get(key), frame) for key in
                  ("matrix", "focalLength", "horizontalAperture", "verticalAperture", "near", "far")}
        return CameraData(name=names[0], **{key: value for key, value in values.items() if value is not None})


def _namesMatch(fullName: str, name: str) -> bool:
    """
    Whether an object of a file (its full path, e.g. /cam_A/cam_AShape) is the one named `name`: its path, the end of
    it, or its parent transform (in Alembic, Maya and FBX files the shape is under a transform of the name seen in
    the 3D software: /cam_A/cam_AShape is "cam_A").
    """
    parts = [part for part in fullName.split("/") if part]
    wanted = [part for part in name.split("/") if part]
    if not wanted:
        return True
    return parts[-len(wanted):] == wanted or (len(parts) > len(wanted) and parts[-len(wanted) - 1:-1] == wanted)


def sampleAt(samples, frame: float):
    """
    The value at a frame of samples: a static value, or {frame: value} interpolated linearly between the
    samples (matrices: translation, rotation and scale), held before the first and after the last one.
    """
    if not isinstance(samples, dict):
        return samples
    if not samples:
        return None
    frames = sorted(samples)
    if frame <= frames[0]:
        return samples[frames[0]]
    if frame >= frames[-1]:
        return samples[frames[-1]]
    index = int(np.searchsorted(frames, frame))
    a, b = frames[index - 1], frames[index]
    weight = (frame - a) / float(b - a)
    va, vb = samples[a], samples[b]
    if isinstance(va, np.ndarray) and va.shape == (4, 4):
        return interpolateMatrices(va, vb, weight)
    if isinstance(va, np.ndarray):
        return va + (vb - va) * weight if va.shape == vb.shape else va
    return va + (vb - va) * weight


_FILES: "OrderedDict[tuple, SceneFile]" = OrderedDict()
_FILES_LOCK = threading.Lock()


def readSceneFile(path: str, fps: float = 24.0) -> SceneFile:
    """ The meshes and cameras of a 3D file (cached while the file does not change). """
    if not path:
        raise RuntimeError("No file.")
    if not os.path.isfile(path):
        raise RuntimeError(f"File not found: {path}")
    key = (os.path.abspath(path), os.path.getmtime(path), os.path.getsize(path), float(fps))
    with _FILES_LOCK:
        if key in _FILES:
            _FILES.move_to_end(key)
            return _FILES[key]
    extension = os.path.splitext(path)[1].lower()
    if extension == ".obj":
        scene = _readObj(path)
    elif extension == ".ply":
        scene = _readPly(path)
    elif extension in (".usd", ".usda", ".usdc", ".usdz"):
        scene = _readUsd(path)
    elif extension == ".abc":
        scene = _readAlembic(path, fps)
    elif extension == ".fbx":
        scene = _readFbx(path, fps)
    else:
        raise RuntimeError(f"Unsupported 3D file format '{extension}' (supported: {', '.join(GEO_EXTENSIONS)}).")
    with _FILES_LOCK:
        _FILES[key] = scene
        while len(_FILES) > 8:
            _FILES.popitem(last=False)
    return scene


def _readObj(path: str) -> SceneFile:
    vertices, counts, indices = [], [], []
    with open(path, "r", errors="replace") as file:
        for line in file:
            if line.startswith("v "):
                values = line.split()
                vertices.append((float(values[1]), float(values[2]), float(values[3])))
            elif line.startswith("f "):
                face = []
                for token in line.split()[1:]:
                    index = int(token.split("/")[0])
                    face.append(index - 1 if index > 0 else len(vertices) + index)
                counts.append(len(face))
                indices.extend(face)
    scene = SceneFile()
    scene.meshes[os.path.splitext(os.path.basename(path))[0]] = {
        "vertices": np.array(vertices, dtype=np.float64).reshape(-1, 3), "faceCounts": counts, "faceIndices": indices}
    return scene


_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2", "ushort": "u2",
              "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4", "float": "f4", "float32": "f4",
              "double": "f8", "float64": "f8"}


def _readPly(path: str) -> SceneFile:
    """ Vertices (x, y, z) and faces of a PLY file (ASCII or binary). """
    with open(path, "rb") as file:
        if file.readline().strip() != b"ply":
            raise RuntimeError("Not a PLY file.")
        fmt, elements = "ascii", []
        while True:
            line = file.readline()
            if not line:
                raise RuntimeError("Truncated PLY header.")
            words = line.decode("ascii", "replace").split()
            if not words:
                continue
            if words[0] == "format":
                fmt = words[1]
            elif words[0] == "element":
                elements.append({"name": words[1], "count": int(words[2]), "properties": []})
            elif words[0] == "property":
                elements[-1]["properties"].append(words[1:])
            elif words[0] == "end_header":
                break
        data = file.read()
    vertices, counts, indices = np.zeros((0, 3)), [], []
    if fmt == "ascii":
        tokens = data.split()
        position = 0
        for element in elements:
            for _ in range(element["count"]):
                values = {}
                for prop in element["properties"]:
                    if prop[0] == "list":
                        count = int(float(tokens[position]))
                        values[prop[-1]] = [int(float(v)) for v in tokens[position + 1:position + 1 + count]]
                        position += 1 + count
                    else:
                        values[prop[-1]] = float(tokens[position])
                        position += 1
                if element["name"] == "vertex":
                    element.setdefault("rows", []).append((values.get("x", 0.0), values.get("y", 0.0), values.get("z", 0.0)))
                elif element["name"] == "face":
                    face = values.get("vertex_indices", values.get("vertex_index", []))
                    counts.append(len(face))
                    indices.extend(face)
            if element["name"] == "vertex":
                vertices = np.array(element.get("rows", []), dtype=np.float64).reshape(-1, 3)
    else:
        order = "<" if fmt == "binary_little_endian" else ">"
        offset = 0
        for element in elements:
            if all(prop[0] != "list" for prop in element["properties"]):
                dtype = np.dtype([(prop[-1], order + _PLY_TYPES[prop[0]]) for prop in element["properties"]])
                table = np.frombuffer(data, dtype=dtype, count=element["count"], offset=offset)
                offset += dtype.itemsize * element["count"]
                if element["name"] == "vertex":
                    vertices = np.c_[table["x"], table["y"], table["z"]].astype(np.float64)
                continue
            for _ in range(element["count"]):
                for prop in element["properties"]:
                    if prop[0] == "list":
                        countType, itemType = np.dtype(order + _PLY_TYPES[prop[1]]), np.dtype(order + _PLY_TYPES[prop[2]])
                        count = int(np.frombuffer(data, countType, 1, offset)[0])
                        offset += countType.itemsize
                        items = np.frombuffer(data, itemType, count, offset)
                        offset += itemType.itemsize * count
                        if element["name"] == "face" and prop[-1] in ("vertex_indices", "vertex_index"):
                            counts.append(count)
                            indices.extend(int(i) for i in items)
                    else:
                        offset += np.dtype(_PLY_TYPES[prop[0]]).itemsize
    scene = SceneFile()
    scene.meshes[os.path.splitext(os.path.basename(path))[0]] = {"vertices": vertices, "faceCounts": counts,
                                                                 "faceIndices": indices}
    return scene


# USD

def _readUsd(path: str) -> SceneFile:
    try:
        from pxr import Usd  # noqa: F401
    except ImportError:
        if os.path.splitext(path)[1].lower() == ".usda" or _isUsdAscii(path):
            return _readUsdAscii(path)
        raise RuntimeError("Reading binary USD files needs the USD Python module (pxr), not available here: "
                           "use a .usda file, or a Meshroom environment with USD.")
    return _readUsdPxr(path)


def _isUsdAscii(path: str) -> bool:
    with open(path, "rb") as file:
        return file.read(8).startswith(b"#usda")


def _usdMatrix(value) -> np.ndarray:
    """ A USD matrix4d (row vectors) as a matrix acting on column vectors. """
    return np.array(value, dtype=np.float64).reshape(4, 4).T


def _readUsdPxr(path: str) -> SceneFile:
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise RuntimeError(f"USD cannot open {path}.")
    start, end = stage.GetStartTimeCode(), stage.GetEndTimeCode()
    frames = list(range(int(math.floor(start)), int(math.ceil(end)) + 1)) if stage.HasAuthoredTimeCodeRange() else []
    times = frames or [Usd.TimeCode.Default()]
    scene = SceneFile()

    def sampled(function):
        if not frames:
            return function(Usd.TimeCode.Default())
        values = {frame: function(Usd.TimeCode(frame)) for frame in frames}
        first = values[frames[0]]
        if all(_same(first, value) for value in values.values()):
            return first
        return values

    cache = {}

    def worldMatrix(prim, time):
        key = (str(prim.GetPath()), time if isinstance(time, (int, float)) else "default")
        if key not in cache:
            cache[key] = _usdMatrix(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time))
        return cache[key]

    for prim in stage.Traverse():
        name = str(prim.GetPath())
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Points):
            pointBased = UsdGeom.PointBased(prim)
            mesh = {"vertices": sampled(lambda t: np.array(pointBased.GetPointsAttr().Get(t) or [], dtype=np.float64)
                                        .reshape(-1, 3)),
                    "matrix": sampled(lambda t: worldMatrix(prim, t))}
            if prim.IsA(UsdGeom.Mesh):
                geomMesh = UsdGeom.Mesh(prim)
                mesh["faceCounts"] = list(geomMesh.GetFaceVertexCountsAttr().Get(times[0]) or [])
                mesh["faceIndices"] = list(geomMesh.GetFaceVertexIndicesAttr().Get(times[0]) or [])
            scene.meshes[name] = mesh
        elif prim.IsA(UsdGeom.Camera):
            camera = UsdGeom.Camera(prim)
            clipping = sampled(lambda t: np.array(camera.GetClippingRangeAttr().Get(t) or (0.1, 10000.0), dtype=np.float64))
            scene.cameras[name] = {
                "matrix": sampled(lambda t: worldMatrix(prim, t)),
                "focalLength": sampled(lambda t: float(camera.GetFocalLengthAttr().Get(t) or 50.0)),
                "horizontalAperture": sampled(lambda t: float(camera.GetHorizontalApertureAttr().Get(t) or 20.955)),
                "verticalAperture": sampled(lambda t: float(camera.GetVerticalApertureAttr().Get(t) or 15.2908)),
                "near": _component(clipping, 0), "far": _component(clipping, 1)}
    return scene


def _same(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and np.allclose(a, b)
    return a == b


def _component(samples, index):
    if isinstance(samples, dict):
        return {frame: float(value[index]) for frame, value in samples.items()}
    return float(samples[index])


class _UsdaParser:
    """
    A reader of the ASCII USD files without the USD module: the prims (Xform, Mesh, Points, Camera...), their
    attributes (static values and time samples) and their transforms (xformOp:transform, translate, rotateXYZ...,
    scale, in xformOpOrder). Composition (references, payloads, variants) is not supported.
    """
    _TOKEN = re.compile(r'"""[\s\S]*?"""|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|<[^>]*>|#[^\n]*|'
                        r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[A-Za-z_][\w:.\-]*|[()\[\]{}=,]|\S')

    def __init__(self, text: str):
        self.tokens = [t for t in self._TOKEN.findall(text) if not t.startswith("#")]
        self.position = 0

    def peek(self, offset=0):
        index = self.position + offset
        return self.tokens[index] if index < len(self.tokens) else None

    def next(self):
        token = self.peek()
        self.position += 1
        return token

    def expect(self, token):
        value = self.next()
        if value != token:
            raise RuntimeError(f"Invalid USD file: '{token}' expected, '{value}' found.")

    def skipGroup(self):
        """ Skip a ( ... ), [ ... ] or { ... } group starting at the current token. """
        pairs = {"(": ")", "[": "]", "{": "}"}
        stack = [pairs[self.next()]]
        while stack:
            token = self.next()
            if token is None:
                raise RuntimeError("Invalid USD file: unbalanced brackets.")
            if token in pairs:
                stack.append(pairs[token])
            elif token == stack[-1]:
                stack.pop()

    def value(self):
        token = self.peek()
        if token in ("(", "["):
            closing = ")" if token == "(" else "]"
            self.next()
            items = []
            while self.peek() != closing:
                if self.peek() == ",":
                    self.next()
                    continue
                items.append(self.value())
            self.next()
            return items
        if token == "{":
            self.next()
            samples = {}
            while self.peek() != "}":
                if self.peek() == ",":
                    self.next()
                    continue
                key = float(self.next())
                self.expect(":") if self.peek() == ":" else None
                samples[key] = self.value()
            self.next()
            return samples
        token = self.next()
        if token is None:
            raise RuntimeError("Invalid USD file: unexpected end.")
        try:
            return float(token)
        except ValueError:
            return token.strip('"')

    def parse(self) -> list:
        """ The root prims: [{"type", "name", "attributes": {name: value}, "children": [...]}]. """
        if self.peek() == "(":
            self.skipGroup()
        prims = []
        while self.peek() is not None:
            prim = self.prim()
            if prim is not None:
                prims.append(prim)
        return prims

    def prim(self):
        specifier = self.next()
        if specifier not in ("def", "over", "class"):
            return None
        primType = ""
        if self.peek() is not None and not self.peek().startswith('"'):
            primType = self.next()
        name = self.next().strip('"')
        if self.peek() == "(":
            self.skipGroup()
        self.expect("{")
        prim = {"type": primType, "name": name, "attributes": {}, "children": []}
        while self.peek() not in ("}", None):
            token = self.peek()
            if token in ("def", "over", "class"):
                child = self.prim()
                if child is not None:
                    prim["children"].append(child)
                continue
            if token == "variantSet":
                while self.peek() != "{":
                    self.next()
                self.skipGroup()
                continue
            self.attribute(prim["attributes"])
        self.next()
        return prim

    def attribute(self, attributes: dict):
        """ One property line: [custom] [uniform] type name [= value] [( metadata )]. """
        words = []
        while self.peek() not in ("=", "(", "}", None) and not (words and self._startsProperty()):
            words.append(self.next())
        if not words:
            self.skipGroup() if self.peek() == "(" else self.next()
            return
        name = words[-1]
        if self.peek() == "=":
            self.next()
            value = self.value()
            if not name.endswith(".connect") and words[0] != "rel":
                attributes[name] = value
        if self.peek() == "(":
            self.skipGroup()

    def _startsProperty(self) -> bool:
        return self.peek() in ("custom", "uniform", "rel", "def", "over", "class")


_ROTATE_OPS = {"xformOp:rotate" + order: order for order in ROTATION_ORDERS}


def _opMatrix(name: str, value) -> Optional[np.ndarray]:
    """ The matrix of a USD xformOp value (column vectors). """
    base = name.split(":")
    opName = ":".join(base[:2])
    if opName == "xformOp:transform":
        return _usdMatrix(value)
    if opName == "xformOp:translate":
        return composeTRS(translate=value)
    if opName == "xformOp:scale":
        return composeTRS(scale=value)
    if opName in _ROTATE_OPS:
        return composeTRS(rotate=value, order=_ROTATE_OPS[opName])
    if opName in ("xformOp:rotateX", "xformOp:rotateY", "xformOp:rotateZ"):
        m = np.eye(4)
        m[:3, :3] = axisRotation("XYZ".index(opName[-1]), float(value))
        return m
    if opName == "xformOp:orient":
        w, x, y, z = (float(v) for v in value)
        m = np.eye(4)
        m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        return m
    return None


def _primLocalMatrix(attributes: dict):
    """ The local transform of a prim, static or {frame: matrix}, from its xformOpOrder. """
    order = attributes.get("xformOpOrder") or []
    if not isinstance(order, list):
        order = [order]
    ops = []
    frames = set()
    for opName in order:
        inverse = opName.startswith("!invert!")
        opName = opName.replace("!invert!", "")
        value = attributes.get(opName + ".timeSamples", attributes.get(opName))
        if value is None:
            continue
        if isinstance(value, dict):
            frames.update(value)
        ops.append((opName, value, inverse))
    if not ops:
        return None

    def at(frame):
        m = np.eye(4)
        for opName, value, inverse in ops:
            op = _opMatrix(opName, sampleAt(_arraySamples(value), frame) if isinstance(value, dict) else np.asarray(value))
            if op is None:
                continue
            m = m @ (np.linalg.inv(op) if inverse else op)
        return m
    if not frames:
        return at(0.0)
    return {frame: at(frame) for frame in sorted(frames)}


def _arraySamples(samples: dict) -> dict:
    return {frame: np.asarray(value, dtype=np.float64) for frame, value in samples.items()}


def _combine(parent, local):
    """ parent @ local, each static or sampled. """
    if parent is None:
        return local
    if local is None:
        return parent
    if not isinstance(parent, dict) and not isinstance(local, dict):
        return parent @ local
    frames = sorted(set(parent if isinstance(parent, dict) else []) | set(local if isinstance(local, dict) else []))
    return {frame: sampleAt(parent, frame) @ sampleAt(local, frame) for frame in frames}


def _attributeSamples(attributes: dict, name: str, convert):
    if name + ".timeSamples" in attributes and isinstance(attributes[name + ".timeSamples"], dict):
        return {frame: convert(value) for frame, value in attributes[name + ".timeSamples"].items()}
    if name in attributes:
        return convert(attributes[name])
    return None


def _readUsdAscii(path: str) -> SceneFile:
    with open(path, "r", errors="replace") as file:
        text = file.read()
    if not text.startswith("#usda"):
        raise RuntimeError("Not an ASCII USD file.")
    prims = _UsdaParser(text.split("\n", 1)[1] if "\n" in text else "").parse()
    scene = SceneFile()

    def visit(prim, parentPath, parentMatrix):
        path = f"{parentPath}/{prim['name']}"
        attributes = prim["attributes"]
        matrix = _combine(parentMatrix, _primLocalMatrix(attributes))
        if prim["type"] in ("Mesh", "Points"):
            vertices = _attributeSamples(attributes, "points", lambda v: np.array(v, dtype=np.float64).reshape(-1, 3))
            mesh = {"vertices": vertices if vertices is not None else np.zeros((0, 3)), "matrix": matrix}
            if prim["type"] == "Mesh":
                counts = _attributeSamples(attributes, "faceVertexCounts", lambda v: [int(c) for c in v])
                indices = _attributeSamples(attributes, "faceVertexIndices", lambda v: [int(i) for i in v])
                mesh["faceCounts"] = sampleAt(counts, 0) if isinstance(counts, dict) else (counts or [])
                mesh["faceIndices"] = sampleAt(indices, 0) if isinstance(indices, dict) else (indices or [])
            scene.meshes[path] = mesh
        elif prim["type"] == "Camera":
            clipping = _attributeSamples(attributes, "clippingRange", lambda v: np.array(v, dtype=np.float64))
            camera = {"matrix": matrix if matrix is not None else np.eye(4)}
            for key, default in (("focalLength", 50.0), ("horizontalAperture", 20.955), ("verticalAperture", 15.2908)):
                value = _attributeSamples(attributes, key, float)
                camera[key] = value if value is not None else default
            if clipping is not None:
                camera["near"], camera["far"] = _component(clipping, 0), _component(clipping, 1)
            scene.cameras[path] = camera
        for child in prim["children"]:
            visit(child, path, matrix)

    for prim in prims:
        visit(prim, "", None)
    return scene


# Alembic (PyAlembic module) and FBX (Autodesk FBX SDK Python bindings), when installed

def _usdReadsAlembic() -> bool:
    """ Whether USD (pxr) is installed with its Alembic file format plugin (usdAbc). """
    try:
        from pxr import Sdf
        return Sdf.FileFormat.FindByExtension("abc") is not None
    except Exception:
        return False


def _readAlembic(path: str, fps: float) -> SceneFile:
    try:
        from alembic import Abc, AbcGeom
    except ImportError as e:
        # Without PyAlembic: through USD when it has its Alembic plugin (usdAbc)
        if _usdReadsAlembic():
            return _readUsdPxr(path)
        # The reason shown: missing, or installed but not loading (a library or numpy of another version)
        raise RuntimeError(f"Reading Alembic files needs the Alembic Python module (PyAlembic) or USD with its Alembic "
                           f"plugin, which cannot be imported here ({e}): export the scene as USD, or use a Meshroom "
                           f"environment with Alembic.") from e
    archive = Abc.IArchive(path)
    scene = SceneFile()

    def sampleTimes(schema):
        sampling = schema.getTimeSampling()
        count = schema.getNumSamples()
        return [sampling.getSampleTime(i) for i in range(count)]

    def framesOf(times):
        return [round(t * fps, 6) for t in times]

    def visit(obj, parentMatrix):
        header = obj.getHeader()
        matrix = parentMatrix
        if AbcGeom.IXform.matches(header):
            schema = AbcGeom.IXform(obj, Abc.WrapExistingFlag.kWrapExisting).getSchema()
            times = sampleTimes(schema)
            local = {}
            for time, frame in zip(times, framesOf(times)):
                sample = schema.getValue(Abc.ISampleSelector(time))
                local[frame] = np.array([[sample.getMatrix()[r][c] for c in range(4)] for r in range(4)]).T
            local = local[next(iter(local))] if len(local) == 1 else local
            matrix = _combine(parentMatrix, local)
        elif AbcGeom.IPolyMesh.matches(header) or AbcGeom.IPoints.matches(header):
            isMesh = AbcGeom.IPolyMesh.matches(header)
            schema = (AbcGeom.IPolyMesh if isMesh else AbcGeom.IPoints)(obj, Abc.WrapExistingFlag.kWrapExisting).getSchema()
            times = sampleTimes(schema)
            vertices, counts, indices = {}, [], []
            for index, (time, frame) in enumerate(zip(times, framesOf(times))):
                sample = schema.getValue(Abc.ISampleSelector(time))
                vertices[frame] = np.array([(p[0], p[1], p[2]) for p in sample.getPositions()], dtype=np.float64)
                if isMesh and index == 0:
                    counts = list(sample.getFaceCounts())
                    # Alembic faces are clockwise: reversed to the counter-clockwise order of USD
                    rawIndices = list(sample.getFaceIndices())
                    position = 0
                    for count in counts:
                        indices.extend(reversed(rawIndices[position:position + count]))
                        position += count
            scene.meshes[obj.getFullName()] = {"vertices": vertices[next(iter(vertices))] if len(vertices) == 1 else vertices,
                                               "faceCounts": counts, "faceIndices": indices, "matrix": matrix}
        elif AbcGeom.ICamera.matches(header):
            schema = AbcGeom.ICamera(obj, Abc.WrapExistingFlag.kWrapExisting).getSchema()
            times = sampleTimes(schema)
            values = {key: {} for key in ("focalLength", "horizontalAperture", "verticalAperture", "near", "far")}
            for time, frame in zip(times, framesOf(times)):
                sample = schema.getValue(Abc.ISampleSelector(time))
                values["focalLength"][frame] = sample.getFocalLength()
                # Alembic apertures are in centimeters
                values["horizontalAperture"][frame] = sample.getHorizontalAperture() * 10.0
                values["verticalAperture"][frame] = sample.getVerticalAperture() * 10.0
                values["near"][frame] = sample.getNearClippingPlane()
                values["far"][frame] = sample.getFarClippingPlane()
            camera = {key: (next(iter(samples.values())) if len(set(samples.values())) == 1 else samples)
                      for key, samples in values.items() if samples}
            camera["matrix"] = matrix if matrix is not None else np.eye(4)
            scene.cameras[obj.getFullName()] = camera
        for i in range(obj.getNumChildren()):
            visit(obj.getChild(i), matrix)

    visit(archive.getTop(), None)
    return scene


def _readFbx(path: str, fps: float) -> SceneFile:
    try:
        import fbx
    except ImportError:
        raise RuntimeError("Reading FBX files needs the Autodesk FBX SDK Python module (fbx), not available here: "
                           "export the scene as USD, or use a Meshroom environment with the FBX SDK.")
    manager = fbx.FbxManager.Create()
    try:
        importer = fbx.FbxImporter.Create(manager, "")
        if not importer.Initialize(path, -1, manager.GetIOSettings()):
            raise RuntimeError(f"The FBX SDK cannot open {path}.")
        fbxScene = fbx.FbxScene.Create(manager, "scene")
        importer.Import(fbxScene)
        importer.Destroy()
        timeSpan = fbxScene.GetGlobalSettings().GetTimelineDefaultTimeSpan()
        mode = fbx.FbxTime.EMode.eCustom if hasattr(fbx.FbxTime, "EMode") else fbx.FbxTime.eCustom
        start, stop = timeSpan.GetStart(), timeSpan.GetStop()
        firstFrame = int(round(start.GetSecondDouble() * fps))
        lastFrame = int(round(stop.GetSecondDouble() * fps))
        frames = list(range(firstFrame, max(firstFrame, lastFrame) + 1))
        del mode

        def timeOf(frame):
            time = fbx.FbxTime()
            time.SetSecondDouble(frame / float(fps))
            return time

        def globalMatrix(node, frame):
            m = node.EvaluateGlobalTransform(timeOf(frame))
            return np.array([[m.Get(r, c) for c in range(4)] for r in range(4)]).T

        def samples(function):
            values = {frame: function(frame) for frame in frames}
            first = values[frames[0]]
            return first if all(_same(first, value) for value in values.values()) else values

        scene = SceneFile()
        # FBX cameras look along +X (Y up): turned to look along -Z
        cameraFix = composeTRS(rotate=(0.0, 90.0, 0.0))
        stack = [fbxScene.GetRootNode()]
        while stack:
            node = stack.pop()
            for i in range(node.GetChildCount() - 1, -1, -1):
                stack.append(node.GetChild(i))
            attribute = node.GetNodeAttribute()
            if attribute is None:
                continue
            kind = attribute.GetAttributeType()
            if kind == fbx.FbxNodeAttribute.EType.eMesh:
                mesh = node.GetMesh()
                points = mesh.GetControlPoints()
                vertices = np.array([(p[0], p[1], p[2]) for p in points], dtype=np.float64)
                counts, indices = [], []
                for polygon in range(mesh.GetPolygonCount()):
                    size = mesh.GetPolygonSize(polygon)
                    counts.append(size)
                    indices.extend(mesh.GetPolygonVertex(polygon, k) for k in range(size))
                scene.meshes[node.GetName()] = {"vertices": vertices, "faceCounts": counts, "faceIndices": indices,
                                                "matrix": samples(lambda f: globalMatrix(node, f))}
            elif kind == fbx.FbxNodeAttribute.EType.eCamera:
                camera = node.GetCamera()
                scene.cameras[node.GetName()] = {
                    "matrix": samples(lambda f: globalMatrix(node, f) @ cameraFix),
                    "focalLength": samples(lambda f: float(camera.FocalLength.EvaluateValue(timeOf(f)))),
                    # FBX film sizes are in inches
                    "horizontalAperture": float(camera.FilmWidth.Get()) * 25.4,
                    "verticalAperture": float(camera.FilmHeight.Get()) * 25.4,
                    "near": float(camera.NearPlane.Get()), "far": float(camera.FarPlane.Get())}
        return scene
    finally:
        manager.Destroy()


def availableFormats() -> Dict[str, bool]:
    """ Whether the optional modules reading USD (binary), Alembic and FBX files are installed. """
    result = {}
    for name, module in (("usd", "pxr"), ("abc", "alembic"), ("fbx", "fbx")):
        try:
            __import__(module)
            result[name] = True
        except ImportError:
            result[name] = False
        except Exception:
            logging.debug("[3D] %s cannot be imported", module, exc_info=True)
            result[name] = False
    return result
