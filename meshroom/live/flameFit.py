"""
FLAME 2023 Open (MPI-IS, CC-BY-4.0: commercial use allowed, with attribution) fitted on the face of the FaceBuilder:
the whole head (skull, ears, neck) that goes with the face, from the statistics of the FLAME scans, without any network.

    loadModel: FLAME (its pickle) and its MediaPipe landmark embedding, from the models folder (MESHROOM_MODELS_PATH/flame);
    fitFlame: its shape (all its directions: identity and expression, unit variance) and a similarity put on the face
        fitted on the photos: its MediaPipe landmarks on theirs, then each vertex of the face on the nearest point of
        FLAME (more correspondences), regularized (the coefficients toward 0: a likely head).

The head is then given to "Fit To AI Head" like a generated one (aiHead: placed by its landmarks, the FaceBuilder head
registered and warped on it; FLAME has no hair: its skull is used too).
"""
import glob
import os
from typing import Optional

import numpy as np

from . import face


class FlameUnavailable(RuntimeError):
    pass


class FlameModel:
    """ FLAME: its template (n, 3) and directions (n, 3, k) in meters, its triangles, and where the MediaPipe landmarks
    are on it (triangles (l,), barycentric coordinates (l, 3), landmark indices (l,)). """

    def __init__(self, template, directions, triangles, embedding):
        self.template = np.asarray(template, dtype=np.float64)
        self.directions = np.asarray(directions, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self.embeddingFaces, self.embeddingWeights, self.embeddingIndices = embedding

    def vertices(self, coefficients) -> np.ndarray:
        return self.template + self.directions[:, :, :len(coefficients)] @ np.asarray(coefficients, dtype=np.float64)

    def landmarkRows(self):
        """ The landmarks as linear functions of the vertices: (corner vertices (l, 3), weights (l, 3)). """
        return self.triangles[self.embeddingFaces], self.embeddingWeights


def modelFolder() -> str:
    return os.path.join(os.environ.get("MESHROOM_MODELS_PATH", ""), "flame")


def loadModel(folder: Optional[str] = None) -> FlameModel:
    """
    FLAME from a folder (default: MESHROOM_MODELS_PATH/flame): its pickle (flame2023_Open.pkl, or any FLAME .pkl) and
    mediapipe_landmark_embedding.npz (there or in the mica folder), both from https://flame.is.tue.mpg.de.
    """
    from .neural import micaRunner
    folder = folder or modelFolder()
    pickles = sorted(glob.glob(os.path.join(folder, "*.pkl")), key=lambda p: ("open" not in p.lower(), p))
    if not pickles:
        raise FlameUnavailable(f"FLAME: no model in {folder} (flame2023_Open.pkl from https://flame.is.tue.mpg.de, "
                               "'FLAME 2023 Open', CC-BY-4.0)")
    embeddingPaths = [os.path.join(folder, "mediapipe_landmark_embedding.npz"),
                      os.path.join(os.path.dirname(folder), "mica", "mediapipe_landmark_embedding.npz")]
    embeddingPath = next((path for path in embeddingPaths if os.path.isfile(path)), None)
    if embeddingPath is None:
        raise FlameUnavailable(f"FLAME: no mediapipe_landmark_embedding.npz in {folder} ('FLAME Mediapipe Landmark "
                               "Embedding', https://flame.is.tue.mpg.de)")
    try:
        template, directions, triangles = micaRunner.loadFlame(pickles[0], components=None)
        embedding = micaRunner.readMediapipeEmbedding(embeddingPath)
    except (OSError, KeyError, ValueError) as e:
        raise FlameUnavailable(f"FLAME: cannot read {pickles[0]} / {embeddingPath}: {e}") from e
    return FlameModel(template, directions, triangles, embedding)


def _similarity(source, target, weights):
    """ (scale, rotation, translation): target ~ scale * rotation @ source + translation, weighted (Umeyama). """
    w = weights / weights.sum()
    ms, mt = w @ source, w @ target
    s, t = source - ms, target - mt
    u, sigma, vt = np.linalg.svd((t * w[:, None]).T @ s)
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ d @ vt
    scale = float(np.trace(np.diag(sigma) @ d) / max((w * (s ** 2).sum(1)).sum(), 1e-18))
    return scale, rotation, mt - scale * rotation @ ms


def _vertexNormals(vertices, triangles):
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    normals = np.zeros_like(vertices)
    faceNormals = np.cross(b - a, c - a)
    for i in range(3):
        np.add.at(normals, triangles[:, i], faceNormals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-18)


def fitFlame(model: FlameModel, faceTarget: np.ndarray, iterations: int = 12, regularization: float = 0.1,
             surfaceFrom: int = 3, maxDistance: float = 1.0, cancel=None, progress=None) -> dict:
    """
    FLAME fitted on a face (468, 3) of the FaceBuilder (its space, centimeters): a similarity and all its coefficients
    (unit variance), alternately; the MediaPipe landmarks of FLAME on the face's, then (from iteration `surfaceFrom`)
    each vertex of the face on the surface of FLAME (along the normal of its nearest vertex, within `maxDistance` cm) too;
    `regularization`: the weight of a coefficient of 1 against 1 mm of error at each correspondence.

    Returns:
        {"vertices": (n, 3) the head in the space of the face (cm), "triangles", "landmarks": (468, 3) (NaN where
        FLAME has none), "coefficients", "error": the RMS error of the correspondences (cm)}.
    """
    from scipy.spatial import cKDTree
    faceTarget = np.asarray(faceTarget, dtype=np.float64)[:face.MESH_VERTICES]
    corners, weights = model.landmarkRows()
    indices = model.embeddingIndices
    count = model.directions.shape[2]
    coefficients = np.zeros(count)
    scale, rotation, translation = 100.0, np.eye(3), np.zeros(3)
    error = 0.0
    landmarkTargets = faceTarget[indices]
    for iteration in range(iterations):
        if cancel:
            cancel()
        vertices = model.vertices(coefficients)
        # The placement: on the landmarks
        landmarkPoints = (vertices[corners] * weights[:, :, None]).sum(1)
        scale, rotation, translation = _similarity(landmarkPoints, landmarkTargets, np.ones(len(indices)))
        # The coefficients for that placement (in the FLAME space, meters): the landmarks (3D), then the face on the
        # surface of FLAME (along its normal at the nearest vertex: free to slide), ridge least squares
        wanted = (landmarkTargets - translation) @ rotation / scale - (model.template[corners] * weights[:, :, None]).sum(1)
        rows = [(model.directions[corners] * weights[:, :, None, None]).sum(1).reshape(-1, count)]
        right = [wanted.reshape(-1)]
        pairs = len(indices)
        if iteration >= surfaceFrom:
            placed = scale * vertices @ rotation.T + translation
            distance, nearest = cKDTree(placed).query(faceTarget)
            close = distance < maxDistance
            normals = _vertexNormals(vertices, model.triangles)[nearest[close]]
            local = (faceTarget[close] - translation) @ rotation / scale
            along = ((local - model.template[nearest[close]]) * normals).sum(1)
            basis = np.einsum("pc,pck->pk", normals, model.directions[nearest[close]])
            root = np.sqrt(0.5)
            rows.append(root * basis)
            right.append(root * along)
            pairs += 0.5 * int(close.sum())
        a, b = np.concatenate(rows), np.concatenate(right)
        # 1 mm (in meters) of error per correspondence against a coefficient of 1
        damping = regularization * (1e-3 ** 2) * pairs
        coefficients = np.linalg.solve(a.T @ a + damping * np.eye(count), a.T @ b)
        error = float(np.sqrt(np.mean((a @ coefficients - b) ** 2)) * scale)
        if progress:
            progress((iteration + 1) / float(iterations))
    vertices = scale * model.vertices(coefficients) @ rotation.T + translation
    landmarks = np.full((face.MESH_VERTICES, 3), np.nan)
    landmarks[indices] = (vertices[corners] * weights[:, :, None]).sum(1)
    return {"vertices": vertices, "triangles": model.triangles, "landmarks": landmarks, "coefficients": coefficients,
            "error": error}
