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
            "error": error, "similarity": (scale, rotation, translation)}


def _rotation(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    k = np.asarray(vector) / angle
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def _reprojection(views, points, indices):
    """ The reprojection residuals (rows, 2) of landmark points (l, 3) (face space) in the views, the rows that count
    (each view's visible landmarks among `indices`), and their jacobians (rows, 2, 3) and landmark of each row. """
    from .headFit import _projection
    residuals, jacobians, which = [], [], []
    for view in views:
        width, height = view["size"]
        pixels = view["pixels"]
        hidden = set(int(i) for i in view.get("hidden") or [])
        keep = np.array([i < len(pixels) and int(i) not in hidden for i in indices])
        if not keep.any():
            continue
        xy, depth, jacobian = _projection(view["camera"], points[keep], width, height)
        residuals.append(xy - pixels[indices[keep]])
        jacobians.append(jacobian)
        which.append(np.where(keep)[0])
    return np.concatenate(residuals), np.concatenate(jacobians), np.concatenate(which)


def fitFlameToViews(model: FlameModel, views, start: dict, faceTarget: Optional[np.ndarray] = None, iterations: int = 10,
                    regularization: float = 1.0, faceWeight: float = 5.0, surfaceWeight: float = 10.0,
                    exact: bool = True, radius: float = 2.0, cancel=None, progress=None) -> dict:
    """
    FLAME fitted on the landmarks of the photos (their reprojection in every view: what makes it fit the images), from
    `start` (fitFlame on the face of the photos): its coefficients and its placement (similarity) by Gauss-Newton,
    regularized (`regularization`: a coefficient of 1 against 1 pixel of error per landmark and view); then, with
    `exact`, its surface moved around each landmark (gaussians of `radius` cm, solved together) so that each lands on
    its pixel in every view. `faceTarget` (the face of the photos, 468 x 3, fitted on all the landmarks of the views):
    its landmarks held near it too (`faceWeight` pixels per centimeter: what the views do not see, e.g. the depth with
    a single view), the whole face on the surface of FLAME (`surfaceWeight` pixels per centimeter), and, with `exact`,
    the face of FLAME then warped onto it (onPhotoFace): FLAME on the face of the images everywhere, not only at its
    landmarks.

    Args:
        views: [{"camera", "size": (w, h), "pixels": (n, 2) the landmarks (Nuke pixels), "hidden": [indices]}].

    Returns:
        fitFlame's dict, with "pixels": the RMS reprojection error (pixels) before and after.
    """
    corners, weights = model.landmarkRows()
    indices = model.embeddingIndices
    count = model.directions.shape[2]
    scale0, rotation0, translation0 = start["similarity"]
    coefficients = np.asarray(start["coefficients"], dtype=np.float64).copy()
    pose = np.zeros(7)  # rotation vector, translation (cm), log of the scale

    def landmarkPoints(c, p):
        local = (model.vertices(c)[corners] * weights[:, :, None]).sum(1)
        rotation = _rotation(p[:3]) @ rotation0
        return scale0 * np.exp(p[6]) * local @ rotation.T + translation0 + p[3:6], rotation

    points, rotation = landmarkPoints(coefficients, pose)
    residuals, _, _ = _reprojection(views, points, indices)
    before = float(np.sqrt((residuals ** 2).sum(1).mean()))
    basis = (model.directions[corners] * weights[:, :, None, None]).sum(1)          # (l, 3, k) meters
    for iteration in range(iterations):
        if cancel:
            cancel()
        points, rotation = landmarkPoints(coefficients, pose)
        residuals, jacobians, which = _reprojection(views, points, indices)
        scale = scale0 * np.exp(pose[6])
        # The coefficients: d xy / d c = J (2, 3) . scale R basis (3, k)
        rowsC = np.einsum("rij,jk,rkm->rim", jacobians, scale * rotation, basis[which])
        # The placement: numerically
        rowsP = np.zeros((len(residuals), 2, 7))
        for k in range(7):
            step = np.zeros(7)
            step[k] = 1e-4
            moved, _ = landmarkPoints(coefficients, pose + step)
            movedResiduals, _, _ = _reprojection(views, moved, indices)
            rowsP[:, :, k] = (movedResiduals - residuals) / 1e-4
        a = np.concatenate([rowsC, rowsP], axis=2).reshape(-1, count + 7)
        b = -residuals.reshape(-1)
        if faceTarget is not None:
            # The landmarks near those of the face of the photos (in 3D)
            offsets = points - np.asarray(faceTarget, dtype=np.float64)[indices]
            rows3C = faceWeight * np.einsum("ij,ljk->lik", scale * rotation, basis)
            rows3P = np.zeros((len(points), 3, 7))
            for k in range(7):
                step = np.zeros(7)
                step[k] = 1e-4
                moved, _ = landmarkPoints(coefficients, pose + step)
                rows3P[:, :, k] = faceWeight * (moved - points) / 1e-4
            a = np.concatenate([a, np.concatenate([rows3C, rows3P], axis=2).reshape(-1, count + 7)])
            b = np.concatenate([b, -faceWeight * offsets.reshape(-1)])
            # The whole face of the photos on the surface of FLAME (not only its landmarks: its width, its jaw...),
            # point to plane at the nearest FLAME vertex
            rowsS, rightS = _surfaceRows(model, coefficients, scale, rotation, translation0 + pose[3:6],
                                         np.asarray(faceTarget, dtype=np.float64), surfaceWeight)
            if len(rowsS):
                a = np.concatenate([a, np.concatenate([rowsS, np.zeros((len(rowsS), 7))], axis=1)])
                b = np.concatenate([b, rightS])
        damping = np.r_[np.full(count, regularization * len(residuals)), np.full(7, 1e-6)]
        delta = np.linalg.solve(a.T @ a + np.diag(damping), a.T @ b - damping * np.r_[coefficients, np.zeros(7)])
        coefficients += delta[:count]
        pose += delta[count:]
        if progress:
            progress(0.8 * (iteration + 1) / float(iterations))
        if np.abs(delta).max() < 1e-5:
            break
    rotation = _rotation(pose[:3]) @ rotation0
    scale = scale0 * np.exp(pose[6])
    translation = translation0 + pose[3:6]
    vertices = scale * model.vertices(coefficients) @ rotation.T + translation
    if exact:
        if faceTarget is not None:
            vertices = onPhotoFace(vertices, model.triangles, np.asarray(faceTarget, dtype=np.float64))
        # (what the warp left: each landmark on its pixel)
        vertices = _exactLandmarks(vertices, model, views, radius)
    landmarks = np.full((face.MESH_VERTICES, 3), np.nan)
    landmarks[indices] = (vertices[corners] * weights[:, :, None]).sum(1)
    residuals, _, _ = _reprojection(views, landmarks[indices], indices)
    after = float(np.sqrt((residuals ** 2).sum(1).mean()))
    if progress:
        progress(1.0)
    return {"vertices": vertices, "triangles": model.triangles, "landmarks": landmarks, "coefficients": coefficients,
            "error": start.get("error", 0.0), "similarity": (scale, rotation, translation), "pixels": (before, after)}


def _photoFaceGeometry(faceTarget):
    """ (triangles of the face of the FaceBuilder, its oval vertices). """
    head = face.canonicalHead()
    triangles = head.triangles()
    faceTriangles = triangles[(triangles < face.MESH_VERTICES).all(1)]
    oval = np.unique(np.asarray(face.headData()["holeEdges"]).ravel())
    return faceTriangles, oval[oval < face.MESH_VERTICES]


def _surfaceRows(model, coefficients, scale, rotation, translation, faceTarget, weight, maxDistance=1.5):
    """ Rows (k, coefficients) and right sides (k,) of the face of the photos on the surface of FLAME placed by the
    similarity: each face vertex (not on the oval) along the normal of its nearest FLAME vertex, in centimeters. """
    from scipy.spatial import cKDTree
    vertices = model.vertices(coefficients)
    placed = scale * vertices @ rotation.T + translation
    _, oval = _photoFaceGeometry(faceTarget)
    inner = np.setdiff1d(np.arange(len(faceTarget)), oval)
    distance, nearest = cKDTree(placed).query(faceTarget[inner])
    close = distance < maxDistance
    if not close.any():
        return np.zeros((0, len(coefficients))), np.zeros(0)
    nearest = nearest[close]
    normals = _vertexNormals(placed, model.triangles)[nearest]
    residual = ((faceTarget[inner][close] - placed[nearest]) * normals).sum(1)
    rows = np.einsum("pi,ij,pjk->pk", normals, scale * rotation, model.directions[nearest])
    return weight * rows, weight * residual


def onPhotoFace(vertices, triangles, faceTarget, maxDistance=1.5):
    """ The FLAME head with its face warped onto the face of the photos (aiHead.warpOntoSurface: its vertices near the
    face, not near its oval, to their nearest points; the rest of the head carried smoothly). """
    from scipy.spatial import cKDTree
    from .aiHead import warpOntoSurface
    faceTriangles, oval = _photoFaceGeometry(faceTarget)
    distance, nearest = cKDTree(faceTarget).query(vertices)
    data = ((distance < maxDistance) & ~np.isin(nearest, oval)).astype(np.float64)
    moves = warpOntoSurface(vertices, triangles, faceTarget, faceTriangles, data, np.zeros(len(vertices)),
                            stiffness=(10.0, 4.0, 1.5, 0.6), maxDistance=maxDistance)
    return vertices + moves


def _exactLandmarks(vertices, model, views, radius, iterations=3):
    """ The surface of FLAME moved so that its landmarks land on their pixels in every view: a gaussian move around
    each landmark (`radius` cm, normalized: moves of neighbors do not add up), all solved together (least squares on
    the reprojections, a little damped: a landmark seen in one view moves across it). """
    corners, weights = model.landmarkRows()
    indices = model.embeddingIndices
    points = (vertices[corners] * weights[:, :, None]).sum(1)
    # The weight of each landmark's move at each vertex (and at each landmark)

    def kernel(at):
        g = np.exp(-((at[:, None, :] - points[None, :, :]) ** 2).sum(-1) / radius ** 2)
        return g / np.maximum(g.sum(1, keepdims=True), 1.0)
    atVertices, atLandmarks = kernel(vertices), kernel(points)
    moves = np.zeros_like(points)
    for _ in range(iterations):
        current = points + atLandmarks @ moves
        residuals, jacobians, which = _reprojection(views, current, indices)
        # d residual / d moves: J (2, 3) at the landmark of the row times its kernel weights over the landmarks
        rows = np.einsum("rij,rl->rilj", jacobians, atLandmarks[which]).reshape(len(residuals) * 2, -1)
        damping = 1e-2 * np.eye(rows.shape[1])
        moves = moves + np.linalg.solve(rows.T @ rows + damping, -rows.T @ residuals.reshape(-1)).reshape(-1, 3)
    return vertices + atVertices @ moves
