"""
FLAME 2023 Open (MPI-IS, CC-BY-4.0: commercial use allowed, with attribution) fitted on the face of the FaceBuilder:
the whole head (skull, ears, neck) that goes with the face, from the statistics of the FLAME scans, without any network.

    loadModel: FLAME (its pickle) and its MediaPipe landmark embedding, from the models folder (MESHROOM_MODELS_PATH/flame);
    fitFlame: its shape (all its directions: identity and expression, unit variance) and a similarity put on the face
        fitted on the photos: its MediaPipe landmarks on theirs, then each vertex of the face on the nearest point of
        FLAME (more correspondences), regularized (the coefficients toward 0: a likely head): only where it starts;
    fitFlameToViews: FLAME on its own on the landmarks of the images (their reprojection), its jaw and its move in each
        view free: nothing of the FaceBuilder face holds it.

The FaceBuilder head is then warped onto it ("Fit To AI Head": aiHead, FLAME in the face space, its moves in the views
those of the cameras of the head; FLAME has no hair: its skull is used too).
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

    def __init__(self, template, directions, triangles, embedding, jaw=None):
        self.template = np.asarray(template, dtype=np.float64)
        self.directions = np.asarray(directions, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self.embeddingFaces, self.embeddingWeights, self.embeddingIndices = embedding
        # The jaw (optional): its skinning weights (n,) and the regressor of its joint (n,)
        self.jawWeights, self.jawRegressor = (None, None) if jaw is None else \
            (np.asarray(jaw[0], dtype=np.float64), np.asarray(jaw[1], dtype=np.float64))

    def vertices(self, coefficients) -> np.ndarray:
        return self.template + self.directions[:, :, :len(coefficients)] @ np.asarray(coefficients, dtype=np.float64)

    def jawJoint(self, vertices) -> np.ndarray:
        return self.jawRegressor @ vertices

    def posed(self, vertices, jaw) -> np.ndarray:
        """ The vertices with the jaw turned by the rotation vector `jaw` around its joint (linear blend skinning, the
        other joints at rest). """
        if self.jawWeights is None or not np.any(jaw):
            return vertices
        joint = self.jawJoint(vertices)
        turned = (vertices - joint) @ _rotation(jaw).T + joint
        return vertices + self.jawWeights[:, None] * (turned - vertices)

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
        jaw = micaRunner.loadFlameJaw(pickles[0])
    except (OSError, KeyError, ValueError) as e:
        raise FlameUnavailable(f"FLAME: cannot read {pickles[0]} / {embeddingPath}: {e}") from e
    return FlameModel(template, directions, triangles, embedding, jaw=jaw)


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


def _poseMatrix(pose, center):
    """ The 4x4 move of a view (rotation vector pose[:3] around `center`, then translation pose[3:6]). """
    rotation = _rotation(pose[:3])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = center - rotation @ center + pose[3:6]
    return matrix


def _visible(view, indices):
    pixels = view["pixels"]
    hidden = set(int(i) for i in view.get("hidden") or [])
    return np.array([i < len(pixels) and int(i) not in hidden for i in indices])


def _reprojection(views, points, indices):
    """ The reprojection residuals (rows, 2) of landmark points (l, 3) (face space) in the views, the rows that count
    (each view's visible landmarks among `indices`), and their jacobians (rows, 2, 3) and landmark of each row. """
    from .headFit import _projection
    residuals, jacobians, which = [], [], []
    for view in views:
        width, height = view["size"]
        keep = _visible(view, indices)
        if not keep.any():
            continue
        xy, depth, jacobian = _projection(view["camera"], points[keep], width, height)
        residuals.append(xy - view["pixels"][indices[keep]])
        jacobians.append(jacobian)
        which.append(np.where(keep)[0])
    return np.concatenate(residuals), np.concatenate(jacobians), np.concatenate(which)


def fitFlameToViews(model: FlameModel, views, start: dict, iterations: int = 15, regularization: float = 0.1,
                    jawRegularization: float = 5.0, viewRotation: float = 2.0, viewTranslation: float = 0.3,
                    exact: bool = True, radius: float = 2.0, cancel=None, progress=None) -> dict:
    """
    FLAME fitted on its own on the landmarks of the photos (their reprojection in every view: nothing else, no face of
    the FaceBuilder), from `start` (fitFlame: only where it begins): its coefficients (identity and expression), its
    jaw, its placement (similarity) and a rigid move of the head in each view (the cameras of the views are only
    estimates: FLAME places itself in each), by Gauss-Newton, regularized (`regularization`: a coefficient of 1 against
    1 pixel of error per landmark and view; `jawRegularization` the same for a jaw turned by 1 radian; a move of a view
    of `viewRotation` degrees or `viewTranslation` cm against 1 pixel); then, with `exact`, its surface moved around each
    landmark (gaussians of `radius` cm, solved together) so that each lands on its pixel in every view.

    Args:
        views: [{"camera", "size": (w, h), "pixels": (n, 2) the landmarks (Nuke pixels), "hidden": [indices]}].

    Returns:
        fitFlame's dict, with "pixels": the RMS reprojection error (pixels) before and after, "jaw": its rotation
        vector, "views": the 4x4 move of the head (face space) in each view (the camera of a view sees the head moved
        by it: camera matrix inv(move) @ matrix).
    """
    corners, weights = model.landmarkRows()
    indices = model.embeddingIndices
    count = model.directions.shape[2]
    hasJaw = model.jawWeights is not None
    scale0, rotation0, translation0 = start["similarity"]
    coefficients = np.asarray(start["coefficients"], dtype=np.float64).copy()
    jaw = np.zeros(3)
    pose = np.zeros(7)  # the placement: rotation vector, translation (cm), log of the scale
    viewPoses = np.zeros((len(views), 6))  # each view: rotation vector, translation (cm)
    jawDirections = model.jawRegressor @ model.directions.reshape(len(model.template), -1) if hasJaw else None
    if hasJaw:
        jawDirections = jawDirections.reshape(3, count)
    basis = model.directions[corners]  # (l, 3 corners, 3, k) meters
    cornerJaw = model.jawWeights[corners] if hasJaw else np.zeros(corners.shape)
    center = (scale0 * (model.vertices(coefficients)[corners] * weights[:, :, None]).sum(1) @ rotation0.T
              + translation0).mean(0)

    def local(c, j):
        """ The landmarks in the FLAME space (meters), the jaw turned. """
        vertices = model.posed(model.vertices(c), j) if hasJaw else model.vertices(c)
        return (vertices[corners] * weights[:, :, None]).sum(1)

    def placed(points, p):
        rotation = _rotation(p[:3]) @ rotation0
        return scale0 * np.exp(p[6]) * points @ rotation.T + translation0 + p[3:6], rotation

    def inView(points, viewPose):
        """ The landmarks moved as the head is in a view. """
        return (points - center) @ _rotation(viewPose[:3]).T + center + viewPose[3:6]

    def viewResiduals(points, view, keep):
        from .headFit import _projection
        width, height = view["size"]
        xy, _, jacobian = _projection(view["camera"], points[keep], width, height)
        return xy - view["pixels"][indices[keep]], jacobian

    keeps = [_visible(view, indices) for view in views]
    active = [v for v in range(len(views)) if keeps[v].any()]
    if not active:
        raise ValueError("FLAME: no landmark seen in the views")
    rows = sum(int(keeps[v].sum()) for v in active)
    perView = [max(int(keeps[v].sum()), 1) for v in range(len(views))]

    def allResiduals(c, j, p, vp):
        points, _ = placed(local(c, j), p)
        return np.concatenate([viewResiduals(inView(points, vp[v]), views[v], keeps[v])[0] for v in active])

    before = float(np.sqrt((allResiduals(coefficients, jaw, pose, viewPoses) ** 2).sum(1).mean()))
    size = count + 3 + 7 + 6 * len(views)
    step = 1e-4
    for iteration in range(iterations):
        if cancel:
            cancel()
        points0 = local(coefficients, jaw)
        points, rotation = placed(points0, pose)
        scale = scale0 * np.exp(pose[6])
        # d landmark (FLAME space) / d coefficients: the corners turned with the jaw (linear blend), and its joint moved
        if hasJaw and np.any(jaw):
            turn = _rotation(jaw)
            w = cornerJaw[:, :, None, None]
            turned = np.einsum("ij,lcjk->lcik", turn, basis)
            moved = np.einsum("ij,jk->ik", np.eye(3) - turn, jawDirections)
            cornerBasis = (1 - w) * basis + w * (turned + moved[None, None])
        else:
            cornerBasis = basis
        landmarkBasis = (cornerBasis * weights[:, :, None, None]).sum(1)  # (l, 3, k)
        blocks, right = [], []
        for v in active:
            view, keep = views[v], keeps[v]
            seen = inView(points, viewPoses[v])
            residual, jacobian = viewResiduals(seen, view, keep)
            viewRotationMatrix = _rotation(viewPoses[v, :3])
            block = np.zeros((len(residual), 2, size))
            # The coefficients: J (2, 3) . Rv s R basis (3, k)
            block[:, :, :count] = np.einsum("rij,jk,rkm->rim", jacobian, viewRotationMatrix @ (scale * rotation),
                                            landmarkBasis[keep])
            # The jaw, the placement, the move of the view: numerically
            for k in range(3 + 7 + 6):
                j, p, vp = jaw.copy(), pose.copy(), viewPoses[v].copy()
                if k < 3:
                    j[k] += step
                elif k < 10:
                    p[k - 3] += step
                else:
                    vp[k - 10] += step
                if k < 3 and not hasJaw:
                    continue
                movedPoints, _ = placed(local(coefficients, j) if k < 3 else points0, p)
                movedResidual, _ = viewResiduals(inView(movedPoints, vp), view, keep)
                column = count + k if k < 10 else count + 10 + 6 * v + (k - 10)
                block[:, :, column] = (movedResidual - residual) / step
            blocks.append(block.reshape(-1, size))
            right.append(-residual.reshape(-1))
        a, b = np.concatenate(blocks), np.concatenate(right)
        # Damping: the coefficients and the jaw toward 0 (a likely face), the moves of the views toward none (each
        # against its own landmarks), the placement free
        damping = np.zeros(size)
        damping[:count] = regularization * rows
        damping[count:count + 3] = jawRegularization * rows if hasJaw else 1.0
        # (the placement nearly free: its depth and its size, that a single view hardly tells apart, held a little:
        # 10% of size or 5 cm against 1 pixel)
        damping[count + 3:count + 9] = 1e-6
        damping[count + 6:count + 9] = rows / 5.0 ** 2
        damping[count + 9] = rows / 0.1 ** 2
        for v in range(len(views)):
            first = count + 10 + 6 * v
            damping[first:first + 3] = perView[v] / np.radians(viewRotation) ** 2
            damping[first + 3:first + 6] = perView[v] / viewTranslation ** 2
        current = np.r_[coefficients, jaw, np.zeros(7), viewPoses.ravel()]
        current[count + 3:count + 10] = 0.0
        delta = np.linalg.solve(a.T @ a + np.diag(damping), a.T @ b - damping * current)

        def cost(d):
            r = allResiduals(coefficients + d[:count], jaw + d[count:count + 3], pose + d[count + 3:count + 10],
                             viewPoses + d[count + 10:].reshape(-1, 6))
            return float((r ** 2).sum() + (damping * (current + d) ** 2).sum())
        # (a step that makes it worse: shortened)
        now = cost(np.zeros(size))
        for _ in range(4):
            if cost(delta) <= now:
                break
            delta = 0.5 * delta
        else:
            break
        coefficients += delta[:count]
        jaw += delta[count:count + 3]
        pose += delta[count + 3:count + 10]
        viewPoses += delta[count + 10:].reshape(-1, 6)
        if progress:
            progress(0.8 * (iteration + 1) / float(iterations))
        if np.abs(delta).max() < 1e-5:
            break
    rotation = _rotation(pose[:3]) @ rotation0
    scale = scale0 * np.exp(pose[6])
    translation = translation0 + pose[3:6]
    vertices = scale * model.posed(model.vertices(coefficients), jaw) @ rotation.T + translation
    moves = [_poseMatrix(viewPoses[v], center) for v in range(len(views))]
    # The views seen by their cameras with the head moved there
    posedViews = [dict(view, camera=view["camera"].copy(matrix=np.linalg.inv(move) @ view["camera"].matrix))
                  for view, move in zip(views, moves)]
    if exact:
        # (what FLAME could not: each landmark on its pixel)
        vertices = _exactLandmarks(vertices, model, posedViews, radius)
    landmarks = np.full((face.MESH_VERTICES, 3), np.nan)
    landmarks[indices] = (vertices[corners] * weights[:, :, None]).sum(1)
    residuals, _, _ = _reprojection(posedViews, landmarks[indices], indices)
    after = float(np.sqrt((residuals ** 2).sum(1).mean()))
    if progress:
        progress(1.0)
    return {"vertices": vertices, "triangles": model.triangles, "landmarks": landmarks, "coefficients": coefficients,
            "error": start.get("error", 0.0), "similarity": (scale, rotation, translation), "pixels": (before, after),
            "jaw": jaw, "views": moves}


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
