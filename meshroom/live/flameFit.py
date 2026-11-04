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


def denseCorrespondence(model: FlameModel):
    """
    Where each of the 468 MediaPipe landmarks is on FLAME (corners (468, 3), weights (468, 3), from the embedding
    (`embedded` (468,) True) or else the nearest FLAME vertex to the canonical MediaPipe face put on the template by the
    embedding landmarks: a correspondence of the models, nothing of the actor), and the landmarks of the face oval.
    """
    from scipy.spatial import cKDTree
    n = face.MESH_VERTICES
    corners, weights = model.landmarkRows()
    allCorners, allWeights = np.zeros((n, 3), np.int64), np.zeros((n, 3))
    embedded = np.zeros(n, bool)
    keep = model.embeddingIndices < n
    allCorners[model.embeddingIndices[keep]] = corners[keep]
    allWeights[model.embeddingIndices[keep]] = weights[keep]
    embedded[model.embeddingIndices[keep]] = True
    canonical = face.canonicalHead().vertices[:n]
    points = (model.template[corners[keep]] * weights[keep][:, :, None]).sum(1)
    scale, rotation, translation = _similarity(canonical[model.embeddingIndices[keep]], points, np.ones(int(keep.sum())))
    placed = scale * canonical @ rotation.T + translation
    _, nearest = cKDTree(model.template).query(placed[~embedded])
    allCorners[~embedded] = nearest[:, None]
    allWeights[~embedded] = [1.0, 0.0, 0.0]
    oval = np.unique(np.asarray(face.headData()["holeEdges"]).ravel())
    return allCorners, allWeights, embedded, oval[oval < n]


def fitFlameToViews(model: FlameModel, views, start: dict, iterations: int = 20, regularization: float = 0.1,
                    jawRegularization: float = 5.0, viewRotation: float = 30.0, viewTranslation: float = 5.0,
                    denseWeight: float = 0.5, outlineFacing: float = 0.3, expressionRegularization: float = 1.0,
                    rigidIterations: int = 5, huber: float = 3.0, exact: bool = True, radius: float = 2.0, cancel=None,
                    progress=None) -> dict:
    """
    FLAME fitted on its own on the landmarks of the photos (their reprojection in every view: nothing else, no face of
    the FaceBuilder), from `start` (fitFlame: only where it begins): its coefficients (identity and expression), its
    jaw, its placement (similarity) and a rigid move of the head in each view (the cameras of the views are only
    estimates: FLAME places itself in each), by Gauss-Newton, regularized (`regularization`: a coefficient of 1 against
    1 pixel of error per landmark and view; `jawRegularization` the same for a jaw turned by 1 radian; a move of a view
    of `viewRotation` degrees or `viewTranslation` cm against 1 pixel; the expression directions after the 300 of
    identity by `expressionRegularization`; its placement alone the first `rigidIterations`; robust: Huber of `huber`
    pixels). All the 468 landmarks: those of the embedding,
    the others (`denseWeight`) where the MediaPipe face is on FLAME (denseCorrespondence); the landmarks of the face
    oval that a view sees on the outline of the face (FLAME turned away there) on the outline of FLAME in that view
    (its vertex nearest to them there, the error across the outline only). Then, with `exact`, its surface moved around
    each landmark of the embedding (gaussians of `radius` cm, solved together) so that each lands on its pixel.

    Args:
        views: [{"camera", "size": (w, h), "pixels": (n, 2) the landmarks (Nuke pixels), "hidden": [indices]}].

    Returns:
        fitFlame's dict, with "pixels": the RMS reprojection error (pixels, the landmarks of the embedding) before and
        after, "jaw": its rotation vector, "views": the 4x4 move of the head (face space) in each view (the camera of a
        view sees the head moved by it: camera matrix inv(move) @ matrix).
    """
    from .headFit import _projection
    count = model.directions.shape[2]
    hasJaw = model.jawWeights is not None
    scale0, rotation0, translation0 = start["similarity"]
    coefficients = np.asarray(start["coefficients"], dtype=np.float64).copy()
    jaw = np.zeros(3)
    pose = np.zeros(7)  # the placement: rotation vector, translation (cm), log of the scale
    viewPoses = np.zeros((len(views), 6))  # each view: rotation vector, translation (cm)
    jawDirections = (model.jawRegressor @ model.directions.reshape(len(model.template), -1)).reshape(3, count) \
        if hasJaw else None
    corners, weights, embedded, oval = denseCorrespondence(model)
    isOval = np.zeros(len(corners), bool)
    isOval[oval] = True
    center = (scale0 * (model.vertices(coefficients)[corners[embedded]] * weights[embedded][:, :, None]).sum(1)
              @ rotation0.T + translation0).mean(0)
    flameFace = np.unique(corners)

    def headVertices(c, j, p):
        """ FLAME in the face space (before the move of a view). """
        vertices = model.posed(model.vertices(c), j) if hasJaw else model.vertices(c)
        rotation = _rotation(p[:3]) @ rotation0
        return scale0 * np.exp(p[6]) * vertices @ rotation.T + translation0 + p[3:6]

    def inView(points, viewPose):
        """ Points moved as the head is in a view. """
        return (points - center) @ _rotation(viewPose[:3]).T + center + viewPose[3:6]

    def at(vertices, rowCorners, rowWeights):
        return (vertices[rowCorners] * rowWeights[:, :, None]).sum(1)

    seen = []
    for view in views:
        pixels = view["pixels"]
        hidden = set(int(i) for i in view.get("hidden") or [])
        seen.append(np.array([i < len(pixels) and i not in hidden for i in range(len(corners))]))

    def correspondences(vertices, v):
        """ The rows of a view: (landmarks, corners, weights, pixel weights, 2D normals or None) for the head
        `vertices` (face space). """
        view = views[v]
        width, height = view["size"]
        moved = inView(vertices, viewPoses[v])
        normals = _vertexNormals(moved, model.triangles)
        eye = np.asarray(view["camera"].matrix, dtype=np.float64)[:3, 3]
        landmarks = np.where(seen[v])[0]
        rowCorners, rowWeights = corners[landmarks].copy(), weights[landmarks].copy()
        pixelWeights = np.where(embedded[landmarks], 1.0, denseWeight)
        normals2 = np.zeros((len(landmarks), 2))
        # The oval landmarks where FLAME turns away from the camera: on its outline
        points = at(moved, rowCorners, rowWeights)
        toEye = eye - points
        toEye /= np.maximum(np.linalg.norm(toEye, axis=1, keepdims=True), 1e-12)
        facing = (at(normals, rowCorners, rowWeights) * toEye).sum(1)
        outline = isOval[landmarks] & (facing < outlineFacing)
        if outline.any():
            candidates = flameFace[np.abs(((eye - moved[flameFace]) / np.maximum(np.linalg.norm(
                eye - moved[flameFace], axis=1, keepdims=True), 1e-12) * normals[flameFace]).sum(1)) < 0.15]
            if len(candidates):
                xy, _, _ = view["camera"].project(moved[candidates], width, height)
                tip, _, _ = view["camera"].project(moved[candidates] + 0.5 * normals[candidates], width, height)
                for row in np.where(outline)[0]:
                    distance = np.linalg.norm(xy - view["pixels"][landmarks[row]], axis=1)
                    best = int(np.argmin(distance))
                    if distance[best] > 0.1 * max(width, height):
                        pixelWeights[row] = 0.0
                        continue
                    direction = tip[best] - xy[best]
                    rowCorners[row], rowWeights[row] = candidates[best], [1.0, 0.0, 0.0]
                    normals2[row] = direction / max(np.linalg.norm(direction), 1e-12)
                    pixelWeights[row] = denseWeight
        return landmarks, rowCorners, rowWeights, pixelWeights, np.where(outline[:, None], normals2, np.nan)

    def residualsOf(vertices, v, rows, viewPose=None):
        landmarks, rowCorners, rowWeights, pixelWeights, normals2 = rows
        view = views[v]
        points = inView(at(vertices, rowCorners, rowWeights), viewPoses[v] if viewPose is None else viewPose)
        xy, _, jacobian = _projection(view["camera"], points, *view["size"])
        residual = xy - view["pixels"][landmarks]
        across = np.isfinite(normals2[:, 0])
        # (on the outline: only across it)
        residual[across] = (residual[across] * normals2[across]).sum(1)[:, None] * normals2[across]
        return residual * pixelWeights[:, None], jacobian

    def embeddedError(vertices, moves):
        errors = []
        for v, view in enumerate(views):
            keep = seen[v] & embedded
            if not keep.any():
                continue
            points = at(vertices, corners[keep], weights[keep]) @ moves[v][:3, :3].T + moves[v][:3, 3]
            xy, _, _ = view["camera"].project(points, *view["size"])
            errors.append(xy - view["pixels"][np.where(keep)[0]])
        return float(np.sqrt((np.concatenate(errors) ** 2).sum(1).mean()))

    if not any(keep.any() for keep in seen):
        raise ValueError("FLAME: no landmark seen in the views")
    before = embeddedError(headVertices(coefficients, jaw, pose), [np.eye(4)] * len(views))
    size = count + 3 + 7 + 6 * len(views)
    step = 1e-4
    rotationDamping, translationDamping = 1.0 / np.radians(viewRotation) ** 2, 1.0 / viewTranslation ** 2
    for iteration in range(iterations):
        if cancel:
            cancel()
        vertices = headVertices(coefficients, jaw, pose)
        rotation = _rotation(pose[:3]) @ rotation0
        scale = scale0 * np.exp(pose[6])
        # d vertex (face space) / d coefficients, the jaw turned (linear blend) and its joint moved
        if hasJaw and np.any(jaw):
            turn = _rotation(jaw)
            moved = (np.eye(3) - turn) @ jawDirections
        allRows, blocks, right, viewRows = 0, [], [], []
        for v in range(len(views)):
            if not seen[v].any():
                viewRows.append(0)
                continue
            rows = correspondences(vertices, v)
            residual, jacobian = residualsOf(vertices, v, rows)
            # Robust (Huber, `huber` pixels): a landmark far off (MediaPipe wrong there: hair, a hidden eye) counts less
            pixelWeights = rows[3]
            raw = np.linalg.norm(residual, axis=1) / np.maximum(pixelWeights, 1e-12)
            pixelWeights = pixelWeights * np.sqrt(np.minimum(1.0, huber / np.maximum(raw, 1e-12)))
            rows = rows[:3] + (pixelWeights,) + rows[4:]
            residual, jacobian = residualsOf(vertices, v, rows)
            _, rowCorners, rowWeights, pixelWeights, _ = rows
            cornerBasis = model.directions[rowCorners]  # (r, 3 corners, 3, k)
            if hasJaw and np.any(jaw):
                cw = model.jawWeights[rowCorners][:, :, None, None]
                cornerBasis = (1 - cw) * cornerBasis + cw * (np.einsum("ij,rcjk->rcik", turn, cornerBasis) + moved)
            basis = (cornerBasis * rowWeights[:, :, None, None]).sum(1)
            block = np.zeros((len(residual), 2, size))
            linear = _rotation(viewPoses[v, :3]) @ (scale * rotation)
            block[:, :, :count] = pixelWeights[:, None, None] * np.einsum("rij,jk,rkm->rim", jacobian, linear, basis)
            across = np.isfinite(rows[4][:, 0])
            if across.any():
                n2 = rows[4][across]
                block[across, :, :count] = np.einsum("ri,rj,rjk->rik", n2, n2, block[across, :, :count])
            # The jaw, the placement, the move of the view: numerically
            for k in range(16):
                if k < 3 and not hasJaw:
                    continue
                j, p, vp = jaw.copy(), pose.copy(), viewPoses[v].copy()
                (j if k < 3 else p if k < 10 else vp)[k if k < 3 else k - 3 if k < 10 else k - 10] += step
                movedResidual, _ = residualsOf(headVertices(coefficients, j, p), v, rows, vp)
                column = count + k if k < 10 else count + 10 + 6 * v + (k - 10)
                block[:, :, column] = (movedResidual - residual) / step
            blocks.append(block.reshape(-1, size))
            right.append(-residual.reshape(-1))
            viewRows.append(float(pixelWeights.sum()))
            allRows += viewRows[-1]
        a, b = np.concatenate(blocks), np.concatenate(right)
        # Damping: the coefficients and the jaw toward 0 (a likely face), the moves of the views toward none (each
        # against its own landmarks: nearly free), the placement free but its depth and size (that a single view
        # hardly tells apart: 10% of size or 5 cm against 1 pixel)
        damping = np.zeros(size)
        damping[:count] = regularization * allRows
        if count > 300:
            # (FLAME: 300 directions of identity, then those of expression: held more, the face of the images is
            # mostly neutral)
            damping[300:count] = expressionRegularization * allRows
        damping[count:count + 3] = jawRegularization * allRows if hasJaw else 1.0
        damping[count + 3:count + 6] = 1e-6
        damping[count + 6:count + 9] = allRows / 5.0 ** 2
        damping[count + 9] = allRows / 0.1 ** 2
        for v in range(len(views)):
            first = count + 10 + 6 * v
            damping[first:first + 3] = max(viewRows[v], 1) * rotationDamping
            damping[first + 3:first + 6] = max(viewRows[v], 1) * translationDamping
        current = np.r_[coefficients, jaw, np.zeros(7), viewPoses.ravel()]
        rightSide = a.T @ b - damping * current
        if iteration < rigidIterations:
            # First its placement alone (the shape not taking what the pose should)
            damping[:count + 3] = 1e12
            rightSide[:count + 3] = 0.0
        delta = np.linalg.solve(a.T @ a + np.diag(damping), rightSide)
        coefficients += delta[:count]
        jaw += delta[count:count + 3]
        pose += delta[count + 3:count + 10]
        viewPoses += delta[count + 10:].reshape(-1, 6)
        if progress:
            progress(0.8 * (iteration + 1) / float(iterations))
        if np.abs(delta).max() < 1e-5 and iteration >= rigidIterations:
            break
    rotation = _rotation(pose[:3]) @ rotation0
    scale = scale0 * np.exp(pose[6])
    translation = translation0 + pose[3:6]
    vertices = headVertices(coefficients, jaw, pose)
    moves = [_poseMatrix(viewPoses[v], center) for v in range(len(views))]
    # The views seen by their cameras with the head moved there
    posedViews = [dict(view, camera=view["camera"].copy(matrix=np.linalg.inv(move) @ view["camera"].matrix))
                  for view, move in zip(views, moves)]
    if exact:
        # (what FLAME could not: each landmark on its pixel)
        vertices = _exactLandmarks(vertices, model, posedViews, radius)
    indices = model.embeddingIndices
    landmarks = np.full((face.MESH_VERTICES, 3), np.nan)
    landmarks[indices] = at(vertices, *model.landmarkRows())
    after = embeddedError(vertices, moves)
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
