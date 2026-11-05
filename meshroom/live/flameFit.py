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


def earVertices(model: FlameModel) -> np.ndarray:
    """ (n,) the vertices of the ears of FLAME: those whose nearest vertex of the canonical head of the FaceBuilder (put
    on the template by the embedding landmarks) is on its ears (headFit.headRegions). """
    from scipy.spatial import cKDTree
    from . import headFit
    head = face.canonicalHead()
    n = face.MESH_VERTICES
    corners, weights = model.landmarkRows()
    keep = model.embeddingIndices < n
    points = (model.template[corners[keep]] * weights[keep][:, :, None]).sum(1)
    scale, rotation, translation = _similarity(head.vertices[model.embeddingIndices[keep]], points,
                                               np.ones(int(keep.sum())))
    placed = scale * head.vertices @ rotation.T + translation
    distance, nearest = cKDTree(placed).query(model.template)
    return headFit.headRegions(head)["ear"][nearest] & (distance < 0.01)


def _earFalloff(model: FlameModel, ears: np.ndarray, faceVertices: np.ndarray, radius: float = 3.5) -> np.ndarray:
    """ (2, n): how much each vertex follows the move of each ear (right: x < 0, then left): a gaussian of `radius`
    cm around its center, none within 1 cm of the face (`faceVertices`), all from 3 cm. """
    from scipy.spatial import cKDTree
    template = model.template * 100.0
    toFace, _ = cKDTree(template[faceVertices]).query(template)
    fade = np.clip((toFace - 1.0) / 2.0, 0.0, 1.0)
    falloff = np.zeros((2, len(template)))
    for side, sign in enumerate((-1, 1)):
        mask = ears & (np.sign(template[:, 0]) == sign)
        if mask.any():
            falloff[side] = np.exp(-((template - template[mask].mean(0)) ** 2).sum(1) / radius ** 2) * fade
    return falloff


def _earOutline(vertices, triangles, ears, view, camera, search, faceVertices=None):
    """
    The rims of the ears in a view and the border of the skin there: [(vertex, direction (2,) out of the ear, offset
    (pixels) of the border along it)]. The vertices of the ears between the triangles facing the camera and the others
    with, just beyond them, nothing of the ear nor of the face (nothing, or the skull: under the hair, not skin); the
    nearest border of the skin (the segmenter's face or body skin) from inside to outside within `search` pixels.
    """
    from .geoRender import rasterize
    width, height = view.size
    xy, depth, _ = camera.project(vertices, width, height)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    facing = (np.cross(b - a, c - a) * (camera.position - (a + b + c) / 3.0)).sum(1) > 0
    front, back = np.zeros(len(vertices), bool), np.zeros(len(vertices), bool)
    for i in range(3):
        front[triangles[facing, i]] = True
        back[triangles[~facing, i]] = True
    candidates = np.where(ears & front & back & (depth > 0))[0]
    if not len(candidates):
        return []
    normals = _vertexNormals(vertices, triangles)
    ahead, _, _ = camera.project(vertices[candidates] + 0.1 * normals[candidates], width, height)
    directions = ahead - xy[candidates]
    lengths = np.linalg.norm(directions, axis=1)
    keep = lengths > 1e-6
    candidates, directions = candidates[keep], directions[keep] / lengths[keep, None]
    # What is just beyond: nothing, or a triangle that is neither of the ear nor of the face
    _, ids = rasterize(xy[:, 0], height - xy[:, 1], depth, triangles, (height, width))
    blocking = ears.copy()
    if faceVertices is not None:
        blocking[faceVertices] = True
    blockingTriangles = blocking[triangles].any(1)
    beyond = xy[candidates] + 2.5 * directions
    columns = np.clip(np.floor(beyond[:, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.floor(height - beyond[:, 1]).astype(int), 0, height - 1)
    hit = ids[rows, columns]
    free = (hit < 0) | ~blockingTriangles[np.maximum(hit, 0)]
    candidates, directions = candidates[free], directions[free]
    steps = np.arange(-int(search), int(search) + 1, dtype=np.float64)
    result = []
    for vertex, direction in zip(candidates, directions):
        line = xy[vertex][None, :] + steps[:, None] * direction[None, :]
        skin = view.skin(line)
        crossings = np.where(skin[:-1] & ~skin[1:])[0]
        if not len(crossings):
            continue
        offsets = steps[crossings] + 0.5
        result.append((int(vertex), direction, float(offsets[np.argmin(np.abs(offsets))])))
    return result


def fitFlameToViews(model: FlameModel, views, start: dict, iterations: int = 20, regularization: float = 0.01,
                    jawRegularization: float = 5.0, viewRotation: float = 30.0, viewTranslation: float = 5.0,
                    denseWeight: float = 0.2, innerWeight: float = 0.0, outlineFacing: float = 0.3,
                    expressionRegularization: float = 0.01,
                    rigidIterations: int = 5, huber: float = 3.0, exact: bool = False, radius: float = 2.0,
                    segmentations=None, earWeight: float = 1.0, earSearch: float = 25.0,
                    earMoveRegularization: float = 0.05, cancel=None,
                    progress=None) -> dict:
    """
    FLAME fitted on its own on the landmarks of the photos (their reprojection in every view: nothing else, no face of
    the FaceBuilder), from `start` (fitFlame: only where it begins), by Gauss-Newton: its identity (its first 300
    directions, shared by the views), the expression (its other directions) and the jaw of each view (the views are
    not taken at the same moment: eyes, brows, mouth differ), its placement (similarity) and a rigid move of the head
    in each view (the cameras of the views are only estimates: FLAME places itself in each). Regularized: a coefficient
    of 1 against `regularization` (identity) or `expressionRegularization` pixels squared per landmark, a jaw turned
    by 1 radian against `jawRegularization`, a move of a view of `viewRotation` degrees or `viewTranslation` cm against
    1 pixel; its placement alone the first `rigidIterations`; robust (Huber of `huber` pixels). All the 468 landmarks:
    those of the embedding, those of the face oval (`denseWeight`) and the others (`innerWeight`: a rough
    correspondence, it pulls the nose and the brows off) where the MediaPipe face is on FLAME (denseCorrespondence); the
    landmarks of the face oval that a view sees on the outline of the face (FLAME turned away there) on the outline of
    FLAME in that view (its vertex nearest to them there, the error across the outline only). With `exact`, the surface
    of each view then moved around each landmark of the embedding (gaussians of `radius` cm) so that each lands on its
    pixel. With `segmentations` (one per view, a headFit.FitView or None: its labels, what the segmenter saw), the
    ears of FLAME on the outline of the head in a view also on the border of the skin there (`earWeight` against a
    landmark, within `earSearch` pixels), FLAME's shapes and each ear moved as a whole (an ear moved by 1 cm against
    `earMoveRegularization` pixels squared per landmark): the ears seen in the images, that no landmark gives.

    Args:
        views: [{"camera", "size": (w, h), "pixels": (n, 2) the landmarks (Nuke pixels), "hidden": [indices]}].

    Returns:
        fitFlame's dict ("vertices", "landmarks", "coefficients": the head neutral: its identity, no expression, the jaw
        closed), with "viewVertices": the head as it is in each view (its expression, its jaw; face space), "jaws",
        "expressions", "pixels": the RMS reprojection error (pixels, the landmarks of the embedding) before and after,
        "views": the 4x4 move of the head (face space) in each view (the camera of a view sees the head moved by it:
        camera matrix inv(move) @ matrix).
    """
    from .headFit import _projection
    count = model.directions.shape[2]
    hasJaw = model.jawWeights is not None
    # FLAME: 300 directions of identity (shared by the views), then those of expression (each view its own, with its
    # jaw: the views are not taken at the same moment)
    identityCount = min(count, 300)
    expressionCount = count - identityCount
    viewCount = len(views)
    scale0, rotation0, translation0 = start["similarity"]
    startCoefficients = np.asarray(start["coefficients"], dtype=np.float64)
    identity = startCoefficients[:identityCount].copy()
    expressions = np.tile(startCoefficients[identityCount:], (viewCount, 1))
    jaws = np.zeros((viewCount, 3))
    pose = np.zeros(7)  # the placement: rotation vector, translation (cm), log of the scale
    viewPoses = np.zeros((viewCount, 6))  # each view: rotation vector, translation (cm)
    jawDirections = (model.jawRegressor @ model.directions.reshape(len(model.template), -1)).reshape(3, count) \
        if hasJaw else None
    corners, weights, embedded, oval = denseCorrespondence(model)
    isOval = np.zeros(len(corners), bool)
    isOval[oval] = True
    center = (scale0 * (model.vertices(startCoefficients)[corners[embedded]] * weights[embedded][:, :, None]).sum(1)
              @ rotation0.T + translation0).mean(0)
    flameFace = np.unique(corners)
    segmentations = list(segmentations or [None] * len(views))
    ears = earVertices(model) if any(seg is not None for seg in segmentations) else None
    # With the segmentation, each ear also moved as a whole (beyond what FLAME's shapes do: where the views see it),
    # smoothly around it and not on the face (earFalloff)
    earMoves = np.zeros((2, 3))
    earFalloff = _earFalloff(model, ears, flameFace) if ears is not None else np.zeros((2, len(model.template)))
    # The columns: identity, then for each view its expression and jaw, then the placement, then the moves of the views
    perView = expressionCount + 3
    placementColumn = identityCount + viewCount * perView
    earColumn = placementColumn + 7 + 6 * viewCount
    size = earColumn + 6

    def viewCoefficients(v):
        return np.r_[identity, expressions[v]]

    def headVertices(c, j, p, e=None):
        """ FLAME in the face space (before the move of a view), its ears moved by `e` (2, 3) (default: earMoves). """
        vertices = model.posed(model.vertices(c), j) if hasJaw else model.vertices(c)
        rotation = _rotation(p[:3]) @ rotation0
        placed = scale0 * np.exp(p[6]) * vertices @ rotation.T + translation0 + p[3:6]
        return placed + earFalloff.T @ (earMoves if e is None else e)

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
        # (the oval: the outline of the face, that the embedding lacks; the others where the canonical MediaPipe face is
        # on FLAME, a rough correspondence: little, it would pull the nose and the brows off)
        pixelWeights = np.where(embedded[landmarks], 1.0, np.where(isOval[landmarks], denseWeight, innerWeight))
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

    def embeddedError(viewVertices, moves):
        errors = []
        for v, view in enumerate(views):
            keep = seen[v] & embedded
            if not keep.any():
                continue
            points = at(viewVertices[v], corners[keep], weights[keep]) @ moves[v][:3, :3].T + moves[v][:3, 3]
            xy, _, _ = view["camera"].project(points, *view["size"])
            errors.append(xy - view["pixels"][np.where(keep)[0]])
        return float(np.sqrt((np.concatenate(errors) ** 2).sum(1).mean()))

    def _earRows(vertices, v, coefficients, jaw, linear):
        """ The rows (k, size) and right sides (k,) of the ears of a view on the border of the skin (pixels). """
        view = views[v]
        width, height = view["size"]
        moved = inView(vertices, viewPoses[v])
        found = _earOutline(moved, model.triangles, ears, segmentations[v], view["camera"], earSearch, flameFace)
        if not found:
            return np.zeros((0, size)), np.zeros(0)
        indices = np.array([vertex for vertex, _, _ in found])
        directions = np.array([direction for _, direction, _ in found])
        offsets = np.array([offset for _, _, offset in found])
        _, _, jacobian = _projection(view["camera"], moved[indices], width, height)
        # Robust as the landmarks (an ear under the hair: the border found is not its own)
        weight = earWeight * np.sqrt(np.minimum(1.0, huber / np.maximum(np.abs(offsets), 1e-12)))
        rows = np.zeros((len(indices), size))
        across = np.einsum("ri,rij->rj", directions, jacobian)  # d (pixels along the direction) / d point
        coefficientRows = np.einsum("rj,jk,rkm->rm", across, linear, model.directions[indices])
        first = identityCount + v * perView
        rows[:, :identityCount] = coefficientRows[:, :identityCount]
        rows[:, first:first + expressionCount] = coefficientRows[:, identityCount:]
        # (not the placement nor the move of the view: the landmarks place the head, the ears do not)
        # The moves of the ears: the falloff at the vertex, turned by the move of the view
        viewRotation = _rotation(viewPoses[v, :3])
        for side in range(2):
            for axis in range(3):
                rows[:, earColumn + 3 * side + axis] = (across @ viewRotation[:, axis]) * earFalloff[side, indices]
        # The vertex moved by `offset` along the direction: the residual is minus it
        return weight[:, None] * rows, weight * offsets

    if not any(keep.any() for keep in seen):
        raise ValueError("FLAME: no landmark seen in the views")
    before = embeddedError([headVertices(startCoefficients, np.zeros(3), pose)] * viewCount, [np.eye(4)] * viewCount)
    step = 1e-4
    rotationDamping, translationDamping = 1.0 / np.radians(viewRotation) ** 2, 1.0 / viewTranslation ** 2
    for iteration in range(iterations):
        if cancel:
            cancel()
        rotation = _rotation(pose[:3]) @ rotation0
        scale = scale0 * np.exp(pose[6])
        allRows, blocks, right, viewRows = 0.0, [], [], []
        for v in range(viewCount):
            if not seen[v].any():
                viewRows.append(0.0)
                continue
            coefficients, jaw = viewCoefficients(v), jaws[v]
            vertices = headVertices(coefficients, jaw, pose)
            rows = correspondences(vertices, v)
            residual, jacobian = residualsOf(vertices, v, rows)
            # Robust (Huber, `huber` pixels): a landmark far off (MediaPipe wrong there: hair, a hidden eye) counts less
            pixelWeights = rows[3]
            raw = np.linalg.norm(residual, axis=1) / np.maximum(pixelWeights, 1e-12)
            pixelWeights = pixelWeights * np.sqrt(np.minimum(1.0, huber / np.maximum(raw, 1e-12)))
            rows = rows[:3] + (pixelWeights,) + rows[4:]
            residual, jacobian = residualsOf(vertices, v, rows)
            _, rowCorners, rowWeights, pixelWeights, _ = rows
            # d vertex (face space) / d coefficients, the jaw turned (linear blend) and its joint moved
            cornerBasis = model.directions[rowCorners]  # (r, 3 corners, 3, k)
            if hasJaw and np.any(jaw):
                turn = _rotation(jaw)
                cw = model.jawWeights[rowCorners][:, :, None, None]
                cornerBasis = (1 - cw) * cornerBasis + cw * (np.einsum("ij,rcjk->rcik", turn, cornerBasis)
                                                             + (np.eye(3) - turn) @ jawDirections)
            basis = (cornerBasis * rowWeights[:, :, None, None]).sum(1)
            linear = _rotation(viewPoses[v, :3]) @ (scale * rotation)
            rowsC = pixelWeights[:, None, None] * np.einsum("rij,jk,rkm->rim", jacobian, linear, basis)
            across = np.isfinite(rows[4][:, 0])
            if across.any():
                n2 = rows[4][across]
                rowsC[across] = np.einsum("ri,rj,rjk->rik", n2, n2, rowsC[across])
            block = np.zeros((len(residual), 2, size))
            first = identityCount + v * perView
            block[:, :, :identityCount] = rowsC[:, :, :identityCount]
            block[:, :, first:first + expressionCount] = rowsC[:, :, identityCount:]
            # The jaw, the placement, the move of the view: numerically
            for k in range(16):
                if k < 3 and not hasJaw:
                    continue
                j, p, vp = jaw.copy(), pose.copy(), viewPoses[v].copy()
                (j if k < 3 else p if k < 10 else vp)[k if k < 3 else k - 3 if k < 10 else k - 10] += step
                movedResidual, _ = residualsOf(headVertices(coefficients, j, p), v, rows, vp)
                column = first + expressionCount + k if k < 3 else placementColumn + (k - 3) if k < 10 else \
                    placementColumn + 7 + 6 * v + (k - 10)
                block[:, :, column] = (movedResidual - residual) / step
            blocks.append(block.reshape(-1, size))
            right.append(-residual.reshape(-1))
            if segmentations[v] is not None and iteration >= rigidIterations:
                earRows, earRight = _earRows(vertices, v, coefficients, jaw, linear)
                if len(earRows):
                    blocks.append(earRows)
                    right.append(earRight)
            viewRows.append(float(pixelWeights.sum()))
            allRows += viewRows[-1]
        a, b = np.concatenate(blocks), np.concatenate(right)
        # Damping: the identity, the expressions and the jaws toward 0 (a likely face: identity against all the
        # landmarks, the expression and the jaw of a view against its own), the moves of the views toward none (nearly
        # free), the placement free but its depth and size (that a single view hardly tells apart: 10% of size or 5 cm
        # against 1 pixel)
        damping = np.zeros(size)
        damping[:identityCount] = regularization * allRows
        for v in range(viewCount):
            first = identityCount + v * perView
            damping[first:first + expressionCount] = expressionRegularization * max(viewRows[v], 1.0)
            damping[first + expressionCount:first + perView] = jawRegularization * max(viewRows[v], 1.0) if hasJaw \
                else 1.0
            pose0 = placementColumn + 7 + 6 * v
            damping[pose0:pose0 + 3] = max(viewRows[v], 1.0) * rotationDamping
            damping[pose0 + 3:pose0 + 6] = max(viewRows[v], 1.0) * translationDamping
        damping[placementColumn:placementColumn + 3] = 1e-6
        damping[placementColumn + 3:placementColumn + 6] = allRows / 5.0 ** 2
        damping[placementColumn + 6] = allRows / 0.1 ** 2
        # (an ear moved by 1 cm against `earMoveRegularization` pixels squared per landmark; held when not seen)
        damping[earColumn:] = earMoveRegularization * allRows if ears is not None else 1.0
        current = np.r_[identity, np.c_[expressions, jaws].ravel(), np.zeros(7), viewPoses.ravel(), earMoves.ravel()]
        rightSide = a.T @ b - damping * current
        if iteration < rigidIterations:
            # First its placement alone (the shape not taking what the pose should)
            damping[:placementColumn] = 1e12
            rightSide[:placementColumn] = 0.0
        delta = np.linalg.solve(a.T @ a + np.diag(damping), rightSide)
        identity += delta[:identityCount]
        perViewDelta = delta[identityCount:placementColumn].reshape(viewCount, perView)
        expressions += perViewDelta[:, :expressionCount]
        jaws += perViewDelta[:, expressionCount:]
        pose += delta[placementColumn:placementColumn + 7]
        viewPoses += delta[placementColumn + 7:earColumn].reshape(-1, 6)
        earMoves += delta[earColumn:].reshape(2, 3)
        if progress:
            progress(0.8 * (iteration + 1) / float(iterations))
        if np.abs(delta).max() < 1e-5 and iteration >= rigidIterations:
            break
    rotation = _rotation(pose[:3]) @ rotation0
    scale = scale0 * np.exp(pose[6])
    translation = translation0 + pose[3:6]
    # The head of the actor: its identity, neutral (no expression, the jaw closed); and as it is in each view
    neutral = np.r_[identity, np.zeros(expressionCount)]
    vertices = headVertices(neutral, np.zeros(3), pose)
    viewVertices = [headVertices(viewCoefficients(v), jaws[v], pose) for v in range(viewCount)]
    moves = [_poseMatrix(viewPoses[v], center) for v in range(viewCount)]
    if exact:
        # (what FLAME could not: each landmark on its pixel, in each view)
        for v, (view, move) in enumerate(zip(views, moves)):
            posed = dict(view, camera=view["camera"].copy(matrix=np.linalg.inv(move) @ view["camera"].matrix))
            viewVertices[v] = _exactLandmarks(viewVertices[v], model, [posed], radius)
    indices = model.embeddingIndices
    landmarks = np.full((face.MESH_VERTICES, 3), np.nan)
    landmarks[indices] = at(vertices, *model.landmarkRows())
    after = embeddedError(viewVertices, moves)
    if progress:
        progress(1.0)
    return {"vertices": vertices, "triangles": model.triangles, "landmarks": landmarks, "coefficients": neutral,
            "error": start.get("error", 0.0), "similarity": (scale, rotation, translation), "pixels": (before, after),
            "jaw": jaws.mean(0), "jaws": jaws, "earMoves": earMoves, "expressions": expressions, "viewVertices": viewVertices,
            "views": moves}


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
