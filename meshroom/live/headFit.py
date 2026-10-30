"""
Automatic fit of the parts of a head that the face landmarks do not see (ears, neck, skull) on the views of a
FaceBuilder: what the pins do by hand, from what networks see in the photos.

- Contours: a segmentation of each view (MediaPipe multiclass selfie segmenter: hair, face skin, body skin, clothes...)
  gives where the skin and the head are. The outlines of the ears and the neck are moved onto the border of the skin
  (both ways); the outline of the skull, hidden by the hair, is only kept inside the head (hair or skin): a skull out
  of it is moved in, inside it nothing is known.
- Depth (optional): a monocular depth network (Depth Anything V2) gives a relative depth of each view; it is scaled on
  the face (placed by its landmarks: 1 / depth = a * disparity + b) and moves the ears and the neck seen on the skin
  along the view axis.

The head is moved by a smooth field (gaussians of the 3D distance around control points of the head, `radius`), the
face kept (but near its oval, face.headMobility), the constraints robust (Cauchy weights: a wrong border is ignored),
fitted on all the views together, again from the new outlines (Gauss-Newton). The result is the field (control points
and their moves): `applyFit` moves a head with it.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from . import face, scene3d

# The categories of the MediaPipe multiclass selfie segmenter
BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHERS = range(6)


class HeadSegmenter:
    """ MediaPipe multiclass selfie segmenter: the confidence of each category at each pixel. """

    def __init__(self, modelPath: str):
        mp = face.importMediaPipe()
        try:
            from mediapipe.tasks import python as tasks
            from mediapipe.tasks.python import vision
        except Exception as e:
            raise face.FaceUnavailable(f"MediaPipe is not installed (pip install mediapipe): {e}") from e
        self._mp = mp
        options = vision.ImageSegmenterOptions(base_options=tasks.BaseOptions(model_asset_path=modelPath),
                                               running_mode=vision.RunningMode.IMAGE,
                                               output_confidence_masks=True, output_category_mask=False)
        try:
            self._segmenter = vision.ImageSegmenter.create_from_options(options)
        except Exception as e:
            raise face.FaceUnavailable(f"cannot load the segmentation model {modelPath}: {e}") from e

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        """ (h, w, 6) confidences of the categories (BACKGROUND...OTHERS) of an image (h, w, 3) uint8, rows from the
        top. """
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._segmenter.segment(image)
        masks = [np.asarray(mask.numpy_view(), dtype=np.float32).reshape(rgb.shape[:2])
                 for mask in result.confidence_masks]
        while len(masks) < 6:
            masks.append(np.zeros(rgb.shape[:2], np.float32))
        return np.stack(masks[:6], axis=2)

    def close(self):
        try:
            self._segmenter.close()
        except Exception:
            pass


def headSegmenter() -> HeadSegmenter:
    """ The segmenter with the model of the models folder (MESHROOM_MODELS_PATH), or FaceUnavailable. """
    from .neural import NeuralUnavailable
    from .neural.models import modelFile
    try:
        path = modelFile("headsegmenter", "mediapipe/selfie_multiclass_256x256.tflite")
    except NeuralUnavailable as e:
        raise face.FaceUnavailable(str(e)) from e
    return HeadSegmenter(path)


class DepthEstimator:
    """ Depth Anything V2 Small (transformers): a relative inverse depth (larger: nearer) of an image. """

    def __init__(self, device=None):
        import os
        from .neural import torchDevice
        from .neural.models import importPackage, modelFile
        transformers = importPackage("depthanything")
        folder = os.path.dirname(modelFile("depthanything", "depthanything/model.safetensors"))
        self.device = torchDevice(device)
        self.processor = transformers.AutoImageProcessor.from_pretrained(folder)
        self.model = transformers.AutoModelForDepthEstimation.from_pretrained(folder).to(self.device).eval()

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """ (h, w) relative inverse depth of an image (h, w, 3) float in [0, 1] (display referred), rows from the top. """
        import torch
        inputs = self.processor(images=(np.clip(rgb, 0, 1) * 255).astype(np.uint8), return_tensors="pt")
        with torch.no_grad():
            predicted = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).predicted_depth
            resized = torch.nn.functional.interpolate(predicted[:, None], size=rgb.shape[:2], mode="bilinear",
                                                      align_corners=False)
        return resized[0, 0].float().cpu().numpy()


def depthEstimator(device=None) -> DepthEstimator:
    """ The depth network (loaded once per device), or NeuralUnavailable. """
    from .neural.models import cachedModel, loadError
    try:
        return cachedModel(("depthanything", str(device)), lambda: DepthEstimator(device))
    except Exception as e:
        from .neural import NeuralUnavailable
        if isinstance(e, NeuralUnavailable):
            raise
        raise loadError("depthanything", e) from e


# The parts of the head

_REGIONS = {}


def headRegions(head: scene3d.GeoData) -> dict:
    """
    {"ear", "neck", "skull"}: masks of the vertices of a head of the FaceBuilder (its topology: the parts are found on
    the canonical head, in centimeters): the ears stick out of the sides, the neck is under the jaw, the skull is the
    rest of the head (not the face).
    """
    count = len(head.vertices)
    if count in _REGIONS:
        return _REGIONS[count]
    vertices = face.canonicalHead().vertices
    if len(vertices) != count:
        raise ValueError("Head regions: not a head of the FaceBuilder.")
    x, y, z = vertices.T
    isHeadPart = np.arange(count) >= face.MESH_VERTICES
    ear = isHeadPart & (np.abs(x) > 7.3) & (y > -4.0) & (y < 4.5) & (z > -6.0) & (z < 1.5)
    neck = isHeadPart & ~ear & (y < -7.0)
    skull = isHeadPart & ~ear & ~neck
    # The neck down to the collar (its lower part and the cut of the mesh are hidden by clothes, wrongly segmented)
    upperNeck = neck & (y > -10.5)
    _REGIONS[count] = {"ear": ear, "neck": neck, "skull": skull, "upperNeck": upperNeck}
    return _REGIONS[count]


# A view

class FitView:
    """
    A view of the head: its camera (camera to face space), size, and what the networks saw on a part of it (the crop
    around the head): `labels` (h, w, 6) confidences of the segmenter, `disparity` (h, w) relative inverse depth or
    None; `origin` (x0, y0): the top-left corner of the crop in the view (pixels, rows from the top).
    """

    def __init__(self, camera, size, labels: np.ndarray, origin=(0, 0), disparity: Optional[np.ndarray] = None,
                 search: Optional[float] = None):
        self.camera, self.size = camera, (int(size[0]), int(size[1]))
        # How far (pixels) the borders are searched from the outline (None: the default of fitHead)
        self.search = search
        self.labels = np.asarray(labels, dtype=np.float32)
        self.origin = (int(origin[0]), int(origin[1]))
        self.disparity = None if disparity is None else np.asarray(disparity, dtype=np.float32)

    def _rowsColumns(self, xy: np.ndarray):
        """ The rows and columns in the crop of Nuke pixels (n, 2), and whether they are in it. """
        columns = np.floor(xy[:, 0]).astype(np.int64) - self.origin[0]
        rows = np.floor(self.size[1] - xy[:, 1]).astype(np.int64) - self.origin[1]
        h, w = self.labels.shape[:2]
        inside = (rows >= 0) & (rows < h) & (columns >= 0) & (columns < w)
        return np.clip(rows, 0, h - 1), np.clip(columns, 0, w - 1), inside

    def skin(self, xy: np.ndarray) -> np.ndarray:
        """ Whether the skin (face or body) is at Nuke pixels (n, 2). """
        rows, columns, inside = self._rowsColumns(xy)
        values = self.labels[rows, columns]
        return inside & (values[:, FACE_SKIN] + values[:, BODY_SKIN] > 0.5)

    def headArea(self, xy: np.ndarray) -> np.ndarray:
        """ Whether the head (skin, hair, or what is worn on it) is at Nuke pixels (n, 2). """
        rows, columns, inside = self._rowsColumns(xy)
        values = self.labels[rows, columns]
        return inside & (values[:, FACE_SKIN] + values[:, BODY_SKIN] + values[:, HAIR] + values[:, OTHERS] > 0.5)

    def disparityAt(self, xy: np.ndarray) -> np.ndarray:
        rows, columns, inside = self._rowsColumns(xy)
        return np.where(inside, self.disparity[rows, columns], np.nan)


# The constraints: (vertex, row (3,): the change of the measure for a move of the vertex, the wanted change, kind)

def _projection(camera, points: np.ndarray, width: int, height: int):
    """ (xy (n, 2), depth (n,), jacobians (n, 2, 3) of the projection). """
    xy, depth, _ = camera.project(points, width, height)
    jacobians = np.zeros((len(points), 2, 3))
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = 1e-3
        moved, _, _ = camera.project(points + step, width, height)
        jacobians[:, :, axis] = (moved - xy) / 1e-3
    return xy, depth, jacobians


def _vertexNormals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    normals = np.zeros_like(vertices)
    faceNormals = np.cross(b - a, c - a)
    for i in range(3):
        np.add.at(normals, triangles[:, i], faceNormals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def contourConstraints(vertices: np.ndarray, triangles: np.ndarray, view: FitView, regions: dict,
                       mobility: np.ndarray, search: float) -> list:
    """
    The constraints of the outlines of the head in a view: the vertices on its outline (between the triangles facing
    the camera and the others, nothing of the head beyond them) of the ears and the neck moved, along the normal of
    the outline in the image, onto the nearest border of the skin within `search` pixels; those of the skull outside
    the head area moved in onto its border. Units: centimeters at the vertex (pixels times its depth / focal).
    """
    width, height = view.size
    camera = view.camera
    xy, depth, jacobians = _projection(camera, vertices, width, height)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    toCamera = camera.position - (a + b + c) / 3.0
    facing = (np.cross(b - a, c - a) * toCamera).sum(1) > 0
    front = np.zeros(len(vertices), dtype=bool)
    back = np.zeros(len(vertices), dtype=bool)
    for i in range(3):
        front[triangles[facing, i]] = True
        back[triangles[~facing, i]] = True
    parts = regions["ear"] | regions["upperNeck"] | regions["skull"]
    candidates = np.where(front & back & (mobility > 0.05) & (depth > 0) & parts)[0]
    if not len(candidates):
        return []
    normals = _vertexNormals(vertices, triangles)
    ahead, _, _ = camera.project(vertices[candidates] + 0.1 * normals[candidates], width, height)
    directions = ahead - xy[candidates]
    lengths = np.linalg.norm(directions, axis=1)
    keep = lengths > 1e-6
    candidates, directions = candidates[keep], directions[keep] / lengths[keep, None]
    # The outline of the head only: beyond it, nothing of the head (an outline inside the head, e.g. an ear over the
    # skull, has the head behind it: its border in the image is not that of this outline)
    from .geoRender import rasterize
    h, w = view.labels.shape[:2]
    _, ids = rasterize(xy[:, 0] - view.origin[0], (height - xy[:, 1]) - view.origin[1], depth, triangles, (h, w))
    beyond = xy[candidates] + 2.5 * directions
    rows, columns, inside = view._rowsColumns(beyond)
    outline = ~inside | (ids[rows, columns] < 0)
    candidates, directions = candidates[outline], directions[outline]
    focal = camera.focalPixels(width)
    steps = np.arange(-int(search), int(search) + 1, dtype=np.float64)
    constraints = []
    for vertex, direction in zip(candidates, directions):
        line = xy[vertex][None, :] + steps[:, None] * direction[None, :]
        isSkull = regions["skull"][vertex]
        inside = view.headArea(line) if isSkull else view.skin(line)
        # The borders from inside (toward -direction) to outside (toward +direction)
        crossings = np.where(inside[:-1] & ~inside[1:])[0]
        if not len(crossings):
            continue
        offsets = steps[crossings] + 0.5
        if isSkull:
            # Only a skull out of the head is moved (in): inside it, the hair hides it
            if inside[len(steps) // 2]:
                continue
            # (the vertex on the border: half a pixel out, not moved)
            offsets = offsets[offsets < -0.5] + 0.5
            if not len(offsets):
                continue
        offset = float(offsets[np.argmin(np.abs(offsets))])
        scale = max(float(depth[vertex]), 1e-6) / max(focal, 1e-6)
        row = scale * (direction @ jacobians[vertex])
        kind = "skull" if isSkull else "neck" if regions["neck"][vertex] else "contour"
        constraints.append((int(vertex), row, scale * offset, kind))
    return constraints


def depthConstraints(vertices: np.ndarray, triangles: np.ndarray, view: FitView, regions: dict,
                     mobility: np.ndarray) -> list:
    """
    The constraints of the depth network on a view: its relative inverse depth scaled on the visible face (placed by its
    landmarks: 1 / depth = a * disparity + b, robust), then the visible vertices of the ears and the neck on the skin
    moved along the view axis to their depth. None when the network is not used, or does not agree with the face.
    """
    if view.disparity is None:
        return []
    from .geoRender import rasterize
    width, height = view.size
    camera = view.camera
    xy, depth, _ = camera.project(vertices, width, height)
    # What the camera sees (a vertex hidden by the head is not constrained)
    h, w = view.labels.shape[:2]
    u = xy[:, 0] - view.origin[0]
    v = (height - xy[:, 1]) - view.origin[1]
    zbuffer, _ = rasterize(u, v, depth, triangles, (h, w))
    rows, columns, inside = view._rowsColumns(xy)
    visible = inside & (depth > 0) & (depth <= zbuffer[rows, columns] * 1.01 + 1e-6)
    onSkin = view.skin(xy)
    disparity = view.disparityAt(xy)
    faceVertices = np.zeros(len(vertices), dtype=bool)
    faceVertices[:face.MESH_VERTICES] = True
    reference = np.where(visible & onSkin & faceVertices & np.isfinite(disparity))[0]
    if len(reference) < 30:
        return []
    # 1 / depth = a * disparity + b, iteratively reweighted (Cauchy)
    A = np.c_[disparity[reference], np.ones(len(reference))]
    target = 1.0 / depth[reference]
    weights = np.ones(len(reference))
    for _ in range(5):
        coefficients, *_ = np.linalg.lstsq(A * weights[:, None], target * weights, rcond=None)
        errors = np.abs(A @ coefficients - target)
        scale = max(1.4826 * float(np.median(errors)), 1e-6)
        weights = 1.0 / np.sqrt(1.0 + (errors / (3.0 * scale)) ** 2)
    a, b = coefficients
    predicted = A @ coefficients
    # Not used when it does not follow the face (its depth relief): the inverse depth correlates, increasing
    if a <= 0 or np.corrcoef(predicted, target)[0, 1] < 0.5:
        return []
    forward = -camera.viewMatrix[2, :3]
    constraints = []
    targets = np.where(visible & onSkin & regions["ear"] & (mobility > 0.05) &
                       np.isfinite(disparity))[0]
    # Not on a depth edge (an ear rim before the skull): the depths around the vertex (3 pixels) must agree
    around = [view.disparityAt(xy[targets] + offset) for offset in ([3, 0], [-3, 0], [0, 3], [0, -3])]
    for k, vertex in enumerate(targets):
        values = np.array([a * disparity[vertex] + b] + [a * values[k] + b for values in around])
        if not np.all(np.isfinite(values)) or values.min() <= 1e-6:
            continue
        depths = 1.0 / values
        if depths.max() - depths.min() > 0.5:
            continue
        constraints.append((int(vertex), forward.copy(), float(depths[0] - depth[vertex]), "depth"))
    return constraints


# The field

def controlPoints(vertices: np.ndarray, mobility: np.ndarray, spacing: float) -> np.ndarray:
    """ Vertices spread over the movable part of a head (farthest point sampling, about `spacing` apart). """
    candidates = np.where(mobility > 0.05)[0]
    if not len(candidates):
        return np.zeros(0, dtype=np.int64)
    chosen = [int(candidates[np.argmax(vertices[candidates, 1])])]
    distance = np.linalg.norm(vertices[candidates] - vertices[chosen[0]], axis=1)
    while True:
        k = int(np.argmax(distance))
        if distance[k] < spacing:
            break
        chosen.append(int(candidates[k]))
        distance = np.minimum(distance, np.linalg.norm(vertices[candidates] - vertices[candidates[k]], axis=1))
    return np.array(chosen, dtype=np.int64)


def basis(vertices: np.ndarray, centers: np.ndarray, radius: float, mobility: np.ndarray) -> np.ndarray:
    """ (n, c): the move of each vertex for a unit move of each control point. """
    points = vertices[centers]
    distances = ((vertices[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    return np.exp(-distances / (radius * radius)) * mobility[:, None]


def applyFit(head: scene3d.GeoData, fit: Optional[dict]) -> np.ndarray:
    """ The vertices of a head moved by a fit (`fitHead`), computed on that head ({} or None: unchanged). """
    vertices = np.asarray(head.vertices, dtype=np.float64)
    if not fit or not fit.get("centers"):
        return vertices
    centers = np.asarray(fit["centers"], dtype=np.int64)
    if centers.max() >= len(vertices):
        return vertices
    moves = np.asarray(fit["weights"], dtype=np.float64).reshape(-1, 3)
    return vertices + _capped(basis(vertices, centers, float(fit["radius"]), face.headMobility(head)) @ moves)


# Robust scale of each kind of constraint (centimeters): beyond a few times this, a constraint barely counts
ROBUST_SCALE = {"contour": 0.6, "neck": 0.5, "skull": 0.6, "depth": 0.8}
# How much each kind counts (the depth: all its constraints together, relative to all the outlines)
KIND_WEIGHT = {"contour": 1.0, "neck": 0.5, "skull": 1.0, "depth": 0.5}
# The largest move of a vertex (centimeters): a head differs from the base by less (a wrong border cannot drag it far)
MAX_MOVE = 2.5


def _capped(moves: np.ndarray) -> np.ndarray:
    """ Moves (n, 3) smoothly limited to MAX_MOVE. """
    lengths = np.linalg.norm(moves, axis=1, keepdims=True)
    factor = np.where(lengths > 1e-9, MAX_MOVE * np.tanh(lengths / MAX_MOVE) / np.maximum(lengths, 1e-9), 1.0)
    return moves * factor


def fitHead(head: scene3d.GeoData, views: List[FitView], radius: float = 3.0, iterations: int = 5,
            regularization: float = 2.0, search: float = 40.0, useDepth: bool = True,
            cancel: Optional[Callable[[], None]] = None, progress: Optional[Callable[[float], None]] = None) -> dict:
    """
    Fit the ears, neck and skull of a head (FaceBuilder topology, face space) on views (FitView): the moves of control
    points (`radius` apart / 0.75, gaussians of `radius`) best explaining the contours (and the depth) of all the views,
    regularized (`regularization`: their moves cost a little), robust, again from the new outlines.

    Returns:
        {"radius", "centers": [vertex indices], "weights": [[x, y, z] per control point], "constraints": count used}.
    """
    vertices = np.asarray(head.vertices, dtype=np.float64)
    triangles = head.triangles()
    regions = headRegions(head)
    mobility = face.headMobility(head)
    radius = max(float(radius), 0.1)
    centers = controlPoints(vertices, mobility, 0.75 * radius)
    if not len(centers) or not views:
        return {"radius": radius, "centers": [], "weights": [], "constraints": 0}
    phi = basis(vertices, centers, radius, mobility)
    weights = np.zeros((len(centers), 3))
    used = 0
    for iteration in range(iterations):
        if cancel:
            cancel()
        current = vertices + _capped(phi @ weights)
        constraints = []
        for view in views:
            constraints += contourConstraints(current, triangles, view, regions, mobility,
                                              view.search if view.search else search)
            if useDepth:
                constraints += depthConstraints(current, triangles, view, regions, mobility)
        used = len(constraints)
        if not constraints:
            break
        rows = np.zeros((len(constraints), 3 * len(centers)))
        rhs = np.zeros(len(constraints))
        # Each kind weighs as a whole (the depth gives many more constraints than the outlines)
        counts = {}
        for constraint in constraints:
            counts[constraint[3]] = counts.get(constraint[3], 0) + 1
        outlines = max(counts.get("contour", 0) + counts.get("neck", 0) + counts.get("skull", 0), 1)
        for k, (vertex, row, wanted, kind) in enumerate(constraints):
            robust = 1.0 / np.sqrt(1.0 + (wanted / (3.0 * ROBUST_SCALE[kind])) ** 2)
            share = KIND_WEIGHT[kind] * (outlines / counts[kind] if kind == "depth" else 1.0)
            root = np.sqrt(share) * robust
            rows[k] = root * np.kron(phi[vertex], row)
            rhs[k] = root * wanted
        # The moves themselves cost (not only their change at this step): what no view sees stays where it was
        damping = regularization
        normal = rows.T @ rows + damping * np.eye(rows.shape[1])
        step = np.linalg.solve(normal, rows.T @ rhs - damping * weights.ravel())
        weights += step.reshape(-1, 3)
        if progress:
            progress((iteration + 1) / float(iterations))
        if np.abs(step).max() < 1e-3:
            break
    return {"radius": radius, "centers": [int(c) for c in centers],
            "weights": [[round(float(v), 5) for v in w] for w in weights], "constraints": used}
