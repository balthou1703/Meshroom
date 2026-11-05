"""
Photometric refinement (after FaceCapture, Technicolor: its "photo cost", the colors of the image at the visible
vertices against the shaded colors of the mesh): a head seen in several views, its shape refined so that the same
skin (one albedo per vertex, shared by the views) lit by each view's light (spherical harmonics of order 2, per color
channel) gives the colors of the images. Gauss-Newton on the CPU, without autodiff: the albedo and the lights solved in
closed form, then the shape (a linear model of it: FLAME's identity) along the image gradients.
"""
from typing import Callable, List, Optional

import numpy as np

from .headFit import _projection

# The 9 real spherical harmonics (bands 0 to 2) of a unit normal, with the cosine lobe convolution (Ramamoorthi)
_SH = (0.886227, 1.023328, 1.023328, 1.023328, 0.858086, 0.858086, 0.858086, 0.247708, 0.429043)


def shBasis(normals: np.ndarray) -> np.ndarray:
    """ (n, 9) the irradiance basis of unit normals (x, y, z). """
    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
    c = _SH
    return np.stack([np.full_like(x, c[0]), c[1] * y, c[2] * z, c[3] * x, c[4] * x * y, c[5] * y * z,
                     c[7] * (3.0 * z * z - 1.0), c[6] * x * z, c[8] * (x * x - y * y)], axis=1)


def vertexNormals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    normals = np.zeros_like(vertices)
    faceNormals = np.cross(b - a, c - a)
    for i in range(3):
        np.add.at(normals, triangles[:, i], faceNormals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def sample(image: np.ndarray, xy: np.ndarray):
    """ (colors (n, c), gradients (n, c, 2) per Nuke pixel, inside (n,)) of an image (h, w, c), rows from the top, at
    Nuke pixel positions (x right, y up; pixel centers at .5): bilinear. """
    h, w = image.shape[:2]
    columns = xy[:, 0] - 0.5
    rows = (h - xy[:, 1]) - 0.5
    inside = (columns >= 0) & (columns <= w - 2) & (rows >= 0) & (rows <= h - 2)
    c0 = np.clip(np.floor(columns).astype(np.int64), 0, w - 2)
    r0 = np.clip(np.floor(rows).astype(np.int64), 0, h - 2)
    fc, fr = np.clip(columns - c0, 0, 1)[:, None], np.clip(rows - r0, 0, 1)[:, None]
    p00, p01 = image[r0, c0], image[r0, c0 + 1]
    p10, p11 = image[r0 + 1, c0], image[r0 + 1, c0 + 1]
    top, bottom = p00 + fc * (p01 - p00), p10 + fc * (p11 - p10)
    colors = top + fr * (bottom - top)
    dColumn = (1 - fr) * (p01 - p00) + fr * (p11 - p10)
    dRow = (1 - fc) * (p10 - p00) + fc * (p11 - p01)
    # (y up: a row down is -1 in y)
    gradients = np.stack([dColumn, -dRow], axis=2)
    return colors, gradients, inside


def visibleVertices(vertices: np.ndarray, triangles: np.ndarray, camera, width: int, height: int,
                    facing: float = 0.25, tolerance: float = 0.004) -> np.ndarray:
    """ (n,) the vertices seen by a camera: in front, facing it (cosine above `facing`), not behind the surface (the
    depth buffer, `tolerance` relative). """
    from .geoRender import rasterize
    xy, depth, inFront = camera.project(vertices, width, height)
    buffer, _ = rasterize(xy[:, 0], height - xy[:, 1], depth, triangles, (height, width))
    columns = np.clip(np.floor(xy[:, 0]).astype(np.int64), 0, width - 1)
    rows = np.clip(np.floor(height - xy[:, 1]).astype(np.int64), 0, height - 1)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    notHidden = depth <= buffer[rows, columns] * (1.0 + tolerance) + 1e-6
    normals = vertexNormals(vertices, triangles)
    toCamera = camera.position[None, :] - vertices
    toCamera /= np.maximum(np.linalg.norm(toCamera, axis=1, keepdims=True), 1e-12)
    return inFront & inside & notHidden & ((normals * toCamera).sum(1) > facing)


def _neighbors(triangles: np.ndarray, count: int):
    from scipy import sparse
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    adjacency = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count)).tocsr()
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float64)
    degree = np.asarray(adjacency.sum(1)).ravel()
    return adjacency, degree


def solveAlbedo(colors: List[np.ndarray], shading: List[np.ndarray], seen: List[np.ndarray], adjacency, degree,
                smoothness: float = 0.5, iterations: int = 20) -> np.ndarray:
    """ (n, c) one albedo per vertex for all the views: the least squares of color = albedo * shading over the views
    where it is seen, then smoothed toward its neighbors (`smoothness`: their weight against the data, Jacobi). """
    count = len(colors[0])
    numerator = np.zeros_like(colors[0])
    denominator = np.zeros_like(colors[0])
    for color, shade, mask in zip(colors, shading, seen):
        weight = mask[:, None].astype(np.float64)
        numerator += weight * color * shade
        denominator += weight * shade * shade
    data = denominator > 1e-9
    albedo = np.where(data, numerator / np.maximum(denominator, 1e-12), 0.0)
    known = data.any(1)
    if known.any():
        albedo[~known] = albedo[known].mean(0)
    for _ in range(iterations):
        mean = (adjacency @ albedo) / np.maximum(degree, 1.0)[:, None]
        albedo = (numerator + smoothness * denominator.mean(0, keepdims=True) * mean) / \
            np.maximum(denominator + smoothness * denominator.mean(0, keepdims=True), 1e-12)
    return albedo[:count]


def solveLight(color: np.ndarray, albedo: np.ndarray, basis: np.ndarray, mask: np.ndarray,
               regularization: float = 1e-3) -> np.ndarray:
    """ (9, c) the spherical harmonics of a view's light: color = albedo * (basis @ light), per channel, least squares
    on the seen vertices. """
    light = np.zeros((basis.shape[1], color.shape[1]))
    for channel in range(color.shape[1]):
        a = basis[mask] * albedo[mask, channel:channel + 1]
        b = color[mask, channel]
        light[:, channel] = np.linalg.solve(a.T @ a + regularization * len(b) * np.eye(basis.shape[1]), a.T @ b)
    return light


def refineIdentity(verticesOf: Callable[[int, np.ndarray], np.ndarray],
                   jacobianOf: Callable[[int, np.ndarray], np.ndarray], identity: np.ndarray, triangles: np.ndarray,
                   cameras, images: List[np.ndarray], region: np.ndarray, masks: Optional[List] = None,
                   iterations: int = 20, regularization: float = 1e-5, prior: float = 1e-5, huber: float = 0.2,
                   cancel=None, progress=None) -> dict:
    """
    The shape of a head refined on the colors of its views: its identity coefficients (a linear model) so that one
    albedo (per vertex, shared) lit by each view's light (spherical harmonics) gives the colors of every view at the
    vertices it sees.

    Args:
        verticesOf: (view, identity) -> (n, 3) the head in the world of that view.
        jacobianOf: (view, vertex indices) -> (k, 3, identity count) d vertex / d identity.
        cameras, images: per view, its camera and image (h, w, 3) float, rows from the top.
        region: (n,) the vertices compared (the skin of the face: not the eyes, the inside of the mouth, the hair).
        masks: per view (h, w) bool or None: where its skin is (a segmentation), the others not compared.
        regularization: a coefficient of 1 against this squared color error per vertex and view; prior: the same toward
        the starting identity (it was fitted on the landmarks).

    Returns:
        {"identity", "albedo" (n, 3), "lights" [(9, 3)], "error": (before, after) RMS color error}.
    """
    identity0 = np.asarray(identity, dtype=np.float64).copy()
    identity = identity0.copy()
    count = len(identity)
    adjacency, degree = _neighbors(np.asarray(triangles, dtype=np.int64), len(region))
    errors = []
    albedo = lights = None
    for iteration in range(iterations + 1):
        if cancel:
            cancel()
        colors, gradients, bases, seen = [], [], [], []
        for v, (camera, image) in enumerate(zip(cameras, images)):
            h, w = image.shape[:2]
            vertices = verticesOf(v, identity)
            mask = region & visibleVertices(vertices, triangles, camera, w, h)
            xy, _, _ = camera.project(vertices, w, h)
            color, gradient, inside = sample(image, xy)
            mask &= inside
            if masks is not None and masks[v] is not None:
                columns = np.clip(np.floor(xy[:, 0]).astype(np.int64), 0, w - 1)
                rows = np.clip(np.floor(h - xy[:, 1]).astype(np.int64), 0, h - 1)
                mask &= np.asarray(masks[v], dtype=bool)[rows, columns]
            basis = shBasis(vertexNormals(vertices, triangles))
            colors.append(color)
            gradients.append(gradient)
            bases.append(basis)
            seen.append(mask)
        # The albedo and the lights, alternately (from a white light)
        if lights is None:
            lights = [np.r_[np.ones((1, 3)) / _SH[0], np.zeros((8, 3))] for _ in cameras]
        for _ in range(3):
            shading = [basis @ light for basis, light in zip(bases, lights)]
            albedo = solveAlbedo(colors, shading, seen, adjacency, degree)
            lights = [solveLight(color, albedo, basis, mask) for color, basis, mask in zip(colors, bases, seen)]
        shading = [basis @ light for basis, light in zip(bases, lights)]
        residuals = [(color - albedo * shade)[mask] for color, shade, mask in zip(colors, shading, seen)]
        stacked = np.concatenate(residuals) if residuals else np.zeros((0, 3))
        errors.append(float(np.sqrt(np.mean(stacked ** 2))) if len(stacked) else 0.0)
        if iteration == iterations:
            break
        # The identity: the colors of the images where the vertices move (their gradients), the shading kept
        a = np.zeros((count, count))
        b = np.zeros(count)
        rows = 0
        for v, camera in enumerate(cameras):
            indices = np.where(seen[v])[0]
            if not len(indices):
                continue
            h, w = images[v].shape[:2]
            vertices = verticesOf(v, identity)
            _, _, projection = _projection(camera, vertices[indices], w, h)  # (k, 2, 3)
            jacobian = jacobianOf(v, indices)  # (k, 3, count)
            # d color / d identity: (k, 3 channels, count)
            rowsV = np.einsum("kcp,kpj,kjm->kcm", gradients[v][indices], projection, jacobian)
            residual = residuals[v]
            # Robust: a color far off (a highlight, a shadow edge, a hair) counts less
            weight = np.sqrt(np.minimum(1.0, huber / np.maximum(np.abs(residual), 1e-12)))
            rowsV = (weight[:, :, None] * rowsV).reshape(-1, count)
            residual = (weight * residual).reshape(-1)
            a += rowsV.T @ rowsV
            b += rowsV.T @ residual
            rows += len(residual)
        damping = (regularization + prior) * max(rows, 1)
        delta = np.linalg.solve(a + damping * np.eye(count),
                                -b - regularization * rows * identity - prior * rows * (identity - identity0))
        identity = identity + delta
        if progress:
            progress((iteration + 1) / float(iterations))
    return {"identity": identity, "albedo": albedo, "lights": lights, "error": (errors[0], errors[-1])}
