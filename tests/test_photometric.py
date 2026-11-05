"""The photometric refinement (meshroom/live/photometric.py): a head's shape refined on the colors of its views."""
import numpy as np

from meshroom.live import face, headFit, photometric
from meshroom.live.geoRender import rasterize


def _render(vertices, triangles, colors, camera, width, height):
    """ The vertex colors of a mesh seen by a camera (interpolated in its triangles), rows from the top. """
    xy, depth, _ = camera.project(vertices, width, height)
    u, v = xy[:, 0], height - xy[:, 1]
    _, ids = rasterize(u, v, depth, triangles, (height, width))
    image = np.zeros((height, width, 3))
    rows, columns = np.where(ids >= 0)
    t = triangles[ids[rows, columns]]
    a, b, c = (np.stack([u[t[:, i]], v[t[:, i]]], 1) for i in range(3))
    e1, e2, p = b - a, c - a, np.stack([columns + 0.5, rows + 0.5], 1) - a
    d = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
    l1 = (p[:, 0] * e2[:, 1] - e2[:, 0] * p[:, 1]) / d
    l2 = (e1[:, 0] * p[:, 1] - p[:, 0] * e1[:, 1]) / d
    image[rows, columns] = ((1 - l1 - l2)[:, None] * colors[t[:, 0]] + l1[:, None] * colors[t[:, 1]]
                            + l2[:, None] * colors[t[:, 2]])
    return image


def test_shBasisAndSampling():
    """ The irradiance of a constant light is constant; a ramp image sampled has its slope as gradient. """
    normals = np.random.default_rng(0).normal(size=(10, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    light = np.zeros((9, 1))
    light[0] = 2.0
    np.testing.assert_allclose(photometric.shBasis(normals) @ light, 2.0 * 0.886227)
    image = np.tile(np.arange(20.0)[None, :, None], (10, 1, 1))
    colors, gradients, inside = photometric.sample(image, np.array([[5.5, 4.0], [7.25, 3.0]]))
    np.testing.assert_allclose(colors[:, 0], [5.0, 6.75])
    np.testing.assert_allclose(gradients[:, 0, 0], 1.0)
    np.testing.assert_allclose(gradients[:, 0, 1], 0.0)
    assert inside.all()


def test_identityRefinedOnTheColors():
    """ A head (the identity model) seen in three views with a skin texture and a light: from an identity a few
    millimeters off, the refined one is nearer the truth. """
    from tests.test_faceBuilder import _headViews
    head = face.canonicalHead()
    triangles = head.triangles()
    count = len(head.vertices)
    modes = headFit.identityModel()
    rng = np.random.default_rng(1)
    truth = rng.normal(0, 1, len(modes))

    def vertices(coefficients):
        return head.vertices + np.tensordot(coefficients, modes, axes=1)
    actor = vertices(truth)
    albedo = 0.6 + 0.15 * np.sin(head.vertices[:, [0]] * 1.3 + [0, 1, 2]) + 0.1 * rng.random((count, 1))
    light = np.zeros((9, 3))
    light[0], light[3], light[1] = 1 / 0.886227, 0.4, 0.2
    size = (640, 480)
    cameras = _headViews((0, 35, -35), size)
    shading = photometric.shBasis(photometric.vertexNormals(actor, triangles)) @ light
    images = [_render(actor, triangles, albedo * shading, camera, *size) for camera in cameras]
    region = np.zeros(count, bool)
    region[:face.MESH_VERTICES] = True
    start = truth + rng.normal(0, 0.6, len(modes))
    result = photometric.refineIdentity(lambda v, c: vertices(c), lambda v, indices: np.moveaxis(modes[:, indices], 0, 2),
                                        start, triangles, cameras, images, region, iterations=12)
    before = np.linalg.norm(vertices(start) - actor, axis=1)[:face.MESH_VERTICES].mean()
    after = np.linalg.norm(vertices(result["identity"]) - actor, axis=1)[:face.MESH_VERTICES].mean()
    print(f"photometric: color {result['error'][0]:.4f} -> {result['error'][1]:.4f}, face {before:.3f} -> {after:.3f} cm")
    assert result["error"][1] < result["error"][0] and after < 0.7 * before
