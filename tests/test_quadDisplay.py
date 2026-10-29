import numpy as np

from meshroom.live import geoRender, scene3d


def test_quadEdgesHideTheDiagonalsOfAGrid():
    # A 3x3 grid of quads, each split into two triangles
    size = 4
    xs, ys = np.meshgrid(np.arange(size, dtype=float), np.arange(size, dtype=float))
    vertices = np.c_[xs.ravel(), ys.ravel(), np.zeros(size * size)]
    triangles = []
    for row in range(size - 1):
        for column in range(size - 1):
            a, b = row * size + column, row * size + column + 1
            c, d = a + size, b + size
            triangles += [(a, b, d), (a, d, c)]
    geo = scene3d.GeoData(vertices, np.full(len(triangles), 3), np.array(triangles).ravel())
    edges = geoRender.quadEdges(geo)
    assert len(geo.edges()) == 33
    # 24 grid edges, no diagonal
    assert len(edges) == 24
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    assert np.allclose(lengths, 1.0)


def test_quadEdgesKeepPolygonsAndSkinnyPairs():
    # A quad face and a triangle making a very skewed pair with nothing: all edges kept
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [5, 0.5, 0]], float)
    geo = scene3d.GeoData(vertices, np.array([4, 3]), np.array([0, 1, 2, 3, 1, 4, 2]))
    assert len(geoRender.quadEdges(geo)) == len(geo.edges())


def test_faceTrackKeepsSurfacePoints():
    from meshroom.live import face
    points = np.array([[10.0, 20.0, 1.0], [30.5, 40.25, 0.0]])
    faces = {1001: {"landmarks": np.zeros((468, 3)), "blendshapes": {}, "surface": points}}
    parsed, _ = face.parseFaces(face.dumpFaces(faces, (100, 100)))
    assert np.allclose(parsed[1001]["surface"], [[10.0, 20.0, 1.0], [30.5, 40.2, 0.0]], atol=0.06)
