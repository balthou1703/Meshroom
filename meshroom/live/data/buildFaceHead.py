"""
Build faceHead.npz: the ears, skull and neck added to the canonical face of MediaPipe (the "head" of the FaceTracker
and FaceBuilder), from the generic neutral head of ICT-FaceKit (MIT license, see LICENSE-ICT-FaceKit.md).

    python -m meshroom.live.data.buildFaceHead canonical_face_model.obj generic_neutral_mesh.obj

- canonical_face_model.obj: mediapipe/modules/face_geometry/data/ (Apache 2.0, github.com/google-ai-edge/mediapipe)
- generic_neutral_mesh.obj: FaceXModel/ of github.com/USC-ICT/ICT-FaceKit

The ICT head (its face and its head and neck geometries) is aligned on the MediaPipe face (68 landmarks, similarity:
both are in centimeters, Y up, looking along +Z), the part covered by the MediaPipe face (inside its face oval) and the
neck under `NECK_CUT` are removed, and the rest is warped (harmonic, within 4 cm) so that the border of the hole lies on
the face oval: the border vertex nearest to each oval vertex is merged with it (the head triangles use the oval
vertices of the face), the others are on an edge of the oval (`hole`: its two oval vertices and the weight of the
second one), which keeps the head closed whatever the face does.

Saved: "faceVertices" (468, 3) and "faceTriangles" of the canonical face (MediaPipe, Apache 2.0, see
LICENSE-MediaPipe.md), "vertices" (m, 3) of the head after the 468 face vertices, "triangles" (k, 3) indexing face +
head vertices, "hole" (h,) head vertices (0 = vertex 468) on the face oval edges, "holeEdges" (h, 2) their oval vertices, "holeWeights" (h,).
"""

import os
import sys

import numpy as np

# Multi-PIE 68 landmarks: in the ICT head (its README) and in the MediaPipe face
ICT_68 = [1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924, 2608, 3272, 4088, 3443, 268, 493,
          1914, 2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942, 4857, 1140, 2075, 1147, 4269, 3360, 1507, 1542,
          1537, 1528, 1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695, 2081, 0, 4275, 6200, 6213, 6346, 6461,
          5518, 5957, 5841, 5702, 5711, 5533, 6216, 6207, 6470, 5517, 5966]
MEDIAPIPE_68 = [162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389, 71, 63, 105, 66, 107,
                336, 296, 334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305, 33, 160, 158, 133, 153, 144, 362, 385, 387,
                263, 373, 380, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87]
# The ICT face and head and neck geometries (its README): vertices [0:11247], polygons [0:11143]
ICT_HEAD_VERTICES = 11248
ICT_HEAD_POLYGONS = 11144
NECK_CUT = -13.0
WARP_DISTANCE = 4.0
# Center of the cylinder used to tell the inside of the face oval
CENTER = np.array([0.0, 0.0, -2.0])


def readObj(path):
    vertices, polygons = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                polygons.append([int(token.split("/")[0]) - 1 for token in line.split()[1:]])
    return np.array(vertices), polygons


def triangulate(polygons):
    return np.array([[p[0], p[i], p[i + 1]] for p in polygons for i in range(1, len(p) - 1)], dtype=np.int64)


def similarity(source, target):
    """ (scale, rotation, translation) best mapping `source` points on `target` points. """
    sourceMean, targetMean = source.mean(0), target.mean(0)
    a, b = source - sourceMean, target - targetMean
    u, s, vt = np.linalg.svd(b.T @ a)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ d @ vt
    scale = (s * np.diag(d)).sum() / (a ** 2).sum()
    return scale, rotation, targetMean - scale * rotation @ sourceMean


def boundaryLoops(triangles):
    """ The boundary loops of a mesh, each in the direction of its border edges. """
    edges = {(t[i], t[(i + 1) % 3]) for t in triangles for i in range(3)}
    following = {a: b for a, b in edges if (b, a) not in edges}
    loops, seen = [], set()
    for start in list(following):
        if start in seen:
            continue
        loop, current = [start], following[start]
        seen.add(start)
        while current != start:
            loop.append(current)
            seen.add(current)
            current = following[current]
        loops.append(loop)
    return loops


def cylinder(points):
    """ (angle around the vertical axis, height): where the face oval is a simple polygon. """
    d = points - CENTER
    return np.c_[np.arctan2(d[:, 0], d[:, 2]), points[:, 1]]


def insidePolygon(points, polygon):
    """ Even-odd test of 2D points in a 2D polygon. """
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    x0, y0 = polygon[:, 0][None], polygon[:, 1][None]
    x1, y1 = np.roll(polygon[:, 0], -1)[None], np.roll(polygon[:, 1], -1)[None]
    crosses = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / np.where(y1 != y0, y1 - y0, 1e-12) + x0)
    return crosses.sum(1) % 2 == 1


def build(facePath, ictPath):
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from scipy.spatial import cKDTree
    face, facePolygons = readObj(facePath)
    faceTriangles = triangulate(facePolygons)
    ict, ictPolygons = readObj(ictPath)
    scale, rotation, translation = similarity(ict[ICT_68[17:]], face[MEDIAPIPE_68[17:]])
    vertices = (scale * (rotation @ ict[:ICT_HEAD_VERTICES].T)).T + translation
    quads = np.array(ictPolygons[:ICT_HEAD_POLYGONS])

    # Remove the part under the MediaPipe face and the neck
    oval = max(boundaryLoops(faceTriangles), key=len)
    inside = insidePolygon(cylinder(vertices), cylinder(face[oval])) & ((vertices - CENTER)[:, 2] > 0)
    keep = ~(inside[quads].any(1) | (vertices[quads].mean(1)[:, 1] < NECK_CUT))
    # A border touching itself at a vertex (two border edges leaving it): the kept polygons there are removed too, until
    # the border of the hole is a simple loop
    while True:
        outgoing = {}
        polygons = quads[keep]
        edges = {(int(p[i]), int(p[(i + 1) % 4])) for p in polygons for i in range(4)}
        for a, b in edges:
            if (b, a) not in edges:
                outgoing[a] = outgoing.get(a, 0) + 1
        pinched = [vertex for vertex, count in outgoing.items() if count > 1]
        if not pinched:
            break
        keep &= ~np.isin(quads, pinched).any(1)
    triangles = triangulate(quads[keep].tolist())
    # Without the polygons repeated in the ICT mesh (same vertices)
    _, unique = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)
    triangles = triangles[np.sort(unique)]
    used = np.unique(triangles)
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[used] = np.arange(len(used))
    head, triangles = vertices[used], remap[triangles]
    loops = sorted(boundaryLoops(triangles), key=lambda loop: head[loop][:, 1].mean())
    neck, hole = loops[0], loops[-1]
    head[neck, 1] = head[neck, 1].mean()

    # Each hole vertex on the oval edge at the same angle around the oval
    ovalCenter = cylinder(face[oval]).mean(0)

    def angles(points):
        uv = cylinder(points) - ovalCenter
        return np.arctan2(uv[:, 1], uv[:, 0])

    def increasing(loop, points):
        if np.unwrap(angles(points[loop]))[-1] < np.unwrap(angles(points[loop]))[0]:
            loop = loop[::-1]
        first = int(np.argmin(np.mod(angles(points[loop]), 2 * np.pi)))
        loop = list(loop[first:]) + list(loop[:first])
        unwrapped = np.unwrap(angles(points[loop]))
        return loop, unwrapped - unwrapped[0] + np.mod(unwrapped[0], 2 * np.pi)

    oval, ovalAngles = increasing(oval, face)
    hole, holeAngles = increasing(hole, head)
    # On the turn of the oval: from its first vertex
    holeAngles = ovalAngles[0] + np.mod(holeAngles - ovalAngles[0], 2 * np.pi)
    # Monotonic along the loop (the border of the hole zigzags a little): the vertices keep their order on the oval
    first = int(np.argmin(holeAngles))
    hole, holeAngles = list(hole[first:]) + list(hole[:first]), np.r_[holeAngles[first:], holeAngles[:first]]
    holeAngles = (np.maximum.accumulate(holeAngles) + np.minimum.accumulate(holeAngles[::-1])[::-1]) / 2.0
    holeAngles = holeAngles + np.linspace(0.0, 1e-4, len(holeAngles))
    # Each oval vertex takes the hole vertex at its angle (they are merged: the head shares the oval vertices of the
    # face); the hole vertices between two of them go on their oval edge, spread as along the border of the hole
    count = len(hole)
    anchors = []
    for j, angle in enumerate(ovalAngles):
        k = int(np.argmin(np.abs(holeAngles - angle)))
        if anchors and k <= anchors[-1]:
            k = anchors[-1] + 1
        anchors.append(k)
    if anchors[-1] >= count:
        raise RuntimeError("The border of the hole has too few vertices for the face oval.")
    merged = {int(hole[k]): int(oval[j]) for j, k in enumerate(anchors)}
    onEdge, holeEdges, weights = [], [], []
    for j, k in enumerate(anchors):
        end = anchors[(j + 1) % len(anchors)] + (count if j == len(anchors) - 1 else 0)
        between = [hole[i % count] for i in range(k, end + 1)]
        steps = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(head[between], axis=0), axis=1))]
        for vertex, position in zip(between[1:-1], steps[1:-1] / steps[-1]):
            onEdge.append(int(vertex))
            holeEdges.append([int(oval[j]), int(oval[(j + 1) % len(oval)])])
            weights.append(float(position))
    holeEdges, weights = np.array(holeEdges, dtype=np.int64), np.array(weights)
    targets = {vertex: face[faceVertex] for vertex, faceVertex in merged.items()}
    targets.update({vertex: (1 - w) * face[a] + w * face[b] for vertex, (a, b), w in zip(onEdge, holeEdges, weights)})

    # Warp the head near the hole (harmonic), the rest stays
    edges = np.unique(np.sort(np.r_[triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=1), axis=0)
    total = len(head)
    adjacency = sparse.coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                                  shape=(total, total)).tocsr()
    laplacian = (sparse.diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency).tocsr()
    distance, _ = cKDTree(head[hole]).query(head)
    fixed = distance > WARP_DISTANCE
    fixed[hole] = True
    shifts = np.zeros((total, 3))
    for vertex, target in targets.items():
        shifts[vertex] = target - head[vertex]
    free, known = np.where(~fixed)[0], np.where(fixed)[0]
    shifts[free] = spsolve(laplacian[free][:, free].tocsc(), -(laplacian[free][:, known] @ shifts[known]))
    head = head + shifts

    # Without the merged vertices: the triangles index the face (its oval vertices) and the head after it
    kept = np.array([v for v in range(total) if v not in merged], dtype=np.int64)
    index = -np.ones(total, dtype=np.int64)
    index[kept] = len(face) + np.arange(len(kept))
    for vertex, faceVertex in merged.items():
        index[vertex] = faceVertex
    return {"faceVertices": face.astype(np.float32), "faceTriangles": faceTriangles.astype(np.int32),
            "vertices": head[kept].astype(np.float32), "triangles": index[triangles].astype(np.int32),
            "hole": (index[onEdge] - len(face)).astype(np.int32), "holeEdges": holeEdges.astype(np.int32),
            "holeWeights": weights.astype(np.float32)}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    data = build(sys.argv[1], sys.argv[2])
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faceHead.npz")
    np.savez_compressed(path, **data)
    print(f"{path}: {len(data['vertices'])} vertices, {len(data['triangles'])} triangles")
