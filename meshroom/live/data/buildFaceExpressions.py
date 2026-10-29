"""
Build faceExpressions.npz: the ARKit expression shapes (blendshapes) of the head of the FaceTracker (data/faceHead.npz:
the MediaPipe face with ears, skull and neck), transferred from the expression shapes of ICT-FaceKit (MIT license, see
LICENSE-ICT-FaceKit.md).

    python -m meshroom.live.data.buildFaceExpressions path/to/ICT-FaceKit/FaceXModel

The ICT head is aligned on the MediaPipe face (similarity on the 68 landmarks, as buildFaceHead.py), then warped onto it
(thin plate spline) by the 68 landmarks and the contours of the eyelids and the inner lips (matched by arc length
between their corners); each vertex of our head takes the expression moves of its closest point on the warped ICT head,
on the same side of the eye and mouth slits (an upper lid vertex follows the upper lid). The contour vertices follow
their contour point exactly. The moves are then smoothed a little over our mesh (its coarser triangles), the contours
kept, the blinks scaled so that they close our eyes (the canonical MediaPipe eye is more open than the ICT one), and
the head vertices on the face oval follow their oval edge (the head stays closed).

ICT names the sides "_L" / "_R" and splits browInnerUp and cheekPuff: the shapes are saved with the ARKit names
(eyeBlinkLeft, ...; the left of the face, +X), the split ones summed. ARKit's tongueOut has no ICT shape.

Saved: "names" (51,) and "deltas" (51, vertices, 3) float16, centimeters, in the space of the canonical head.
"""

import json
import os
import sys

import numpy as np

from meshroom.live.data import buildFaceHead

# The contours of MediaPipe: (lower arc, upper arc), both from the same corner to the other
EYE_RIGHT = ([33, 7, 163, 144, 145, 153, 154, 155, 133], [33, 246, 161, 160, 159, 158, 157, 173, 133])
EYE_LEFT = ([263, 249, 390, 373, 374, 380, 381, 382, 362], [263, 466, 388, 387, 386, 385, 384, 398, 362])
LIPS = ([78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308], [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308])
CONTOURS = (EYE_RIGHT, EYE_LEFT, LIPS)
# The middle of the lids, to close the eyes with the blinks: (shape, upper lid, lower lid)
BLINKS = (("eyeBlink_R", 159, 145), ("eyeBlink_L", 386, 374))
# The ARKit shapes (without tongueOut), in the order of ARKit
ARKIT = ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
         "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
         "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
         "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
         "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
         "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
         "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
         "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
         "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"]
SMOOTHING_STEPS = 4


def readVertices(path, count):
    """ The first `count` vertices of an OBJ file. """
    vertices = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
                if len(vertices) == count:
                    break
    return np.array(vertices)


def arkitName(ictName):
    """ The ARKit name of an ICT shape (browInnerUp_L: browInnerUp, its two halves are summed). """
    for suffix, side in (("_L", "Left"), ("_R", "Right")):
        if ictName.endswith(suffix):
            base = ictName[:-2]
            return base if base in ("browInnerUp", "cheekPuff") else base + side
    return ictName


def arcs(loop, points, startCorner, endCorner):
    """ The two arcs of a closed loop between its vertices nearest to two corners: (lower, upper), from the start. """
    loop = list(loop)
    a = loop[int(np.argmin(np.linalg.norm(points[loop] - startCorner, axis=1)))]
    b = loop[int(np.argmin(np.linalg.norm(points[loop] - endCorner, axis=1)))]
    i, j = loop.index(a), loop.index(b)
    one = [loop[(i + k) % len(loop)] for k in range((j - i) % len(loop) + 1)]
    other = [loop[(i - k) % len(loop)] for k in range((i - j) % len(loop) + 1)]
    return (one, other) if points[one, 1].mean() < points[other, 1].mean() else (other, one)


def arcFractions(points, arc):
    """ The position of each vertex along an arc, in [0, 1] of its length. """
    steps = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points[arc], axis=0), axis=1))]
    return steps / max(steps[-1], 1e-12)


def pointsAlong(points, arc, fractions):
    """ (vertex a, vertex b, weight of b) of the points at `fractions` of the length of an arc. """
    along = arcFractions(points, arc)
    result = []
    for fraction in fractions:
        k = int(np.clip(np.searchsorted(along, fraction, side="right") - 1, 0, len(arc) - 2))
        weight = (fraction - along[k]) / max(along[k + 1] - along[k], 1e-12)
        result.append((arc[k], arc[k + 1], float(np.clip(weight, 0.0, 1.0))))
    return result


def closestOnTriangles(p, a, b, c):
    """ The closest points of points `p` on triangles (a, b, c) (all (m, 3)), and their barycentric weights. """
    ab, ac = b - a, c - a
    d1, d2 = (ab * (p - a)).sum(1), (ac * (p - a)).sum(1)
    d3, d4 = (ab * (p - b)).sum(1), (ac * (p - b)).sum(1)
    d5, d6 = (ab * (p - c)).sum(1), (ac * (p - c)).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def ratio(x, y):
        return x / np.where(np.abs(y) > 1e-30, y, 1e-30)
    v, w = ratio(vb, va + vb + vc), ratio(vc, va + vb + vc)
    weights = np.c_[1.0 - v - w, v, w]
    # The vertex and edge regions (Ericson, Real-Time Collision Detection)
    done = np.zeros(len(p), dtype=bool)
    for region, values in (((d1 <= 0) & (d2 <= 0), lambda: np.c_[np.ones_like(d1), 0 * d1, 0 * d1]),
                           ((d3 >= 0) & (d4 <= d3), lambda: np.c_[0 * d1, np.ones_like(d1), 0 * d1]),
                           ((vc <= 0) & (d1 >= 0) & (d3 <= 0),
                            lambda: np.c_[1 - ratio(d1, d1 - d3), ratio(d1, d1 - d3), 0 * d1]),
                           ((d6 >= 0) & (d5 <= d6), lambda: np.c_[0 * d1, 0 * d1, np.ones_like(d1)]),
                           ((vb <= 0) & (d2 >= 0) & (d6 <= 0),
                            lambda: np.c_[1 - ratio(d2, d2 - d6), 0 * d1, ratio(d2, d2 - d6)]),
                           ((va <= 0) & (d4 >= d3) & (d5 >= d6),
                            lambda: np.c_[0 * d1, 1 - ratio(d4 - d3, d4 - d3 + d5 - d6), ratio(d4 - d3, d4 - d3 + d5 - d6)])):
        region = region & ~done
        weights[region] = values()[region]
        done |= region
    return weights[:, :1] * a + weights[:, 1:2] * b + weights[:, 2:] * c, weights


def slitSides(points, face):
    """ 0 away from the eyes and the mouth; near one (+-1 right eye, +-2 left eye, +-3 mouth): + above its slit (the
    line through its corners and the middle of its contour), - below. """
    sides = np.zeros(len(points), dtype=np.int64)
    for code, (lower, upper), margin in ((1, EYE_RIGHT, 1.2), (2, EYE_LEFT, 1.2), (3, LIPS, 1.6)):
        first, last = face[lower[0]], face[lower[-1]]
        low, high = sorted([first[0], last[0]])
        center = 0.5 * (first + last)
        near = (points[:, 0] > low - 0.4) & (points[:, 0] < high + 0.4) & (np.abs(points[:, 1] - center[1]) < margin) \
            & (points[:, 2] > center[2] - 1.5)
        middle = 0.5 * (face[lower].mean(0) + face[upper].mean(0))
        line = np.array(sorted([(first[0], first[1]), (middle[0], middle[1]), (last[0], last[1])]))
        side = np.where(points[:, 1] >= np.interp(points[:, 0], line[:, 0], line[:, 1]), code, -code)
        sides[near] = side[near]
    return sides


def build(ictFolder):
    from scipy import sparse
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import cKDTree
    from meshroom.live import face as faceModule
    with open(os.path.join(ictFolder, "vertex_indices.json")) as f:
        ictNames = json.load(f)["expressions"]
    count = buildFaceHead.ICT_HEAD_VERTICES
    neutral, polygons = buildFaceHead.readObj(os.path.join(ictFolder, "generic_neutral_mesh.obj"))
    neutral = neutral[:count]
    head = faceModule.canonicalHead()
    vertices, face = head.vertices, head.vertices[:faceModule.MESH_VERTICES]
    scale, rotation, translation = buildFaceHead.similarity(neutral[buildFaceHead.ICT_68[17:]],
                                                            face[buildFaceHead.MEDIAPIPE_68[17:]])
    aligned = (scale * (rotation @ neutral.T)).T + translation
    ictDeltas = np.stack([(scale * (rotation @ (readVertices(os.path.join(ictFolder, f"{name}.obj"), count) - neutral).T)).T
                          for name in ictNames])
    triangles = buildFaceHead.triangulate(polygons[:buildFaceHead.ICT_HEAD_POLYGONS])
    loops = buildFaceHead.boundaryLoops(triangles)

    # The warp of the ICT head onto the MediaPipe face: the contours (by arc length), the other 68 landmarks
    sources, targets, onContour = [], [], {}
    for lower, upper in CONTOURS:
        center = face[lower + upper].mean(0)
        loop = min(loops, key=lambda loop: np.linalg.norm(aligned[loop].mean(0) - center))
        ictLower, ictUpper = arcs(loop, aligned, face[lower[0]], face[lower[-1]])
        for contour, ictArc in ((lower, ictLower), (upper, ictUpper)):
            for vertex, (a, b, w) in zip(contour, pointsAlong(aligned, ictArc, arcFractions(face, contour))):
                onContour[vertex] = (a, b, w)
                sources.append((1 - w) * aligned[a] + w * aligned[b])
                targets.append(face[vertex])
    contourVertices = {v for lower, upper in CONTOURS for v in lower + upper}
    for ictIndex, faceIndex in zip(buildFaceHead.ICT_68, buildFaceHead.MEDIAPIPE_68):
        if faceIndex not in contourVertices:
            sources.append(aligned[ictIndex])
            targets.append(face[faceIndex])
    sources, targets = np.array(sources), np.array(targets)
    warp = RBFInterpolator(sources, targets - sources, kernel="thin_plate_spline", smoothing=1e-3)
    # Faded out away from the face (the skull of our head is the ICT one, only aligned)
    distance, _ = cKDTree(face).query(aligned)
    warped = aligned + np.clip(1.0 - (distance - 1.0) / 3.0, 0.0, 1.0)[:, None] * warp(aligned)

    # Each vertex on its closest point of the warped ICT head, on the same side of the slits
    triangleSides = slitSides(warped[triangles].mean(1), face)
    sides = slitSides(vertices, face)
    sides[faceModule.MESH_VERTICES:] = 0
    _, near = cKDTree(warped[triangles].mean(1)).query(vertices, k=48)
    corners = np.zeros((len(vertices), 3), dtype=np.int64)
    weights = np.zeros((len(vertices), 3))
    for i, point in enumerate(vertices):
        candidates = near[i][triangleSides[near[i]] != -sides[i]] if sides[i] else near[i]
        closest, w = closestOnTriangles(np.repeat(point[None], len(candidates), 0),
                                        *(warped[triangles[candidates, j]] for j in range(3)))
        k = int(np.argmin(np.linalg.norm(closest - point, axis=1)))
        corners[i], weights[i] = triangles[candidates[k]], w[k]
    for vertex, (a, b, w) in onContour.items():
        corners[vertex], weights[vertex] = (a, b, b), (1.0 - w, w, 0.0)
    deltas = np.einsum("vj,kvjd->kvd", weights, ictDeltas[:, corners])

    # Smoothed over our mesh (its triangles are coarser than the ICT ones), the contours kept
    edges = head.edges()
    total = len(vertices)
    pairs = (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])
    adjacency = sparse.coo_matrix((np.ones(2 * len(edges)), pairs), shape=(total, total)).tocsr()
    adjacency.data[:] = 1.0
    average = sparse.diags(1.0 / np.maximum(np.asarray(adjacency.sum(1)).ravel(), 1.0)) @ adjacency
    kept = sorted(contourVertices)
    for k in range(len(deltas)):
        moves = deltas[k]
        for _ in range(SMOOTHING_STEPS):
            smoothed = 0.5 * moves + 0.5 * (average @ moves)
            smoothed[kept] = moves[kept]
            moves = smoothed
        deltas[k] = moves
    # The blinks close our eyes: the middle of the upper lid down to the height of the lower one
    for name, up, low in BLINKS:
        k = ictNames.index(name)
        opening = vertices[up, 1] - vertices[low, 1]
        closing = deltas[k][low, 1] - deltas[k][up, 1]
        deltas[k] *= opening / max(closing, 1e-6)
    # The head vertices on the face oval stay on their oval edge
    data = faceModule.headData()
    for vertex, (a, b), w in zip(data["hole"], data["holeEdges"], data["holeWeights"]):
        deltas[:, faceModule.MESH_VERTICES + vertex] = (1 - w) * deltas[:, a] + w * deltas[:, b]

    # The ARKit shapes
    shapes = {}
    for name, moves in zip(ictNames, deltas):
        target = arkitName(name)
        shapes[target] = shapes[target] + moves if target in shapes else moves
    missing = [name for name in ARKIT if name not in shapes]
    if missing:
        raise RuntimeError(f"ICT shapes missing: {missing}")
    return {"names": np.array(ARKIT), "deltas": np.stack([shapes[name] for name in ARKIT]).astype(np.float16)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    result = build(sys.argv[1])
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faceExpressions.npz")
    np.savez_compressed(path, **result)
    print(f"{path}: {len(result['names'])} shapes of {result['deltas'].shape[1]} vertices")
