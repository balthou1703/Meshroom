"""FaceBuilder and the head: the canonical face of MediaPipe with ears, skull and neck following it."""

import json
import os

import numpy as np

import meshroom
from meshroom.core import pluginManager
from meshroom.core.graph import Graph
from meshroom.live import Evaluator, buildPlan, face, scene3d
from meshroom.live.actions import ActionContext
from meshroom.live.evaluator import _snapshotValue

_COMPOSITING_FOLDER = os.path.join(os.path.dirname(meshroom.__file__), "nodes", "compositing")
if not pluginManager.isNodeDescRegistered("Grade"):
    pluginManager.addPluginFromBuiltInFolder("compositing", _COMPOSITING_FOLDER)


def _procrustes(a, b):
    """ RMS distance between two point sets after the best similarity. """
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, sv, vt = np.linalg.svd(a0.T @ b0)
    scale = sv.sum() / (a0 ** 2).sum()
    return float(np.sqrt(np.mean(np.sum((scale * a0 @ (u @ vt) - b0) ** 2, axis=1))))


def actorFace():
    """ The canonical face, wider at the jaw and with a longer nose (a smooth, symmetric difference). """
    actor = face.canonicalFace().vertices.copy()
    actor[:, 0] *= 1.0 + 0.12 * np.clip(-actor[:, 1] / 10.0, 0.0, 1.0)
    actor[:, 2] += 1.2 * np.exp(-(actor[:, 0] ** 2 + (actor[:, 1] + 1.0) ** 2) / 6.0)
    return actor


def test_canonicalHead():
    head = face.canonicalHead()
    assert face.isHead(head) and len(head.vertices) == face.headVertexCount() > 4000
    # The face first, in the landmark order; the head reaches behind it (skull) and to its sides (ears)
    np.testing.assert_allclose(head.vertices[:face.MESH_VERTICES], face.canonicalFace().vertices, atol=1e-5)
    low, high = head.bounds()
    assert high[0] - low[0] > 14.0 and high[2] - low[2] > 18.0
    part = face.facePart(head)
    assert len(part.vertices) == face.MESH_VERTICES
    np.testing.assert_array_equal(np.sort(np.sort(part.triangles(), axis=1), axis=0),
                                  np.sort(np.sort(face.canonicalFace().triangles(), axis=1), axis=0))
    # Closed but at the neck: every edge has two triangles, but those of the neck
    edges = {}
    for t in head.triangles():
        for k in range(3):
            edge = tuple(sorted((t[k], t[(k + 1) % 3])))
            edges[edge] = edges.get(edge, 0) + 1
    border = [edge for edge, count in edges.items() if count == 1]
    assert head.vertices[np.unique(border)][:, 1].max() < -12.0


def test_headFollowsTheFace():
    head = face.canonicalHead()
    canonical = face.canonicalFace()
    # A rigid move with a scale: the whole head moves with it
    rotation = scene3d.composeTRS((0, 0, 0), (10.0, 30.0, -5.0))[:3, :3]
    moved = scene3d.GeoData(1.2 * canonical.vertices @ rotation.T + [1.0, 2.0, 3.0], topology=canonical.topology)
    np.testing.assert_allclose(face.headOf(moved).vertices, 1.2 * head.vertices @ rotation.T + [1.0, 2.0, 3.0], atol=1e-4)
    # The jaw opens: the back of the skull and the top of the ears stay
    opened = canonical.vertices.copy()
    low = opened[:, 1] < opened[13, 1]
    opened[low, 1] -= 2.0 * np.clip((opened[13, 1] - opened[low, 1]) / 4.0, 0.0, 1.0)
    result = face.headOf(scene3d.GeoData(opened, topology=canonical.topology)).vertices
    np.testing.assert_allclose(result[:face.MESH_VERTICES], opened, atol=1e-9)
    back = head.vertices[:, 2] < -8.0
    assert np.abs(result[back] - head.vertices[back]).max() < 1e-6
    # The head stays joined to the face oval: its border vertices on the oval edges
    data = face.headData()
    onOval = (1 - data["holeWeights"])[:, None] * opened[data["holeEdges"][:, 0]] \
        + data["holeWeights"][:, None] * opened[data["holeEdges"][:, 1]]
    np.testing.assert_allclose(result[face.MESH_VERTICES + data["hole"]], onOval, atol=1e-5)


def _views(actor, yaws, focal=50.0, size=(800, 1000)):
    width, height = size
    camera = face.viewCamera(focal, width, height)
    views = []
    for index, yaw in enumerate(yaws):
        head = scene3d.composeTRS((0.5 * index, -1.0, -45.0), (3.0, yaw, 2.0))
        xy, _, _ = camera.project(scene3d.transformPoints(head, actor), width, height)
        views.append({"pixels": xy, "size": size, "hidden": []})
    return views


def test_buildShapeFromViews():
    actor = actorFace()
    base = face.canonicalFace()
    offsets, views = face.buildShape(base, _views(actor, (-35, -15, 0, 20, 40)), focal=50.0)
    assert _procrustes(base.vertices + offsets, actor) < 0.5 * _procrustes(base.vertices, actor)
    assert all(view["error"] < 2.0 for view in views)


def test_focalEstimated():
    actor = actorFace()
    base = face.canonicalFace()
    width, height = 800, 1000
    for focal in (24.0, 85.0):
        view = _views(actor, (15,), focal=focal, size=(width, height))[0]
        estimated = face.estimateFocal(scene3d.GeoData(actor, topology=base.topology), view["pixels"], width, height)
        assert abs(np.log(estimated / focal)) < 0.25


class _PhotoLandmarker:
    """ The landmarks of the actor in the photo: its views stored by "frame" (the Constant gives no clue). """

    def __init__(self, landmarks):
        self.landmarks = list(landmarks)

    def detect(self, rgb, timestampMs):
        return self.landmarks.pop(0), {}

    def close(self):
        pass


def test_faceBuilderNode(monkeypatch):
    actor = actorFace()
    size = (800, 1000)
    views = _views(actor, (-30, 0, 30), size=size)
    normalized = [np.c_[v["pixels"][:, 0] / size[0], (size[1] - v["pixels"][:, 1]) / size[1], np.zeros(468)]
                  for v in views]
    monkeypatch.setattr(face, "landmarker", lambda minConfidence=0.5: _PhotoLandmarker(normalized))
    monkeypatch.setattr(face, "_cropAround", lambda landmarks, width, height: (0, 0, width, height))
    graph = Graph("builder")
    photos = [graph.addNewNode("Constant", width=size[0], height=size[1]) for _ in range(3)]
    builder = graph.addNewNode("FaceBuilder", estimateFocal=False, focal=50.0)
    for photo, name in zip(photos, ("input", "input1", "input2")):
        graph.addEdge(photo.output, builder.attribute(name))
    graph.update()
    values = {attr.name: _snapshotValue(attr) for attr in builder.attributes
              if not attr.isOutput and attr.type not in ("Image", "PushButtonParam")}
    evaluator = Evaluator()
    context = ActionContext(values, evaluator, buildPlan(photos[0]), 1, nodePlan=buildPlan(builder),
                            inputPlans={"input1": buildPlan(photos[1]), "input2": buildPlan(photos[2])})
    result = builder.nodeDesc.runAction("buildHead", context)
    stored = json.loads(result["views"])
    assert [view["input"] for view in stored] == ["input", "input1", "input2"]
    builder.faceShape.value, builder.views.value = result["faceShape"], result["views"]
    graph.update()
    plan = buildPlan(builder)
    head = Evaluator().data(plan, plan.target, "geoOutput", 1)
    original = Evaluator().data(plan, plan.target, "originalOutput", 1)
    assert face.isHead(head) and face.isHead(original)
    n = face.MESH_VERTICES
    assert _procrustes(head.vertices[:n], actor) < 0.5 * _procrustes(original.vertices[:n], actor)
    # The camera of the view shown sees the head where the photo has the face
    camera = Evaluator().data(plan, plan.target, "cameraOutput", 1)
    xy, _, _ = camera.project(head.vertices[:n], *size)
    assert np.median(np.hypot(*(xy - views[0]["pixels"]).T)) < 3.0
    assert Evaluator().render(plan, 1).rgba()[:, :, 2].max() > 0.5
    # A FaceTracker fed with the head tracks it (its face fitted, its ears and skull following)
    tracker = graph.addNewNode("FaceTracker")
    graph.addEdge(builder.geoOutput, tracker.face)
    graph.update()
    trackerPlan = buildPlan(tracker)
    shape = Evaluator().data(trackerPlan, trackerPlan.target, "shapeOutput", 1)
    np.testing.assert_allclose(shape.vertices, head.vertices, atol=1e-6)
    assert builder.nodeDesc.runAction("clearHead", context) == {"faceShape": "", "views": ""}


def test_rigidHeadFollowKeepsTheSkull():
    """ An expression (FaceTracker): the skull moves rigidly with the upper face, a skewed face does not shear it. """
    head = face.canonicalHead()
    canonical = face.canonicalFace().vertices
    skewed = canonical.copy()
    skewed[:, 0] += 0.3 * skewed[:, 1]
    result = face.HeadFollow.of(head.vertices).follow(skewed, rigid=True)
    back = head.vertices[:, 2] < -8.0
    a, b = head.vertices[back], result[back]
    np.testing.assert_allclose(np.linalg.norm(a - a.mean(0), axis=1), np.linalg.norm(b - b.mean(0), axis=1), atol=1e-6)


def test_rigDoesNotFoldTheFace():
    """ Wrong landmarks and pins far off around the nose and the mouth (a profile, an occlusion, pins misplaced): the
    rig stays within what a face can do and the mesh never folds, exact pins or not. """
    from meshroom.live import faceRig
    base = face.canonicalFace()
    width, height = 1000, 800
    camera = face.viewCamera(50.0, width, height)
    head = scene3d.composeTRS((0.0, -1.0, -45.0), (5.0, 55.0, 80.0))
    pixels, _, _ = camera.project(scene3d.transformPoints(head, base.vertices), width, height)
    rng = np.random.default_rng(3)
    near = np.linalg.norm(base.vertices - base.vertices[faceRig.LANDMARK["noseTip"]], axis=1) < 4.0
    pixels[near] += rng.normal(0, 25.0, (int(near.sum()), 2))
    pinned = [1, 129, 358, 13, 14, 61]
    pins = (base.vertices[pinned], pixels[pinned] + rng.normal(0, 40.0, (len(pinned), 2)))
    reference = faceRig.triangleNormals(base.vertices, base.triangles())
    for exactPins in (False, True):
        report = {}
        geo, headMatrix, _, _ = face.solveFace(base, pixels, camera, width, height, model="rig", stiffness=0.3,
                                               pins=pins, exactPins=exactPins, report=report)
        local = scene3d.transformPoints(np.linalg.inv(headMatrix), geo.vertices)
        assert (faceRig.triangleNormals(local, base.triangles()) * reference).sum(1).min() > 0.0
        params = report["rigParams"]
        assert faceRig.JAW_LIMITS[0][0] - 1e-9 <= params[-2] <= faceRig.JAW_LIMITS[0][1] + 1e-9


def test_lyingFaceIsDetectedUpright():
    """ A face lying on its side: the crop is turned so that the landmarker sees it upright, the landmarks come back
    in the image. """
    width, height = 300, 200
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    # The "face": a bright square with a mark on its top edge (upright when the mark is at the top of what is seen)
    rgb[50:150, 100:200] = 0.5
    rgb[50:60, 140:160] = 1.0
    turnedRgb = np.rot90(rgb, -1)  # Lying: the mark on the right side

    class UprightOnly:
        """ Finds the face only when its mark is at the top; its landmarks: the mark (33 and 263 its ends). """
        turns = []

        def detect(self, image, timestampMs):
            rows, columns = np.nonzero(image[:, :, 0] > 200)
            if not len(rows) or (rows.max() - rows.min()) > (columns.max() - columns.min()):
                return None, {}
            h, w = image.shape[:2]
            landmarks = np.zeros((478, 3))
            landmarks[:, 0], landmarks[:, 1] = columns.mean() / w, rows.mean() / h
            landmarks[33, 0], landmarks[263, 0] = columns.min() / w, columns.max() / w
            return landmarks, {}
    height2, width2 = turnedRgb.shape[:2]
    found, _ = face._detectUpright(UprightOnly(), np.ascontiguousarray(turnedRgb), (0, 0, width2, height2), 0)
    assert found is not None
    # The mark of the lying face: rows 140-160 of the turned image, at its right (columns 140-150)
    rows, columns = np.nonzero(turnedRgb[:, :, 0] > 0.9)
    assert abs(found[0, 0] * width2 - columns.mean()) < 2.0 and abs(found[0, 1] * height2 - rows.mean()) < 2.0
    assert abs(abs(face.faceRoll(found, width2, height2)) - 90.0) < 10.0


def test_darkFaceUnderBrightSkyIsBrightened():
    crop = np.full((100, 100, 3), 0.05, dtype=np.float32)
    crop[:10] = 1.0  # A bright sky above
    exposed = face._exposed(crop)
    assert 0.3 < exposed[50, 50, 0] < 0.5
