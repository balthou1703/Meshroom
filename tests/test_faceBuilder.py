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
    np.testing.assert_allclose(shape.vertices, head.vertices, atol=1e-3)
    assert builder.nodeDesc.runAction("clearHead", context) == {"faceShape": "", "views": "", "headFit": ""}


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
    crop[40:60, 40:60] = 0.2  # The face, in the shade
    crop[:10] = 1.0  # A bright sky at the border
    exposed = face._exposed(crop)
    assert 0.8 < exposed[50, 50, 0] <= 0.91


def _texture(points):
    x, y, z = points.T
    value = np.sin(2.1 * x + 1.3 * y) * np.sin(1.7 * z - 2.1 * x) + np.sin(2.3 * y + 1.9 * z) * np.sin(2.1 * x - 1.2 * z) \
        + 0.5 * np.sin(4.7 * x - 3.3 * y + 3.1 * z)
    return (0.5 + 0.2 * value).astype(np.float32)


def _renderFace(mesh, cameraToFace, camera, width, height):
    """ The textured face (its texture fixed on the face) seen by the camera: a z-buffer over its triangles. """
    from meshroom.live import ImageBuffer, geotrack
    focal = camera.focalPixels(width)
    xy, depth = geotrack.projectPoints(cameraToFace, mesh.vertices, width, height, focal)
    lum = np.full((height, width), 0.05, np.float32)
    zbuffer = np.full((height, width), np.inf)
    for a, b, c in mesh.triangles():
        p = xy[[a, b, c]]
        x0, x1 = int(max(0, np.floor(p[:, 0].min()))), int(min(width - 1, np.ceil(p[:, 0].max())))
        y0, y1 = int(max(0, np.floor(p[:, 1].min()))), int(min(height - 1, np.ceil(p[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1])
        if abs(d) < 1e-12:
            continue
        w0 = ((p[1, 1] - p[2, 1]) * (gx - p[2, 0]) + (p[2, 0] - p[1, 0]) * (gy - p[2, 1])) / d
        w1 = ((p[2, 1] - p[0, 1]) * (gx - p[2, 0]) + (p[0, 0] - p[2, 0]) * (gy - p[2, 1])) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * depth[a] + w1 * depth[b] + w2 * depth[c]
        # Nuke y up: row = height - 1 - y
        rows, columns = (height - 1 - (gy - 0.5)).astype(int)[inside], (gx - 0.5).astype(int)[inside]
        z = z[inside]
        point = (w0[inside, None] * mesh.vertices[a] + w1[inside, None] * mesh.vertices[b] + w2[inside, None] * mesh.vertices[c])
        nearer = z < zbuffer[rows, columns]
        zbuffer[rows[nearer], columns[nearer]] = z[nearer]
        lum[rows[nearer], columns[nearer]] = _texture(point[nearer])
    return ImageBuffer.fromRGBA(np.stack([lum, lum, lum, np.ones_like(lum)], -1), (width, height))


def test_headPoseFollowedByTheSurface():
    """ The head turning: its pose followed by the textured surface of its rigid parts (forehead, nose bridge...). """
    from meshroom.live import geotrack
    mesh = face.canonicalFace()
    width, height = 320, 240
    camera = scene3d.CameraData(np.eye(4), 50.0, 36.0, 27.0)

    def poseAt(frame):
        # camera-to-face: the inverse of the head pose (head at 45 cm, turning and nodding)
        head = scene3d.composeTRS((0.3 * frame, 0.0, -45.0), (1.0 * frame, 2.0 * frame, 0.0))
        return np.linalg.inv(head)
    tracker = face.HeadPoseTracker(mesh, camera, width, height, method="flow")
    images = {f: _renderFace(mesh, poseAt(f), camera, width, height) for f in range(7)}
    pose = poseAt(0)
    for f in range(1, 7):
        pose = tracker.step(images[f - 1], images[f], pose, f - 1, f)
        # One in ten of the followed points kept for display, most of them fitting the pose
        assert 5 <= len(tracker.points) and tracker.points[:, 2].mean() > 0.7
        assert np.all((tracker.points[:, 0] >= 0) & (tracker.points[:, 0] <= width))
    focal = camera.focalPixels(width)
    xy, _ = geotrack.projectPoints(pose, mesh.vertices, width, height, focal)
    truth, _ = geotrack.projectPoints(poseAt(6), mesh.vertices, width, height, focal)
    assert np.abs(xy - truth).max() < 1.5


def _plateHeadPose(frame):
    """ camera-to-face of the synthetic plate: the head 45 cm away, turning and nodding. """
    return np.linalg.inv(scene3d.composeTRS((0.3 * frame, 0.0, -45.0), (1.0 * frame, 2.0 * frame, 0.0)))


from meshroom.core import desc as _desc  # noqa: E402
from meshroom.live import ImageInfo as _ImageInfo  # noqa: E402


class SyntheticFacePlate(_desc.ImageOpNode):
    """ The textured canonical face turning (_plateHeadPose), seen by the default camera of the FaceTracker. """
    category = "Test"
    inputs = []
    outputs = [_desc.Image(name="output", label="Output", description="")]

    def getInfo(self, ctx, inputsInfo):
        return _ImageInfo(360, 240, 0, 16)

    def render(self, ctx, inputs):
        camera = scene3d.CameraData(np.eye(4), 35.0, 36.0, 24.0)
        image = _renderFace(face.canonicalFace(), _plateHeadPose(ctx.frame), camera, 360, 240)
        # A noisy plate (grain, as in a dark shot)
        rgba = np.array(image.rgba(), dtype=np.float32)
        rgba[:, :, :3] += np.random.default_rng(ctx.frame).normal(0, 0.03, rgba[:, :, :3].shape).astype(np.float32)
        return image.withRGBA(rgba)


import pytest  # noqa: E402


@pytest.mark.parametrize("followSurface, model", [(True, "arkit"), (False, "arkit"), (True, "rig")])
def test_faceTrackerFollowsTheHeadSurface(followSurface, model):
    """ Track Face (flow) with "Follow Head Surface": the pose of each frame from the surface, kept with the frame. """
    from tests.utils import registerNodeDesc, unregisterNodeDesc
    from meshroom.live import geotrack
    registerNodeDesc(SyntheticFacePlate)
    try:
        width, height = 360, 240
        camera = scene3d.CameraData(np.eye(4), 35.0, 36.0, 24.0)
        canonical = face.canonicalFace()
        graph = Graph("surface")
        plate = graph.addNewNode("SyntheticFacePlate")
        tracker = graph.addNewNode("FaceTracker", trackWith="flow", temporalSmoothing=0.0, smoothing=0.0,
                                   meshType="face", followSurface=followSurface, deformModel=model)
        graph.addEdge(plate.output, tracker.input)
        xy, _ = geotrack.projectPoints(_plateHeadPose(0), canonical.vertices, width, height,
                                       camera.focalPixels(width))
        start = np.c_[xy[:, 0] / width, (height - xy[:, 1]) / height, np.zeros(len(xy))]
        tracker.faceTrack.value = face.dumpFaces({0: {"landmarks": start, "blendshapes": {}}}, (width, height))
        graph.update()
        values = {attr.name: _snapshotValue(attr) for attr in tracker.attributes
                  if not attr.isOutput and attr.type not in ("Image", "PushButtonParam")}
        context = ActionContext(values, Evaluator(), buildPlan(plate), 0, nodePlan=buildPlan(tracker))
        result = tracker.nodeDesc.runAction("trackFaceForward", context)
        faces, _ = face.parseFaces(result["faceTrack"])
        assert (faces[14]["pose"] is not None) == followSurface
        # Some of the points followed on the surface kept for display
        assert (faces[14].get("surface") is not None) == followSurface
        tracker.faceTrack.value = result["faceTrack"]
        graph.update()
        plan = buildPlan(tracker)
        head = Evaluator().data(plan, plan.target, "headMatrix", 14)
        solved, _ = geotrack.projectPoints(np.linalg.inv(head), canonical.vertices, width, height, camera.focalPixels(width))
        truth, _ = geotrack.projectPoints(_plateHeadPose(14), canonical.vertices, width, height, camera.focalPixels(width))
        assert np.median(np.hypot(*(solved - truth).T)) < 1.5
        # The face (no expression on the plate): the rig keeps it near the truth
        geo = Evaluator().data(plan, plan.target, "geoOutput", 14)
        fitted, _, _ = camera.project(geo.vertices[:face.MESH_VERTICES], width, height)
        assert np.median(np.hypot(*(fitted - truth).T)) < 1.5
    finally:
        unregisterNodeDesc(SyntheticFacePlate)


def _headViews(yaws, size=(1280, 960)):
    """ Cameras looking at the canonical head (60 cm away, turned by each yaw), camera to face space. """
    cameras = []
    for yaw in yaws:
        matrix = scene3d.composeTRS((0, 0, 0), (0, yaw, 0)) @ scene3d.composeTRS((0, 1.0, 60.0), (0, 0, 0))
        cameras.append(face.viewCamera(50.0, *size).copy(matrix=matrix))
    return cameras


def test_headPinsWarpTheEarsAndSkull():
    """ Pins on the ears and the skull, placed where an actor with wider ears and a higher skull has them. """
    head = face.canonicalHead()
    vertices = head.vertices
    index = np.arange(len(vertices))
    ears = (np.abs(vertices[:, 0]) > 6.0) & (index >= face.MESH_VERTICES) & (vertices[:, 1] > -3) & (vertices[:, 1] < 5)
    top = (vertices[:, 1] > 9.0) & (index >= face.MESH_VERTICES)
    truth = vertices.copy()
    truth[ears, 0] += np.sign(vertices[ears, 0]) * 1.0
    truth[top, 1] += 0.8 * (vertices[top, 1] - 9.0) / (vertices[:, 1].max() - 9.0)
    size = (1280, 960)
    cameras = _headViews((0, 70, -70), size)
    picks = list(np.where(ears)[0][::max(1, ears.sum() // 3)]) + list(np.where(top)[0][::max(1, top.sum() // 2)])
    locations, targets, pinCameras = [], [], []
    for camera in cameras:
        xy, _, _ = camera.project(truth[picks], *size)
        for k, vertex in enumerate(picks):
            locations.append(face.surfaceLocation(vertices, head.triangles(), vertices[vertex]))
            targets.append(xy[k])
            pinCameras.append(camera)
    warped = face.warpToPins(head, locations, np.array(targets), pinCameras, [size] * len(locations))
    before, after = np.linalg.norm(vertices - truth, axis=1), np.linalg.norm(warped - truth, axis=1)
    # The pinned points reach their place (seen from 3 views: in 3D), the ears and the skull follow, the face stays
    # (but near its oval, where the head joins it)
    assert after[picks].mean() < 0.01 * before[picks].mean()
    assert after[ears].mean() < 0.3 * before[ears].mean() and after[top].mean() < 0.6 * before[top].mean()
    oval = np.unique(np.asarray(face.headData()["holeEdges"]).ravel())
    faceVertices = vertices[:face.MESH_VERTICES]
    toOval = np.sqrt(((faceVertices[:, None] - vertices[oval][None]) ** 2).sum(-1)).min(axis=1)
    assert np.abs(warped[:face.MESH_VERTICES][toOval > 2.0] - faceVertices[toOval > 2.0]).max() < 1e-9
    # A pin dragged far in one view: reached, its surroundings moved along (no spike), the rest of the head not
    camera = cameras[1]
    xy, _, _ = camera.project(vertices[[picks[0]]], *size)
    target = xy + [[-80.0, -60.0]]
    moved = face.warpToPins(head, [face.surfaceLocation(vertices, head.triangles(), vertices[picks[0]])], target,
                            [camera], [size], radius=4.0)
    reached, _, _ = camera.project(moved[[picks[0]]], *size)
    assert np.hypot(*(reached[0] - target[0])) < 0.5
    shift = np.linalg.norm(moved - vertices, axis=1)
    distance = np.linalg.norm(vertices - vertices[picks[0]], axis=1)
    assert shift[distance < 1.0].min() > 0.8 * shift.max() and shift[distance > 15.0].max() < 0.01
    # No pin: the head as it is
    np.testing.assert_array_equal(face.warpToPins(head, [], np.zeros((0, 2)), [], []), vertices)


def test_faceBuilderPins():
    """ The pins of the views of a FaceBuilder: added on the head, dragged, the head follows. """
    size = (1280, 960)
    cameras = _headViews((0, 70), size)
    graph = Graph("builderPins")
    photos = [graph.addNewNode("Constant", width=size[0], height=size[1]) for _ in range(2)]
    builder = graph.addNewNode("FaceBuilder")
    for photo, name in zip(photos, ("input", "input1")):
        graph.addEdge(photo.output, builder.attribute(name))
    views = [{"input": name, "frame": 1, "still": True, "size": list(size), "landmarks": [], "focal": 50.0,
              "cameraToFace": [float(v) for v in camera.matrix.ravel()]} for name, camera in zip(("input", "input1"), cameras)]
    builder.views.value = json.dumps(views)
    builder.showView.value = "input1"
    graph.update()
    plan = buildPlan(builder)
    evaluator = Evaluator()
    from meshroom.live.evaluator import RenderContext
    ctx = RenderContext(plan.nodes[plan.target], 1, evaluator, plan)
    original = face.canonicalHead().vertices
    # The tip of the left ear (seen from the side view): picked where the head is seen
    ear = int(np.argmax(original[:, 0] * (np.arange(len(original)) >= face.MESH_VERTICES)))
    xy, _, _ = cameras[1].project(original[[ear]], *size)
    point = builder.nodeDesc.pickPoint(ctx, *xy[0])
    assert point is not None and np.linalg.norm(point - original[ear]) < 1.0
    # Dragged 25 pixels to the left in that view
    target = xy[0] + [-25.0, 0.0]
    builder.pins1.value = json.dumps([{"name": "pin1", "enabled": True, "point": [float(v) for v in point],
                                       "keys": {"1": [float(target[0]), float(target[1])]}, "format": list(size)}])
    graph.update()
    plan = buildPlan(builder)
    ctx = RenderContext(plan.nodes[plan.target], 1, Evaluator(), plan)
    head = builder.nodeDesc.head(ctx)
    projections = builder.nodeDesc.pinProjections(ctx, lambda output: None)
    assert np.hypot(*(np.array(projections["pin1"]) - target)) < 1.0
    moved, _, _ = cameras[1].project(head.vertices[[ear]], *size)
    assert np.hypot(*(moved[0] - target)) < 3.0
    # The face did not move
    assert np.abs(head.vertices[:face.MESH_VERTICES] - original[:face.MESH_VERTICES]).max() < 0.1


# Fit of the ears, neck and skull on segmented views ("Fit Ears, Neck And Skull")

def _actorHead():
    """ The canonical head with wider ears and neck and a higher skull (smooth bumps; the face unchanged). """
    from meshroom.live import headFit
    head = face.canonicalHead()
    vertices = head.vertices
    regions = headFit.headRegions(head)
    mobility = face.headMobility(head)

    def bump(center, radius, move):
        weight = np.exp(-((vertices - center) ** 2).sum(1) / radius ** 2) * mobility
        return weight[:, None] * np.asarray(move, dtype=np.float64)[None]
    actor = vertices.copy()
    for side in (-1, 1):
        sideMask = np.sign(vertices[:, 0]) == side
        actor += bump(vertices[regions["ear"] & sideMask].mean(0), 3.0, (side * 1.0, 0, 0))
        actor += bump(vertices[regions["neck"] & sideMask].mean(0), 4.0, (side * 0.8, 0, 0))
    actor += bump(vertices[np.argmax(vertices[:, 1])], 6.0, (0, 0.8, 0))
    return actor


_LABEL_COLORS = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)}  # background, hair, body skin, face skin


def _labelImage(vertices, camera, size):
    """ The segmentation of a head seen by a camera, as colors (_LABEL_COLORS), rows from the top. """
    from meshroom.live import geoRender, headFit
    head = face.canonicalHead()
    triangles = head.triangles()
    regions = headFit.headRegions(head)
    classes = np.full(len(triangles), headFit.HAIR)
    classes[(triangles < face.MESH_VERTICES).all(1)] = headFit.FACE_SKIN
    classes[(regions["ear"][triangles] | regions["neck"][triangles]).any(1)] = headFit.BODY_SKIN
    width, height = size
    xy, depth, _ = camera.project(vertices, width, height)
    _, ids = geoRender.rasterize(xy[:, 0], height - xy[:, 1], depth, triangles, (height, width))
    rgb = np.zeros((height, width, 3), np.float32)
    covered = ids >= 0
    rgb[covered] = np.array([_LABEL_COLORS[c] for c in range(4)], np.float32)[classes[ids[covered]]]
    return rgb


class _ColorSegmenter:
    """ A segmenter reading the classes back from _LABEL_COLORS. """

    def segment(self, rgb):
        from meshroom.live import headFit
        labels = np.zeros(rgb.shape[:2] + (6,), np.float32)
        r, g, b = (rgb[:, :, i] > 127 for i in range(3))
        labels[:, :, headFit.HAIR] = r & ~g & ~b
        labels[:, :, headFit.BODY_SKIN] = g & ~r & ~b
        labels[:, :, headFit.FACE_SKIN] = b & ~r & ~g
        labels[:, :, headFit.BACKGROUND] = ~(r | g | b)
        return labels

    def close(self):
        pass


def test_headFitOnSegmentedViews():
    """ The ears, neck and skull fitted on the segmentation of views of an actor (headFit.fitHead). """
    from meshroom.live import headFit
    head = face.canonicalHead()
    actor = _actorHead()
    size = (640, 480)
    cameras = _headViews((0, 60, -60, 120, -120), size)
    segmenter = _ColorSegmenter()
    views = []
    for camera in cameras:
        rgb = _labelImage(actor, camera, size)
        views.append(headFit.FitView(camera, size, segmenter.segment((rgb * 255).astype(np.uint8)), (0, 0)))
    fit = headFit.fitHead(head, views, radius=3.0, useDepth=False)
    fitted = headFit.applyFit(head, fit)
    before = np.linalg.norm(head.vertices - actor, axis=1)[face.MESH_VERTICES:].mean()
    after = np.linalg.norm(fitted - actor, axis=1)[face.MESH_VERTICES:].mean()
    assert after < 0.65 * before
    # The face itself does not move (its landmarks placed it), but near its oval
    still = face.headMobility(head)[:face.MESH_VERTICES] == 0
    assert np.abs(fitted[:face.MESH_VERTICES][still] - head.vertices[:face.MESH_VERTICES][still]).max() < 1e-9
    # With the depth of the views (an inverse depth up to an affine map, as a monocular network gives it)
    from meshroom.live import geoRender
    depthViews = []
    for view, camera in zip(views, cameras):
        xy, depth, _ = camera.project(actor, *size)
        zbuffer, _ = geoRender.rasterize(xy[:, 0], size[1] - xy[:, 1], depth, head.triangles(), (size[1], size[0]))
        disparity = np.where(np.isfinite(zbuffer), 3.0 / np.where(np.isfinite(zbuffer), zbuffer, 1.0) - 0.01, 0.0)
        depthViews.append(headFit.FitView(camera, size, view.labels, (0, 0), disparity.astype(np.float32)))
    withDepth = headFit.applyFit(head, headFit.fitHead(head, depthViews, radius=3.0, useDepth=True))
    assert np.linalg.norm(withDepth - actor, axis=1)[face.MESH_VERTICES:].mean() < 0.55 * before


class LabelPlate(_desc.ImageOpNode):
    """ The segmentation colors of the actor's head (_actorHead) seen by camera `view` of _headViews((0, 60, -60)). """
    category = "Test"
    inputs = [_desc.IntParam(name="view", label="View", value=0, range=(0, 2, 1), description="")]
    outputs = [_desc.Image(name="output", label="Output", description="")]

    def getInfo(self, ctx, inputsInfo):
        return _ImageInfo(640, 480)

    def render(self, ctx, inputs):
        from meshroom.live import ImageBuffer
        camera = _headViews((0, 60, -60), (640, 480))[ctx.value("view")]
        rgb = _labelImage(_actorHead(), camera, (640, 480))
        # Linear values: the display encoding (2.2) gives the colors back
        return ImageBuffer.fromRGBA(np.dstack([rgb, np.ones(rgb.shape[:2], np.float32)]), (640, 480))


def test_faceBuilderFitsEarsNeckAndSkull(monkeypatch):
    from tests.utils import registerNodeDesc, unregisterNodeDesc
    from meshroom.live import headFit
    from meshroom.live.evaluator import RenderContext
    registerNodeDesc(LabelPlate)
    try:
        monkeypatch.setattr(headFit, "headSegmenter", lambda: _ColorSegmenter())
        size = (640, 480)
        cameras = _headViews((0, 60, -60), size)
        graph = Graph("builderFit")
        plates = [graph.addNewNode("LabelPlate", view=index) for index in range(3)]
        # (the free fit: this actor's bumps are not in the identity model)
        builder = graph.addNewNode("FaceBuilder", fitDepth=False, predictHead=False)
        names = ("input", "input1", "input2")
        for plate, name in zip(plates, names):
            graph.addEdge(plate.output, builder.attribute(name))
        builder.views.value = json.dumps([{"input": name, "frame": 1, "still": True, "size": list(size),
                                           "landmarks": [], "focal": 50.0,
                                           "cameraToFace": [float(v) for v in camera.matrix.ravel()]}
                                          for name, camera in zip(names, cameras)])
        graph.update()
        values = {attr.name: _snapshotValue(attr) for attr in builder.attributes
                  if not attr.isOutput and attr.type not in ("Image", "PushButtonParam")}
        context = ActionContext(values, Evaluator(), buildPlan(plates[0]), 1, nodePlan=buildPlan(builder),
                                inputPlans={"input1": buildPlan(plates[1]), "input2": buildPlan(plates[2])})
        result = builder.nodeDesc.runAction("fitHeadContours", context)
        builder.headFit.value = result["headFit"]
        graph.update()
        plan = buildPlan(builder)
        head = builder.nodeDesc.head(RenderContext(plan.nodes[plan.target], 1, Evaluator(), plan))
        actor, original = _actorHead(), face.canonicalHead().vertices
        n = face.MESH_VERTICES
        assert np.linalg.norm(head.vertices - actor, axis=1)[n:].mean() < \
            0.8 * np.linalg.norm(original - actor, axis=1)[n:].mean()
        assert builder.nodeDesc.runAction("clearHead", context)["headFit"] == ""
    finally:
        unregisterNodeDesc(LabelPlate)


# The identity model: the proportions of the whole head from its face

def _identityActor(seed=3, spread=1.2):
    from meshroom.live import headFit
    modes = headFit.identityModel()
    coefficients = np.random.default_rng(seed).normal(0, spread, len(modes))
    return face.canonicalHead().vertices + np.tensordot(coefficients, modes, axes=1)


def test_identityPredictsTheHeadFromTheFace():
    """ A head of the model: its face alone predicts its skull, ears and neck better than following the face. """
    from meshroom.live import headFit
    rest = face.canonicalHead().vertices
    n = face.MESH_VERTICES
    for seed in (1, 2, 3):
        actor = _identityActor(seed)
        faceTarget = actor[:n]
        followed = face.HeadFollow.of(rest).follow(faceTarget)
        predicted = headFit.identityHead(rest, faceTarget, headFit.predictIdentity(faceTarget, rest))
        np.testing.assert_allclose(predicted[:n], faceTarget, atol=1e-9)
        errorFollowed = np.linalg.norm(followed - actor, axis=1)[n:].mean()
        errorPredicted = np.linalg.norm(predicted - actor, axis=1)[n:].mean()
        print(f"seed {seed}: canonical {np.linalg.norm(rest - actor, axis=1)[n:].mean():.3f}, followed "
              f"{errorFollowed:.3f}, predicted {errorPredicted:.3f}")
        assert errorPredicted < 0.8 * errorFollowed


def test_identityFittedOnTheOutlines():
    """ The identity refined on the segmented views of the actor: closer than predicted from the face alone. """
    from meshroom.live import headFit
    head = face.canonicalHead()
    rest = head.vertices
    n = face.MESH_VERTICES
    size = (640, 480)
    cameras = _headViews((0, 60, -60, 120, -120), size)
    segmenter = _ColorSegmenter()
    errors = []
    for seed in (1, 3):
        actor = _identityActor(seed)
        views = [headFit.FitView(camera, size, segmenter.segment((_labelImage(actor, camera, size) * 255).astype(np.uint8)),
                                 (0, 0)) for camera in cameras]
        predicted = headFit.identityHead(rest, actor[:n], headFit.predictIdentity(actor[:n], rest))
        fit = headFit.fitIdentity(rest, actor[:n], head.topology, views, useDepth=False)
        fitted = headFit.identityHead(rest, actor[:n], fit["identity"])
        # The ears and skull (the neck mostly follows the face: its bend differs from view to view)
        parts = headFit.headRegions(head)
        parts = (parts["ear"] | parts["skull"]) & ~parts["neck"]
        parts[:n] = False
        errorPredicted = np.linalg.norm(predicted - actor, axis=1)[parts].mean()
        errorFitted = np.linalg.norm(fitted - actor, axis=1)[parts].mean()
        print(f"seed {seed}: predicted {errorPredicted:.3f}, fitted {errorFitted:.3f} ({fit['constraints']} constraints)")
        assert fit["constraints"] > 50 and errorFitted < 1.05 * errorPredicted
        errors.append((errorPredicted, errorFitted))
    assert sum(e[1] for e in errors) < 0.9 * sum(e[0] for e in errors)


def test_neckBentOnTheVerticalOfTheView():
    """ A bowed head: the base of its neck goes down along the vertical of the image, the head itself unchanged. """
    from meshroom.live import headFit
    rest = face.canonicalHead().vertices
    angle = np.radians(40.0)
    # The camera up in the space of the head: the head bowed by 40 degrees (the camera vertical tilted backward)
    up = np.array([0.0, np.cos(angle), -np.sin(angle)])
    matrix = np.eye(4)
    matrix[:3, 1] = up
    matrix[:3, 2] = np.cross(np.array([1.0, 0.0, 0.0]), up)
    bent = headFit.bendNeck(rest, matrix)
    neck = headFit.headRegions(face.canonicalHead())["neck"]
    np.testing.assert_array_equal(bent[~neck], rest[~neck])
    weights, ring = headFit.neckBendWeights(len(rest))
    base = weights > 0.99
    pivot = rest[ring].mean(axis=0)
    direction = bent[base].mean(axis=0) - pivot
    direction /= np.linalg.norm(direction)
    restDirection = rest[base].mean(axis=0) - pivot
    restDirection /= np.linalg.norm(restDirection)
    # Turned by the angle between the vertical of the head and that of the view, toward it
    turned = np.degrees(np.arccos(np.clip(direction @ restDirection, -1, 1)))
    assert 35.0 < turned < 45.0 and direction @ -up > restDirection @ -up
    np.testing.assert_array_equal(headFit.bendNeck(rest, matrix, amount=0.0), rest)
