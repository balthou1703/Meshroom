"""FaceTracker: a face mesh with the MediaPipe topology fitted on landmarks, non rigid (meshroom/live/face.py)."""

import json
import os

import numpy as np
import pytest

import meshroom
from meshroom.core import pluginManager
from meshroom.core.graph import Graph
from meshroom.live import Evaluator, buildPlan, face, scene3d
from meshroom.live.actions import ActionContext
from meshroom.live.evaluator import _snapshotValue

_COMPOSITING_FOLDER = os.path.join(os.path.dirname(meshroom.__file__), "nodes", "compositing")
if not pluginManager.isNodeDescRegistered("Grade"):
    pluginManager.addPluginFromBuiltInFolder("compositing", _COMPOSITING_FOLDER)

W, H = 640, 480


def syntheticFace():
    """ A face-like surface with 468 vertices (18 x 26 grid, bulged toward +Z), in centimeters. """
    columns, rows = 18, 26
    u, v = np.meshgrid(np.linspace(-1, 1, columns), np.linspace(-1, 1, rows))
    vertices = np.c_[7.0 * u.ravel(), 10.0 * v.ravel(), 6.0 * np.sqrt(np.clip(1.2 - 0.5 * u.ravel() ** 2 - 0.4 * v.ravel() ** 2, 0, None))]
    faces = []
    for r in range(rows - 1):
        for c in range(columns - 1):
            a = r * columns + c
            faces.append([a, a + 1, a + columns + 1, a + columns])
    return scene3d.GeoData(vertices, [4] * len(faces), np.array(faces).ravel())


FACE = syntheticFace()
CAMERA = scene3d.CameraData(np.eye(4), 35.0, 36.0, 24.0)
# The rows of the "mouth" (deformed) and the rigid landmarks used by the tests (the forehead and the sides)
MOUTH = [i for i in range(468) if 6 <= i // 18 <= 8 and 5 <= i % 18 <= 12]
RIGID = [i for i in range(468) if i // 18 >= 16 or i % 18 in (0, 17)]


def headAt(frame):
    """ The true head pose: 60 cm in front of the camera, turning. """
    return scene3d.composeTRS((2.0 * frame, -1.0, -60.0), (0.0, 4.0 * frame, 2.0 * frame))


def deformedAt(frame):
    """ The face in its space at a frame: the mouth opening (moved down). """
    vertices = FACE.vertices.copy()
    vertices[MOUTH, 1] -= 0.4 * frame
    return vertices


def landmarksAt(frame):
    """ The normalized landmarks MediaPipe would give (x, y from the top left corner, z unused). """
    world = scene3d.transformPoints(headAt(frame), deformedAt(frame))
    xy, _, _ = CAMERA.project(world, W, H)
    return np.c_[xy[:, 0] / W, (H - xy[:, 1]) / H, np.zeros(len(xy))]


def test_rigidAndNonRigidFit():
    pixels = face.toPixels(landmarksAt(4), W, H)
    geo, head, _, error = face.solveFace(FACE, pixels, CAMERA, W, H, rigid=RIGID)
    assert error < 1e-3
    np.testing.assert_allclose(head, headAt(4), atol=1e-4)
    # Non rigid: every vertex is seen on its landmark, the mouth where it moved
    xy, _, _ = CAMERA.project(geo.vertices, W, H)
    np.testing.assert_allclose(xy, pixels, atol=1e-6)
    # (the vertices keep their neutral depth: a few hundredths of the 1.6 cm the mouth moved)
    np.testing.assert_allclose(geo.vertices[MOUTH], scene3d.transformPoints(headAt(4), deformedAt(4))[MOUTH], atol=0.15)
    assert geo.topology is FACE.topology
    rigid, _, _, _ = face.solveFace(FACE, pixels, CAMERA, W, H, rigid=RIGID, deform=False)
    np.testing.assert_allclose(rigid.vertices, scene3d.transformPoints(headAt(4), FACE.vertices), atol=1e-4)


def test_storageAndSmoothing():
    faces = {frame: {"landmarks": landmarksAt(frame), "blendshapes": {"jawOpen": 0.1 * frame}} for frame in range(1, 6)}
    parsed, imageFormat = face.parseFaces(face.dumpFaces(faces, (W, H)))
    assert imageFormat == (W, H) and sorted(parsed) == [1, 2, 3, 4, 5]
    np.testing.assert_allclose(parsed[3]["landmarks"], faces[3]["landmarks"], atol=1e-5)
    assert parsed[2]["blendshapes"] == {"jawOpen": 0.2}
    # Smoothed: a jitter at one frame is softened; outside the tracked frames, the nearest one
    noisy = dict(parsed)
    noisy[3] = {"landmarks": parsed[3]["landmarks"] + 0.01, "blendshapes": {}}
    assert np.abs(face.landmarksAt(noisy, 3, 1.0) - parsed[3]["landmarks"]).max() < 0.006
    np.testing.assert_allclose(face.landmarksAt(parsed, 9), parsed[5]["landmarks"])


def test_landmarkTracks():
    faces = {frame: {"landmarks": landmarksAt(frame)} for frame in (1, 2)}
    tracks = face.landmarkTracks(faces, W, H, indices=[0, 467])
    assert [len(track["keys"]) for track in tracks] == [2, 2] and tracks[1]["format"] == [W, H]
    np.testing.assert_allclose(tracks[0]["keys"]["2"], face.toPixels(landmarksAt(2), W, H)[0], atol=1e-3)


def test_topologyIsChecked():
    with pytest.raises(face.FaceUnavailable, match="468"):
        face.checkTopology(scene3d.GeoData(np.zeros((100, 3))))


def faceGraph(tmp_path):
    graph = Graph("face")
    plate = graph.addNewNode("Constant", width=W, height=H)
    path = tmp_path / "face.obj"
    with open(path, "w") as file:
        file.writelines(f"v {x} {y} {z}\n" for x, y, z in FACE.vertices)
        starts = np.r_[0, np.cumsum(FACE.faceCounts)[:-1]]
        for start, count in zip(starts, FACE.faceCounts):
            file.write("f " + " ".join(str(i + 1) for i in FACE.faceIndices[start:start + count]) + "\n")
    readGeo = graph.addNewNode("ReadGeo")
    readGeo.file.value = str(path)
    tracker = graph.addNewNode("FaceTracker")
    graph.addEdge(plate.output, tracker.input)
    graph.addEdge(readGeo.output, tracker.face)
    graph.update()
    return graph, plate, tracker


class FakeLandmarker:
    """ The landmarks of the synthetic face, frame after frame (the frame from the time stamp, 24 fps from frame 1). """

    def detect(self, rgb, timestampMs):
        assert rgb.dtype == np.uint8 and rgb.shape == (H, W, 3)
        frame = 1 + int(round(timestampMs * 24 / 1000.0))
        return landmarksAt(frame), {"jawOpen": 0.5}

    def close(self):
        pass


def test_faceTrackerNode(tmp_path, monkeypatch):
    monkeypatch.setattr(face, "RIGID_LANDMARKS", tuple(RIGID))
    monkeypatch.setattr(face, "landmarker", lambda minConfidence=0.5: FakeLandmarker())
    # The fake landmarker sees the whole frame (the crops are tested by test_smallDarkFaceFoundInALargePlate)
    monkeypatch.setattr(face, "_cropAround", lambda landmarks, width, height: (0, 0, width, height))
    graph, plate, tracker = faceGraph(tmp_path)
    tracker.exportTracks.value = "all"
    graph.update()
    values = {attr.name: _snapshotValue(attr) for attr in tracker.attributes
              if not attr.isOutput and attr.type not in ("Image", "PushButtonParam")}
    context = ActionContext(values, Evaluator(), buildPlan(plate), 1, nodePlan=buildPlan(tracker))
    result = tracker.nodeDesc.runAction("trackFace", context)
    assert len(json.loads(result["tracks"])) == 468
    tracker.faceTrack.value = result["faceTrack"]
    tracker.smoothing.value = 0.0
    graph.update()
    plan = buildPlan(tracker)
    evaluator = Evaluator()
    # The constant plate is a still image: frame 1 only
    geo = evaluator.data(plan, plan.target, "geoOutput", 1)
    np.testing.assert_allclose(evaluator.data(plan, plan.target, "headMatrix", 1), headAt(1), atol=1e-3)
    xy, _, _ = CAMERA.project(geo.vertices, W, H)
    # The landmarks are stored with 5 decimals (normalized)
    np.testing.assert_allclose(xy, face.toPixels(landmarksAt(1), W, H), atol=0.01)
    image = evaluator.render(plan, 1)
    assert image.rgba()[:, :, 2].max() > 0.5


class SquareFinder:
    """ A fake landmarker finding a bright square (the "face") when it is large enough in the image it is given. """

    def __init__(self):
        self.sizes = []

    def detect(self, rgb, timestampMs):
        self.sizes.append(rgb.shape[:2])
        rows, columns = np.nonzero(rgb[:, :, 0] > 160)
        if not len(rows) or columns.max() - columns.min() + 1 < 0.15 * rgb.shape[1]:
            return None, {}
        h, w = rgb.shape[:2]
        # 478 landmarks on the square (normalized to the image given), z relative to its width
        t = np.linspace(0.0, 1.0, face.LANDMARKS)
        x = (columns.min() + t * (columns.max() + 1 - columns.min())) / w
        y = (rows.min() + t * (rows.max() + 1 - rows.min())) / h
        return np.c_[x, y, np.full(len(t), 10.0 / w)], {}


def test_smallDarkFaceFoundInALargePlate():
    from meshroom.live import ImageBuffer
    width, height = 3000, 1000
    rgba = np.full((height, width, 4), 0.002, np.float32)
    rgba[:, :, 3] = 1.0
    # A dark 80 pixel "face", rows 300 to 379 from the top, columns 2400 to 2479
    rgba[300:380, 2400:2480, :3] = 0.03
    plate = ImageBuffer.fromRGBA(rgba, (width, height))
    finder = SquareFinder()
    faces = face.trackFaces(lambda frame: plate, [1, 2], finder)
    assert sorted(faces) == [1, 2]
    for frame in (1, 2):
        landmarks = faces[frame]["landmarks"]
        np.testing.assert_allclose([landmarks[:, 0].min() * width, landmarks[:, 0].max() * width], [2400, 2480], atol=1e-6)
        np.testing.assert_allclose([landmarks[:, 1].min() * height, landmarks[:, 1].max() * height], [300, 380], atol=1e-6)
        # z rescaled to the width of the whole frame
        np.testing.assert_allclose(landmarks[:, 2], 10.0 / width)
    # Frame 2: found at once in the crop around the face of frame 1
    assert finder.sizes[-1][0] < 300 and finder.sizes[-1][0] == finder.sizes[-1][1]


def test_mediapipeImportedWithoutMatplotlib(monkeypatch, tmp_path):
    """ A package importing matplotlib in its __init__ as mediapipe does (and deleting its submodule names). """
    import sys
    package = tmp_path / "mediapipe"
    (package / "python").mkdir(parents=True)
    (package / "tasks").mkdir()
    (package / "python" / "__init__.py").write_text("import matplotlib.pyplot as plt\nimport cv2\n")
    (package / "tasks" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("from mediapipe.python import *\nimport mediapipe.tasks as tasks\ndel python\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("matplotlib", "matplotlib.pyplot", "cv2"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import importlib.util
    realFindSpec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *args: None if name in ("matplotlib", "cv2") else realFindSpec(name, *args))
    try:
        assert face.importMediaPipe().tasks is not None
        # The stand-ins are gone: matplotlib is still missing for everything else
        assert "matplotlib" not in sys.modules and "cv2" not in sys.modules
    finally:
        for name in [name for name in sys.modules if name.split(".")[0] == "mediapipe"]:
            del sys.modules[name]


def test_maskLimitsTheSearchAndHidesLandmarks():
    from meshroom.live import ImageBuffer
    width, height = 1200, 400
    rgba = np.full((height, width, 4), 0.01, np.float32)
    rgba[:, :, 3] = 1.0
    # Two "faces": the left one masked out, the right one half masked
    rgba[100:200, 100:200, :3] = 0.5
    rgba[100:200, 900:1000, :3] = 0.5
    plate = ImageBuffer.fromRGBA(rgba, (width, height))
    mask = np.zeros((height, width), np.float32)
    mask[:, 600:960] = 1.0
    finder = SquareFinder()
    faces = face.trackFaces(lambda frame: plate, [1], finder, maskAt=lambda frame: mask)
    landmarks = faces[1]["landmarks"]
    # The right face (cut by the mask at 960: the finder sees its unmasked part), the left one never seen
    assert landmarks[:, 0].min() * width >= 900 - 1e-6
    hidden = faces[1]["hidden"]
    assert hidden == [i for i in range(face.LANDMARKS) if landmarks[i, 0] * width >= 960]
    # Stored, and no track key where hidden
    stored, _ = face.parseFaces(face.dumpFaces(faces, (width, height)))
    assert stored[1]["hidden"] == hidden
    tracks = face.landmarkTracks(stored, width, height)
    assert all(not track["keys"] for i, track in enumerate(tracks) if i in hidden)


def test_hiddenLandmarksDoNotMoveTheHead():
    landmarks = landmarksAt(3)
    pixels = face.toPixels(landmarks, W, H)
    wrong = pixels.copy()
    # Landmarks pulled away (e.g. a hand in front of the face), masked out
    hidden = RIGID[: len(RIGID) // 3]
    wrong[hidden] += 40.0
    geo, head, _, _ = face.solveFace(FACE, wrong, CAMERA, W, H, rigid=RIGID, deform=True, hidden=hidden)
    np.testing.assert_allclose(head, headAt(3), atol=1e-3)
    # Their vertices follow the head (not pulled)
    xy, _, _ = CAMERA.project(geo.vertices[hidden], W, H)
    np.testing.assert_allclose(xy, pixels[hidden], atol=0.05)


def test_pinsPullTheFaceNonRigid():
    pixels = face.toPixels(landmarksAt(2), W, H)
    # A pin on a mouth vertex, dragged 15 pixels to the right: the mouth follows, the far side of the face does not
    pinned = MOUTH[len(MOUTH) // 2]
    target = pixels[pinned] + [15.0, 0.0]
    geo, head, _, _ = face.solveFace(FACE, pixels, CAMERA, W, H, rigid=RIGID, deform=True,
                                     pins=(FACE.vertices[[pinned]], target[None]), pinRadius=2.5)
    xy, _, _ = CAMERA.project(geo.vertices, W, H)
    np.testing.assert_allclose(xy[pinned], target, atol=0.3)
    far = np.linalg.norm(FACE.vertices - FACE.vertices[pinned], axis=1) > 8.0
    np.testing.assert_allclose(xy[far], pixels[far], atol=0.3)
    # A pin on the surface between vertices goes through its position too
    point = FACE.vertices[FACE.triangles()[100]].mean(axis=0)
    geo, _, _, _ = face.solveFace(FACE, pixels, CAMERA, W, H, rigid=RIGID, deform=True,
                                  pins=(point[None], np.array([[300.0, 200.0]])), pinRadius=3.0)
    (triangle, weights), = face.pinLocations(FACE, point[None])
    onFace = geo.vertices[FACE.triangles()[triangle]].T @ weights
    xy, _, _ = CAMERA.project(onFace[None], W, H)
    np.testing.assert_allclose(xy[0], [300.0, 200.0], atol=1.0)


def test_trackRangeLimitsTheFrames():
    from meshroom.live.params import trackRange
    assert trackRange({}, range(1, 6)) == [1, 2, 3, 4, 5]
    assert trackRange({"limitTrackRange": True, "trackFirst": 2, "trackLast": 4}, range(1, 6)) == [2, 3, 4]
    assert trackRange({"limitTrackRange": True, "trackFirst": 4, "trackLast": 2}, range(5, 0, -1)) == [4, 3, 2]
    assert trackRange({"limitTrackRange": False, "trackFirst": 2, "trackLast": 4}, [1, 5]) == [1, 5]


def test_faceTrackerPins(tmp_path, monkeypatch):
    from meshroom.live.evaluator import RenderContext
    monkeypatch.setattr(face, "RIGID_LANDMARKS", tuple(RIGID))
    monkeypatch.setattr(face, "landmarker", lambda minConfidence=0.5: FakeLandmarker())
    monkeypatch.setattr(face, "_cropAround", lambda landmarks, width, height: (0, 0, width, height))
    graph, plate, tracker = faceGraph(tmp_path)
    values = {attr.name: _snapshotValue(attr) for attr in tracker.attributes
              if not attr.isOutput and attr.type not in ("Image", "PushButtonParam")}
    context = ActionContext(values, Evaluator(), buildPlan(plate), 1, nodePlan=buildPlan(tracker))
    tracker.faceTrack.value = tracker.nodeDesc.runAction("trackFace", context)["faceTrack"]
    tracker.smoothing.value = 0.0
    graph.update()
    plan = buildPlan(tracker)
    evaluator = Evaluator()
    # A pin picked on the face where the mouth vertex is seen: its point on the neutral face is that vertex
    pinned = MOUTH[len(MOUTH) // 2]
    geo = evaluator.data(plan, plan.target, "geoOutput", 1)
    (x, y), = CAMERA.project(geo.vertices[[pinned]], W, H)[0]
    ctx = RenderContext(plan.nodes[plan.target], 1, evaluator, plan)
    point = tracker.nodeDesc.pickPoint(ctx, x, y)
    np.testing.assert_allclose(point, FACE.vertices[pinned], atol=0.05)
    # Dragged 12 pixels up: the face goes through it
    tracker.pins.value = json.dumps([{"name": "pin1", "enabled": True, "point": list(point), "format": [W, H],
                                      "keys": {"1": [x, y + 12.0]}, "keyframes": [1]}])
    graph.update()
    plan = buildPlan(tracker)
    geo = Evaluator().data(plan, plan.target, "geoOutput", 1)
    xy, _, _ = CAMERA.project(geo.vertices[[pinned]], W, H)
    np.testing.assert_allclose(xy[0], [x, y + 12.0], atol=0.5)
