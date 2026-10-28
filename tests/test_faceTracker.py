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


@pytest.fixture(autouse=True)
def _uprightSyntheticFace(monkeypatch):
    """ The synthetic face has no eyes: its landmarks 33 and 263 tell no roll (the crops are not turned). """
    monkeypatch.setattr(face, "faceRoll", lambda landmarks, width, height: 0.0)


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
    # Each vertex exactly on its landmark
    tracker.deformModel.value = "surface"
    tracker.stiffness.value = 0.0
    graph.update()
    plan = buildPlan(tracker)
    evaluator = Evaluator()
    # The constant plate is a still image: frame 1 only
    geo = evaluator.data(plan, plan.target, "geoOutput", 1)
    np.testing.assert_allclose(evaluator.data(plan, plan.target, "headMatrix", 1), headAt(1), atol=1e-3)
    # The head (default "Mesh"): the face, then the ears, skull and neck
    assert face.isHead(geo)
    xy, _, _ = CAMERA.project(geo.vertices[:face.MESH_VERTICES], W, H)
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
    (triangle, weights), = face.pinLocations(FACE, point[None])
    free, _, _, _ = face.solveFace(FACE, pixels, CAMERA, W, H, rigid=RIGID, deform=True)
    start, _, _ = CAMERA.project((free.vertices[FACE.triangles()[triangle]].T @ weights)[None], W, H)
    target = start[0] + [12.0, -8.0]
    geo, _, _, _ = face.solveFace(FACE, pixels, CAMERA, W, H, rigid=RIGID, deform=True,
                                  pins=(point[None], target[None]), pinRadius=3.0)
    onFace = geo.vertices[FACE.triangles()[triangle]].T @ weights
    xy, _, _ = CAMERA.project(onFace[None], W, H)
    np.testing.assert_allclose(xy[0], target, atol=1.0)
    # A pin too far for the face (a smooth warp would fold it): the face goes toward it without folding
    from meshroom.live import faceRig
    geo, head, _, _ = face.solveFace(FACE, pixels, CAMERA, W, H, rigid=RIGID, deform=True,
                                     pins=(point[None], np.array([[start[0][0] - 80.0, start[0][1] + 20.0]])),
                                     pinRadius=3.0)
    local = scene3d.transformPoints(np.linalg.inv(head), geo.vertices)
    reference = faceRig.triangleNormals(FACE.vertices, FACE.triangles())
    assert (faceRig.triangleNormals(local, FACE.triangles()) * reference).sum(1).min() > 0.0


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
    # The pins pull the vertices around them (the rig model of a MediaPipe mesh: see test_rigPins)
    tracker.deformModel.value = "surface"
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


def _movingPlate(frame, width=320, height=240):
    """ A textured plate moving 2 pixels right and 1 down per frame. """
    from meshroom.live import ImageBuffer
    rows, columns = np.mgrid[0:height, 0:width]
    x, y = columns + 0.5 - 2.0 * frame, rows + 0.5 - 1.0 * frame
    lum = (0.5 + 0.12 * (np.sin(0.21 * x + 0.13 * y) + np.sin(0.17 * y - 0.09 * x) + np.sin(0.31 * x) * np.sin(0.27 * y)))
    lum = lum.astype(np.float32)
    return ImageBuffer.fromRGBA(np.stack([lum, lum, lum, np.ones_like(lum)], -1), (width, height))


def test_faceFollowsTheFlow():
    width, height = 320, 240
    t = np.linspace(0.3, 0.7, face.LANDMARKS)
    start = np.c_[t, 1.0 - t, np.zeros_like(t)]
    faces, reason = face.propagateFaces(_movingPlate, [0, 1, 2, 3, 4], start)
    assert reason == "" and sorted(faces) == [1, 2, 3, 4]
    moved = faces[4]["landmarks"]
    inside = (start[:, 0] * width > 30) & (start[:, 0] * width < width - 30)
    np.testing.assert_allclose((moved[inside, 0] - start[inside, 0]) * width, 8.0, atol=0.5)
    np.testing.assert_allclose((moved[inside, 1] - start[inside, 1]) * height, 4.0, atol=0.5)
    assert faces[4]["source"] == "flow"
    # Backward: the other way
    back, _ = face.propagateFaces(_movingPlate, [4, 3, 2], moved)
    np.testing.assert_allclose((back[2]["landmarks"][inside, 0] - moved[inside, 0]) * width, -4.0, atol=0.5)


def test_mediapipeTakenWhereItAgreesWithTheFlow():
    t = np.linspace(0.3, 0.7, face.LANDMARKS)
    start = np.c_[t, 1.0 - t, np.zeros_like(t)]

    class Finder:
        def __init__(self, shift):
            self.shift = shift

        def detect(self, rgb, timestampMs):
            # The landmarks "found" in the crop: the flowed ones shifted (normalized to the crop: whole frame here)
            return self.landmarks, {"jawOpen": 0.2}
    flowed, _ = face.propagateFaces(_movingPlate, [0, 1], start)
    for shift, expected in ((0.5, "mediapipe"), (40.0, "flow")):
        finder = Finder(shift)
        landmarks = flowed[1]["landmarks"].copy()
        landmarks[:, 0] += shift / 320.0
        finder.landmarks = landmarks
        original = face._cropAround
        face._cropAround = lambda l, w, h: (0, 0, w, h)
        try:
            faces, _ = face.propagateFaces(_movingPlate, [0, 1], start, detector=finder)
        finally:
            face._cropAround = original
        assert faces[1]["source"] == expected


def test_faceTrackingStopsAtKeyframesAndRefines(tmp_path, monkeypatch):
    from meshroom.nodes.compositing.FaceNodes import FaceTracker
    frames = list(range(1000, 1011))
    assert FaceTracker._sequence("trackFaceForward", 1002, frames, [1002, 1006]) == [1002, 1003, 1004, 1005]
    assert FaceTracker._sequence("trackFaceBackward", 1006, frames, [1002, 1006]) == [1006, 1005, 1004, 1003]
    assert FaceTracker._sequence("trackFaceStepForward", 1005, frames, [1002, 1006]) == [1005, 1006]
    assert FaceTracker._sequence("trackFaceForward", 1007, frames, [1002, 1006]) == list(range(1007, 1011))


def test_refineBlendsBothDirections():
    from meshroom.nodes.compositing.FaceNodes import FaceTracker

    class Context:
        frames = list(range(0, 11))
    starts = {0: np.zeros((3, 3)), 4: np.ones((3, 3))}

    def propagate(start, sequence, callback):
        # Forward from 0 keeps 0, backward from 4 keeps 1
        return {f: {"landmarks": start.copy(), "blendshapes": {}, "hidden": [], "source": "flow"} for f in sequence[1:]}
    result = {}
    FaceTracker()._refine(Context(), {}, [0, 4], lambda f: starts[f], propagate, lambda f, found: result.__setitem__(f, found))
    assert sorted(result) == [1, 2, 3]
    np.testing.assert_allclose(result[1]["landmarks"], 0.25)
    np.testing.assert_allclose(result[3]["landmarks"], 0.75)
    with pytest.raises(RuntimeError):
        FaceTracker()._refine(Context(), {}, [0, 1], lambda f: starts[0], propagate, lambda f, found: None)


def test_stiffFaceIgnoresWrongLandmarks():
    frame = 3
    truth = face.toPixels(landmarksAt(frame), W, H)
    pixels = truth.copy()
    # A landmark drifted far away, and hidden ones given wrong positions
    wrong, hidden = MOUTH[3], [i for i in range(468) if 10 <= i // 18 <= 11 and 6 <= i % 18 <= 9]
    pixels[wrong] += [45.0, -30.0]
    pixels[hidden] += 25.0
    errors = {}
    for stiffness in (0.0, 0.3):
        geo, _, _, _ = face.solveFace(FACE, pixels, CAMERA, W, H, rigid=RIGID, deform=True, hidden=hidden,
                                      stiffness=stiffness)
        xy, _, _ = CAMERA.project(geo.vertices, W, H)
        errors[stiffness] = np.hypot(*(xy - truth).T)
    # Exact: the wrong landmark folds the mesh; stiff: it is ignored, the others still followed (the mouth opens)
    assert errors[0.0][wrong] > 40.0 and errors[0.3][wrong] < 4.0
    assert np.median(errors[0.3]) < 0.5 and errors[0.3][MOUTH].max() < 3.5 and errors[0.3][hidden].max() < 3.5


def test_flowFollowsTheRegularizedFace():
    t = np.linspace(0.3, 0.7, face.LANDMARKS)
    start = np.c_[t, 1.0 - t, np.zeros_like(t)]
    calls = []

    def regularize(frame, landmarks, hidden):
        calls.append(frame)
        fitted = landmarks.copy()
        fitted[:, 0] = landmarks[:, 0].mean()  # A "model" putting every point on one column
        return fitted
    faces, _ = face.propagateFaces(_movingPlate, [0, 1, 2], start, regularize=regularize)
    assert calls == [1, 2] and np.ptp(faces[2]["landmarks"][:, 0]) < 1e-9


def test_temporalSmoothingSteadiesTheFace(tmp_path, monkeypatch):
    monkeypatch.setattr(face, "RIGID_LANDMARKS", tuple(RIGID))
    graph, plate, tracker = faceGraph(tmp_path)
    rng = np.random.default_rng(3)
    # The landmarks of a steady head, with 1.5 pixels of noise at each frame
    truth = landmarksAt(0)
    faces = {f: {"landmarks": truth + np.c_[rng.normal(0, 1.5 / W, (468, 2)), np.zeros(468)], "blendshapes": {}}
             for f in range(1, 16)}
    tracker.faceTrack.value = face.dumpFaces(faces, (W, H))
    tracker.smoothing.value = 0.0
    jitter = {}
    for sigma in (0.0, 2.0):
        tracker.temporalSmoothing.value = sigma
        graph.update()
        plan = buildPlan(tracker)
        evaluator = Evaluator()
        heads = [evaluator.data(plan, plan.target, "headMatrix", f) for f in range(5, 12)]
        geos = [evaluator.data(plan, plan.target, "geoOutput", f).vertices for f in range(5, 12)]
        # How much the head and the face move from frame to frame (they should not: the head is still)
        jitter[sigma] = (max(np.abs(b[:3, 3] - a[:3, 3]).max() for a, b in zip(heads, heads[1:])),
                         max(np.abs(b - a).max() for a, b in zip(geos, geos[1:])))
    assert jitter[2.0][0] < 0.5 * jitter[0.0][0] and jitter[2.0][1] < 0.5 * jitter[0.0][1]


def test_referenceFrameIsTheBaseOfTheFit(tmp_path, monkeypatch):
    monkeypatch.setattr(face, "RIGID_LANDMARKS", tuple(RIGID))
    graph, plate, tracker = faceGraph(tmp_path)
    # The actor's face is wider than the mesh (a shape the landmarks show at every frame); the mouth opens over time
    faces = {}
    for f in range(1, 8):
        world = scene3d.transformPoints(headAt(f), deformedAt(f) * [1.15, 1.0, 1.0])
        xy, _, _ = CAMERA.project(world, W, H)
        faces[f] = {"landmarks": np.c_[xy[:, 0] / W, (H - xy[:, 1]) / H, np.zeros(len(xy))], "blendshapes": {}}
    tracker.faceTrack.value = face.dumpFaces(faces, (W, H))
    tracker.smoothing.value = 0.0
    tracker.temporalSmoothing.value = 0.0
    # Supple (the mouth of the synthetic grid is a block moved down: a stiff face rounds its edges)
    tracker.stiffness.value = 0.02
    tracker.deformModel.value = "surface"
    values = {attr.name: _snapshotValue(attr) for attr in tracker.attributes
              if not attr.isOutput and attr.type not in ("Image", "PushButtonParam")}
    context = ActionContext(values, Evaluator(), buildPlan(plate), 2, nodePlan=buildPlan(tracker))
    assert tracker.nodeDesc.runAction("setReference", context) == {"referenceFrame": 2, "useReference": True}
    tracker.referenceFrame.value, tracker.useReference.value = 2, True
    graph.update()
    plan = buildPlan(tracker)
    evaluator = Evaluator()
    for f in (2, 5, 7, 1):
        geo = evaluator.data(plan, plan.target, "geoOutput", f)
        xy, _, _ = CAMERA.project(geo.vertices[:face.MESH_VERTICES], W, H)
        errors = np.hypot(*(xy - face.toPixels(faces[f]["landmarks"], W, H)).T)
        assert np.median(errors) < 0.3 and errors.max() < 1.5
    # The other frames are fitted on the face of the reference frame (wider than the neutral mesh)
    rest = evaluator.cachedData(plan.target, "_rawFace", 2).vertices
    assert np.ptp(rest[:, 0]) > 1.05 * np.ptp(FACE.vertices[:, 0])


def test_rigFitsExpressionsAndKeepsRigidPartsRigid():
    from meshroom.live import faceRig
    rig = faceRig.FaceRig(FACE.vertices)
    rng = np.random.default_rng(5)
    params = rng.normal(0, 0.3, rig.paramCount)
    # Within what each control can do (the nose and the cheeks barely move)
    limits = np.repeat([faceRig.controlLimit(name)[0] / 0.1 for name, *_ in faceRig.CONTROLS], 3)
    params[:-2] = np.clip(params[:-2], -0.8 * limits, 0.8 * limits)
    params[-2:] = [0.12, 0.03]  # The jaw open, a little sideways
    expression = rig.deform(params)
    world = scene3d.transformPoints(headAt(2), expression)
    truth, _, _ = CAMERA.project(world, W, H)
    pixels = truth.copy()
    # A landmark far from every control (a rigid part) given a wrong position
    rigid = int(np.argmin(rig.weights.sum(axis=1) + rig.jawWeight))
    pixels[rigid] += [35.0, 20.0]
    report = {}
    geo, head, _, _ = face.solveFace(FACE, pixels, CAMERA, W, H, rigid=RIGID, deform=True, model="rig",
                                     stiffness=0.01, report=report)
    xy, _, _ = CAMERA.project(geo.vertices, W, H)
    errors = np.hypot(*(xy - truth).T)
    assert np.median(errors) < 0.3 and errors.max() < 1.5 and errors[rigid] < 1.5
    assert len(report["rigParams"]) == rig.paramCount
    # The rigid parts never move with the rig (only the head does)
    moved = np.linalg.norm(rig.deform(rng.normal(0, 1.0, rig.paramCount)) - FACE.vertices, axis=1)
    assert moved[rigid] < 0.05 * np.linalg.norm(FACE.vertices[faceRig.LANDMARK["sideR"]] - FACE.vertices[faceRig.LANDMARK["sideL"]])


def test_rigPins():
    from meshroom.live import faceRig
    rig = faceRig.FaceRig(FACE.vertices)
    pixels = face.toPixels(landmarksAt(2), W, H)
    # A pin on the mouth corner control, dragged: the control follows it (its area moves), the rigid parts do not
    corner = faceRig.LANDMARK["mouthR"]
    target = pixels[corner] + [0.0, -12.0]
    rest, _, _, _ = face.solveFace(FACE, pixels, CAMERA, W, H, rigid=RIGID, model="rig", stiffness=0.3)
    geo, _, _, _ = face.solveFace(FACE, pixels, CAMERA, W, H, rigid=RIGID, model="rig", stiffness=0.3,
                                  pins=(FACE.vertices[[corner]], target[None]))
    before, _, _ = CAMERA.project(rest.vertices, W, H)
    xy, _, _ = CAMERA.project(geo.vertices, W, H)
    # Exact at the pin (the rig, then a local warp for what it cannot do)
    assert np.hypot(*(xy[corner] - target)) < 0.5
    still = (rig.weights.sum(axis=1) + rig.jawWeight) < 0.01
    assert still.sum() > 20 and np.abs(xy[still] - before[still]).max() < 1.5


def test_rigStaysWithinWhatAFaceCanDo():
    from meshroom.live import faceRig
    rng = np.random.default_rng(2)
    pixels = face.toPixels(landmarksAt(2), W, H)
    # Wild landmarks on one side (a profile: MediaPipe guesses the hidden side)
    side = FACE.vertices[:, 0] > 2.0
    pixels[side] += rng.normal(0, 30, (int(side.sum()), 2)) + [60.0, -40.0]
    report = {}
    geo, head, _, _ = face.solveFace(FACE, pixels, CAMERA, W, H, rigid=RIGID, model="rig", stiffness=0.3, report=report)
    params = report["rigParams"]
    assert np.abs(params[:-2]).max() < faceRig.TRANSLATION_LIMIT + 0.3
    assert faceRig.JAW_LIMITS[0][0] - 0.1 < params[-2] < faceRig.JAW_LIMITS[0][1] + 0.1
    local = scene3d.transformPoints(np.linalg.inv(head), geo.vertices)
    width = np.linalg.norm(FACE.vertices[faceRig.LANDMARK["sideR"]] - FACE.vertices[faceRig.LANDMARK["sideL"]])
    assert np.linalg.norm(local - FACE.vertices, axis=1).max() < 0.35 * width


def _procrustes(a, b):
    """ RMS distance between two point sets after the best similarity (rotation, translation, scale). """
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, sv, vt = np.linalg.svd(a0.T @ b0)
    rotation = u @ vt
    scale = sv.sum() / (a0 ** 2).sum()
    return float(np.sqrt(np.mean(np.sum((scale * a0 @ rotation - b0) ** 2, axis=1))))


def test_fitShapeFindsTheActorsFace():
    # The actor: wider jaw, longer nose than the mesh (a smooth, symmetric difference), seen turning the head
    actor = FACE.vertices.copy()
    actor[:, 0] *= 1.0 + 0.12 * np.clip(-actor[:, 1] / 10.0, 0.0, 1.0)
    actor[:, 2] += 1.5 * np.exp(-(actor[:, 0] ** 2 + actor[:, 1] ** 2) / 8.0)
    observations = []
    for f, yaw in enumerate((-25, -12, 0, 12, 25, 35)):
        head = scene3d.composeTRS((1.0 * f - 2.0, -1.0, -60.0), (5.0, yaw, 3.0))
        xy, _, _ = CAMERA.project(scene3d.transformPoints(head, actor), W, H)
        observations.append({"pixels": xy, "hidden": [], "camera": CAMERA, "weight": 1.0})
    offsets = face.fitShape(FACE, observations, W, H)
    before = _procrustes(FACE.vertices, actor)
    after = _procrustes(FACE.vertices + offsets, actor)
    assert after < 0.5 * before
    # Stored
    np.testing.assert_allclose(face.parseShape(face.dumpShape(offsets)), offsets, atol=1e-5)


def test_fitShapeAction(tmp_path, monkeypatch):
    monkeypatch.setattr(face, "RIGID_LANDMARKS", tuple(RIGID))
    graph, plate, tracker = faceGraph(tmp_path)
    actor = FACE.vertices.copy()
    actor[:, 0] *= 1.12
    faces = {}
    for f, yaw in enumerate((-20, -8, 4, 16, 28), start=1):
        head = scene3d.composeTRS((0.5 * f, -1.0, -60.0), (4.0, yaw, 2.0))
        xy, _, _ = CAMERA.project(scene3d.transformPoints(head, actor), W, H)
        faces[f] = {"landmarks": np.c_[xy[:, 0] / W, (H - xy[:, 1]) / H, np.zeros(len(xy))], "blendshapes": {}}
    tracker.faceTrack.value = face.dumpFaces(faces, (W, H))
    graph.update()
    values = {attr.name: _snapshotValue(attr) for attr in tracker.attributes
              if not attr.isOutput and attr.type not in ("Image", "PushButtonParam")}
    context = ActionContext(values, Evaluator(), buildPlan(plate), 1, nodePlan=buildPlan(tracker))
    monkeypatch.setattr(ActionContext, "frames", property(lambda self: [1, 2, 3, 4, 5]))
    result = tracker.nodeDesc.runAction("fitShape", context)
    tracker.faceShape.value = result["faceShape"]
    graph.update()
    plan = buildPlan(tracker)
    shape = Evaluator().data(plan, plan.target, "shapeOutput", 1)
    assert face.isHead(shape)
    assert _procrustes(shape.vertices[:face.MESH_VERTICES], actor) < 0.6 * _procrustes(FACE.vertices, actor)
    assert tracker.nodeDesc.runAction("clearShape", context) == {"faceShape": ""}


def test_rigPinOnARigidPartOrientsTheHead():
    from meshroom.live import faceRig
    rig = faceRig.FaceRig(FACE.vertices)
    pixels = face.toPixels(landmarksAt(2), W, H)
    # A pin on a rigid part dragged: the head turns toward it (the pins weigh as much as the landmarks), nothing is
    # warped there (rig mode: a rigid part stays rigid, "Exact Pins" or not)
    rigid = int(np.argmin(rig.mobility))
    assert rig.mobility[rigid] < 0.05
    target = pixels[rigid] + [10.0, 6.0]
    kwargs = dict(rigid=RIGID, model="rig", stiffness=0.3, pins=(FACE.vertices[[rigid]], target[None]), pinRadius=2.0)
    geo, head, _, _ = face.solveFace(FACE, pixels, CAMERA, W, H, **kwargs)
    unwarped, _, _, _ = face.solveFace(FACE, pixels, CAMERA, W, H, exactPins=False, **kwargs)
    np.testing.assert_allclose(geo.vertices, unwarped.vertices, atol=1e-9)
    free, _, _, _ = face.solveFace(FACE, pixels, CAMERA, W, H, rigid=RIGID, model="rig", stiffness=0.3)
    xy, _, _ = CAMERA.project(geo.vertices, W, H)
    xyFree, _, _ = CAMERA.project(free.vertices, W, H)
    assert np.hypot(*(xy[rigid] - target)) < 0.6 * np.hypot(*(xyFree[rigid] - target))
    # The face keeps its shape around the pin: the same rigid part, moved as a whole
    local = scene3d.transformPoints(np.linalg.inv(head), geo.vertices)
    near = np.linalg.norm(FACE.vertices - FACE.vertices[rigid], axis=1) < 3.0
    assert np.abs(local[near] - FACE.vertices[near]).max() < 0.2


def test_exactPinOnTheMouthIsReached():
    from meshroom.live import faceRig
    rig = faceRig.FaceRig(FACE.vertices)
    pixels = face.toPixels(landmarksAt(2), W, H)
    mobile = int(np.argmax(rig.mobility * (FACE.vertices[:, 1] < 0)))
    target = pixels[mobile] + [6.0, -4.0]
    geo, _, _, _ = face.solveFace(FACE, pixels, CAMERA, W, H, rigid=RIGID, model="rig", stiffness=0.3,
                                  pins=(FACE.vertices[[mobile]], target[None]), pinRadius=2.0)
    xy, _, _ = CAMERA.project(geo.vertices, W, H)
    assert np.hypot(*(xy[mobile] - target)) < 0.5
