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
