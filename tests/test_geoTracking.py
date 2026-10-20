"""Tracking: tracks tracked together (tracking.trackMany), pins of a Resection, geometry tracking (geotrack.py)."""

import json
import os

import numpy as np
import pytest

import meshroom
from meshroom.core import desc, pluginManager
from meshroom.core.graph import Graph
from meshroom.live import Evaluator, ImageBuffer, buildPlan, geotrack, scene3d, tracking
from meshroom.live.actions import ActionContext
from meshroom.live.evaluator import _snapshotValue
from meshroom.live.imageBuffer import ImageInfo

from .utils import registerNodeDesc, unregisterNodeDesc

_COMPOSITING_FOLDER = os.path.join(os.path.dirname(meshroom.__file__), "nodes", "compositing")
if not pluginManager.isNodeDescRegistered("Grade"):
    pluginManager.addPluginFromBuiltInFolder("compositing", _COMPOSITING_FOLDER)

W, H = 320, 180
_CUBE = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
_CUBE_FACES = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]
CUBE = scene3d.GeoData(_CUBE, [4] * 6, sum(_CUBE_FACES, []))


def cameraAt(frame) -> scene3d.CameraData:
    """ The true camera of the synthetic plate: turning around the cube and rising. """
    position = (2.5 * np.sin(0.03 * frame), 1.5 + 0.03 * frame, 6.0 * np.cos(0.03 * frame))
    return scene3d.CameraData(scene3d.lookAt(position, (0, 0, 0)), 35, 36, 24)


def _texture(points):
    x, y, z = points.T
    value = np.sin(9.1 * x + 4.3 * y) * np.sin(8.7 * z - 3.1 * x) + np.sin(11.3 * y + 5.9 * z) * np.sin(10.1 * x - 4.2 * z) \
        + 0.5 * np.sin(23.7 * x - 17.3 * y + 13.1 * z)
    return (0.5 + 0.2 * value).astype(np.float32)


def renderCube(camera) -> ImageBuffer:
    """ The textured cube seen by a camera (ray casting), on a dark background. """
    rows, columns = np.mgrid[0:H, 0:W]
    x, y = columns + 0.5, H - rows - 0.5
    focal = camera.focalPixels(W)
    directions = np.stack([(x - W / 2) / focal, (y - H / 2) / focal, -np.ones_like(x, dtype=float)], -1).reshape(-1, 3)
    directions = directions @ camera.matrix[:3, :3].T
    origin = camera.position
    with np.errstate(divide="ignore", invalid="ignore"):
        near, far = (-1 - origin) / directions, (1 - origin) / directions
    entry = np.nanmax(np.minimum(near, far), axis=1)
    exit = np.nanmin(np.maximum(near, far), axis=1)
    hit = (exit >= entry) & (exit > 0)
    lum = np.full(len(directions), 0.05, np.float32)
    lum[hit] = _texture(origin + directions[hit] * entry[hit, np.newaxis])
    lum = lum.reshape(H, W)
    return ImageBuffer.fromRGBA(np.stack([lum, lum, lum, np.ones_like(lum)], -1), (W, H))


def pixelError(matrix, frame) -> float:
    """ The largest distance in pixels between the cube corners seen with `matrix` and with the true camera. """
    focal = cameraAt(frame).focalPixels(W)
    xy, _ = geotrack.projectPoints(matrix, _CUBE, W, H, focal)
    truth, _ = geotrack.projectPoints(cameraAt(frame).matrix, _CUBE, W, H, focal)
    return float(np.abs(xy - truth).max())


class SyntheticCubePlate(desc.ImageOpNode):
    """ The textured cube filmed by cameraAt, frames 0 to 20. """
    category = "Test"
    inputs = []
    outputs = [desc.Image(name="output", label="Output", description="")]

    def getInfo(self, ctx, inputsInfo):
        return ImageInfo(W, H, 0, 20)

    def render(self, ctx, inputs):
        return renderCube(cameraAt(ctx.frame))


def setup_module():
    registerNodeDesc(SyntheticCubePlate)


def teardown_module():
    unregisterNodeDesc(SyntheticCubePlate)


# Tracks tracked together

def _blobs(frame, height=80, width=160):
    rows, columns = np.mgrid[0:height, 0:width]
    x, y = columns + 0.5, height - rows - 0.5
    lum = np.zeros((height, width), np.float32)
    for cx, cy, dx, dy in ((30, 40, 2.0, 1.0), (110, 30, -1.5, 1.5)):
        bx, by = cx + dx * frame, cy + dy * frame
        lum += np.exp(-((x - bx) ** 2 + (y - by) ** 2) / 32.0) + 0.4 * np.exp(-((x - bx - 5) ** 2 + (y - by + 3) ** 2) / 8.0)
    rgba = np.zeros((height, width, 4), np.float32)
    rgba[:, :, :3] = lum[:, :, np.newaxis]
    return ImageBuffer.fromRGBA(rgba, (width, height))


def test_tracksAreTrackedTogetherFrameAfterFrame():
    rendered = []

    def imageAt(frame):
        rendered.append(frame)
        return _blobs(frame)
    tracks = [{"keys": {"0": [30, 40]}, "pattern": 21, "search": 41},
              {"keys": {"0": [110, 30]}, "pattern": 21, "search": 41}]
    seen = []
    keys = tracking.trackMany(tracks, list(range(9)), imageAt, onKey=lambda index, frame, *_: seen.append((frame, index)))
    # Each frame is rendered once, all the tracks move to it before the next one
    assert rendered == list(range(9))
    assert seen[:4] == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert keys[0]["8"] == pytest.approx([46, 48], abs=0.05)
    assert keys[1]["8"] == pytest.approx([98, 42], abs=0.05)


def test_eachTrackStopsAtItsOwnFrame():
    tracks = [{"keys": {"0": [30, 40]}}, {"keys": {"0": [110, 30]}}]
    keys = tracking.trackMany(tracks, list(range(9)), _blobs, stopAt=[4, None])
    assert sorted(keys[0], key=int) == ["1", "2", "3"]
    assert len(keys[1]) == 8


def test_trackFramesIsOneTrackTracked():
    scores = {}
    keys = tracking.trackFrames({"keys": {"0": [30, 40]}}, list(range(5)), _blobs, scores=scores)
    assert keys["4"] == pytest.approx([38, 44], abs=0.05)
    assert sorted(scores, key=int) == ["1", "2", "3", "4"]


def test_pinsModelRejectsASlippingPin():
    from meshroom.nodes.compositing.ThreeDNodes import _PinsModel
    focal = cameraAt(0).focalPixels(W)
    tracks = [{"point": point.tolist()} for point in _CUBE]
    model = _PinsModel(tracks, W, H, focal, cameraAt(0).matrix)
    truth, _ = geotrack.projectPoints(cameraAt(1).matrix, _CUBE, W, H, focal)
    positions = {index: tuple(xy) for index, xy in enumerate(truth)}
    positions[3] = (positions[3][0] + 12.0, positions[3][1] - 7.0)
    assert model.reject(1, positions) == {3}
    # The pins are searched where the pose (moving at constant speed) puts them
    predicted = model.predict(2, positions)
    expected, _ = geotrack.projectPoints(cameraAt(2).matrix, _CUBE, W, H, focal)
    assert max(np.hypot(predicted[i][0] - expected[i][0], predicted[i][1] - expected[i][1]) for i in range(8)) < 1.0


# Geometry tracking

def test_robustPoseIgnoresWrongMatches():
    focal = cameraAt(0).focalPixels(W)
    rng = np.random.default_rng(3)
    points = rng.uniform(-1, 1, (60, 3))
    positions, _ = geotrack.projectPoints(cameraAt(5).matrix, points, W, H, focal)
    positions[:10] += rng.uniform(15, 30, (10, 2))
    matrix, inliers, error = geotrack.robustPose(points, positions, W, H, focal, cameraAt(0).matrix)
    assert not inliers[:10].any() and inliers[10:].all()
    assert error < 0.01 and pixelError(matrix, 5) < 0.01


def test_visibleSamplesFaceTheCamera():
    samples = geotrack.SurfaceSamples(CUBE, count=5000)
    camera = scene3d.CameraData(scene3d.lookAt((0, 0, 6), (0, 0, 0)), 35, 36, 24)
    indices, _ = samples.visible(camera.matrix, W, H, camera.focalPixels(W), margin=0)
    # Only the front face (z = 1) is seen
    assert len(indices) and np.allclose(samples.points[indices, 2], 1.0)


@pytest.mark.parametrize("method", geotrack.METHODS)
def test_geometryFollowsThePlate(method):
    focal = cameraAt(0).focalPixels(W)
    poses, reason = geotrack.trackPose(geotrack.SurfaceSamples(CUBE), geotrack.TrackSettings(method), list(range(11)),
                                       lambda frame: renderCube(cameraAt(frame)), cameraAt(0).matrix, W, H, focal)
    assert reason == "" and sorted(poses) == list(range(1, 11))
    assert max(pixelError(poses[frame], frame) for frame in poses) < 2.0


def test_alignFindsTheRightPosition():
    focal = cameraAt(0).focalPixels(W)
    rough = cameraAt(6).matrix.copy()
    rough[:3, 3] += [0.12, -0.08, 0.2]
    assert pixelError(rough, 6) > 8.0
    matrix, inliers, _ = geotrack.alignPose(geotrack.SurfaceSamples(CUBE), geotrack.TrackSettings(), cameraAt(0).matrix,
                                            renderCube(cameraAt(0)), rough, renderCube(cameraAt(6)), W, H, focal)
    assert inliers >= 8 and pixelError(matrix, 6) < 1.5


def test_blendPosesTrustsTheNearestKeyframe():
    a, b = np.eye(4), np.eye(4)
    b[:3, 3] = [4.0, 0, 0]
    blended = geotrack.blendPoses({1: a, 2: a, 3: a}, {1: b, 2: b, 3: b}, 0, 4)
    assert [round(blended[frame][0, 3], 6) for frame in (1, 2, 3)] == [1.0, 2.0, 3.0]
    assert geotrack.blendPoses({1: a}, {}, 0, 4) == {1: a}


def test_posesAreStoredAsJson():
    poses = {3: cameraAt(3).matrix, 7: cameraAt(7).matrix}
    text = geotrack.dumpPoses(poses, [3])
    parsed, keyframes = geotrack.parsePoses(text)
    assert keyframes == [3] and sorted(parsed) == [3, 7]
    np.testing.assert_allclose(parsed[7], poses[7], atol=1e-8)
    assert geotrack.dumpPoses({}, []) == "" and geotrack.parsePoses("") == ({}, [])
    with pytest.raises(ValueError):
        geotrack.parsePoses('{"poses": {"1": [1, 2]}}')


# The Resection node

def resectionGraph(tmp_path):
    """ The synthetic plate, the cube and a Resection whose 8 pins place the cube at frame 0 (a keyframe). """
    graph = Graph("geoTracking")
    plate = graph.addNewNode("SyntheticCubePlate")
    path = tmp_path / "cube.obj"
    with open(path, "w") as file:
        file.writelines(f"v {x} {y} {z}\n" for x, y, z in _CUBE)
        file.writelines("f " + " ".join(str(i + 1) for i in face) + "\n" for face in _CUBE_FACES)
    readGeo = graph.addNewNode("ReadGeo")
    readGeo.file.value = str(path)
    resection = graph.addNewNode("Resection")
    graph.addEdge(plate.output, resection.input)
    graph.addEdge(readGeo.output, resection.geo)
    xy, _, _ = cameraAt(0).project(_CUBE, W, H)
    resection.tracks.value = json.dumps([{"name": f"pin{i}", "enabled": True, "point": _CUBE[i].tolist(),
                                          "keys": {"0": xy[i].tolist()}, "keyframes": [0], "format": [W, H]}
                                         for i in range(8)])
    graph.update()
    return graph, plate, resection


def runAction(resection, plate, name, frame):
    """ Run an action of the Resection as the interface does (ActionContext), returns its result. """
    values = {attr.name: _snapshotValue(attr) for attr in resection.attributes
              if not attr.isOutput and attr.type not in ("Image", "PushButtonParam")}
    context = ActionContext(values, Evaluator(), buildPlan(plate), frame, nodePlan=buildPlan(resection))
    return resection.nodeDesc.runAction(name, context)


def cameraOutput(resection, frame):
    plan = buildPlan(resection)
    return Evaluator().data(plan, plan.target, "cameraOutput", frame)


def test_resectionTracksTheGeometry(tmp_path):
    graph, plate, resection = resectionGraph(tmp_path)
    result = runAction(resection, plate, "trackGeoForward", 0)
    poses, keyframes = geotrack.parsePoses(result["geoTrack"])
    # From the keyframe of the pins to the end of the plate
    assert keyframes == [0] and sorted(poses) == list(range(21))
    resection.geoTrack.value = result["geoTrack"]
    graph.update()
    assert pixelError(cameraOutput(resection, 12).matrix, 12) < 3.0


def test_resectionAlignsTheGeometryAtAFrame(tmp_path):
    graph, plate, resection = resectionGraph(tmp_path)
    rough = cameraAt(6).matrix.copy()
    rough[:3, 3] += [0.12, -0.08, 0.2]
    resection.geoTrack.value = geotrack.dumpPoses({6: rough}, [])
    graph.update()
    assert pixelError(cameraOutput(resection, 6).matrix, 6) > 8.0
    result = runAction(resection, plate, "alignGeo", 6)
    poses, keyframes = geotrack.parsePoses(result["geoTrack"])
    assert keyframes == [6] and pixelError(poses[6], 6) < 1.5


def test_resectionRefinesBetweenKeyframes(tmp_path):
    graph, plate, resection = resectionGraph(tmp_path)
    resection.geoTrack.value = geotrack.dumpPoses({8: cameraAt(8).matrix}, [8])
    graph.update()
    poses, keyframes = geotrack.parsePoses(runAction(resection, plate, "refineGeo", 0)["geoTrack"])
    assert keyframes == [0, 8] and sorted(poses) == list(range(9))
    assert max(pixelError(poses[frame], frame) for frame in range(1, 8)) < 2.0
    assert runAction(resection, plate, "clearGeoTrack", 0) == {"geoTrack": ""}


def test_trackedPosesPrevailExceptAtThePinKeyframes(tmp_path):
    graph, plate, resection = resectionGraph(tmp_path)
    shifted = cameraAt(0).matrix.copy()
    shifted[:3, 3] += [0.3, 0, 0]
    resection.geoTrack.value = geotrack.dumpPoses({0: shifted, 5: shifted}, [])
    graph.update()
    # At the keyframe of the pins, they solve from the stored pose (back to the truth); elsewhere the stored pose
    assert pixelError(cameraOutput(resection, 0).matrix, 0) < 0.01
    np.testing.assert_allclose(cameraOutput(resection, 5).matrix, shifted)


def test_aFourPinSlipIsRejected():
    from meshroom.nodes.compositing.ThreeDNodes import _PinsModel
    focal = cameraAt(0).focalPixels(W)
    points = _CUBE[[0, 2, 5, 7]]
    model = _PinsModel([{"point": point.tolist()} for point in points], W, H, focal, cameraAt(0).matrix)
    truth, _ = geotrack.projectPoints(cameraAt(1).matrix, points, W, H, focal)
    positions = {index: tuple(xy) for index, xy in enumerate(truth)}
    assert model.reject(1, positions) == set()
    positions[2] = (positions[2][0] + 25.0, positions[2][1])
    assert model.reject(1, positions) == {2}


def test_trackedPosesAreNotReplacedByThePinsSolve(tmp_path):
    graph, plate, resection = resectionGraph(tmp_path)
    tracked = cameraAt(3).matrix
    resection.geoTrack.value = geotrack.dumpPoses({3: tracked}, [])
    graph.update()
    plan = buildPlan(resection)
    evaluator = Evaluator()
    # Frame 5 solved first (its pins chain goes through frame 3), then frame 3: still the tracked pose
    evaluator.data(plan, plan.target, "cameraOutput", 5)
    np.testing.assert_allclose(evaluator.data(plan, plan.target, "cameraOutput", 3).matrix, tracked)


# Drawing of the geometry (meshroom/live/geoRender.py)

def test_depthBufferCoversTheCubeAsRayCasting():
    from meshroom.live import geoRender
    from meshroom.live.imageBuffer import BBox
    camera = cameraAt(3)
    u, v, z, _ = geoRender.screenPositions(CUBE, camera, W, H, BBox(0, 0, W, H))
    zbuffer, ids = geoRender.rasterize(u, v, z, CUBE.triangles(), (H, W))
    # The background of the synthetic render is 0.05: the cube is where the rays hit it
    hit = renderCube(camera).rgba()[:, :, 0] != np.float32(0.05)
    covered = np.isfinite(zbuffer)
    assert (covered == hit).mean() > 0.995 and ((ids >= 0) == covered).all()
    # The depth of the cube center row is between the near and far faces
    assert 4.0 < np.nanmin(np.where(covered, zbuffer, np.nan)) < 6.5


def test_featureEdgesOfACube():
    from meshroom.live import geoRender
    # The diagonals of the quads (flat) are not drawn; the 12 edges of the cube are creases
    edges = geoRender.featureEdges(CUBE, cameraAt(0).position)
    assert len(edges) == 12 and {tuple(e) for e in edges} == {tuple(e) for e in CUBE.edges()}


def test_hiddenEdgesAreNotDrawn():
    from meshroom.live import geoRender
    from meshroom.live.imageBuffer import BBox
    camera = scene3d.CameraData(scene3d.lookAt((0, 0, 6), (0, 0, 0)), 35, 36, 24)
    drawn = {}
    for hidden in (False, True):
        rgba = np.zeros((H, W, 4), np.float32)
        geoRender.drawGeometry(rgba, BBox(0, 0, W, H), CUBE, camera, W, H, "wireframe", (1, 1, 1), 1.0,
                               hiddenLines=hidden)
        drawn[hidden] = rgba[:, :, 3] > 0.5
    # Facing the cube: the back square (smaller) is drawn only without the hidden lines
    back, _ = geotrack.projectPoints(camera.matrix, _CUBE[:4], W, H, camera.focalPixels(W))
    row, column = int(H - back[0, 1]), int(round(back[0, 0] + (back[1, 0] - back[0, 0]) / 2))
    assert drawn[False][row - 1:row + 2, column].any() and not drawn[True][row - 1:row + 2, column].any()
    assert drawn[True].sum() < drawn[False].sum()


def test_shadedDisplay():
    from meshroom.live import geoRender
    from meshroom.live.imageBuffer import BBox
    rgba = np.zeros((H, W, 4), np.float32)
    geoRender.drawGeometry(rgba, BBox(0, 0, W, H), CUBE, cameraAt(3), W, H, "shaded", (0.0, 1.0, 0.0), 1.0)
    hit = renderCube(cameraAt(3)).rgba()[:, :, 0] != np.float32(0.05)
    # The silhouette lines spill over the edge by a pixel (antialiased)
    assert rgba[hit, 1].mean() > 0.3 and rgba[~hit, 1].mean() < 0.02 and rgba[:, :, 0].max() == 0.0


def test_topologySharedAcrossFrames(tmp_path):
    from meshroom.live import Evaluator
    graph, plate, resection = resectionGraph(tmp_path)
    readGeo = resection.geo.inputRootLink.node
    readGeo.translate.value = {"x": 1.0, "y": 0.0, "z": 0.0}
    graph.update()
    plan = buildPlan(readGeo)
    evaluator = Evaluator()
    first, second = (evaluator.data(plan, plan.target, "output", frame) for frame in (1, 2))
    # The faces (and their edges, computed once) are shared by the geometry of every frame
    assert first is not second and first.topology is second.topology
    assert first.edges() is second.edges()
