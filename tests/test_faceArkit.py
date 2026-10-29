""" The ARKit rig of the FaceTracker (faceRig.BlendshapeRig, data/faceExpressions.npz). """
import numpy as np

from meshroom.live import face, faceRig, scene3d

W, H = 1280, 720
CAMERA = scene3d.CameraData(np.eye(4), 50.0, 36.0, 24.0)
CAMERA_TO_FACE = np.linalg.inv(scene3d.composeTRS((1.0, -2.0, -50.0), (5.0, 20.0, 3.0)))


def _project(local):
    """ The pixels (Nuke) of face-space points seen with CAMERA_TO_FACE. """
    view = np.linalg.inv(CAMERA_TO_FACE)
    world = local @ view[:3, :3].T + view[:3, 3]
    xy, _, _ = CAMERA.project(world, W, H)
    return xy


def _weights(rig, **values):
    params = np.zeros(rig.paramCount)
    for name, value in values.items():
        params[rig.names.index(name)] = value
    return params


def test_expressionShapes():
    names, deltas = faceRig.expressionShapes()
    head = face.canonicalHead()
    assert len(names) == 51 and deltas.shape == (51, len(head.vertices), 3)
    assert "jawOpen" in names and "eyeBlinkLeft" in names and "tongueOut" not in names
    rig = faceRig.rigOf(head.vertices, "arkit", head=True)
    vertices = head.vertices
    # Left: the left of the face (+X, MediaPipe's 263 eye corner)
    smile = rig.deform(_weights(rig, mouthSmileLeft=1.0)) - vertices
    assert np.linalg.norm(smile[291]) > 0.5 and np.linalg.norm(smile[61]) < 0.05
    # The blinks close the eyes
    for name, up, low in (("eyeBlinkLeft", 386, 374), ("eyeBlinkRight", 159, 145)):
        closed = rig.deform(_weights(rig, **{name: 1.0}))
        assert abs(closed[up, 1] - closed[low, 1]) < 0.02 < vertices[up, 1] - vertices[low, 1]
    # The jaw opens the mouth; the skull and the nose bridge never move
    opened = rig.deform(_weights(rig, jawOpen=0.5))
    assert np.linalg.norm(opened[13] - opened[14]) > np.linalg.norm(vertices[13] - vertices[14]) + 1.0
    everything = rig.deform(np.ones(rig.paramCount)) - vertices
    # (the back of the head; the forehead moves with the brows)
    skull = (vertices[:, 2] < -3.0) & (vertices[:, 1] > 3.0)
    # (all 51 shapes at once: the scalp follows the brows a little)
    assert skull.sum() > 50 and np.linalg.norm(everything[skull], axis=1).max() < 0.25
    assert rig.mobility[skull].max() < 0.35 and rig.mobility[13] == 1.0


def test_arkitRigFitsTheExpression():
    head = face.canonicalHead()
    rig = faceRig.rigOf(head.vertices, "arkit", head=True)
    truth = _weights(rig, jawOpen=0.35, mouthSmileLeft=0.7, mouthSmileRight=0.5, eyeBlinkRight=0.9, browInnerUp=0.6,
                     browOuterUpLeft=0.4)
    pixels = _project(rig.deform(truth))[:face.MESH_VERTICES] + np.random.default_rng(0).normal(0, 1.0, (468, 2))
    report = {}
    geo, _, _, _ = face.solveFace(head, pixels, CAMERA, W, H, model="arkit", stiffness=0.05, report=report)
    params = report["rigParams"]
    assert np.abs(params - truth).max() < 0.1
    assert params.min() >= 0.0 and params.max() <= 1.0
    xy, _, _ = CAMERA.project(geo.vertices, W, H)
    assert np.hypot(*(xy[:468] - _project(rig.deform(truth))[:468]).T).mean() < 0.6
    # The face only (468 vertices): the same rig on it
    faceMesh = face.facePart(head) if hasattr(face, "facePart") else None
    if faceMesh is not None and len(faceMesh.vertices) == face.MESH_VERTICES:
        report = {}
        face.solveFace(faceMesh, pixels, CAMERA, W, H, model="arkit", stiffness=0.05, report=report)
        assert np.abs(report["rigParams"] - truth).max() < 0.12


def test_arkitPoseKeptAndRigFollowed():
    head = face.canonicalHead()
    rig = faceRig.rigOf(head.vertices, "arkit", head=True)
    truth = _weights(rig, jawOpen=0.2, mouthPucker=0.6)
    pixels = _project(rig.deform(truth))[:face.MESH_VERTICES]
    # An expression pin pulling the mouth corner: reached, the rig face (followed to the next frame) without the warp
    target = pixels[61] + [0.0, -8.0]
    report = {}
    geo, _, cameraToFace, _ = face.solveFace(head, pixels, CAMERA, W, H, model="arkit", stiffness=0.05, report=report,
                                             pins=(head.vertices[[61]], target[None], np.array([True])),
                                             pose=CAMERA_TO_FACE)
    np.testing.assert_allclose(cameraToFace, CAMERA_TO_FACE, atol=1e-9)
    xy, _, _ = CAMERA.project(geo.vertices, W, H)
    assert np.hypot(*(xy[61] - target)) < 0.5
    followed, _, _ = CAMERA.project(report["rigWorld"], W, H)
    assert len(followed) == len(geo.vertices)
    # The skull does not move with the expression pin
    skull = head.vertices[:, 1] > head.vertices[10, 1] + 2.0
    assert np.abs(xy[skull] - _project(head.vertices)[skull]).max() < 0.5


def test_mediaPipeScoresAsPrior():
    rig = faceRig.rigOf(face.canonicalHead().vertices, "arkit", head=True)
    prior = rig.fromMediaPipe({"_neutral": 0.1, "jawOpen": 0.4, "eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.2, "tongueOut": 1.0})
    assert prior[rig.names.index("jawOpen")] == 0.4
    # The two sides averaged: the landmarks tell them apart
    assert prior[rig.names.index("eyeBlinkLeft")] == prior[rig.names.index("eyeBlinkRight")] == 0.5
    assert rig.fromMediaPipe({}) is None
    # A shape the landmarks barely see, guided by the prior
    head = face.canonicalHead()
    pixels = _project(head.vertices)[:face.MESH_VERTICES]
    puff = np.zeros(rig.paramCount)
    puff[rig.names.index("cheekPuff")] = 1.0
    reports = []
    for weight in (0.0, 50.0):
        report = {}
        face.solveFace(head, pixels, CAMERA, W, H, model="arkit", stiffness=0.05, report=report, prior=puff,
                       priorWeight=weight, pose=CAMERA_TO_FACE)
        reports.append(report["rigParams"][rig.names.index("cheekPuff")])
    assert reports[1] > reports[0] + 0.05


def test_quadEdgesOfTheTrackedHead():
    from meshroom.live import geoRender
    head = face.canonicalHead()
    assert len(geoRender.quadEdges(head)) < 0.75 * len(head.edges())
