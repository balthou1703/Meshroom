"""3D compositing: camera / geo / matrix streams, Camera, ReadGeo, MergeGeo, Resection, matrices of the 2D nodes."""

import json
import os

import numpy as np
import pytest

import meshroom
from meshroom.core import desc, pluginManager
from meshroom.core.graph import Graph, loadGraph
from meshroom.live import Evaluator, buildPlan, scene3d
from meshroom.live.evaluator import EvaluationError, RenderContext, isEvaluable

_COMPOSITING_FOLDER = os.path.join(os.path.dirname(meshroom.__file__), "nodes", "compositing")
if not pluginManager.isNodeDescRegistered("Grade"):
    pluginManager.addPluginFromBuiltInFolder("compositing", _COMPOSITING_FOLDER)


def setup_module():
    # The subgraph nodes (registered and unregistered by other test modules)
    from meshroom.nodes.subgraph.GraphInput import GraphInput
    from meshroom.nodes.subgraph.GraphOutput import GraphOutput
    from meshroom.nodes.subgraph.SubGraph import SubGraph
    from .utils import registerNodeDesc
    for nodeDesc in (GraphInput, GraphOutput, SubGraph):
        if not pluginManager.isNodeDescRegistered(nodeDesc.__name__):
            registerNodeDesc(nodeDesc)


_CUBE = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
_CUBE_FACES = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]


def writeCube(path):
    with open(path, "w") as file:
        file.writelines(f"v {x} {y} {z}\n" for x, y, z in _CUBE)
        file.writelines("f " + " ".join(str(i + 1) for i in face) + "\n" for face in _CUBE_FACES)
    return str(path)


def data(node, output, frame=1, evaluator=None):
    plan = buildPlan(node)
    return (evaluator or Evaluator()).data(plan, plan.target, output, frame)


def resectionGraph(tmp_path, width=640, height=360):
    graph = Graph("resection")
    plate = graph.addNewNode("Constant", width=width, height=height)
    readGeo = graph.addNewNode("ReadGeo")
    readGeo.file.value = writeCube(tmp_path / "cube.obj")
    camera = graph.addNewNode("Camera")
    camera.translate.value = {"x": 0.0, "y": 0.0, "z": 8.0}
    resection = graph.addNewNode("Resection")
    graph.addEdge(plate.output, resection.input)
    graph.addEdge(readGeo.output, resection.geo)
    graph.addEdge(camera.output, resection.camera)
    graph.update()
    return graph, plate, readGeo, camera, resection


def pinsFor(truthByFrame, indices, width=640, height=360, keyframes=None):
    """ Pins on the cube vertices `indices`, at their projection by the cameras {frame: CameraData}. """
    pins = []
    for i in indices:
        keys = {}
        for frame, truth in truthByFrame.items():
            xy, _, _ = truth.project(_CUBE[i:i + 1], width, height)
            keys[str(frame)] = xy[0].tolist()
        pins.append({"name": f"pin{i}", "enabled": True, "point": _CUBE[i].tolist(), "keys": keys,
                     "keyframes": keyframes or [min(truthByFrame)], "format": [width, height]})
    return json.dumps(pins)


class TestStreams:

    def test_streamsConnectOnlyToTheirType(self):
        graph = Graph("streams")
        camera = graph.addNewNode("Camera")
        other = graph.addNewNode("Camera")
        readGeo = graph.addNewNode("ReadGeo")
        transform = graph.addNewNode("Transform")
        assert other.input.validateIncomingConnection(camera.output)
        assert not other.input.validateIncomingConnection(readGeo.output)
        assert not transform.input.validateIncomingConnection(camera.output)
        assert other.inputMatrix.validateIncomingConnection(transform.matrix)
        assert not other.inputMatrix.validateIncomingConnection(camera.output)

    def test_everyCompositingNodeHasAHiddenMatrixOutput(self):
        for nodeType in ("Grade", "Transform", "Merge", "Camera", "ReadGeo", "Resection", "CornerPin"):
            nodeDesc = pluginManager.getNodeDescProvider(nodeType).nodeDescClass
            matrices = [output for output in nodeDesc.outputs if output.name == "matrix"]
            assert len(matrices) == 1 and isinstance(matrices[0], desc.Matrix) and not matrices[0].exposed, nodeType

    def test_matrixOfANodeWithoutTransformIsIdentity(self):
        graph = Graph("identity")
        constant = graph.addNewNode("Constant", width=8, height=4)
        grade = graph.addNewNode("Grade")
        graph.addEdge(constant.output, grade.input)
        graph.update()
        np.testing.assert_allclose(data(grade, "matrix"), np.eye(4))

    def test_dataNodesAreNotEvaluableImages(self):
        graph = Graph("data")
        camera = graph.addNewNode("Camera")
        constant = graph.addNewNode("Constant", width=8, height=4)
        assert not isEvaluable(camera)
        assert isEvaluable(constant)

    def test_streamChangeUpstreamChangesTheUid(self):
        graph = Graph("uid")
        camera = graph.addNewNode("Camera")
        other = graph.addNewNode("Camera")
        graph.addEdge(camera.matrix, other.inputMatrix)
        graph.update()
        before = other._uid
        camera.translate.value = {"x": 1.0, "y": 0.0, "z": 10.0}
        graph.update()
        assert other._uid != before

    def test_disabledCameraPassesItsInput(self):
        graph = Graph("disabled")
        source = graph.addNewNode("Camera")
        source.focalLength.value = 85.0
        camera = graph.addNewNode("Camera")
        graph.addEdge(source.output, camera.input)
        camera.internalAttribute("disable").value = True
        graph.update()
        assert data(camera, "output").focalLength == 85.0
        np.testing.assert_allclose(data(camera, "matrix"), np.eye(4))


class TestMatrices2D:

    def test_transformDrivenByTheMatrixOfAnother(self):
        graph = Graph("matrix")
        constant = graph.addNewNode("Constant", width=64, height=36)
        transform = graph.addNewNode("Transform")
        driven = graph.addNewNode("Transform")
        graph.addEdge(constant.output, transform.input)
        graph.addEdge(constant.output, driven.input)
        transform.translate.value = {"x": 5.5, "y": 2.0}
        transform.rotate.value = 12.0
        graph.addEdge(transform.matrix, driven.inputMatrix)
        graph.update()
        evaluator = Evaluator()
        np.testing.assert_allclose(evaluator.render(buildPlan(transform), 1).rgba(),
                                   evaluator.render(buildPlan(driven), 1).rgba())

    def test_cornerPinMatrixMapsTheCorners(self):
        graph = Graph("cornerPin")
        constant = graph.addNewNode("Constant", width=100, height=50)
        pin = graph.addNewNode("CornerPin")
        graph.addEdge(constant.output, pin.input)
        pin.to3.value = {"x": 1800.0, "y": 1000.0}
        graph.update()
        matrix = scene3d.extract2D(data(pin, "matrix"))
        corner = matrix @ np.array([1920.0, 1080.0, 1.0])
        np.testing.assert_allclose(corner[:2] / corner[2], [1800.0, 1000.0], atol=1e-6)

    def test_bakeTransformMatrix(self):
        graph = Graph("bake")
        constant = graph.addNewNode("Constant", width=64, height=36)
        transform = graph.addNewNode("Transform")
        transform.translate.value = {"x": 3.0, "y": -2.0}
        transform.rotate.value = 25.0
        transform.scale.value = {"x": 1.5, "y": 0.75}
        transform.skewX.value = 10.0
        driven = graph.addNewNode("Transform")
        graph.addEdge(constant.output, transform.input)
        graph.addEdge(constant.output, driven.input)
        graph.update()
        matrix = data(transform, "matrix")
        plan = buildPlan(driven)
        baked = driven.nodeDesc.bakeStream(RenderContext(plan.targetNode, 1, Evaluator(), plan), "inputMatrix", {1: matrix})
        for name, value in baked.items():
            driven.attribute(name).value = value
        graph.update()
        np.testing.assert_allclose(data(driven, "matrix"), matrix, atol=1e-9)


class TestScene3D:

    def test_eulerRoundTrip(self):
        rng = np.random.default_rng(0)
        for order in scene3d.ROTATION_ORDERS:
            for _ in range(20):
                rotation = rng.uniform(-170, 170, 3)
                matrix = scene3d.eulerMatrix(rotation, order)
                np.testing.assert_allclose(scene3d.eulerMatrix(scene3d.matrixToEuler(matrix, order), order), matrix, atol=1e-9)

    def test_projectionAndRay(self):
        camera = scene3d.CameraData(scene3d.lookAt((3, 2, 8), (0, 0, 0)), 35, 36, 24)
        xy, depth, inFront = camera.project(np.array([[0.0, 0.0, 0.0]]), 1920, 1080)
        np.testing.assert_allclose(xy[0], [960, 540], atol=1e-6)
        assert inFront[0] and depth[0] == pytest.approx(np.sqrt(77))
        origin, direction = camera.ray(100.0, 200.0, 1920, 1080)
        back, _, _ = camera.project((origin + 5 * direction)[np.newaxis], 1920, 1080)
        np.testing.assert_allclose(back[0], [100.0, 200.0], atol=1e-6)

    def test_readObjAndPly(self, tmp_path):
        geo = scene3d.readSceneFile(writeCube(tmp_path / "cube.obj")).geo(1)
        assert len(geo.vertices) == 8 and len(geo.faceCounts) == 6
        assert len(geo.triangles()) == 12 and len(geo.edges()) == 12
        ply = tmp_path / "cube.ply"
        header = ("ply\nformat binary_little_endian 1.0\nelement vertex 8\nproperty float x\nproperty float y\n"
                  "property float z\nelement face 6\nproperty list uchar int vertex_indices\nend_header\n")
        body = _CUBE.astype("<f4").tobytes() + b"".join(np.uint8(4).tobytes() + np.array(face, "<i4").tobytes()
                                                        for face in _CUBE_FACES)
        ply.write_bytes(header.encode() + body)
        geoPly = scene3d.readSceneFile(str(ply)).geo(1)
        np.testing.assert_allclose(geoPly.vertices, _CUBE)
        assert geoPly.faceIndices.tolist() == geo.faceIndices.tolist()

    def test_readUsdaCameraAndMesh(self, tmp_path):
        from meshroom.live import camera as cameraModule
        solve = {"width": 1920, "height": 1080, "cameras": {
            "1": {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 5.0]},
            "3": {"R": np.eye(3).tolist(), "t": [2.0, 0.0, 5.0]}}}
        path = str(tmp_path / "camera.usda")
        cameraModule.exportCameraUsd(solve, path, 50.0, 36.0)
        scene = scene3d.readSceneFile(path)
        assert list(scene.cameras) == ["/Camera"]
        cameraAt2 = scene.camera(2)
        assert cameraAt2.focalLength == 50.0 and cameraAt2.horizontalAperture == 36.0
        # The camera of the CV solve at t = (1, 0, 5): its center at (-1, 0, -5), USD world (y, z flipped): (-1, 0, 5)
        np.testing.assert_allclose(cameraAt2.position, [-1.0, 0.0, 5.0], atol=1e-6)
        mesh = tmp_path / "mesh.usda"
        mesh.write_text('''#usda 1.0
(
    upAxis = "Y"
)
def Xform "root" (
    kind = "component"
)
{
    double3 xformOp:translate.timeSamples = {
        1: (0, 0, 0),
        11: (10, 0, 0),
    }
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Mesh "quad"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)] (
            interpolation = "vertex"
        )
        float3 xformOp:scale = (2, 2, 2)
        uniform token[] xformOpOrder = ["xformOp:scale"]
        rel material:binding = </root/material>
    }
}
''')
        geo = scene3d.readSceneFile(str(mesh)).geo(6)
        np.testing.assert_allclose(geo.vertices[2], [7.0, 2.0, 0.0])
        assert geo.faceCounts.tolist() == [4]

    def test_unsupportedFormatsSayWhy(self, tmp_path):
        path = tmp_path / "scene.abc"
        path.write_bytes(b"not an alembic")
        if not scene3d.availableFormats()["abc"]:
            with pytest.raises(RuntimeError, match="Alembic"):
                scene3d.readSceneFile(str(path))


class TestNodes3D:

    def test_cameraFromParameters(self):
        graph = Graph("camera")
        camera = graph.addNewNode("Camera")
        camera.translate.value = {"x": 1.0, "y": 2.0, "z": 3.0}
        camera.rotate.value = {"x": 0.0, "y": 90.0, "z": 0.0}
        graph.update()
        result = data(camera, "output")
        np.testing.assert_allclose(result.position, [1, 2, 3])
        # Turned 90 degrees around Y: looks along -X
        np.testing.assert_allclose(result.matrix[:3, :3] @ [0, 0, -1], [-1, 0, 0], atol=1e-9)

    def test_cameraFollowsTheMatrixOfAnother(self):
        graph = Graph("follow")
        leader = graph.addNewNode("Camera")
        leader.translate.value = {"x": 4.0, "y": 0.0, "z": 0.0}
        follower = graph.addNewNode("Camera")
        follower.focalLength.value = 85.0
        graph.addEdge(leader.matrix, follower.inputMatrix)
        graph.update()
        result = data(follower, "output")
        np.testing.assert_allclose(result.position, [4, 0, 0])
        assert result.focalLength == 85.0

    def test_cameraBakedFromAnInputCamera(self):
        graph = Graph("bakeCamera")
        source = graph.addNewNode("Camera")
        source.rotationOrder.value = "XYZ"
        source.translate.value = {"x": 1.0, "y": 2.0, "z": 3.0}
        source.rotate.value = {"x": 10.0, "y": 20.0, "z": 30.0}
        source.focalLength.value = 50.0
        camera = graph.addNewNode("Camera")
        graph.addEdge(source.output, camera.input)
        graph.update()
        plan = buildPlan(camera)
        evaluator = Evaluator()
        values = {frame: RenderContext(plan.targetNode, frame, evaluator, plan).inputData("input") for frame in (1, 2)}
        baked = camera.nodeDesc.bakeStream(RenderContext(plan.targetNode, 1, evaluator, plan), "input", values)
        graph.removeEdge(camera.input)
        for name, value in baked.items():
            camera.attribute(name).value = value
        graph.update()
        result = data(camera, "output")
        np.testing.assert_allclose(result.matrix, values[1].matrix, atol=1e-9)
        assert result.focalLength == 50.0

    def test_readGeoTransformAndMerge(self, tmp_path):
        graph = Graph("geo")
        first = graph.addNewNode("ReadGeo")
        first.file.value = writeCube(tmp_path / "cube.obj")
        second = graph.addNewNode("ReadGeo")
        second.file.value = first.file.value
        second.translate.value = {"x": 10.0, "y": 0.0, "z": 0.0}
        merge = graph.addNewNode("MergeGeo")
        graph.addEdge(first.output, merge.input1)
        graph.addEdge(second.output, merge.input3)
        graph.update()
        geo = data(merge, "output")
        assert len(geo.vertices) == 16 and len(geo.faceCounts) == 12
        assert geo.vertices[8:, 0].min() == pytest.approx(9.0)
        assert geo.faceIndices.max() == 15

    def test_readGeoMissingFileFails(self):
        graph = Graph("missing")
        readGeo = graph.addNewNode("ReadGeo")
        readGeo.file.value = "/nonexistent/geo.obj"
        graph.update()
        with pytest.raises(EvaluationError, match="not found"):
            data(readGeo, "output")


class TestResection:

    def test_withoutPinsTheInputCamera(self, tmp_path):
        graph, plate, readGeo, camera, resection = resectionGraph(tmp_path)
        np.testing.assert_allclose(data(resection, "cameraOutput").position, [0, 0, 8])
        np.testing.assert_allclose(data(resection, "geoMatrix"), np.eye(4), atol=1e-9)

    def test_solveFromPins(self, tmp_path):
        graph, plate, readGeo, camera, resection = resectionGraph(tmp_path)
        truth = scene3d.CameraData(scene3d.lookAt((2, 1, 7), (0, 0, 0)), 35, 36, 24)
        resection.tracks.value = pinsFor({1: truth}, range(6))
        graph.update()
        evaluator = Evaluator()
        solved = data(resection, "cameraOutput", evaluator=evaluator)
        np.testing.assert_allclose(solved.matrix, truth.matrix, atol=1e-4)
        np.testing.assert_allclose(data(resection, "cameraMatrix", evaluator=evaluator), truth.matrix, atol=1e-4)
        # The geometry moved by the geo matrix, seen from the input camera, is where the plate shows it
        geoMatrix = data(resection, "geoMatrix", evaluator=evaluator)
        inputCamera = data(camera, "output")
        moved, _, _ = inputCamera.project(scene3d.transformPoints(geoMatrix, _CUBE), 640, 360)
        expected, _, _ = truth.project(_CUBE, 640, 360)
        np.testing.assert_allclose(moved, expected, atol=1e-2)
        # The plate with the wireframe drawn over it
        image = Evaluator().render(buildPlan(resection), 1)
        assert image.format == (640, 360) and image.rgba()[:, :, 1].max() > 0.5

    def test_onePinMovesTheGeometryUnderIt(self, tmp_path):
        graph, plate, readGeo, camera, resection = resectionGraph(tmp_path)
        resection.tracks.value = json.dumps([{"name": "pin1", "point": [1.0, 1.0, 1.0], "keys": {"1": [400.0, 250.0]},
                                              "keyframes": [1], "format": [640, 360]}])
        graph.update()
        solved = data(resection, "cameraOutput")
        xy, _, _ = solved.project(np.array([[1.0, 1.0, 1.0]]), 640, 360)
        np.testing.assert_allclose(xy[0], [400.0, 250.0], atol=0.5)
        # Mostly a move parallel to the image: the distance to the geometry barely changes
        assert np.linalg.norm(solved.position) == pytest.approx(8.0, rel=0.05)

    def test_trackedPinsSolveEachFrame(self, tmp_path):
        graph, plate, readGeo, camera, resection = resectionGraph(tmp_path)
        truths = {frame: scene3d.CameraData(scene3d.lookAt((0.3 * frame, 1, 7 - 0.1 * frame), (0, 0, 0)), 35, 36, 24)
                  for frame in range(1, 11)}
        resection.tracks.value = pinsFor(truths, range(8), keyframes=[1])
        graph.update()
        evaluator = Evaluator()
        for frame in (10, 5):
            solved = data(resection, "cameraOutput", frame, evaluator)
            np.testing.assert_allclose(solved.position, truths[frame].position, atol=1e-3)

    def test_pickPointOnTheGeometry(self, tmp_path):
        graph, plate, readGeo, camera, resection = resectionGraph(tmp_path)
        plan = buildPlan(resection)
        ctx = RenderContext(plan.targetNode, 1, Evaluator(), plan)
        point = resection.nodeDesc.pickPoint(ctx, 320.0, 180.0)
        np.testing.assert_allclose(point, [0.0, 0.0, 1.0], atol=1e-9)
        assert resection.nodeDesc.pickPoint(ctx, 5.0, 5.0) is None

    def test_cameraPortThroughASubGraph(self, tmp_path):
        from meshroom.core.subgraph.editing import collapseIntoSubGraph
        graph, plate, readGeo, camera, resection = resectionGraph(tmp_path)
        camera.translate.value = {"x": 0.0, "y": 0.0, "z": 12.0}
        graph.update()
        subGraphNode, _ = collapseIntoSubGraph(graph, [camera])
        graph.update()
        assert resection.camera.inputRootLink.node is subGraphNode
        np.testing.assert_allclose(data(resection, "cameraOutput").position, [0, 0, 12])


def test_compositingTemplateHasACompositingSubGraph():
    path = os.path.join(os.path.dirname(meshroom.__file__), "nodes", "compositing", "compositing.mgt")
    graph = loadGraph(path)
    assert graph.workspaceMode == "compositing"
    subGraphs = [node for node in graph.nodes if node.nodeType == "SubGraph"]
    assert [node.label for node in subGraphs] == ["compositing_1"]
    write = next(node for node in graph.nodes if node.nodeType == "Write")
    assert write.image.inputRootLink.node is subGraphs[0]


class TestBake:

    def test_bakeAnimatedParameter(self):
        from meshroom.live import animation as anim, bake
        graph = Graph("bakeParameter")
        source = graph.addNewNode("Transform")
        target = graph.addNewNode("Transform")
        source.internalAttribute("animation").value = anim.dumpAnimation(
            {"rotate": [[1.0, 0.0], [10.0, 90.0]], "translate": [[1.0, {"x": 0.0, "y": 5.0}], [10.0, {"x": 9.0, "y": 5.0}]]})
        graph.addEdge(source.rotate, target.rotate)
        graph.addEdge(source.translate.value.get("x"), target.translate.value.get("x"))
        graph.update()
        assert bake.canBake(target.rotate)
        baked = bake.bakedParameter(target.rotate)
        assert baked == {"rotate": [[1.0, 0.0], [10.0, 90.0]]}
        bakedX = bake.bakedParameter(target.translate.value.get("x"))
        assert [key[1]["x"] for key in bakedX["translate"]] == [0.0, 9.0]
        # The other child keeps the value of the node
        assert all(key[1]["y"] == 0.0 for key in bakedX["translate"])
        changes, animation = bake.applyToNode(target, baked, 4)
        assert changes[0][1] == pytest.approx(30.0)
        assert anim.parseAnimation(animation)["rotate"] == [[1.0, 0.0], [10.0, 90.0]]

    def test_bakeStaticParameter(self):
        from meshroom.live import bake
        graph = Graph("bakeStatic")
        source = graph.addNewNode("Grade")
        target = graph.addNewNode("Grade")
        source.mix.value = 0.25
        graph.addEdge(source.mix, target.mix)
        graph.update()
        assert bake.bakedParameter(target.mix) == {"mix": 0.25}

    def test_imageEdgesAreNotBaked(self):
        from meshroom.live import bake
        graph = Graph("bakeImage")
        constant = graph.addNewNode("Constant")
        grade = graph.addNewNode("Grade")
        resection = graph.addNewNode("Resection")
        readGeo = graph.addNewNode("ReadGeo")
        graph.addEdge(constant.output, grade.input)
        graph.addEdge(readGeo.output, resection.geo)
        assert not bake.canBake(grade.input)
        assert not bake.canBake(resection.geo)

    def test_bakeCameraFollowingAResection(self, tmp_path):
        from meshroom.live import animation as anim, bake
        graph, plate, readGeo, camera, resection = resectionGraph(tmp_path)
        truths = {frame: scene3d.CameraData(scene3d.lookAt((0.3 * frame, 1, 7), (0, 0, 0)), 35, 36, 24)
                  for frame in range(1, 6)}
        resection.tracks.value = pinsFor(truths, range(8), keyframes=[1])
        follower = graph.addNewNode("Camera")
        graph.addEdge(resection.cameraOutput, follower.input)
        graph.update()
        assert bake.canBake(follower.input)
        baked = bake.bakedValues(follower.input, range(1, 6), 1)
        changes, animation = bake.applyToNode(follower, baked, 1)
        graph.removeEdge(follower.input)
        for attribute, value in changes:
            attribute.value = value
        follower.internalAttribute("animation").value = animation
        graph.update()
        assert follower.source.value == "parameters"
        assert "translate" in anim.parseAnimation(animation)
        for frame in (1, 3, 5):
            np.testing.assert_allclose(data(follower, "output", frame).position, truths[frame].position, atol=1e-3)


def test_compositingSubGraphsOfAGraph():
    from meshroom.core.subgraph.editing import SubGraphStack, compositingSubGraphs, newCompositingLabel
    graph = Graph("workspace")
    assert compositingSubGraphs(graph) == [] and newCompositingLabel(graph) == "compositing_1"
    other = graph.addNewNode("SubGraph")
    assert compositingSubGraphs(graph) == []
    labelled = graph.addNewNode("SubGraph")
    labelled.internalAttribute("label").value = "compositing_1"
    assert compositingSubGraphs(graph) == [labelled] and newCompositingLabel(graph) == "compositing_2"
    # A SubGraph holding compositing nodes, whatever its label
    inner = SubGraphStack()
    content = inner.enter(graph, other)
    content.addNewNode("Grade")
    inner.exit(content)
    assert set(compositingSubGraphs(graph)) == {labelled, other}


def test_writeOutsideTheCompositingSubGraph(tmp_path):
    """ The Write of the compositing template renders the image given by the compositing_1 subgraph. """
    from meshroom.core.subgraph.editing import collapseIntoSubGraph
    from meshroom.live import imageIO
    graph = Graph("write")
    graph.saveAsTemp(str(tmp_path))
    constant = graph.addNewNode("Constant", width=4, height=2)
    constant.rgba.value.get("r").value = 0.25
    grade = graph.addNewNode("Grade", multiply={"r": 2.0, "g": 2.0, "b": 2.0, "a": 2.0})
    write = graph.addNewNode("Write", file=str(tmp_path / "out" / "render.####.tif"), dataType="float",
                             colorspace="linear")
    graph.addEdge(constant.output, grade.input)
    graph.addEdge(grade.output, write.image)
    collapseIntoSubGraph(graph, [constant, grade])
    graph.update()
    os.makedirs(write.internalFolder, exist_ok=True)
    write.createChunks()
    write.preprocess()
    for chunk in write.chunks:
        write.prepareLogger(chunk.range.iteration)
        chunk.process(True, True)
        write.restoreLogger()
    write.postprocess()
    image = imageIO.readImage(str(tmp_path / "out" / "render.0001.tif"), "linear")
    np.testing.assert_allclose(image.rgba()[0, 0, 0], 0.5, rtol=1e-6)


def test_keysInASubGraphSurviveWritingItBack():
    """ Writing a subgraph back (e.g. saving inside it) keeps the cache keys of its nodes. """
    from meshroom.core.subgraph.editing import SubGraphStack, collapseIntoSubGraph
    graph = Graph("keys")
    constant = graph.addNewNode("Constant", width=8, height=4)
    grade = graph.addNewNode("Grade")
    graph.addEdge(constant.output, grade.input)
    subGraphNode, _ = collapseIntoSubGraph(graph, [constant, grade])
    stack = SubGraphStack()
    inner = stack.enter(graph, subGraphNode)
    innerGrade = next(node for node in inner.nodes if node.nodeType == "Grade")
    before = buildPlan(innerGrade).target
    uidBefore = subGraphNode._uid
    innerGrade.position = type(innerGrade.position)(500, 500)
    stack.commitAll(inner)
    graph.update()
    assert subGraphNode._uid != uidBefore
    assert buildPlan(innerGrade).target == before


def test_bakeKeepsWhatAnInvertedTransformApplied():
    graph = Graph("bakeInvert")
    constant = graph.addNewNode("Constant", width=64, height=36)
    transform = graph.addNewNode("Transform", rotate=20.0, translate={"x": 4.0, "y": 1.0})
    driven = graph.addNewNode("Transform", invert=True)
    graph.addEdge(constant.output, transform.input)
    graph.addEdge(constant.output, driven.input)
    graph.addEdge(transform.matrix, driven.inputMatrix)
    graph.update()
    applied = data(driven, "matrix")
    plan = buildPlan(driven)
    baked = driven.nodeDesc.bakeStream(RenderContext(plan.targetNode, 1, Evaluator(), plan), "inputMatrix",
                                       {1: data(transform, "matrix")})
    graph.removeEdge(driven.inputMatrix)
    for name, value in baked.items():
        driven.attribute(name).value = value
    graph.update()
    np.testing.assert_allclose(data(driven, "matrix"), applied, atol=1e-9)


def test_streamSubclassOfAPlugin():
    from meshroom.core import attribute

    class Mask(desc.Image):
        pass
    assert Mask(name="mask").getInstanceType() is attribute.Image
    assert desc.Camera(name="camera").getInstanceType() is attribute.Camera


def test_compositingSubGraphsInNaturalOrder():
    from meshroom.core.subgraph.editing import compositingSubGraphs
    graph = Graph("order")
    for label in ("compositing_10", "compositing_2"):
        graph.addNewNode("SubGraph").internalAttribute("label").value = label
    assert [node.label for node in compositingSubGraphs(graph)] == ["compositing_2", "compositing_10"]


def test_resectionCenterGeometry(tmp_path):
    """ "Center Geometry": the starting camera steps back along its view axis until the geometry is centered. """
    graph, plate, readGeo, camera, resection = resectionGraph(tmp_path)
    # A camera inside the cube, looking along -Z
    camera.translate.value = {"x": 0.2, "y": 0.1, "z": 0.5}
    resection.centerGeometry.value = True
    graph.update()
    solved = data(resection, "cameraOutput")
    np.testing.assert_allclose(solved.matrix[:3, :3], np.eye(3), atol=1e-9)
    xy, _, inFront = solved.project(_CUBE, 640, 360)
    assert inFront.all()
    assert (xy[:, 0] > 0).all() and (xy[:, 0] < 640).all() and (xy[:, 1] > 0).all() and (xy[:, 1] < 360).all()
    center, _, _ = solved.project(np.zeros((1, 3)), 640, 360)
    np.testing.assert_allclose(center[0], [320, 180], atol=1e-6)


def test_objectNamesMatchTheirTransform():
    from meshroom.live.scene3d import _namesMatch
    assert _namesMatch("/cam_022b_040_base/cam_022b_040_baseShape", "cam_022b_040_base")
    assert _namesMatch("/cam_022b_040_base/cam_022b_040_baseShape", "cam_022b_040_baseShape")
    assert _namesMatch("/root/cam_A/cam_AShape", "/root/cam_A")
    assert _namesMatch("/root/cam_A/cam_AShape", "")
    assert not _namesMatch("/root/cam_A/cam_AShape", "cam")
    assert not _namesMatch("/root/cam_A/cam_AShape", "root")
