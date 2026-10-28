"""Tests for the live compositing engine (meshroom.live) and the compositing nodes."""

import os
import threading

import numpy as np
import pytest

import meshroom
from meshroom.core import desc, pluginManager
from meshroom.core.graph import Graph, loadGraph
from meshroom.live import (
    BBox, CancelToken, EvalStatus, EvaluationCancelled, EvaluationError, Evaluator, ImageBuffer, ImageCache,
    ImageInfo, buildPlan, imageIO, ops,
)

_COMPOSITING_FOLDER = os.path.join(os.path.dirname(meshroom.__file__), "nodes", "compositing")
if not pluginManager.isNodeDescRegistered("Grade"):
    pluginManager.addPluginFromBuiltInFolder("compositing", _COMPOSITING_FOLDER)


def setColor(constantNode, r=0.0, g=0.0, b=0.0, a=1.0):
    for channel, value in zip("rgba", (r, g, b, a)):
        constantNode.rgba.value.get(channel).value = value


def constantGraph():
    """ Constant(red) -> Grade -> Merge.A ; Constant(blue) -> Merge.B """
    graph = Graph("live")
    red = graph.addNewNode("Constant", width=8, height=4)
    blue = graph.addNewNode("Constant", width=8, height=4)
    grade = graph.addNewNode("Grade")
    merge = graph.addNewNode("Merge")
    setColor(red, r=0.5, a=1.0)
    setColor(blue, b=1.0, a=1.0)
    graph.addEdge(red.output, grade.input)
    graph.addEdge(grade.output, merge.A)
    graph.addEdge(blue.output, merge.B)
    graph.update()
    return graph, red, blue, grade, merge


class TestImageAttribute:

    def test_imageOpNodeHasNoFlowPins(self):
        graph, red, _, grade, _ = constantGraph()
        names = [attr.name for attr in grade.internalAttributes]
        assert "flowInputs" not in names
        assert "flowOutput" not in names

    def test_imageOpNodeIsNotComputableInBatch(self):
        graph, red, _, grade, _ = constantGraph()
        assert grade.isImageOpNode
        assert not grade.isComputableType
        assert grade.isComputed  # never blocks a batch node downstream

    def test_uidPropagatesThroughImageLinks(self):
        graph, red, _, grade, merge = constantGraph()
        before = (grade._uid, merge._uid)
        setColor(red, r=0.25)
        graph.update()
        assert grade._uid != before[0]
        assert merge._uid != before[1]

    def test_imageConnectsOnlyToImage(self):
        graph, red, _, grade, merge = constantGraph()
        assert grade.input.validateIncomingConnection(red.output)
        assert not grade.input.validateIncomingConnection(red.width)

    def test_saveAndLoad(self, tmp_path):
        graph, red, blue, grade, merge = constantGraph()
        graph.workspaceMode = "compositing"
        filepath = str(tmp_path / "comp.mg")
        graph.save(filepath)
        loaded = loadGraph(filepath)
        assert loaded.workspaceMode == "compositing"
        assert loaded.node(merge.name).A.inputRootLink.node.name == grade.name
        assert loaded.node(merge.name)._uid == merge._uid


class TestImageBuffer:

    def test_layersAndRGBA(self):
        image = ImageBuffer.fromRGBA(np.ones((2, 3, 3), dtype=np.float32))
        image = image.withLayer("depth", np.full((2, 3, 1), 5.0), ("Z",))
        assert image.layerNames == ["rgba", "depth"]
        rgba = image.rgba()
        assert rgba.shape == (2, 3, 4)
        assert rgba[0, 0, 3] == 0.0  # missing alpha is 0
        assert image.layer("depth")[0, 0, 0] == 5.0

    def test_buffersAreReadOnly(self):
        pixels = np.zeros((2, 2, 4), dtype=np.float32)
        image = ImageBuffer.fromRGBA(pixels)
        with pytest.raises(ValueError):
            image.layer("rgba")[0, 0, 0] = 1.0

    def test_splitChannelNames(self):
        groups = imageIO.splitChannelNames(["R", "G", "B", "A", "Z", "diffuse.R", "diffuse.G", "diffuse.B"])
        assert list(groups.keys()) == ["rgba", "depth", "diffuse"]
        assert groups["diffuse"] == [(5, "R"), (6, "G"), (7, "B")]


class TestImageCache:

    def test_lruEvictionInBytes(self):
        def image():
            return ImageBuffer.constant(4, 4, (1, 1, 1, 1))  # 256 bytes
        size = image().nbytes
        cache = ImageCache(maxBytes=size * 2)
        cache.put(("a", 1), image())
        cache.put(("b", 1), image())
        assert cache.get(("a", 1)) is not None  # "a" becomes the most recently used
        cache.put(("c", 1), image())
        assert ("b", 1) not in cache
        assert ("a", 1) in cache and ("c", 1) in cache
        assert cache.usedBytes == size * 2
        assert cache.cachedFrames("a") == [1]

    def test_sharedArraysCountedOnce(self):
        image = ImageBuffer.constant(4, 4, (1, 1, 1, 1))
        image = image.withLayer("depth", np.zeros((4, 4, 1)), ("Z",))
        regraded = image.withRGBA(np.ones((4, 4, 4), dtype=np.float32))  # shares the depth layer
        cache = ImageCache(maxBytes=10 ** 6)
        cache.put(("a", 1), image)
        cache.put(("passThrough", 1), image)
        cache.put(("b", 1), regraded)
        assert cache.usedBytes == image.nbytes + 4 * 4 * 4 * 4
        cache.discardUids(["a", "passThrough"])
        assert cache.usedBytes == regraded.nbytes


class TestEvaluator:

    def test_renderBranchAndCache(self):
        graph, red, blue, grade, merge = constantGraph()
        evaluator = Evaluator()
        image = evaluator.render(buildPlan(merge), 1)
        assert image.width == 8 and image.height == 4
        np.testing.assert_allclose(image.rgba()[0, 0], [0.5, 0, 0, 1])
        for node in (red, blue, grade, merge):
            assert evaluator.status(node._uid, 1)[0] == EvalStatus.SUCCESS

        # Changing the grade re-renders grade and merge only
        grade.gain.value = {"r": 2.0, "g": 2.0, "b": 2.0, "a": 2.0}
        graph.update()
        rendered = []
        evaluator.listener = lambda uid, frame, status, message: status == EvalStatus.RUNNING and rendered.append(uid)
        image = evaluator.render(buildPlan(merge), 1)
        np.testing.assert_allclose(image.rgba()[0, 0], [1.0, 0, 0, 1])
        assert set(rendered) == {grade._uid, merge._uid}

    def test_onlyTheEvaluatedBranchIsComputed(self):
        graph, red, blue, grade, merge = constantGraph()
        evaluator = Evaluator()
        evaluator.render(buildPlan(grade), 1)
        assert evaluator.status(grade._uid, 1)[0] == EvalStatus.SUCCESS
        assert evaluator.status(blue._uid, 1)[0] == EvalStatus.NONE
        assert evaluator.status(merge._uid, 1)[0] == EvalStatus.NONE

    def test_cachedResultMarksItsBranchSucceeded(self):
        graph, red, blue, grade, merge = constantGraph()
        evaluator = Evaluator()
        plan = buildPlan(merge)
        evaluator.render(plan, 1)
        evaluator.clearStatuses()
        evaluator.render(plan, 1)  # served from the cache
        for node in (red, blue, grade, merge):
            assert evaluator.status(node._uid, 1)[0] == EvalStatus.SUCCESS
        evaluator.cache.discardUids([merge._uid])
        assert (merge._uid, 1) not in evaluator.cache
        assert (grade._uid, 1) in evaluator.cache

    def test_errorAndUpstreamError(self):
        graph = Graph("live")
        read = graph.addNewNode("Read", file="/does/not/exist.exr")
        grade = graph.addNewNode("Grade")
        graph.addEdge(read.output, grade.input)
        graph.update()
        evaluator = Evaluator()
        with pytest.raises(EvaluationError) as error:
            evaluator.render(buildPlan(grade), 1)
        assert error.value.nodeName == read.name
        assert evaluator.status(read._uid, 1)[0] == EvalStatus.ERROR
        assert evaluator.status(grade._uid, 1)[0] == EvalStatus.UPSTREAM_ERROR

    def test_disconnectedRequiredInput(self):
        graph = Graph("live")
        grade = graph.addNewNode("Grade")
        graph.update()
        evaluator = Evaluator()
        with pytest.raises(EvaluationError):
            evaluator.render(buildPlan(grade), 1)
        status, message = evaluator.status(grade._uid, 1)
        assert status == EvalStatus.ERROR
        assert "input" in message

    def test_cancel(self):
        graph, red, blue, grade, merge = constantGraph()
        token = CancelToken()
        token.cancel()
        with pytest.raises(EvaluationCancelled):
            Evaluator().render(buildPlan(merge), 1, token)

    def test_renderInWorkerThread(self):
        graph, red, blue, grade, merge = constantGraph()
        plan = buildPlan(merge)  # snapshot in the graph thread
        result = {}
        thread = threading.Thread(target=lambda: result.update(image=Evaluator().render(plan, 1)))
        thread.start()
        thread.join()
        np.testing.assert_allclose(result["image"].rgba()[0, 0], [0.5, 0, 0, 1])

    def test_batchDependencyNotComputed(self):
        class FileProducer(desc.Node):
            inputs = []
            outputs = [desc.File(name="output", label="Output", description="", value="{nodeCacheFolder}/img.exr")]

        from .utils import registeredNodeTypes
        with registeredNodeTypes([FileProducer]):
            graph = Graph("live")
            producer = graph.addNewNode("FileProducer")
            read = graph.addNewNode("Read")
            graph.addEdge(producer.output, read.file)
            graph.update()
            evaluator = Evaluator()
            with pytest.raises(EvaluationError) as error:
                evaluator.render(buildPlan(read), 1)
            assert "not computed" in error.value.message


class TestOps:

    def test_gradeFormula(self):
        pixels = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        out = ops.grade(pixels, blackpoint=0.0, whitepoint=1.0, lift=0.1, gain=0.9)
        np.testing.assert_allclose(out[0, 0], [0.1, 0.5, 0.9], rtol=1e-6)
        out = ops.grade(pixels, gamma=2.0)
        np.testing.assert_allclose(out[0, 0], [0.0, np.sqrt(0.5), 1.0], rtol=1e-6)

    def test_mergeOver(self):
        a = np.array([[[0.5, 0, 0, 0.5]]], dtype=np.float32)
        b = np.array([[[0, 0, 1, 1]]], dtype=np.float32)
        np.testing.assert_allclose(ops.merge(a, b, "over")[0, 0], [0.5, 0, 0.5, 1.0])
        np.testing.assert_allclose(ops.merge(a, b, "under")[0, 0], [0, 0, 1, 1])
        np.testing.assert_allclose(ops.merge(a, b, "mask")[0, 0], [0, 0, 0.5, 0.5])

    def test_translateFollowsNukeConvention(self):
        bbox = BBox(0, 0, 4, 4)
        pixels = np.zeros((4, 4, 1), dtype=np.float32)
        pixels[3, 0, 0] = 1.0  # bottom-left pixel
        matrix = ops.transformMatrix(translate=(1.25, 1.25))
        out = ops.warpAffine(pixels, bbox, matrix, bbox, "impulse")
        assert out[2, 1, 0] == 1.0  # moved one pixel right and up (impulse: nearest pixel)
        assert out.sum() == 1.0
        assert ops.integerTranslation(ops.transformMatrix(translate=(3, -2))) == (3, -2)
        assert ops.integerTranslation(matrix) is None

    def test_blurPreservesEnergyAndGrowsBBox(self):
        for size in (2.0, 20.0):  # small kernel and box blurs
            pixels = np.zeros((21, 21, 1), dtype=np.float32)
            pixels[10, 10, 0] = 1.0
            out = ops.blur(pixels, size, size)
            radius = ops.blurRadius(size)
            assert out.shape == (21 + 2 * radius, 21 + 2 * radius, 1)
            assert out[10 + radius, 10 + radius, 0] < 1.0
            assert abs(out.sum() - 1.0) < 1e-4
            # Symmetric
            np.testing.assert_allclose(out, out[::-1, ::-1], atol=1e-6)

    def test_boxBlurApproximatesGaussian(self):
        sigma = 10.0
        radii = ops.boxRadii(sigma)
        variance = sum(((2 * r + 1) ** 2 - 1) / 12.0 for r in radii)
        assert abs(variance ** 0.5 - sigma) < 1.0

    def test_alignToBBox(self):
        pixels = np.arange(4, dtype=np.float32).reshape(2, 2, 1)  # rows: top [0, 1], bottom [2, 3]
        out = ops.alignToBBox(pixels, BBox(0, 0, 2, 2), BBox(-1, -1, 3, 3))
        assert out.shape == (4, 4, 1)
        assert out[2, 1, 0] == 2  # bottom-left pixel at (0, 0)
        assert out.sum() == pixels.sum()
        crop = ops.alignToBBox(pixels, BBox(0, 0, 2, 2), BBox(1, 0, 2, 1))
        assert crop[0, 0, 0] == 3  # bottom-right pixel
        assert ops.fitToFormat(pixels, 3, 3)[2, 0, 0] == 2

    def test_integerLutMatchesFormula(self):
        raw = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(2, axis=2)
        out = imageIO.integerToFloat(raw, [True, False])
        np.testing.assert_allclose(out[:, :, 0].ravel(), imageIO.srgbToLinear(np.arange(256) / 255.0), rtol=1e-6)
        np.testing.assert_allclose(out[:, :, 1].ravel(), np.arange(256) / 255.0, rtol=1e-6)


class TestReadWrite:

    def _writeSequence(self, folder, frames):
        for frame in frames:
            image = ImageBuffer.constant(4, 2, (frame / 10.0, 0, 0, 1))
            imageIO.writeImage(imageIO.framePath(str(folder / "img.####.tif"), frame), image, "float", "linear")

    def test_sequencePaths(self, tmp_path):
        assert imageIO.framePath("/a/img.####.exr", 12) == os.path.join("/a", "img.0012.exr")
        assert imageIO.framePath("/a/img.%03d.exr", 7) == os.path.join("/a", "img.007.exr")
        self._writeSequence(tmp_path, [3, 4, 5])
        assert imageIO.detectFrameRange(str(tmp_path / "img.####.tif")) == (3, 5)

    def test_readSequenceHoldsOutsideRange(self, tmp_path):
        self._writeSequence(tmp_path, [3, 4, 5])
        graph = Graph("live")
        read = graph.addNewNode("Read", file=str(tmp_path / "img.####.tif"), colorspace="linear")
        graph.update()
        evaluator = Evaluator()
        plan = buildPlan(read)
        assert evaluator.info(plan) == ImageInfo(4, 2, 3, 5)
        np.testing.assert_allclose(evaluator.render(plan, 4).rgba()[0, 0, 0], 0.4, rtol=1e-6)
        np.testing.assert_allclose(evaluator.render(plan, 10).rgba()[0, 0, 0], 0.5, rtol=1e-6)

    def test_writeNodeRendersFrames(self, tmp_path):
        self._writeSequence(tmp_path, [1, 2])
        graph = Graph("live")
        graph.saveAsTemp(str(tmp_path))
        read = graph.addNewNode("Read", file=str(tmp_path / "img.####.tif"), colorspace="linear")
        grade = graph.addNewNode("Grade", multiply={"r": 2.0, "g": 2.0, "b": 2.0, "a": 2.0})
        write = graph.addNewNode("Write", file=str(tmp_path / "out" / "render.####.tif"), dataType="float",
                                 colorspace="linear")
        graph.addEdge(read.output, grade.input)
        graph.addEdge(grade.output, write.image)
        graph.update()
        assert write.isComputableType
        assert write.evaluateSize() == 2

        os.makedirs(write.internalFolder, exist_ok=True)
        write.createChunks()
        assert write.size == 2
        write.preprocess()
        for chunk in write.chunks:
            write.prepareLogger(chunk.range.iteration)
            chunk.process(True, True)
            write.restoreLogger()
        write.postprocess()

        for frame in (1, 2):
            image = imageIO.readImage(str(tmp_path / "out" / f"render.{frame:04d}.tif"), "linear")
            np.testing.assert_allclose(image.rgba()[0, 0, 0], frame / 10.0 * 2, rtol=1e-6)


def test_compositingTemplate():
    """
    The "compositing" pipeline template is registered and opens in the compositing workspace: the compositing nodes
    are in the "compositing_1" SubGraph node, rendered by the Write outside.
    """
    from meshroom.core.subgraph.editing import SubGraphStack
    if not pluginManager.isNodeDescRegistered("SubGraph"):
        subGraphFolder = os.path.join(os.path.dirname(meshroom.__file__), "nodes", "subgraph")
        pluginManager.addPluginFromBuiltInFolder("subgraph", subGraphFolder)
    templates = pluginManager.getPipelineTemplates()
    assert "compositing" in templates
    graph = Graph("")
    graph.initFromTemplate(templates["compositing"])
    assert graph.workspaceMode == "compositing"
    subGraphNode = graph.node("Write_1").image.inputRootLink.node
    assert subGraphNode.nodeType == "SubGraph" and subGraphNode.label == "compositing_1"
    inner = SubGraphStack().enter(graph, subGraphNode)
    assert {node.nodeType for node in inner.nodes} >= {"Read", "Grade"}


class TestBBoxNodes:

    def _graphWith(self, nodeType, **values):
        graph = Graph("live")
        source = graph.addNewNode("Constant", width=20, height=10)
        setColor(source, r=1.0, a=1.0)
        node = graph.addNewNode(nodeType, **values)
        graph.addEdge(source.output, node.input)
        graph.update()
        return graph, source, node

    def test_integerTranslationOnlyMovesTheBBox(self):
        graph, source, transform = self._graphWith("Transform")
        transform.translate.value.get("x").value = 30.0
        transform.translate.value.get("y").value = -3.0
        graph.update()
        evaluator = Evaluator()
        image = evaluator.render(buildPlan(transform), 1)
        assert image.bbox == BBox(30, -3, 50, 7)
        assert image.layer("rgba") is evaluator.cache.get((source._uid, 1)).layer("rgba")

    def test_rotationGrowsTheBBox(self):
        graph, source, transform = self._graphWith("Transform", rotate=45.0)
        image = Evaluator().render(buildPlan(transform), 1)
        assert image.bbox.width > 20 and image.bbox.height > 10
        assert (image.width, image.height) == (20, 10)  # The format does not change

    def test_blurGrowsTheBBox(self):
        graph, source, blur = self._graphWith("Blur", size={"x": 10.0, "y": 10.0})
        image = Evaluator().render(buildPlan(blur), 1)
        radius = ops.blurRadius(10.0)
        assert image.bbox == BBox(-radius, -radius, 20 + radius, 10 + radius)
        np.testing.assert_allclose(image.layer("rgba")[:, :, 0].sum(), 200.0, rtol=1e-4)

    def test_mergeUnionOfBBoxes(self):
        graph = Graph("live")
        background = graph.addNewNode("Constant", width=20, height=10)
        foreground = graph.addNewNode("Constant", width=20, height=10)
        transform = graph.addNewNode("Transform")
        merge = graph.addNewNode("Merge")
        setColor(background, b=1.0, a=1.0)
        setColor(foreground, r=1.0, a=1.0)
        transform.translate.value.get("x").value = 15.0
        graph.addEdge(foreground.output, transform.input)
        graph.addEdge(background.output, merge.B)
        graph.addEdge(transform.output, merge.A)
        graph.update()
        image = Evaluator().render(buildPlan(merge), 1)
        assert image.bbox == BBox(0, 0, 35, 10)
        rgba = image.layer("rgba")
        np.testing.assert_allclose(rgba[0, 0], [0, 0, 1, 1])  # B only
        np.testing.assert_allclose(rgba[0, 20], [1, 0, 0, 1])  # A over B
        np.testing.assert_allclose(rgba[0, 30], [1, 0, 0, 1])  # A outside of B (overscan)
        merge.bbox.value = "B"
        graph.update()
        assert Evaluator().render(buildPlan(merge), 1).bbox == BBox(0, 0, 20, 10)

    def test_writeCropsToFormatOutsideEXR(self, tmp_path):
        image = ImageBuffer.fromRGBA(np.ones((4, 6, 4), dtype=np.float32), format=(4, 2), bbox=BBox(-1, -1, 5, 3))
        path = str(tmp_path / "crop.tif")
        imageIO.writeImage(path, image, "float", "linear")
        read = imageIO.readImage(path, "linear")
        assert (read.width, read.height) == (4, 2)
        assert read.bbox == BBox(0, 0, 4, 2)


class TestPassThrough:

    def test_neutralNodesPassThrough(self):
        graph = Graph("live")
        source = graph.addNewNode("Constant", width=8, height=4)
        grade = graph.addNewNode("Grade")
        blur = graph.addNewNode("Blur")
        transform = graph.addNewNode("Transform")
        graph.addEdge(source.output, grade.input)
        graph.addEdge(grade.output, blur.input)
        graph.addEdge(blur.output, transform.input)
        graph.update()
        evaluator = Evaluator()
        rendered = []
        evaluator.listener = lambda uid, frame, status, message: status == EvalStatus.RUNNING and rendered.append(uid)
        image = evaluator.render(buildPlan(transform), 1)
        assert image is evaluator.cache.get((source._uid, 1))
        assert rendered == [source._uid]  # Nothing else computed
        assert evaluator.status(blur._uid, 1)[0] == EvalStatus.SUCCESS
        assert evaluator.cache.usedBytes == image.nbytes  # Counted once

    def test_mergeWithoutAPassesBThrough(self):
        graph = Graph("live")
        source = graph.addNewNode("Constant", width=8, height=4)
        merge = graph.addNewNode("Merge")
        graph.addEdge(source.output, merge.B)
        graph.update()
        evaluator = Evaluator()
        assert evaluator.render(buildPlan(merge), 1) is evaluator.cache.get((source._uid, 1))

    def test_disabledNode(self):
        graph, red, blue, grade, merge = constantGraph()
        grade.gain.value = {"r": 3.0, "g": 3.0, "b": 3.0, "a": 3.0}
        grade.internalAttribute("disable").value = True
        graph.update()
        evaluator = Evaluator()
        image = evaluator.render(buildPlan(grade), 1)
        np.testing.assert_allclose(image.rgba()[0, 0], [0.5, 0, 0, 1])
        # A disabled Merge passes B through
        merge.internalAttribute("disable").value = True
        graph.update()
        np.testing.assert_allclose(evaluator.render(buildPlan(merge), 1).rgba()[0, 0], [0, 0, 1, 1])
        # A disabled source outputs transparent black
        red.internalAttribute("disable").value = True
        graph.update()
        np.testing.assert_allclose(evaluator.render(buildPlan(red), 1).rgba()[0, 0], [0, 0, 0, 0])

    def test_disableChangesTheUid(self):
        graph, red, blue, grade, merge = constantGraph()
        before = merge._uid
        grade.internalAttribute("disable").value = True
        graph.update()
        assert merge._uid != before


class TestMask:

    def test_maskChannel(self):
        graph = Graph("live")
        source = graph.addNewNode("Constant", width=4, height=2)
        mask = graph.addNewNode("Constant", width=4, height=2)
        grade = graph.addNewNode("Grade", gain={"r": 2.0, "g": 2.0, "b": 2.0, "a": 2.0}, maskChannel="red")
        setColor(source, r=0.5, g=0.5, b=0.5, a=1.0)
        setColor(mask, r=1.0, a=0.0)
        graph.addEdge(source.output, grade.input)
        graph.addEdge(mask.output, grade.mask)
        graph.update()
        evaluator = Evaluator()
        np.testing.assert_allclose(evaluator.render(buildPlan(grade), 1).rgba()[0, 0, 0], 1.0)  # red mask = 1
        grade.maskChannel.value = "alpha"
        graph.update()
        np.testing.assert_allclose(evaluator.render(buildPlan(grade), 1).rgba()[0, 0, 0], 0.5)  # alpha mask = 0
        grade.invertMask.value = True
        graph.update()
        np.testing.assert_allclose(evaluator.render(buildPlan(grade), 1).rgba()[0, 0, 0], 1.0)


def test_cancelDuringAnOperation():
    graph = Graph("live")
    source = graph.addNewNode("Constant", width=64, height=256)
    blur = graph.addNewNode("Blur", size={"x": 30.0, "y": 30.0})
    graph.addEdge(source.output, blur.input)
    graph.update()
    token = CancelToken()
    evaluator = Evaluator()
    plan = buildPlan(blur)
    evaluator.render(buildPlan(source), 1)  # the source is cached: only the blur remains
    calls = []
    original = ops.parallelBands

    def cancelAfterFirstBand(func, length, cancel=None, minBand=16):
        def cancelling():
            calls.append(1)
            if len(calls) > 1:
                token.cancel()
            if cancel:
                cancel()
        return original(func, length, cancelling, minBand)
    ops.parallelBands = cancelAfterFirstBand
    try:
        with pytest.raises(EvaluationCancelled):
            evaluator.render(plan, 1, token)
    finally:
        ops.parallelBands = original
    assert evaluator.status(blur._uid, 1)[0] == EvalStatus.NONE


def test_listFolderGroupsSequences(tmp_path):
    from meshroom.live.sequences import frameRanges, listFolder
    for frame in list(range(1001, 1011)) + [1013]:
        (tmp_path / f"render.{frame:04d}.exr").write_bytes(b"")
    for frame in (1, 2, 10):
        (tmp_path / f"plate_{frame}.jpg").write_bytes(b"")
    (tmp_path / "still.png").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    (tmp_path / "shots").mkdir()
    entries = listFolder(str(tmp_path))
    names = [(entry["type"], entry["name"]) for entry in entries]
    assert names[0] == ("folder", "shots")
    assert ("sequence", "render.####.exr") in names
    assert ("sequence", "plate_#.jpg") in names
    assert ("file", "still.png") in names
    assert all(name != "notes.txt" for _, name in names)
    render = next(entry for entry in entries if entry["name"] == "render.####.exr")
    assert render["frames"] == "1001-1010, 1013" and render["count"] == 11
    assert render["path"] == str(tmp_path / "render.####.exr")
    assert any(entry["name"] == "notes.txt" for entry in listFolder(str(tmp_path), imagesOnly=False))
    assert frameRanges([1, 2, 3, 5]) == "1-3, 5"
    # The Read node understands the unpadded pattern
    assert imageIO.framePath(str(tmp_path / "plate_#.jpg"), 10) == str(tmp_path / "plate_10.jpg")
    assert imageIO.detectFrameRange(str(tmp_path / "plate_#.jpg")) == (1, 10)


class TestNukeNodes:
    """ The nodes of the Color, Channel, Keyer, Filter, Transform and Draw menus. """

    def _render(self, nodeType, color=(0.2, 0.4, 0.6, 1.0), size=(16, 8), **values):
        graph = Graph("live")
        source = graph.addNewNode("Constant", width=size[0], height=size[1])
        setColor(source, *color)
        node = graph.addNewNode(nodeType, **values)
        graph.addEdge(source.output, node.input)
        graph.update()
        evaluator = Evaluator()
        return evaluator.render(buildPlan(node), 1), evaluator.cache.get((source._uid, 1))

    def test_neutralColorNodesPassThrough(self):
        for nodeType in ("ColorCorrect", "Saturation", "HueShift", "Exposure", "Multiply", "Add", "Sharpen", "Erode",
                         "Median", "Mirror", "Shuffle", "Dot", "Reformat"):
            values = {"width": 16, "height": 8} if nodeType == "Reformat" else {}
            image, source = self._render(nodeType, **values)
            assert image is source, nodeType

    def test_colorNodes(self):
        image, _ = self._render("Exposure", stops=1.0)
        np.testing.assert_allclose(image.rgba()[0, 0], [0.4, 0.8, 1.2, 1.0], rtol=1e-6)
        image, _ = self._render("Saturation", saturation=0.0)
        luma = 0.2126 * 0.2 + 0.7152 * 0.4 + 0.0722 * 0.6
        np.testing.assert_allclose(image.rgba()[0, 0], [luma, luma, luma, 1.0], rtol=1e-5)
        image, _ = self._render("Invert", channels="rgba")
        np.testing.assert_allclose(image.rgba()[0, 0], [0.8, 0.6, 0.4, 0.0], atol=1e-6)
        image, _ = self._render("Clamp", maximum=0.5)  # rgba by default: the alpha is clamped too
        np.testing.assert_allclose(image.rgba()[0, 0], [0.2, 0.4, 0.5, 0.5], rtol=1e-6)
        image, _ = self._render("HueShift", hueRotation=360.0)
        np.testing.assert_allclose(image.rgba()[0, 0], [0.2, 0.4, 0.6, 1.0], atol=1e-5)

    def test_channelNodes(self):
        image, _ = self._render("Shuffle", red="blue", blue="red", alpha="black")
        np.testing.assert_allclose(image.rgba()[0, 0], [0.6, 0.4, 0.2, 0.0], rtol=1e-6)
        image, _ = self._render("Premult", color=(0.5, 0.5, 0.5, 0.5))
        np.testing.assert_allclose(image.rgba()[0, 0], [0.25, 0.25, 0.25, 0.5], rtol=1e-6)
        image, _ = self._render("Unpremult", color=(0.25, 0.25, 0.25, 0.5))
        np.testing.assert_allclose(image.rgba()[0, 0], [0.5, 0.5, 0.5, 0.5], rtol=1e-6)

    def test_keyers(self):
        image, _ = self._render("Keyer", color=(0.5, 0.5, 0.5, 1.0), a=0.25, b=0.75)
        assert abs(image.rgba()[0, 0, 3] - 0.5) < 1e-5
        screenColor = {"r": 0.1, "g": 0.9, "b": 0.1, "a": 1.0}
        image, _ = self._render("ChromaKeyer", color=(0.1, 0.9, 0.1, 1.0), screenColor=screenColor)
        assert image.rgba()[0, 0, 3] < 0.05  # The screen is keyed out
        image, _ = self._render("ChromaKeyer", color=(0.8, 0.3, 0.2, 1.0), screenColor=screenColor)
        assert image.rgba()[0, 0, 3] > 0.99  # The foreground is kept

    def test_erodeAndDilate(self):
        graph = Graph("live")
        shape = graph.addNewNode("Shape", shape="square", width=40, height=40)
        shape.attribute("shapeSize").value = {"x": 20.0, "y": 20.0}
        dilate = graph.addNewNode("Erode", size=-2.0)
        erode = graph.addNewNode("Erode", size=2.0)
        graph.addEdge(shape.output, dilate.input)
        graph.addEdge(shape.output, erode.input)
        graph.update()
        evaluator = Evaluator()
        area = lambda node: evaluator.render(buildPlan(node), 1).rgba()[:, :, 3].sum()
        assert area(erode) < area(shape) < area(dilate)
        assert abs(area(dilate) - 24 * 24) < 1 and abs(area(erode) - 16 * 16) < 1

    def test_transformNodes(self):
        image, _ = self._render("Reformat", width=32, height=16)
        assert image.format == (32, 16)
        np.testing.assert_allclose(image.rgba()[4, 16], [0.2, 0.4, 0.6, 1.0], rtol=1e-5)
        image, _ = self._render("Crop", left=2.0, bottom=1.0, right=10.0, top=6.0, reformat=True)
        assert image.format == (8, 5) and image.bbox == BBox(0, 0, 8, 5)
        image, _ = self._render("Mirror", horizontal=True)
        assert image.bbox == BBox(0, 0, 16, 8)

    def test_generators(self):
        graph = Graph("live")
        checker = graph.addNewNode("CheckerBoard", width=8, height=8, boxSize=4.0)
        wheel = graph.addNewNode("ColorWheel", width=64, height=64)
        circle = graph.addNewNode("Shape", shape="circle", width=100, height=100)
        circle.attribute("shapeSize").value = {"x": 40.0, "y": 40.0}
        graph.update()
        evaluator = Evaluator()
        rgba = evaluator.render(buildPlan(checker), 1).rgba()
        assert not np.allclose(rgba[0, 0], rgba[0, 4])  # Two colors
        np.testing.assert_allclose(rgba[0, 0], rgba[4, 4])
        rgba = evaluator.render(buildPlan(wheel), 1).rgba()
        assert rgba[0, 0, 3] == 0 and rgba[32, 32, 3] == 1
        assert rgba[32, 62, 0] > 0.9 and rgba[32, 62, 1] < 0.2  # Red on the right
        alpha = evaluator.render(buildPlan(circle), 1).rgba()[:, :, 3]
        assert abs(alpha.sum() - np.pi * 20 * 20) < 30

    def test_copyChannel(self):
        graph = Graph("live")
        b = graph.addNewNode("Constant", width=4, height=4)
        a = graph.addNewNode("Constant", width=4, height=4)
        setColor(b, 0.1, 0.2, 0.3, 1.0)
        setColor(a, 0.9, 0.0, 0.0, 0.25)
        copy = graph.addNewNode("Copy")
        graph.addEdge(b.output, copy.B)
        graph.addEdge(a.output, copy.A)
        graph.update()
        np.testing.assert_allclose(Evaluator().render(buildPlan(copy), 1).rgba()[0, 0], [0.1, 0.2, 0.3, 0.25])


def test_renderTimeIsRecorded():
    graph, red, blue, grade, merge = constantGraph()
    evaluator = Evaluator()
    evaluator.render(buildPlan(merge), 1)
    assert evaluator.renderTime(merge._uid, 1) > 0
    assert evaluator.renderTime(grade._uid, 1) == 0.0  # Neutral grade: passed through


def test_upgradeBlurSavedWithAFloatSize(tmp_path):
    """ A value that changed type (float -> 2D group) is upgraded when loading (compositing nodes hold no data). """
    import json
    graph = Graph("live")
    source = graph.addNewNode("Constant")
    blur = graph.addNewNode("Blur")
    graph.addEdge(source.output, blur.input)
    graph.update()
    path = str(tmp_path / "old.mg")
    graph.save(path)
    data = json.loads(open(path).read())
    data["graph"][blur.name]["inputs"]["size"] = 12.0
    data["graph"][blur.name]["inputs"]["sizeY"] = 4.0
    data["header"]["nodesVersions"]["Blur"] = "0.2"
    with open(path, "w") as file:
        json.dump(data, file)
    loaded = loadGraph(path)
    upgraded = loaded.node(blur.name)
    assert not upgraded.isCompatibilityNode
    assert upgraded.attribute("size").getSerializedValue() == {"x": 12.0, "y": 4.0}
    assert upgraded.input.inputRootLink.node.name == source.name


def test_nodeColorFallsBackToTheDescriptorColor():
    graph = Graph("live")
    blur = graph.addNewNode("Blur")
    assert blur.color == "#CC804E"  # Filter category color (Nuke chart)
    blur.internalAttribute("color").value = "#123456"
    assert blur.color == "#123456"  # A user color wins

    class NoColorNode(desc.Node):
        inputs = []
        outputs = []

    from .utils import registeredNodeTypes
    with registeredNodeTypes([NoColorNode]):
        assert graph.addNewNode("NoColorNode").color == ""


class TestFormatInputs:
    """ Generators take the format of their optional input, drawing nodes can use it as format only. """

    def _graph(self):
        graph = Graph("live")
        source = graph.addNewNode("Constant", width=12, height=6)
        setColor(source, 0.0, 0.0, 1.0, 1.0)
        return graph, source

    def test_generatorTakesTheFormatOfItsInputWithoutRenderingIt(self):
        graph, source = self._graph()
        checker = graph.addNewNode("CheckerBoard", width=100, height=100, boxSize=2.0)
        graph.addEdge(source.output, checker.input)
        graph.update()
        assert not checker.width.enabled
        evaluator = Evaluator()
        image = evaluator.render(buildPlan(checker), 1)
        assert image.format == (12, 6)
        assert evaluator.cache.get((source._uid, 1)) is None  # The input pixels are not needed
        assert evaluator.status(source._uid, 1)[0] == EvalStatus.NONE

    def test_disabledGeneratorDoesNotPassItsFormatInputThrough(self):
        graph, source = self._graph()
        constant = graph.addNewNode("Constant")
        graph.addEdge(source.output, constant.input)
        constant.internalAttribute("disable").value = True
        graph.update()
        image = Evaluator().render(buildPlan(constant), 1)
        assert image.format == (12, 6)
        np.testing.assert_allclose(image.rgba()[0, 0], [0, 0, 0, 0])

    def test_shapeInputUsage(self):
        graph, source = self._graph()
        shape = graph.addNewNode("Shape", shape="square", width=100, height=100)
        shape.attribute("shapeSize").value = {"x": 2.0, "y": 2.0}
        graph.addEdge(source.output, shape.input)
        graph.update()
        evaluator = Evaluator()
        background = evaluator.render(buildPlan(shape), 1).rgba()
        np.testing.assert_allclose(background[0, 0], [0, 0, 1, 1])  # Drawn over the input
        shape.inputUsage.value = "format"
        graph.update()
        image = evaluator.render(buildPlan(shape), 1)
        assert image.format == (12, 6)
        np.testing.assert_allclose(image.rgba()[0, 0], [0, 0, 0, 0])  # Transparent background
        np.testing.assert_allclose(image.rgba()[3, 6], [1, 1, 1, 1])  # The square in the center

    def test_ramp(self):
        graph = Graph("live")
        ramp = graph.addNewNode("Ramp", width=11, height=3)
        ramp.point0.value = {"x": 0.5, "y": 1.5}
        ramp.point1.value = {"x": 10.5, "y": 1.5}
        graph.update()
        rgba = Evaluator().render(buildPlan(ramp), 1).rgba()
        np.testing.assert_allclose(rgba[1, :, 0], np.linspace(0, 1, 11), atol=1e-5)
        np.testing.assert_allclose(rgba[1, 5], [0.5, 0.5, 0.5, 0.5], atol=1e-5)
        # Vertical ramp: y up, point 1 at the top
        ramp.point1.value = {"x": 0.5, "y": 2.5}
        graph.update()
        rgba = Evaluator().render(buildPlan(ramp), 1).rgba()
        assert rgba[0, 0, 0] == 1.0 and rgba[2, 0, 0] == 0.0


def test_keylight():
    graph = Graph("live")
    screen = graph.addNewNode("Constant", width=4, height=4)
    foreground = graph.addNewNode("Constant", width=4, height=4)
    setColor(screen, 0.1, 0.8, 0.15, 1.0)
    setColor(foreground, 0.7, 0.4, 0.3, 1.0)
    keyScreen = graph.addNewNode("Keylight")
    keyForeground = graph.addNewNode("Keylight")
    for key, source in ((keyScreen, screen), (keyForeground, foreground)):
        key.screenColor.value = {"r": 0.1, "g": 0.8, "b": 0.15, "a": 1.0}
        graph.addEdge(source.output, key.input)
    graph.update()
    evaluator = Evaluator()
    assert evaluator.render(buildPlan(keyScreen), 1).rgba()[0, 0, 3] < 0.01
    rgba = evaluator.render(buildPlan(keyForeground), 1).rgba()
    np.testing.assert_allclose(rgba[0, 0], [0.7, 0.4, 0.3, 1.0], atol=1e-5)  # No spill: unchanged
    # The core matte forces the foreground
    graph.addEdge(foreground.output, keyScreen.core)
    keyScreen.view.value = "screenMatte"
    graph.update()
    np.testing.assert_allclose(evaluator.render(buildPlan(keyScreen), 1).rgba()[0, 0], [1, 1, 1, 1])


class TestTimeNodes:

    def _sequenceGraph(self, tmp_path, frames=(1, 2, 3, 4)):
        TestReadWrite()._writeSequence(tmp_path, frames)
        graph = Graph("live")
        read = graph.addNewNode("Read", file=str(tmp_path / "img.####.tif"), colorspace="linear")
        return graph, read

    @staticmethod
    def _red(evaluator, node, frame):
        return round(float(evaluator.render(buildPlan(node), frame).rgba()[0, 0, 0]), 4)

    def test_readFrameOffsetAndOutsideRange(self, tmp_path):
        graph, read = self._sequenceGraph(tmp_path)
        read.frameOffset.value = 10
        graph.update()
        evaluator = Evaluator()
        assert evaluator.info(buildPlan(read)) == ImageInfo(4, 2, 11, 14)
        assert self._red(evaluator, read, 12) == 0.2
        assert self._red(evaluator, read, 1) == 0.1  # Held
        read.outsideRange.value = "black"
        graph.update()
        assert self._red(evaluator, read, 1) == 0.0

    def test_timeOffsetAndFrameHold(self, tmp_path):
        graph, read = self._sequenceGraph(tmp_path)
        offset = graph.addNewNode("TimeOffset", timeOffset=2)
        hold = graph.addNewNode("FrameHold", firstFrame=3)
        graph.addEdge(read.output, offset.input)
        graph.addEdge(read.output, hold.input)
        graph.update()
        evaluator = Evaluator()
        assert evaluator.info(buildPlan(offset)) == ImageInfo(4, 2, 3, 6)
        assert self._red(evaluator, offset, 4) == 0.2
        assert evaluator.status(read._uid, 2)[0] == EvalStatus.SUCCESS
        assert evaluator.status(read._uid, 4)[0] == EvalStatus.NONE  # Only the needed frame is read
        assert [self._red(evaluator, hold, f) for f in (1, 2, 4)] == [0.3, 0.3, 0.3]
        hold.increment.value = 2
        hold.firstFrame.value = 1
        graph.update()
        assert [self._red(evaluator, hold, f) for f in (1, 2, 3, 4)] == [0.1, 0.1, 0.3, 0.3]
        offset.reverse.value = True
        offset.timeOffset.value = 0
        graph.update()
        assert [self._red(evaluator, offset, f) for f in (1, 4)] == [0.4, 0.1]

    def test_retime(self, tmp_path):
        graph, read = self._sequenceGraph(tmp_path)
        retime = graph.addNewNode("Retime", speed=0.5)
        graph.addEdge(read.output, retime.input)
        graph.update()
        evaluator = Evaluator()
        assert evaluator.info(buildPlan(retime)) == ImageInfo(4, 2, 1, 7)
        assert [self._red(evaluator, retime, f) for f in (1, 3, 7)] == [0.1, 0.2, 0.4]
        retime.filter.value = "blend"
        graph.update()
        assert self._red(evaluator, retime, 2) == 0.15
        retime.speed.value = 2.0
        retime.reverse.value = True
        retime.filter.value = "nearest"
        graph.update()
        assert evaluator.info(buildPlan(retime)) == ImageInfo(4, 2, 1, 2)
        assert [self._red(evaluator, retime, f) for f in (1, 2)] == [0.4, 0.2]
        # Input range 2-3 stretched onto the output range 11-20
        retime.reverse.value = False
        for name, value in (("useInputFirst", True), ("inputFirst", 2), ("useInputLast", True), ("inputLast", 3),
                            ("useOutputFirst", True), ("outputFirst", 11), ("useOutputLast", True), ("outputLast", 20)):
            retime.attribute(name).value = value
        graph.update()
        assert evaluator.info(buildPlan(retime)) == ImageInfo(4, 2, 11, 20)
        assert [self._red(evaluator, retime, f) for f in (11, 15, 16, 20)] == [0.2, 0.2, 0.3, 0.3]


def test_compositingNodeWithAnOldVersionIsUpgradedWhenLoading(tmp_path):
    import json
    graph, red, blue, grade, merge = constantGraph()
    path = str(tmp_path / "old.mg")
    graph.save(path)
    data = json.loads(open(path).read())
    data["header"]["nodesVersions"]["Constant"] = "0.0"
    del data["graph"][red.name]["inputs"]["width"]
    with open(path, "w") as file:
        json.dump(data, file)
    loaded = loadGraph(path)
    assert not loaded.node(red.name).isCompatibilityNode
    assert loaded.node(grade.name).input.inputRootLink.node.name == red.name


class TestRoto:

    _SQUARE = {"name": "Square", "closed": True, "points": [
        {"x": 2, "y": 2}, {"x": 6, "y": 2}, {"x": 6, "y": 6}, {"x": 2, "y": 6}]}

    def test_rasterizeIsAntialiasedAndExact(self):
        from meshroom.live import roto
        square = roto.shapesMatte([self._SQUARE], BBox(0, 0, 10, 10))
        assert square.sum() == pytest.approx(16.0)
        assert square[9 - 2, 2] == 1.0 and square[9 - 1, 2] == 0.0  # Rows from the top (y up)
        shifted = dict(self._SQUARE, points=[dict(p, x=p["x"] + 0.5) for p in self._SQUARE["points"]])
        matte = roto.shapesMatte([shifted], BBox(0, 0, 10, 10))
        assert matte.sum() == pytest.approx(16.0)
        assert matte[5, 2] == pytest.approx(0.5) and matte[5, 6] == pytest.approx(0.5)

    def test_bezierCircleArea(self):
        from meshroom.live import roto
        k = 0.5523 * 20
        circle = {"closed": True, "points": [
            {"x": 70, "y": 50, "inX": 0, "inY": -k, "outX": 0, "outY": k},
            {"x": 50, "y": 70, "inX": k, "inY": 0, "outX": -k, "outY": 0},
            {"x": 30, "y": 50, "inX": 0, "inY": k, "outX": 0, "outY": -k},
            {"x": 50, "y": 30, "inX": -k, "inY": 0, "outX": k, "outY": 0}]}
        assert roto.shapesMatte([circle], BBox(0, 0, 100, 100)).sum() == pytest.approx(np.pi * 400, rel=2e-3)
        # Open, hidden, inverted shapes
        # An open curve is a stroke (4 pixels wide by default), not filled
        assert 0 < roto.shapesMatte([dict(circle, closed=False)], BBox(0, 0, 100, 100)).sum() < np.pi * 400 / 2
        assert roto.shapesMatte([dict(circle, visible=False)], BBox(0, 0, 100, 100)).sum() == 0
        inverted = roto.shapesMatte([dict(circle, invert=True)], BBox(0, 0, 100, 100)).sum()
        assert inverted == pytest.approx(10000 - np.pi * 400, rel=1e-3)

    def test_rotoNode(self):
        import json
        graph = Graph("live")
        source = graph.addNewNode("Constant", width=10, height=10)
        setColor(source, 0.0, 0.0, 1.0, 1.0)
        rotoNode = graph.addNewNode("Roto")
        rotoNode.shapes.value = json.dumps([dict(self._SQUARE, feather=0.0)])
        graph.addEdge(source.output, rotoNode.input)
        graph.update()
        evaluator = Evaluator()
        rgba = evaluator.render(buildPlan(rotoNode), 1).rgba()
        np.testing.assert_allclose(rgba[5, 4], [1, 1, 1, 1])  # White over the input
        np.testing.assert_allclose(rgba[0, 0], [0, 0, 1, 1])
        rotoNode.outputMode.value = "replaceAlpha"
        graph.update()
        rgba = evaluator.render(buildPlan(rotoNode), 1).rgba()
        np.testing.assert_allclose(rgba[5, 4], [0, 0, 1, 1])
        np.testing.assert_allclose(rgba[0, 0], [0, 0, 1, 0])  # The matte replaces the alpha
        rotoNode.shapes.value = "not json"
        graph.update()
        with pytest.raises(EvaluationError):
            evaluator.render(buildPlan(rotoNode), 1)


class TestChapters:

    def test_foldBands(self):
        from meshroom.core.chapters import FOLDED_GAP, foldBands, foldShift, isFoldedAway
        bands = foldBands([(500, False), (0, True), (200, True)])
        assert bands == [(0, 200, 200 - FOLDED_GAP), (200, 500, 300 - FOLDED_GAP)]
        assert not isFoldedAway(bands, 0) and not isFoldedAway(bands, 200)  # The lines stay visible
        assert isFoldedAway(bands, 100) and isFoldedAway(bands, 300)
        assert foldShift(bands, -50) == 0
        assert foldShift(bands, 200) == 200 - FOLDED_GAP  # The next line moves onto the folded one
        assert foldShift(bands, 600) == 500 - 2 * FOLDED_GAP
        # The last chapter cannot fold: there is no line below it
        assert foldBands([(0, False), (100, True)]) == []

    def test_unfoldYIsTheInverseOfTheFolding(self):
        from meshroom.core.chapters import FOLDED_GAP, foldBands, foldShift, unfoldY
        bands = foldBands([(0, True), (200, False), (500, True), (800, False)])
        for y in (-100, 0, 200, 260, 500, 800, 1000):
            assert unfoldY(bands, y - foldShift(bands, y)) == y
        # Between a folded line and the next one: onto the next line
        assert unfoldY(bands, FOLDED_GAP / 2) == 200
        assert unfoldY([], 42) == 42

    def test_chapterNodeSaveAndLoad(self, tmp_path):
        if not pluginManager.isNodeDescRegistered("Chapter"):
            pluginManager.addPluginFromBuiltInFolder(
                "general", os.path.join(os.path.dirname(meshroom.__file__), "nodes", "general"))
        graph = Graph("chapters")
        chapter = graph.addNewNode("Chapter")
        assert chapter.isBackdropNode
        assert not chapter.isComputableType
        chapter.internalAttribute("folded").value = True
        chapter.internalAttribute("label").value = "Keying"
        path = str(tmp_path / "chapters.mg")
        graph.save(path)
        loaded = loadGraph(path).node(chapter.name)
        assert not loaded.isCompatibilityNode
        assert loaded.internalAttribute("folded").value is True
        assert loaded.label == "Keying"


class TestRound5:

    _SQUARE = TestRoto._SQUARE

    def test_rotoShapeKeysAreInterpolated(self):
        from meshroom.live import roto
        moved = [dict(point, x=point["x"] + 10) for point in self._SQUARE["points"]]
        shape = dict(self._SQUARE, keys=[{"frame": 1001, "points": self._SQUARE["points"]},
                                         {"frame": 1011, "points": moved}])
        assert roto.pointsAtFrame(shape, 990)[0]["x"] == 2  # Held before the first key
        assert roto.pointsAtFrame(shape, 1006)[0]["x"] == pytest.approx(7)
        assert roto.pointsAtFrame(shape, 2000)[0]["x"] == 12
        matte = roto.shapesMatte([shape], BBox(0, 0, 20, 10), frame=1006)
        assert matte[9 - 3, 7] == 1.0 and matte[9 - 3, 6] == 0.0

    def test_rotoFollowsATrackerAndAddsAFeather(self):
        import json
        graph = Graph("live")
        source = graph.addNewNode("Constant", width=20, height=10)
        tracker = graph.addNewNode("Tracker")
        tracker.tracks.value = json.dumps([{"name": "t1", "keys": {"1": [5, 5], "11": [15, 5]}}])
        rotoNode = graph.addNewNode("Roto", outputMode="replaceAlpha")
        shape = dict(self._SQUARE, link={"track": "t1", "point": -1, "frame": 1})
        rotoNode.shapes.value = json.dumps([shape])
        graph.addEdge(source.output, tracker.input)
        graph.addEdge(source.output, rotoNode.input)
        graph.addEdge(tracker.output, rotoNode.tracker)
        graph.update()
        evaluator = Evaluator()
        alpha = evaluator.render(buildPlan(rotoNode), 6).rgba()[:, :, 3]
        assert alpha[9 - 3, 7] == 1.0 and alpha[9 - 3, 6] == 0.0  # Moved by 5 pixels
        assert evaluator.status(tracker._uid, 6)[0] == EvalStatus.NONE  # The tracker image is not needed
        rotoNode.feather.value = 4.0
        graph.update()
        alpha = evaluator.render(buildPlan(rotoNode), 1).rgba()[:, :, 3]
        assert 0 < alpha[9 - 3, 1] < 1  # Soft edge
        # The tracker matchmoves its input with its tracks
        tracker.transform.value = "matchmove"
        tracker.referenceFrame.value = 1
        graph.update()
        image = evaluator.render(buildPlan(tracker), 11)
        assert image.bbox == BBox(10, 0, 30, 10)

    def test_rotoOutputModes(self):
        import json
        graph = Graph("live")
        source = graph.addNewNode("Constant", width=10, height=10)
        setColor(source, 0.5, 0.5, 0.5, 0.5)
        rotoNode = graph.addNewNode("Roto")
        rotoNode.shapes.value = json.dumps([self._SQUARE])
        graph.addEdge(source.output, rotoNode.input)
        evaluator = Evaluator()
        expected = {"addAlpha": ([0.5, 0.5, 0.5, 1.0], [0.5, 0.5, 0.5, 0.5]),
                    "cutOut": ([0.5, 0.5, 0.5, 0.5], [0, 0, 0, 0])}
        for mode, (inside, outside) in expected.items():
            rotoNode.outputMode.value = mode
            graph.update()
            rgba = evaluator.render(buildPlan(rotoNode), 1).rgba()
            np.testing.assert_allclose(rgba[5, 4], inside, atol=1e-6)
            np.testing.assert_allclose(rgba[0, 0], outside, atol=1e-6)

    def test_trackingFollowsAMovingFeature(self):
        from meshroom.live.tracking import trackFrames

        def imageAt(frame):
            height, width = 80, 120
            rows, columns = np.mgrid[0:height, 0:width]
            x, y = columns + 0.5, height - rows - 0.5
            cx, cy = 30 + 2.5 * frame, 40 + 1.25 * frame
            blob = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 32.0)
            blob += 0.4 * np.exp(-((x - cx - 5) ** 2 + (y - cy + 3) ** 2) / 8.0)
            rgba = np.zeros((height, width, 4), np.float32)
            rgba[:, :, :3] = blob[:, :, np.newaxis]
            return ImageBuffer.fromRGBA(rgba, (width, height))

        keys = trackFrames({"keys": {"0": [30, 40]}, "pattern": 21, "search": 41}, list(range(9)), imageAt)
        assert len(keys) == 8
        assert keys["8"] == pytest.approx([50, 50], abs=0.05)

    def test_cloneOutputsTheClonedBranch(self):
        graph, red, blue, grade, merge = constantGraph()
        clone = graph.addNewNode("Clone")
        graph.addEdge(grade.output, clone.source)
        exposure = graph.addNewNode("Exposure", stops=1.0)
        graph.addEdge(clone.output, exposure.input)
        graph.update()
        evaluator = Evaluator()
        np.testing.assert_allclose(evaluator.render(buildPlan(exposure), 1).rgba()[0, 0], [1.0, 0, 0, 1.0])
        setColor(red, r=0.25)
        graph.update()  # A change behind the cloned node reaches the clone
        np.testing.assert_allclose(evaluator.render(buildPlan(exposure), 1).rgba()[0, 0], [0.5, 0, 0, 1.0])

    def test_splitVec2IsSavedWithTheNode(self, tmp_path):
        graph = Graph("live")
        blur = graph.addNewNode("Blur")
        blur.internalAttribute("splitVec2").value = "size"
        path = str(tmp_path / "split.mg")
        graph.save(path)
        assert loadGraph(path).node(blur.name).internalAttribute("splitVec2").value == "size"


class TestCurves:

    def test_monotoneCurve(self):
        from meshroom.live import curves
        points = [[0, 0], [0.5, 0.8], [1, 1]]
        values = curves.evaluate(points, np.linspace(0, 1, 101))
        assert values[0] == 0 and values[50] == pytest.approx(0.8) and values[-1] == pytest.approx(1)
        assert np.all(np.diff(values) >= -1e-12)  # No overshoot
        assert curves.evaluate(points, [2.0])[0] == pytest.approx(1 + 1.0 * curves._slopes(
            np.array([0, 0.5, 1.0]), np.array([0, 0.8, 1.0]))[-1])  # Straight line beyond the end
        assert curves.evaluate([], [0.3])[0] == pytest.approx(0.3)
        assert curves.isIdentity([[0, 0], [1, 1]]) and not curves.isIdentity([[0, 0], [1, 2]])

    def test_colorLookupAndPickers(self):
        import json
        from meshroom.live import curves
        graph = Graph("live")
        source = graph.addNewNode("Constant", width=4, height=4)
        setColor(source, 0.2, 0.4, 0.6, 1.0)
        lookup = graph.addNewNode("ColorLookup")
        graph.addEdge(source.output, lookup.input)
        graph.update()
        evaluator = Evaluator()
        assert evaluator.render(buildPlan(lookup), 1) is evaluator.cache.get((source._uid, 1))  # Identity
        values = curves.lookupPoints(curves.parseCurves(lookup.curves.value), "rgb", (0.2, 0.4, 0.6), (0.3, 0.4, 0.5))
        lookup.curves.value = json.dumps(values)
        graph.update()
        np.testing.assert_allclose(evaluator.render(buildPlan(lookup), 1).rgba()[0, 0], [0.3, 0.4, 0.5, 1.0], atol=1e-6)

    def test_colorLookupHSV(self):
        import json
        from meshroom.live import curves
        graph = Graph("live")
        source = graph.addNewNode("Constant", width=4, height=4)
        setColor(source, 0.8, 0.2, 0.2, 1.0)  # A red
        lookup = graph.addNewNode("ColorLookupHSV")
        lookup.curves.value = json.dumps(curves.lookupPoints({}, "hsv", (0.8, 0.2, 0.2), (0.2, 0.8, 0.2)))
        graph.addEdge(source.output, lookup.input)
        graph.update()
        np.testing.assert_allclose(Evaluator().render(buildPlan(lookup), 1).rgba()[0, 0], [0.2, 0.8, 0.2, 1.0], atol=1e-4)

    def test_matchCurves(self):
        from meshroom.live import curves
        rng = np.random.default_rng(1)
        source = rng.random((64, 64, 4))
        target = source * 0.5 + 0.1
        matched = curves.matchCurvesRGB(source, target)
        for index, name in enumerate(("red", "green", "blue")):
            np.testing.assert_allclose(curves.evaluate(matched[name], source[:, :, index]), target[:, :, index], atol=0.03)
        shifted = curves.hsvToRgb(curves.rgbToHsv(source[:, :, :3]) + np.array([0.1, 0, 0]))
        hsvCurves = curves.matchCurvesHSV(source, shifted)
        assert hsvCurves["hue"][0][1] == pytest.approx(0.1, abs=0.01)

    def test_timeWarp(self, tmp_path):
        import json
        graph, read = TestTimeNodes()._sequenceGraph(tmp_path)
        warp = graph.addNewNode("TimeWarp")
        graph.addEdge(read.output, warp.input)
        graph.update()
        evaluator = Evaluator()
        assert evaluator.render(buildPlan(warp), 2) is evaluator.render(buildPlan(read), 2)  # No point: identity
        warp.lookup.value = json.dumps({"time": [[1, 4], [10, 1]]})  # Backwards, 10 frames
        graph.update()
        assert evaluator.info(buildPlan(warp)) == ImageInfo(4, 2, 1, 10)
        red = lambda frame: round(float(evaluator.render(buildPlan(warp), frame).rgba()[0, 0, 0]), 4)
        assert [red(1), red(10)] == [0.4, 0.1]


def test_rotoDoesNotNeedTheTrackerImage():
    import json
    graph = Graph("live")
    tracker = graph.addNewNode("Tracker")  # No input: its image cannot be computed
    tracker.tracks.value = json.dumps([{"name": "t1", "keys": {"1": [0, 0], "3": [2, 0]}}])
    rotoNode = graph.addNewNode("Roto", width=10, height=10)
    rotoNode.shapes.value = json.dumps([dict(TestRoto._SQUARE, link={"track": "t1", "point": -1, "frame": 1})])
    graph.addEdge(tracker.output, rotoNode.tracker)
    graph.update()
    alpha = Evaluator().render(buildPlan(rotoNode), 3).rgba()[:, :, 3]
    assert alpha.sum() == pytest.approx(16.0) and alpha[9 - 2, 4] == 1.0 and alpha[9 - 2, 3] == 0.0


def test_trackingKeepsTheFramesTrackedBeforeAStop():
    from meshroom.live import EvaluationCancelled
    from meshroom.live.tracking import trackFrames

    def imageAt(frame):
        if frame >= 3:
            raise EvaluationCancelled()
        rgba = np.zeros((40, 40, 4), np.float32)
        rgba[15 + frame:25 + frame, 15:25, :3] = 1.0
        return ImageBuffer.fromRGBA(rgba, (40, 40))

    keys = trackFrames({"keys": {"0": [20, 20]}, "pattern": 15, "search": 25}, [0, 1, 2, 3, 4], imageAt)
    assert sorted(keys) == ["1", "2"]


def test_retimeOutputFirstFrameIsUpgraded(tmp_path):
    import json
    graph = Graph("live")
    retime = graph.addNewNode("Retime")
    path = str(tmp_path / "retime.mg")
    graph.save(path)
    data = json.loads(open(path).read())
    data["header"]["nodesVersions"]["Retime"] = "0.1"
    inputs = data["graph"][retime.name]["inputs"]
    for name in ("useInputFirst", "inputFirst", "useInputLast", "inputLast", "useOutputFirst", "outputFirst",
                 "useOutputLast", "outputLast"):
        inputs.pop(name, None)
    inputs.update({"keepFirstFrame": False, "outputFirstFrame": 1001})
    with open(path, "w") as file:
        json.dump(data, file)
    loaded = loadGraph(path).node(retime.name)
    assert loaded.useOutputFirst.value is True and loaded.outputFirst.value == 1001


class TestRound6:

    def test_fittedTransforms(self):
        from meshroom.live import geometry
        src = np.array([[0, 0], [10, 0], [10, 10], [0, 10.0]])
        angle = np.radians(30)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        dst = (src @ rotation.T) * 2 + [5, 7]
        matrix = geometry.fitTransform("translate rotate scale", src, dst)
        np.testing.assert_allclose((matrix @ np.c_[src, np.ones(4)].T).T[:, :2], dst, atol=1e-9)
        assert np.allclose(geometry.fitTransform("translate", src, src + [3, 4])[:2, 2], [3, 4])
        warped = np.array([[1, 2], [12, 1], [9, 13], [-1, 9.0]])
        homography = geometry.fitTransform("perspective", src, warped)
        projected = (homography @ np.c_[src, np.ones(4)].T).T
        np.testing.assert_allclose(projected[:, :2] / projected[:, 2:], warped, atol=1e-9)
        with pytest.raises(ValueError):
            geometry.fitTransform("perspective", src[:3], warped[:3])

    def _trackerGraph(self, tracks):
        import json
        graph = Graph("live")
        source = graph.addNewNode("CheckerBoard", width=40, height=40, boxSize=5.0)
        tracker = graph.addNewNode("Tracker")
        tracker.tracks.value = json.dumps(tracks)
        graph.addEdge(source.output, tracker.input)
        return graph, source, tracker

    def test_stabilizeAndCornerPin(self):
        tracks = [{"name": f"t{i}", "keys": {"1": p, "2": [p[0] + 3, p[1] + 2]}}
                  for i, p in enumerate([[5, 5], [35, 5], [35, 35], [5, 35]])]
        graph, source, tracker = self._trackerGraph(tracks)
        stabilize = graph.addNewNode("Stabilize", model="translate", referenceFrame=1)
        pin = graph.addNewNode("CornerPin", tracks="t0,t1,t2,t3", referenceFrame=1)
        for node in (stabilize, pin):
            graph.addEdge(source.output, node.input)
            graph.addEdge(tracker.output, node.tracker)
        graph.update()
        evaluator = Evaluator()
        assert evaluator.render(buildPlan(stabilize), 2).bbox == BBox(-3, -2, 37, 38)  # Moved back by (3, 2)
        assert evaluator.render(buildPlan(pin), 2).bbox == BBox(3, 2, 43, 42)  # Moved with the tracks
        assert evaluator.render(buildPlan(pin), 1) is evaluator.cache.get((source._uid, 1))  # Reference: identity

    def test_motionBlur(self):
        tracks = [{"name": "t", "keys": {"1": [10, 10], "3": [30, 10]}}]  # 10 pixels per frame
        graph, source, tracker = self._trackerGraph(tracks)
        blur = graph.addNewNode("MotionBlur", shutter=1.0, samples=11)
        graph.addEdge(source.output, blur.input)
        graph.addEdge(tracker.output, blur.tracker)
        graph.update()
        image = Evaluator().render(buildPlan(blur), 2)
        assert image.bbox == BBox(-5, 0, 45, 40)
        # A square, smeared horizontally only
        square = graph.addNewNode("Shape", shape="square", width=40, height=40)
        square.attribute("shapeSize").value = {"x": 10.0, "y": 10.0}
        graph.addEdge(square.output, blur.input)
        graph.update()
        alpha = Evaluator().render(buildPlan(blur), 2).rgba()
        offset = 5  # bbox x0 = -5
        assert 0 < alpha[20, 13 + offset, 3] < 0.5 < 0.85 < alpha[20, 20 + offset, 3]
        assert alpha[12, 20 + offset, 3] == 0  # Not above the square

    def test_reformatCrop(self):
        graph = Graph("live")
        source = graph.addNewNode("Constant", width=10, height=10)
        transform = graph.addNewNode("Transform")
        transform.attribute("translate").value = {"x": 5.0, "y": 0.0}
        reformat = graph.addNewNode("Reformat", width=10, height=10, crop=True)
        graph.addEdge(source.output, transform.input)
        graph.addEdge(transform.output, reformat.input)
        graph.update()
        assert Evaluator().render(buildPlan(reformat), 1).bbox == BBox(5, 0, 10, 10)

    def test_rotoLinksWithSeveralTrackers(self):
        from meshroom.live import roto
        square = TestRoto._SQUARE
        tracks = {"tracker": [{"name": "a", "keys": {"1": [0, 0], "2": [1, 0]}}],
                  "tracker2": [{"name": "b", "keys": {"1": [0, 0], "2": [0, 5]}}]}
        shape = dict(square, links=[{"input": "tracker", "track": "a", "point": -1, "frame": 1},
                                    {"input": "tracker2", "track": "b", "point": 2, "frame": 1}])
        points = roto.shapeAtFrame(shape, 2, tracks)["points"]
        assert (points[0]["x"], points[0]["y"]) == (3, 2)  # Whole shape: track a
        assert (points[2]["x"], points[2]["y"]) == (6, 11)  # Its own link: track b only
        # The former single link
        legacy = dict(square, link={"track": "a", "point": -1, "frame": 1})
        assert roto.shapeAtFrame(legacy, 2, tracks)["points"][0]["x"] == 3


class TestCameraTracker:

    @staticmethod
    def _syntheticTracks(seed=3, frames=30):
        rng = np.random.default_rng(seed)
        width, height, focal = 1920, 1080, 1500.0
        points = np.c_[rng.uniform(-4, 4, 200), rng.uniform(-2, 2, 200), rng.uniform(6, 14, 200)]
        tracks = [{"name": f"p{i}", "keys": {}} for i in range(len(points))]
        centers = {}
        for frame in range(1, frames + 1):
            angle = np.radians(0.4 * frame)
            rotation = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
            center = np.array([0.08 * frame, 0.01 * frame, 0.0])
            centers[frame] = center
            camera = (points - center) @ rotation.T
            uv = camera[:, :2] / camera[:, 2:] * focal + [width / 2, height / 2]
            for i, (u, v) in enumerate(uv):
                if 0 <= u < width and 0 <= v < height:
                    tracks[i]["keys"][str(frame)] = [u + rng.normal(0, 0.2), height - v + rng.normal(0, 0.2)]
        return tracks, centers, (width, height, focal)

    def test_solveAndExport(self, tmp_path):
        from meshroom.live import camera
        tracks, centers, (width, height, focal) = self._syntheticTracks()
        solve = camera.solveCamera(tracks, width, height, focal)
        assert solve["error"] < 0.6 and len(solve["frames"]) >= 25 and len(solve["points"]) > 150
        # The trajectory, up to the scale and the origin of the solve
        solved = {int(f): -np.array(c["R"]).T @ np.array(c["t"]) for f, c in solve["cameras"].items()}
        frames = sorted(solved)
        a = np.array([solved[f] for f in frames]) - solved[frames[0]]
        b = np.array([centers[f] for f in frames]) - centers[frames[0]]
        scale = np.linalg.norm(b) / np.linalg.norm(a)
        assert np.abs(a * scale - b).max() < 0.05
        cameraFile, pointsFile = str(tmp_path / "camera.usda"), str(tmp_path / "points.usda")
        camera.exportCameraUsd(solve, cameraFile, 35.0, 36.0)
        camera.exportPointsUsd(solve, pointsFile)
        text = open(cameraFile).read()
        assert text.startswith("#usda 1.0") and "xformOp:transform.timeSamples" in text and "focalLength = 35" in text
        assert "point3f[] points" in open(pointsFile).read()

    def test_autotrackAction(self):
        import json
        from meshroom.live.actions import ActionContext
        rng = np.random.default_rng(0)
        texture = rng.random((300, 400)).astype(np.float32)
        # Smooth texture (box blur) so that the features can be matched
        texture = sum(np.roll(np.roll(texture, i, 0), j, 1) for i in range(-2, 3) for j in range(-2, 3)) / 25.0

        graph = Graph("live")
        cameraTracker = graph.addNewNode("CameraTracker", features=40)
        graph.update()
        values = buildPlan(cameraTracker).targetNode.values

        def render(frame):
            pixels = texture[40 + frame:40 + frame + 160, 40 + 2 * frame:40 + 2 * frame + 240]
            rgba = np.zeros((160, 240, 4), np.float32)
            rgba[:, :, :3] = pixels[:, :, np.newaxis] * 3
            return ImageBuffer.fromRGBA(rgba, (240, 160))

        context = ActionContext(values, currentFrame=1)
        context.render = render
        context.info = ImageInfo(240, 160, 1, 10)
        result = cameraTracker.nodeDesc.runAction("autotrack", context)
        tracks = json.loads(result["tracks"])
        assert len(tracks) >= 30
        track = max(tracks, key=lambda t: len(t["keys"]))
        first, last = min(track["keys"], key=int), max(track["keys"], key=int)
        frames = int(last) - int(first)
        moved = np.array(track["keys"][last]) - np.array(track["keys"][first])
        np.testing.assert_allclose(moved, [-2 * frames, frames], atol=0.05)  # The image moves left and up


def test_homographyKeepsThePlaneInFront():
    from meshroom.live import geometry, ops
    # A strongly receding plane: the horizon passes between the origin and the corners
    src = np.array([[100, 100], [200, 100], [200, 200], [100, 200.0]])
    dst = np.array([[100, 100], [200, 100], [160, 120], [140, 120.0]])
    matrix = geometry.homography(src, dst)
    weights = matrix[2] @ np.c_[src, np.ones(4)].T
    assert np.all(weights > 0)
    projected = (matrix @ np.c_[src, np.ones(4)].T)
    np.testing.assert_allclose((projected[:2] / projected[2]).T, dst, atol=1e-6)
    assert ops.transformedBBox(BBox(100, 100, 200, 200), matrix) == BBox(100, 100, 200, 120)


class TestRound7:

    def _blurGraph(self, tracks, **values):
        import json
        graph = Graph("live")
        square = graph.addNewNode("Shape", shape="square", width=80, height=40)
        square.attribute("shapeSize").value = {"x": 60.0, "y": 10.0}
        tracker = graph.addNewNode("Tracker")
        tracker.tracks.value = json.dumps(tracks)
        blur = graph.addNewNode("MotionBlur", shutter=1.0, **values)
        graph.addEdge(square.output, tracker.input)
        graph.addEdge(square.output, blur.input)
        graph.addEdge(tracker.output, blur.tracker)
        graph.update()
        return graph, blur

    def test_motionBlurFactorAndVectorField(self):
        moving = [{"name": "t", "keys": {"1": [10, 20], "3": [30, 20]}}]  # 10 pixels per frame
        graph, blur = self._blurGraph(moving, factor=2.0)
        assert Evaluator().render(buildPlan(blur), 2).bbox == BBox(-10, 0, 90, 40)  # 20 pixels long
        # Vector field: the left track moves, the right one does not
        field = [{"name": "left", "keys": {"1": [10, 20], "3": [30, 20]}},
                 {"name": "right", "keys": {"1": [70, 20], "3": [70, 20]}}]
        graph, blur = self._blurGraph(field, model="vector field")
        alpha = Evaluator().render(buildPlan(blur), 2).rgba()[:, :, 3]
        offset = 6  # The bbox grows by the largest motion (5 pixels) + 1
        row = 20
        assert 0.2 < alpha[row, 12 + offset] < 0.8  # Left edge of the square: blurred
        assert alpha[row, 69 + offset] > 0.9 and alpha[row, 71 + offset] < 0.1  # Right edge: (almost) sharp

    def test_tracksAreScaledToTheImageFormat(self):
        from meshroom.live import tracking
        track = {"name": "t", "format": [1024, 576], "keys": {"1": [512, 288]}, "pattern": 21, "search": 61}
        scaled = tracking.scaleTracks([track], 4096, 2304)[0]
        assert scaled["keys"]["1"] == [2048, 1152] and scaled["pattern"] == 84
        assert tracking.scaleTracks([dict(track, format=None)], 4096, 2304)[0]["keys"]["1"] == [512, 288]

    def test_motionBlurSpeed(self):
        import time
        graph = Graph("live")
        source = graph.addNewNode("CheckerBoard", width=1920, height=1080)
        tracker = graph.addNewNode("Tracker")
        import json
        tracker.tracks.value = json.dumps([{"name": "t", "keys": {"1": [0, 0], "3": [60, 0]}}])
        blur = graph.addNewNode("MotionBlur", shutter=1.0)
        graph.addEdge(source.output, tracker.input)
        graph.addEdge(source.output, blur.input)
        graph.addEdge(tracker.output, blur.tracker)
        graph.update()
        evaluator = Evaluator()
        evaluator.render(buildPlan(source), 2)
        start = time.perf_counter()
        evaluator.render(buildPlan(blur), 2)
        assert time.perf_counter() - start < 3.0  # 30 pixels, 31 samples on 1080p


class TestAnimation:

    def test_valueAtInterpolatesNumbersAndHoldsTheOthers(self):
        from meshroom.live import animation as anim
        keys = anim.parseAnimation('{"gain": [[10, 3.0], [0, 1.0]]}')["gain"]
        assert keys[0][0] == 0.0  # Sorted by frame
        assert anim.valueAt(keys, -5) == 1.0 and anim.valueAt(keys, 5) == 2.0 and anim.valueAt(keys, 20) == 3.0
        size = [[0, {"x": 0.0, "y": 2.0}], [4, {"x": 4.0, "y": 2.0}]]
        assert anim.valueAt(size, 1) == {"x": 1.0, "y": 2.0}
        assert anim.valueAt([[0, True], [4, False]], 3) is True
        assert anim.valueAt([[0, 1], [4, 4]], 1) == 2  # Integers stay integers (1.75 rounded)
        animation = anim.setKey({}, "gain", 2, 5.0)
        animation = anim.setKey(animation, "gain", 2, 6.0)
        assert animation == {"gain": [[2.0, 6.0]]}
        assert anim.removeKey(animation, "gain", 2) == {}
        assert anim.dumpAnimation({}) == "" and anim.parseAnimation("not json") == {}

    def _animatedGraph(self):
        from meshroom.live import animation as anim
        graph = Graph("live")
        constant = graph.addNewNode("Constant", width=4, height=4)
        setColor(constant, r=1.0, g=1.0, b=1.0, a=1.0)
        multiply = graph.addNewNode("Multiply")
        graph.addEdge(constant.output, multiply.input)
        multiply.internalAttribute(anim.ANIMATION).value = anim.dumpAnimation({"value": [[1, 0.0], [11, 1.0]]})
        graph.update()
        return graph, multiply

    def test_animatedValueIsRenderedPerFrame(self):
        graph, multiply = self._animatedGraph()
        evaluator = Evaluator()
        plan = buildPlan(multiply)
        assert evaluator.render(plan, 1).rgba()[0, 0, 0] == pytest.approx(0.0)
        assert evaluator.render(plan, 6).rgba()[0, 0, 0] == pytest.approx(0.5)
        assert evaluator.render(plan, 20).rgba()[0, 0, 0] == pytest.approx(1.0)

    def test_storedValueOfAnAnimatedParameterIsNotInTheUid(self):
        from meshroom.live import animation as anim
        graph, multiply = self._animatedGraph()
        uid = multiply._uid
        multiply.attribute("value").value = 0.7  # Mirrors another frame: same render
        graph.update()
        assert multiply._uid == uid
        multiply.internalAttribute(anim.ANIMATION).value = anim.dumpAnimation({"value": [[1, 0.0], [11, 2.0]]})
        graph.update()
        assert multiply._uid != uid  # The keys are part of the uid
        multiply.internalAttribute(anim.ANIMATION).value = ""
        graph.update()
        uid = multiply._uid
        multiply.attribute("value").value = 0.2  # Not animated any more: the value is in the uid
        graph.update()
        assert multiply._uid != uid

    def test_parameterLinkedToAnAnimatedOneFollowsItsKeys(self):
        graph, multiply = self._animatedGraph()
        follower = graph.addNewNode("Multiply")
        graph.addEdge(multiply.output, follower.input)
        graph.addEdge(multiply.attribute("value"), follower.attribute("value"))
        graph.update()
        uid = follower._uid
        multiply.attribute("value").value = 0.9  # The mirrored value of another frame
        graph.update()
        assert follower._uid == uid
        evaluator = Evaluator()
        plan = buildPlan(follower)
        assert evaluator.render(plan, 6).rgba()[0, 0, 0] == pytest.approx(0.25)  # 0.5 * 0.5
        assert evaluator.render(plan, 11).rgba()[0, 0, 0] == pytest.approx(1.0)

    def test_animationIsSavedAndLoaded(self, tmp_path):
        graph, multiply = self._animatedGraph()
        path = str(tmp_path / "animated.mg")
        graph.save(path)
        loaded = loadGraph(path)
        node = loaded.node(multiply.name)
        assert Evaluator().render(buildPlan(node), 6).rgba()[0, 0, 0] == pytest.approx(0.5)


class TestRound7Nodes:

    def test_colorCorrectGainPerChannel(self):
        graph = Graph("live")
        constant = graph.addNewNode("Constant", width=4, height=4)
        setColor(constant, r=0.5, g=0.5, b=0.5, a=1.0)
        correct = graph.addNewNode("ColorCorrect")
        graph.addEdge(constant.output, correct.input)
        graph.update()
        assert Evaluator().render(buildPlan(correct), 1).rgba()[0, 0, 0] == pytest.approx(0.5)
        for channel, value in zip("rgba", (2.0, 1.0, 0.0, 1.0)):
            correct.attribute("gain").value.get(channel).value = value
        graph.update()
        assert Evaluator().render(buildPlan(correct), 1).rgba()[0, 0].tolist() == pytest.approx([1.0, 0.5, 0.0, 1.0])

    def test_colorCorrectSavedWithFloatsIsUpgraded(self, tmp_path):
        import json
        graph = Graph("live")
        correct = graph.addNewNode("ColorCorrect")
        path = str(tmp_path / "old.mg")
        graph.save(path)
        with open(path) as f:
            data = json.load(f)
        nodeData = data["graph"][correct.name]
        nodeData["version"] = "0.2"
        nodeData["inputs"]["gain"] = 2.0
        with open(path, "w") as f:
            json.dump(data, f)
        loaded = loadGraph(path)
        node = loaded.node(correct.name)
        assert node.attribute("gain").getSerializedValue() == {"r": 2.0, "g": 2.0, "b": 2.0, "a": 2.0}

    def _defocusGraph(self, depthValue, **values):
        graph = Graph("live")
        square = graph.addNewNode("Shape", shape="square", width=64, height=64)
        square.attribute("shapeSize").value = {"x": 20.0, "y": 20.0}
        depth = graph.addNewNode("Constant", width=64, height=64)
        setColor(depth, r=depthValue, g=depthValue, b=depthValue, a=1.0)
        defocus = graph.addNewNode("ZDefocus", math="linear", focalPlane=0.5, size=8.0, **values)
        graph.addEdge(square.output, defocus.input)
        graph.addEdge(depth.output, defocus.depthImage)
        graph.update()
        return graph, defocus

    def test_zDefocusBlursOutOfTheFocalPlane(self):
        graph, defocus = self._defocusGraph(0.5)
        sharp = Evaluator().render(buildPlan(defocus), 1).rgba()[:, :, 3]
        assert set(np.unique(np.round(sharp, 3))) <= {0.0, 1.0}  # In focus: unchanged
        graph, defocus = self._defocusGraph(1.0)  # 0.5 behind: radius 4
        blurred = Evaluator().render(buildPlan(defocus), 1).rgba()[:, :, 3]
        assert blurred.sum() == pytest.approx(sharp.sum(), rel=0.02)  # The energy is kept
        assert 0.2 < blurred[32, 22] < 0.8 and blurred[32, 32] == pytest.approx(1.0, abs=1e-3)
        setup = graph.node(defocus.name)
        setup.attribute("outputMode").value = "focal plane setup"
        graph.update()
        tinted = Evaluator().render(buildPlan(setup), 1).rgba()
        assert tinted[32, 32, 2] > tinted[32, 32, 0]  # Behind: blue

    def test_bokehKernels(self):
        from meshroom.live import defocus
        disc = defocus.bokehKernel(10.0)
        hexagon = defocus.bokehKernel(10.0, blades=6)
        assert disc.sum() == pytest.approx(1.0) and hexagon.sum() == pytest.approx(1.0)
        assert hexagon.shape == disc.shape and not np.allclose(hexagon, disc)
        wide = defocus.bokehKernel(10.0, aspect=2.0)
        assert wide.shape[1] > 1.8 * wide.shape[0]
        assert defocus.bokehKernel(0.2).shape == (1, 1)

    def test_zDefocusNearBlursOverTheBackground(self):
        from meshroom.live import defocus
        # Left half near (blurred), right half in focus: the near color spreads over the right
        rgba = np.zeros((40, 80, 4), np.float32)
        rgba[:, :40] = (1, 0, 0, 1)
        rgba[:, 40:] = (0, 0, 1, 1)
        radius = np.zeros((40, 80), np.float32)
        radius[:, :40] = -6.0
        out = defocus.defocus(rgba, radius, 6.0, 4, lambda r: defocus.bokehKernel(r))
        assert out[20, 42, 0] > 0.1  # Red over the sharp blue
        assert out[20, 70].tolist() == pytest.approx([0, 0, 1, 1], abs=1e-3)
        assert out[:, :, 3].min() == pytest.approx(1.0, abs=1e-3)  # No dark seams

    def test_zDefocusSpeed(self):
        import time
        from meshroom.live import defocus
        rgba = np.random.default_rng(0).random((1080, 1920, 4), dtype=np.float32)
        radius = np.tile(np.linspace(-30, 30, 1920, dtype=np.float32), (1080, 1))
        start = time.perf_counter()
        defocus.defocus(rgba, radius, 30.0, 8, lambda r: defocus.bokehKernel(r, blades=6))
        assert time.perf_counter() - start < 8.0

    def test_evaluatorKeepsTheLayersOfTheRenderedNodes(self):
        graph, red, blue, grade, merge = constantGraph()
        evaluator = Evaluator()
        assert evaluator.layerNames(red._uid) is None
        evaluator.render(buildPlan(merge), 1)
        assert evaluator.layerNames(red._uid) == ["rgba"]

    def test_zDefocusUniformBlurIsAnEvenDisc(self):
        from meshroom.live import defocus
        rgba = np.zeros((41, 41, 4), np.float32)
        rgba[20, 20] = 1.0
        out = defocus.defocus(rgba, np.full((41, 41), 3.0, np.float32), 50.0, 8, lambda r: defocus.bokehKernel(r))
        # The point becomes a disc of radius 3: no sharp center left
        assert out[20, 20, 0] < 0.1 and out[20, 22, 0] == pytest.approx(out[20, 20, 0], rel=0.1)


class TestRound8:

    def test_openRotoCurveIsAStroke(self):
        from meshroom.live import BBox, roto
        line = {"name": "c", "closed": False, "width": 4.0,
                "points": [{"x": 10, "y": 20}, {"x": 50, "y": 20}]}
        matte = roto.shapesMatte([line], BBox(0, 0, 64, 40))
        assert matte.sum() == pytest.approx(40 * 4 + np.pi * 4, rel=0.03)  # Segment + round caps
        row = 40 - 1 - 20  # Nuke y 20 (row of the pixel [20, 21))
        assert matte[row, 30] == pytest.approx(1.0, abs=1e-3) and matte[row - 4, 30] == 0.0
        # Closed (the default): a polygon of no area
        assert roto.shapesMatte([dict(line, closed=True)], BBox(0, 0, 64, 40)).sum() == pytest.approx(0.0, abs=1e-6)
        # A bent open curve: no gap at the joint
        bent = dict(line, points=[{"x": 10, "y": 10}, {"x": 30, "y": 30}, {"x": 50, "y": 10}])
        assert roto.shapesMatte([bent], BBox(0, 0, 64, 40))[40 - 1 - 30, 30] > 0.9

    def test_frameRangeFollowsTheFirstRead(self, tmp_path):
        from meshroom.live.frameRange import followedFrameRange
        ranges = {}

        def infoOf(node):
            first, last = ranges.get(node.name, (None, None))
            return ImageInfo(10, 10, first, last)

        graph = Graph("live")
        assert followedFrameRange(graph, infoOf, None) == (None, None)
        read = graph.addNewNode("Read")
        value, uid = followedFrameRange(graph, infoOf, None)
        assert value == {"source": read.name}  # No file yet: no range, but this Read is followed
        graph.setFrameRange(value)
        ranges[read.name] = (1001, 1100)
        read.attribute("file").value = "/plate.####.tif"  # Changes its uid
        value, uid = followedFrameRange(graph, infoOf, uid)
        assert value == {"source": read.name, "first": 1001, "last": 1100}
        graph.setFrameRange(value)
        # A second Read does not change the range
        other = graph.addNewNode("Read")
        ranges[other.name] = (1, 50)
        assert followedFrameRange(graph, infoOf, uid)[0] is None
        # Entered by hand: it stays
        graph.setFrameRange({"first": 5, "last": 8, "source": ""})
        ranges[read.name] = (1, 10)
        read.attribute("file").value = "/other.####.exr"
        assert followedFrameRange(graph, infoOf, uid)[0] is None
        # Saved with the graph
        path = str(tmp_path / "range.mg")
        graph.save(path)
        assert loadGraph(path).frameRange == {"first": 5, "last": 8, "source": ""}

    def test_cameraTrackerPreviewAndMask(self):
        from meshroom.live import camera
        rng = np.random.default_rng(3)
        texture = rng.random((140, 200)).astype(np.float32)
        # Blurred noise: corners to track; the next frame moves 3 pixels to the right
        kernel = np.ones(5, np.float32) / 5
        texture = np.apply_along_axis(lambda r: np.convolve(r, kernel, "same"), 1, texture)
        texture = np.apply_along_axis(lambda c: np.convolve(c, kernel, "same"), 0, texture)

        def frame(shift):
            rgba = np.zeros((140, 200, 4), np.float32)
            rgba[:, :, :3] = np.roll(texture, shift, axis=1)[:, :, np.newaxis]
            rgba[:, :, 3] = 1
            return ImageBuffer.fromRGBA(rgba, (200, 140))

        features = camera.previewFeatures(frame(0), frame(3), 20, 11, 31, 0.7)
        assert len(features) > 5
        matched = [f for f in features if f["score"] is not None and f["score"] > 0.9]
        assert len(matched) > 0.8 * len(features)
        assert all(abs(f["nx"] - f["x"] - 3) < 1.01 for f in matched)
        # Mask: only the right half
        mask = np.zeros((140, 200), np.float32)
        mask[:, 100:] = 1
        masked = camera.previewFeatures(frame(0), None, 20, 11, 31, 0.7, mask)
        assert masked and all(f["x"] >= 100 for f in masked)
        drawn = camera.drawPreview(frame(0).rgba(), frame(0).bbox, masked, 11, 31, 0.7, mask)
        assert drawn[:, :80, 0].max() <= texture.max() * 0.35 + 1e-6  # Masked out: darker (away from the search squares)
        # Autotrack with a mask
        tracks = camera.autoTrack(lambda f: frame(3 * (f - 1)), [1, 2, 3], count=20, pattern=11, search=31,
                                  maskAt=lambda f, bbox: mask)
        assert tracks and all(position[0] >= 97 for track in tracks for position in track["keys"].values())

    @staticmethod
    def _texture(width=160, height=120, seed=5):
        rng = np.random.default_rng(seed)
        texture = rng.random((height, width)).astype(np.float32)
        kernel = np.ones(7, np.float32) / 7
        for axis in (0, 1):
            texture = np.apply_along_axis(lambda r: np.convolve(r, kernel, "same"), axis, texture)
        return texture

    @staticmethod
    def _shiftedImage(texture, dx, dy):
        """ The texture moved by (dx, dy) pixels (Nuke: dy up), bilinear, as an ImageBuffer. """
        from meshroom.live import flow
        h, w = texture.shape
        rows, columns = np.mgrid[0:h, 0:w].astype(np.float32)
        values = flow.sampleRows(texture, rows + dy, columns - dx)
        rgba = np.zeros((h, w, 4), np.float32)
        rgba[:, :, :3] = values[:, :, np.newaxis]
        rgba[:, :, 3] = 1
        return ImageBuffer.fromRGBA(rgba, (w, h))

    def test_opticalFlowOfATranslation(self):
        from meshroom.live import flow
        texture = self._texture()
        a, b = self._shiftedImage(texture, 0, 0), self._shiftedImage(texture, 3.0, 2.0)
        field = flow.computeFlow(a, b)
        interior = field[20:-20, 20:-20]
        assert np.median(interior[:, :, 0]) == pytest.approx(3.0, abs=0.15)
        assert np.median(interior[:, :, 1]) == pytest.approx(2.0, abs=0.15)
        # Half way: the image moved by half the motion
        backward = flow.computeFlow(b, a)
        middle = flow.interpolate(a, b, field, backward, 0.5).rgba()[20:-20, 20:-20, 0]
        expected = self._shiftedImage(texture, 1.5, 1.0).rgba()[20:-20, 20:-20, 0]
        blend = 0.5 * (a.rgba()[20:-20, 20:-20, 0] + b.rgba()[20:-20, 20:-20, 0])
        assert np.abs(middle - expected).mean() < 0.3 * np.abs(blend - expected).mean()

    def test_opticalFlowNodeAndRetimeMotion(self, tmp_path):
        from meshroom.live import flow
        texture = self._texture()
        for frame in range(1, 6):
            imageIO.writeImage(imageIO.framePath(str(tmp_path / "plate.####.tif"), frame),
                               self._shiftedImage(texture, 2.0 * frame, 0), "float", "linear")
        graph = Graph("live")
        read = graph.addNewNode("Read", file=str(tmp_path / "plate.####.tif"), colorspace="linear")
        vectors = graph.addNewNode("OpticalFlow")
        retime = graph.addNewNode("Retime", speed=0.5, filter="motion")
        graph.addEdge(read.output, vectors.input)
        graph.addEdge(read.output, retime.input)
        graph.update()
        evaluator = Evaluator()
        image = evaluator.render(buildPlan(vectors), 2)
        forward = flow.flowLayer(image, flow.FORWARD)[20:-20, 20:-20]
        backward = flow.flowLayer(image, flow.BACKWARD)[20:-20, 20:-20]
        assert np.median(forward[:, :, 0]) == pytest.approx(2.0, abs=0.15)
        assert np.median(backward[:, :, 0]) == pytest.approx(-2.0, abs=0.15)
        # Retime at half speed: output frame 2 is the input frame 1.5 (computed on the fly, then with the vectors)
        expected = self._shiftedImage(texture, 3.0, 0).rgba()[20:-20, 20:-20, 0]
        onTheFly = evaluator.render(buildPlan(retime), 2).rgba()[20:-20, 20:-20, 0]
        assert np.abs(onTheFly - expected).mean() < 0.02
        graph.addEdge(vectors.output, retime.vectors)
        retime.attribute("motionVectors").value = "vectors input"
        graph.update()
        withVectors = evaluator.render(buildPlan(retime), 2).rgba()[20:-20, 20:-20, 0]
        assert np.abs(withVectors - expected).mean() < 0.02

    def test_rotoShapeMovedAlongTheFlow(self):
        from meshroom.live import roto
        shape = {"name": "s", "points": [{"x": 10, "y": 10, "inX": 0, "inY": 0, "outX": 5, "outY": 0},
                                         {"x": 20, "y": 10}, {"x": 15, "y": 20}]}
        moved = roto.advectShape(shape, 1, 2, lambda x, y: (2.0, -1.0))
        assert [key["frame"] for key in moved["keys"]] == [1, 2]
        first = moved["keys"][1]["points"][0]
        assert (first["x"], first["y"], first["outX"]) == (12.0, 9.0, 5.0)
        assert roto.pointsAtFrame(moved, 1.5)[1]["x"] == pytest.approx(21.0)

    def test_cameraTrackerPreviewNode(self):
        graph = Graph("live")
        checker = graph.addNewNode("CheckerBoard", width=200, height=140, boxSize=20.0)
        tracker = graph.addNewNode("CameraTracker", preview=True, features=10)
        graph.addEdge(checker.output, tracker.input)
        graph.update()
        out = Evaluator().render(buildPlan(tracker), 1).rgba()
        source = Evaluator().render(buildPlan(checker), 1).rgba()
        assert not np.allclose(out, source)  # Features drawn
        tracker.attribute("preview").value = False
        graph.update()
        assert np.allclose(Evaluator().render(buildPlan(tracker), 1).rgba(), source)

    def test_rotoFlowTrackingKeepsTheStartFrame(self):
        from meshroom.live import roto
        points = lambda x: [{"x": x, "y": 0}, {"x": x + 10, "y": 0}, {"x": x, "y": 10}]
        shape = {"name": "s", "keys": [{"frame": 1, "points": points(0)}, {"frame": 21, "points": points(20)}]}
        moved = roto.advectShape(shape, 11, 12, lambda x, y: (1.0, 0.0))
        assert roto.pointsAtFrame(moved, 11)[0]["x"] == pytest.approx(10.0)  # Unchanged where it starts
        assert roto.pointsAtFrame(moved, 12)[0]["x"] == pytest.approx(11.0)

    def test_retimeMotionOfAnImageWithoutAlpha(self):
        from meshroom.live import flow
        texture = self._texture()
        rgb = np.repeat(texture[:, :, np.newaxis], 3, axis=2)
        image0 = ImageBuffer.fromRGBA(rgb, (160, 120))
        image1 = ImageBuffer.fromRGBA(np.roll(rgb, 2, axis=1), (160, 120))
        assert image0.channels("rgba") == ("R", "G", "B")
        forward, backward = flow.computeFlow(image0, image1), flow.computeFlow(image1, image0)
        middle = flow.interpolate(image0, image1, forward, backward, 0.5)
        assert middle.channels("rgba") == ("R", "G", "B", "A") and middle.rgba().shape == (120, 160, 4)

    def test_featuresAreFoundBesideABrightLight(self):
        from meshroom.live import camera
        texture = self._texture(400, 300) * 0.02  # A dark plate
        texture[140:160, 190:210] = 50.0  # A light
        features = camera.detectFeatures(texture, BBox(0, 0, 400, 300), 50, minDistance=15, margin=20)
        assert len(features) > 20


class TestLensDistortion:

    def test_modelsAreInvertedExactly(self):
        from meshroom.live import lens
        x, y = np.meshgrid(np.linspace(-0.8, 0.8, 9), np.linspace(-0.5, 0.5, 7))
        cases = [(lens.CLASSIC, [0.05, 1.3, 0.01, -0.02, 0.01], 1.0),
                 (lens.RADIAL4, [-0.04, 0.002, -0.001, 0.01, 0.0005, 0.0003, 20.0, 0.01], 1.0),
                 (lens.ANAMORPHIC4, [0.03, 0.02, 0.01, -0.01, 0.005, 0.004, 0.001, 0.002, 0.0005, 0.0004, 1.5, 1.01, 0.99],
                  2.0)]
        for model, params, pa in cases:
            ux, uy = lens.undistortDn(model, params, x, y, pa)
            dx, dy = lens.distortDn(model, params, ux, uy, pa)
            assert np.abs(dx - x).max() < 1e-8 and np.abs(dy - y).max() < 1e-8, model

    def test_radialDistortionBarrel(self):
        from meshroom.live import lens
        # Barrel distortion (negative degree 2): the undistorted position is closer to the center
        ux, uy = lens.undistortDn(lens.RADIAL4, [-0.1, 0, 0, 0, 0, 0, 0, 0], np.array([0.5]), np.array([0.0]))
        assert ux[0] == pytest.approx(0.5 * (1 - 0.1 * 0.25)) and uy[0] == 0.0

    def test_lensDistortionNodeAndNukeImport(self, tmp_path):
        script = tmp_path / "lens.nk"
        script.write_text("""
LD_3DE4_Radial_Standard_Degree_4 {
 direction undistort
 tde4_focal_length_cm 3.5
 tde4_filmback_width_cm 3.6
 tde4_filmback_height_cm 2.4
 tde4_lens_center_offset_x_cm 0.01
 tde4_lens_center_offset_y_cm 0
 tde4_pixel_aspect 1
 "Distortion - Degree 2" -0.05
 "Quartic Distortion - Degree 4" 0.002
 name LD_3DE4_1
}
""")
        from meshroom.live import lens
        values = lens.parseNukeLensNode(script.read_text())
        assert values["model"] == lens.RADIAL4 and values["distortionDegree2"] == -0.05
        assert values["filmbackWidth"] == 3.6 and values["lensCenterOffsetX"] == 0.01
        graph = Graph("live")
        checker = graph.addNewNode("CheckerBoard", width=120, height=80, boxSize=10.0)
        node = graph.addNewNode("LensDistortion")
        graph.addEdge(checker.output, node.input)
        for name, value in values.items():
            node.attribute(name).value = value
        node.attribute("direction").value = "STMap undistort"
        graph.update()
        stmap = Evaluator().render(buildPlan(node), 1).rgba()
        # Barrel: the corners of the undistorted image come from further out in the plate... with a negative
        # coefficient the distorted position of a corner is beyond it
        assert stmap[0, 0, 0] < 0.5 / 120 and stmap[40, 60, 0] == pytest.approx(60.5 / 120, abs=0.01)
        node.attribute("direction").value = "undistort"
        graph.update()
        undistorted = Evaluator().render(buildPlan(node), 1).rgba()
        node.attribute("direction").value = "distort"
        graph.update()
        assert undistorted.shape == (80, 120, 4)
        # Distort then undistort: back to the checkerboard in the center
        distort = graph.addNewNode("LensDistortion")
        graph.addEdge(node.output, distort.input)
        for name, value in values.items():
            distort.attribute(name).value = value
        distort.attribute("direction").value = "undistort"
        graph.update()
        roundTrip = Evaluator().render(buildPlan(distort), 1).rgba()
        source = Evaluator().render(buildPlan(checker), 1).rgba()
        assert np.abs(roundTrip[20:60, 30:90] - source[20:60, 30:90]).mean() < 0.05

    def test_nukeImportDefaultsCurvesAndUnsupportedModels(self):
        from meshroom.live import lens
        values = lens.parseNukeLensNode("""LD_3DE4_Anamorphic_Standard_Degree_4 {
 "Cx02 - Degree 2" {{curve x1001 0.01 0.02 x1010 0.05}}
 tde4_pixel_aspect 2
}""")
        assert values["model"] == lens.ANAMORPHIC4 and values["squeezeX"] == 1.0 and values["cy02Degree2"] == 0.0
        assert values["curves"]["cx02Degree2"] == [[1001.0, 0.01], [1002.0, 0.02], [1010.0, 0.05]]
        assert values["pixelAspect"] == 2.0
        with pytest.raises(ValueError):
            lens.parseNukeLensNode("LD_3DE4_Some_Future_Model {\n}")

    def test_undistortWithoutSourceIsBlack(self):
        from meshroom.live import lens
        positions = lens.sourcePositions(lens.RADIAL4, [-0.3, 0, 0, 0, 0, 0, 0, 0], {}, 96, 54, "undistort")
        assert np.isfinite(positions).all()
        # The corners have no distorted source: far outside
        assert abs(positions[0, 0, 0]) > 1000 and 40 < positions[27, 48, 0] < 56

    def test_retimeSavedWithVectorsUsesThem(self, tmp_path):
        import json
        graph = Graph("live")
        checker = graph.addNewNode("CheckerBoard")
        flowNode = graph.addNewNode("OpticalFlow")
        retime = graph.addNewNode("Retime", filter="motion")
        graph.addEdge(checker.output, flowNode.input)
        graph.addEdge(checker.output, retime.input)
        graph.addEdge(flowNode.output, retime.vectors)
        path = str(tmp_path / "old.mg")
        graph.save(path)
        with open(path) as f:
            data = json.load(f)
        data["graph"][retime.name]["inputs"].pop("motionVectors", None)
        data["header"]["nodesVersions"]["Retime"] = "0.3"
        with open(path, "w") as f:
            json.dump(data, f)
        assert loadGraph(path).node(retime.name).attribute("motionVectors").value == "vectors input"

    def test_featuresAvoidTheGrainOfDarkAreas(self):
        from meshroom.live import camera
        rng = np.random.default_rng(1)
        lum = np.zeros((200, 400), np.float32)
        lum[:, :200] = 0.002 + 0.001 * rng.standard_normal((200, 200))  # Dark grain
        texture = TestRound8._texture(200, 200) * 0.25 + 0.05
        lum[:, 200:] = texture + 0.002 * rng.standard_normal((200, 200))
        features = camera.detectFeatures(lum, BBox(0, 0, 400, 200), 30, minDistance=15, margin=15)
        assert sum(1 for x, _ in features if x >= 200) > 0.8 * len(features)


class TestCameraTracksInOtherNodes:

    def test_motionBlurWithCameraTrackerTracks(self):
        import json
        graph = Graph("live")
        square = graph.addNewNode("Shape", shape="square", width=80, height=40)
        square.attribute("shapeSize").value = {"x": 60.0, "y": 10.0}
        cameraTracker = graph.addNewNode("CameraTracker")
        # Autotrack-like: features living on a few frames; the ones ended before frame 2 must not count as still
        tracks = [{"name": f"auto{i}", "auto": True, "keys": {"1": [10 + i, 20], "3": [30 + i, 20]}} for i in range(3)]
        tracks += [{"name": f"old{i}", "auto": True, "keys": {"-5": [50, 10], "0": [50, 10]}} for i in range(5)]
        cameraTracker.attribute("tracks").value = json.dumps(tracks)
        blur = graph.addNewNode("MotionBlur", shutter=1.0)
        graph.addEdge(square.output, cameraTracker.input)
        graph.addEdge(square.output, blur.input)
        graph.addEdge(cameraTracker.output, blur.tracker)
        graph.update()
        # 10 pixels per frame over a shutter of 1 frame: the bbox grows by 5 on both sides
        assert Evaluator().render(buildPlan(blur), 2).bbox == BBox(-5, 0, 85, 40)

    def test_trackerImportsTheTracksOfACameraTracker(self):
        import json
        from meshroom.live.actions import ActionContext
        from meshroom.nodes.compositing.TrackerNodes import Tracker
        imported = [{"name": "auto1", "keys": {"1": [1, 2]}}, {"name": "track1", "keys": {"1": [3, 4]}}]
        context = ActionContext({"tracks": json.dumps([{"name": "track1", "keys": {"1": [5, 6]}}]), "importReplace": False},
                                inputValues={"importFrom": {"tracks": json.dumps(imported)}})
        result = Tracker().runAction("importTracks", context)
        assert [t["name"] for t in json.loads(result["tracks"])] == ["track1", "auto1", "track1_2"]
        assert result["_disconnect"] == ["importFrom"]
        with pytest.raises(RuntimeError):
            Tracker().runAction("importTracks", ActionContext({"tracks": "[]", "importReplace": False}))


_NUKE_SCRIPT = r"""#! /usr/local/Nuke14.0v5/libnuke-14.0.5.so -nx
version 14.0 v5
define_window_layout_xml {<?xml version="1.0" encoding="UTF-8"?>
<layout version="1.0"/>
}
Root {
 inputs 0
 name /shots/test.nk
 first_frame 1001
 last_frame 1100
 format "2048 1152 0 0 2048 1152 1 2K_Super_35"
}
Read {
 inputs 0
 file_type exr
 file /plates/plate.%04d.exr
 format "2048 1152 0 0 2048 1152 1 2K"
 first 1001
 last 1100
 origfirst 1001
 origlast 1100
 colorspace "ACES - ACEScg"
 name Read1
 xpos -40
 ypos -300
}
set N1 [stack 0]
Blur {
 size {{curve x1001 0 x1010 10} 4}
 name Blur1
 xpos -40
 ypos -200
}
Grade {
 white 1.2
 gamma {1.1 1.1 1.1 1}
 name Grade1
 xpos -40
 ypos -150
}
Constant {
 inputs 0
 color {1 0 0 1}
 name Constant1
 xpos 100
 ypos -300
}
Transform {
 translate {10 20}
 rotate 5
 scale 2
 center {1024 576}
 name Transform1
 xpos 100
 ypos -200
}
Merge2 {
 inputs 2
 operation plus
 name Merge1
 xpos -40
 ypos -100
}
push $N1
clone node7f1|Blur|21 Blur {
 size {{Blur1.size}}
 name Blur2
 xpos 200
 ypos -200
}
set C7f1 [stack 0]
SomeGizmo {
 strength 3
 name SomeGizmo1
 xpos 200
 ypos -150
}
clone $C7f1 {
 xpos 300
 ypos -150
}
Viewer {
 frame 1001
 name Viewer1
 xpos 300
 ypos 0
}
"""


class TestNukeImport:

    def test_parseScript(self):
        from meshroom.live import nuke
        nodes, root = nuke.parseScript(_NUKE_SCRIPT)
        assert root["first_frame"] == 1001 and nuke.rootFrameRange(root) == (1001, 1100)
        byName = {node.name: node for node in nodes}
        assert byName["Blur1"].inputs[0] is byName["Read1"]
        merge = byName["Merge1"]
        # The top of the stack is input 0 (B): the Transform branch; A is the Grade branch
        assert merge.inputs[0] is byName["Transform1"] and merge.inputs[1] is byName["Grade1"]
        assert byName["Blur2"].inputs[0] is byName["Read1"]
        size = byName["Blur1"].knobs["size"]
        assert isinstance(size[0], nuke.Curve) and size[0].keys == [[1001.0, 0.0], [1010.0, 10.0]] and size[1] == 4
        assert isinstance(byName["Blur2"].knobs["size"], nuke.Expression)
        clones = [node for node in nodes if node.cloneOf is not None]
        assert len(clones) == 1 and clones[0].inputs[0] is byName["SomeGizmo1"]

    def test_importScript(self):
        import json
        from meshroom.live import nuke
        graph = Graph("live")
        nodes, report, root = nuke.importScript(graph, _NUKE_SCRIPT)
        byLabel = {node.label: node for node in nodes}
        read = byLabel["Read1"]
        assert read.attribute("file").value == "/plates/plate.####.exr"
        blur = byLabel["Blur1"]
        assert json.loads(blur.internalAttribute("animation").value)["size"] == [[1001, {"x": 0.0, "y": 4.0}],
                                                                                [1010, {"x": 10.0, "y": 4.0}]]
        assert byLabel["Grade1"].attribute("gain").getSerializedValue() == pytest.approx({"r": 1.2, "g": 1.2, "b": 1.2, "a": 1.2})
        assert byLabel["Grade1"].attribute("gamma").getSerializedValue() == pytest.approx({"r": 1.1, "g": 1.1, "b": 1.1, "a": 1.0})
        transform = byLabel["Transform1"]
        assert transform.attribute("translate").getSerializedValue() == {"x": 10.0, "y": 20.0}
        assert transform.attribute("scale").getSerializedValue() == {"x": 2.0, "y": 2.0}
        assert transform.attribute("centerOnImage").value is False
        merge = byLabel["Merge1"]
        assert merge.attribute("operation").value == "plus"
        assert merge.attribute("B").inputLink.node is transform
        assert merge.attribute("A").inputLink.node is byLabel["Grade1"]
        constant = byLabel["Constant1"]
        assert constant.attribute("width").value == 2048 and constant.attribute("rgba").getSerializedValue()["r"] == 1.0
        # Link between knobs: Blur2.size = Blur1.size
        assert byLabel["Blur2"].attribute("size").inputLink is blur.attribute("size")
        unsupported = byLabel["SomeGizmo1"]
        assert unsupported.nodeType == "NukeUnsupported" and unsupported.attribute("nukeClass").value == "SomeGizmo"
        clone = next(node for node in nodes if node.nodeType == "Clone")
        assert clone.attribute("source").inputLink.node is byLabel["Blur2"]
        assert any("SomeGizmo" in line for line in report)
        assert all(node.nodeType != "Viewer" for node in nodes)
        # The imported graph renders (the Read has no file here: render the Constant branch)
        graph.update()
        assert Evaluator().render(buildPlan(transform), 1).rgba().shape[2] == 4

    def test_importedNodesSurviveCopyPaste(self):
        import json
        from meshroom.live import nuke
        tmp = Graph("")
        nuke.importScript(tmp, _NUKE_SCRIPT)
        data = json.loads(json.dumps(tmp.serializePartial(list(tmp.nodes))))
        pasted = Graph("")
        pasted._deserialize(data)
        target = Graph("live")
        nodes = target.importGraphContent(pasted)
        byLabel = {node.label: node for node in nodes}
        assert json.loads(byLabel["Blur1"].internalAttribute("animation").value)["size"][1][0] == 1010
        assert byLabel["Blur2"].attribute("size").inputLink is byLabel["Blur1"].attribute("size")
        clone = next(node for node in nodes if node.nodeType == "Clone")
        assert clone.attribute("source").inputLink.node is byLabel["Blur2"]


class TestProxy:

    def test_proxyRenderIsTheFullRenderReduced(self, tmp_path):
        from meshroom.live import evaluator as ev
        rgba = np.zeros((64, 96, 4), np.float32)
        rgba[16:48, 24:72] = 1.0
        imageIO.writeImage(str(tmp_path / "plate.tif"), ImageBuffer.fromRGBA(rgba, (96, 64)), "float", "linear")
        graph = Graph("live")
        read = graph.addNewNode("Read", file=str(tmp_path / "plate.tif"), colorspace="linear")
        transform = graph.addNewNode("Transform")
        transform.attribute("translate").value = {"x": 8.0, "y": 4.0}
        blur = graph.addNewNode("Blur")
        blur.attribute("size").value = {"x": 4.0, "y": 4.0}
        graph.addEdge(read.output, transform.input)
        graph.addEdge(transform.output, blur.input)
        graph.update()
        evaluator = Evaluator()
        full = evaluator.render(buildPlan(blur), 1)
        half = evaluator.render(buildPlan(blur, proxy=0.5), 1)
        assert half.format == (48, 32) and full.format == (96, 64)
        reduced = ops.downscaleImage(full, 2)
        common = half.bbox.intersection(reduced.bbox)
        a = ops.alignToBBox(half.rgba(), half.bbox, common)[:, :, 3]
        b = ops.alignToBBox(reduced.rgba(), reduced.bbox, common)[:, :, 3]
        assert np.abs(a - b).mean() < 0.02
        # Cached apart from the full resolution
        assert evaluator.cache.get((ev.proxyKey(blur._uid, 0.5), 1)) is half

    def test_rotoShapesAtProxyScale(self):
        import json
        graph = Graph("live")
        rotoNode = graph.addNewNode("Roto", width=64, height=64)
        square = [{"x": 16, "y": 16}, {"x": 48, "y": 16}, {"x": 48, "y": 48}, {"x": 16, "y": 48}]
        rotoNode.attribute("shapes").value = json.dumps([{"name": "s", "closed": True, "points": square}])
        graph.update()
        half = Evaluator().render(buildPlan(rotoNode, proxy=0.5), 1)
        assert half.format == (32, 32)
        assert half.rgba()[:, :, 3].sum() == pytest.approx(16 * 16, rel=0.02)


class TestRegionOfInterest:

    def _graph(self):
        graph = Graph("live")
        checker = graph.addNewNode("CheckerBoard", width=200, height=120, boxSize=10.0)
        transform = graph.addNewNode("Transform")
        transform.attribute("translate").value = {"x": 3.5, "y": -2.25}
        transform.attribute("rotate").value = 10.0
        blur = graph.addNewNode("Blur")
        blur.attribute("size").value = {"x": 6.0, "y": 6.0}
        grade = graph.addNewNode("Grade", gain={"r": 2.0, "g": 2.0, "b": 2.0, "a": 2.0})
        graph.addEdge(checker.output, transform.input)
        graph.addEdge(transform.output, blur.input)
        graph.addEdge(blur.output, grade.input)
        graph.update()
        return graph, checker, transform, blur, grade

    def test_regionRenderMatchesTheFullRender(self):
        graph, checker, transform, blur, grade = self._graph()
        region = BBox(60, 30, 110, 70)
        full = Evaluator().render(buildPlan(grade), 1)
        evaluator = Evaluator()
        part = evaluator.render(buildPlan(grade), 1, region=region)
        assert part.bbox == region
        expected = ops.alignToBBox(full.rgba(), full.bbox, region)
        assert np.abs(part.rgba() - expected).max() < 1e-4
        # Upstream, the blur only computed what the region needs
        blurred = evaluator.cache.get((blur._uid, 1))
        assert blurred.bbox.width < 80
        # A whole render is not served by the region in cache
        whole = evaluator.render(buildPlan(grade), 1)
        assert whole.bbox == full.bbox
        # A region inside the cached one is
        again = evaluator.render(buildPlan(grade), 1, region=BBox(0, 0, 50, 50))
        assert again is whole


class _FakeOcioProcessor:
    def __init__(self, function):
        self.function = function

    def getDefaultCPUProcessor(self):
        return self

    def applyRGB(self, buffer):
        buffer[...] = self.function(buffer)


class _FakeOcioColorSpace:
    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name


class _FakeOcioConfig:
    """ A minimal stand-in for a PyOpenColorIO config: "lin" (scene_linear) and "log2" (log2 encoding). """
    roles = {"scene_linear": "lin", "color_picking": "log2"}
    functions = {("lin", "log2"): lambda v: np.log2(np.maximum(v, 1e-6)) / 8.0 + 0.5,
                 ("log2", "lin"): lambda v: 2.0 ** ((v - 0.5) * 8.0)}

    def getColorSpaceNames(self):
        return ["lin", "log2"]

    def getColorSpace(self, name):
        name = self.roles.get(name, name)
        return _FakeOcioColorSpace(name) if name in ("lin", "log2") else None

    def getProcessor(self, *args):
        if len(args) == 1:  # a display / view transform
            return _FakeOcioProcessor(lambda v: np.clip(v, 0, 1) ** 0.5)
        return _FakeOcioProcessor(self.functions[args])

    def getDisplays(self):
        return ["Screen"]

    def getViews(self, display):
        return ["Sqrt"]

    def getDefaultDisplay(self):
        return "Screen"

    def getDefaultView(self, display):
        return "Sqrt"


class TestColorManagement:

    @pytest.fixture(autouse=True)
    def builtinConfig(self, monkeypatch):
        from meshroom.live import color
        monkeypatch.delenv("OCIO", raising=False)
        color.setConfig(None)
        yield
        color.setConfig(None)

    def test_primariesMatrices(self):
        from meshroom.live import color
        acescgToRec709 = color.primariesMatrix("ap1", "rec709")
        np.testing.assert_allclose(acescgToRec709, [[1.70505, -0.62179, -0.08326],
                                                    [-0.13026, 1.14080, -0.01055],
                                                    [-0.02400, -0.12897, 1.15297]], atol=2e-3)
        aces2065ToRec709 = color.primariesMatrix("ap0", "rec709")
        np.testing.assert_allclose(aces2065ToRec709[0], [2.52169, -1.13413, -0.38756], atol=2e-3)
        # White stays white
        np.testing.assert_allclose(acescgToRec709.sum(axis=1), 1.0, atol=1e-6)

    def test_builtinRoundTrips(self):
        from meshroom.live import color
        config = color.config()
        assert not config.isOcio and config.workingSpace() == "linear"
        values = np.array([[[0.001, 0.18, 0.5], [1.0, 2.0, 4.0]]], dtype=np.float32)
        for name in config.colorspaceNames():
            encoded = config.convert(values, "linear", name)
            np.testing.assert_allclose(config.convert(encoded, name, "linear"), values, rtol=2e-3, atol=1e-5,
                                       err_msg=name)

    def test_builtinKnownValues(self):
        from meshroom.live import color
        config = color.config()

        def encode(name, value):
            return float(config.convert(np.full((1, 1, 3), value, np.float32), "linear", name)[0, 0, 0])
        assert encode("sRGB", 0.214041) == pytest.approx(0.5, abs=1e-4)
        assert encode("Cineon", 1.0) == pytest.approx(685.0 / 1023.0, abs=1e-5)
        assert encode("Cineon", 0.0) == pytest.approx(95.0 / 1023.0, abs=1e-5)
        assert encode("ACEScct", 0.18) == pytest.approx(0.4135884, abs=1e-5)
        assert encode("Gamma2.2", 0.5) == pytest.approx(0.5 ** (1 / 2.2), abs=1e-5)
        # LogC encodes ARRI Wide Gamut 3: a grey stays grey
        assert encode("AlexaV3LogC", 0.18) == pytest.approx(0.391007, abs=1e-4)
        assert config.resolve("ACES - ACEScg") == "ACEScg" and config.resolve("scene_linear") == "linear"
        assert config.resolve("nope") is None
        assert config.curve("linear", "sRGB") is not None and config.curve("linear", "ACEScg") is None

    def test_readWriteColorspaces(self, tmp_path):
        from meshroom.live import color
        rgba = np.random.default_rng(1).uniform(0.0, 2.0, (4, 6, 4)).astype(np.float32)
        rgba[:, :, 3] = 1.0
        image = ImageBuffer.fromRGBA(rgba, (6, 4))
        for name in ("Cineon", "ACEScg", "rec709"):
            path = str(tmp_path / f"{name}.tif")
            imageIO.writeImage(path, image, "float", name)
            raw = imageIO.readImage(path, "linear").rgba()
            expected = color.config().convert(rgba[:, :, :3], "linear", name)
            np.testing.assert_allclose(raw[:, :, :3], expected, rtol=1e-5, atol=1e-6, err_msg=name)
            np.testing.assert_allclose(raw[:, :, 3], 1.0)
            np.testing.assert_allclose(imageIO.readImage(path, name).rgba(), rgba, rtol=1e-4, atol=1e-5, err_msg=name)
        # 16-bit integer: the curve goes through the lookup table
        path = str(tmp_path / "cineon16.tif")
        clipped = ImageBuffer.fromRGBA(np.clip(rgba, 0.0, 1.0), (6, 4))
        imageIO.writeImage(path, clipped, "uint16", "Cineon")
        np.testing.assert_allclose(imageIO.readImage(path, "Cineon").rgba()[:, :, :3], clipped.rgba()[:, :, :3],
                                   rtol=2e-3, atol=2e-4)
        with pytest.raises(ValueError):
            imageIO.readImage(path, "nope")

    def test_ocioColorSpaceNode(self):
        from meshroom.live import color
        graph = Graph("live")
        constant = graph.addNewNode("Constant", width=4, height=2)
        setColor(constant, r=0.18, g=0.5, b=1.0, a=0.5)
        convert = graph.addNewNode("OCIOColorSpace", inColorspace="linear", outColorspace="sRGB")
        graph.addEdge(constant.output, convert.input)
        graph.update()
        evaluator = Evaluator()
        pixel = evaluator.render(buildPlan(convert), 1).rgba()[0, 0]
        np.testing.assert_allclose(pixel[:3], color.linearToSrgb(np.array([0.18, 0.5, 1.0])), rtol=1e-5)
        assert pixel[3] == pytest.approx(0.5)
        convert.attribute("inColorspace").value = "ACEScg"
        convert.attribute("outColorspace").value = "linear"
        graph.update()
        pixel = evaluator.render(buildPlan(convert), 1).rgba()[0, 0]
        np.testing.assert_allclose(pixel[:3], color.primariesMatrix("ap1", "rec709") @ [0.18, 0.5, 1.0], rtol=1e-4)
        convert.attribute("outColorspace").value = "ACEScg"
        graph.update()
        np.testing.assert_allclose(evaluator.render(buildPlan(convert), 1).rgba()[0, 0], [0.18, 0.5, 1.0, 0.5], rtol=1e-6)

    def test_displayTransforms(self):
        from meshroom.live import color
        from meshroom.live.display import displayPixels
        config = color.config()
        assert "sRGB / Standard" in config.displayTransforms() and config.defaultDisplayTransform() == "sRGB / Standard"
        image = ImageBuffer.fromRGBA(np.array([[[0.18, 0.5, 2.0, 1.0], [0.0, 1.0, 0.25, 1.0]]], np.float32), (2, 1))
        default = displayPixels(image, "rgba", "rgb", 0.0, 1.0)
        assert np.array_equal(displayPixels(image, "rgba", "rgb", 0.0, 1.0,
                                            transform=config.displayTransform("sRGB / Standard")), default)
        raw = displayPixels(image, "rgba", "rgb", 0.0, 1.0, transform=config.displayTransform("sRGB / Raw"))
        assert list(raw[0, 0, :3]) == [46, 127, 255]
        p3 = displayPixels(image, "rgba", "rgb", 0.0, 1.0, transform=config.displayTransform("Display P3 / Standard"))
        assert not np.array_equal(p3, default)
        # A grey stays grey through the P3 primaries
        grey = ImageBuffer.fromRGBA(np.full((1, 1, 4), 0.18, np.float32), (1, 1))
        greyP3 = displayPixels(grey, "rgba", "rgb", 0.0, 1.0, transform=config.displayTransform("Display P3 / Standard"))
        assert len(set(greyP3[0, 0, :3])) == 1
        # A processor (OCIO) transform
        processed = displayPixels(image, "rgba", "rgb", 0.0, 1.0,
                                  transform=color.DisplayTransform(processor=lambda rgb: np.clip(rgb, 0, 1) ** 0.5))
        assert list(processed[0, 1, :3]) == [0, 255, 128]

    def test_ocioConfig(self, monkeypatch, tmp_path):
        import sys
        import types
        from meshroom.live import color
        fake = types.ModuleType("PyOpenColorIO")
        fake.Config = types.SimpleNamespace(CreateFromFile=lambda path: _FakeOcioConfig())
        fake.DisplayViewTransform = lambda **kwargs: kwargs
        monkeypatch.setitem(sys.modules, "PyOpenColorIO", fake)
        monkeypatch.setenv("OCIO", str(tmp_path / "config.ocio"))
        config = color.loadConfig()
        assert config.isOcio and config.colorspaceNames() == ["lin", "log2"]
        assert config.workingSpace() == "lin" and config.defaultSpace(True) == "log2"
        assert config.displayTransforms() == ["Screen / Sqrt"]
        values = np.array([[[0.25, 1.0, 4.0]]], np.float32)
        np.testing.assert_allclose(config.convert(values, "lin", "log2"), [[[0.25, 0.5, 0.75]]], rtol=1e-6)
        # Read: a float file in "log2" comes back linear
        path = str(tmp_path / "log.tif")
        imageIO.writeImage(path, ImageBuffer.constant(2, 2, (0.25, 0.5, 0.75, 1.0)), "float", "lin")
        np.testing.assert_allclose(imageIO.readImage(path, "log2").rgba()[0, 0, :3], [0.25, 1.0, 4.0], rtol=1e-5)
        with pytest.raises(ValueError):
            imageIO.readImage(path, "sRGB")
        transform = config.displayTransform("Screen / Sqrt")
        assert transform.processor is not None
        np.testing.assert_allclose(transform.processor(np.array([[0.25, 1.0, 4.0]], np.float32)), [[0.5, 1.0, 1.0]])
        # PyOpenColorIO missing: the builtin colorspaces
        monkeypatch.setitem(sys.modules, "PyOpenColorIO", None)
        assert not color.loadConfig().isOcio

    def test_nukeColorspaces(self):
        from meshroom.live import nuke
        script = """
Read {
 file /plates/a.####.exr
 colorspace "ACES - ACEScg"
 name Read1
}
OCIOColorSpace {
 in_colorspace scene_linear
 out_colorspace Cineon
 name OCIOColorSpace1
}
Read {
 file /plates/b.####.exr
 colorspace "default (sRGB)"
 name Read2
}
Read {
 file /plates/c.####.exr
 colorspace "Unknown Space"
 name Read3
}
"""
        graph = Graph("live")
        nodes, report, _ = nuke.importScript(graph, script)
        byLabel = {node.label: node for node in nodes}
        assert byLabel["Read1"].attribute("colorspace").value == "ACEScg"
        assert byLabel["Read2"].attribute("colorspace").value == "auto"
        assert byLabel["Read3"].attribute("colorspace").value == "auto"
        assert any("Unknown Space" in line for line in report)
        convert = byLabel["OCIOColorSpace1"]
        assert convert.nodeType == "OCIOColorSpace"
        assert convert.attribute("inColorspace").value == "linear"
        assert convert.attribute("outColorspace").value == "Cineon"
        assert convert.attribute("input").inputLink.node is byLabel["Read1"]


class TestGradePerChannel:

    def test_gradeValuesPerChannel(self):
        graph = Graph("live")
        constant = graph.addNewNode("Constant", width=4, height=2)
        setColor(constant, r=0.5, g=0.5, b=0.5, a=1.0)
        grade = graph.addNewNode("Grade", channels="rgba", gain={"r": 2.0, "g": 1.0, "b": 0.5, "a": 0.5},
                                 offset={"r": 0.0, "g": 0.1, "b": 0.0, "a": 0.0})
        graph.addEdge(constant.output, grade.input)
        graph.update()
        pixel = Evaluator().render(buildPlan(grade), 1).rgba()[0, 0]
        assert pixel.tolist() == pytest.approx([1.0, 0.6, 0.25, 0.5])
        # "all": the same values on the rgba layer
        grade.attribute("channels").value = "all"
        graph.update()
        assert Evaluator().render(buildPlan(grade), 1).rgba()[0, 0].tolist() == pytest.approx([1.0, 0.6, 0.25, 0.5])

    def test_gradeSavedWithFloatsIsUpgraded(self, tmp_path):
        import json
        graph = Graph("live")
        grade = graph.addNewNode("Grade")
        path = str(tmp_path / "old.mg")
        graph.save(path)
        with open(path) as f:
            data = json.load(f)
        nodeData = data["graph"][grade.name]
        nodeData["version"] = "0.2"
        nodeData["inputs"]["gain"] = 2.0
        nodeData["inputs"]["gamma"] = 1.5
        with open(path, "w") as f:
            json.dump(data, f)
        node = loadGraph(path).node(grade.name)
        assert node.attribute("gain").getSerializedValue() == {"r": 2.0, "g": 2.0, "b": 2.0, "a": 2.0}
        assert node.attribute("gamma").getSerializedValue() == {"r": 1.5, "g": 1.5, "b": 1.5, "a": 1.5}

    def test_nukeGradePerChannel(self):
        from meshroom.live import nuke
        script = """
Grade {
 white {1.2 1 0.8 1}
 add 0.1
 name Grade1
}
"""
        nodes, report, _ = nuke.importScript(Graph("live"), script)
        grade = nodes[0]
        assert grade.attribute("gain").getSerializedValue() == pytest.approx({"r": 1.2, "g": 1.0, "b": 0.8, "a": 1.0})
        assert grade.attribute("offset").getSerializedValue() == pytest.approx({"r": 0.1, "g": 0.1, "b": 0.1, "a": 0.1})
        assert not report


class TestLensModels:

    def test_newModelsRoundTrip(self):
        from meshroom.live import lens
        rng = np.random.default_rng(3)
        x, y = rng.uniform(-0.7, 0.7, 50), rng.uniform(-0.4, 0.4, 50)
        anamorphic6 = [0.03, 0.02, 0.01, -0.01, 0.005, 0.004, 0.001, 0.002, 0.0005, 0.0004,
                       0.002, -0.001, 0.0005, 0.0003, 0.0002, 0.0001, 0.0001, 0.0002, 2.0, 1.01, 0.99]
        cases = [(lens.ANAMORPHIC6, anamorphic6, 2.0, 0.0),
                 (lens.RESCALED4, anamorphic6[:10] + [2.0, 1.01, 0.99, 1.1], 1.0, 0.0),
                 (lens.FISHEYE8, [0.02, -0.01, 0.002, -0.0005], 1.0, 0.9)]
        for model, params, pa, focal in cases:
            assert len(params) == len(lens.MODEL_PARAMETERS[model])
            ux, uy = lens.undistortDn(model, params, x, y, pa, focal)
            dx, dy = lens.distortDn(model, params, ux, uy, pa, focalDn=focal)
            np.testing.assert_allclose(dx, x, atol=1e-6, err_msg=model)
            np.testing.assert_allclose(dy, y, atol=1e-6, err_msg=model)

    def test_modelsReduceToTheStandardDegree4(self):
        from meshroom.live import lens
        x, y = np.array([0.3, -0.5, 0.1]), np.array([0.2, 0.1, -0.4])
        degree4 = [0.03, 0.02, 0.01, -0.01, 0.005, 0.004, 0.001, 0.002, 0.0005, 0.0004, 10.0, 1.02, 0.98]
        expected = lens.undistortDn(lens.ANAMORPHIC4, degree4, x, y, 1.5)
        degree6 = degree4[:10] + [0.0] * 8 + degree4[10:]
        np.testing.assert_allclose(lens.undistortDn(lens.ANAMORPHIC6, degree6, x, y, 1.5), expected)
        np.testing.assert_allclose(lens.undistortDn(lens.RESCALED4, degree4 + [1.0], x, y, 1.5), expected)
        # Rescale: the polynomial is evaluated on the radius times Rescale
        rescaled = lens.undistortDn(lens.RESCALED4, degree4 + [2.0], x, y, 1.0)
        scaled = lens.undistortDn(lens.ANAMORPHIC4, degree4, x * 2.0, y * 2.0, 1.0)
        np.testing.assert_allclose(rescaled, np.array(scaled) / 2.0, atol=1e-12)

    def test_fisheyeProjection(self):
        from meshroom.live import lens
        focal = 0.8
        ux, uy = lens.undistortDn(lens.FISHEYE8, [0.0] * 4, np.array([0.5, 0.0]), np.array([0.0, 0.0]), 1.0, focal)
        # Equisolid angle: r = 2 f sin(a / 2) -> plain: f tan(a)
        angle = 2.0 * np.arcsin(0.5 / (2.0 * focal))
        assert ux[0] == pytest.approx(focal * np.tan(angle)) and uy[0] == 0.0 and ux[1] == 0.0
        with pytest.raises(ValueError):
            lens.sourcePositions(lens.FISHEYE8, [0.0] * 4, {}, 16, 9, "undistort")
        camera = {"filmbackWidth": 3.6, "filmbackHeight": 2.4, "focalLength": 1.2}
        assert lens.focalDn(camera, lens.filmback(16, 9, camera)) == pytest.approx(1.2 / (0.5 * np.hypot(3.6, 2.4)))
        positions = lens.sourcePositions(lens.FISHEYE8, [0.01, 0.0, 0.0, 0.0], camera, 32, 18, "undistort")
        assert np.isfinite(positions).all()

    def test_nukeFisheyeNode(self):
        from meshroom.live import lens, nuke
        script = """LD_3DE4_Radial_Fisheye_Degree_8 {
 tde4_focal_length_cm 1.2
 tde4_filmback_width_cm 3.6
 tde4_filmback_height_cm 2.4
 "Distortion - Degree 2" 0.02
 "Degree 8" -0.001
 name LD1
}
"""
        values = lens.parseNukeLensNode(script)
        assert values["model"] == lens.FISHEYE8 and values["focalLength"] == 1.2 and values["degree8"] == -0.001
        nodes, report, _ = nuke.importScript(Graph("live"), script)
        node = nodes[0]
        assert node.nodeType == "LensDistortion" and node.attribute("model").value == lens.FISHEYE8
        assert node.attribute("focalLength").value == pytest.approx(1.2)
        assert node.attribute("distortionDegree2").value == pytest.approx(0.02)


class TestHiddenListInputs:

    def test_hiddenInputsAfterRemoval(self):
        shift = desc.ImageOpNode.hiddenInputsAfterRemoval
        value = "input,inputs[0],inputs[2],inputs[3].x,mask,inputs2[3]"
        # Removing the element 1: the hidden 2 and 3 become 1 and 2; other inputs are kept
        assert shift(value, "inputs", 1) == "input,inputs[0],inputs[1],inputs[2].x,mask,inputs2[3]"
        # Removing a hidden element forgets it
        assert shift(value, "inputs", 0) == "input,inputs[1],inputs[2].x,mask,inputs2[3]"
        assert shift(value, "inputs", 5) == value
        assert shift("", "inputs", 0) == ""


class TestNukeStackOrder:

    def test_topOfTheStackIsInputZero(self):
        """ A template as written by Nuke: Grade (image + mask), Switch and Merge2 (B: background, A: plate). """
        from meshroom.live import nuke
        script = """
Read {
 inputs 0
 name MASK
}
Read {
 inputs 0
 name SRC
}
Grade {
 inputs 1+1
 white 0
 name Grade1
}
Read {
 inputs 0
 name BG
}
CheckerBoard2 {
 inputs 0
 name CheckerBoard1
}
Switch {
 inputs 2
 name Switch1
}
Merge2 {
 inputs 2
 name Merge1
}
"""
        nodes, _ = nuke.parseScript(script)
        byName = {node.name: node for node in nodes}
        assert byName["Grade1"].inputs == [byName["SRC"], byName["MASK"]]
        assert byName["Switch1"].inputs == [byName["CheckerBoard1"], byName["BG"]]
        assert byName["Merge1"].inputs == [byName["Switch1"], byName["Grade1"]]


class TestSwitch:

    def _graph(self):
        graph = Graph("live")
        red = graph.addNewNode("Constant", width=4, height=2)
        setColor(red, r=1.0)
        blue = graph.addNewNode("Constant", width=4, height=2)
        setColor(blue, b=1.0)
        switch = graph.addNewNode("Switch")
        graph.addEdge(red.output, switch.input)
        graph.addEdge(blue.output, switch.input1)
        graph.update()
        return graph, red, blue, switch

    def test_which(self):
        graph, red, blue, switch = self._graph()
        evaluator = Evaluator()
        assert evaluator.render(buildPlan(switch), 1).rgba()[0, 0].tolist() == [1, 0, 0, 1]
        switch.attribute("which").value = 1
        graph.update()
        assert evaluator.render(buildPlan(switch), 1).rgba()[0, 0].tolist() == [0, 0, 1, 1]
        # Not connected: transparent black
        switch.attribute("which").value = 3
        graph.update()
        image = evaluator.render(buildPlan(switch), 1)
        assert image.format == (4, 2) and image.rgba()[0, 0].tolist() == [0, 0, 0, 0]

    def test_expressionReadsOtherNodes(self):
        graph, red, blue, switch = self._graph()
        read = graph.addNewNode("Read")
        read.internalAttribute("label").value = "BG"
        switch.attribute("useExpression").value = True
        switch.attribute("expression").value = "1 if node('BG').file else 0"
        graph.update()
        evaluator = Evaluator()
        plan = buildPlan(switch)
        assert evaluator.render(plan, 1).rgba()[0, 0].tolist() == [1, 0, 0, 1]
        # The file of BG is set: the plan key changes (the cached image of the switch is not reused)
        read.attribute("file").value = "/plates/bg.####.exr"
        plan2 = buildPlan(switch)
        assert plan2.target != plan.target
        assert evaluator.render(plan2, 1).rgba()[0, 0].tolist() == [0, 0, 1, 1]
        # Nuke Python, frames and several lines
        switch.attribute("expression").value = "nuke.toNode('BG')['file'].value() and frame > 5 and 1 or 0"
        assert evaluator.render(buildPlan(switch), 1).rgba()[0, 0].tolist() == [1, 0, 0, 1]
        assert evaluator.render(buildPlan(switch), 6).rgba()[0, 0].tolist() == [0, 0, 1, 1]
        switch.attribute("expression").value = "if frame > 3:\n    result = 1\nelse:\n    result = 0"
        assert evaluator.render(buildPlan(switch), 4).rgba()[0, 0].tolist() == [0, 0, 1, 1]
        switch.attribute("expression").value = "().__class__"
        with pytest.raises(EvaluationError):
            evaluator.render(buildPlan(switch), 1)

    def test_nukeSwitchAndReformatBox(self):
        from meshroom.live import expressions, nuke
        assert expressions.nukePythonExpression(
            '"\\[python 1\\\\ if\\\\ nuke.toNode(\'BG\')\\\\\\[\'file\'\\\\].value()\\\\ else\\\\ 0]"') == \
            "1 if nuke.toNode('BG')['file'].value() else 0"
        script = r"""
Read {
 inputs 0
 name SRC
}
Read {
 inputs 0
 name BG
}
Reformat {
 type "to box"
 box_width {{SRC.width}}
 box_height {{SRC.height}}
 box_fixed true
 name Reformat2
}
CheckerBoard2 {
 inputs 0
 name CheckerBoard1
}
Switch {
 inputs 2
 which {{"\[python 1\\ if\\ nuke.toNode('BG')\\\['file'\\\].value()\\ else\\ 0]"}}
 name Switch1
}
"""
        nodes, report, _ = nuke.importScript(Graph("live"), script)
        byLabel = {node.label: node for node in nodes}
        switch = byLabel["Switch1"]
        assert switch.nodeType == "Switch" and switch.attribute("useExpression").value is True
        assert switch.attribute("expression").value == "1 if nuke.toNode('BG')['file'].value() else 0"
        assert switch.attribute("input").inputLink.node is byLabel["CheckerBoard1"]
        assert switch.attribute("input1").inputLink.node is byLabel["Reformat2"]
        assert byLabel["Reformat2"].attribute("formatInput").inputLink.node is byLabel["SRC"]
        assert not [line for line in report if "Switch1" in line or "Reformat2" in line]

    def test_inputNotChosenMayFail(self):
        """ A Read without file on the input that is not chosen does not make the switch fail. """
        graph, red, blue, switch = self._graph()
        read = graph.addNewNode("Read")
        graph.addEdge(read.output, switch.input1)
        graph.update()
        evaluator = Evaluator()
        plan = buildPlan(switch)
        assert evaluator.info(plan).width == 4
        assert evaluator.render(plan, 1).rgba()[0, 0].tolist() == [1, 0, 0, 1]

    def test_reformatToTheFormatOfAnInput(self):
        graph = Graph("live")
        plate = graph.addNewNode("Constant", width=8, height=4)
        small = graph.addNewNode("Constant", width=4, height=2)
        reformat = graph.addNewNode("Reformat", width=100, height=100)
        graph.addEdge(plate.output, reformat.input)
        graph.addEdge(small.output, reformat.formatInput)
        graph.update()
        assert Evaluator().render(buildPlan(reformat), 1).format == (4, 2)


class TestLensAnimationImport:

    def test_animatedLDNodeOfANukeScript(self):
        """ The animated knobs of a LD_3DE4 node become keys, used frame by frame. """
        import json
        from meshroom.live import nuke
        script = """
Constant {
 inputs 0
 format "64 36 0 0 64 36 1 small"
 color {0.5 0.5 0.5 1}
 name C1
}
LD_3DE4_Radial_Standard_Degree_4 {
 direction undistort
 tde4_focal_length_cm {{curve x1001 3.5 x1010 5}}
 tde4_filmback_width_cm 3.6
 tde4_filmback_height_cm 2.025
 "Distortion - Degree 2" {{curve x1001 -0.05 -0.06 x1010 -0.2}}
 name LD1
}
"""
        graph = Graph("live")
        nodes, report, _ = nuke.importScript(graph, script)
        lensNode = next(node for node in nodes if node.nodeType == "LensDistortion")
        animation = json.loads(lensNode.internalAttribute("animation").value)
        assert [key[0] for key in animation["distortionDegree2"]] == [1001, 1002, 1010]
        graph.update()
        evaluator = Evaluator()
        plan = buildPlan(lensNode)
        first, last = evaluator.render(plan, 1001).rgba(), evaluator.render(plan, 1010).rgba()
        assert np.abs(first - last).max() > 0.1


def _movingBlobSequence(folder, frames, speed=2.0, size=(64, 48)):
    """ A bright soft blob moving right by `speed` pixels per frame, written as float tifs. """
    width, height = size
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    for frame in frames:
        cx, cy = 20.0 + speed * (frame - frames[0]), 24.0
        blob = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 4.0 ** 2)).astype(np.float32)
        texture = 0.3 + 0.1 * np.sin(xs * 0.7 - speed * (frame - frames[0]) * 0.7) * np.cos(ys * 0.5)
        rgba = np.zeros((height, width, 4), np.float32)
        rgba[:, :, :3] = (blob + texture)[:, :, np.newaxis]
        rgba[:, :, 3] = 1.0
        imageIO.writeImage(imageIO.framePath(str(folder / "blob.####.tif"), frame),
                           ImageBuffer.fromRGBA(rgba[::-1].copy(), (width, height)), "float", "linear")


class TestRotoFlowTracking:

    def test_trackShapesSetsAKeyPerFrame(self, tmp_path):
        import json
        from meshroom.live import roto
        from meshroom.live.actions import ActionContext
        frames = [1, 2, 3, 4, 5]
        _movingBlobSequence(tmp_path, frames)
        graph = Graph("live")
        read = graph.addNewNode("Read", file=str(tmp_path / "blob.####.tif"), colorspace="linear")
        rotoNode = graph.addNewNode("Roto")
        graph.addEdge(read.output, rotoNode.input)
        square = [{"x": x, "y": y} for x, y in ((14, 18), (26, 18), (26, 30), (14, 30))]
        rotoNode.attribute("shapes").value = json.dumps([{"name": "s", "closed": True, "points": square}])
        graph.update()
        evaluator = Evaluator()
        values = {attr.name: attr.value for attr in rotoNode.attributes if not attr.isOutput and attr.type == "StringParam"}
        published = []
        context = ActionContext(values, evaluator, buildPlan(read), currentFrame=1,
                                publish=lambda result, frame: published.append(frame))
        result = rotoNode.nodeDesc.runAction("flowForward", context)
        shape = roto.parseShapes(result["shapes"])[0]
        assert [key["frame"] for key in shape["keys"]] == frames
        # Each step is published as it is tracked (the viewer follows)
        assert published == frames[1:]
        xs = [np.mean([p["x"] for p in roto.pointsAtFrame(shape, f)]) for f in frames]
        assert xs[-1] - xs[0] == pytest.approx(8.0, abs=2.0)


class TestCameraSolveRobustness:

    def test_missingFramesAreInterpolated(self):
        from meshroom.live import camera
        tracks, centers, (width, height, focal) = TestCameraTracker._syntheticTracks()
        # Frame 15 has too few tracks to be solved
        for track in tracks[6:]:
            track["keys"].pop("15", None)
        solve = camera.solveCamera(tracks, width, height, focal)
        assert 15 not in solve["frames"] and solve["interpolated"] == [15]
        assert sorted(int(f) for f in solve["cameras"]) == list(range(1, 31))
        solved = {int(f): -np.array(c["R"]).T @ np.array(c["t"]) for f, c in solve["cameras"].items()}
        np.testing.assert_allclose(solved[15], (solved[14] + solved[16]) / 2.0, atol=1e-6)

    def test_estimateFocal(self):
        from meshroom.live import camera
        tracks, _, (width, height, focal) = TestCameraTracker._syntheticTracks()
        candidates = [mm / 36.0 * width for mm in (16, 24, 35, 50, 85)]
        estimated = camera.estimateFocal(tracks, width, height, candidates)
        assert estimated == pytest.approx(focal, rel=0.15)

    def test_autotrackPublishesTheTracks(self):
        import json
        from meshroom.live.actions import ActionContext
        rng = np.random.default_rng(0)
        texture = rng.random((300, 400)).astype(np.float32)
        texture = sum(np.roll(np.roll(texture, i, 0), j, 1) for i in range(-2, 3) for j in range(-2, 3)) / 25.0
        graph = Graph("live")
        cameraTracker = graph.addNewNode("CameraTracker", features=40)
        graph.update()

        def render(frame):
            pixels = texture[40 + frame:40 + frame + 160, 40 + 2 * frame:40 + 2 * frame + 240]
            rgba = np.zeros((160, 240, 4), np.float32)
            rgba[:, :, :3] = pixels[:, :, np.newaxis] * 3
            return ImageBuffer.fromRGBA(rgba, (240, 160))

        published = []
        context = ActionContext(buildPlan(cameraTracker).targetNode.values, currentFrame=1,
                                publish=lambda values, frame: published.append((frame, len(json.loads(values["tracks"])))))
        context.render = render
        context.info = ImageInfo(240, 160, 1, 10)
        cameraTracker.nodeDesc.runAction("autotrack", context)
        # The first frame tracked is published at once
        assert published and published[0][0] == 2 and published[0][1] >= 20


def test_curvePointInterpolations():
    from meshroom.live import curves
    smooth = [[0, 0], [1, 1], [2, 0]]
    # Linear: straight lines to the neighbors (a corner at the point)
    linear = [[0, 0], [1, 1, "linear"], [2, 0]]
    np.testing.assert_allclose(curves.evaluate(linear, [0.5, 1.0, 1.5]), [0.5, 1.0, 0.5], atol=1e-9)
    assert curves.evaluate(smooth, [0.5])[0] != 0.5
    # Constant: held to the next point (a step), flat after a constant last point
    step = [[0, 0, "constant"], [1, 1, "constant"], [2, 3, "constant"]]
    np.testing.assert_allclose(curves.evaluate(step, [0.0, 0.5, 0.99, 1.0, 1.7, 2.0, 5.0, -1.0]),
                               [0, 0, 0, 1, 1, 3, 3, 0], atol=1e-9)
    # Stored: the smooth points without mode; an unknown mode is smooth
    assert curves.sortedPoints([[1, 1, "linear"], [0, 0, "smooth"], [2, 2, "bogus"]]) == [[0, 0], [1, 1, "linear"], [2, 2]]
    assert curves.isIdentity([[0, 0], [1, 1, "linear"]]) and not curves.isIdentity([[0, 0, "constant"], [1, 1]])


def test_compositingTemplateIsRegistered():
    # "New Compositing" opens the "compositing" template of the compositing plugin (meshroom/nodes/compositing)
    import meshroom.core
    meshroom.core.initPipelines()
    path = {k.lower(): v for k, v in meshroom.core.pipelineTemplates.items()}.get("compositing")
    assert path and os.path.isfile(path) and path.endswith("compositing.mgt")


def test_browserListsFilesOfKindsAndResolvesTypedPaths(tmp_path):
    from meshroom.live.sequences import listFolder, resolvePath
    (tmp_path / "shots").mkdir()
    for name in ("a.mg", "b.mgt", "c.exr", "notes.txt"):
        (tmp_path / name).write_text("{}")
    names = [entry["name"] for entry in listFolder(str(tmp_path), False, [".mgt", ".mg"])]
    assert names == ["shots", "a.mg", "b.mgt"]
    assert resolvePath("shots", str(tmp_path)) == {"type": "folder", "path": str(tmp_path / "shots")}
    assert resolvePath(str(tmp_path / "a.mg"), "/") == {"type": "file", "path": str(tmp_path / "a.mg")}
    assert resolvePath("nothing", str(tmp_path))["type"] == "missing"
    assert resolvePath("~", "/")["path"] == os.path.expanduser("~")
