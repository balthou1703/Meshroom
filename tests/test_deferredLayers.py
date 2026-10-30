""" Deferred layers (imageBuffer.LazyPixels): the layers nobody looks at are not read nor computed. """
import os
from collections import OrderedDict

import numpy as np

import meshroom
from meshroom.core import pluginManager
from meshroom.live import BBox, ImageBuffer, imageIO, ops
from meshroom.live.imageBuffer import LazyPixels, deferred

if not pluginManager.isNodeDescRegistered("Grade"):
    pluginManager.addPluginFromBuiltInFolder("compositing", os.path.join(os.path.dirname(meshroom.__file__), "nodes",
                                                                         "compositing"))


def _image(calls):
    rng = np.random.default_rng(1)
    rgba = rng.random((6, 8, 4)).astype(np.float32)
    diffuse = rng.random((6, 8, 3)).astype(np.float32)

    def load():
        calls.append("diffuse")
        return diffuse
    layers = OrderedDict([("rgba", (("R", "G", "B", "A"), rgba)),
                          ("diffuse", (("R", "G", "B"), LazyPixels((6, 8, 3), load)))])
    return ImageBuffer(layers, (8, 6)), rgba, diffuse


def test_lazyLayerReadOnUse():
    calls = []
    image, rgba, diffuse = _image(calls)
    assert image.nbytes == rgba.nbytes + diffuse.nbytes and image.shape("diffuse") == (6, 8, 3)
    np.testing.assert_array_equal(image.rgba(), rgba)
    # Derived images keep it deferred
    graded = image.withRGBA(image.rgba() * 2.0).mapLayers(lambda p: p + 1.0, ["rgba"])
    cropped = graded.mapLayersDeferred(lambda p: ops.alignToBBox(p, graded.bbox, BBox(1, 1, 5, 4)), BBox(1, 1, 5, 4))
    assert calls == []
    np.testing.assert_array_equal(cropped.layer("diffuse"), ops.alignToBBox(diffuse, image.bbox, BBox(1, 1, 5, 4)))
    # Read once, shared
    np.testing.assert_array_equal(image.layer("diffuse"), diffuse)
    assert calls == ["diffuse"]
    assert not image.layer("diffuse").flags.writeable


def test_deferredOfDeferred():
    calls = []
    source = LazyPixels((2, 2, 1), lambda: calls.append(1) or np.ones((2, 2, 1), np.float32))
    doubled = deferred(lambda p: p * 2, source, (2, 2, 1))
    assert not calls
    np.testing.assert_array_equal(doubled.get(), np.full((2, 2, 1), 2.0))
    assert calls == [1]


def test_proxyAndMergeKeepOtherLayersDeferred():
    from meshroom.core.graph import Graph
    from meshroom.live import Evaluator, buildPlan
    calls = []
    image, rgba, diffuse = _image(calls)
    small = ops.downscaleImage(image, 2)
    assert calls == [] and small.shape("diffuse") == (3, 4, 3)
    np.testing.assert_allclose(small.layer("diffuse")[0, 0], diffuse[:2, :2].mean(axis=(0, 1)), rtol=1e-5)
    assert calls == ["diffuse"]

    # Merge: the layers of B other than its main one follow the new bbox when used
    calls.clear()
    image, rgba, diffuse = _image(calls)
    registerNodeDesc(LazySource)
    try:
        LazySource.image = image
        graph = Graph("merge")
        source = graph.addNewNode("LazySource")
        constant = graph.addNewNode("Constant", width=4, height=4)
        transform = graph.addNewNode("Transform", translate={"x": 20.0, "y": 0.0})
        graph.addEdge(constant.output, transform.input)
        merge = graph.addNewNode("Merge")
        graph.addEdge(source.output, merge.B)
        graph.addEdge(transform.output, merge.A)
        result = Evaluator().render(buildPlan(merge), 1)
        assert calls == [] and result.layerNames == ["rgba", "diffuse"] and result.bbox != image.bbox
        np.testing.assert_array_equal(result.layer("diffuse"), ops.alignToBBox(diffuse, image.bbox, result.bbox))
        assert calls == ["diffuse"]
    finally:
        unregisterNodeDesc(LazySource)


from meshroom.core import desc as _desc  # noqa: E402
from meshroom.live import ImageInfo as _ImageInfo  # noqa: E402
from tests.utils import registerNodeDesc, unregisterNodeDesc  # noqa: E402


class LazySource(_desc.ImageOpNode):
    """ Gives LazySource.image (with its deferred layers). """
    category = "Test"
    inputs = []
    outputs = [_desc.Image(name="output", label="Output", description="")]
    image = None

    def getInfo(self, ctx, inputsInfo):
        return _ImageInfo(8, 6)

    def render(self, ctx, inputs):
        return LazySource.image


class _FakeSpec:
    def __init__(self, names, name=None, width=8, height=6, x=0, y=0, basetype="float"):
        self.channelnames = names
        self.nchannels = len(names)
        self.width, self.height, self.x, self.y = width, height, x, y
        self.full_width, self.full_height, self.full_x, self.full_y = 8, 6, 0, 0
        self.format = type("Format", (), {"basetype": basetype})()
        self.extra_attribs = []
        self._name = name

    def getattribute(self, name):
        return self._name if name in ("oiio:subimagename", "name") else None


class _FakeOIIO:
    """ A multi-part file: a beauty part (RGBA) and a "diffuse" part (unprefixed RGB, smaller data window). """
    UINT8, UINT16 = "uint8", "uint16"
    reads = []

    def __init__(self):
        rng = np.random.default_rng(2)
        self.parts = [(_FakeSpec(["R", "G", "B", "A"]), rng.random((6, 8, 4)).astype(np.float32)),
                      (_FakeSpec(["R", "G", "B"], "diffuse", width=4, height=3, x=2, y=1),
                       rng.random((3, 4, 3)).astype(np.float32))]
        fake = self

        class ImageInput:
            @staticmethod
            def open(path):
                return ImageInput()

            def __init__(self):
                self.current = 0

            def seek_subimage(self, subimage, mip):
                self.current = subimage
                return subimage < len(fake.parts)

            def spec(self):
                return fake.parts[self.current][0]

            def read_image(self, subimage, mip, first, last, type):
                fake.reads.append((subimage, first, last))
                return fake.parts[subimage][1][:, :, first:last]

            def geterror(self):
                return ""

            def close(self):
                pass
        self.ImageInput = ImageInput

    @staticmethod
    def geterror():
        return ""


def test_multiPartFileReadsItsPartsWhenUsed(tmp_path, monkeypatch):
    fake = _FakeOIIO()
    _FakeOIIO.reads = []
    monkeypatch.setattr(imageIO, "oiio", fake)
    path = tmp_path / "beauty.exr"
    path.write_bytes(b"")
    image = imageIO.readImage(str(path), "linear")
    assert image.layerNames == ["rgba", "diffuse"]
    # The beauty only
    assert fake.reads == [(0, 0, 4)]
    np.testing.assert_allclose(image.rgba(), fake.parts[0][1], rtol=1e-6)
    diffuse = image.layer("diffuse")
    assert fake.reads == [(0, 0, 4), (1, 0, 3)]
    # The part placed in the data window of the image (its own window: rows from the top)
    assert diffuse.shape == (6, 8, 3)
    np.testing.assert_allclose(diffuse[1:4, 2:6], fake.parts[1][1], rtol=1e-6)
    assert diffuse[0].max() == 0.0
