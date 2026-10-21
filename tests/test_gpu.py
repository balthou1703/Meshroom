"""
GPU acceleration (meshroom/live/gpu.py, gpuOps.py) and the neural networks (meshroom/live/neural).

Without PyTorch: the fallbacks on the CPU and the logic around the networks (with fake networks). With PyTorch (even
without a GPU): each GPU kernel, run on the "cpu" torch device, against its CPU version.
"""

import os

import numpy as np
import pytest

import meshroom
from meshroom.core import pluginManager
from meshroom.live import ImageBuffer, geotrack, gpu, ops, scene3d, tracking
from meshroom.live.evaluator import EvaluationCancelled
from meshroom.live.imageBuffer import BBox

_COMPOSITING_FOLDER = os.path.join(os.path.dirname(meshroom.__file__), "nodes", "compositing")
if not pluginManager.isNodeDescRegistered("Grade"):
    pluginManager.addPluginFromBuiltInFolder("compositing", _COMPOSITING_FOLDER)


@pytest.fixture(autouse=True)
def freshDetection():
    gpu.reset()
    yield
    gpu.reset()


def _image(seed=0, height=64, width=96):
    rng = np.random.default_rng(seed)
    rgba = rng.random((height, width, 4)).astype(np.float32)
    return rgba


# Device detection and fallback

def test_noGpuWithoutTorchOrWhenTurnedOff(monkeypatch):
    monkeypatch.setattr(gpu, "torchModule", lambda: None)
    assert gpu.device(True) is None and "PyTorch" in gpu.status()
    gpu.reset()
    monkeypatch.setenv("MESHROOM_GPU", "0")
    assert gpu.device(True) is None and "MESHROOM_GPU" in gpu.status()
    assert gpu.device(False) is None


def test_runFallsBackOnTheCpu(monkeypatch):
    monkeypatch.setattr(gpu, "availableDevice", lambda: "fakeGpu")
    calls = []

    def failing(device):
        calls.append(device)
        raise RuntimeError("CUDA out of memory")
    assert gpu.run(True, failing, lambda: "cpu", "Test") == "cpu" and calls == ["fakeGpu"]
    assert gpu.run(True, lambda device: device, lambda: "cpu") == "fakeGpu"
    assert gpu.run(False, lambda device: device, lambda: "cpu") == "cpu"

    def cancelled(device):
        raise EvaluationCancelled()
    with pytest.raises(EvaluationCancelled):
        gpu.run(True, cancelled, lambda: "cpu")


def test_nodesWithAGpuParameter():
    from meshroom.core.graph import Graph
    graph = Graph("gpu")
    for nodeType in ("Blur", "Sharpen", "Erode", "Median", "ZDefocus", "OpticalFlow", "Retime", "TimeWarp", "Tracker",
                     "CameraTracker", "Resection"):
        node = graph.addNewNode(nodeType)
        assert node.useGpu.value is True, nodeType


def test_blurOnDeviceWithoutGpuIsTheCpuBlur(monkeypatch):
    monkeypatch.setattr(gpu, "torchModule", lambda: None)
    pixels = _image()
    np.testing.assert_array_equal(ops.blurOnDevice(pixels, 5.0, 9.0, True), ops.blur(pixels, 5.0, 9.0))


def test_matchManyIsMatchPatternOneAfterTheOther():
    lum = _image(1)[:, :, 0]
    bbox = BBox(0, 0, lum.shape[1], lum.shape[0])
    centers = np.array([[30.5, 30.5], [60.2, 20.7], [2.0, 2.0]])
    patterns = np.stack([tracking._patch(lum, bbox, x, y, 9) if tracking._patch(lum, bbox, x, y, 9) is not None
                         else np.zeros((9, 9), np.float32) for x, y in centers])
    # A device without PyTorch: the GPU part fails and the CPU one is used
    for device in (None, "noTorchDevice"):
        matches = tracking.matchMany(patterns, lum, bbox, centers + 1.0, 21, device)
        for (x, y), pattern, match in zip(centers + 1.0, patterns, matches):
            expected = tracking.matchPattern(pattern, lum, bbox, x, y, 21)
            if expected is None:
                assert np.isnan(match).all()
            else:
                np.testing.assert_allclose(match, expected)


# The networks, without PyTorch or with fake ones

def test_modelsFolderIsRequired(monkeypatch, tmp_path):
    from meshroom.live.neural import NeuralUnavailable, models
    monkeypatch.delenv(models.MODELS_PATH_VARIABLE, raising=False)
    with pytest.raises(NeuralUnavailable, match="MESHROOM_MODELS_PATH"):
        models.modelsPath()
    monkeypatch.setenv(models.MODELS_PATH_VARIABLE, str(tmp_path))
    with pytest.raises(NeuralUnavailable, match="download tapir"):
        models.modelFile("tapir", "tapnet/bootstapir_checkpoint_v2.pt")
    assert models.MODELS["cotracker"].nonCommercial and not models.MODELS["tapir"].nonCommercial


def test_raftFallsBackOnLucasKanade(monkeypatch):
    from meshroom.live import flow, neural

    def unavailable(*args, **kwargs):
        raise neural.NeuralUnavailable("no PyTorch here")
    import meshroom.live.neural.raft as raft
    monkeypatch.setattr(raft, "computeFlow", unavailable)
    a = ImageBuffer.fromRGBA(_image(2))
    b = ImageBuffer.fromRGBA(np.roll(_image(2), 2, axis=1))
    np.testing.assert_array_equal(flow.computeFlow(a, b, method="raft"), flow.computeFlow(a, b))


class FakePointTracker:
    """ Moves every point by (+2, -1) pixels per frame (network coordinates: y down), hides the second at frame 3. """
    chunk = 3

    def __init__(self):
        self.calls = []

    def track(self, video, queries, size):
        self.calls.append((len(video), queries.copy()))
        steps = np.arange(len(video))[:, None, None]
        positions = queries[None] + steps * np.array([2.0, -1.0])
        visible = np.ones((len(video), len(queries)), dtype=bool)
        return positions, visible


def _flatImage(frame):
    return ImageBuffer.fromRGBA(np.full((60, 80, 4), 0.5, np.float32), (80, 60))


def test_trackManyNeuralFollowsTheNetwork():
    tracker = FakePointTracker()
    tracks = [{"keys": {"0": [10.0, 30.0]}, "pattern": 9}, {"keys": {"0": [40.0, 20.0]}, "pattern": 9}]
    keys = tracking.trackManyNeural(tracks, list(range(8)), _flatImage, tracker, stopAt=[None, 5])
    # A flat image: the pattern cannot refine, the positions of the network are kept (y up: +1 per frame)
    assert keys[0]["7"] == pytest.approx([24.0, 37.0])
    assert sorted(keys[1], key=int) == ["1", "2", "3", "4"]
    # Chunks of 3 frames, each starting from the positions found at the end of the previous one
    assert [length for length, _ in tracker.calls] == [4, 4, 2]
    np.testing.assert_allclose(tracker.calls[1][1][0], [16.0, 60 - 33.0])


def test_autoTrackWithAMatcher():
    from meshroom.live import camera

    class FakeMatcher:
        """ Every frame: the same 3 features, moved by 1 pixel to the right. """

        def features(self, image):
            return object(), image.format[1]

        def match(self, imageA, imageB, featuresA=None, featuresB=None):
            frame = imageA.frame
            points = np.array([[10.0 + frame, 10.0], [30.0 + frame, 20.0], [50.0 + frame, 40.0]])
            return points, points + [1.0, 0.0], np.array([0.9, 0.8, 0.1])

    def imageAt(frame):
        image = _flatImage(frame)
        image.frame = frame
        return image
    tracks = camera.autoTrackMatched(imageAt, list(range(5)), FakeMatcher(), count=10)
    # The third match is too poor; the two others make one track each through the 5 frames
    assert len(tracks) == 2
    assert tracks[0]["keys"] == {str(f): [10.0 + f, 10.0] for f in range(5)}


def test_alignWithMatchesFindsThePose():
    from tests.test_geoTracking import CUBE, H, W, cameraAt, pixelError
    focal = cameraAt(0).focalPixels(W)
    samples = geotrack.SurfaceSamples(CUBE, count=200000)

    class ExactMatcher:
        """ Points of the cube seen at frame 0 and where they are at frame 6. """

        def match(self, imageA, imageB):
            indices, positions = samples.visible(cameraAt(0).matrix, W, H, focal, margin=0)
            chosen = indices[::max(1, len(indices) // 300)]
            a, _ = geotrack.projectPoints(cameraAt(0).matrix, samples.points[chosen], W, H, focal)
            b, _ = geotrack.projectPoints(cameraAt(6).matrix, samples.points[chosen], W, H, focal)
            return a, b, np.ones(len(a))
    far = cameraAt(6).matrix.copy()
    far[:3, 3] += [1.0, -0.6, 1.5]
    matrix, inliers, _ = geotrack.alignWithMatches(samples, cameraAt(0).matrix, None, far, None, W, H, focal,
                                                   ExactMatcher())
    assert inliers >= 50 and pixelError(matrix, 6) < 1.0


def test_resectionAlignTellsWhenANetworkIsMissing(tmp_path, monkeypatch):
    from tests.test_geoTracking import SyntheticCubePlate, resectionGraph, runAction
    from tests.utils import registerNodeDesc, unregisterNodeDesc
    monkeypatch.delenv("MESHROOM_MODELS_PATH", raising=False)
    registerNodeDesc(SyntheticCubePlate)
    try:
        graph, plate, resection = resectionGraph(tmp_path)
        resection.alignMethod.value = "lightglue"
        graph.update()
        with pytest.raises(RuntimeError, match="Align with lightglue"):
            runAction(resection, plate, "alignGeo", 6)
    finally:
        unregisterNodeDesc(SyntheticCubePlate)


def test_downloadArguments(tmp_path, capsys):
    from meshroom.live.neural import download
    with pytest.raises(SystemExit):
        download.main(["unknownModel", "--path", str(tmp_path)])
    assert "unknown models" in capsys.readouterr().err


# The GPU kernels against their CPU versions (PyTorch needed, on its "cpu" device)

@pytest.fixture
def torchCpu():
    torch = pytest.importorskip("torch")
    return torch.device("cpu")


@pytest.mark.parametrize("sizes", [(0.0, 3.0), (2.5, 2.5), (9.0, 30.0)])
def test_gpuBlur(torchCpu, sizes):
    from meshroom.live import gpuOps
    pixels = _image(3)
    np.testing.assert_allclose(gpuOps.blur(pixels, *sizes, torchCpu), ops.blur(pixels, *sizes), atol=1e-5)


@pytest.mark.parametrize("useMax", [False, True])
def test_gpuErodeDilate(torchCpu, useMax):
    from meshroom.live import gpuOps
    from meshroom.nodes.compositing.FilterNodes import _extremumAxis
    channels = _image(4)[:, :, :2].copy()
    expected = _extremumAxis(_extremumAxis(channels, 3, 1, useMax), 3, 0, useMax)
    np.testing.assert_allclose(gpuOps.extremum(channels, 3, useMax, torchCpu), expected)


def test_gpuMedian(torchCpu):
    ndimage = pytest.importorskip("scipy.ndimage")
    from meshroom.live import gpuOps
    pixels = _image(5)
    expected = ndimage.median_filter(pixels, size=(5, 5, 1), mode="constant", cval=0.0)
    np.testing.assert_allclose(gpuOps.median(pixels, 5, torchCpu, maxBytes=200000), expected)


def test_gpuFftConvolve(torchCpu):
    from meshroom.live import defocus, gpuOps
    pixels = _image(6)
    kernel = defocus.bokehKernel(4.0)
    np.testing.assert_allclose(gpuOps.fftConvolve(pixels, kernel, torchCpu), defocus.fftConvolve(pixels, kernel),
                               atol=1e-5)


def test_gpuOpticalFlow(torchCpu):
    from meshroom.live import flow, gpuOps
    rng = np.random.default_rng(7)
    a = flow._boxFilter(rng.random((80, 120)).astype(np.float32), 2)
    b = np.roll(a, (1, 2), axis=(0, 1))
    np.testing.assert_allclose(gpuOps.lucasKanade(a, b, 11, 3, torchCpu), flow._lucasKanade(a, b, 11, 3), atol=1e-3)


def test_gpuMatchPatterns(torchCpu):
    from meshroom.live import gpuOps
    rng = np.random.default_rng(8)
    lum = rng.random((90, 120)).astype(np.float32)
    bbox = BBox(-5, 3, 115, 93)
    centers = np.array([[30.5, 40.5], [70.3, 60.8], [100.0, 20.0], [-4.0, 10.0]])
    patterns = []
    for x, y in centers:
        pattern = tracking._patch(lum, bbox, x + 1.3, y - 0.7, 11)
        patterns.append(pattern if pattern is not None else np.zeros((11, 11), np.float32))
    result = gpuOps.matchPatterns(np.stack(patterns), lum, bbox, centers, 31, torchCpu)
    for (x, y), pattern, match in zip(centers, patterns, result):
        expected = tracking.matchPattern(pattern, lum, bbox, x, y, 31)
        if expected is None:
            assert np.isnan(match).all()
        else:
            np.testing.assert_allclose(match, expected, atol=1e-4)


def test_gpuTrackManyMatchesTheCpu(torchCpu):
    from tests.test_geoTracking import _blobs
    tracks = [{"keys": {"0": [30, 40]}, "pattern": 21, "search": 41},
              {"keys": {"0": [110, 30]}, "pattern": 21, "search": 41}]
    cpu = tracking.trackMany(tracks, list(range(9)), _blobs)
    onDevice = tracking.trackMany(tracks, list(range(9)), _blobs, device=torchCpu)
    for a, b in zip(cpu, onDevice):
        assert sorted(a) == sorted(b)
        for frame in a:
            assert a[frame] == pytest.approx(b[frame], abs=1e-2)


def test_geometryTrackingOnTheDevice(torchCpu):
    from tests.test_geoTracking import CUBE, H, W, cameraAt, pixelError, renderCube
    focal = cameraAt(0).focalPixels(W)
    poses, reason = geotrack.trackPose(geotrack.SurfaceSamples(CUBE), geotrack.TrackSettings("patches", device=torchCpu),
                                       list(range(6)), lambda frame: renderCube(cameraAt(frame)), cameraAt(0).matrix,
                                       W, H, focal)
    assert reason == "" and max(pixelError(poses[frame], frame) for frame in poses) < 2.0


def test_scene3dUnchanged():
    # The pose conversions used by MegaPose (OpenCV object-to-camera) round trip
    camera = scene3d.lookAt((1.0, 2.0, 5.0), (0, 0, 0))
    rotation, translation = scene3d.cvPose(camera)
    np.testing.assert_allclose(scene3d.cameraMatrixFromCvPose(rotation, translation), camera, atol=1e-12)


def test_gpuDepthBuffer(torchCpu):
    from meshroom.live import geoRender
    from meshroom.live.imageBuffer import BBox
    from tests.test_geoTracking import CUBE, H, W, cameraAt
    u, v, z, _ = geoRender.screenPositions(CUBE, cameraAt(3), W, H, BBox(0, 0, W, H))
    zCpu, idsCpu = geoRender.rasterize(u, v, z, CUBE.triangles(), (H, W))
    zGpu, idsGpu = geoRender.rasterize(u, v, z, CUBE.triangles(), (H, W), device=torchCpu)
    assert (np.isfinite(zCpu) == np.isfinite(zGpu)).mean() > 0.999
    both = np.isfinite(zCpu) & np.isfinite(zGpu)
    np.testing.assert_allclose(zGpu[both], zCpu[both], rtol=1e-5)


def test_aGpuTooOldForPyTorchIsNotUsed(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_capability(index):
            return (5, 2)

        @staticmethod
        def get_device_name(index):
            return "Quadro M4000"

        @staticmethod
        def get_arch_list():
            return ["sm_70", "sm_75", "sm_80", "sm_86", "compute_86"]

    class FakeTorch:
        cuda = FakeCuda

        @staticmethod
        def device(*args):
            return args

    monkeypatch.delenv("MESHROOM_GPU", raising=False)
    monkeypatch.setattr(gpu, "torchModule", lambda: FakeTorch)
    assert gpu.device(True) is None
    assert "Quadro M4000" in gpu.status() and "5.2" in gpu.status()
