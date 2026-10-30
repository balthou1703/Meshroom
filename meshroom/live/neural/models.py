"""
The networks, where their weights are and how to get them.

Weights are read from the folder of MESHROOM_MODELS_PATH (never downloaded while Meshroom runs):
    <models>/torch/hub/checkpoints/   RAFT (torchvision), ALIKED and LightGlue (lightglue package): the torch hub cache
    <models>/tapnet/                  BootsTAPIR checkpoint
    <models>/cotracker/               CoTracker3 checkpoint
    <models>/megapose/                MegaPose data folder (MEGAPOSE_DATA_DIR)
    <models>/mediapipe/               MediaPipe Face Landmarker, multiclass selfie segmenter (FaceTracker, FaceBuilder)
    <models>/depthanything/           Depth Anything V2 Small (transformers format): depth of the FaceBuilder views

Fill it once, on a machine with internet access: python -m meshroom.live.neural.download --path <models> [names]
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Dict, List, Optional

from . import NeuralUnavailable, requireTorch

MODELS_PATH_VARIABLE = "MESHROOM_MODELS_PATH"


class Model:
    """ A network: its Python package, its license and its files in the models folder. """

    def __init__(self, name: str, label: str, package: str, pipName: str, license: str, files: List[str],
                 urls: Optional[Dict[str, str]] = None, nonCommercial: bool = False):
        self.name, self.label = name, label
        self.package, self.pipName = package, pipName
        self.license, self.nonCommercial = license, nonCommercial
        # Paths relative to the models folder, and the URL to download each of them from (when not from torch hub)
        self.files = files
        self.urls = urls or {}


MODELS = {
    "raft": Model("raft", "RAFT optical flow (torchvision raft_large)", "torchvision", "torchvision",
                  "BSD-3-Clause", ["torch/hub/checkpoints"]),
    "tapir": Model("tapir", "BootsTAPIR point tracker", "tapnet", "tapnet @ git+https://github.com/google-deepmind/tapnet",
                   "Apache-2.0 (code), CC-BY-4.0 (weights)", ["tapnet/bootstapir_checkpoint_v2.pt"],
                   {"tapnet/bootstapir_checkpoint_v2.pt":
                    "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt"}),
    "cotracker": Model("cotracker", "CoTracker3 point tracker (offline)", "cotracker",
                       "cotracker @ git+https://github.com/facebookresearch/co-tracker",
                       "CC-BY-NC-4.0: non-commercial use only", ["cotracker/scaled_offline.pth"],
                       {"cotracker/scaled_offline.pth":
                        "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"},
                       nonCommercial=True),
    "lightglue": Model("lightglue", "LightGlue matcher with ALIKED features", "lightglue",
                       "lightglue @ git+https://github.com/cvg/LightGlue", "Apache-2.0 (LightGlue), BSD-3-Clause (ALIKED)",
                       ["torch/hub/checkpoints"]),
    "megapose": Model("megapose", "MegaPose object pose (RGB)", "megapose", "megapose (megapose6d) or happypose",
                      "Apache-2.0", ["megapose"]),
    "facelandmarker": Model("facelandmarker", "MediaPipe Face Landmarker (FaceTracker)", "mediapipe", "mediapipe",
                            "Apache-2.0", ["mediapipe/face_landmarker.task", "mediapipe/canonical_face_model.obj"],
                            {"mediapipe/face_landmarker.task":
                             "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
                             "face_landmarker.task",
                             "mediapipe/canonical_face_model.obj": [
                                 "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/"
                                 "face_geometry/data/canonical_face_model.obj",
                                 "https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/"
                                 "face_geometry/data/canonical_face_model.obj"]}),
    "headsegmenter": Model("headsegmenter", "MediaPipe multiclass selfie segmenter (FaceBuilder: hair, skin, clothes)",
                           "mediapipe", "mediapipe", "Apache-2.0", ["mediapipe/selfie_multiclass_256x256.tflite"],
                           {"mediapipe/selfie_multiclass_256x256.tflite":
                            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/"
                            "float32/latest/selfie_multiclass_256x256.tflite"}),
    "depthanything": Model("depthanything", "Depth Anything V2 Small (FaceBuilder: depth of the views)", "transformers",
                           "transformers", "Apache-2.0",
                           ["depthanything/config.json", "depthanything/model.safetensors",
                            "depthanything/preprocessor_config.json"],
                           {f"depthanything/{name}":
                            f"https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/main/{name}"
                            for name in ("config.json", "model.safetensors", "preprocessor_config.json")}),
}


def modelsPath() -> str:
    """ The models folder (MESHROOM_MODELS_PATH), or NeuralUnavailable when it is not set or does not exist. """
    path = os.environ.get(MODELS_PATH_VARIABLE, "").strip()
    if not path:
        raise NeuralUnavailable(f"{MODELS_PATH_VARIABLE} is not set (the folder of the network weights)")
    if not os.path.isdir(path):
        raise NeuralUnavailable(f"{MODELS_PATH_VARIABLE} ({path}) is not a folder")
    return path


def modelFile(name: str, relative: str) -> str:
    """ The path of a weights file of a model, or NeuralUnavailable telling how to get it. """
    path = os.path.join(modelsPath(), relative)
    if not os.path.isfile(path):
        raise NeuralUnavailable(f"the weights of {MODELS[name].label} are missing ({path}): run "
                                f"'python -m meshroom.live.neural.download {name}' on a machine with internet access")
    return path


_hubLock = threading.Lock()


def useTorchHub():
    """ Point the torch hub cache (where torchvision and lightglue look for their weights) to the models folder. """
    torch = requireTorch()
    hub = os.path.join(modelsPath(), "torch", "hub")
    with _hubLock:
        if os.path.normpath(torch.hub.get_dir()) != os.path.normpath(hub):
            torch.hub.set_dir(hub)
    return hub


def importPackage(name: str):
    """ The Python package of a model, or NeuralUnavailable telling how to install it. """
    import importlib
    model = MODELS[name]
    try:
        return importlib.import_module(model.package)
    except Exception as e:
        raise NeuralUnavailable(f"the package '{model.package}' of {model.label} is not installed "
                                f"(pip install \"{model.pipName}\"): {e}") from e


_cache = {}
_cacheLock = threading.Lock()
_loadLock = threading.Lock()


def cachedModel(key, factory: Callable):
    """ A loaded network kept for the next uses (one per model, settings and device): factory() the first time. """
    with _cacheLock:
        if key in _cache:
            return _cache[key]
    # One load at a time: two threads asking for the same network load it once
    with _loadLock:
        with _cacheLock:
            if key in _cache:
                return _cache[key]
        model = factory()
        with _cacheLock:
            return _cache.setdefault(key, model)


def clearCache():
    """ Forget the loaded networks (their memory on the GPU is given back). """
    from ..gpu import emptyCache
    with _cacheLock:
        _cache.clear()
    emptyCache()


def loadError(name: str, error: Exception) -> NeuralUnavailable:
    """ The NeuralUnavailable of a network whose weights could not be loaded (missing from the models folder). """
    return NeuralUnavailable(f"cannot load {MODELS[name].label} ({error}); its weights must be in "
                             f"{os.environ.get(MODELS_PATH_VARIABLE, '$' + MODELS_PATH_VARIABLE)}: run "
                             f"'python -m meshroom.live.neural.download {name}' on a machine with internet access")
