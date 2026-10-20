"""
MegaPose (megapose6d, or its happypose version): the pose of a known object in an RGB image from its mesh, starting from
a box around it. The Resection uses it to "Align" its geometry when it is far from its place: the box is where the
current (rough) pose shows the geometry; the pose found is then refined on the image (geotrack.alignPose).

Its data folder (the models and their renders cache) is <models>/megapose (MEGAPOSE_DATA_DIR). It needs a CUDA GPU.
"""
from __future__ import annotations

import importlib
import os
import tempfile

import numpy as np

from . import NeuralUnavailable, displayRGB, requireTorch
from .models import loadError, modelsPath

MODEL_NAME = "megapose-1.0-RGB-multi-hypothesis"

# Where each name lives in megapose6d, then in happypose
_SYMBOLS = {
    "RigidObject": ("megapose.datasets.object_dataset", "happypose.toolbox.datasets.object_dataset"),
    "RigidObjectDataset": ("megapose.datasets.object_dataset", "happypose.toolbox.datasets.object_dataset"),
    "ObjectData": ("megapose.datasets.scene_dataset", "happypose.toolbox.datasets.scene_dataset"),
    "ObservationTensor": ("megapose.inference.types", "happypose.toolbox.inference.types"),
    "make_detections_from_object_data": ("megapose.inference.utils", "happypose.toolbox.inference.utils"),
    "NAMED_MODELS": ("megapose.utils.load_model", "happypose.toolbox.utils.load_model",
                     "happypose.pose_estimators.megapose.utils.load_model"),
    "load_named_model": ("megapose.utils.load_model", "happypose.toolbox.utils.load_model",
                         "happypose.pose_estimators.megapose.utils.load_model"),
}


def _symbol(name: str):
    errors = []
    for module in _SYMBOLS[name]:
        try:
            return getattr(importlib.import_module(module), name)
        except Exception as e:
            errors.append(f"{module}: {e}")
    raise NeuralUnavailable(f"MegaPose is not installed (megapose6d or happypose): {'; '.join(errors)}")


def _prepareData():
    """ The data folder of MegaPose, set in its environment variables before it is imported. """
    folder = os.path.join(modelsPath(), "megapose")
    if not os.path.isdir(folder):
        raise NeuralUnavailable(f"the MegaPose data folder is missing ({folder}): run "
                                "'python -m meshroom.live.neural.download megapose' on a machine with internet access")
    os.environ.setdefault("MEGAPOSE_DATA_DIR", folder)
    os.environ.setdefault("HAPPYPOSE_DATA_DIR", folder)
    return folder


def _writeMesh(geo, folder: str) -> str:
    """ The geometry written as an OBJ file (world coordinates, meters for MegaPose). """
    path = os.path.join(folder, "object.obj")
    with open(path, "w") as file:
        file.writelines(f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in geo.vertices)
        for triangle in geo.triangles():
            file.write("f " + " ".join(str(int(i) + 1) for i in triangle) + "\n")
    return path


def estimatePose(image, geo, roughMatrix: np.ndarray, focalPixels: float, device) -> np.ndarray:
    """
    The camera-to-world matrix whose view of the geometry `geo` (at rest in the world) fits the image, from a rough
    one (its view of the geometry gives the box MegaPose starts from). Raises NeuralUnavailable when MegaPose cannot
    be used.
    """
    torch = requireTorch()
    if device is None or getattr(device, "type", "") != "cuda":
        raise NeuralUnavailable("MegaPose needs a CUDA GPU")
    if geo is None or not len(geo.triangles()):
        raise NeuralUnavailable("MegaPose needs a mesh (faces), not a point cloud")
    _prepareData()
    from ..geotrack import projectPoints
    from ..imageBuffer import BBox
    from ..scene3d import cameraMatrixFromCvPose
    width, height = image.format
    xy, depth = projectPoints(roughMatrix, geo.vertices, width, height, focalPixels)
    xy = xy[depth > 0]
    if not len(xy):
        raise NeuralUnavailable("the geometry is behind the camera: place it roughly first")
    # The box (OpenCV pixels: from the top left corner) where the rough pose shows the geometry
    x0, x1 = np.clip([xy[:, 0].min(), xy[:, 0].max()], 0, width - 1)
    y0, y1 = np.clip([height - xy[:, 1].max(), height - xy[:, 1].min()], 0, height - 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise NeuralUnavailable("the geometry is out of the image: place it roughly first")
    rgb = (displayRGB(image, BBox.fromFormat(width, height)) * 255.0 + 0.5).astype(np.uint8)
    K = np.array([[focalPixels, 0.0, width / 2.0], [0.0, focalPixels, height / 2.0], [0.0, 0.0, 1.0]])
    RigidObject, RigidObjectDataset = _symbol("RigidObject"), _symbol("RigidObjectDataset")
    ObjectData, ObservationTensor = _symbol("ObjectData"), _symbol("ObservationTensor")
    makeDetections = _symbol("make_detections_from_object_data")
    namedModels, loadNamedModel = _symbol("NAMED_MODELS"), _symbol("load_named_model")
    with tempfile.TemporaryDirectory(prefix="meshroomMegaPose") as folder:
        objects = RigidObjectDataset([RigidObject(label="geometry", mesh_path=_writeMesh(geo, folder), mesh_units="m")])
        try:
            # The renderer of the estimator is bound to the mesh: loaded for each call (the weights are cached on disk)
            estimator = loadNamedModel(MODEL_NAME, objects).cuda()
        except Exception as e:
            raise loadError("megapose", e) from e
        detections = makeDetections([ObjectData(label="geometry", bbox_modal=np.array([x0, y0, x1, y1]))]).cuda()
        observation = ObservationTensor.from_numpy(rgb, None, K).cuda()
        with torch.no_grad():
            output, _ = estimator.run_inference_pipeline(observation, detections=detections,
                                                         **namedModels[MODEL_NAME]["inference_parameters"])
        poses = output.poses.to("cpu").numpy()
    if not len(poses):
        raise NeuralUnavailable("MegaPose found no pose")
    # Object (= world) to camera, OpenCV axes
    transform = poses[0]
    return cameraMatrixFromCvPose(transform[:3, :3], transform[:3, 3])

