"""
Face tracking (the FaceTracker node): the landmarks of a face found at each frame by MediaPipe Face Landmarker
(Google, Apache-2.0: 478 points, 52 expression coefficients), then a face mesh fitted on them, non rigid.

- `trackFaces` runs the landmarker through frames (video mode: it follows the face from frame to frame) and keeps,
  for each frame, the landmarks (normalized: x, y in [0, 1] from the top left corner, z the relative depth) and the
  expression coefficients ("blendshapes").
- `solveFace` fits a face mesh whose vertex i is the landmark i (the MediaPipe face topology: 468 vertices, or 478
  with the irises): the head pose from the landmarks that move little with the expressions (forehead, nose bridge,
  eye corners, temples), then every vertex moved, at its depth, onto the ray of its landmark: the mouth and the
  eyelids follow the face. Any mesh with that topology works (the canonical face of MediaPipe, or a scan of the
  actor retopologized on it): its own shape is the neutral face.

Positions are in Nuke pixels (y up); cameras are camera-to-world matrices (looking along -Z).
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import scene3d

LANDMARKS = 478
MESH_VERTICES = 468

# Landmarks that move little with the expressions: the head pose is solved with them
RIGID_LANDMARKS = (1, 4, 5, 6, 8, 9, 10, 151, 168, 195, 197, 33, 133, 362, 263, 127, 356, 234, 454, 21, 251, 54, 284,
                   103, 332, 67, 297, 109, 338)

# Regions of the landmarks (MediaPipe face mesh), for the names of the tracks: filled from MediaPipe when installed
_REGIONS = ("lips", "leftEye", "rightEye", "leftEyebrow", "rightEyebrow", "leftIris", "rightIris", "nose", "faceOval")


class FaceUnavailable(RuntimeError):
    """ The face landmarker cannot be used: MediaPipe or its model is missing. """


def landmarkRegions() -> Dict[int, str]:
    """
    The region of each landmark ({index: "lips"...}) from the connections of MediaPipe (its tasks API, or the former
    solutions API), {} without it.
    """
    regions = {}
    try:
        # MediaPipe tasks (0.10 and later): Connection(start, end)
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as connections
        prefix = "FACE_LANDMARKS_"
    except Exception:
        try:
            from mediapipe.python.solutions import face_mesh_connections as connections
            prefix = "FACEMESH_"
        except Exception:
            return regions
    names = {"lips": "LIPS", "leftEye": "LEFT_EYE", "rightEye": "RIGHT_EYE", "leftEyebrow": "LEFT_EYEBROW",
             "rightEyebrow": "RIGHT_EYEBROW", "leftIris": "LEFT_IRIS", "rightIris": "RIGHT_IRIS", "nose": "NOSE",
             "faceOval": "FACE_OVAL"}
    for region in _REGIONS:
        for connection in getattr(connections, prefix + names[region], None) or ():
            pair = (connection.start, connection.end) if hasattr(connection, "start") else connection
            for index in pair:
                regions.setdefault(int(index), region)
    return regions


def trackName(index: int, regions: Dict[int, str]) -> str:
    return f"{regions.get(index, 'face')}_{index:03d}"


# The landmarker

class FaceLandmarker:
    """ MediaPipe Face Landmarker in video mode (the face followed from frame to frame). """

    def __init__(self, modelPath: str, minConfidence: float = 0.5):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as tasks
            from mediapipe.tasks.python import vision
        except Exception as e:
            raise FaceUnavailable(f"MediaPipe is not installed (pip install mediapipe): {e}") from e
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=tasks.BaseOptions(model_asset_path=modelPath), running_mode=vision.RunningMode.VIDEO,
            num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=False,
            min_face_detection_confidence=minConfidence, min_face_presence_confidence=minConfidence,
            min_tracking_confidence=minConfidence)
        try:
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            raise FaceUnavailable(f"cannot load the face landmarker model {modelPath}: {e}") from e

    def detect(self, rgb: np.ndarray, timestampMs: int):
        """
        The landmarks (478, 3) of the face in an image (h, w, 3) uint8, rows from the top, and its expression
        coefficients {name: value}; (None, {}) when there is no face.
        """
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, int(timestampMs))
        if not result.face_landmarks:
            return None, {}
        landmarks = np.array([[point.x, point.y, point.z] for point in result.face_landmarks[0]], dtype=np.float64)
        blendshapes = {}
        if getattr(result, "face_blendshapes", None):
            blendshapes = {category.category_name: float(category.score) for category in result.face_blendshapes[0]}
        return landmarks, blendshapes

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass


def landmarker(minConfidence: float = 0.5) -> FaceLandmarker:
    """ The face landmarker with the model of the models folder (MESHROOM_MODELS_PATH), or FaceUnavailable. """
    from .neural import NeuralUnavailable
    from .neural.models import modelFile
    try:
        path = modelFile("facelandmarker", "mediapipe/face_landmarker.task")
    except NeuralUnavailable as e:
        raise FaceUnavailable(str(e)) from e
    return FaceLandmarker(path, minConfidence)


def trackFaces(imageAt: Callable, frames: List[int], detector, fps: float = 24.0, cancel=None,
               progress: Optional[Callable[[int, int], None]] = None,
               onFrame: Optional[Callable[[int, dict], None]] = None) -> Dict[int, dict]:
    """
    The face at each of `frames` (in order): {frame: {"landmarks": (478, 3) normalized, "blendshapes": {...}}}, the
    frames without face left out. `detector.detect(rgb uint8, timestamp ms)` (see FaceLandmarker).
    """
    from .imageBuffer import BBox
    from .neural import displayRGB
    faces = {}
    for index, frame in enumerate(frames):
        if cancel:
            cancel()
        image = imageAt(frame)
        rgb = (displayRGB(image, BBox.fromFormat(*image.format)) * 255.0 + 0.5).astype(np.uint8)
        # Video mode: increasing time stamps, whatever the direction
        landmarks, blendshapes = detector.detect(rgb, int(round(index * 1000.0 / max(fps, 1e-3))))
        if landmarks is not None:
            faces[frame] = {"landmarks": landmarks, "blendshapes": blendshapes}
            if onFrame:
                onFrame(frame, faces[frame])
        if progress:
            progress(index + 1, len(frames))
    return faces


# Storage (the "faceTrack" parameter): {"format": [w, h], "frames": {"1001": {"l": [x, y, z, ...], "b": {...}}}}

def dumpFaces(faces: Dict[int, dict], imageFormat: Tuple[int, int]) -> str:
    if not faces:
        return ""
    return json.dumps({"format": [int(imageFormat[0]), int(imageFormat[1])],
                       "frames": {str(frame): {"l": [round(float(v), 5) for v in np.asarray(face["landmarks"]).ravel()],
                                               "b": {name: round(float(value), 4)
                                                     for name, value in (face.get("blendshapes") or {}).items()}}
                                  for frame, face in sorted(faces.items())}})


def parseFaces(text: str) -> Tuple[Dict[int, dict], Optional[Tuple[int, int]]]:
    """ (the faces {frame: {"landmarks", "blendshapes"}}, the format they were tracked on) of a "faceTrack" value. """
    if not text or not text.strip():
        return {}, None
    try:
        data = json.loads(text)
        faces = {int(frame): {"landmarks": np.array(face["l"], dtype=np.float64).reshape(-1, 3),
                              "blendshapes": dict(face.get("b") or {})}
                 for frame, face in (data.get("frames") or {}).items()}
        imageFormat = tuple(data["format"]) if data.get("format") else None
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        raise ValueError(f"Invalid face track: {e}") from e
    return faces, imageFormat


def landmarksAt(faces: Dict[int, dict], frame: int, smoothing: float = 0.0) -> Optional[np.ndarray]:
    """
    The landmarks (n, 3) at a frame: smoothed in time by a gaussian of `smoothing` frames (MediaPipe jitters a little),
    the nearest tracked frame where there is none. None without any face.
    """
    if not faces:
        return None
    if smoothing <= 0:
        nearest = frame if frame in faces else min(faces, key=lambda f: (abs(f - frame), f))
        return faces[nearest]["landmarks"]
    radius = int(np.ceil(3 * smoothing))
    weights, total = 0.0, None
    for other in range(frame - radius, frame + radius + 1):
        face = faces.get(other)
        if face is None:
            continue
        weight = float(np.exp(-0.5 * ((other - frame) / smoothing) ** 2))
        total = face["landmarks"] * weight if total is None else total + face["landmarks"] * weight
        weights += weight
    if total is None:
        return landmarksAt(faces, frame, 0.0)
    return total / weights


def toPixels(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """ Normalized landmarks (from the top left corner) to Nuke pixels (n, 2) of a width x height format. """
    return np.c_[landmarks[:, 0] * width, height - landmarks[:, 1] * height]


def landmarkTracks(faces: Dict[int, dict], width: int, height: int, indices=None, smoothing: float = 0.0) -> List[dict]:
    """ The landmarks as tracks of a Tracker (named by their region: lips_013...), keys at the tracked frames. """
    regions = landmarkRegions()
    if not faces:
        return []
    count = min(len(face["landmarks"]) for face in faces.values())
    indices = [i for i in (indices if indices is not None else range(count)) if i < count]
    tracks = [{"name": trackName(i, regions), "enabled": True, "pattern": 15, "search": 41, "auto": True,
               "format": [width, height], "keys": {}} for i in indices]
    for frame in sorted(faces):
        pixels = toPixels(landmarksAt(faces, frame, smoothing), width, height)
        for track, i in zip(tracks, indices):
            track["keys"][str(frame)] = [round(float(pixels[i, 0]), 3), round(float(pixels[i, 1]), 3)]
    return tracks


def featureIndices() -> List[int]:
    """ The landmarks of the features (lips, eyes, eyebrows, irises, nose, face oval): all of them without MediaPipe. """
    regions = landmarkRegions()
    return sorted(regions) if regions else list(range(LANDMARKS))


# The face mesh

def canonicalFace() -> scene3d.GeoData:
    """
    The canonical face of MediaPipe (468 vertices, the landmark order; centimeters, looking along +Z): from the
    MediaPipe package when it has it, else from the models folder. FaceUnavailable when there is none.
    """
    candidates = []
    try:
        import mediapipe
        candidates.append(os.path.join(os.path.dirname(mediapipe.__file__), "modules", "face_geometry", "data",
                                       "canonical_face_model.obj"))
    except Exception:
        pass
    try:
        from .neural.models import modelsPath
        candidates.append(os.path.join(modelsPath(), "mediapipe", "canonical_face_model.obj"))
    except Exception:
        pass
    for path in candidates:
        if os.path.isfile(path):
            return scene3d.readSceneFile(path).geo(0)
    raise FaceUnavailable("the canonical face mesh of MediaPipe is missing: run 'python -m meshroom.live.neural.download "
                          "facelandmarker', or connect a face mesh with the MediaPipe topology to 'Face'")


def checkTopology(mesh: scene3d.GeoData):
    """ FaceUnavailable unless the mesh has the MediaPipe face topology (468 or 478 vertices in the landmark order). """
    if mesh is None or len(mesh.vertices) not in (MESH_VERTICES, LANDMARKS):
        count = 0 if mesh is None else len(mesh.vertices)
        raise FaceUnavailable(f"the face mesh must have the MediaPipe face topology ({MESH_VERTICES} or {LANDMARKS} "
                              f"vertices in the landmark order), not {count} vertices")


def _defaultStart(mesh: scene3d.GeoData) -> np.ndarray:
    """ A camera looking at the face from the front (+Z), far enough to see it whole. """
    low, high = mesh.bounds()
    center = (low + high) / 2.0
    size = float(np.max(high - low)) or 1.0
    return scene3d.lookAt(center + np.array([0.0, 0.0, 4.0 * size]), center)


def solveFace(mesh: scene3d.GeoData, pixels: np.ndarray, camera: scene3d.CameraData, width: int, height: int,
              initial: Optional[np.ndarray] = None, rigid=RIGID_LANDMARKS, deform: bool = True):
    """
    The face seen in the image: the pose of the head from the `rigid` landmarks, then each vertex moved at its depth
    onto the ray of its landmark (`deform`).

    Args:
        mesh: the face (vertex i: landmark i), its shape the neutral face, in its own space.
        pixels: the landmarks (n, 2) in Nuke pixels.
        camera: the plate camera (its lens, and its pose: the world).
        initial: a camera-to-face matrix to start the solve from (e.g. the previous frame).

    Returns:
        (the face as a GeoData in world space, the head matrix (face space to world), the camera-to-face matrix, the
        RMS error of the rigid landmarks in pixels).
    """
    from .geotrack import robustPose
    count = min(len(mesh.vertices), len(pixels))
    rigid = [i for i in rigid if i < count]
    focal = camera.focalPixels(width)
    start = initial if initial is not None else _defaultStart(mesh)
    cameraToFace, _, error = robustPose(mesh.vertices[rigid], pixels[rigid], width, height, focal, start)
    head = camera.matrix @ np.linalg.inv(cameraToFace)
    view = np.linalg.inv(cameraToFace)
    inCamera = mesh.vertices @ view[:3, :3].T + view[:3, 3]
    if deform:
        # Each vertex at its depth, on the ray of its landmark (the camera looks along -Z)
        depth = -inCamera[:count, 2]
        rays = np.c_[(pixels[:count, 0] - width / 2.0) / focal, (pixels[:count, 1] - height / 2.0) / focal,
                     -np.ones(count)]
        inCamera = inCamera.copy()
        inCamera[:count] = rays * depth[:, None]
    world = inCamera @ camera.matrix[:3, :3].T + camera.matrix[:3, 3]
    return scene3d.GeoData(world, name="face", topology=mesh.topology), head, cameraToFace, error
