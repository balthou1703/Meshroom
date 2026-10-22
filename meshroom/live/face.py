"""
Face tracking (the FaceTracker node): the landmarks of a face found at each frame by MediaPipe Face Landmarker
(Google, Apache-2.0: 478 points, 52 expression coefficients), then a face mesh fitted on them, non rigid.

- `trackFaces` runs the landmarker through frames and keeps, for each frame, the landmarks (normalized: x, y in [0, 1] from the top left corner, z the relative depth) and the
  expression coefficients ("blendshapes"). The landmarker sees the face in a crop around where it was at the previous
  frame (a small face in a large plate is found as well as a close-up), else in the whole frame, else in tiles; each
  crop is brightened when dark.
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


def _forgetMediaPipe(before):
    """ Remove from sys.modules the mediapipe modules a failed import left (not in `before`): imported again, its
    package would not get them as attributes ("name 'python' is not defined"). """
    import sys
    for name in [name for name in sys.modules if (name == "mediapipe" or name.startswith("mediapipe."))
                 and name not in before]:
        sys.modules.pop(name, None)


def importMediaPipe():
    """
    The mediapipe module, or FaceUnavailable. `import mediapipe` also loads its drawing utilities, which need
    matplotlib and OpenCV, not used here: when they are not installed, stand-ins are given to that import only (put
    before importing: a mediapipe import failing halfway cannot be done again in the same session).
    """
    import importlib
    import importlib.util
    import sys
    import types
    if "mediapipe" in sys.modules and hasattr(sys.modules["mediapipe"], "tasks"):
        return sys.modules["mediapipe"]
    # The leftovers of an earlier failed import (in this session, by this or other code) forgotten first
    _forgetMediaPipe(set())
    stubs = []

    def stub(name):
        stubs.append(name)
        sys.modules[name] = types.ModuleType(name)
        if name == "matplotlib":
            sys.modules["matplotlib.pyplot"] = sys.modules["matplotlib"].pyplot = types.ModuleType("matplotlib.pyplot")

    def missing(name):
        if name in sys.modules:
            return False
        try:
            return importlib.util.find_spec(name) is None
        except (ImportError, ValueError):
            return True
    before = set(sys.modules)
    try:
        for name in ("matplotlib", "cv2"):
            if missing(name):
                stub(name)
        for _ in range(3):
            try:
                return importlib.import_module("mediapipe")
            except ModuleNotFoundError as e:
                name = (e.name or "").split(".")[0]
                if name not in ("matplotlib", "cv2") or name in stubs:
                    raise
                # Installed but broken (e.g. built for another numpy): a stand-in, and the import done again cleanly
                _forgetMediaPipe(before)
                stub(name)
        raise ImportError("mediapipe cannot be imported")
    except Exception as e:
        _forgetMediaPipe(before)
        raise FaceUnavailable(f"MediaPipe is not installed (pip install mediapipe): {e}") from e
    finally:
        for name in stubs:
            sys.modules.pop(name, None)
            if name == "matplotlib":
                sys.modules.pop("matplotlib.pyplot", None)


def landmarkRegions() -> Dict[int, str]:
    """
    The region of each landmark ({index: "lips"...}) from the connections of MediaPipe (its tasks API, or the former
    solutions API), {} without it.
    """
    regions = {}
    try:
        importMediaPipe()
    except FaceUnavailable:
        return regions
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
    """
    MediaPipe Face Landmarker on single images (trackFaces gives it a crop around the face of the previous frame: in
    video mode MediaPipe would follow the face in the coordinates of the previous crop).
    """

    def __init__(self, modelPath: str, minConfidence: float = 0.5):
        mp = importMediaPipe()
        try:
            from mediapipe.tasks import python as tasks
            from mediapipe.tasks.python import vision
        except Exception as e:
            raise FaceUnavailable(f"MediaPipe is not installed (pip install mediapipe): {e}") from e
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=tasks.BaseOptions(model_asset_path=modelPath), running_mode=vision.RunningMode.IMAGE,
            num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=False,
            min_face_detection_confidence=minConfidence, min_face_presence_confidence=minConfidence,
            min_tracking_confidence=minConfidence)
        try:
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            raise FaceUnavailable(f"cannot load the face landmarker model {modelPath}: {e}") from e

    def detect(self, rgb: np.ndarray, timestampMs: int = 0):
        """
        The landmarks (478, 3) of the face in an image (h, w, 3) uint8, rows from the top (normalized: x, y in [0, 1],
        z relative to the width), and its expression coefficients {name: value}; (None, {}) when there is no face.
        """
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect(image)
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


def _cropAround(landmarks: np.ndarray, width: int, height: int, scale: float = 2.2) -> Tuple[int, int, int, int]:
    """ The square (x0, y0, x1, y1) in pixels, rows from the top, `scale` times the face of normalized `landmarks`,
    moved (and shrunk) to stay in the image. """
    x, y = landmarks[:, 0] * width, landmarks[:, 1] * height
    size = int(round(min(max(64.0, scale * max(x.max() - x.min(), y.max() - y.min())), width, height)))
    cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    x0 = int(round(min(max(cx - size / 2.0, 0), width - size)))
    y0 = int(round(min(max(cy - size / 2.0, 0), height - size)))
    return x0, y0, x0 + size, y0 + size


def _searchRegions(width: int, height: int, area=None) -> List[Tuple[int, int, int, int]]:
    """ Where a face is searched when none was found at the previous frame: the whole frame (or `area`: (x0, y0, x1,
    y1), rows from the top), then squares of its height (and half of it) overlapping by half (a small face in a large
    plate: the landmarker works on ~256 pixels). """
    x0, y0, x1, y1 = area if area is not None else (0, 0, width, height)
    regionWidth, regionHeight = x1 - x0, y1 - y0
    regions = [(x0, y0, x1, y1)]
    side = min(regionWidth, regionHeight)
    for size in (side, side // 2):
        if size < 128 or (size == side and regionWidth == regionHeight):
            continue
        step = max(1, size // 2)
        xs = sorted(set(list(range(x0, x1 - size + 1, step)) + [x1 - size]))
        ys = sorted(set(list(range(y0, y1 - size + 1, step)) + [y1 - size]))
        regions += [(x, y, x + size, y + size) for y in ys for x in xs]
    return regions


def _maskArea(mask: np.ndarray, margin: int = 16):
    """ The box (x0, y0, x1, y1) around the white part (above 0.5) of a mask (h, w), rows from the top, with a margin;
    None when it is all black. """
    rows, columns = np.nonzero(mask >= 0.5)
    if not len(rows):
        return None
    height, width = mask.shape
    return (max(0, int(columns.min()) - margin), max(0, int(rows.min()) - margin),
            min(width, int(columns.max()) + 1 + margin), min(height, int(rows.max()) + 1 + margin))


def hiddenLandmarks(landmarks: np.ndarray, mask: np.ndarray) -> List[int]:
    """ The landmarks (normalized) where the mask (h, w, rows from the top) is below 0.5 (masked out). """
    height, width = mask.shape
    columns = np.clip(np.floor(landmarks[:, 0] * width).astype(np.int64), 0, width - 1)
    rows = np.clip(np.floor(landmarks[:, 1] * height).astype(np.int64), 0, height - 1)
    return [int(i) for i in np.flatnonzero(mask[rows, columns] < 0.5)]


def _detectIn(detector, rgb: np.ndarray, region, timestampMs: int):
    """ The landmarks (normalized to the whole image) and blendshapes of the face in a region of `rgb` (floats in [0,
    1]), brightened when dark; (None, {}) without face. """
    x0, y0, x1, y1 = region
    crop = rgb[y0:y1, x0:x1]
    # Dark crops brightened (the landmarker is trained on well exposed photos), at most 8 times (noise)
    level = float(np.percentile(crop.max(axis=2), 99.9)) if crop.size else 1.0
    if level < 0.6:
        crop = np.clip(crop * min(8.0, 0.9 / max(level, 1e-6)), 0.0, 1.0)
    landmarks, blendshapes = detector.detect((crop * 255.0 + 0.5).astype(np.uint8), timestampMs)
    if landmarks is None:
        return None, {}
    height, width = rgb.shape[:2]
    cropWidth, cropHeight = x1 - x0, y1 - y0
    landmarks = np.asarray(landmarks, dtype=np.float64).copy()
    landmarks[:, 0] = (landmarks[:, 0] * cropWidth + x0) / width
    landmarks[:, 1] = (landmarks[:, 1] * cropHeight + y0) / height
    landmarks[:, 2] *= cropWidth / float(width)
    return landmarks, blendshapes


def trackFaces(imageAt: Callable, frames: List[int], detector, fps: float = 24.0, cancel=None,
               progress: Optional[Callable[[int, int], None]] = None,
               onFrame: Optional[Callable[[int, dict], None]] = None,
               maskAt: Optional[Callable[[int], np.ndarray]] = None) -> Dict[int, dict]:
    """
    The face at each of `frames` (in order): {frame: {"landmarks": (478, 3) normalized, "blendshapes": {...},
    "hidden": [masked out landmarks]}}, the frames without face left out. `detector.detect(rgb uint8, timestamp ms)`
    (see FaceLandmarker) is given a crop around the face of the previous frame, else (or if it finds none there) the
    whole frame, then tiles of it. `maskAt(frame)`: the mask (h, w) over the format, rows from the top: the face is
    searched in its white part only (the rest blackened), and the landmarks it masks out are "hidden" (not used to
    place the head, e.g. behind an occluding object).
    """
    from .imageBuffer import BBox
    from .neural import displayRGB
    faces = {}
    previous = None
    for index, frame in enumerate(frames):
        if cancel:
            cancel()
        image = imageAt(frame)
        rgb = displayRGB(image, BBox.fromFormat(*image.format))
        height, width = rgb.shape[:2]
        mask = np.clip(np.asarray(maskAt(frame), dtype=np.float32), 0.0, 1.0) if maskAt else None
        area = None
        if mask is not None:
            rgb = rgb * mask[:, :, None]
            area = _maskArea(mask)
        regions = _searchRegions(width, height, area) if (mask is None or area is not None) else []
        if previous is not None and regions:
            regions.insert(0, _cropAround(previous, width, height))
        landmarks, blendshapes = None, {}
        timestamp = int(round(index * 1000.0 / max(fps, 1e-3)))
        for region in regions:
            if cancel:
                cancel()
            landmarks, blendshapes = _detectIn(detector, rgb, region, timestamp)
            if landmarks is not None:
                break
        previous = landmarks
        if landmarks is not None:
            faces[frame] = {"landmarks": landmarks, "blendshapes": blendshapes,
                            "hidden": hiddenLandmarks(landmarks, mask) if mask is not None else []}
            if onFrame:
                onFrame(frame, faces[frame])
        if progress:
            progress(index + 1, len(frames))
    return faces


def propagateFaces(imageAt: Callable, frames: List[int], start: np.ndarray, detector=None, flowMethod: str = "lucasKanade",
                   useGpu: bool = True, agreement: float = 0.06, cancel=None,
                   progress: Optional[Callable[[int, int], None]] = None,
                   onFrame: Optional[Callable[[int, dict], None]] = None,
                   maskAt: Optional[Callable[[int], np.ndarray]] = None,
                   regularize: Optional[Callable[[int, np.ndarray, list], np.ndarray]] = None) -> Tuple[Dict[int, dict], str]:
    """
    The face followed from its landmarks `start` (n, 3) normalized at frames[0] through `frames` (consecutive, forward or
    backward) by the optical flow of the plate (`flowMethod`: "lucasKanade" or "raft"): each landmark moves with the
    flow. With a `detector` (MediaPipe, see trackFaces), the face it finds around the flowed landmarks is taken when it
    agrees with them (its landmarks within `agreement` x the size of the face, in median): MediaPipe does not drift,
    the flow keeps the face where MediaPipe is lost or wrong (occlusions, profile, blur). Masked out landmarks are
    "hidden" (as in trackFaces). `regularize(frame, landmarks, hidden)`: the landmarks of the face model fitted on the
    flowed ones (e.g. a stiff face through them): what is followed stays a face (no drifting point breaks it).

    Returns:
        ({frame: {"landmarks", "blendshapes", "hidden", "source": "flow" or "mediapipe"}} of frames[1:], why it
        stopped ("" at the end)).
    """
    from .evaluator import EvaluationCancelled, EvaluationError
    from .flow import computeFlow, sampleRows
    from .imageBuffer import BBox
    from .neural import displayRGB
    faces = {}
    current = np.asarray(start, dtype=np.float64).copy()
    try:
        imageA = imageAt(frames[0])
    except (EvaluationCancelled, EvaluationError) as e:
        return faces, getattr(e, "message", "") or "cancelled"
    width, height = imageA.format
    for index, frame in enumerate(frames[1:], start=1):
        try:
            if cancel:
                cancel()
            imageB = imageAt(frame)
            field = computeFlow(imageA, imageB, cancel=cancel, useGpu=useGpu, method=flowMethod)
            bbox = imageA.bbox
            x, y = current[:, 0] * width, current[:, 1] * height  # Pixels from the top left corner
            # The flow at each landmark (rows of the bbox from its top; Nuke y = height - y)
            motion = sampleRows(field, (bbox.y1 - (height - y) - 0.5).astype(np.float32),
                                (x - bbox.x0 - 0.5).astype(np.float32)).astype(np.float64)
        except EvaluationCancelled:
            return faces, "cancelled"
        except EvaluationError as e:
            return faces, f"frame {frame}: {e.message}"
        flowed = current.copy()
        # The flow is in Nuke pixels (y up): rows go down
        flowed[:, 0] = (x + motion[:, 0]) / width
        flowed[:, 1] = (y - motion[:, 1]) / height
        mask = np.clip(np.asarray(maskAt(frame), dtype=np.float32), 0.0, 1.0) if maskAt else None
        if regularize is not None:
            flowed = regularize(frame, flowed, hiddenLandmarks(flowed, mask) if mask is not None else [])
        landmarks, blendshapes, source = flowed, {}, "flow"
        if detector is not None:
            rgb = displayRGB(imageB, BBox.fromFormat(width, height))
            if mask is not None:
                rgb = rgb * mask[:, :, None]
            found, foundShapes = _detectIn(detector, rgb, _cropAround(flowed, width, height),
                                           int(round(index * 1000.0 / 24.0)))
            if found is not None:
                count = min(len(found), len(flowed))
                size = max(np.ptp(flowed[:count, 0]) * width, np.ptp(flowed[:count, 1]) * height, 1.0)
                distance = np.hypot((found[:count, 0] - flowed[:count, 0]) * width, (found[:count, 1] - flowed[:count, 1]) * height)
                if float(np.median(distance)) <= agreement * size:
                    landmarks, blendshapes, source = found, foundShapes, "mediapipe"
        faces[frame] = {"landmarks": landmarks, "blendshapes": blendshapes, "source": source,
                        "hidden": hiddenLandmarks(landmarks, mask) if mask is not None else []}
        if onFrame:
            onFrame(frame, faces[frame])
        if progress:
            progress(index, len(frames) - 1)
        current, imageA = landmarks, imageB
    return faces, ""


# Storage (the "faceTrack" parameter): {"format": [w, h], "frames": {"1001": {"l": [x, y, z, ...], "b": {...},
# "h": [hidden landmarks] (when there are)}}}

def _dumpFace(face: dict) -> dict:
    data = {"l": [round(float(v), 5) for v in np.asarray(face["landmarks"]).ravel()],
            "b": {name: round(float(value), 4) for name, value in (face.get("blendshapes") or {}).items()}}
    if face.get("hidden"):
        data["h"] = [int(i) for i in face["hidden"]]
    if face.get("source"):
        data["s"] = face["source"]
    return data


def dumpFaces(faces: Dict[int, dict], imageFormat: Tuple[int, int]) -> str:
    if not faces:
        return ""
    return json.dumps({"format": [int(imageFormat[0]), int(imageFormat[1])],
                       "frames": {str(frame): _dumpFace(face) for frame, face in sorted(faces.items())}})


def parseFaces(text: str) -> Tuple[Dict[int, dict], Optional[Tuple[int, int]]]:
    """ (the faces {frame: {"landmarks", "blendshapes"}}, the format they were tracked on) of a "faceTrack" value. """
    if not text or not text.strip():
        return {}, None
    try:
        data = json.loads(text)
        faces = {int(frame): {"landmarks": np.array(face["l"], dtype=np.float64).reshape(-1, 3),
                              "blendshapes": dict(face.get("b") or {}), "hidden": [int(i) for i in face.get("h") or []],
                              "source": face.get("s", "")}
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
    """ The landmarks as tracks of a Tracker (named by their region: lips_013...), keys at the tracked frames (except
    where they are hidden by the mask). """
    regions = landmarkRegions()
    if not faces:
        return []
    count = min(len(face["landmarks"]) for face in faces.values())
    indices = [i for i in (indices if indices is not None else range(count)) if i < count]
    tracks = [{"name": trackName(i, regions), "enabled": True, "pattern": 15, "search": 41, "auto": True,
               "format": [width, height], "keys": {}} for i in indices]
    for frame in sorted(faces):
        pixels = toPixels(landmarksAt(faces, frame, smoothing), width, height)
        hidden = set(faces[frame].get("hidden") or ())
        for track, i in zip(tracks, indices):
            # A masked out landmark has no key at that frame
            if i not in hidden:
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
        # Found without importing mediapipe (its drawing utilities need matplotlib)
        import importlib.util
        spec = importlib.util.find_spec("mediapipe")
        if spec is not None and spec.origin:
            candidates.append(os.path.join(os.path.dirname(spec.origin), "modules", "face_geometry", "data",
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


# Pins: points of the face (in the space of its neutral mesh) placed by hand in the plate, and tracked

def surfaceLocation(vertices: np.ndarray, triangles: np.ndarray, point: np.ndarray) -> Tuple[int, np.ndarray]:
    """ (the triangle, the barycentric coordinates (3,)) of the point of a mesh surface nearest to `point`. """
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    ab, ac, ap = b - a, c - a, np.asarray(point, dtype=np.float64) - a
    # The barycentric coordinates of the point projected on each triangle plane, clamped into the triangle
    d00, d01, d11 = (ab * ab).sum(1), (ab * ac).sum(1), (ac * ac).sum(1)
    d20, d21 = (ap * ab).sum(1), (ap * ac).sum(1)
    denominator = np.where(np.abs(d00 * d11 - d01 * d01) > 1e-20, d00 * d11 - d01 * d01, 1e-20)
    v = np.clip((d11 * d20 - d01 * d21) / denominator, 0.0, 1.0)
    w = np.clip((d00 * d21 - d01 * d20) / denominator, 0.0, 1.0)
    total = np.maximum(v + w, 1.0)
    v, w = v / total, w / total
    closest = a + ab * v[:, None] + ac * w[:, None]
    best = int(np.argmin(((closest - point) ** 2).sum(1)))
    return best, np.array([1.0 - v[best] - w[best], v[best], w[best]])


def pinLocations(mesh: scene3d.GeoData, points: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """ The place of each pin point (neutral face space) on the mesh: (triangle, barycentric coordinates). """
    triangles = mesh.triangles()
    return [surfaceLocation(mesh.vertices, triangles, point) for point in np.asarray(points, dtype=np.float64).reshape(-1, 3)]


def _onSurface(vertices: np.ndarray, triangles: np.ndarray, locations) -> np.ndarray:
    return np.array([vertices[triangles[t]].T @ weights for t, weights in locations]).reshape(-1, 3)


def solveFace(mesh: scene3d.GeoData, pixels: np.ndarray, camera: scene3d.CameraData, width: int, height: int,
              initial: Optional[np.ndarray] = None, rigid=RIGID_LANDMARKS, deform: bool = True, hidden=(),
              pins: Optional[Tuple[np.ndarray, np.ndarray]] = None, pinRadius: float = 2.5, stiffness: float = 0.0):
    """
    The face seen in the image: the pose of the head from the `rigid` landmarks, then each vertex moved at its depth
    onto the ray of its landmark (`deform`). The `hidden` landmarks (masked out) are not used: the head is placed
    with the other rigid ones (all of them if fewer than 6 are left), and their vertices follow the head only.
    `pins`: (points (k, 3) on the neutral face, their image positions (k, 2), Nuke pixels): they place the head with
    the rigid landmarks (counted three times), then, non rigid, the face is moved around each of them (a smooth
    interpolation of their remaining errors, over `pinRadius` units of the mesh) so that it goes through them.
    `stiffness` > 0: the face is not moved vertex by vertex onto its landmarks but as a smooth surface (the moves of
    neighboring vertices kept alike, `regularizedShifts`): a wrong landmark (drift, occlusion) is ignored, the hidden
    ones follow their neighbors, the mesh does not fold; 0: each vertex exactly on its landmark.

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
    hidden = {int(i) for i in hidden}
    rigid = [i for i in rigid if i < count]
    visibleRigid = [i for i in rigid if i not in hidden]
    rigid = visibleRigid if len(visibleRigid) >= 6 else rigid
    focal = camera.focalPixels(width)
    start = initial if initial is not None else _defaultStart(mesh)
    fitPoints, fitPixels = mesh.vertices[rigid], pixels[rigid]
    locations = []
    if pins is not None and len(pins[0]):
        locations = pinLocations(mesh, pins[0])
        pinPoints = _onSurface(mesh.vertices, mesh.triangles(), locations)
        pinPixels = np.asarray(pins[1], dtype=np.float64).reshape(-1, 2)
        # Placed by hand: weighted more than a landmark
        fitPoints = np.r_[fitPoints, np.repeat(pinPoints, 3, axis=0)]
        fitPixels = np.r_[fitPixels, np.repeat(pinPixels, 3, axis=0)]
    cameraToFace, _, error = robustPose(fitPoints, fitPixels, width, height, focal, start)
    head = camera.matrix @ np.linalg.inv(cameraToFace)
    view = np.linalg.inv(cameraToFace)
    inCamera = mesh.vertices @ view[:3, :3].T + view[:3, 3]
    if deform:
        # Each vertex at its depth, on the ray of its landmark (the camera looks along -Z)
        depth = -inCamera[:count, 2]
        moved = np.ones(count, dtype=bool)
        moved[[i for i in hidden if i < count]] = False
        targets = pixels[:count].astype(np.float64)
        if stiffness > 0:
            # A smooth surface through the landmarks: the shifts from the rigid face regularized over the mesh
            safe = np.where(depth > 1e-9, depth, 1e-9)
            rigidPixels = np.c_[width / 2.0 + focal * inCamera[:count, 0] / safe,
                                height / 2.0 + focal * inCamera[:count, 1] / safe]
            shifts = regularizedShifts(mesh, targets - rigidPixels, moved.astype(np.float64), stiffness)
            targets = rigidPixels + shifts
            moved[:] = True
        rays = np.c_[(targets[:, 0] - width / 2.0) / focal, (targets[:, 1] - height / 2.0) / focal, -np.ones(count)]
        inCamera = inCamera.copy()
        inCamera[:count][moved] = (rays * depth[:, None])[moved]
        if locations:
            inCamera = _followPins(inCamera, mesh, locations, pinPixels, width, height, focal, pinRadius)
    world = inCamera @ camera.matrix[:3, :3].T + camera.matrix[:3, 3]
    return scene3d.GeoData(world, name="face", topology=mesh.topology), head, cameraToFace, error


_LAPLACIANS = {}


def _laplacian(mesh: scene3d.GeoData, count: int):
    """ The graph Laplacian (count, count, sparse, normalized: I - D^-1 A) of the first `count` vertices of a mesh. """
    from scipy import sparse
    key = (id(mesh.topology), count)
    cached = _LAPLACIANS.get(key)
    if cached is not None and cached[0] is mesh.topology:
        return cached[1]
    edges = mesh.edges()
    edges = edges[(edges[:, 0] < count) & (edges[:, 1] < count)]
    adjacency = sparse.coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                                  shape=(count, count)).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = sparse.diags(np.where(degree > 0, 1.0 / np.maximum(degree, 1e-12), 0.0))
    laplacian = (sparse.identity(count) - inverse @ adjacency).tocsr()
    if len(_LAPLACIANS) > 16:
        _LAPLACIANS.clear()
    _LAPLACIANS[key] = (mesh.topology, laplacian)
    return laplacian


def regularizedShifts(mesh: scene3d.GeoData, shifts: np.ndarray, weights: np.ndarray, stiffness: float,
                      rounds: int = 4) -> np.ndarray:
    """
    The shifts (n, 2) of the vertices of a face (pixels, from the rigid face) closest to the measured ones (weighted:
    0 for a hidden landmark) while staying smooth over the mesh (the Laplacian of the shifts, times `stiffness`):
    min sum w |s - shift|^2 + stiffness |L s|^2. Robust: a landmark far from the smooth surface of the others (drift,
    wrong detection) is weighted down, again and again.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    count = len(shifts)
    laplacian = _laplacian(mesh, count)
    regularization = float(stiffness) * (laplacian.T @ laplacian) + 1e-6 * sparse.identity(count)
    base = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if base.sum() < 3:
        return np.zeros_like(shifts)
    current = base.copy()
    result = np.zeros_like(shifts)
    for _ in range(rounds):
        system = (sparse.diags(current) + regularization).tocsc()
        result = np.c_[spsolve(system, current * shifts[:, 0]), spsolve(system, current * shifts[:, 1])]
        residuals = np.hypot(*(result - shifts).T)
        measured = base > 0
        scale = max(1.0, 1.4826 * float(np.median(residuals[measured])))
        # Cauchy weights: within 3 typical errors, nearly 1
        current = base / (1.0 + (residuals / (3.0 * scale)) ** 2)
    return result


def _followPins(inCamera: np.ndarray, mesh: scene3d.GeoData, locations, pinPixels: np.ndarray, width: int, height: int,
                focal: float, radius: float) -> np.ndarray:
    """
    The face (vertices in camera space) moved in the image, each vertex at its depth, so that the pins go through
    their positions: their errors interpolated over the neutral face by gaussians of `radius` (exact at the pins).
    """
    def project(points):
        z = np.where(points[:, 2] < -1e-9, -points[:, 2], 1e-9)
        return np.c_[width / 2.0 + focal * points[:, 0] / z, height / 2.0 + focal * points[:, 1] / z], z
    triangles = mesh.triangles()
    centers = _onSurface(mesh.vertices, triangles, locations)
    radius = max(float(radius), 1e-6)
    # The shift of each vertex for a unit error of each pin (gaussians around the pins), and what it gives at the pins
    # (interpolated from the vertices of their triangle): solved so that each pin lands on its position
    vertexKernel = np.exp(-((mesh.vertices[:, None, :] - centers[None, :, :]) ** 2).sum(-1) / (radius * radius))
    atPins = np.array([weights @ vertexKernel[triangles[t]] for t, weights in locations])
    count = len(inCamera)
    moved = inCamera.copy()
    # Twice: the perspective makes the first step a little short
    for _ in range(2):
        current, _ = project(_onSurface(moved, triangles, locations))
        try:
            weights = np.linalg.solve(atPins + 1e-6 * np.eye(len(centers)), pinPixels - current)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(atPins, pinPixels - current, rcond=None)[0]
        shift = vertexKernel @ weights
        n = min(count, len(shift))
        xy, depth = project(moved[:n])
        xy = xy + shift[:n]
        moved[:n] = np.c_[(xy[:, 0] - width / 2.0) / focal, (xy[:, 1] - height / 2.0) / focal, -np.ones(n)] * depth[:, None]
    return moved
