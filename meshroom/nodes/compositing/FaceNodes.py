"""
FaceTracker: a face followed in the plate, non rigid (MediaPipe Face Landmarker, see meshroom/live/face.py).
"""
__version__ = "0.1"

import numpy as np

from meshroom.core import desc
from meshroom.live import scene3d
from meshroom.live.gpu import gpuParam
from meshroom.live.params import imageInput, maskParams, trackRangeParams

# "MediaPipe Expressions" of 1: the scores of MediaPipe weigh about as much as the landmarks in the fit of the shapes
EXPRESSION_PRIOR_SCALE = 50.0

_COLOR = "#6E9E8A"


class FaceTracker(desc.ImageOpNode):
    category = "Comp 3D"
    color = _COLOR
    # The tracks edited in the viewer (and tracked by its Track buttons) are the pins; "tracks" are the landmarks
    editableTracks = "pins"
    documentation = """
Tracks a face in the plate, mouth and eyes included (non rigid), like KeenTools FaceTracker.

- "Track Face": MediaPipe Face Landmarker finds the 478 landmarks of the face at every frame of the plate, with 52
  expression coefficients (it looks around the face of the previous frame, else in the whole frame, then in tiles:
  a small face in a large plate is found; dark frames are brightened). They are stored in the node.
- The face mesh is then fitted at each frame: the head pose from the landmarks that move little with the expressions
  (forehead, nose bridge, eye corners, temples), then, with "Non Rigid", every vertex moved at its depth onto its
  landmark: the lips and the eyelids follow.
- "Face": any mesh with the MediaPipe face topology (468 or 478 vertices in the landmark order), e.g. a scan of the
  actor retopologized on it, or a head of the FaceBuilder; its shape is the neutral face. Without it: the canonical
  face of MediaPipe.
- "Mesh": the face only, or the head: the face with ears, skull and neck that follow it (they are not fitted: the face
  is, the rest of the head goes with it).

Outputs: "Face" (the face mesh at each frame, in the world of the camera), "Head Matrix" (the rigid pose of the head:
face space to world, e.g. for the "Input Matrix" of a ReadGeo), "Camera" (the plate camera), and the plate with the face
drawn. Pins help where MediaPipe is wrong or lost (occlusions, profile, motion blur), as in GeoTracker: in the viewer, "Add
pin" and click on the face, then drag the pin onto its place in the plate; track the pins forward / backward like
tracks. They place the head with the landmarks (weighted more), then, with "Non Rigid", the face around each pin is
moved until it goes through it (smoothly, over "Pin Radius"): a pin on a lip corner pulls the lip, not the head.

The optional "Mask" limits the search to its white part and leaves out the landmarks it masks out (an
occluding object: its roto, inverted). Its tracks (the landmarks, named by region: lips_013, leftEye_033...) can drive a Roto, a Stabilize or a
CornerPin like those of a Tracker: connect it to their "Tracker" input.

Needs MediaPipe (pip install mediapipe) and its model in MESHROOM_MODELS_PATH
(python -m meshroom.live.neural.download facelandmarker).
"""
    inputs = [
        imageInput(description="The plate."),
        desc.Camera(name="camera", label="Camera", optional=True,
                    description="The lens of the plate and its pose (the world). Without it: 35 mm on a 36 mm film "
                                "back, at the origin, looking along -Z."),
        desc.Geo(name="face", label="Face", optional=True,
                 description="A face mesh with the MediaPipe face topology (468 or 478 vertices in the landmark order), "
                             "e.g. a scan of the actor retopologized on it, or a head of the FaceBuilder. Without it: "
                             "the canonical face of MediaPipe."),
        desc.ChoiceParam(name="meshType", label="Mesh", value="head", values=["head", "face"],
                         description="head: the face with ears, skull and neck following it (only the face is fitted). "
                                     "face: the face mesh only."),
        desc.Image(name="mask", label="Mask", optional=True, infoOnly=True,
                   description="Optional mask: the face is searched in its white part only, and the landmarks it "
                               "masks out (e.g. behind foliage: a roto of it, inverted) do not place the head nor "
                               "move their vertices, and give no track key."),
        *maskParams(),
        desc.FloatParam(name="minConfidence", label="Min Confidence", value=0.5, range=(0.1, 0.95, 0.05),
                        invalidate=False,
                        description="A face is found (and followed) when MediaPipe is at least this confident."),
        desc.BoolParam(name="detectLying", label="Detect Lying Faces", value=True, invalidate=False,
                       description="A face lying on its side (turned more than 45 degrees in the image) is searched in "
                                   "the image turned by a quarter or half turn, so that MediaPipe sees it upright. Off: "
                                   "the image as it is."),
        desc.ChoiceParam(name="exportTracks", label="Tracks", value="features", values=["features", "all", "none"],
                         invalidate=False,
                         description="Landmarks given as tracks by 'Track Face': the features (lips, eyes, eyebrows, "
                                     "irises, nose, face oval), all 478, or none."),
        desc.ChoiceParam(name="trackWith", label="Track With", value="mediapipe+flow",
                         values=["mediapipe", "flow", "mediapipe+flow"], invalidate=False,
                         description="mediapipe: MediaPipe finds the face at each frame on its own. flow: the face of the "
                                     "current frame (fitted, pins included) follows the optical flow of the plate. "
                                     "mediapipe+flow: followed by the flow, MediaPipe taken where it agrees with it (no "
                                     "drift, and the flow where MediaPipe is lost or wrong: occlusions, profile, blur). "
                                     "With flow, start from a well placed frame."),
        desc.ChoiceParam(name="flowMethod", label="Optical Flow", value="lucasKanade", values=["lucasKanade", "raft"],
                         invalidate=False,
                         description="The optical flow followed: Lucas-Kanade, or the RAFT network (large motions, "
                                     "sharp edges; PyTorch and its weights needed, else Lucas-Kanade)."),
        desc.BoolParam(name="followSurface", label="Follow Head Surface", value=True, invalidate=False,
                       description="With flow or mediapipe+flow: the pose of the head is followed by the surface of "
                                   "its rigid parts (forehead, nose bridge, temples, cheekbones: many textured points, "
                                   "as the Resection follows a geometry, the masked parts left out), not by the "
                                   "landmarks: much less drift; the landmarks then only give the expression. Kept with "
                                   "each tracked frame (the keyframes with rigid pins are placed by their pins)."),
        gpuParam(),
        desc.PushButtonParam(name="trackFaceBackward", label="Track Face Backward", semantic="liveAction",
                             description="Track the face from the current frame back to the start of the track range."),
        desc.PushButtonParam(name="trackFaceForward", label="Track Face Forward", semantic="liveAction",
                             description="Track the face from the current frame to the end of the track range."),
        desc.PushButtonParam(name="trackFaceStepBackward", label="Track Face One Frame Backward", semantic="liveAction",
                             description="Track the face to the previous frame only (and show it)."),
        desc.PushButtonParam(name="trackFaceStepForward", label="Track Face One Frame Forward", semantic="liveAction",
                             description="Track the face to the next frame only (and show it)."),
        desc.PushButtonParam(name="refineFace", label="Refine", semantic="liveAction",
                             description="Track the face again between each pair of keyframes (frames with pins), from "
                                         "both ends, and blend the two."),
        desc.PushButtonParam(name="trackFace", label="Track Face", semantic="liveAction",
                             description="Find the face at every frame of the plate (MediaPipe Face Landmarker)."),
        desc.PushButtonParam(name="clearFace", label="Clear", semantic="liveAction",
                             description="Remove the tracked face."),
        *trackRangeParams(),
        desc.StringParam(name="pins", label="Pins", value="[]", semantic="tracks",
                         description="Pins as JSON: a tracked image position and its point on the neutral face "
                                     "(\"point\": [x, y, z]); \"expression\": true for an expression pin. Edited in "
                                     "the viewer: a rigid pin (cyan, the default) only places the head (rotation, "
                                     "position); an expression pin (magenta, toggled in the viewer) deforms the face "
                                     "(rig model: its controls, then a local warp with 'Exact Pins')."),
        desc.ChoiceParam(name="deformModel", label="Non Rigid Model", value="arkit", values=["arkit", "rig", "surface"],
                         description="arkit: the face moved by the 51 expression shapes of ARKit (eyeBlinkLeft, "
                                     "jawOpen, mouthSmileRight...; from ICT-FaceKit), each weighted from 0 to 1: only "
                                     "expressions a face can make, the skull and nose bridge rigid; the pins move the "
                                     "shapes. rig: a simpler rig of a few controls placed on the features (jaw, brows, "
                                     "eyelids, lips, mouth corners, cheeks). surface: every vertex free, a smooth "
                                     "surface through the landmarks (finer, less plausible)."),
        desc.FloatParam(name="expressionPrior", label="MediaPipe Expressions", value=0.1, range=(0.0, 1.0, 0.01),
                        enabled=lambda node: node.deformModel.value == "arkit",
                        description="With the arkit model: how much the expressions MediaPipe reads (its blendshape "
                                    "scores, frames found by MediaPipe) guide the shapes, besides the landmarks. 0: "
                                    "the landmarks only."),
        desc.FloatParam(name="stiffness", label="Stiffness", value=0.3, range=(0.0, 5.0, 0.05),
                        description="With 'Non Rigid': rig: how much the controls stay at rest (less: stronger "
                                    "expressions, more noise). surface: how much the face keeps a smooth shape (0: "
                                    "each vertex exactly on its landmark, a wrong landmark folds the mesh; higher: a "
                                    "smooth surface through the landmarks, wrong or hidden ones following their "
                                    "neighbors)."),
        desc.BoolParam(name="showPinReach", label="Show Pin Reach", value=True,
                       description="With 'Non Rigid': the face is tinted where the pins move it (orange at a pin, "
                                   "fading over 'Pin Radius')."),
        desc.BoolParam(name="exactPins", label="Exact Pins", value=True,
                       description="With 'Non Rigid' rig: after the rig, the face is warped locally (over 'Pin Radius') "
                                   "so that each pin is exactly on its position, on the parts the rig moves only "
                                   "(mouth, jaw, eyelids, brows, cheeks, nose tip): a pin on a rigid part (forehead, "
                                   "nose bridge, temples, the sides of the head) only orients the head. Off: the pins "
                                   "orient the head and move the rig controls, nothing else."),
        desc.FloatParam(name="pinRadius", label="Pin Radius", value=2.5, range=(0.1, 20.0, 0.1),
                        description="How far around a pin the face is moved to go through it (non rigid), in the "
                                    "units of the face mesh (centimeters for the canonical face)."),
        desc.FloatParam(name="smoothing", label="Landmark Smoothing", value=1.0, range=(0.0, 5.0, 0.1),
                        description="Gaussian smoothing of the landmarks in time, in frames (0: as tracked)."),
        desc.BoolParam(name="useReference", label="Use Reference Frame", value=False,
                       description="Fit the face from the reference frame (where it is seen at its best, placed with "
                                   "pins): its fitted face becomes the shape of the actor that the other frames only "
                                   "change the expression of, and they are fitted from it outward, each from the "
                                   "previous one (a steady pose)."),
        desc.IntParam(name="referenceFrame", label="Reference Frame", value=1001, range=(0, 10000, 1),
                      enabled=lambda node: node.useReference.value,
                      description="The reference frame ('Set Reference Frame': the current frame)."),
        desc.PushButtonParam(name="setReference", label="Set Reference Frame", semantic="liveAction",
                             description="The current frame becomes the reference frame."),
        desc.PushButtonParam(name="fitShape", label="Fit Shape", semantic="liveAction",
                             description="Find the shape of the actor's face from the tracked frames of the track range "
                                         "(seen from several angles; the keyframes with pins count more): one rest face "
                                         "for the whole shot, the rig then only changes its expression (as the head "
                                         "of KeenTools FaceBuilder). Track the face first; fit again after new "
                                         "keyframes."),
        desc.PushButtonParam(name="clearShape", label="Clear Shape", semantic="liveAction",
                             description="Back to the face mesh as it is (no fitted shape)."),
        desc.StringParam(name="faceShape", label="Face Shape", value="", semantic="faceShape", advanced=True,
                         description="The shape of the actor found by 'Fit Shape': offsets of the vertices of the face "
                                     "mesh (JSON)."),
        desc.FloatParam(name="temporalSmoothing", label="Temporal Smoothing", value=1.5, range=(0.0, 6.0, 0.1),
                        description="Gaussian smoothing in time (frames) of the fitted face: its head pose and its shape "
                                    "in face space (the expression), so that it moves as a face does: no jitter, no part "
                                    "turning back and forth in depth. The keyframes keep their own fit. 0: each frame on "
                                    "its own."),
        desc.BoolParam(name="deform", label="Non Rigid", value=True,
                       description="The face follows the expressions (mouth, eyes); off: only the rigid head pose."),
        desc.ChoiceParam(name="display", label="Display", value="wireframe",
                         values=["shaded", "edges", "wireframe", "points", "landmarks", "none"],
                         description="What is drawn over the plate: the face mesh (shaded, its edges, its wireframe or "
                                     "its vertices), the landmarks, or nothing."),
        desc.FloatParam(name="opacity", label="Opacity", value=0.6, range=(0.0, 1.0, 0.01),
                        description="Opacity of the face drawn over the plate."),
        desc.BoolParam(name="showSurfaceTracks", label="Show Surface Tracks", value=True, invalidate=False,
                       description="Some of the points followed on the head by 'Follow Head Surface' at the frame (one in "
                                   "ten): green where they fit the head pose, red where they were rejected."),
        desc.StringParam(name="faceTrack", label="Face Track", value="", semantic="faceTrack", visible=False,
                         advanced=True, description="The landmarks of each frame (JSON, written by 'Track Face')."),
        desc.StringParam(name="tracks", label="Landmark Tracks", value="[]", semantic="autoTracks",
                         description="The landmarks as tracks (JSON), for the Roto, Stabilize, CornerPin."),
    ]
    outputs = [
        desc.Geo(name="geoOutput", label="Face", description="The face mesh at each frame, in the world of the camera."),
        desc.Matrix(name="headMatrix", label="Head Matrix", description="The pose of the head: face space to world."),
        desc.Camera(name="cameraOutput", label="Camera", description="The plate camera (the input camera, or the default one)."),
        desc.Geo(name="shapeOutput", label="Shape",
                 description="The rest face (face space): the face mesh with the shape of the actor found by 'Fit "
                             "Shape' (e.g. to write it, and connect it to 'Face' of other FaceTrackers)."),
        desc.Image(name="output", label="Output", description="The plate, with the face drawn over it."),
    ]

    def isIdentity(self, ctx):
        return "input" if ctx.value("display") == "none" else None

    # The face

    def camera(self, ctx, frame=None):
        """ The plate camera: the "camera" input, or 35 mm on a 36 mm film back at the origin. """
        camera = ctx.inputData("camera", frame)
        return camera if camera is not None else scene3d.CameraData(np.eye(4), 35.0, 36.0, 24.0)

    def inputMesh(self, ctx, frame=None):
        """ The "face" input (a face or a head of MediaPipe topology), or the canonical face. """
        from meshroom.live import face
        mesh = ctx.inputData("face", frame)
        try:
            mesh = mesh if mesh is not None else face.canonicalFace()
            face.checkTopology(mesh)
        except face.FaceUnavailable as e:
            raise RuntimeError(str(e)) from e
        return mesh

    def baseMesh(self, ctx, frame=None):
        """ The face mesh (MediaPipe topology) fitted: the "face" input (the face of a head), or the canonical face. """
        from meshroom.live import face
        return face.facePart(self.inputMesh(ctx, frame))

    def headRest(self, ctx, frame=None):
        """ The rest head ("Mesh" head): the rest face (`mesh`, with its fitted shape) with ears, skull and neck; None
        for the face only. """
        from meshroom.live import face
        if ctx.value("meshType") != "head":
            return None
        rest = self.mesh(ctx, frame)
        source = self.inputMesh(ctx, frame)
        if face.isHead(source):
            # A head connected: its own ears, skull and neck, following the fitted shape
            return scene3d.GeoData(face.HeadFollow.of(source.vertices).follow(rest.vertices), name="head",
                                   topology=source.topology)
        return face.headOf(rest)

    def withHead(self, ctx, local, frame=None):
        """ The vertices (face space) of the head of a fitted face (`local`: its vertices), and its topology; the
        face as it is for "Mesh" face. """
        from meshroom.live import face
        rest = self.headRest(ctx, frame)
        if rest is None:
            return local, self.mesh(ctx, frame).topology
        # An expression: the skull follows the face rigidly (its shape is that of the rest head)
        return face.HeadFollow.of(rest.vertices).follow(local, rigid=True), rest.topology

    _SHAPED = {}

    def mesh(self, ctx, frame=None):
        """ The rest face: the base mesh with the shape of the actor ("Fit Shape") when it was fitted. """
        from meshroom.live import face
        base = self.baseMesh(ctx, frame)
        text = ctx.value("faceShape")
        if not text:
            return base
        key = (id(base.topology), hash(base.vertices.tobytes()), hash(text))
        shaped = self._SHAPED.get(key)
        if shaped is None:
            try:
                offsets = face.parseShape(text)
            except ValueError as e:
                raise RuntimeError(str(e)) from e
            if offsets is None or len(offsets) != len(base.vertices):
                return base  # Fitted on another mesh
            if len(self._SHAPED) > 8:
                self._SHAPED.clear()
            shaped = self._SHAPED[key] = scene3d.GeoData(base.vertices + offsets, name=base.name, topology=base.topology)
        return shaped

    def drawSurfaceTracks(self, ctx, rgba, image):
        """ The points followed on the surface of the head at the frame (Follow Head Surface), in place. """
        try:
            faces, trackFormat = self._parsedFaces(ctx)
        except ValueError:
            return
        points = (faces.get(ctx.frame) or {}).get("surface")
        if points is None or not len(points):
            return
        xy = points[:, :2].copy()
        if trackFormat and tuple(trackFormat) != (image.width, image.height):
            xy *= [image.width / float(trackFormat[0]), image.height / float(trackFormat[1])]
        fits = points[:, 2] > 0.5
        scene3d.drawPoints(rgba, image.bbox, xy[fits], (0.3, 1.0, 0.3), 1.0, 3)
        scene3d.drawPoints(rgba, image.bbox, xy[~fits], (1.0, 0.25, 0.2), 1.0, 3)

    # The last face track parsed for display: (its text, (faces, format))
    _displayedTrack = (None, ({}, None))

    @classmethod
    def _parsedFaces(cls, ctx):
        """ The faces and format of the "faceTrack" parameter, parsed once for the frames displayed. """
        from meshroom.live import face
        text = ctx.value("faceTrack")
        entry = cls._displayedTrack
        if entry[0] != text:
            entry = cls._displayedTrack = (text, face.parseFaces(text))
        return entry[1]

    def faces(self, ctx):
        from meshroom.live import face
        try:
            return face.parseFaces(ctx.value("faceTrack"))[0]
        except ValueError as e:
            raise RuntimeError(str(e)) from e

    @staticmethod
    def pins(ctx, info=None):
        """ The enabled pins with a point on the face, in the pixels of the plate. """
        from meshroom.live import tracking
        try:
            tracks = tracking.parseTracks(ctx.value("pins"))
        except ValueError as e:
            raise RuntimeError(f"Invalid pins: {e}") from e
        info = info or ctx.inputInfo("input")
        if info is not None:
            tracks = tracking.scaleTracks(tracks, info.width, info.height)
        return [track for track in tracks if track.get("enabled", True) and len(track.get("point") or []) == 3]

    @staticmethod
    def pinsAt(pins, frame):
        """ (points on the neutral face (k, 3), image positions (k, 2)) of the pins with a key at this frame (placed by
        hand or tracked): as in GeoTracker, a pin only constrains the frames where it has a key. """
        points, positions = [], []
        for pin in pins:
            key = (pin.get("keys") or {}).get(str(int(frame)))
            if key is None or int(frame) != frame:
                continue
            points.append(pin["point"])
            positions.append((float(key[0]), float(key[1])))
        return np.array(points, dtype=np.float64).reshape(-1, 3), np.array(positions, dtype=np.float64).reshape(-1, 2)

    @staticmethod
    def isExpressionPin(pin) -> bool:
        """ An expression pin deforms the face (mouth, eyelids...); a rigid pin (the default) only places the head. """
        return bool(pin.get("expression", False))

    @classmethod
    def solvePinsAt(cls, pins, frame):
        """ The pins of a frame for face.solveFace: (points, positions, expression flags), or None without pins. """
        keyed = [pin for pin in pins if int(frame) == frame and (pin.get("keys") or {}).get(str(int(frame))) is not None]
        points, positions = cls.pinsAt(keyed, frame)
        if not len(points):
            return None
        return points, positions, np.array([cls.isExpressionPin(pin) for pin in keyed], dtype=bool)

    def pinProjections(self, ctx, cachedData) -> dict:
        """ {pin name: [x, y]}: where each pin is seen at the frame, on the fitted face (`cachedData(output)`: the cached
        output at the frame, or None), in the pixels of the plate of `ctx`. """
        from meshroom.live import face
        geo = cachedData("geoOutput")
        info = ctx.inputInfo("input")
        pins = self.pins(ctx, info) if info is not None else []
        if geo is None or geo.isEmpty or not pins:
            return {}
        mesh = self.mesh(ctx)
        triangles = mesh.triangles()
        locations = face.pinLocations(self.baseMesh(ctx), np.array([pin["point"] for pin in pins], dtype=np.float64))
        onFace = np.array([geo.vertices[triangles[t]].T @ w for t, w in locations])
        xy, _, inFront = self.camera(ctx).project(onFace, info.width, info.height)
        return {pin["name"]: [float(x), float(y)] for pin, (x, y), front in zip(pins, xy, inFront) if front}

    def pinReachColors(self, ctx, info, count):
        """ The colors of the vertices showing where the pins of the frame move the face ("Show Pin Reach"): from the
        face color to orange at a pin, over "Pin Radius"; None when not shown (rigid, no pin at this frame). """
        from meshroom.live import face
        if not (ctx.value("showPinReach") and ctx.value("deform")):
            return None
        # The expression pins of this frame (a keyframe): the others do not move the face here, the rigid ones never
        pins = [pin for pin in self.pins(ctx, info)
                if self.isExpressionPin(pin) or not face.isRigModel(ctx.value("deformModel"))]
        points, _ = self.pinsAt(pins, ctx.frame)
        if not len(points):
            return None
        mesh = self.mesh(ctx)
        locations = face.pinLocations(self.baseMesh(ctx), points)
        triangles = mesh.triangles()
        if face.isRigModel(ctx.value("deformModel")):
            # The areas of the controls (or shapes) under each pin (what the pin moves; a pin on a rigid part moves none)
            try:
                rig = face.rigFor(mesh, ctx.value("deformModel"))
            except ValueError:
                return None
            areas = rig.areas
            reach = np.zeros(len(mesh.vertices))
            for t, w in locations:
                share = w @ areas[triangles[t]]
                reach = np.maximum(reach, np.clip(areas @ share / max(float(share @ share), 1e-6), 0.0, 1.0)
                                   * (share.max() > 0.05))
            # And the local warp around each pin ("Exact Pins": what the rig cannot do), on what the rig moves only
            movable = [(t, w) for t, w in locations if float(w @ rig.mobility[triangles[t]]) >= 0.2]
            if ctx.value("exactPins") and movable:
                centers = face._onSurface(mesh.vertices, triangles, movable)
                radius = max(float(ctx.value("pinRadius")), 1e-6)
                distances = np.linalg.norm(mesh.vertices[:, None, :] - centers[None, :, :], axis=2)
                reach = np.maximum(reach, np.exp(-(distances.min(axis=1) / radius) ** 2) * rig.mobility)
            reach = reach[:count, None]
        else:
            centers = face._onSurface(mesh.vertices, triangles, locations)
            radius = max(float(ctx.value("pinRadius")), 1e-6)
            distances = np.linalg.norm(mesh.vertices[:, None, :] - centers[None, :, :], axis=2)
            reach = np.exp(-(distances.min(axis=1) / radius) ** 2)[:count, None]
        colors = np.array([0.35, 0.8, 1.0]) * (1.0 - reach) + np.array([1.0, 0.5, 0.1]) * reach
        if len(colors) < count:
            colors = np.r_[colors, np.tile([0.35, 0.8, 1.0], (count - len(colors), 1))]
        return colors

    def pickPoint(self, ctx, x, y):
        """ The point of the neutral face under the image position (x, y) (where the fitted face is seen), or None. """
        from meshroom.live import face
        info = ctx.inputInfo("input")
        geo = ctx.ownData("geoOutput", ctx.frame)
        if info is None or geo is None or geo.isEmpty:
            raise RuntimeError("Track the face first (Track Face): the pins are put on it.")
        origin, direction = self.camera(ctx).ray(x, y, info.width, info.height)
        hit = scene3d.raycast(geo, origin, direction)
        if hit is None:
            return None
        mesh = self.baseMesh(ctx)
        triangle, weights = face.surfaceLocation(geo.vertices, geo.triangles(), hit)
        corners = geo.triangles()[triangle]
        if corners.max() >= len(mesh.vertices):
            # On the ears, skull or neck: the pins are on the face (what is fitted)
            return None
        # The same place on the base face mesh (the pins stay valid when the shape is fitted again)
        return mesh.vertices[corners].T @ weights

    def reference(self, ctx):
        """ The reference frame ("Use Reference Frame"), or None. """
        return int(ctx.value("referenceFrame")) if ctx.value("useReference") else None

    def _fit(self, ctx, frame, rest, initial):
        """ The face fitted at a frame on the `rest` shape (the neutral face, or the face of the reference frame):
        (vertices in face space, head matrix); None before tracking. """
        from meshroom.live import face
        faces = self.faces(ctx)
        info = ctx.inputInfo("input")
        if not faces or info is None:
            return None
        landmarks = face.landmarksAt(faces, frame, ctx.value("smoothing"))
        pixels = face.toPixels(landmarks, info.width, info.height)
        tracked = faces.get(frame) or faces[min(faces, key=lambda f: (abs(f - frame), f))]
        solvePins = self.solvePinsAt(self.pins(ctx, info), frame)
        neutral = self.baseMesh(ctx, frame)
        if solvePins is not None and rest is not neutral:
            # The pins are points of the base face mesh: the same places on the rest shape
            solvePins = (face._onSurface(rest.vertices, rest.triangles(), face.pinLocations(neutral, solvePins[0])),
                         solvePins[1], solvePins[2])
        camera = self.camera(ctx, frame)
        prior, priorWeight = self.expressionPrior(ctx, rest, tracked)
        geo, head, _, _ = face.solveFace(rest, pixels, camera, info.width, info.height, initial=initial,
                                         rigid=face.RIGID_LANDMARKS, deform=ctx.value("deform"),
                                         hidden=tracked.get("hidden") or (), pins=solvePins,
                                         pinRadius=ctx.value("pinRadius"), stiffness=ctx.value("stiffness"),
                                         model=ctx.value("deformModel"), exactPins=ctx.value("exactPins"),
                                         pose=(faces.get(frame) or {}).get("pose"), prior=prior, priorWeight=priorWeight)
        inverse = np.linalg.inv(head)
        return geo.vertices @ inverse[:3, :3].T + inverse[:3, 3], head

    @staticmethod
    def expressionPrior(ctx, rest, tracked):
        """ (the ARKit shapes MediaPipe read on a tracked frame, their weight) for face.solveFace: (None, 0) without
        (another model, a frame followed by the flow, "MediaPipe Expressions" at 0). """
        from meshroom.live import face
        weight = float(ctx.value("expressionPrior"))
        if ctx.value("deformModel") != "arkit" or weight <= 0 or not (tracked or {}).get("blendshapes"):
            return None, 0.0
        try:
            prior = face.rigFor(rest, "arkit").fromMediaPipe(tracked["blendshapes"])
        except ValueError:
            return None, 0.0
        # 1: as much as the landmarks (the prior is one row per shape, the landmarks 468 points)
        return (prior, weight * EXPRESSION_PRIOR_SCALE) if prior is not None else (None, 0.0)

    def rawSolve(self, ctx, frame):
        """
        The face fitted at a frame on its own (no temporal smoothing): (its vertices in face space (n, 3), the head
        matrix), cached; None before tracking. Each fit starts from the head pose of a neighbor frame (continuity: no
        flip of the pose between frames). With a reference frame: the frames are fitted from it outward, each from the
        previous one, on the face of the reference frame (the shape of the actor seen at its best) instead of the
        neutral face: the other frames only change its expression.
        """
        evaluator, uid = ctx.evaluator, ctx.planNode.uid
        cached = evaluator.cachedData(uid, "_rawFace", frame)
        if cached is not None:
            return cached.vertices, evaluator.cachedData(uid, "_rawHead", frame)
        faces = self.faces(ctx)
        if not faces or ctx.inputInfo("input") is None:
            return None
        reference = self.reference(ctx)

        def store(f, result):
            evaluator.storeData(uid, "_rawHead", f, result[1])
            evaluator.storeData(uid, "_rawFace", f, scene3d.GeoData(result[0]))
            return result
        if reference is None:
            camera = self.camera(ctx, frame)
            initial = None
            for neighbor in (frame - 1, frame + 1):
                head = evaluator.cachedData(uid, "_rawHead", neighbor)
                if head is not None:
                    # camera-to-face = inverse(head) @ camera
                    initial = np.linalg.inv(head) @ camera.matrix
                    break
            return store(frame, self._fit(ctx, frame, self.mesh(ctx, frame), initial))
        # The reference frame on the neutral face, then frame after frame toward `frame` on the reference face
        neutral = self.mesh(ctx, frame)
        referenceFit = evaluator.cachedData(uid, "_rawFace", reference)
        if referenceFit is None:
            store(reference, self._fit(ctx, reference, neutral, None))
        if frame == reference:
            return self.rawSolve(ctx, frame)
        rest = scene3d.GeoData(evaluator.cachedData(uid, "_rawFace", reference).vertices, topology=neutral.topology)
        step = 1 if frame > reference else -1
        previous = reference
        result = None
        for f in range(reference + step, frame + step, step):
            ctx.checkCancelled()
            cached = evaluator.cachedData(uid, "_rawFace", f)
            if cached is not None and f != frame:
                previous = f
                continue
            head = evaluator.cachedData(uid, "_rawHead", previous)
            initial = np.linalg.inv(head) @ self.camera(ctx, f).matrix if head is not None else None
            result = store(f, self._fit(ctx, f, rest, initial))
            previous = f
        return result

    def solve(self, ctx, frame=None):
        """
        (the face GeoData in world space, the head matrix) at a frame; an empty GeoData and the identity before
        tracking. With "Temporal Smoothing", the head pose and the shape of the face (in face space: its expression)
        are averaged over the neighbor frames (gaussian): no jitter of the pose, no part of the face turning back and
        forth in depth from one frame to the next. The keyframes (frames with pins) keep their own fit.
        """
        from scipy.spatial.transform import Rotation
        frame = ctx.frame if frame is None else frame
        raw = self.rawSolve(ctx, frame)
        if raw is None:
            return scene3d.GeoData(), np.eye(4)
        sigma = float(ctx.value("temporalSmoothing"))
        info = ctx.inputInfo("input")
        keyed = len(self.pinsAt(self.pins(ctx, info), frame)[0]) > 0 or frame == self.reference(ctx)
        local, head = raw
        if sigma > 0 and not keyed:
            faces = self.faces(ctx)
            first, last = min(faces), max(faces)
            radius = int(np.ceil(2.0 * sigma))
            neighbors = [f for f in range(frame - radius, frame + radius + 1) if first <= f <= last]
            samples = [(f, self.rawSolve(ctx, f)) for f in neighbors]
            samples = [(f, value) for f, value in samples if value is not None and len(value[0]) == len(local)]
            if len(samples) > 1:
                weights = np.array([np.exp(-0.5 * ((f - frame) / sigma) ** 2) for f, _ in samples])
                weights /= weights.sum()
                local = np.einsum("k,knd->nd", weights, np.stack([value[0] for _, value in samples]))
                heads = np.stack([value[1] for _, value in samples])
                head = np.eye(4)
                # The rotations averaged (quaternions), the translations too; the scale of the head kept
                scales = np.linalg.norm(heads[:, :3, :3], axis=1)
                rotation = Rotation.from_matrix(heads[:, :3, :3] / scales[:, None, :]).mean(weights).as_matrix()
                head[:3, :3] = rotation * np.einsum("k,kd->d", weights, scales)[None, :]
                head[:3, 3] = np.einsum("k,kd->d", weights, heads[:, :3, 3])
        local, topology = self.withHead(ctx, local, frame)
        world = local @ head[:3, :3].T + head[:3, 3]
        return scene3d.GeoData(world, name="face", topology=topology), head

    def computeData(self, ctx, outputName):
        if outputName == "cameraOutput":
            return self.camera(ctx)
        if outputName == "geoOutput":
            return self.solve(ctx)[0]
        if outputName in ("headMatrix", "matrix"):
            return self.solve(ctx)[1]
        if outputName == "shapeOutput":
            return self.headRest(ctx) or self.mesh(ctx)
        return super().computeData(ctx, outputName)

    def render(self, ctx, inputs):
        from meshroom.live import face, geoRender
        image = inputs["input"]
        rgba = np.array(image.rgba(), dtype=np.float32)
        width, height = image.width, image.height
        display, alpha = ctx.value("display"), ctx.value("opacity")
        if display == "landmarks":
            faces = self.faces(ctx)
            if faces:
                pixels = face.toPixels(face.landmarksAt(faces, ctx.frame, ctx.value("smoothing")), width, height)
                scene3d.drawPoints(rgba, image.bbox, pixels, (1.0, 0.85, 0.2), alpha, 2)
        else:
            geo = ctx.ownData("geoOutput", ctx.frame)
            if geo is not None and not geo.isEmpty:
                from meshroom.live import gpu
                geoRender.drawGeometry(rgba, image.bbox, geo, self.camera(ctx), width, height, display,
                                       (0.35, 0.8, 1.0), alpha, creaseDegrees=60.0, useGpu=gpu.enabled(ctx),
                                       vertexColors=self.pinReachColors(ctx, image.info, len(geo.vertices)), quads=True)
        if ctx.value("showSurfaceTracks"):
            self.drawSurfaceTracks(ctx, rgba, image)
        # The pins: where the face puts their point, linked to where they are (the error)
        geo = ctx.ownData("geoOutput", ctx.frame)
        points, positions = self.pinsAt(self.pins(ctx, image.info), ctx.frame)
        if len(points) and geo is not None and not geo.isEmpty:
            mesh = self.mesh(ctx)
            triangles = mesh.triangles()
            onFace = np.array([geo.vertices[triangles[t]].T @ w for t, w in face.pinLocations(self.baseMesh(ctx), points)])
            xy, _, _ = self.camera(ctx).project(onFace, width, height)
            scene3d.drawSegments(rgba, image.bbox, xy, positions, (1.0, 0.3, 0.3), 1.0)
            for dx in range(-3, 4):
                scene3d.drawSegments(rgba, image.bbox, xy + [dx, -3], xy + [dx, 3], (1.0, 0.85, 0.2), 1.0)
        return image.withRGBA(rgba)

    # Actions

    _TRACK_ACTIONS = ("trackFace", "trackFaceForward", "trackFaceBackward", "trackFaceStepForward",
                      "trackFaceStepBackward", "refineFace")

    def runAction(self, name, context):
        """
        "Track Face" (every frame of the track range), forward / backward from the current frame, one frame forward /
        backward, or "Clear". With "Track With" mediapipe, MediaPipe finds the face at each frame on its own; with flow
        or mediapipe+flow, the face is followed from the current frame (its fitted face, pins included) by the optical
        flow, MediaPipe taken where it agrees (mediapipe+flow). The landmarks of the other frames are kept.
        """
        import json
        import time
        from meshroom.live import face
        if name == "clearFace":
            return {"faceTrack": "", "tracks": "[]"}
        if name == "setReference":
            return {"referenceFrame": int(context.currentFrame), "useReference": True}
        if name == "clearShape":
            return {"faceShape": ""}
        if name == "fitShape":
            return self.fitShape(context)
        if name not in self._TRACK_ACTIONS:
            raise RuntimeError(f"Unknown action: {name}")
        if context.info is None:
            raise RuntimeError("Connect the plate to the input.")
        width, height = context.info.width, context.info.height
        try:
            faces, _ = face.parseFaces(context.value("faceTrack"))
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        method = context.value("trackWith")
        frame, frames = context.currentFrame, context.frames
        if name != "refineFace" and (name != "trackFace" or method != "mediapipe"):
            if frame not in frames:
                raise RuntimeError("The current frame is outside the track range (Limit Track Range).")
        lastPublish = [0.0]

        def onFrame(f, found):
            faces[f] = found
            now = time.monotonic()
            if now - lastPublish[0] >= 0.5:
                lastPublish[0] = now
                context.publish({"faceTrack": face.dumpFaces(faces, (width, height))}, f)
        maskAt = None
        if context.hasInput("mask"):
            from meshroom.live import ops
            from meshroom.live.imageBuffer import BBox
            channel, invert = context.value("maskChannel"), context.value("invertMask")

            def maskAt(f):
                return ops.maskValues(context.renderInput("mask", f), channel, invert,
                                      BBox.fromFormat(width, height))[:, :, 0]
        detector = None
        if method != "flow" or not faces:
            try:
                detector = face.landmarker(context.value("minConfidence"))
            except face.FaceUnavailable as e:
                raise RuntimeError(f"Track Face: {e}") from e
            detector.turnLying = bool(context.values.get("detectLying", True))
        keyframes = self.keyframes(context)
        flowMethod = "raft" if context.value("flowMethod") == "raft" else "lucasKanade"
        useGpu = bool(context.value("useGpu")) if "useGpu" in context.values else True

        def startAt(f):
            """ The landmarks the flow starts from at a frame: its fitted face (pins included). """
            if f not in faces:
                # Nothing to start from at this frame: MediaPipe finds the face there first
                if detector is None:
                    raise RuntimeError(f"No face at frame {f} to follow: track it with MediaPipe first.")
                face.trackFaces(context.render, [f], detector, maskAt=maskAt, onFrame=onFrame)
                if f not in faces:
                    raise RuntimeError(f"No face found at frame {f} to follow (lower 'Min Confidence', or start from "
                                       f"a frame where the face is found).")
            return self.fittedLandmarks(context.nodeContext(f), faces, f)

        regularize = self.regularizer(context.nodeContext(frame), width, height)

        def poseAt(f):
            """ The camera-to-face matrix of the fitted face at a frame (pins included), None when there is none. """
            ctx = context.nodeContext(f)
            geo, head = self.solve(ctx, f)
            return None if geo.isEmpty else np.linalg.inv(head) @ self.camera(ctx, f).matrix

        def poseTracker(f):
            """ The tracker of the head pose by the surface of its rigid parts ("Follow Head Surface"), or None. """
            if not context.values.get("followSurface", True):
                return None
            from meshroom.live import gpu
            ctx = context.nodeContext(f)
            return face.HeadPoseTracker(self.mesh(ctx, f), self.camera(ctx, f), width, height,
                                        method="raft" if flowMethod == "raft" else "flow",
                                        device=gpu.device(useGpu))

        def propagate(start, sequence, callback):
            tracker = poseTracker(sequence[0])
            found, reason = face.propagateFaces(
                context.render, sequence, start, detector if method == "mediapipe+flow" else None, flowMethod=flowMethod,
                useGpu=useGpu, cancel=context.checkCancelled, onFrame=callback, maskAt=maskAt, regularize=regularize,
                progress=lambda done, count: context.progress(done / float(count), "Track Face"),
                poseTracker=tracker, startPose=poseAt(sequence[0]) if tracker is not None else None)
            if reason and reason != "cancelled":
                raise RuntimeError(f"Stopped: {reason}.")
            return found
        last = frame
        try:
            if name == "refineFace":
                self._refine(context, faces, keyframes, startAt, propagate, onFrame)
            elif method == "mediapipe":
                sequence = frames if name == "trackFace" else self._sequence(name, frame, frames, keyframes)
                face.trackFaces(context.render, sequence, detector, cancel=context.checkCancelled,
                                progress=lambda done, count: context.progress(done / float(count), "Track Face"),
                                onFrame=onFrame, maskAt=maskAt)
                last = sequence[-1]
            else:
                start = startAt(frame)
                directions = ["trackFaceForward", "trackFaceBackward"] if name == "trackFace" else [name]
                for direction in directions:
                    sequence = self._sequence(direction, frame, frames, keyframes)
                    if len(sequence) < 2:
                        continue
                    found = propagate(start, sequence, onFrame)
                    last = max(found, key=sequence.index) if found else frame
        finally:
            if detector is not None:
                detector.close()
        if not faces:
            raise RuntimeError("No face found in the plate (lower 'Min Confidence'?).")
        export = context.value("exportTracks")
        indices = None if export == "all" else face.featureIndices()
        tracks = [] if export == "none" else face.landmarkTracks(faces, width, height, indices)
        result = {"faceTrack": face.dumpFaces(faces, (width, height)), "tracks": json.dumps(tracks)}
        if name in ("trackFaceStepForward", "trackFaceStepBackward"):
            # The viewer shows the frame tracked
            result["_frame"] = last
        return result

    def fitShape(self, context, maxFrames: int = 40):
        """
        "Fit Shape": the shape of the actor from the tracked frames of the track range (up to `maxFrames`, spread, all
        the keyframes included and weighted 3 times): one rest face for the shot, fitted with the head pose and the
        expression of each frame (face.fitShape).
        """
        from meshroom.live import face
        if context.info is None:
            raise RuntimeError("Connect the plate to the input.")
        width, height = context.info.width, context.info.height
        try:
            faces, _ = face.parseFaces(context.value("faceTrack"))
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        inRange = set(context.frames)
        tracked = sorted(f for f in faces if f in inRange)
        if len(tracked) < 2:
            raise RuntimeError("Fit Shape needs the face tracked on several frames (Track Face first).")
        keyframes = [f for f in self.keyframes(context) if f in faces]
        others = [f for f in tracked if f not in keyframes]
        count = max(0, maxFrames - len(keyframes))
        chosen = sorted(set(keyframes) | {others[int(i)] for i in np.linspace(0, len(others) - 1, min(count, len(others)))}
                        if others else set(keyframes))
        ctx = context.nodeContext()
        base = self.baseMesh(ctx)
        pins = self.pins(ctx, context.info)
        observations = []
        for f in chosen:
            observations.append({"pixels": face.toPixels(faces[f]["landmarks"], width, height),
                                 "hidden": faces[f].get("hidden") or [], "camera": self.camera(ctx, f),
                                 "weight": 3.0 if f in keyframes else 1.0, "pins": self.solvePinsAt(pins, f)})
        offsets = face.fitShape(base, observations, width, height, cancel=context.checkCancelled,
                                progress=lambda done: context.progress(done, "Fit Shape"))
        return {"faceShape": face.dumpShape(offsets)}

    @staticmethod
    def _sequence(name, frame, frames, keyframes=()):
        """ The frames tracked by an action from `frame`: to the next keyframe (excluded, as in FaceTracker: it has its
        own fit) or the end of the range, forward / backward; or one frame. """
        forward = name in ("trackFaceForward", "trackFaceStepForward")
        sequence = [f for f in frames if f >= frame] if forward else [f for f in frames if f <= frame][::-1]
        if "Step" in name:
            return sequence[:2]
        for index, f in enumerate(sequence[1:], start=1):
            if f in keyframes:
                return sequence[:index]
        return sequence

    @staticmethod
    def keyframes(context):
        """ The keyframes of the face: the frames where pins were placed (they have a key there). """
        from meshroom.live import tracking
        try:
            pins = tracking.parseTracks(context.value("pins"))
        except ValueError:
            return []
        return sorted({int(f) for pin in pins if pin.get("enabled", True) for f in (pin.get("keys") or {})})

    def _refine(self, context, faces, keyframes, startAt, propagate, onFrame):
        """ "Refine": between each pair of keyframes in the track range, the face followed from both ends and the two
        blended, weighted toward the nearest keyframe (its drift is the smallest), as Refine in FaceTracker. """
        inRange = set(context.frames)
        spans = [(a, b) for a, b in zip(keyframes, keyframes[1:]) if b - a > 1 and a in inRange and b in inRange]
        if not spans:
            raise RuntimeError("Refine needs two keyframes (frames with pins) with frames between them.")
        for a, b in spans:
            forward = propagate(startAt(a), list(range(a, b)), None)
            backward = propagate(startAt(b), list(range(b, a, -1)), None)
            for f in range(a + 1, b):
                one, other = forward.get(f), backward.get(f)
                if one is None and other is None:
                    continue
                if one is None or other is None:
                    onFrame(f, one or other)
                    continue
                weight = (f - a) / float(b - a)
                count = min(len(one["landmarks"]), len(other["landmarks"]))
                blended = one["landmarks"][:count] * (1.0 - weight) + other["landmarks"][:count] * weight
                nearest = one if weight < 0.5 else other
                onFrame(f, {"landmarks": blended, "blendshapes": nearest["blendshapes"], "hidden": nearest["hidden"],
                            "source": "refined"})

    def regularizer(self, ctx, width, height):
        """ function(frame, landmarks, hidden): the landmarks of the face fitted on flowed ones (rigid pose, stiff non
        rigid face, pins of the frame): what the flow follows stays a face (a drifting point does not break it). """
        from meshroom.live import face
        mesh = self.mesh(ctx)
        deform, stiffness = ctx.value("deform"), max(float(ctx.value("stiffness")), 0.05)
        pins = self.pins(ctx)

        def regularize(frame, landmarks, hidden, pose=None, blendshapes=None):
            camera = self.camera(ctx, frame)
            pixels = face.toPixels(landmarks, width, height)
            report = {}
            prior, priorWeight = self.expressionPrior(ctx, mesh, {"blendshapes": blendshapes})
            geo, _, _, _ = face.solveFace(mesh, pixels, camera, width, height, rigid=face.RIGID_LANDMARKS,
                                          deform=deform, hidden=hidden, stiffness=stiffness,
                                          pins=self.solvePinsAt(pins, frame),
                                          pinRadius=ctx.value("pinRadius"), model=ctx.value("deformModel"),
                                          exactPins=ctx.value("exactPins"), pose=pose, report=report,
                                          prior=prior, priorWeight=priorWeight)
            # A rig: the face it makes is followed to the next frame (flow -> rig), not the warp of the pins (a local
            # correction of this frame only: it would drift from frame to frame)
            followed = report.get("rigWorld")
            xy, _, _ = camera.project(followed if followed is not None else geo.vertices, width, height)
            count = min(len(xy), len(landmarks))
            fitted = landmarks.copy()
            fitted[:count, 0] = xy[:count, 0] / width
            fitted[:count, 1] = (height - xy[:count, 1]) / height
            return fitted
        return regularize

    def fittedLandmarks(self, ctx, faces, frame):
        """ The landmarks (normalized) of the face fitted at a frame (pins and non rigid included): what the flow
        follows. The irises (not in a 468 vertex mesh) are kept from the tracked landmarks. """
        from meshroom.live import face
        info = ctx.inputInfo("input")
        geo, _ = self.solve(ctx, frame)
        tracked = faces[frame]["landmarks"]
        if geo.isEmpty:
            return tracked
        xy, _, _ = self.camera(ctx, frame).project(geo.vertices, info.width, info.height)
        # The landmarks of the mesh: the irises too when it has them (478 vertices; not a head)
        count = min(len(xy), face.LANDMARKS) if not face.isHead(geo) else face.MESH_VERTICES
        landmarks = tracked.copy() if len(tracked) >= count else np.zeros((count, 3))
        landmarks[:count, 0] = xy[:count, 0] / info.width
        landmarks[:count, 1] = (info.height - xy[:count, 1]) / info.height
        return landmarks


_VIEW_INPUTS = ("input", "input1", "input2", "input3")
# The pins of each view (FaceBuilder)
_VIEW_PINS = {"input": "pins", "input1": "pins1", "input2": "pins2", "input3": "pins3"}


class FaceBuilder(desc.ImageOpNode):
    category = "Comp 3D"
    color = _COLOR
    documentation = """
Builds the head of an actor from photos (or frames) of their face, like KeenTools FaceBuilder: a base head (the
canonical face of MediaPipe with ears, skull and neck following it) warped onto the actor's face.

- Connect one to four views: each one a photo, or a sequence (every frame a view, up to "Max Frames" spread over it),
  e.g. a Read of photos from the front, the sides, three quarters. Different sizes and lenses are fine.
- "Build Head": MediaPipe finds the 478 landmarks of the face in each view, then the shape of the face is fitted on
  all the views at once, each with its head pose and expression (the face rig) and its focal length (estimated, or
  "Focal Length"). The views from different angles give the depth one view does not. "Fit Again" fits on the stored
  landmarks again (e.g. after changing "Smoothness").
- The ears, skull and neck are not seen by the landmarks: with "Head From Face", they take the proportions that go
  with the face (a statistical model of whole heads, ICT-FaceKit); "Fit Ears, Neck And Skull" refines them on the
  outlines of the views (and their depth). Pins
  place them, as in KeenTools FaceBuilder: in the viewer (the view shown, "Show View"), "Add a pin", click on the head
  (an ear, the top or the back of the skull, the jaw line...), then drag the pin onto its place in the photo. The head
  follows live, a smooth move around each pin ("Pin Radius"); pins near each other in several views move it in 3D,
  a pin in one view moves it across that view.

Outputs: "Head" (the warped head, face space: centimeters, looking along +Z), "Original" (the base head), "Camera"
(the camera of the view shown, the head at the origin) and the view shown with the head drawn over it. Connect "Head"
to the "Face" input of a FaceTracker: it then tracks the actor's own head.

Needs MediaPipe (pip install mediapipe) and its model in MESHROOM_MODELS_PATH
(python -m meshroom.live.neural.download facelandmarker).
"""
    inputs = [
        desc.Image(name=name, label=f"View {index + 1}", optional=index > 0,
                   description=f"View {index + 1} of the actor's face: a photo, or a sequence (each frame a view).")
        for index, name in enumerate(_VIEW_INPUTS)
    ] + [
        desc.Geo(name="face", label="Base", optional=True,
                 description="The base face or head (MediaPipe face topology: 468 or 478 vertices in the landmark "
                             "order, or a head of the FaceBuilder). Without it: the canonical head."),
        desc.ChoiceParam(name="meshType", label="Mesh", value="head", values=["head", "face"],
                         description="head: the face with ears, skull and neck following it. face: the face only."),
        desc.FloatParam(name="minConfidence", label="Min Confidence", value=0.5, range=(0.1, 0.95, 0.05),
                        invalidate=False, description="A face is found when MediaPipe is at least this confident."),
        desc.IntParam(name="maxFrames", label="Max Frames", value=12, range=(1, 60, 1), invalidate=False,
                      description="The views taken from a sequence: at most this many frames, spread over it."),
        desc.BoolParam(name="estimateFocal", label="Estimate Focal Length", value=True, invalidate=False,
                       description="Estimate the focal length of each view from how the face is seen (close: strong "
                                   "perspective); off: 'Focal Length' for all."),
        desc.FloatParam(name="focal", label="Focal Length", value=50.0, range=(8.0, 300.0, 1.0), invalidate=False,
                        enabled=lambda node: not node.estimateFocal.value,
                        description="Focal length of the views (mm, on a 36 mm wide film back: the 35 mm equivalent)."),
        desc.FloatParam(name="smoothness", label="Smoothness", value=3.0, range=(0.0, 20.0, 0.1), invalidate=False,
                        description="How smooth the change of shape is (less: closer to the landmarks, more noise)."),
        desc.FloatParam(name="symmetry", label="Symmetry", value=1.0, range=(0.0, 10.0, 0.1), invalidate=False,
                        description="How much the change of shape stays symmetric."),
        desc.FloatParam(name="closeness", label="Stay Near Base", value=0.05, range=(0.0, 1.0, 0.01), invalidate=False,
                        description="How much the shape stays near the base face."),
        desc.FloatParam(name="pinRadius", label="Pin Radius", value=4.0, range=(0.5, 20.0, 0.1),
                        description="How far around a pin the head moves (centimeters on the canonical head: an ear "
                                    "is about 6 cm high). More: a wider, smoother move."),
        *[desc.StringParam(name=pinsName, label=f"Pins (View {index + 1})", value="[]", semantic="tracks",
                           advanced=True,
                           description=f"Pins of View {index + 1} (JSON): a point of the base head (\"point\") and "
                                       "where it is in the view (its key). Edited in the viewer.")
          for index, pinsName in enumerate(_VIEW_PINS.values())],
        desc.PushButtonParam(name="buildHead", label="Build Head", semantic="liveAction",
                             description="Find the face in each view and fit the head on all of them."),
        desc.BoolParam(name="predictHead", label="Head From Face", value=True,
                       description="The skull, ears and neck take the proportions that go with the face (the identity "
                                   "model of ICT-FaceKit: 50 modes learned on scans of whole heads, the face fitted on "
                                   "the landmarks); off: they only follow the face (stretched with it)."),
        desc.PushButtonParam(name="fitHeadContours", label="Fit Ears, Neck And Skull", semantic="liveAction",
                             description="After 'Build Head': fit the ears, the neck and the skull on the views, as "
                                         "the pins would (MediaPipe segmentation: the outlines of the ears and neck "
                                         "on the skin, the skull inside the hair; with 'Use Depth', the depth of the "
                                         "Depth Anything network). The pins still apply after it."),
        desc.FloatParam(name="fitRadius", label="Fit Radius", value=3.0, range=(1.0, 10.0, 0.1), invalidate=False,
                        enabled=lambda node: not node.predictHead.value,
                        description="'Fit Ears, Neck And Skull' without 'Head From Face': how smooth the move is "
                                    "(centimeters around each control point; more: smoother, less detail)."),
        desc.BoolParam(name="fitDepth", label="Use Depth", value=True, invalidate=False,
                       description="'Fit Ears, Neck And Skull' also uses the depth of the views (Depth Anything V2: "
                                   "pip install transformers, python -m meshroom.live.neural.download depthanything); "
                                   "without the network, only the outlines."),
        desc.StringParam(name="headFit", label="Head Fit", value="", advanced=True,
                         description="The fit of the ears, neck and skull (JSON: control points and their moves)."),
        desc.PushButtonParam(name="refitHead", label="Fit Again", semantic="liveAction",
                             description="Fit the head again on the landmarks found by 'Build Head' (after changing "
                                         "the parameters of the fit)."),
        desc.PushButtonParam(name="clearHead", label="Clear", semantic="liveAction",
                             description="Back to the base head."),
        desc.StringParam(name="faceShape", label="Face Shape", value="", semantic="faceShape", advanced=True,
                         description="The shape of the actor: offsets of the vertices of the base face (JSON)."),
        desc.StringParam(name="views", label="Views", value="", visible=False, advanced=True,
                         description="The views (JSON): their input, frame, size, landmarks, focal length and pose."),
        desc.ChoiceParam(name="showView", label="Show View", value="input", values=list(_VIEW_INPUTS),
                         description="The view output (and drawn on): input (View 1), input1 (View 2)..."),
        desc.ChoiceParam(name="display", label="Display", value="wireframe",
                         values=["shaded", "edges", "wireframe", "points", "landmarks", "none"],
                         description="What is drawn over the view: the head, the landmarks, or nothing."),
        desc.FloatParam(name="opacity", label="Opacity", value=0.6, range=(0.0, 1.0, 0.01),
                        description="Opacity of the head drawn over the view."),
    ]
    outputs = [
        desc.Geo(name="geoOutput", label="Head", description="The head of the actor (face space)."),
        desc.Geo(name="originalOutput", label="Original", description="The base head (face space)."),
        desc.Camera(name="cameraOutput", label="Camera",
                    description="The camera of the view shown at this frame, the head at the origin."),
        desc.Image(name="output", label="Output", description="The view shown, with the head drawn over it."),
    ]

    # The views

    def shownInput(self, ctx):
        name = ctx.value("showView")
        return name if ctx.hasInput(name) else "input"

    def infoOnlyInputs(self, ctx):
        # Only the view shown is rendered
        shown = self.shownInput(ctx)
        return [name for name in _VIEW_INPUTS if name != shown]

    def getInfo(self, ctx, inputsInfo):
        info = inputsInfo.get(self.shownInput(ctx))
        return info if info is not None else super().getInfo(ctx, inputsInfo)

    def isIdentity(self, ctx):
        return self.shownInput(ctx) if ctx.value("display") == "none" else None

    @staticmethod
    def storedViews(ctx_or_context):
        import json
        text = ctx_or_context.value("views")
        if not text:
            return []
        try:
            return json.loads(text)
        except ValueError as e:
            raise RuntimeError(f"Invalid views: {e}") from e

    def viewAt(self, ctx, frame=None):
        """ The stored view of the input shown at a frame (a photo: any frame), or None. """
        frame = ctx.frame if frame is None else frame
        shown = self.shownInput(ctx)
        views = [view for view in self.storedViews(ctx) if view["input"] == shown]
        for view in views:
            if view.get("still") or int(view["frame"]) == int(frame):
                return view
        return None

    # The head

    def inputMesh(self, ctx):
        from meshroom.live import face
        mesh = ctx.inputData("face")
        try:
            mesh = mesh if mesh is not None else face.canonicalHead()
            face.checkTopology(mesh)
        except face.FaceUnavailable as e:
            raise RuntimeError(str(e)) from e
        return mesh

    def baseMesh(self, ctx):
        """ The face fitted (MediaPipe topology): the face of the base. """
        from meshroom.live import face
        return face.facePart(self.inputMesh(ctx))

    def original(self, ctx):
        """ The base head (or face, "Mesh" face). """
        from meshroom.live import face
        source = self.inputMesh(ctx)
        if ctx.value("meshType") != "head":
            return face.facePart(source)
        return face.headOf(source)

    def head(self, ctx, withFit=True):
        """ The head of the actor: the base with the fitted shape (its ears, skull and neck following the face), then
        the fit of its ears, neck and skull and its pins (`withFit`; without: the head the landmarks give). """
        from meshroom.live import face
        base = self.baseMesh(ctx)
        try:
            offsets = face.parseShape(ctx.value("faceShape"))
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        original = self.original(ctx)
        hasShape = offsets is not None and len(offsets) == len(base.vertices)
        if ctx.value("meshType") != "head":
            fitted = scene3d.GeoData(base.vertices + offsets, name="face", topology=base.topology) if hasShape else original
        elif self.usesIdentity(ctx, original):
            # The proportions of the whole head from its face (the identity model), or fitted on the views
            from meshroom.live import headFit
            faceTarget = base.vertices + offsets if hasShape else original.vertices[:face.MESH_VERTICES]
            coefficients = self.storedIdentity(ctx) if withFit else None
            if coefficients is None:
                coefficients = headFit.predictIdentity(faceTarget, original.vertices)
            fitted = scene3d.GeoData(headFit.identityHead(original.vertices, faceTarget, coefficients), name="head",
                                     topology=original.topology)
        elif hasShape:
            fitted = scene3d.GeoData(face.HeadFollow.of(original.vertices).follow(base.vertices + offsets), name="head",
                                     topology=original.topology)
        else:
            fitted = original
        if not withFit:
            return fitted
        fitted = self.headFitted(ctx, fitted)
        return self.pinned(ctx, fitted, original)

    @staticmethod
    def usesIdentity(ctx, original) -> bool:
        """ Whether the head comes from the identity model ("Head From Face"; a head of the FaceBuilder topology). """
        from meshroom.live import face, headFit
        return bool(ctx.value("predictHead")) and face.isHead(original) and headFit.hasIdentityModel(original.vertices)

    @staticmethod
    def storedIdentity(ctx):
        """ The identity coefficients fitted on the views ("Fit Ears, Neck And Skull"), or None. """
        import json
        text = ctx.value("headFit")
        if not text:
            return None
        try:
            identity = json.loads(text).get("identity")
        except (ValueError, AttributeError):
            return None
        return np.asarray(identity, dtype=np.float64) if identity else None

    def headFitted(self, ctx, head):
        """ The head with its ears, neck and skull fitted on the views ("Fit Ears, Neck And Skull"), if done. """
        import json
        from meshroom.live import face, headFit
        text = ctx.value("headFit")
        if not text or not face.isHead(head):
            return head
        try:
            fit = json.loads(text)
        except ValueError as e:
            raise RuntimeError(f"Invalid head fit: {e}") from e
        return scene3d.GeoData(headFit.applyFit(head, fit), name=head.name, topology=head.topology)

    # Pins

    _PINNED = {}

    def viewPins(self, ctx, view):
        """ (points on the base head (k, 3), their positions (k, 2) in the view) of the pins of a stored view: their key
        at its frame (a photo: the key nearest to it). """
        from meshroom.live import tracking
        try:
            tracks = tracking.parseTracks(ctx.value(_VIEW_PINS.get(view["input"], "pins")))
        except ValueError as e:
            raise RuntimeError(f"Invalid pins: {e}") from e
        tracks = tracking.scaleTracks(tracks, *view["size"])
        points, positions = [], []
        frame = int(view["frame"])
        for track in tracks:
            keys = track.get("keys") or {}
            if not track.get("enabled", True) or len(track.get("point") or []) != 3 or not keys:
                continue
            key = keys.get(str(frame))
            if key is None and view.get("still"):
                key = keys[min(keys, key=lambda f: (abs(int(f) - frame), int(f)))]
            if key is None:
                continue
            points.append(track["point"])
            positions.append((float(key[0]), float(key[1])))
        return np.array(points, dtype=np.float64).reshape(-1, 3), np.array(positions, dtype=np.float64).reshape(-1, 2)

    def pinned(self, ctx, fitted, original):
        """ The fitted head moved onto the pins of all the views (face.warpToPins), cached. """
        import hashlib
        from meshroom.live import face
        views = [view for view in self.storedViews(ctx) if view.get("cameraToFace")]
        texts = [ctx.value(name) or "" for name in _VIEW_PINS.values()]
        if not views or all(text.strip() in ("", "[]") for text in texts):
            return fitted
        key = hashlib.sha1(np.ascontiguousarray(fitted.vertices).tobytes() + repr(
            (texts, ctx.value("views"), ctx.value("pinRadius"))).encode()).hexdigest()
        cached = FaceBuilder._PINNED.get(key)
        if cached is None:
            locations, targets, cameras, sizes = [], [], [], []
            for view in views:
                points, positions = self.viewPins(ctx, view)
                if not len(points):
                    continue
                camera = self.viewCamera(ctx, view)
                for location, position in zip(face.pinLocations(original, points), positions):
                    locations.append(location)
                    targets.append(position)
                    cameras.append(camera)
                    sizes.append(tuple(view["size"]))
            if not locations:
                return fitted
            cached = face.warpToPins(fitted, locations, np.array(targets), cameras, sizes,
                                     radius=float(ctx.value("pinRadius")))
            if len(FaceBuilder._PINNED) > 16:
                FaceBuilder._PINNED.clear()
            FaceBuilder._PINNED[key] = cached
        return scene3d.GeoData(cached, name=fitted.name, topology=fitted.topology)

    def pickPoint(self, ctx, x, y):
        """ The point of the base head under the image position (x, y) of the view shown (where the head is seen), or
        None. """
        from meshroom.live import face
        view = self.viewAt(ctx)
        if view is None or not view.get("cameraToFace"):
            raise RuntimeError("Build the head first (Build Head): the pins are put on it, in the views.")
        head = self.head(ctx)
        width, height = view["size"]
        origin, direction = self.viewCamera(ctx, view).ray(x, y, width, height)
        hit = scene3d.raycast(head, origin, direction)
        if hit is None:
            return None
        triangle, weights = face.surfaceLocation(head.vertices, head.triangles(), hit)
        # The same place on the base head (the pins stay valid when the head is built again)
        return self.original(ctx).vertices[head.triangles()[triangle]].T @ weights

    def pinProjections(self, ctx, cachedData) -> dict:
        """ {pin name: [x, y]}: where each pin of the view shown is seen on the head, in the pixels of the view. """
        from meshroom.live import face, tracking
        view = self.viewAt(ctx)
        if view is None or not view.get("cameraToFace"):
            return {}
        try:
            tracks = tracking.parseTracks(ctx.value(_VIEW_PINS[self.shownInput(ctx)]))
        except ValueError:
            return {}
        tracks = [track for track in tracks if len(track.get("point") or []) == 3]
        if not tracks:
            return {}
        head = self.head(ctx)
        locations = face.pinLocations(self.original(ctx), np.array([track["point"] for track in tracks]))
        onHead = face._onSurface(head.vertices, head.triangles(), locations)
        xy, _, inFront = self.viewCamera(ctx, view).project(onHead, *view["size"])
        return {track["name"]: [float(px), float(py)] for track, (px, py), front in zip(tracks, xy, inFront) if front}

    def viewCamera(self, ctx, view):
        """ The camera of a stored view, the head at the origin. """
        from meshroom.live import face
        width, height = view["size"]
        return face.viewCamera(view["focal"], width, height).copy(
            matrix=np.array(view["cameraToFace"], dtype=np.float64).reshape(4, 4))

    def computeData(self, ctx, outputName):
        if outputName == "geoOutput":
            return self.head(ctx)
        if outputName == "originalOutput":
            return self.original(ctx)
        if outputName == "cameraOutput":
            view = self.viewAt(ctx)
            return self.viewCamera(ctx, view) if view is not None else None
        return super().computeData(ctx, outputName)

    def render(self, ctx, inputs):
        from meshroom.live import face, geoRender
        image = inputs[self.shownInput(ctx)]
        rgba = np.array(image.rgba(), dtype=np.float32)
        width, height = image.width, image.height
        view = self.viewAt(ctx)
        display, alpha = ctx.value("display"), ctx.value("opacity")
        if view is not None and list(view["size"]) == [width, height]:
            if display == "landmarks":
                pixels = face.toPixels(np.array(view["landmarks"], dtype=np.float64), width, height)
                scene3d.drawPoints(rgba, image.bbox, pixels, (1.0, 0.85, 0.2), alpha, 2)
            else:
                from meshroom.live import gpu
                camera = self.viewCamera(ctx, view)
                geoRender.drawGeometry(rgba, image.bbox, self.head(ctx), camera, width, height, display,
                                       (0.35, 0.8, 1.0), alpha, creaseDegrees=60.0, useGpu=gpu.enabled(ctx), quads=True)
        return image.withRGBA(rgba)

    # Actions

    def runAction(self, name, context):
        """ "Build Head": the landmarks of each view, then the shape fitted on all of them; "Fit Again": the fit on
        the stored landmarks; "Clear". """
        import json
        from meshroom.live import face
        if name == "clearHead":
            return {"faceShape": "", "views": "", "headFit": ""}
        if name == "fitHeadContours":
            return self.fitHeadContours(context)
        if name not in ("buildHead", "refitHead"):
            raise RuntimeError(f"Unknown action: {name}")
        if name == "buildHead":
            stored = self.detectViews(context)
        else:
            stored = self.storedViews(context)
            if not stored:
                raise RuntimeError("Build the head first (Build Head).")
        views = [{"pixels": face.toPixels(np.array(view["landmarks"], dtype=np.float64), *view["size"]),
                  "size": tuple(view["size"]), "hidden": view.get("hidden") or []} for view in stored]
        ctx = context.nodeContext()
        base = self.baseMesh(ctx)
        offsets, fitted = face.buildShape(
            base, views, focal=None if context.value("estimateFocal") else context.value("focal"),
            smoothness=context.value("smoothness"), symmetry=context.value("symmetry"),
            closeness=context.value("closeness"), cancel=context.checkCancelled,
            progress=lambda done: context.progress(done, "Build Head"))
        for view, result in zip(stored, fitted):
            view["focal"] = round(float(result["focal"]), 3)
            view["cameraToFace"] = [round(float(v), 6) for v in np.asarray(result["cameraToFace"]).ravel()]
            view["error"] = round(float(result["error"]), 3)
        return {"faceShape": face.dumpShape(offsets), "views": json.dumps(stored)}

    def fitHeadContours(self, context):
        """ "Fit Ears, Neck And Skull": each view segmented (and its depth estimated), then headFit.fitHead on all. """
        import json
        from meshroom.live import BBox, face, gpu, headFit
        from meshroom.live.neural import NeuralUnavailable, displayRGB
        stored = [view for view in self.storedViews(context) if view.get("cameraToFace")]
        if not stored:
            raise RuntimeError("Build the head first (Build Head): the ears, neck and skull are fitted on its views.")
        ctx = context.nodeContext()
        head = self.head(ctx, withFit=False)
        if not face.isHead(head):
            raise RuntimeError("Fit Ears, Neck And Skull: the mesh is a face only ('Mesh': head).")
        try:
            segmenter = headFit.headSegmenter()
        except face.FaceUnavailable as e:
            raise RuntimeError(f"Fit Ears, Neck And Skull: {e}") from e
        estimator, depthMessage = None, ""
        if context.value("fitDepth"):
            try:
                estimator = headFit.depthEstimator(gpu.availableDevice())
            except NeuralUnavailable as e:
                depthMessage = f" (without depth: {e})"
        views = []
        try:
            for index, view in enumerate(stored):
                context.checkCancelled()
                context.progress(0.6 * index / len(stored), "Segment the views")
                width, height = view["size"]
                image = context.renderInput(view["input"], int(view["frame"]))
                camera = self.viewCamera(ctx, view)
                xy, _, _ = camera.project(head.vertices, width, height)
                # The crop around the head (rows from the top), with room for the borders searched around it
                x0, x1 = xy[:, 0].min(), xy[:, 0].max()
                y0, y1 = height - xy[:, 1].max(), height - xy[:, 1].min()
                size = max(x1 - x0, y1 - y0)
                margin = 0.25 * size
                left, top = int(max(0, x0 - margin)), int(max(0, y0 - margin))
                right, bottom = int(min(width, x1 + margin)), int(min(height, y1 + margin))
                if right - left < 16 or bottom - top < 16:
                    continue
                rgb = displayRGB(image, BBox.fromFormat(width, height))[top:bottom, left:right]
                labels = segmenter.segment((rgb * 255).astype(np.uint8))
                disparity = estimator.estimate(rgb) if estimator is not None else None
                views.append(headFit.FitView(camera, (width, height), labels, (left, top), disparity,
                                             search=float(np.clip(0.08 * size, 15, 150))))
        finally:
            segmenter.close()
        if not views:
            raise RuntimeError("Fit Ears, Neck And Skull: the head is not in the views.")
        original = self.original(ctx)
        progress = lambda done: context.progress(0.6 + 0.4 * done, "Fit the head")  # noqa: E731
        if self.usesIdentity(ctx, original):
            # The proportions of the whole head (identity coefficients) fitted on the views: a human head whatever the
            # segmentation does
            fit = headFit.fitIdentity(original.vertices, head.vertices[:face.MESH_VERTICES], original.topology, views,
                                      useDepth=estimator is not None, cancel=context.checkCancelled, progress=progress)
        else:
            fit = headFit.fitHead(head, views, radius=context.value("fitRadius"), useDepth=estimator is not None,
                                  cancel=context.checkCancelled, progress=progress)
        if not fit["constraints"]:
            raise RuntimeError("Fit Ears, Neck And Skull: no outline of the ears, neck or skull found in the views.")
        import logging
        logging.info(f"[FaceBuilder] Head fitted on {len(views)} views, {fit['constraints']} constraints{depthMessage}")
        fit.pop("constraints", None)
        return {"headFit": json.dumps(fit)}

    def detectViews(self, context):
        """ The views of the connected inputs with a face: {input, frame, still, size, landmarks, hidden}. """
        from meshroom.live import face
        try:
            detector = face.landmarker(context.value("minConfidence"))
        except face.FaceUnavailable as e:
            raise RuntimeError(f"Build Head: {e}") from e
        sources = []
        for name in _VIEW_INPUTS:
            if not context.hasInput(name):
                continue
            frames = list(context.inputFrames(name))
            still = len(frames) == 1
            count = max(1, int(context.value("maxFrames")))
            if len(frames) > count:
                frames = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, count)]
            sources.append((name, frames, still))
        total = max(1, sum(len(frames) for _, frames, _ in sources))
        views, done = [], [0]
        try:
            for name, frames, still in sources:
                sizes = {}

                def imageAt(f, name=name, sizes=sizes):
                    image = context.renderInput(name, f)
                    sizes[f] = image.format
                    return image
                # Each view on its own (no search around the face of the previous one: photos are unrelated)
                for f in frames:
                    found = face.trackFaces(imageAt, [f], detector, cancel=context.checkCancelled)
                    done[0] += 1
                    context.progress(0.3 * done[0] / total, "Find the faces")
                    if f in found:
                        landmarks = found[f]["landmarks"]
                        views.append({"input": name, "frame": int(f), "still": still, "size": list(sizes[f]),
                                      "landmarks": [[round(float(v), 5) for v in point] for point in landmarks],
                                      "hidden": list(found[f].get("hidden") or [])})
        finally:
            detector.close()
        if not views:
            raise RuntimeError("No face found in the views (connect photos of the face, lower 'Min Confidence'?).")
        return views
