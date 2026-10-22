"""
FaceTracker: a face followed in the plate, non rigid (MediaPipe Face Landmarker, see meshroom/live/face.py).
"""
__version__ = "0.1"

import numpy as np

from meshroom.core import desc
from meshroom.live import scene3d
from meshroom.live.gpu import gpuParam
from meshroom.live.params import imageInput, maskParams, trackRangeParams

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
  actor retopologized on it; its shape is the neutral face. Without it: the canonical face of MediaPipe.

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
                             "e.g. a scan of the actor retopologized on it. Without it: the canonical face of MediaPipe."),
        desc.Image(name="mask", label="Mask", optional=True, infoOnly=True,
                   description="Optional mask: the face is searched in its white part only, and the landmarks it "
                               "masks out (e.g. behind foliage: a roto of it, inverted) do not place the head nor "
                               "move their vertices, and give no track key."),
        *maskParams(),
        desc.FloatParam(name="minConfidence", label="Min Confidence", value=0.5, range=(0.1, 0.95, 0.05),
                        invalidate=False,
                        description="A face is found (and followed) when MediaPipe is at least this confident."),
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
                                     "(\"point\": [x, y, z]). Edited in the viewer."),
        desc.FloatParam(name="stiffness", label="Stiffness", value=0.3, range=(0.0, 5.0, 0.05),
                        description="With 'Non Rigid': how much the face keeps a smooth shape. 0: each vertex exactly "
                                    "on its landmark (a wrong landmark folds the mesh); higher: a smooth surface through "
                                    "the landmarks, wrong or hidden ones following their neighbors."),
        desc.BoolParam(name="showPinReach", label="Show Pin Reach", value=True,
                       description="With 'Non Rigid': the face is tinted where the pins move it (orange at a pin, "
                                   "fading over 'Pin Radius')."),
        desc.FloatParam(name="pinRadius", label="Pin Radius", value=2.5, range=(0.1, 20.0, 0.1),
                        description="How far around a pin the face is moved to go through it (non rigid), in the "
                                    "units of the face mesh (centimeters for the canonical face)."),
        desc.FloatParam(name="smoothing", label="Landmark Smoothing", value=1.0, range=(0.0, 5.0, 0.1),
                        description="Gaussian smoothing of the landmarks in time, in frames (0: as tracked)."),
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
        desc.StringParam(name="faceTrack", label="Face Track", value="", semantic="faceTrack", visible=False,
                         advanced=True, description="The landmarks of each frame (JSON, written by 'Track Face')."),
        desc.StringParam(name="tracks", label="Landmark Tracks", value="[]", semantic="autoTracks",
                         description="The landmarks as tracks (JSON), for the Roto, Stabilize, CornerPin."),
    ]
    outputs = [
        desc.Geo(name="geoOutput", label="Face", description="The face mesh at each frame, in the world of the camera."),
        desc.Matrix(name="headMatrix", label="Head Matrix", description="The pose of the head: face space to world."),
        desc.Camera(name="cameraOutput", label="Camera", description="The plate camera (the input camera, or the default one)."),
        desc.Image(name="output", label="Output", description="The plate, with the face drawn over it."),
    ]

    def isIdentity(self, ctx):
        return "input" if ctx.value("display") == "none" else None

    # The face

    def camera(self, ctx, frame=None):
        """ The plate camera: the "camera" input, or 35 mm on a 36 mm film back at the origin. """
        camera = ctx.inputData("camera", frame)
        return camera if camera is not None else scene3d.CameraData(np.eye(4), 35.0, 36.0, 24.0)

    def mesh(self, ctx, frame=None):
        """ The face mesh (MediaPipe topology): the "face" input, or the canonical face. """
        from meshroom.live import face
        mesh = ctx.inputData("face", frame)
        try:
            mesh = mesh if mesh is not None else face.canonicalFace()
            face.checkTopology(mesh)
        except face.FaceUnavailable as e:
            raise RuntimeError(str(e)) from e
        return mesh

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
        locations = face.pinLocations(mesh, np.array([pin["point"] for pin in pins], dtype=np.float64))
        onFace = np.array([geo.vertices[triangles[t]].T @ w for t, w in locations])
        xy, _, inFront = self.camera(ctx).project(onFace, info.width, info.height)
        return {pin["name"]: [float(x), float(y)] for pin, (x, y), front in zip(pins, xy, inFront) if front}

    def pinReachColors(self, ctx, info, count):
        """ The colors of the vertices showing where the pins of the frame move the face ("Show Pin Reach"): from the
        face color to orange at a pin, over "Pin Radius"; None when not shown (rigid, no pin at this frame). """
        from meshroom.live import face
        if not (ctx.value("showPinReach") and ctx.value("deform")):
            return None
        # The pins of this frame (a keyframe): the others do not move the face here
        points, _ = self.pinsAt(self.pins(ctx, info), ctx.frame)
        if not len(points):
            return None
        mesh = self.mesh(ctx)
        centers = face._onSurface(mesh.vertices, mesh.triangles(), face.pinLocations(mesh, points))
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
        mesh = self.mesh(ctx)
        triangle, weights = face.surfaceLocation(geo.vertices, geo.triangles(), hit)
        # The same place on the neutral face
        return mesh.vertices[mesh.triangles()[triangle]].T @ weights

    def rawSolve(self, ctx, frame):
        """
        The face fitted at a frame on its own (no temporal smoothing): (its vertices in face space (n, 3), the head
        matrix), cached; None before tracking. Starts from the head pose of a neighbor frame already solved (continuity:
        no flip of the pose between frames).
        """
        from meshroom.live import face
        evaluator, uid = ctx.evaluator, ctx.planNode.uid
        cached = evaluator.cachedData(uid, "_rawFace", frame)
        if cached is not None:
            return cached.vertices, evaluator.cachedData(uid, "_rawHead", frame)
        faces = self.faces(ctx)
        info = ctx.inputInfo("input")
        if not faces or info is None:
            return None
        landmarks = face.landmarksAt(faces, frame, ctx.value("smoothing"))
        pixels = face.toPixels(landmarks, info.width, info.height)
        tracked = faces.get(frame) or faces[min(faces, key=lambda f: (abs(f - frame), f))]
        pinPoints, pinPositions = self.pinsAt(self.pins(ctx, info), frame)
        camera = self.camera(ctx, frame)
        initial = None
        for neighbor in (frame - 1, frame + 1):
            head = evaluator.cachedData(uid, "_rawHead", neighbor)
            if head is not None:
                # camera-to-face = inverse(head) @ camera
                initial = np.linalg.inv(head) @ camera.matrix
                break
        geo, head, _, _ = face.solveFace(self.mesh(ctx, frame), pixels, camera, info.width, info.height,
                                         initial=initial, rigid=face.RIGID_LANDMARKS, deform=ctx.value("deform"),
                                         hidden=tracked.get("hidden") or (),
                                         pins=(pinPoints, pinPositions) if len(pinPoints) else None,
                                         pinRadius=ctx.value("pinRadius"), stiffness=ctx.value("stiffness"))
        inverse = np.linalg.inv(head)
        local = geo.vertices @ inverse[:3, :3].T + inverse[:3, 3]
        evaluator.storeData(uid, "_rawHead", frame, head)
        evaluator.storeData(uid, "_rawFace", frame, scene3d.GeoData(local))
        return local, head

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
        keyed = len(self.pinsAt(self.pins(ctx, info), frame)[0]) > 0
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
        world = local @ head[:3, :3].T + head[:3, 3]
        mesh = self.mesh(ctx, frame)
        return scene3d.GeoData(world, name="face", topology=mesh.topology), head

    def computeData(self, ctx, outputName):
        if outputName == "cameraOutput":
            return self.camera(ctx)
        if outputName == "geoOutput":
            return self.solve(ctx)[0]
        if outputName in ("headMatrix", "matrix"):
            return self.solve(ctx)[1]
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
                                       vertexColors=self.pinReachColors(ctx, image.info, len(geo.vertices)))
        # The pins: where the face puts their point, linked to where they are (the error)
        geo = ctx.ownData("geoOutput", ctx.frame)
        points, positions = self.pinsAt(self.pins(ctx, image.info), ctx.frame)
        if len(points) and geo is not None and not geo.isEmpty:
            mesh = self.mesh(ctx)
            triangles = mesh.triangles()
            onFace = np.array([geo.vertices[triangles[t]].T @ w for t, w in face.pinLocations(mesh, points)])
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

        def propagate(start, sequence, callback):
            found, reason = face.propagateFaces(
                context.render, sequence, start, detector if method == "mediapipe+flow" else None, flowMethod=flowMethod,
                useGpu=useGpu, cancel=context.checkCancelled, onFrame=callback, maskAt=maskAt, regularize=regularize,
                progress=lambda done, count: context.progress(done / float(count), "Track Face"))
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

        def regularize(frame, landmarks, hidden):
            camera = self.camera(ctx, frame)
            pixels = face.toPixels(landmarks, width, height)
            pinPoints, pinPositions = self.pinsAt(pins, frame)
            geo, _, _, _ = face.solveFace(mesh, pixels, camera, width, height, rigid=face.RIGID_LANDMARKS,
                                          deform=deform, hidden=hidden, stiffness=stiffness,
                                          pins=(pinPoints, pinPositions) if len(pinPoints) else None,
                                          pinRadius=ctx.value("pinRadius"))
            xy, _, _ = camera.project(geo.vertices, width, height)
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
        count = min(len(xy), face.LANDMARKS)
        landmarks = tracked.copy() if len(tracked) >= count else np.zeros((count, 3))
        landmarks[:count, 0] = xy[:count, 0] / info.width
        landmarks[:count, 1] = (info.height - xy[:count, 1]) / info.height
        return landmarks
