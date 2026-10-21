"""
FaceTracker: a face followed in the plate, non rigid (MediaPipe Face Landmarker, see meshroom/live/face.py).
"""
__version__ = "0.1"

import numpy as np

from meshroom.core import desc
from meshroom.live import scene3d
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
        desc.PushButtonParam(name="trackFace", label="Track Face", semantic="liveAction",
                             description="Find the face at every frame of the plate (MediaPipe Face Landmarker)."),
        desc.PushButtonParam(name="clearFace", label="Clear", semantic="liveAction",
                             description="Remove the tracked face."),
        *trackRangeParams(),
        desc.StringParam(name="pins", label="Pins", value="[]", semantic="tracks",
                         description="Pins as JSON: a tracked image position and its point on the neutral face "
                                     "(\"point\": [x, y, z]). Edited in the viewer."),
        desc.FloatParam(name="pinRadius", label="Pin Radius", value=2.5, range=(0.1, 20.0, 0.1),
                        description="How far around a pin the face is moved to go through it (non rigid), in the "
                                    "units of the face mesh (centimeters for the canonical face)."),
        desc.FloatParam(name="smoothing", label="Smoothing", value=1.0, range=(0.0, 5.0, 0.1),
                        description="Gaussian smoothing of the landmarks in time, in frames (0: as tracked)."),
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

    def solve(self, ctx, frame=None):
        """ (the face GeoData in world space, the head matrix) at a frame; an empty GeoData and the identity before
        tracking. """
        from meshroom.live import face
        frame = ctx.frame if frame is None else frame
        faces = self.faces(ctx)
        info = ctx.inputInfo("input")
        if not faces or info is None:
            return scene3d.GeoData(), np.eye(4)
        landmarks = face.landmarksAt(faces, frame, ctx.value("smoothing"))
        pixels = face.toPixels(landmarks, info.width, info.height)
        tracked = faces.get(frame) or faces[min(faces, key=lambda f: (abs(f - frame), f))]
        pinPoints, pinPositions = self.pinsAt(self.pins(ctx, info), frame)
        geo, head, _, _ = face.solveFace(self.mesh(ctx, frame), pixels, self.camera(ctx, frame), info.width, info.height,
                                         rigid=face.RIGID_LANDMARKS, deform=ctx.value("deform"),
                                         hidden=tracked.get("hidden") or (),
                                         pins=(pinPoints, pinPositions) if len(pinPoints) else None,
                                         pinRadius=ctx.value("pinRadius"))
        return geo, head

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
                                       (0.35, 0.8, 1.0), alpha, creaseDegrees=60.0, useGpu=gpu.enabled(ctx))
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

    def runAction(self, name, context):
        """ "Track Face" (the landmarks of every frame, and their tracks) or "Clear". """
        import json
        import time
        from meshroom.live import face
        if name == "clearFace":
            return {"faceTrack": "", "tracks": "[]"}
        if name != "trackFace":
            raise RuntimeError(f"Unknown action: {name}")
        if context.info is None:
            raise RuntimeError("Connect the plate to the input.")
        try:
            detector = face.landmarker(context.value("minConfidence"))
        except face.FaceUnavailable as e:
            raise RuntimeError(f"Track Face: {e}") from e
        width, height = context.info.width, context.info.height
        faces = {}
        lastPublish = [0.0]

        def onFrame(frame, found):
            faces[frame] = found
            now = time.monotonic()
            if now - lastPublish[0] >= 0.5:
                lastPublish[0] = now
                context.publish({"faceTrack": face.dumpFaces(faces, (width, height))}, frame)
        maskAt = None
        if context.hasInput("mask"):
            from meshroom.live import ops
            from meshroom.live.imageBuffer import BBox
            channel, invert = context.value("maskChannel"), context.value("invertMask")

            def maskAt(frame):
                return ops.maskValues(context.renderInput("mask", frame), channel, invert,
                                      BBox.fromFormat(width, height))[:, :, 0]
        try:
            face.trackFaces(context.render, context.frames, detector, cancel=context.checkCancelled,
                            progress=lambda done, count: context.progress(done / float(count), "Track Face"),
                            onFrame=onFrame, maskAt=maskAt)
        finally:
            detector.close()
        if not faces:
            raise RuntimeError("No face found in the plate (lower 'Min Confidence'?).")
        export = context.value("exportTracks")
        indices = None if export == "all" else face.featureIndices()
        tracks = [] if export == "none" else face.landmarkTracks(faces, width, height, indices)
        return {"faceTrack": face.dumpFaces(faces, (width, height)), "tracks": json.dumps(tracks)}
