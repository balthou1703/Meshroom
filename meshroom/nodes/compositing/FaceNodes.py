"""
FaceTracker: a face followed in the plate, non rigid (MediaPipe Face Landmarker, see meshroom/live/face.py).
"""
__version__ = "0.1"

import numpy as np

from meshroom.core import desc
from meshroom.live import scene3d
from meshroom.live.params import imageInput

_COLOR = "#6E9E8A"


class FaceTracker(desc.ImageOpNode):
    category = "Comp 3D"
    color = _COLOR
    documentation = """
Tracks a face in the plate, mouth and eyes included (non rigid), like KeenTools FaceTracker.

- "Track Face": MediaPipe Face Landmarker finds the 478 landmarks of the face at every frame of the plate (it follows
  the face from frame to frame), with 52 expression coefficients. They are stored in the node.
- The face mesh is then fitted at each frame: the head pose from the landmarks that move little with the expressions
  (forehead, nose bridge, eye corners, temples), then, with "Non Rigid", every vertex moved at its depth onto its
  landmark: the lips and the eyelids follow.
- "Face": any mesh with the MediaPipe face topology (468 or 478 vertices in the landmark order), e.g. a scan of the
  actor retopologized on it; its shape is the neutral face. Without it: the canonical face of MediaPipe.

Outputs: "Face" (the face mesh at each frame, in the world of the camera), "Head Matrix" (the rigid pose of the head:
face space to world, e.g. for the "Input Matrix" of a ReadGeo), "Camera" (the plate camera), and the plate with the face
drawn. Its tracks (the landmarks, named by region: lips_013, leftEye_033...) can drive a Roto, a Stabilize or a
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
        geo, head, _, _ = face.solveFace(self.mesh(ctx, frame), pixels, self.camera(ctx, frame), info.width, info.height,
                                         rigid=face.RIGID_LANDMARKS, deform=ctx.value("deform"))
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
        try:
            face.trackFaces(context.render, context.frames, detector, cancel=context.checkCancelled,
                            progress=lambda done, count: context.progress(done / float(count), "Track Face"),
                            onFrame=onFrame)
        finally:
            detector.close()
        if not faces:
            raise RuntimeError("No face found in the plate (lower 'Min Confidence'?).")
        export = context.value("exportTracks")
        indices = None if export == "all" else face.featureIndices()
        tracks = [] if export == "none" else face.landmarkTracks(faces, width, height, indices)
        return {"faceTrack": face.dumpFaces(faces, (width, height)), "tracks": json.dumps(tracks)}
