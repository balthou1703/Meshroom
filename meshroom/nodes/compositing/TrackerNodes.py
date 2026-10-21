"""
2D tracking (Nuke "Transform > Tracker"): Tracker, and the nodes using its tracks: Stabilize,
MotionBlur (and CornerPin, Roto in their own modules).

The tracks are edited in the viewer (Tracker selected): add a track with the toolbar, drag it on a
feature, then track forward or backward (meshroom/live/tracking.py). The node passes its input
through, or applies the average motion of its enabled tracks (match-move or stabilize). Other nodes
can read its tracks: a Roto connected to it can link its shapes to a track.
"""
__version__ = "0.1"

import numpy as np

from meshroom.core import desc
from meshroom.live.gpu import gpuParam
from meshroom.live.params import imageInput, imageOutput, maskParams, pointTrackerParam, trackRangeParams


class Tracker(desc.ImageOpNode):
    category = "Comp Transform"
    color = "#A57AAA"
    documentation = __doc__
    inputs = [
        imageInput(description="Image to track."),
        desc.StringParam(name="tracks", label="Tracks", description="Tracks as JSON (edited in the viewer).",
                         value="[]", semantic="tracks"),
        desc.ChoiceParam(name="transform", label="Transform", value="none", values=["none", "matchmove", "stabilize"],
                         description="none: the input is passed through. matchmove: the image moves with the average of "
                                     "the enabled tracks. stabilize: the opposite motion (the tracked features stay still)."),
        desc.IntParam(name="referenceFrame", label="Reference Frame",
                      description="Frame where the transform is the identity.", value=1, range=(-1000, 10000, 1)),
        desc.ChoiceParam(name="filter", label="Filter", value="bilinear", values=["impulse", "bilinear"],
                         description="Pixel filter of the transform."),
        desc.Image(name="importFrom", label="Import From", optional=True, infoOnly=True,
                   description="A CameraTracker (or another Tracker) whose tracks 'Import Tracks' copies into this Tracker."),
        desc.BoolParam(name="importReplace", label="Replace Tracks",
                       description="'Import Tracks' replaces the tracks of this Tracker (else they are added).", value=False),
        desc.PushButtonParam(name="importTracks", label="Import Tracks", semantic="liveAction",
                             description="Copy the tracks of the node connected to 'Import From' into this Tracker (they "
                                         "can then be edited), and disconnect it."),
        pointTrackerParam(),
        gpuParam(),
        *trackRangeParams(),
    ]
    outputs = imageOutput("Input image, or moved with the tracks.")
    valuesOnlyInputs = ("importFrom",)

    def infoOnlyInputs(self, ctx):
        return ["importFrom"]

    def runAction(self, name, context):
        """ "Import Tracks": the tracks of the node connected to "importFrom" added to (or replacing) the tracks. """
        import json
        from meshroom.live import tracking
        if name != "importTracks":
            raise RuntimeError(f"Unknown action: {name}")
        values = context.inputValues("importFrom")
        if values is None:
            raise RuntimeError("Connect a CameraTracker (or a Tracker) to 'Import From'.")
        try:
            imported = tracking.parseTracks(values.get("tracks", ""))
            current = [] if context.value("importReplace") else tracking.parseTracks(context.value("tracks"))
        except ValueError as e:
            raise RuntimeError(f"Invalid tracks: {e}") from e
        if not imported:
            raise RuntimeError("The connected node has no track (run its Autotrack first).")
        names = {track.get("name") for track in current}
        for track in imported:
            track = dict(track)
            base = track.get("name") or "track"
            name, index = base, 1
            while name in names:
                index += 1
                name = f"{base}_{index}"
            track["name"] = name
            names.add(name)
            current.append(track)
        return {"tracks": json.dumps(current), "_disconnect": ["importFrom"]}

    def _offset(self, ctx):
        """ Average motion (dx, dy) of the enabled tracks between the reference frame and the current frame. """
        from meshroom.live import tracking
        try:
            tracks = tracking.parseTracks(ctx.value("tracks"))
            info = ctx.inputInfo("input")
            if info is not None:
                tracks = tracking.scaleTracks(tracks, info.width, info.height)
        except ValueError:
            return 0.0, 0.0
        offsets = [tracking.trackOffset(tracks, track.get("name", ""), ctx.value("referenceFrame"), ctx.frame)
                   for track in tracks]
        offsets = [offset for offset in offsets if offset is not None]
        if not offsets:
            return 0.0, 0.0
        dx = sum(offset[0] for offset in offsets) / len(offsets)
        dy = sum(offset[1] for offset in offsets) / len(offsets)
        return (dx, dy) if ctx.value("transform") == "matchmove" else (-dx, -dy)

    def transformMatrix(self, ctx):
        """ The translation of the tracks (match-move) or its opposite (stabilize); identity with transform 'none'. """
        if ctx.value("transform") == "none":
            return None
        dx, dy = self._offset(ctx)
        return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])

    def isIdentity(self, ctx):
        if ctx.value("transform") == "none":
            return "input"
        dx, dy = self._offset(ctx)
        return "input" if abs(dx) < 1e-6 and abs(dy) < 1e-6 else None

    def render(self, ctx, inputs):
        from meshroom.live import BBox, ops
        image = inputs["input"]
        dx, dy = self._offset(ctx)
        matrix = ops.transformMatrix(translate=(dx, dy))
        shift = ops.integerTranslation(matrix)
        if shift is not None:
            return image.mapLayers(lambda pixels: pixels, bbox=image.bbox.translated(*shift))
        limit = BBox(-image.width, -image.height, 2 * image.width, 2 * image.height)
        bbox = ops.transformedBBox(image.bbox, matrix).clampedTo(limit)
        return image.mapLayers(
            lambda pixels: ops.warpAffine(pixels, image.bbox, matrix, bbox, ctx.value("filter"), ctx.checkCancelled),
            bbox=bbox)


def _trackerInputs(description):
    return [
        imageInput(description=description),
        desc.Image(name="tracker", label="Tracker", description="Tracker (or CameraTracker) whose tracks are used.",
                   optional=True, infoOnly=True),
        desc.StringParam(name="tracks", label="Tracks", semantic="trackNames",
                         description="The tracks used, chosen in the list of the connected Tracker (names comma "
                                     "separated; empty: all the enabled tracks).",
                         value=""),
    ]


def _connectedTracks(ctx):
    """ The tracks of the connected Tracker, in the pixels of the input image. """
    from meshroom.live import tracking
    try:
        tracks = tracking.parseTracks((ctx.inputValues("tracker") or {}).get("tracks", ""))
    except ValueError as e:
        raise RuntimeError(f"Invalid tracks: {e}") from e
    info = ctx.inputInfo("input")
    return tracking.scaleTracks(tracks, info.width, info.height) if info is not None else tracks


class Stabilize(desc.ImageOpNode):
    category = "Comp Transform"
    color = "#A57AAA"
    documentation = """
Stabilizes the input with the tracks of the connected Tracker: the transform moving the tracks
from the current frame back to their position at the reference frame is fitted by least squares
and applied, so that the tracked features stay still (or the opposite with "Match-Move").

Transforms and the tracks they need: translate (1), rotate (2), scale (2), translate rotate (2),
translate rotate scale (2), perspective (4: a homography, 8 values, from 4 tracks or more).
Rotate and scale alone happen around the center of the tracks.
"""
    inputs = _trackerInputs("Image to stabilize.") + [
        desc.ChoiceParam(name="model", label="Transform", value="translate rotate scale",
                         values=["translate", "rotate", "scale", "translate rotate", "translate rotate scale",
                                 "perspective"],
                         description="Components of the motion compensated."),
        desc.IntParam(name="referenceFrame", label="Reference Frame", description="Frame left unchanged.",
                      value=1, range=(-1000, 10000, 1)),
        desc.BoolParam(name="matchMove", label="Match-Move",
                       description="Apply the motion of the tracks instead of removing it.", value=False),
        desc.ChoiceParam(name="filter", label="Filter", description="Pixel filter.", value="bilinear",
                         values=["impulse", "bilinear"]),
    ]
    outputs = imageOutput("Stabilized image.")
    valuesOnlyInputs = ("tracker",)

    def infoOnlyInputs(self, ctx):
        return ["tracker"]

    def _matrix(self, ctx):
        from meshroom.live import geometry, tracking
        tracks = _connectedTracks(ctx)
        current, reference = tracking.trackPoints(tracks, tracking.trackNames(ctx.value("tracks")), ctx.frame,
                                                  ctx.value("referenceFrame"), covering=True)
        try:
            # Moves the features from where they are now back to where they were at the reference frame
            matrix = geometry.fitTransform(ctx.value("model"), current, reference)
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        return np.linalg.inv(matrix) if ctx.value("matchMove") else matrix

    def isIdentity(self, ctx):
        if ctx.frame == ctx.value("referenceFrame"):
            return "input"
        try:
            return "input" if np.allclose(self._matrix(ctx), np.eye(3), atol=1e-9) else None
        except Exception:
            return None

    def transformMatrix(self, ctx):
        if ctx.frame == ctx.value("referenceFrame"):
            return None
        return self._matrix(ctx)

    def render(self, ctx, inputs):
        from meshroom.live import ops
        return ops.transformImage(inputs["input"], self._matrix(ctx), ctx.value("filter"), ctx.checkCancelled)


class MotionBlur(desc.ImageOpNode):
    category = "Comp Filter"
    color = "#CC804E"
    documentation = """
Motion blur from tracking: the image is averaged along the motion of the tracks of the connected
Tracker during the shutter (as if the camera or the object had moved while the shutter was open).

- translate: the average motion of the tracks (the fastest);
- translate rotate scale: the motion fitted on the tracks (2 tracks or more);
- vector field: every track gives the motion around it (all the tracks, or those named): each pixel
  moves with the tracks near it (inverse distance weighting), e.g. several objects moving differently.

"Factor" multiplies the motion (2: twice as long a blur). The number of images averaged follows the
length of the motion, up to "Samples".
"""
    inputs = _trackerInputs("Image to blur.") + [
        desc.ChoiceParam(name="model", label="Motion", value="translate",
                         values=["translate", "translate rotate scale", "vector field"],
                         description="translate: average motion. translate rotate scale: fitted motion. vector field: "
                                     "every track moves the pixels around it."),
        desc.FloatParam(name="factor", label="Factor", description="Multiplier of the motion (length of the blur).",
                        value=1.0, range=(0.0, 10.0, 0.05)),
        desc.FloatParam(name="shutter", label="Shutter", description="Shutter duration in frames (0.5: 180 degrees).",
                        value=0.5, range=(0.0, 2.0, 0.05)),
        desc.ChoiceParam(name="shutterOffset", label="Shutter Offset", value="centered", values=["centered", "start", "end"],
                         description="Shutter interval relative to the frame: centered on it, starting or ending at it."),
        desc.IntParam(name="samples", label="Samples", description="Maximum number of images averaged.", value=32,
                      range=(2, 128, 1)),
    ]
    outputs = imageOutput("Blurred image.")
    valuesOnlyInputs = ("tracker",)

    def infoOnlyInputs(self, ctx):
        return ["tracker"]

    def _interval(self, ctx):
        """ (start, end) of the shutter, in frames relative to the current frame. """
        shutter = ctx.value("shutter")
        start = {"centered": -shutter / 2.0, "start": 0.0, "end": -shutter}[ctx.value("shutterOffset")]
        return start, start + shutter

    @staticmethod
    def _times(start, end, count):
        return [start + (end - start) * i / (count - 1) for i in range(count)]

    def _points(self, ctx, offset):
        """ Track positions at the current frame and at `frame + offset`. """
        from meshroom.live import tracking
        tracks = _connectedTracks(ctx)
        # The tracks known over the whole shutter (the autotrack of a CameraTracker: tracks of a few frames)
        start, end = self._interval(ctx)
        names = tracking.trackNames(ctx.value("tracks"))
        covered = [track for track in tracks if tracking.trackCovers(track, ctx.frame + start, ctx.frame + end)]
        current, other = tracking.trackPoints(covered or tracks, names, ctx.frame, ctx.frame + offset)
        if len(current) == 0:
            raise RuntimeError("No track: connect a Tracker or a CameraTracker with tracks.")
        return current, other

    def _sampleCount(self, ctx, length):
        """ About one image per pixel of motion, up to "Samples". """
        return int(min(ctx.value("samples"), max(2, np.ceil(length) + 1)))

    def isIdentity(self, ctx):
        if ctx.value("shutter") <= 0 or ctx.value("factor") <= 0:
            return "input"
        try:
            start, end = self._interval(ctx)
            a = self._points(ctx, start)
            b = self._points(ctx, end)
        except Exception:
            return None
        still = np.allclose(a[0], a[1], atol=1e-6) and np.allclose(b[0], b[1], atol=1e-6)
        return "input" if still else None

    def render(self, ctx, inputs):
        from meshroom.live import geometry, ops
        image = inputs["input"]
        factor = ctx.value("factor")
        start, end = self._interval(ctx)
        model = ctx.value("model")

        if model == "translate":
            # Average motion of the tracks, sampled along the shutter (curved paths are followed)
            current, first = self._points(ctx, start)
            _, last = self._points(ctx, end)
            length = np.linalg.norm((last - first).mean(axis=0)) * factor
            offsets = []
            for t in self._times(start, end, self._sampleCount(ctx, length)):
                a, b = self._points(ctx, t)
                offsets.append(tuple((b - a).mean(axis=0) * factor))
            return ops.shiftAccumulate(image, offsets, ctx.checkCancelled)

        if model == "translate rotate scale":
            current, _ = self._points(ctx, 0.0)
            corners = np.array([[image.bbox.x0, image.bbox.y0], [image.bbox.x1, image.bbox.y1]], dtype=np.float64)

            def matrixAt(t):
                a, b = self._points(ctx, t)
                fit = geometry.fitTransform("translate rotate scale" if len(a) >= 2 else "translate", a, b)
                # The motion scaled by the factor: interpolation between the identity and the fitted motion
                return np.eye(3) + (fit - np.eye(3)) * factor

            # Samples from the largest displacement of the image corners
            ends = [matrixAt(start), matrixAt(end)]
            moved = [(m[:2, :2] @ corners.T).T + m[:2, 2] for m in ends]
            length = float(np.abs(moved[1] - moved[0]).max())
            images = [ops.transformImage(image, matrixAt(t), "bilinear", ctx.checkCancelled)
                      for t in self._times(start, end, self._sampleCount(ctx, length))]
            return _average(image, images)

        # Vector field: the motion of each track over the shutter, interpolated between the tracks
        current, first = self._points(ctx, start)
        _, last = self._points(ctx, end)
        duration = max(end - start, 1e-6)
        velocities = (last - first) / duration * factor  # Pixels per frame
        bbox = image.bbox
        maxShift = float(np.abs(velocities).max()) * max(abs(start), abs(end)) + 1
        out = bbox.expanded(int(np.ceil(maxShift)), int(np.ceil(maxShift)))
        field = _velocityField(current, velocities, out)
        length = float(np.linalg.norm(velocities, axis=1).max()) * duration
        times = self._times(start, end, self._sampleCount(ctx, length))
        return ops.vectorBlur(image, field, out, times, ctx.checkCancelled)


def _average(reference, images):
    """ Average of transformed copies of an image (all its layers), over the union of their bboxes. """
    from collections import OrderedDict
    from meshroom.live import ops
    bbox = images[0].bbox
    for other in images[1:]:
        bbox = bbox.union(other.bbox)
    layers = OrderedDict()
    for name, channels, _ in reference.layers():
        total = None
        for other in images:
            pixels = ops.alignToBBox(other.layer(name), other.bbox, bbox)
            total = pixels.astype(np.float32) if total is None else total + pixels
        layers[name] = (channels, total / np.float32(len(images)))
    return reference.withPixels(layers, bbox)


def _velocityField(points: np.ndarray, velocities: np.ndarray, bbox, step: int = 16) -> np.ndarray:
    """
    Per-pixel velocity (h, w, 2) over `bbox`, interpolated from the velocities of the tracks at
    `points` (inverse squared distance weighting), computed on a coarse grid then resized.
    """
    from meshroom.live import ops
    if len(points) == 1:
        return np.broadcast_to(velocities[0].astype(np.float32), (bbox.height, bbox.width, 2)).copy()
    rows = max(2, bbox.height // step + 1)
    columns = max(2, bbox.width // step + 1)
    xs = np.linspace(bbox.x0 + 0.5, bbox.x1 - 0.5, columns)
    ys = np.linspace(bbox.y1 - 0.5, bbox.y0 + 0.5, rows)
    gridX, gridY = np.meshgrid(xs, ys)
    distance2 = (gridX[:, :, np.newaxis] - points[:, 0]) ** 2 + (gridY[:, :, np.newaxis] - points[:, 1]) ** 2
    weights = 1.0 / (distance2 + 1.0)
    weights /= weights.sum(axis=2, keepdims=True)
    coarse = np.einsum("hwn,nc->hwc", weights, velocities)
    return ops.resizeBilinear(coarse, bbox.height, bbox.width)


class CameraTracker(desc.ImageOpNode):
    category = "Comp Transform"
    color = "#A57AAA"
    documentation = """
Camera tracking of a sequence: "Autotrack" detects and tracks features through the whole frame
range (in the background, with progress); "Solve Camera" computes the motion of the camera and the
3D positions of the features (the focal length and the film back must be known, or "Estimate Focal
Length" searches the focal length); the frames whose camera cannot be solved are interpolated between
the solved ones (the summary gives their number); "Export USD"
writes the camera (animated) and the point cloud as .usda files. Nothing is written otherwise: the
node stays live, passing its input through.

Its tracks are those of a 2D Tracker: connect its output to the "Tracker" input of a Roto,
CornerPin, Stabilize or MotionBlur to use them.

"Preview" shows what the autotrack would do at the current frame with the current parameters: the
features detected (pattern squares, faint search areas), matched in the next frame (green: above the
minimum correlation, red: lost) with their motion. The optional "Mask" input limits the features to
its white areas (e.g. a roto of a moving object, inverted): masked out areas are shown darker.
"""
    inputs = [
        imageInput(description="Sequence to track."),
        desc.Image(name="mask", label="Mask", optional=True,
                   description="Optional mask: features are only detected and tracked where it is white."),
        *maskParams(),
        desc.BoolParam(name="preview", label="Preview",
                       description="Show the features the autotrack would detect at the current frame, and their match "
                                   "in the next frame, with the current parameters.", value=False),
        desc.IntParam(name="features", label="Features", description="Number of features tracked at each frame.",
                      value=150, range=(20, 1000, 10)),
        desc.IntParam(name="pattern", label="Pattern Size", description="Size of the tracked patterns in pixels.",
                      value=15, range=(7, 51, 2)),
        desc.IntParam(name="search", label="Search Size", description="Size of the search areas in pixels.",
                      value=41, range=(15, 201, 2)),
        desc.FloatParam(name="minScore", label="Min Correlation",
                        description="A feature is lost when its match is worse than this (1: perfect).",
                        value=0.75, range=(0.3, 0.99, 0.01)),
        desc.ChoiceParam(name="detector", label="Features", value="patches", values=["patches", "lightglue"],
                         invalidate=False,
                         description="patches: corners tracked by their patterns from frame to frame. lightglue: ALIKED "
                                     "features matched by LightGlue (robust to fast motions and changes of look; PyTorch "
                                     "and its weights needed, else patches)."),
        gpuParam(),
        *trackRangeParams(),
        desc.PushButtonParam(name="autotrack", label="Autotrack", semantic="liveAction",
                             description="Detect and track features through the frame range of the input."),
        desc.FloatParam(name="focalLength", label="Focal Length", description="Focal length of the lens in mm.",
                        value=35.0, range=(4.0, 300.0, 0.5)),
        desc.FloatParam(name="filmBackWidth", label="Film Back Width",
                        description="Width of the sensor in mm (36: full frame, 24.89: Super 35).",
                        value=36.0, range=(1.0, 70.0, 0.01)),
        desc.BoolParam(name="estimateFocal", label="Estimate Focal Length",
                       description="Unknown focal length: the solve tries focal lengths from 12 to 180 mm and keeps the "
                                   "one reprojecting the tracks the best (slower); the result is written in 'Focal Length'.",
                       value=False),
        desc.PushButtonParam(name="solve", label="Solve Camera", semantic="liveAction",
                             description="Solve the camera motion and the 3D points from the tracks."),
        desc.File(name="cameraFile", label="Camera File", description="USD file of the camera (.usda) written by "
                  "'Export USD' (folder button: choose where).", value="", exposed=False,
                  semantic="saveFile:USD files (*.usda *.usd)"),
        desc.File(name="pointsFile", label="Points File", description="USD file of the point cloud (.usda) written by "
                  "'Export USD' (folder button: choose where).", value="", exposed=False,
                  semantic="saveFile:USD files (*.usda *.usd)"),
        desc.FloatParam(name="sceneScale", label="Scene Scale",
                        description="Scale of the exported scene (the solve has an arbitrary scale).",
                        value=1.0, range=(0.001, 1000.0, 0.01)),
        desc.FloatParam(name="fps", label="Frame Rate", description="Frame rate written in the USD files.",
                        value=24.0, range=(1.0, 120.0, 1.0)),
        desc.PushButtonParam(name="exportUsd", label="Export USD", semantic="liveAction",
                             description="Write the camera and the point cloud files."),
        # Part of the uid: the nodes reading the tracks (Roto, Stabilize...) are rendered again when they change
        desc.StringParam(name="tracks", label="Tracks", description="Tracks of the autotrack (JSON).", value="[]",
                         semantic="autoTracks"),
        desc.StringParam(name="cameraSolve", label="Solve", description="Result of the camera solve (JSON).",
                         value="", semantic="cameraSolve", invalidate=False),
    ]
    outputs = imageOutput("The input, unchanged (the tracks are read by the nodes connected to it).")

    def isIdentity(self, ctx):
        return None if ctx.value("preview") else "input"

    def infoOnlyInputs(self, ctx):
        # The mask is only rendered for the preview
        return [] if ctx.value("preview") else ["mask"]

    def render(self, ctx, inputs):
        """ Preview of the autotrack at the current frame (see the documentation). """
        from meshroom.live import camera, ops
        image = inputs["input"]
        info = ctx.inputInfo("input")
        nextImage = None
        if info is None or not info.hasFrameRange or ctx.frame < info.lastFrame:
            nextImage = ctx.renderInput("input", ctx.frame + 1)
        mask = None
        if inputs.get("mask") is not None:
            mask = ops.maskValues(inputs["mask"], ctx.value("maskChannel"), ctx.value("invertMask"), image.bbox)[:, :, 0]
        features = camera.previewFeatures(image, nextImage, ctx.value("features"), ctx.value("pattern"),
                                          ctx.value("search"), ctx.value("minScore"), mask)
        rgba = camera.drawPreview(image.rgba(), image.bbox, features, ctx.value("pattern"), ctx.value("search"),
                                  ctx.value("minScore"), mask)
        return image.withRGBA(rgba)

    def runAction(self, name, context):
        """ Run an action (button) in a worker thread: returns the new values {attribute name: value}. """
        import json
        import os
        from meshroom.live import camera, ops, tracking
        if name == "autotrack":
            maskAt = None
            if context.hasInput("mask"):
                channel, invert = context.value("maskChannel"), context.value("invertMask")

                def maskAt(frame, bbox):
                    return ops.maskValues(context.renderInput("mask", frame), channel, invert, bbox)[:, :, 0]
            import time
            lastPublish = [0.0]

            def onFrame(frame, tracksSoFar):
                # Shown live (at most 4 times a second): the tracks grow in the viewer, which follows the frame
                now = time.monotonic()
                if now - lastPublish[0] >= 0.25:
                    lastPublish[0] = now
                    context.publish({"tracks": json.dumps(tracksSoFar), "cameraSolve": ""}, frame)

            from meshroom.live import gpu
            device = gpu.device(gpu.enabled(context))
            tracks = None
            if context.value("detector") == "lightglue":
                from meshroom.live.neural import NeuralUnavailable, warnUnavailable

                def autoTrackMatched(onDevice):
                    from meshroom.live.neural.matching import matcher as lightGlue
                    return camera.autoTrackMatched(context.render, context.frames, lightGlue(onDevice),
                                                   count=context.value("features"), checkCancelled=context.checkCancelled,
                                                   progress=lambda f: context.progress(f, "Autotrack (LightGlue)"),
                                                   maskAt=maskAt, onFrame=onFrame, pattern=context.value("pattern"),
                                                   search=context.value("search"))
                try:
                    # On the GPU, or on the CPU if the GPU cannot run it
                    tracks = gpu.withFallback(device, autoTrackMatched, "LightGlue")
                except NeuralUnavailable as e:
                    warnUnavailable("LightGlue", e)
            if tracks is None:
                tracks = camera.autoTrack(context.render, context.frames, count=context.value("features"),
                                          pattern=context.value("pattern"), search=context.value("search"),
                                          minScore=context.value("minScore"), checkCancelled=context.checkCancelled,
                                          progress=lambda f: context.progress(f, "Autotrack"), maskAt=maskAt,
                                          onFrame=onFrame, device=device)
            return {"tracks": json.dumps(tracks), "cameraSolve": ""}
        if name == "solve":
            tracks = tracking.parseTracks(context.value("tracks"))
            if not tracks:
                raise RuntimeError("No track: run the Autotrack first.")
            info = context.info
            if info is None:
                raise RuntimeError("The input is not connected.")
            tracks = tracking.scaleTracks(tracks, info.width, info.height)
            filmBack = context.value("filmBackWidth")
            result = {}
            if context.value("estimateFocal"):
                candidates = [mm / filmBack * info.width for mm in (12, 16, 20, 24, 28, 32, 35, 40, 50, 65, 85, 100, 135, 180)]
                focalPixels = camera.estimateFocal(tracks, info.width, info.height, candidates,
                                                   checkCancelled=context.checkCancelled,
                                                   progress=lambda f: context.progress(0.6 * f, "Estimate focal length"))
                result["focalLength"] = round(focalPixels / info.width * filmBack, 2)
                result["estimateFocal"] = False
            else:
                focalPixels = context.value("focalLength") / filmBack * info.width
            solve = camera.solveCamera(tracks, info.width, info.height, focalPixels,
                                       checkCancelled=context.checkCancelled,
                                       progress=lambda f: context.progress((0.6 + 0.4 * f) if result else f, "Solve"))
            result["cameraSolve"] = json.dumps(solve)
            return result
        if name == "exportUsd":
            solveText = context.value("cameraSolve")
            if not solveText:
                raise RuntimeError("Solve the camera first.")
            solve = json.loads(solveText)
            cameraFile, pointsFile = context.value("cameraFile"), context.value("pointsFile")
            if not cameraFile and not pointsFile:
                raise RuntimeError("Choose where to write the camera and / or points file (folder buttons).")
            # A path without extension gets ".usda"
            cameraFile, pointsFile = (path + ".usda" if path and not os.path.splitext(path)[1] else path
                                      for path in (cameraFile, pointsFile))
            for path in (cameraFile, pointsFile):
                if path and os.path.dirname(path):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
            if cameraFile:
                camera.exportCameraUsd(solve, cameraFile, context.value("focalLength"), context.value("filmBackWidth"),
                                       fps=context.value("fps"), scale=context.value("sceneScale"))
            if pointsFile:
                camera.exportPointsUsd(solve, pointsFile, scale=context.value("sceneScale"))
            return {}
        raise RuntimeError(f"Unknown action: {name}")
