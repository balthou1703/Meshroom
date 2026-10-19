"""
3D nodes of the compositing workspace: Camera, ReadGeo, MergeGeo and Resection (a GeoTracker: the pose of a
camera, or of a geometry, from pins placed on the geometry and tracked in the plate).

They are connected through camera, geo and matrix streams (desc.Camera, desc.Geo, desc.Matrix), computed per
frame in memory like the images (meshroom/live/scene3d.py). Every compositing node also has a "matrix" output,
shown when the node is expanded: the transform it applies at the current frame, to drive another node
("Input Matrix" of a Camera, a ReadGeo or a Transform).
"""
__version__ = "0.1"

import numpy as np

from meshroom.core import desc
from meshroom.live import scene3d
from meshroom.live.params import imageInput

_COLOR = "#6E9E8A"
_FILE_FILTER = "3D files (*.usd *.usda *.usdc *.usdz *.abc *.fbx *.obj *.ply)"


def vec3(name, label, description, value, range, enabled=True):
    """ A 3D value {"x", "y", "z"}. """
    return desc.GroupAttribute(
        name=name, label=label, description=description, enabled=enabled,
        items=[desc.FloatParam(name=axis, label=axis.upper(), description=f"{axis.upper()} {description[0].lower()}"
                               f"{description[1:]}", value=float(component), range=range, enabled=enabled)
               for axis, component in zip("xyz", value)])


def _xyz(value):
    return (value["x"], value["y"], value["z"])


def _rotationOrder(description="Order of the rotations (first letter first), as in Nuke's 'rotation order'."):
    return desc.ChoiceParam(name="rotationOrder", label="Rotation Order", description=description, value="ZXY",
                            values=list(scene3d.ROTATION_ORDERS))


def _inputMatrix(description):
    return desc.Matrix(name="inputMatrix", label="Input Matrix", exposed=False, optional=True, description=description)


def _trsKeys(values, order, frames=None):
    """ Keys of translate / rotate / scale reproducing 4x4 matrices {frame: matrix} (rotations kept continuous). """
    translate, rotate, scale = [], [], []
    previous = None
    for frame in sorted(values):
        t, r, s = scene3d.decomposeTRS(values[frame], order)
        if previous is not None:
            # The Euler angle the closest to the previous frame (no jump of 360 degrees)
            r = tuple(angle + 360.0 * round((last - angle) / 360.0) for angle, last in zip(r, previous))
        previous = r
        translate.append([frame, dict(zip("xyz", (float(v) for v in t)))])
        rotate.append([frame, dict(zip("xyz", (float(v) for v in r)))])
        scale.append([frame, dict(zip("xyz", (float(v) for v in s)))])
    return translate, rotate, scale


def _constantOrKeys(keys):
    """ The value of keys [[frame, value], ...] when it never changes, else the keys. """
    first = keys[0][1]
    if all(value == first for _, value in keys):
        return first
    return keys


class Camera(desc.ImageOpNode):
    category = "Comp 3D"
    color = _COLOR
    documentation = """
A camera: its transform and its lens, set by its parameters (animatable), read from a file (USD, Alembic, FBX:
the camera named in "Camera Name", or the first one), or taken from the camera connected to "Input Camera"
(e.g. the camera solved by a Resection).

"Input Matrix" (shown when the node is expanded) replaces the transform with a matrix: e.g. the "matrix"
output of another Camera, to follow its animation. Right click on its edge > "Disconnect and Bake" turns what
it gives into keys of the parameters.

Conventions: Y up, the camera looks along -Z; the horizontal aperture fits the width of the image.
"""
    inputs = [
        desc.Camera(name="input", label="Input Camera", optional=True,
                    description="When connected, the camera of this node (its own transform and lens are not used)."),
        desc.ChoiceParam(name="source", label="Source", value="parameters", values=["parameters", "file"],
                         description="parameters: the transform and lens below. file: the camera of a 3D file."),
        desc.File(name="file", label="File", value="", semantic="openFile:" + _FILE_FILTER,
                  enabled=lambda node: node.source.value == "file",
                  description="USD (.usda without the USD module), Alembic or FBX file (Alembic and FBX need their "
                              "Python modules)."),
        desc.StringParam(name="cameraName", label="Camera Name", value="",
                         enabled=lambda node: node.source.value == "file",
                         description="Name or path of the camera in the file (empty: the first camera)."),
        desc.IntParam(name="frameOffset", label="Frame Offset", value=0, range=(-10000, 10000, 1),
                      enabled=lambda node: node.source.value == "file",
                      description="Frame of the file shown at frame 0 (+10: the frame 1001 of the file at 1011)."),
        desc.FloatParam(name="fps", label="Frame Rate", value=24.0, range=(1.0, 120.0, 1.0),
                        enabled=lambda node: node.source.value == "file",
                        description="Frames per second of the Alembic / FBX file (USD: its time codes are frames)."),
        vec3("translate", "Translate", "Position of the camera.", (0.0, 0.0, 10.0), (-1000.0, 1000.0, 0.01),
             enabled=lambda node: node.source.value == "parameters"),
        vec3("rotate", "Rotate", "Rotation of the camera in degrees.", (0.0, 0.0, 0.0), (-180.0, 180.0, 0.1),
             enabled=lambda node: node.source.value == "parameters"),
        _rotationOrder(),
        desc.FloatParam(name="focalLength", label="Focal Length", value=35.0, range=(1.0, 500.0, 0.1),
                        enabled=lambda node: node.source.value == "parameters", description="Focal length in mm."),
        desc.FloatParam(name="horizontalAperture", label="Horizontal Aperture", value=36.0, range=(1.0, 100.0, 0.01),
                        enabled=lambda node: node.source.value == "parameters",
                        description="Width of the film back in mm (36: full frame, 24.89: Super 35)."),
        desc.FloatParam(name="verticalAperture", label="Vertical Aperture", value=24.0, range=(1.0, 100.0, 0.01),
                        enabled=lambda node: node.source.value == "parameters", description="Height of the film back in mm."),
        desc.FloatParam(name="near", label="Near", value=0.1, range=(0.0001, 100.0, 0.01),
                        description="Near clipping distance: the geometry closer to the camera is not drawn."),
        desc.FloatParam(name="far", label="Far", value=10000.0, range=(1.0, 1000000.0, 1.0),
                        description="Far clipping distance."),
        _inputMatrix("When connected, the transform of the camera (camera to world): e.g. the 'matrix' output of "
                     "another Camera, to follow its animation."),
    ]
    outputs = [desc.Camera(name="output", label="Camera", description="The camera at each frame.")]

    def getInfo(self, ctx, inputsInfo):
        raise RuntimeError("A Camera gives no image (connect it to a Resection, or use its matrix).")

    def computeData(self, ctx, outputName):
        if outputName == "matrix":
            return ctx.ownData("output", ctx.frame).matrix
        camera = ctx.inputData("input")
        if camera is None:
            if ctx.value("source") == "file":
                camera = scene3d.readSceneFile(ctx.value("file"), ctx.value("fps")).camera(
                    ctx.frame - ctx.value("frameOffset"), ctx.value("cameraName"))
                camera = camera.copy(near=ctx.value("near"), far=ctx.value("far"))
            else:
                camera = scene3d.CameraData(
                    scene3d.composeTRS(_xyz(ctx.value("translate")), _xyz(ctx.value("rotate")),
                                       order=ctx.value("rotationOrder")),
                    ctx.value("focalLength"), ctx.value("horizontalAperture"), ctx.value("verticalAperture"),
                    ctx.value("near"), ctx.value("far"))
        matrix = ctx.inputData("inputMatrix")
        if matrix is not None:
            camera = camera.copy(matrix=matrix)
        return camera

    def bakeStream(self, ctx, inputName, values):
        """ Keys of the transform (and of the lens for an input camera) giving what the input gave. """
        order = ctx.value("rotationOrder")
        if inputName == "inputMatrix":
            translate, rotate, _ = _trsKeys(values, order)
            return {"source": "parameters", "translate": _constantOrKeys(translate), "rotate": _constantOrKeys(rotate)}
        if inputName == "input":
            translate, rotate, _ = _trsKeys({frame: camera.matrix for frame, camera in values.items()}, order)
            result = {"source": "parameters", "translate": _constantOrKeys(translate), "rotate": _constantOrKeys(rotate)}
            for name in ("focalLength", "horizontalAperture", "verticalAperture", "near", "far"):
                result[name] = _constantOrKeys([[frame, float(getattr(camera, name))]
                                                for frame, camera in sorted(values.items())])
            return result
        return None


class ReadGeo(desc.ImageOpNode):
    category = "Comp 3D"
    color = _COLOR
    documentation = """
Reads a geometry (meshes, point clouds) from a file: OBJ, PLY, USD (.usda without the USD module), Alembic or
FBX (with their Python modules). Animated files (time samples) are read at each frame. "Translate", "Rotate",
"Scale" transform it, or "Input Matrix" (shown when the node is expanded), e.g. the "Geo Matrix" output of a
Resection to put the geometry where it is in the plate.
"""
    inputs = [
        desc.File(name="file", label="File", value="", semantic="openFile:" + _FILE_FILTER,
                  description="OBJ, PLY, USD, Alembic or FBX file."),
        desc.StringParam(name="objectName", label="Object Name", value="",
                         description="Name or path of the geometry in the file (empty: all of them)."),
        desc.IntParam(name="frameOffset", label="Frame Offset", value=0, range=(-10000, 10000, 1),
                      description="Frame of the file shown at frame 0 (+10: the frame 1001 of the file at 1011)."),
        desc.FloatParam(name="fps", label="Frame Rate", value=24.0, range=(1.0, 120.0, 1.0),
                        description="Frames per second of the Alembic / FBX file (USD: its time codes are frames)."),
        vec3("translate", "Translate", "Translation of the geometry.", (0.0, 0.0, 0.0), (-1000.0, 1000.0, 0.01)),
        vec3("rotate", "Rotate", "Rotation of the geometry in degrees.", (0.0, 0.0, 0.0), (-180.0, 180.0, 0.1)),
        vec3("scale", "Scale", "Scale of the geometry.", (1.0, 1.0, 1.0), (0.0, 10.0, 0.01)),
        desc.FloatParam(name="uniformScale", label="Uniform Scale", value=1.0, range=(0.0, 100.0, 0.01),
                        description="Scale applied on the 3 axes."),
        _rotationOrder(),
        _inputMatrix("When connected, the transform of the geometry (replacing translate / rotate / scale): e.g. "
                     "the 'Geo Matrix' of a Resection."),
    ]
    outputs = [desc.Geo(name="output", label="Geo", description="The geometry at each frame, in world space.")]

    def getInfo(self, ctx, inputsInfo):
        raise RuntimeError("A ReadGeo gives no image (connect it to a Resection).")

    def transformMatrix(self, ctx):
        matrix = ctx.inputData("inputMatrix")
        if matrix is not None:
            return matrix
        uniform = ctx.value("uniformScale")
        scale = [v * uniform for v in _xyz(ctx.value("scale"))]
        return scene3d.composeTRS(_xyz(ctx.value("translate")), _xyz(ctx.value("rotate")), scale,
                                  ctx.value("rotationOrder"))

    def computeData(self, ctx, outputName):
        if outputName == "output":
            geo = scene3d.readSceneFile(ctx.value("file"), ctx.value("fps")).geo(ctx.frame - ctx.value("frameOffset"),
                                                                                 ctx.value("objectName"))
            return geo.transformed(self.transformMatrix(ctx))
        return super().computeData(ctx, outputName)

    def bakeStream(self, ctx, inputName, values):
        if inputName != "inputMatrix":
            return None
        translate, rotate, scale = _trsKeys(values, ctx.value("rotationOrder"))
        return {"translate": _constantOrKeys(translate), "rotate": _constantOrKeys(rotate),
                "scale": _constantOrKeys(scale), "uniformScale": 1.0}


class MergeGeo(desc.ImageOpNode):
    category = "Comp 3D"
    color = _COLOR
    documentation = """
Merges up to 4 geometries into one.
"""
    inputs = [desc.Geo(name=f"input{i}", label=f"Input {i}", optional=True, description="A geometry.")
              for i in range(1, 5)]
    outputs = [desc.Geo(name="output", label="Geo", description="The geometries together.")]

    def getInfo(self, ctx, inputsInfo):
        raise RuntimeError("A MergeGeo gives no image (connect it to a Resection).")

    def computeData(self, ctx, outputName):
        if outputName == "output":
            return scene3d.GeoData.merged([ctx.inputData(f"input{i}") for i in range(1, 5)])
        return super().computeData(ctx, outputName)


class Resection(desc.ImageOpNode):
    category = "Comp 3D"
    color = _COLOR
    documentation = """
Finds the camera (or the motion of the geometry) matching a geometry to the plate, like GeoTracker: pins put on
the geometry are moved onto the plate, and tracked.

- Evaluate the Resection: the plate is shown with the geometry over it (wireframe), seen from the solved camera.
- Select it, "Add pin" in the toolbar of the viewer and click on the geometry: a pin is stuck to the surface (or
  to the nearest point of a point cloud); drag it onto its place in the plate. The pose follows: 1 pin moves the
  geometry in the image, 2 pins also turn and scale it, 3 pins and more solve the pose (4 or more: exactly).
- Track the pins forward / backward (as the tracks of a Tracker): the pose is solved at every frame with them;
  the frames where pins were placed by hand are their keyframes.

The lens (focal length, film back) and the starting pose come from "Camera" (without it: 35 mm on a 36 mm film
back, looking at the geometry). "Center Geometry" steps the starting camera back along its view axis until the whole
geometry is centered in the view (a large geometry, the camera close to its center: hard to find or to grab). The solve keeps the geometry still: "Camera" is the camera moving around it,
"Camera Matrix" its transform; "Geo Matrix" is the transform moving the geometry in front of the input camera
instead (connect it to the "Input Matrix" of a ReadGeo). Right click on an edge > "Disconnect and Bake" turns them
into keys of a Camera or a ReadGeo.
"""
    inputs = [
        imageInput(description="The plate."),
        desc.Geo(name="geo", label="Geo", description="The geometry to match (e.g. a ReadGeo)."),
        desc.Camera(name="camera", label="Camera", optional=True,
                    description="The lens and the starting pose (the pose of the frames without pins)."),
        desc.BoolParam(name="centerGeometry", label="Center Geometry", value=False,
                       description="Start from the camera stepped back along its view axis until the whole geometry is "
                                   "centered in the view (e.g. a large geometry, the camera close to its center: hard to "
                                   "find or to grab). The pins then solve from there."),
        desc.StringParam(name="tracks", label="Pins", value="[]", semantic="tracks",
                         description="Pins as JSON: a tracked image position (as the tracks of a Tracker) and its point "
                                     "on the geometry (\"point\": [x, y, z]). Edited in the viewer."),
        desc.ChoiceParam(name="display", label="Display", value="wireframe", values=["wireframe", "points", "none"],
                         description="What is drawn over the plate: the edges of the geometry, its vertices, or nothing."),
        desc.FloatParam(name="opacity", label="Opacity", value=0.6, range=(0.0, 1.0, 0.01),
                        description="Opacity of the geometry drawn over the plate."),
        desc.IntParam(name="maxEdges", label="Max Edges", value=100000, range=(1000, 2000000, 1000), advanced=True,
                      description="Number of edges drawn at most (a dense mesh is thinned out)."),
    ]
    outputs = [
        desc.Camera(name="cameraOutput", label="Camera",
                    description="The solved camera (the lens of the input camera), the geometry staying still."),
        desc.Matrix(name="cameraMatrix", label="Camera Matrix", description="Transform of the solved camera."),
        desc.Matrix(name="geoMatrix", label="Geo Matrix",
                    description="Transform of the geometry matching the plate, seen from the input camera."),
        desc.Image(name="output", label="Output", description="The plate, with the geometry and the pins drawn over it."),
    ]

    def isIdentity(self, ctx):
        return "input" if ctx.value("display") == "none" else None

    # Pins

    @staticmethod
    def pins(ctx, info=None):
        """ The enabled pins with a point on the geometry, in the pixels of the plate. """
        from meshroom.live import tracking
        try:
            tracks = tracking.parseTracks(ctx.value("tracks"))
        except ValueError as e:
            raise RuntimeError(f"Invalid pins: {e}") from e
        info = info or ctx.inputInfo("input")
        if info is not None:
            tracks = tracking.scaleTracks(tracks, info.width, info.height)
        return [track for track in tracks if track.get("enabled", True) and len(track.get("point") or []) == 3]

    @staticmethod
    def _pinsAt(pins, frame):
        from meshroom.live import tracking
        points, positions = [], []
        for pin in pins:
            keys = tracking.trackKeys(pin)
            # A pin counts between its first and last keys; a pin with a single key holds it
            if not keys or (len(keys) > 1 and not keys[0][0] <= frame <= keys[-1][0]):
                continue
            points.append(pin["point"])
            positions.append(tracking.trackPosition(pin, frame))
        return np.array(points, dtype=np.float64).reshape(-1, 3), np.array(positions, dtype=np.float64).reshape(-1, 2)

    def inputCamera(self, ctx, frame=None):
        """ The camera of the "camera" input at a frame, or a camera looking at the geometry. """
        camera = ctx.inputData("camera", frame)
        if camera is not None:
            return camera
        return scene3d.framingCamera(ctx.inputData("geo", frame))

    def startCamera(self, ctx, frame=None, info=None):
        """ The pose the solve starts from: the input camera, or with "Center Geometry", stepped back to center it. """
        camera = self.inputCamera(ctx, frame)
        if not ctx.value("centerGeometry"):
            return camera
        info = info or ctx.inputInfo("input")
        aspect = info.width / float(info.height) if info is not None and info.height else 16.0 / 9.0
        return scene3d.centeredCamera(camera, ctx.inputData("geo", frame), aspect)

    # Solve

    def _solve(self, ctx, frame, initial, pins, info):
        points, positions = self._pinsAt(pins, frame)
        camera = self.startCamera(ctx, frame, info)
        if not len(points):
            return camera.copy(matrix=initial if initial is not None else camera.matrix), 0.0
        start = initial if initial is not None else camera.matrix
        matrix, error = scene3d.solvePose(points, positions, info.width, info.height, camera.focalPixels(info.width),
                                          start)
        return camera.copy(matrix=matrix), error

    def computeData(self, ctx, outputName):
        if outputName == "cameraMatrix" or outputName == "matrix":
            return ctx.ownData("cameraOutput", ctx.frame).matrix
        if outputName == "geoMatrix":
            solved = ctx.ownData("cameraOutput", ctx.frame)
            return self.inputCamera(ctx).matrix @ np.linalg.inv(solved.matrix)
        if outputName != "cameraOutput":
            return super().computeData(ctx, outputName)
        info = ctx.inputInfo("input")
        if info is None:
            raise RuntimeError("Connect the plate to the input.")
        pins = self.pins(ctx, info)
        if not pins:
            return self.startCamera(ctx, info=info)
        # Solved frame after frame from the nearest frame with pins placed by hand (each pose starts from the one
        # of the previous frame): the poses on the way are cached
        keyframes = sorted({int(f) for pin in pins for f in (pin.get("keyframes") or [])} or
                           {int(f) for pin in pins for f in (pin.get("keys") or {})})
        frame = ctx.frame
        anchor = min(keyframes, key=lambda f: (abs(f - frame), f)) if keyframes else frame
        step = 1 if frame >= anchor else -1
        evaluator, uid = ctx.evaluator, ctx.planNode.uid
        initial = None
        for current in range(anchor, frame + step, step):
            cached = evaluator.cachedData(uid, "cameraOutput", current) if current != frame else None
            if cached is not None:
                initial = cached.matrix
                continue
            ctx.checkCancelled()
            solved, _ = self._solve(ctx, current, initial, pins, info)
            initial = solved.matrix
            if current != frame:
                evaluator.storeData(uid, "cameraOutput", current, solved)
        return solved

    def reprojectionError(self, ctx) -> float:
        """ RMS distance in pixels between the pins and their points seen from the solved camera, at the frame. """
        info = ctx.inputInfo("input")
        points, positions = self._pinsAt(self.pins(ctx, info), ctx.frame)
        if not len(points):
            return 0.0
        camera = ctx.ownData("cameraOutput", ctx.frame)
        xy, _, _ = camera.project(points, info.width, info.height)
        return float(np.sqrt(np.mean(np.sum((xy - positions) ** 2, axis=1))))

    def pickPoint(self, ctx, x, y):
        """ The point of the geometry under the image position (x, y) seen from the solved camera, or None. """
        info = ctx.inputInfo("input")
        geo = ctx.inputData("geo")
        if info is None or geo is None:
            return None
        camera = ctx.ownData("cameraOutput", ctx.frame)
        if len(geo.faceCounts):
            origin, direction = camera.ray(x, y, info.width, info.height)
            return scene3d.raycast(geo, origin, direction)
        return scene3d.nearestVertexOnScreen(geo, camera, x, y, info.width, info.height, radius=12.0)

    def render(self, ctx, inputs):
        image = inputs["input"]
        geo = ctx.inputData("geo")
        camera = ctx.ownData("cameraOutput", ctx.frame)
        rgba = np.array(image.rgba(), dtype=np.float32)
        width, height = image.width, image.height
        alpha = ctx.value("opacity")
        if ctx.value("display") == "wireframe":
            scene3d.drawGeo(rgba, image.bbox, geo, camera, width, height, (0.35, 1.0, 0.45), alpha,
                            maxEdges=ctx.value("maxEdges"))
        elif geo is not None and not geo.isEmpty:
            xy, _, inFront = camera.project(geo.vertices, width, height)
            scene3d.drawPoints(rgba, image.bbox, xy[inFront], (0.35, 1.0, 0.45), alpha, 2)
        # The points of the pins seen from the solved camera, linked to the pins (the error)
        points, positions = self._pinsAt(self.pins(ctx, image.info), ctx.frame)
        if len(points):
            xy, _, _ = camera.project(points, width, height)
            scene3d.drawSegments(rgba, image.bbox, xy, positions, (1.0, 0.3, 0.3), 1.0)
            for dx in range(-3, 4):
                scene3d.drawSegments(rgba, image.bbox, xy + [dx, -3], xy + [dx, 3], (1.0, 0.85, 0.2), 1.0)
        return image.withRGBA(rgba)
