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
from meshroom.live.gpu import gpuParam
from meshroom.live.params import imageInput, maskParams, pointTrackerParam, trackRangeParams

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

- Evaluate the Resection: the plate is shown with the geometry over it, seen from the solved camera ("Display":
  shaded surface, feature edges, wireframe or points, the hidden parts removed).
- Select it, "Add pin" in the toolbar of the viewer and click on the geometry: a pin is stuck to the surface (or
  to the nearest point of a point cloud); drag it onto its place in the plate. The pose follows: 1 pin moves the
  geometry in the image, 2 pins also turn and scale it, 3 pins and more solve the pose (4 or more: exactly).
- Track the pins forward / backward (as the tracks of a Tracker): all the pins move together frame after frame, each
  searched where the pose solved with the others puts it; a pin that stops fitting the pose stops there. The pose is
  solved at every frame with them; the frames where pins were placed by hand are their keyframes. As in GeoTracker, a
  pin only constrains the frames where it has a key (placed or tracked): elsewhere it follows the geometry (shown
  faint), and the poses between two solved frames are interpolated.
- Or track the geometry itself ("Track Geometry" backward / forward, from the current frame to the next keyframe): the
  pins only place it at a keyframe, then many points of its surface follow the plate, by optical flow (or patches);
  each pose is checked against the keyframe to limit the drift. "Align" finds the right position at the current frame
  from a rough one (the geometry put near its place by hand), against the nearest keyframe, and makes it a keyframe.
  "Refine" tracks again between each pair of keyframes from both ends and blends them (the drift spread between them).
  The tracked poses are stored in "Geometry Track" and prevail over the pins, except at the keyframes of the pins
  (where the pins adjust them); "Clear Geometry Track" removes them. The optional "Mask" limits the points followed
  (and the outline snapped by "Align With" silhouette) to its white areas: an occluding object or a moving part left
  out with an inverted roto.

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
        desc.Image(name="mask", label="Mask", optional=True, infoOnly=True,
                   description="Optional mask: the geometry is only followed (Track Geometry, Align, Refine) where it "
                               "is white, e.g. a roto of an occluding object or of a moving part, inverted."),
        *maskParams(),
        desc.BoolParam(name="centerGeometry", label="Center Geometry", value=False,
                       description="Start from the camera stepped back along its view axis until the whole geometry is "
                                   "centered in the view (e.g. a large geometry, the camera close to its center: hard to "
                                   "find or to grab). The pins then solve from there."),
        desc.StringParam(name="tracks", label="Pins", value="[]", semantic="tracks",
                         description="Pins as JSON: a tracked image position (as the tracks of a Tracker) and its point "
                                     "on the geometry (\"point\": [x, y, z]). Edited in the viewer."),
        desc.ChoiceParam(name="trackMethod", label="Track Geometry With", value="flow", values=["flow", "raft", "patches"],
                         invalidate=False,
                         description="How the points of the geometry follow the plate: flow (the Lucas-Kanade optical "
                                     "flow of the plate), raft (the RAFT optical flow network: large motions, sharp "
                                     "edges; PyTorch and its weights needed, else flow) or patches (the image around "
                                     "each point searched in the next frame)."),
        desc.ChoiceParam(name="alignMethod", label="Align With", value="patches", values=["patches", "silhouette", "lightglue", "megapose"],
                         invalidate=False,
                         description="How 'Align' finds the right position: patches (the points of the nearest keyframe "
                                     "searched close to where the current pose puts them: the geometry must be roughly "
                                     "placed), silhouette (the outline of the geometry snapped onto the edges of "
                                     "the image: no keyframe nor network; the geometry roughly placed, its outline "
                                     "standing out of the background), lightglue (features of the keyframe matched by LightGlue: far from its "
                                     "place, other angle), megapose (MegaPose finds the pose of the mesh in the image, "
                                     "without keyframe, from where the current pose shows it; a CUDA GPU needed). Then "
                                     "refined on the patches."),
        desc.FloatParam(name="silhouetteDistance", label="Silhouette Search", value=40.0, range=(5.0, 200.0, 1.0),
                        invalidate=False, advanced=True,
                        description="Align With silhouette: how far (pixels) the edges of the image are searched across "
                                    "the outline of the geometry. Larger: from farther, but more risk of snapping onto "
                                    "another edge."),
        pointTrackerParam(),
        gpuParam(),
        *trackRangeParams(),
        desc.IntParam(name="trackPoints", label="Track Points", value=150, range=(30, 1000, 10), invalidate=False,
                      description="Number of points of the geometry (on textured parts of the plate) followed at each "
                                  "frame by 'Track Geometry'."),
        desc.BoolParam(name="trackAnchored", label="Anchor To Keyframe", value=True, invalidate=False, advanced=True,
                       description="Check each tracked pose against the keyframe the track started from (less drift, "
                                   "slower)."),
        desc.PushButtonParam(name="trackGeoBackward", label="Track Geometry Backward", semantic="liveAction",
                             description="Track the geometry from the current frame back to the previous keyframe."),
        desc.PushButtonParam(name="trackGeoForward", label="Track Geometry Forward", semantic="liveAction",
                             description="Track the geometry from the current frame to the next keyframe."),
        desc.PushButtonParam(name="alignGeo", label="Align", semantic="liveAction",
                             description="Find the right position of the geometry at the current frame (from a rough "
                                         "one), against the nearest keyframe; the frame becomes a keyframe."),
        desc.PushButtonParam(name="refineGeo", label="Refine", semantic="liveAction",
                             description="Track the geometry again between each pair of keyframes, from both ends, and "
                                         "blend the two tracks."),
        desc.PushButtonParam(name="clearGeoTrack", label="Clear Geometry Track", semantic="liveAction",
                             description="Remove the poses tracked with 'Track Geometry' (the pins solve again)."),
        desc.StringParam(name="geoTrack", label="Geometry Track", value="", semantic="geoTrack", advanced=True,
                         description="Poses tracked with 'Track Geometry' and the keyframes (JSON)."),
        desc.PushButtonParam(name="deformBackward", label="Deform With Flow Backward", semantic="liveAction",
                             description="Deform the geometry with the optical flow of the plate (non rigid), from the "
                                         "current frame back to the start of the track range."),
        desc.PushButtonParam(name="deformForward", label="Deform With Flow Forward", semantic="liveAction",
                             description="Deform the geometry with the optical flow of the plate (non rigid), from the "
                                         "current frame to the end of the track range."),
        desc.PushButtonParam(name="clearDeform", label="Clear Deformation", semantic="liveAction",
                             description="Remove the deformation (the rigid geometry)."),
        desc.IntParam(name="deformPoints", label="Deform Points", value=400, range=(20, 5000, 10), invalidate=False,
                      advanced=True,
                      description="Number of vertices of the geometry following the flow ('Deform With Flow'): the "
                                  "others follow them smoothly. More: finer deformations, more noise. Taken at the "
                                  "first deformation (Clear Deformation to change it)."),
        desc.StringParam(name="geoDeform", label="Geometry Deformation", value="", semantic="geoDeform", advanced=True,
                         description="The deformation by 'Deform With Flow' (JSON: control vertices and their offsets "
                                     "in pixels at each frame)."),
        desc.ChoiceParam(name="display", label="Display", value="wireframe",
                         values=["shaded", "edges", "wireframe", "points", "none"],
                         description="What is drawn over the plate: shaded (the surface lit from the camera, with its "
                                     "edges), edges (silhouettes, creases and open edges only: the shape of a dense "
                                     "mesh), wireframe (all the edges of the faces), points (the vertices), or nothing."),
        desc.BoolParam(name="hiddenLines", label="Hide Hidden Lines", value=True,
                       description="The edges and points behind the surface of the geometry are not drawn."),
        desc.FloatParam(name="creaseAngle", label="Crease Angle", value=40.0, range=(1.0, 90.0, 1.0), advanced=True,
                        description="'shaded' and 'edges': an edge between two faces at more than this angle (degrees) "
                                    "is drawn."),
        desc.FloatParam(name="opacity", label="Opacity", value=0.6, range=(0.0, 1.0, 0.01),
                        description="Opacity of the geometry drawn over the plate."),
        desc.IntParam(name="maxEdges", label="Max Edges", value=500000, range=(1000, 2000000, 1000), advanced=True,
                      description="Number of edges drawn at most (a dense mesh is thinned out)."),
    ]
    outputs = [
        desc.Camera(name="cameraOutput", label="Camera",
                    description="The solved camera (the lens of the input camera), the geometry staying still."),
        desc.Matrix(name="cameraMatrix", label="Camera Matrix", description="Transform of the solved camera."),
        desc.Matrix(name="geoMatrix", label="Geo Matrix",
                    description="Transform of the geometry matching the plate, seen from the input camera."),
        desc.Geo(name="geoOutput", label="Geo",
                 description="The geometry deformed by 'Deform With Flow' (in the world of the input geometry, seen by "
                             "'Camera'); the input geometry when it is not deformed."),
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
        """ (points (k, 3), image positions (k, 2)) of the pins with a key at this frame (placed by hand or tracked): as
        in GeoTracker, a pin only constrains the frames where it has a key; elsewhere it follows the geometry. """
        points, positions = [], []
        for pin in pins:
            key = (pin.get("keys") or {}).get(str(int(frame)))
            if key is None or int(frame) != frame:
                continue
            points.append(pin["point"])
            positions.append((float(key[0]), float(key[1])))
        return np.array(points, dtype=np.float64).reshape(-1, 3), np.array(positions, dtype=np.float64).reshape(-1, 2)

    @staticmethod
    def pinFrames(pins):
        """ The frames where at least one pin has a key (placed by hand or tracked): the frames the pins solve. """
        return sorted({int(f) for pin in pins for f in (pin.get("keys") or {})})

    @staticmethod
    def pinKeyframes(pins):
        """ The frames where pins were placed by hand (all the frames of their keys for pins without keyframes). """
        return sorted({int(f) for pin in pins for f in (pin.get("keyframes") or [])} or
                      {int(f) for pin in pins for f in (pin.get("keys") or {})})

    @staticmethod
    def geoTrack(ctx):
        """ The poses tracked with "Track Geometry" ({frame: camera-to-world matrix}) and their keyframes. """
        from meshroom.live import geotrack
        try:
            return geotrack.parsePoses(ctx.value("geoTrack"))
        except ValueError as e:
            raise RuntimeError(str(e)) from e

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
        if outputName == "geoOutput":
            return self.deformedGeo(ctx)
        if outputName != "cameraOutput":
            return super().computeData(ctx, outputName)
        info = ctx.inputInfo("input")
        if info is None:
            raise RuntimeError("Connect the plate to the input.")
        pins = self.pins(ctx, info)
        keyframes = self.pinKeyframes(pins)
        frame = ctx.frame
        # A pose tracked with "Track Geometry" prevails; at a keyframe of the pins, they adjust it
        poses, _ = self.geoTrack(ctx)
        if frame in poses:
            if frame in keyframes:
                return self._solve(ctx, frame, poses[frame], pins, info)[0]
            return self.startCamera(ctx, frame, info).copy(matrix=poses[frame])
        # The frames with a pose: tracked, or solved by the pins with a key there
        anchors = sorted(set(self.pinFrames(pins)) | set(poses))
        if not anchors:
            return self.startCamera(ctx, info=info)
        if frame not in anchors:
            # Between two of them, the pose is interpolated; before the first and after the last, held
            before = [f for f in anchors if f < frame]
            after = [f for f in anchors if f > frame]
            if before and after:
                a, b = before[-1], after[0]
                matrix = scene3d.interpolateMatrices(ctx.ownData("cameraOutput", a).matrix,
                                                     ctx.ownData("cameraOutput", b).matrix, (frame - a) / float(b - a))
            else:
                matrix = ctx.ownData("cameraOutput", before[-1] if before else after[0]).matrix
            return self.startCamera(ctx, frame, info).copy(matrix=matrix)
        # Solved from frame with pins to frame with pins, from the nearest tracked pose (or the first frame with pins),
        # each solve starting from the previous pose: the poses on the way are cached
        roots = [f for f in anchors if f in poses] or [anchors[0]]
        root = min(roots, key=lambda f: (abs(f - frame), f))
        path = [f for f in anchors if min(root, frame) <= f <= max(root, frame)]
        path = path if root <= frame else path[::-1]
        evaluator, uid = ctx.evaluator, ctx.planNode.uid
        initial, solved = None, None
        for current in path:
            if current in poses and current != frame:
                initial = ctx.ownData("cameraOutput", current).matrix
                continue
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

    # Tracking

    def trackingModel(self, ctx, tracks):
        """ The pins tracked together (meshroom/live/tracking.py): all on the geometry, they move with one pose. """
        info = ctx.inputInfo("input")
        if info is None:
            return None
        camera = self.startCamera(ctx, info=info)
        try:
            start = ctx.ownData("cameraOutput", ctx.frame).matrix
        except Exception:
            start = camera.matrix
        return _PinsModel(tracks, info.width, info.height, camera.focalPixels(info.width), start)

    def runAction(self, name, context):
        """ Geometry tracking actions (see the documentation): returns {"geoTrack": the new poses}. """
        if name == "clearGeoTrack":
            return {"geoTrack": ""}
        if name == "clearDeform":
            return {"geoDeform": ""}
        if name not in ("trackGeoForward", "trackGeoBackward", "alignGeo", "refineGeo", "deformForward", "deformBackward"):
            raise RuntimeError(f"Unknown action: {name}")
        return _GeoTrackAction(self, context).run(name)

    def deformedGeo(self, ctx):
        """ The input geometry deformed by "Deform With Flow" at the frame (seen from the solved camera). """
        from meshroom.live import geotrack
        geo = ctx.inputData("geo")
        try:
            deform = geotrack.parseDeform(ctx.value("geoDeform"))
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        info = ctx.inputInfo("input")
        offsets = geotrack.deformOffsetsAt(deform, ctx.frame) if deform else None
        if geo is None or geo.isEmpty or offsets is None or info is None or len(deform["indices"]) != len(offsets) \
                or int(deform["indices"].max()) >= len(geo.vertices):
            return geo
        camera = ctx.ownData("cameraOutput", ctx.frame)
        weights = geotrack.DeformWeights(geo, deform["indices"], deform["radius"])
        vertices = geotrack.deformVertices(geo.vertices, camera.matrix, info.width, info.height,
                                           camera.focalPixels(info.width), weights.shifts(offsets))
        return scene3d.GeoData(vertices, name=geo.name, topology=geo.topology)

    def reprojectionError(self, ctx) -> float:
        """ RMS distance in pixels between the pins and their points seen from the solved camera, at the frame. """
        info = ctx.inputInfo("input")
        points, positions = self._pinsAt(self.pins(ctx, info), ctx.frame)
        if not len(points):
            return 0.0
        camera = ctx.ownData("cameraOutput", ctx.frame)
        xy, _, _ = camera.project(points, info.width, info.height)
        return float(np.sqrt(np.mean(np.sum((xy - positions) ** 2, axis=1))))

    def pinProjections(self, ctx, cachedData) -> dict:
        """ {pin name: [x, y]}: where each pin is seen at the frame, its point projected by the solved camera
        (`cachedData(output)`: the cached output at the frame, or None), in the pixels of the plate of `ctx`. """
        camera = cachedData("cameraOutput")
        info = ctx.inputInfo("input")
        pins = self.pins(ctx, info) if info is not None else []
        if camera is None or not pins:
            return {}
        xy, _, inFront = camera.project(np.array([pin["point"] for pin in pins], dtype=np.float64), info.width, info.height)
        return {pin["name"]: [float(x), float(y)] for pin, (x, y), front in zip(pins, xy, inFront) if front}

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
        geo = ctx.ownData("geoOutput", ctx.frame)
        camera = ctx.ownData("cameraOutput", ctx.frame)
        from meshroom.live import geoRender
        rgba = np.array(image.rgba(), dtype=np.float32)
        width, height = image.width, image.height
        from meshroom.live import gpu
        geoRender.drawGeometry(rgba, image.bbox, geo, camera, width, height, ctx.value("display"), (0.35, 1.0, 0.45),
                               ctx.value("opacity"), ctx.value("creaseAngle"), ctx.value("hiddenLines"),
                               ctx.value("maxEdges"), useGpu=gpu.enabled(ctx))
        # The points of the pins seen from the solved camera, linked to the pins (the error)
        points, positions = self._pinsAt(self.pins(ctx, image.info), ctx.frame)
        if len(points):
            xy, _, _ = camera.project(points, width, height)
            scene3d.drawSegments(rgba, image.bbox, xy, positions, (1.0, 0.3, 0.3), 1.0)
            for dx in range(-3, 4):
                scene3d.drawSegments(rgba, image.bbox, xy + [dx, -3], xy + [dx, 3], (1.0, 0.85, 0.2), 1.0)
        return image.withRGBA(rgba)


class _PinsModel:
    """
    The pins of a Resection tracked together (a tracking.TrackingModel): at each frame the pose is solved with the pins
    found; the pins far from where it puts their point are rejected, and each pin is searched in the next frame where
    the pose, moving at constant speed, puts it.
    """

    def __init__(self, tracks, width, height, focalPixels, start):
        self.points = {index: np.array(track["point"], dtype=np.float64) for index, track in enumerate(tracks)
                       if len(track.get("point") or []) == 3}
        self.width, self.height, self.focalPixels = width, height, focalPixels
        self.poses = [np.asarray(start, dtype=np.float64)]

    def predict(self, frame, positions):
        from meshroom.live import geotrack
        indices = [index for index in positions if index in self.points]
        # With fewer than 3 pins the pose is not determined: each pin follows its own motion
        if len(self.points) < 3 or not indices:
            return {}
        current = self.poses[-1]
        predicted = current @ np.linalg.inv(self.poses[-2]) @ current if len(self.poses) > 1 else current
        xy, depth = geotrack.projectPoints(predicted, np.array([self.points[i] for i in indices]), self.width,
                                           self.height, self.focalPixels)
        return {index: (float(x), float(y)) for index, (x, y), d in zip(indices, xy, depth)
                if d > 0 and np.isfinite(x) and np.isfinite(y)}

    def reject(self, frame, positions):
        from meshroom.live import geotrack
        indices = [index for index in positions if index in self.points]
        if len(indices) < 3:
            return set()
        points = np.array([self.points[i] for i in indices])
        found = np.array([positions[i] for i in indices])
        matrix, inliers, _ = geotrack.robustPose(points, found, self.width, self.height, self.focalPixels, self.poses[-1])
        rejected = {index for index, inlier in zip(indices, inliers) if not inlier} if len(indices) >= 5 else set()
        if len(indices) == 4:
            # 4 pins: the pose of the other 3 is always exact, the pin far from where it puts its point slipped
            # (occlusion, a similar feature nearby) when the 4 do not fit together
            matrix, errors = self._worstOfFour(points, found, matrix)
            if errors is not None:
                rejected = {indices[errors]}
        if not np.all(np.isfinite(matrix)):
            return rejected
        self.poses = [self.poses[-1], matrix]
        return rejected

    def _worstOfFour(self, points, found, matrix):
        """
        (the pose, the index of the slipped pin or None) of 4 pins: when they do not fit together, any 3 of them fit
        exactly, so the slipped one is the pin farthest from where the pose of the previous frame, moving at constant
        speed, puts it (relative to the others).
        """
        from meshroom.live import geotrack
        xy, _ = geotrack.projectPoints(matrix, points, self.width, self.height, self.focalPixels)
        if np.hypot(*(xy - found).T).max() <= 3.0:
            return matrix, None
        current = self.poses[-1]
        predicted = current @ np.linalg.inv(self.poses[-2]) @ current if len(self.poses) > 1 else current
        expected, _ = geotrack.projectPoints(predicted, points, self.width, self.height, self.focalPixels)
        deviation = np.hypot(*(found - expected).T)
        worst = int(np.argmax(deviation))
        others = np.delete(deviation, worst)
        if deviation[worst] <= 3.0 + 3.0 * float(np.median(others)):
            return matrix, None
        keep = np.arange(4) != worst
        solved, _ = scene3d.solvePose(points[keep], found[keep], self.width, self.height, self.focalPixels, matrix)
        return solved, worst


class _GeoTrackAction:
    """ The geometry tracking actions of a Resection (Track Geometry, Align, Refine), run in a worker thread. """

    def __init__(self, nodeDesc, context):
        from meshroom.live import geotrack
        self.nodeDesc, self.context = nodeDesc, context
        self.ctx = context.nodeContext()
        self.info = self.ctx.inputInfo("input")
        if self.info is None:
            raise RuntimeError("Connect the plate to the input.")
        geo = self.ctx.inputData("geo")
        if geo is None or geo.isEmpty:
            raise RuntimeError("Connect a geometry to 'Geo'.")
        self.samples = geotrack.SurfaceSamples(geo)
        from meshroom.live import gpu
        self.device = gpu.device(gpu.enabled(context))
        self.settings = geotrack.TrackSettings(context.value("trackMethod"), features=context.value("trackPoints"),
                                               anchored=context.value("trackAnchored"), device=self.device)
        self.geo = geo
        self.focalPixels = nodeDesc.startCamera(self.ctx, info=self.info).focalPixels(self.info.width)
        self.poses, self.keyframes = nodeDesc.geoTrack(self.ctx)
        self.hasMask = context.hasInput("mask")
        pins = nodeDesc.pins(self.ctx, self.info)
        self.allKeyframes = sorted(set(self.keyframes) | set(nodeDesc.pinKeyframes(pins)))
        self._lastPublish = 0.0

    def imageAt(self, frame):
        """ The plate at a frame, with its mask (geotrack.setMask) when the "mask" input is connected. """
        from meshroom.live import geotrack, ops
        image = self.context.render(frame)
        mask = None
        if self.hasMask:
            mask = ops.maskValues(self.context.renderInput("mask", frame), self.context.value("maskChannel"),
                                  self.context.value("invertMask"), image.bbox)[:, :, 0]
        # Always set: the image may come from the cache, given a mask by an earlier action
        return geotrack.setMask(image, mask)

    def poseAt(self, frame):
        """ The camera-to-world matrix of the Resection at a frame (when the action started). """
        return self.ctx.ownData("cameraOutput", frame).matrix

    def publish(self, frame=None, force=False):
        """ Show the poses tracked so far (at most 4 times a second): the viewer follows the tracking. """
        import time
        from meshroom.live import geotrack
        now = time.monotonic()
        if force or now - self._lastPublish >= 0.25:
            self._lastPublish = now
            self.context.publish({"geoTrack": geotrack.dumpPoses(self.poses, self.keyframes)}, frame)

    def result(self):
        from meshroom.live import geotrack
        return {"geoTrack": geotrack.dumpPoses(self.poses, self.keyframes)}

    def track(self, frames, progress=None):
        """ Track from the pose at frames[0] through `frames`; the poses found are stored. Returns why it stopped. """
        from meshroom.live import geotrack

        def onPose(frame, matrix, inliers, error):
            self.poses[frame] = matrix
            self.publish(frame)
        tracked, reason = geotrack.trackPose(self.samples, self.settings, frames, self.imageAt,
                                             self.poseAt(frames[0]), self.info.width, self.info.height, self.focalPixels,
                                             cancel=self.context.checkCancelled, progress=progress, onPose=onPose)
        return tracked, reason

    def deform(self, name):
        """ "Deform With Flow" forward / backward from the current frame to the end of the track range. """
        import time
        from meshroom.live import geotrack
        frame, frames = self.context.currentFrame, self.context.frames
        if frame not in frames:
            raise RuntimeError("The current frame is outside the track range (Limit Track Range).")
        sequence = [f for f in frames if f >= frame] if name == "deformForward" else [f for f in frames if f <= frame][::-1]
        if len(sequence) < 2:
            raise RuntimeError("Nothing to deform: the current frame is the end of the plate (or of the track range).")
        try:
            deform = geotrack.parseDeform(self.context.value("geoDeform"))
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        if deform and (len(deform["indices"]) == 0 or int(deform["indices"].max()) >= len(self.geo.vertices)):
            deform = {}  # Made on another geometry
        if not deform:
            indices = geotrack.deformControls(self.geo, self.context.value("deformPoints"))
            deform = {"indices": indices, "radius": geotrack.deformRadius(self.geo, indices), "frames": {}}
        lastPublish = [0.0]

        def onFrame(f, offsets):
            now = time.monotonic()
            if now - lastPublish[0] >= 0.25:
                lastPublish[0] = now
                self.context.publish({"geoDeform": geotrack.dumpDeform(deform)}, f)
        deform, reason = geotrack.trackDeformation(
            self.geo, self.settings, sequence, self.imageAt, self.poseAt, self.info.width, self.info.height,
            self.focalPixels, deform, cancel=self.context.checkCancelled, onFrame=onFrame,
            progress=lambda i, n: self.context.progress(i / float(n), "Deform With Flow"))
        result = {"geoDeform": geotrack.dumpDeform(deform)}
        if reason and reason != "cancelled":
            self.context.publish(result, None)
            raise RuntimeError(f"Stopped: {reason}. The frames deformed so far are kept.")
        return result

    def run(self, name):
        frame = self.context.currentFrame
        frames = self.context.frames
        if name in ("deformForward", "deformBackward"):
            return self.deform(name)
        if name in ("trackGeoForward", "trackGeoBackward"):
            if frame not in frames:
                raise RuntimeError("The current frame is outside the track range (Limit Track Range).")
            step = 1 if name == "trackGeoForward" else -1
            # To the next keyframe (excluded), or the end of the frame range
            ahead = [f for f in self.allKeyframes if (f - frame) * step > 0]
            end = min(ahead, key=lambda f: abs(f - frame)) - step if ahead else (frames[-1] if step > 0 else frames[0])
            sequence = list(range(frame, end + step, step))
            if len(sequence) < 2:
                raise RuntimeError("Nothing to track: the next keyframe or the end of the plate is next to this frame.")
            # The start frame is a keyframe of the track (its pose, from the pins or aligned, is the reference)
            self.poses[frame] = self.poseAt(frame)
            self.keyframes = sorted(set(self.keyframes) | {frame})
            tracked, reason = self.track(sequence, lambda i, n: self.context.progress(i / float(n), "Track Geometry"))
            if reason:
                # Lost or stopped: the poses of an earlier track after it would not join the new ones
                for other in sequence[1:]:
                    if other not in tracked and other not in self.keyframes:
                        self.poses.pop(other, None)
        elif name == "alignGeo":
            self.poses[frame] = self.align(frame)
            self.keyframes = sorted(set(self.keyframes) | {frame})
            reason = ""
        else:
            reason = self.refine()
        self.publish(force=True)
        if reason and reason != "cancelled":
            raise RuntimeError(f"Stopped: {reason}. The poses tracked so far are kept.")
        return self.result()

    def align(self, frame):
        """ The right pose at `frame` ("Align"), with the method of "Align With". """
        from meshroom.live import geotrack
        from meshroom.live.evaluator import EvaluationCancelled
        from meshroom.live.neural import NeuralUnavailable
        method = self.context.value("alignMethod")
        image, rough = self.imageAt(frame), self.poseAt(frame)
        others = [f for f in self.allKeyframes if f != frame]
        reference = min(others, key=lambda f: (abs(f - frame), f)) if others else None
        if reference is None and method not in ("megapose", "silhouette"):
            raise RuntimeError("No keyframe to align to: place the geometry with pins at another frame first "
                               "(or align with silhouette or megapose).")
        try:
            if method == "silhouette":
                self.context.progress(0.1, "Align the silhouette")
                rough, _, _ = geotrack.alignToSilhouette(self.geo, rough, image, self.info.width, self.info.height,
                                                         self.focalPixels,
                                                         maxDistance=self.context.value("silhouetteDistance"),
                                                         cancel=self.context.checkCancelled)
                if reference is None:
                    return rough
            elif method == "megapose":
                from meshroom.live import gpu
                from meshroom.live.neural import megapose
                self.context.progress(0.1, "Align with MegaPose")
                try:
                    rough = megapose.estimatePose(image, self.geo, rough, self.focalPixels, gpu.usable(self.device))
                except (NeuralUnavailable, EvaluationCancelled):
                    raise
                except Exception as e:
                    gpu.failed("MegaPose", e)
                    raise RuntimeError(f"MegaPose failed on the GPU ({gpu.status()}): {e}") from e
                if reference is None:
                    return rough
            elif method == "lightglue":
                from meshroom.live import gpu
                from meshroom.live.neural.matching import matcher
                self.context.progress(0.1, f"Align on frame {reference} with LightGlue")
                samples = geotrack.SurfaceSamples(self.geo, count=200000)
                referenceImage = self.imageAt(reference)
                # On the GPU, or on the CPU if the GPU cannot run it
                rough, _, _ = gpu.withFallback(self.device, lambda device: geotrack.alignWithMatches(
                    samples, self.poseAt(reference), referenceImage, rough, image, self.info.width, self.info.height,
                    self.focalPixels, matcher(device)), "LightGlue")
        except NeuralUnavailable as e:
            raise RuntimeError(f"Align with {method}: {e}") from e
        except geotrack.TrackingLost as e:
            raise RuntimeError(f"Cannot align with {method}: {e}.") from e
        self.context.progress(0.6, f"Align on frame {reference}")
        try:
            matrix, _, _ = geotrack.alignPose(self.samples, self.settings, self.poseAt(reference),
                                              self.imageAt(reference), rough, image, self.info.width,
                                              self.info.height, self.focalPixels, cancel=self.context.checkCancelled)
        except geotrack.TrackingLost as e:
            if method != "patches":
                # The pose of the network, not refined
                return rough
            raise RuntimeError(f"Cannot align on frame {reference}: {e}. Move the geometry closer to its place.") from e
        return matrix

    def refine(self):
        """ Each span between two keyframes tracked from both ends, the two tracks blended. Returns why it stopped. """
        from meshroom.live import geotrack
        # The spans inside the track range ("Limit Track Range")
        inRange = set(self.context.frames)
        spans = [(a, b) for a, b in zip(self.allKeyframes, self.allKeyframes[1:]) if b - a > 1 and a in inRange and b in inRange]
        if not spans:
            raise RuntimeError("Refine needs two keyframes (pins placed by hand or aligned frames) with frames between them.")
        total = sum(b - a for a, b in spans) * 2
        done = [0]
        reasons = []

        def progress(i, n):
            self.context.progress((done[0] + i) / float(total), "Refine")
        for first, last in spans:
            for frame in (first, last):
                # The pose shown at the keyframe (at a keyframe of the pins: adjusted by them)
                self.poses[frame] = self.poseAt(frame)
            forward, reason = geotrack.trackPose(self.samples, self.settings, list(range(first, last + 1)),
                                                 self.imageAt, self.poses[first], self.info.width,
                                                 self.info.height, self.focalPixels, self.context.checkCancelled,
                                                 progress)
            done[0] += last - first
            if reason == "cancelled":
                return reason
            backward, reasonBack = geotrack.trackPose(self.samples, self.settings, list(range(last, first - 1, -1)),
                                                      self.imageAt, self.poses[last], self.info.width,
                                                      self.info.height, self.focalPixels, self.context.checkCancelled,
                                                      progress)
            done[0] += last - first
            self.poses.update(geotrack.blendPoses(forward, backward, first, last))
            self.keyframes = sorted(set(self.keyframes) | {first, last})
            self.publish(first)
            if reasonBack == "cancelled":
                return reasonBack
            if reason and reasonBack:
                reasons.append(f"frames {first}-{last}: {reason}")
        return "; ".join(reasons)
