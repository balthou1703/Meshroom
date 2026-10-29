"""
A rig of the face (MediaPipe topology), as the head models of face trackers (KeenTools FaceTracker): the face is not
moved vertex by vertex but by a few controls placed on its features, each moving its area smoothly, so that the
rigid parts (forehead, skull, nose bridge, temples) stay rigid and the expressions stay plausible:

- the jaw: a rotation (opening, sideways) of the lower part of the face (below the lips) around a hinge near the ears;
- brows (inner, outer, each side), eyelids (upper, lower, each side), mouth corners, upper lip, lower lip, cheeks:
  a translation each (3D, in face space), of their area (gaussian falloff; the upper lip and the upper lids do not
  move the lower ones and the reverse).

`fitRig` finds the head pose and the controls putting the face on its landmarks (robust, regularized toward the rest
shape, and toward a prior: e.g. the controls of the previous frame), and on the pins (weighted more): the pins move
the controls, not the vertices.

Built from the landmark indices of the MediaPipe face mesh on any mesh with that topology (its rest shape: the neutral
face, a scan of the actor, or the face of the reference frame).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Landmarks of the MediaPipe face mesh (R / L: the right / left of the face)
LANDMARK = {
    "sideR": 234, "sideL": 454, "subnasale": 2, "forehead": 10, "chin": 152,
    "eyeOuterR": 33, "eyeInnerR": 133, "eyeInnerL": 362, "eyeOuterL": 263,
    "lidUpR": 159, "lidLowR": 145, "lidUpL": 386, "lidLowL": 374,
    "browInR": 107, "browOutR": 70, "browInL": 336, "browOutL": 300,
    "mouthR": 61, "mouthL": 291, "lipUp": 0, "lipLow": 17, "lipUpIn": 13, "lipLowIn": 14,
    "cheekR": 50, "cheekL": 280, "noseTip": 1, "nostrilR": 129, "nostrilL": 358,
}

# The translation controls: (name, center landmark, radius in face widths, area: None, "upperLip", "lowerLip",
# ("aboveEye", side) or ("belowEye", side))
CONTROLS = (
    ("browInR", "browInR", 0.08, None), ("browOutR", "browOutR", 0.08, None),
    ("browInL", "browInL", 0.08, None), ("browOutL", "browOutL", 0.08, None),
    ("lidUpR", "lidUpR", 0.045, ("aboveEye", "R")), ("lidLowR", "lidLowR", 0.045, ("belowEye", "R")),
    ("lidUpL", "lidUpL", 0.045, ("aboveEye", "L")), ("lidLowL", "lidLowL", 0.045, ("belowEye", "L")),
    ("mouthR", "mouthR", 0.09, None), ("mouthL", "mouthL", 0.09, None),
    ("lipUp", "lipUp", 0.07, "upperLip"), ("lipLow", "lipLow", 0.07, "lowerLip"),
    ("cheekR", "cheekR", 0.11, None), ("cheekL", "cheekL", 0.11, None),
    ("noseTip", "noseTip", 0.06, None), ("nostrilR", "nostrilR", 0.05, None), ("nostrilL", "nostrilL", 0.05, None),
)
# The jaw: opening, sideways (radians)
JAW_PARAMS = 2
# Soft limits of the parameters: a translation of a control beyond 2 units (a fifth of the face width), the jaw beyond
# these angles (radians), cost a lot (a face cannot do more)
TRANSLATION_LIMIT = 2.0
JAW_LIMITS = ((-0.05, 0.6), (-0.15, 0.15))
# Per control: (soft limit of its translation, in face widths; cost of moving it, times the stiffness). The nose and the
# cheeks barely move (their landmarks are often wrong on a profile: they would fold the mesh), the lips and the mouth
# corners move the most
CONTROL_LIMITS = {
    "brow": (0.08, 1.0), "lid": (0.05, 1.0), "mouth": (0.15, 1.0), "lip": (0.12, 1.0),
    "cheek": (0.05, 2.0), "nose": (0.015, 8.0), "nostril": (0.015, 8.0),
}


def controlLimit(name: str) -> Tuple[float, float]:
    """ (soft limit in face widths, cost factor) of a control by its name. """
    for prefix in ("nostril", "nose", "brow", "lid", "mouth", "lip", "cheek"):
        if name.startswith(prefix):
            return CONTROL_LIMITS[prefix]
    return (0.2, 1.0)
# The moves of the controls toward / away from the camera are poorly seen: they cost more
DEPTH_COST = 3.0
# Cost of a parameter beyond its limit, in pixels per unit
LIMIT_STRENGTH = 2000.0


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class FaceRig:
    """ The rig of a face (its rest vertices (n, 3), MediaPipe topology; its up and right directions are taken from
    the face itself: chin to forehead, one side to the other). """

    def __init__(self, rest: np.ndarray):
        rest = np.asarray(rest, dtype=np.float64)
        self.rest = rest
        point = {name: rest[index] for name, index in LANDMARK.items() if index < len(rest)}
        if len(point) < len(LANDMARK):
            raise ValueError("The face rig needs the MediaPipe face topology (468 vertices or more).")
        self.width = max(float(np.linalg.norm(point["sideR"] - point["sideL"])), 1e-6)
        # The axes of the face: right (one side to the other), up (chin to forehead, made orthogonal), forward
        right = point["sideL"] - point["sideR"]
        right /= np.linalg.norm(right)
        up = point["forehead"] - point["chin"]
        up -= right * float(up @ right)
        up /= max(np.linalg.norm(up), 1e-12)
        self.axes = np.stack([right, up, np.cross(right, up)])
        across, height = rest @ right, rest @ up
        # Below the line of the lips (through the corners and the middle of the inner lips): the jaw
        middle = 0.5 * (point["lipUpIn"] + point["lipLowIn"])
        line = sorted([(p @ right, p @ up) for p in (point["mouthR"], middle, point["mouthL"])])
        lineHeight = np.interp(across, [p[0] for p in line], [p[1] for p in line])
        self.jawWeight = _smoothstep((lineHeight - height) / (0.02 * self.width) + 0.5)
        # The hinge of the jaw: between the sides of the face, at the height of the nose base
        sides = 0.5 * (point["sideR"] + point["sideL"])
        self.pivot = sides + up * float((point["subnasale"] - sides) @ up)
        weights = []
        for _, center, radius, area in CONTROLS:
            distance = np.linalg.norm(rest - point[center], axis=1)
            weight = np.exp(-(distance / (radius * self.width)) ** 2)
            if area == "upperLip":
                weight = weight * (1.0 - self.jawWeight)
            elif area == "lowerLip":
                weight = weight * self.jawWeight
            elif isinstance(area, tuple):
                side = area[1]
                eyeHeight = 0.5 * float((point[f"eyeOuter{side}"] + point[f"eyeInner{side}"]) @ up)
                above = _smoothstep((height - eyeHeight) / (0.01 * self.width) + 0.5)
                weight = weight * (above if area[0] == "aboveEye" else 1.0 - above)
            weights.append(weight)
        # (n, controls): the share of each control in the move of each vertex
        self.weights = np.stack(weights, axis=1)
        # How much each vertex is moved by the rig (0: a rigid part), for the local warp of the pins
        self.mobility = np.clip(np.maximum(self.weights.max(axis=1), self.jawWeight), 0.0, 1.0)
        self.controlCount = len(CONTROLS)
        # A parameter of 1 moves its area by a tenth of the face width (the parameters have similar scales)
        self.unit = 0.1 * self.width

    @property
    def paramCount(self) -> int:
        return 3 * self.controlCount + JAW_PARAMS

    def jawRotation(self, params: np.ndarray) -> np.ndarray:
        """ The rotation of the jaw: opening (around the right axis of the face), then sideways (around its up axis). """
        from .camera import rodrigues
        return rodrigues(self.axes[0] * params[-2]) @ rodrigues(self.axes[1] * params[-1])

    def deform(self, params: np.ndarray) -> np.ndarray:
        """ The vertices (n, 3) of the face for rig parameters (paramCount,): the jaw, then the controls. """
        params = np.asarray(params, dtype=np.float64)
        jawed = self.pivot + (self.rest - self.pivot) @ self.jawRotation(params).T
        vertices = self.rest + self.jawWeight[:, None] * (jawed - self.rest)
        translations = params[:3 * self.controlCount].reshape(self.controlCount, 3) * self.unit
        return vertices + self.weights @ translations

    @property
    def areas(self) -> np.ndarray:
        """ (n, controls + 1): how much each control (and the jaw) moves each vertex, in [0, 1]. """
        return np.c_[self.weights, self.jawWeight]

    def limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """ (low, high) soft limits of the parameters: each control its own (in parameter units: a tenth of the face
        width), the jaw its angles. """
        limits = np.array([controlLimit(name) for name, *_ in CONTROLS[:self.controlCount]])
        perControl = np.repeat(limits[:, 0] / 0.1, 3)
        return (np.r_[-perControl, [limit[0] for limit in JAW_LIMITS]],
                np.r_[perControl, [limit[1] for limit in JAW_LIMITS]])

    def costs(self) -> np.ndarray:
        """ The cost of each parameter (times the stiffness): its control's, the moves toward the camera more. """
        limits = np.array([controlLimit(name) for name, *_ in CONTROLS[:self.controlCount]])
        forward = np.abs(self.axes[2])
        return np.r_[np.tile(1.0 + (DEPTH_COST - 1.0) * forward, self.controlCount) * np.repeat(np.sqrt(limits[:, 1]), 3),
                     np.ones(JAW_PARAMS)]

    def vertexJacobian(self, params: np.ndarray) -> np.ndarray:
        """ d vertices / d parameters (n, 3, paramCount) in face space. """
        params = np.asarray(params, dtype=np.float64)
        n, controls = len(self.rest), self.controlCount
        jacobian = np.zeros((n, 3, self.paramCount))
        for axis in range(3):
            # Control c moves along the face axis `axis`
            jacobian[:, axis, axis:3 * controls:3] = self.weights * self.unit
        vertices = self.deform(params)
        for j in range(JAW_PARAMS):
            step = np.zeros(len(params))
            step[3 * controls + j] = 1e-6
            jacobian[:, :, 3 * controls + j] = (self.deform(params + step) - vertices) / 1e-6
        return jacobian


# The slits of the MediaPipe face (its triangles cross them, from one lid or lip to the other): (lower, upper) contours
SLITS = (([33, 7, 163, 144, 145, 153, 154, 155, 133], [33, 246, 161, 160, 159, 158, 157, 173, 133]),
         ([263, 249, 390, 373, 374, 380, 381, 382, 362], [263, 466, 388, 387, 386, 385, 384, 398, 362]),
         ([78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308], [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]))


def acrossSlits(triangles: np.ndarray) -> np.ndarray:
    """ Whether each triangle crosses an eye or the mouth (a corner on each lid or lip): closing them turns it. """
    across = np.zeros(len(triangles), dtype=bool)
    for lower, upper in SLITS:
        onLower = np.isin(triangles, lower[1:-1]).any(axis=1)
        onUpper = np.isin(triangles, upper[1:-1]).any(axis=1)
        corners = np.isin(triangles, [lower[0], lower[-1]]).any(axis=1)
        across |= (onLower & onUpper) | (corners & (onLower | onUpper) & np.isin(triangles, lower + upper).all(axis=1))
    return across


# Within this move (centimeters on the canonical head), a vertex moved by an expression shape counts as fully mobile
BLENDSHAPE_MOBILE_MOVE = 0.5


class BlendshapeRig:
    """
    The ARKit rig of the face (data/faceExpressions.npz: 51 expression shapes of the head of the FaceTracker,
    transferred from ICT-FaceKit): the face is its rest shape plus the moves of the shapes, each weighted in [0, 1]
    (eyeBlinkLeft, jawOpen, mouthSmileRight...: the names of ARKit, of MediaPipe, of the face rigs of Maya, Unreal and
    Blender add-ons). The moves are made for the canonical head: on another rest shape (the actor's), they are scaled
    by the width of its face. Same interface as FaceRig (fitRig).
    """

    # The shapes are made by hand: some triangles of the lips nearly fold in a wide open mouth, which is fine; only the
    # triangles turned over (more than ~105 degrees) tone the expression down
    foldLimit = -0.25
    # The eyes and the mouth close (the triangles across them are not checked)
    slitsClose = True

    def __init__(self, rest: np.ndarray, head: bool = False):
        rest = np.asarray(rest, dtype=np.float64)
        self.rest = rest
        names, deltas = expressionShapes()
        self.names = list(names)
        count = len(rest)
        faceCount = 468
        if count < faceCount or (head and count != deltas.shape[1]):
            raise ValueError("The ARKit face rig needs the MediaPipe face topology (or the head of the FaceTracker).")
        moves = np.zeros((len(names), count, 3))
        if head:
            moves[:] = deltas
        else:
            # A face mesh (468, or 478 with the irises: they do not move)
            moves[:, :faceCount] = deltas[:, :faceCount]
        canonical = _canonicalWidth(deltas)
        width = float(np.linalg.norm(rest[LANDMARK["sideR"]] - rest[LANDMARK["sideL"]]))
        self.scale = width / canonical if canonical > 0 else 1.0
        self.width = width
        # (n, 3, shapes)
        self._jacobian = np.ascontiguousarray(np.transpose(moves * self.scale, (1, 2, 0)))
        self._flat = self._jacobian.reshape(-1, len(names))
        lengths = np.linalg.norm(moves, axis=2)
        self._areas = (lengths / np.maximum(lengths.max(axis=1, keepdims=True), 1e-9)).T
        self.mobility = np.clip(lengths.max(axis=0) / BLENDSHAPE_MOBILE_MOVE, 0.0, 1.0)
        # Shapes moving mostly in depth (toward the camera: puffs, puckers, rolls) are poorly seen: they cost more
        depth = np.sqrt((moves[:, :, 2] ** 2).sum(1) / np.maximum((lengths ** 2).sum(1), 1e-12))
        self._costs = 1.0 + (DEPTH_COST - 1.0) * depth

    @property
    def paramCount(self) -> int:
        return len(self.names)

    @property
    def areas(self) -> np.ndarray:
        """ (n, shapes): how much each shape moves each vertex, in [0, 1] of its largest move. """
        return self._areas

    def deform(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        return self.rest + (self._flat @ params).reshape(-1, 3)

    def limits(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.paramCount), np.ones(self.paramCount)

    def costs(self) -> np.ndarray:
        return self._costs

    def vertexJacobian(self, params: np.ndarray) -> np.ndarray:
        return self._jacobian

    def fromMediaPipe(self, blendshapes: dict) -> Optional[np.ndarray]:
        """
        The parameters MediaPipe sees (its blendshape scores, ARKit names), None without any. The two sides are
        averaged (left and right of MediaPipe are not reliably those of the face): the landmarks tell them apart.
        """
        if not blendshapes:
            return None
        params = np.array([float(blendshapes.get(name, 0.0)) for name in self.names])
        for k, name in enumerate(self.names):
            for side, other in (("Left", "Right"), ("Right", "Left")):
                if name.endswith(side):
                    mirrored = name[:-len(side)] + other
                    if mirrored in blendshapes:
                        params[k] = 0.5 * (float(blendshapes.get(name, 0.0)) + float(blendshapes[mirrored]))
        return np.clip(params, 0.0, 1.0)


_EXPRESSIONS = {}


def expressionShapes() -> Tuple[np.ndarray, np.ndarray]:
    """ (names (51,), moves (51, head vertices, 3) in centimeters) of the ARKit shapes of the canonical head. """
    if not _EXPRESSIONS:
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "faceExpressions.npz")
        with np.load(path) as data:
            _EXPRESSIONS["names"] = np.array(data["names"]).astype(str)
            _EXPRESSIONS["deltas"] = np.array(data["deltas"], dtype=np.float64)
    return _EXPRESSIONS["names"], _EXPRESSIONS["deltas"]


def _canonicalWidth(deltas: np.ndarray) -> float:
    """ The width of the canonical face (sideR to sideL) the shapes were made on. """
    if "width" not in _EXPRESSIONS:
        from . import face
        vertices = face.canonicalHead().vertices
        _EXPRESSIONS["width"] = float(np.linalg.norm(vertices[LANDMARK["sideR"]] - vertices[LANDMARK["sideL"]]))
    return _EXPRESSIONS["width"]


# The rig models: "rig" (FaceRig: controls placed on the features), "arkit" (BlendshapeRig: the ARKit shapes)
RIG_MODELS = ("rig", "arkit")


_RIGS = {}


def rigOf(rest: np.ndarray, model: str = "rig", head: bool = False):
    """ The rig of a rest shape (cached by its values): a FaceRig ("rig"), or a BlendshapeRig ("arkit"; `head`: the
    rest shape is the head of the FaceTracker, not only the face). """
    rest = np.asarray(rest, dtype=np.float64)
    key = (model, head, rest.shape, hash(rest.tobytes()))
    rig = _RIGS.get(key)
    if rig is None:
        if len(_RIGS) > 8:
            _RIGS.clear()
        rig = _RIGS[key] = BlendshapeRig(rest, head) if model == "arkit" else FaceRig(rest)
    return rig


def _vertexNormals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    faceNormals = np.cross(b - a, c - a)
    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, triangles[:, i], faceNormals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def fitRig(rig: FaceRig, pixels: np.ndarray, weights: np.ndarray, width: int, height: int, focal: float,
           cameraToFace: np.ndarray, pins: Optional[Tuple[list, np.ndarray]] = None, triangles=None,
           prior: Optional[np.ndarray] = None, stiffness: float = 0.3, priorWeight: float = 0.0,
           pinWeight=20.0, rounds: int = 3, iterations: int = 12, solve: str = "all"):
    """
    The head pose and the rig parameters putting the face on the landmarks (`pixels` (n, 2), Nuke pixels, `weights`
    (n,): 0 for a hidden landmark) seen by the camera, and on the pins ((locations (triangle, barycentric weights) on the
    mesh, their positions (k, 2)), counted `pinWeight` times). Regularized toward the rest shape (`stiffness`) and the
    `prior` parameters (`priorWeight`, e.g. the previous frame). Robust (Cauchy weights of the landmarks, again and
    again); the landmarks on the side of the face turned away from the camera (a profile: MediaPipe guesses them) are
    not used; the controls stay within what a face can do (soft limits), their moves in depth (poorly seen) cost more.
    Levenberg-Marquardt, from `cameraToFace` (the rigid pose). `pinWeight`: a number, or one per pin. `solve`: "all"
    (the pose and the rig together), "pose" (the head pose only, the rig kept at `prior`), "rig" (the rig only, the
    pose kept).

    Returns:
        (camera-to-face matrix, rig parameters, the face vertices (n, 3) in face space).
    """
    from . import scene3d
    from .camera import rodrigues
    count = min(len(pixels), len(rig.rest))
    observed = np.c_[pixels[:count, 0] - width / 2.0, -(pixels[:count, 1] - height / 2.0)]
    base = np.clip(np.asarray(weights[:count], dtype=np.float64), 0.0, None)
    rotation0, translation0 = scene3d.cvPose(scene3d.toMatrix4(cameraToFace))
    prior = np.zeros(rig.paramCount) if prior is None else np.asarray(prior, dtype=np.float64)
    pinLocations, pinTargets = (pins if pins is not None and len(pins[0]) else ([], np.zeros((0, 2))))
    pinTargets = np.c_[pinTargets[:, 0] - width / 2.0, -(pinTargets[:, 1] - height / 2.0)] if len(pinLocations) else pinTargets
    pinRoots = np.sqrt(np.clip(np.broadcast_to(np.asarray(pinWeight, dtype=np.float64), (len(pinLocations),)), 0.0, None))

    regularization = np.sqrt(stiffness) * 10.0
    # The cost of each parameter (e.g. the moves toward the camera, poorly seen, cost more) and its limits
    costs = rig.costs()
    low, high = rig.limits()
    normals = _vertexNormals(rig.rest, triangles) if triangles is not None else None
    priorScale = np.sqrt(priorWeight) * 10.0 if priorWeight > 0 else 0.0

    def evaluate(rotation, x, landmarkWeights, withJacobian):
        """ The residuals (and their Jacobian) of the parameters x = (rotation increment 3, translation 3, rig) with
        the rotation `rotation` (the increment folded into it at each iteration: its derivative is exact at 0). """
        params = x[6:]
        vertices = rig.deform(params)
        rotated = vertices @ rotation.T
        camera = rotated + translation0 + x[3:6]
        z = np.where(camera[:, 2] > 1e-9, camera[:, 2], 1e-9)
        projected = focal * camera[:, :2] / z[:, None]
        root = np.sqrt(landmarkWeights)
        parts = [((projected[:count] - observed) * root[:, None]).ravel()]
        pinRows = None
        if len(pinLocations):
            onFace = np.array([projected[triangles[t]].T @ w for t, w in pinLocations])
            parts.append(((onFace - pinTargets) * pinRoots[:, None]).ravel())
        parts.append(regularization * costs * params)
        # Limits: 2000 pixels per unit beyond them (stronger than the pins: a face cannot do more)
        beyond = np.minimum(params - low, 0.0) + np.maximum(params - high, 0.0)
        parts.append(LIMIT_STRENGTH * beyond)
        if priorScale:
            parts.append(priorScale * (params - prior))
        residual = np.concatenate(parts)
        if not withJacobian:
            return residual, projected, None
        n, k = len(vertices), len(x)
        # d camera point / d parameters (n, 3, k)
        dCamera = np.zeros((n, 3, k))
        px, py, pz = rotated[:, 0], rotated[:, 1], rotated[:, 2]
        dCamera[:, 1, 0], dCamera[:, 2, 0] = -pz, py
        dCamera[:, 0, 1], dCamera[:, 2, 1] = pz, -px
        dCamera[:, 0, 2], dCamera[:, 1, 2] = -py, px
        dCamera[:, 0, 3] = dCamera[:, 1, 4] = dCamera[:, 2, 5] = 1.0
        # The moves of the rig (face space) in the camera
        dCamera[:, :, 6:] = np.einsum("ij,njk->nik", rotation, rig.vertexJacobian(params))
        # d projection / d camera point (n, 2, 3)
        dProjection = np.zeros((n, 2, 3))
        dProjection[:, 0, 0] = dProjection[:, 1, 1] = focal / z
        dProjection[:, 0, 2] = -focal * camera[:, 0] / (z * z)
        dProjection[:, 1, 2] = -focal * camera[:, 1] / (z * z)
        dProjected = np.einsum("nij,njk->nik", dProjection, dCamera)
        rows = [(dProjected[:count] * root[:, None, None]).reshape(-1, k)]
        if len(pinLocations):
            pinRows = np.array([np.einsum("j,jik->ik", w, dProjected[triangles[t]]) for t, w in pinLocations])
            rows.append((pinRows * pinRoots[:, None, None]).reshape(-1, k))
        identity = np.zeros((len(params), k))
        identity[:, 6:] = np.eye(len(params))
        rows.append(regularization * costs[:, None] * identity)
        outside = ((params < low) | (params > high)).astype(np.float64)
        rows.append(LIMIT_STRENGTH * outside[:, None] * identity)
        if priorScale:
            rows.append(priorScale * identity)
        return residual, projected, np.concatenate(rows)

    x = np.r_[np.zeros(6), prior]
    # The parameters solved (the others kept)
    active = np.ones(len(x), dtype=bool)
    if solve == "pose":
        active[6:] = False
    elif solve == "rig":
        active[:6] = False
    landmarkWeights = base.copy()
    for _ in range(rounds):
        damping = 1e-3
        for _ in range(iterations):
            current, projected, jacobian = evaluate(rotation0, x, landmarkWeights, True)
            normal, gradient = jacobian.T @ jacobian, jacobian.T @ current
            improved = False
            # Up to a strong damping: a step across a limit (stiff) is retried shorter
            for _ in range(12):
                try:
                    reduced = normal[active][:, active]
                    delta = np.zeros(len(x))
                    delta[active] = -np.linalg.solve(reduced + damping * np.diag(np.diag(reduced) + 1e-9), gradient[active])
                except np.linalg.LinAlgError:
                    break
                candidate = x + delta
                candidateRotation = rodrigues(candidate[:3]) @ rotation0
                candidate[:3] = 0.0
                residual, _, _ = evaluate(candidateRotation, candidate, landmarkWeights, False)
                if residual @ residual < current @ current:
                    x, rotation0 = candidate, candidateRotation
                    damping = max(damping / 3.0, 1e-9)
                    improved = True
                    break
                damping *= 4.0
            if not improved or np.linalg.norm(delta) < 1e-8:
                break
        _, projected, _ = evaluate(rotation0, x, landmarkWeights, False)
        # The landmarks far from the fit of the others weighted down (Cauchy, 3 typical errors, at least 2 pixels)
        errors = np.hypot(*(projected[:count] - observed).T)
        measured = base > 0
        scale = max(2.0, 1.4826 * float(np.median(errors[measured]))) if measured.any() else 2.0
        landmarkWeights = base / (1.0 + (errors / (3.0 * scale)) ** 2)
        if normals is not None:
            # The side of the face turned away from the camera: its landmarks are guesses
            rotation = rodrigues(x[:3]) @ rotation0
            points = rig.deform(x[6:]) @ rotation.T + translation0 + x[3:6]
            facing = -np.einsum("ij,ij->i", (normals @ rotation.T), points / np.linalg.norm(points, axis=1, keepdims=True))
            landmarkWeights = landmarkWeights * np.clip((facing[:count] + 0.05) / 0.2, 0.0, 1.0)
    rotation = rodrigues(x[:3]) @ rotation0
    matrix = scene3d.cameraMatrixFromCvPose(rotation, translation0 + x[3:6])
    # Never beyond the limits, never folded (controls pulled against each other: the expression toned down until no
    # triangle turns over)
    params = np.clip(x[6:], low, high)
    if triangles is not None:
        # The triangles across the eyes and the mouth turn when they close (BlendshapeRig: blinks, mouthClose)
        checked = triangles[~acrossSlits(triangles)] if getattr(rig, "slitsClose", False) else triangles
        params = params * unfoldedFraction(lambda t: rig.deform(params * t), rig.rest, checked,
                                           limit=getattr(rig, "foldLimit", 0.15))
    return matrix, params, rig.deform(params)


def triangleNormals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    normals = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]], vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def unfoldedFraction(deformed, rest: np.ndarray, triangles: np.ndarray, steps: int = 8, limit: float = 0.15) -> float:
    """
    The largest fraction t in [0, 1] of a deformation (`deformed(t)`: the vertices) for which no triangle turns over
    (or nearly: the cosine of its normal with its `rest` one below `limit`, 0.15: more than ~80 degrees), by bisection;
    1 when the whole deformation is fine. The triangles already turned at t = 0 are left out, and the degenerate ones
    (nearly no area: e.g. along the face oval of a head), whose normal turns for any move.
    """
    rest = np.asarray(rest, dtype=np.float64)
    reference = triangleNormals(rest, triangles)
    corners = rest[triangles]
    areas = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    # The triangles already turned at t = 0 (e.g. by another step) are not counted
    counted = ((triangleNormals(deformed(0.0), triangles) * reference).sum(1) >= limit) & \
        (areas > 1e-3 * max(float(np.median(areas)), 1e-30))

    def folded(t):
        # Turned over, or nearly (more than ~80 degrees from its rest orientation)
        return bool((((triangleNormals(deformed(t), triangles) * reference).sum(1) < limit) & counted).any())
    if not folded(1.0):
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(steps):
        middle = 0.5 * (low + high)
        if folded(middle):
            high = middle
        else:
            low = middle
    return low
