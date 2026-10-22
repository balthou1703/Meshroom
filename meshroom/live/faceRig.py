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
JAW_LIMITS = ((-0.15, 0.6), (-0.25, 0.25))
# The moves of the controls toward / away from the camera are poorly seen: they cost more
DEPTH_COST = 3.0


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


_RIGS = {}


def rigOf(rest: np.ndarray) -> FaceRig:
    """ The rig of a rest shape (cached by its values). """
    rest = np.asarray(rest, dtype=np.float64)
    key = (rest.shape, hash(rest.tobytes()))
    rig = _RIGS.get(key)
    if rig is None:
        if len(_RIGS) > 8:
            _RIGS.clear()
        rig = _RIGS[key] = FaceRig(rest)
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
           pinWeight: float = 20.0, rounds: int = 3, iterations: int = 12):
    """
    The head pose and the rig parameters putting the face on the landmarks (`pixels` (n, 2), Nuke pixels, `weights`
    (n,): 0 for a hidden landmark) seen by the camera, and on the pins ((locations (triangle, barycentric weights) on the
    mesh, their positions (k, 2)), counted `pinWeight` times). Regularized toward the rest shape (`stiffness`) and the
    `prior` parameters (`priorWeight`, e.g. the previous frame). Robust (Cauchy weights of the landmarks, again and
    again); the landmarks on the side of the face turned away from the camera (a profile: MediaPipe guesses them) are
    not used; the controls stay within what a face can do (soft limits), their moves in depth (poorly seen) cost more.
    Levenberg-Marquardt, from `cameraToFace` (the rigid pose).

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

    controls = rig.controlCount
    regularization = np.sqrt(stiffness) * 10.0
    # The cost of each parameter (the translations along the forward axis of the face cost more)
    forward = np.abs(rig.axes[2])
    costs = np.r_[np.tile(1.0 + (DEPTH_COST - 1.0) * forward, controls), np.ones(JAW_PARAMS)]
    low = np.r_[np.full(3 * controls, -TRANSLATION_LIMIT), [limit[0] for limit in JAW_LIMITS]]
    high = np.r_[np.full(3 * controls, TRANSLATION_LIMIT), [limit[1] for limit in JAW_LIMITS]]
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
            parts.append(((onFace - pinTargets) * np.sqrt(pinWeight)).ravel())
        parts.append(regularization * costs * params)
        # Soft limits: 100 pixels per unit beyond them
        beyond = np.minimum(params - low, 0.0) + np.maximum(params - high, 0.0)
        parts.append(100.0 * beyond)
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
        for axis in range(3):
            # Control c moves along the face axis `axis`: rotation[:, axis] in the camera
            dCamera[:, :, 6 + axis:6 + 3 * controls:3] = (rig.weights * rig.unit)[:, None, :] * rotation[:, axis][None, :, None]
        for j in range(JAW_PARAMS):
            step = np.zeros(len(params))
            step[3 * controls + j] = 1e-6
            dCamera[:, :, 6 + 3 * controls + j] = ((rig.deform(params + step) - vertices) / 1e-6) @ rotation.T
        # d projection / d camera point (n, 2, 3)
        dProjection = np.zeros((n, 2, 3))
        dProjection[:, 0, 0] = dProjection[:, 1, 1] = focal / z
        dProjection[:, 0, 2] = -focal * camera[:, 0] / (z * z)
        dProjection[:, 1, 2] = -focal * camera[:, 1] / (z * z)
        dProjected = np.einsum("nij,njk->nik", dProjection, dCamera)
        rows = [(dProjected[:count] * root[:, None, None]).reshape(-1, k)]
        if len(pinLocations):
            pinRows = np.array([np.einsum("j,jik->ik", w, dProjected[triangles[t]]) for t, w in pinLocations])
            rows.append((pinRows * np.sqrt(pinWeight)).reshape(-1, k))
        identity = np.zeros((len(params), k))
        identity[:, 6:] = np.eye(len(params))
        rows.append(regularization * costs[:, None] * identity)
        outside = ((params < low) | (params > high)).astype(np.float64)
        rows.append(100.0 * outside[:, None] * identity)
        if priorScale:
            rows.append(priorScale * identity)
        return residual, projected, np.concatenate(rows)

    x = np.r_[np.zeros(6), prior]
    landmarkWeights = base.copy()
    for _ in range(rounds):
        damping = 1e-3
        for _ in range(iterations):
            current, projected, jacobian = evaluate(rotation0, x, landmarkWeights, True)
            normal, gradient = jacobian.T @ jacobian, jacobian.T @ current
            improved = False
            for _ in range(6):
                try:
                    delta = -np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-9), gradient)
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
    return matrix, x[6:], rig.deform(x[6:])
