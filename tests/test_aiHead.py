""" The FaceBuilder head registered on a generated head (meshroom/live/aiHead.py), without the generator: a known
head stands for it, and a fake landmarker finds its face on the renders. """
import numpy as np

from meshroom.live import aiHead, face, headFit


def _actor(seed=1, out=1.2, back=0.8):
    """ A head of the identity model with its ears further out and back than the model has them. """
    head = face.canonicalHead()
    vertices = head.vertices
    regions = headFit.headRegions(head)
    mobility = face.headMobility(head)
    modes = headFit.identityModel()
    actor = vertices + np.tensordot(np.random.default_rng(seed).normal(0, 1.0, len(modes)), modes, axes=1)
    for side in (-1, 1):
        mask = regions["ear"] & (np.sign(vertices[:, 0]) == side)
        weight = np.exp(-((vertices - vertices[mask].mean(0)) ** 2).sum(1) / 3.5 ** 2) * mobility
        actor += weight[:, None] * np.array([side * out, 0.0, -back])[None]
    return actor


class _FakeLandmarker:
    """ Sees the face of the head it is given on the last render (aiHead._render, recorded): its 468 first vertices
    projected, when the face is turned toward the camera. """

    def __init__(self, faceVertices):
        self.faceVertices = faceVertices
        self.frame = None

    def detect(self, image):
        camera, center, rotation, offset, size = self.frame
        moved = (self.faceVertices - center) @ rotation.T + offset
        # The face toward the camera: the nose tip (1) nearer than the ears of the face (234, 454)
        if moved[1, 2] < max(moved[234, 2], moved[454, 2]) + 1e-6 * 0:
            return None, {}
        if not (moved[1, 2] > moved[234, 2] and moved[1, 2] > moved[454, 2]) or \
                moved[10, 1] < moved[152, 1]:
            return None, {}
        xy, _, _ = camera.project(moved, size, size)
        landmarks = np.c_[xy[:, 0] / size, (size - xy[:, 1]) / size, np.zeros(len(xy))]
        return np.r_[landmarks, np.zeros((10, 3))], {}


def test_registerOnAGeneratedHead(monkeypatch):
    head = face.canonicalHead()
    rest = head.vertices
    n = face.MESH_VERTICES
    actor = _actor()
    # The "generated" head: the actor somewhere else, turned (Z up), at another scale
    angle = np.radians(25.0)
    turn = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    generated = 0.031 * actor @ turn.T + np.array([0.2, -0.1, 0.05])
    triangles = head.triangles()
    colors = np.full((len(generated), 3), 0.7)
    detector = _FakeLandmarker(generated[:n])
    render = aiHead._render

    def recording(*args, **kwargs):
        image, buffer, frame = render(*args, **kwargs)
        detector.frame = frame
        return image, buffer, frame
    monkeypatch.setattr(aiHead, "_render", recording)
    landmarks = aiHead.findLandmarks(generated, triangles, colors, detector=detector, size=256)
    seen = np.isfinite(landmarks).all(1)
    assert seen.sum() > 0.8 * n
    # Lifted on its surface: near the face vertices (the size of the generated head: ~ 0.031 per cm)
    assert np.median(np.linalg.norm(landmarks[seen] - generated[:n][seen], axis=1)) < 0.031 * 0.3
    # Put on the face of the FaceBuilder (here: the actor's face, in its space)
    matrix = aiHead.alignment(landmarks, actor[:n])
    aligned = aiHead.transform(generated, matrix)
    assert np.median(np.linalg.norm(aligned - actor, axis=1)) < 0.3
    registration = aiHead.registerHead(rest, head.topology, aligned, triangles, aiHead.transform(landmarks, matrix))
    result = aiHead.registeredHead(rest, registration)
    ears = headFit.headRegions(head)["ear"]
    predicted = headFit.identityHead(rest, actor[:n], headFit.predictIdentity(actor[:n], rest))
    # (on the surface: the distance of the ears to the actor's)
    from scipy.spatial import cKDTree
    surface = cKDTree(actor)
    errorPredicted = surface.query(predicted[ears])[0].mean()
    errorRegistered = surface.query(result[ears])[0].mean()
    print(f"ears: predicted {errorPredicted:.2f} cm, registered {errorRegistered:.2f} cm")
    assert errorRegistered < 0.3 * errorPredicted and errorRegistered < 0.15
    # The face on the landmarks found
    assert np.median(np.linalg.norm(result[:n] - actor[:n], axis=1)) < 0.3


def test_readGlb(tmp_path):
    """ A binary glTF with a texture: its vertices, triangles and the texture color at each vertex. """
    import io
    import json
    import struct
    pytest = __import__("pytest")
    Image = pytest.importorskip("PIL.Image")
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    triangles = np.array([[0, 1, 2], [1, 3, 2]], np.uint16)
    uv = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]], np.float32)
    texture = np.zeros((8, 8, 3), np.uint8)
    texture[:4, :4] = (255, 0, 0)
    texture[4:, 4:] = (0, 0, 255)
    png = io.BytesIO()
    Image.fromarray(texture).save(png, format="PNG")
    parts = [vertices.tobytes(), triangles.tobytes() + b"\0\0", uv.tobytes(), png.getvalue()]
    views, offset = [], 0
    for part in parts:
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(part)})
        offset += len(part)
        offset += (-offset) % 4
    binary = b"".join(part + b"\0" * ((-len(part)) % 4) for part in parts)
    document = {
        "asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
                      {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"},
                      {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"}],
        "images": [{"bufferView": 3, "mimeType": "image/png"}], "textures": [{"source": 0}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 2}, "indices": 1, "material": 0}]}]}
    text = json.dumps(document).encode()
    text += b" " * ((-len(text)) % 4)
    glb = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(text) + 8 + len(binary)) + \
        struct.pack("<I4s", len(text), b"JSON") + text + struct.pack("<I4s", len(binary), b"BIN\0") + binary
    path = tmp_path / "head.glb"
    path.write_bytes(glb)
    readVertices, readTriangles, colors = aiHead.loadMesh(str(path))
    np.testing.assert_allclose(readVertices, vertices)
    np.testing.assert_array_equal(readTriangles, triangles)
    np.testing.assert_allclose(colors[0], (1, 0, 0))
    np.testing.assert_allclose(colors[3], (0, 0, 1))
