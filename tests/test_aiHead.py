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
    assert errorRegistered < 0.5 * errorPredicted and errorRegistered < 0.25
    # The face on the landmarks found
    assert np.median(np.linalg.norm(result[:n] - actor[:n], axis=1)) < 0.3
    # The base of the neck keeps its shape: it only follows the face
    neck = headFit.headRegions(head)["neck"]
    base = neck & (rest[:, 1] < rest[neck, 1].min() + 0.3)
    followed = face.HeadFollow.of(rest).follow(result[:n])
    assert np.linalg.norm(result[base] - followed[base], axis=1).max() < 0.1
    # With the face fitted on the photos: that face, exactly
    photoFace = actor[:n] + 0.05
    registration = aiHead.registerHead(rest, head.topology, aligned, triangles, aiHead.transform(landmarks, matrix),
                                       faceTarget=photoFace)
    np.testing.assert_allclose(aiHead.registeredHead(rest, registration)[:n], photoFace, atol=1e-6)


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


def test_generateRunsItsRunner(tmp_path, monkeypatch):
    """ aiHead.generate: the image written, the runner run in another process (Meshroom's Python for TripoSR), its
    output read back; its failure reported with the end of its log. """
    runner = tmp_path / "fakeRunner.py"
    runner.write_text(
        "import sys\n"
        "import numpy as np\n"
        "image, output = sys.argv[1], sys.argv[2]\n"
        "assert sys.argv[3] == '--root'\n"
        "print('[fake] generating', flush=True)\n"
        "if 'fail' in output:\n"
        "    print('[fake] out of memory', flush=True)\n"
        "    sys.exit(3)\n"
        "np.savez(output, vertices=np.zeros((3, 3), np.float32), faces=np.array([[0, 1, 2]], np.int32))\n")
    monkeypatch.setattr(aiHead, "runnerPath", lambda model="trellis2": str(runner))
    monkeypatch.setenv("MESHROOM_TRIPOSR_ROOT", str(tmp_path))
    rgba = np.zeros((16, 16, 4))
    rgba[4:12, 4:12] = (0.8, 0.5, 0.4, 1.0)
    lines = []
    output = aiHead.generate(rgba, str(tmp_path / "head.npz"), model="triposr", log=lines.append)
    assert (tmp_path / "head_input.png").exists() and lines == ["[fake] generating"]
    vertices, triangles, colors = aiHead.loadMesh(output)
    assert vertices.shape == (3, 3) and triangles.shape == (1, 3) and colors.shape == (3, 3)
    try:
        aiHead.generate(rgba, str(tmp_path / "fail.npz"), model="triposr")
        raise AssertionError("no error")
    except aiHead.AIHeadUnavailable as e:
        assert "out of memory" in str(e) and "TripoSR" in str(e)
    # TripoSR needs no clone (its code is in Meshroom); TRELLIS.2 needs its environment
    monkeypatch.delenv("MESHROOM_TRIPOSR_ROOT")
    runner.write_text(runner.read_text().replace("assert sys.argv[3] == '--root'", "assert len(sys.argv) == 3"))
    aiHead.generate(rgba, str(tmp_path / "other.npz"), model="triposr")
    monkeypatch.delenv("MESHROOM_TRELLIS2_PYTHON", raising=False)
    try:
        aiHead.generate(rgba, str(tmp_path / "trellis.npz"), model="trellis2")
        raise AssertionError("no error")
    except aiHead.AIHeadUnavailable as e:
        assert "MESHROOM_TRELLIS2_PYTHON" in str(e)


def test_earsUnderTheHairOfTheGeneratedHeadStay():
    """ A generated head whose ears are under a mass of hair (its surface 2.5 cm out around them): the ears are not
    pulled into it, the skull stays the one the face gives. """
    head = face.canonicalHead()
    rest = head.vertices
    n = face.MESH_VERTICES
    actor = _actor(4)
    regions = headFit.headRegions(head)
    normals = aiHead.vertexNormals(actor, head.triangles())
    around = np.zeros(len(actor))
    for side in (-1, 1):
        center = actor[regions["ear"] & (np.sign(rest[:, 0]) == side)].mean(0)
        around = np.maximum(around, np.exp(-((actor - center) ** 2).sum(1) / 4.0 ** 2))
    generated = actor + 2.5 * around[:, None] * normals
    registration = aiHead.registerHead(rest, head.topology, generated, head.triangles(), actor[:n], faceTarget=actor[:n])
    result = aiHead.registeredHead(rest, registration)
    predicted = headFit.identityHead(rest, actor[:n], headFit.predictIdentity(actor[:n], rest))
    moved = np.linalg.norm(result - predicted, axis=1)
    assert moved[regions["ear"]].max() < 0.5 and moved[regions["skull"]].max() < 0.5


def test_warpOntoTheGeneratedHead():
    """ The head warped onto the generated surface (non-rigid ICP): its face and ears on it, "AI Face" dosing the face,
    the base of the neck held, the warp smooth. """
    from scipy.spatial import cKDTree
    head = face.canonicalHead()
    rest = head.vertices
    n = face.MESH_VERTICES
    triangles = head.triangles()
    actor = _actor(5)
    # The face fitted on the photos: the actor's, a bit wider (what the generated head corrects)
    photoFace = actor[:n] * np.array([1.06, 1.0, 1.0])
    registration = aiHead.registerHead(rest, head.topology, actor, triangles, actor[:n], faceTarget=photoFace)
    start = headFit.straightNeck(aiHead.registeredHead(rest, registration))
    data, fixed = aiHead.warpWeights(rest)
    moves = aiHead.warpOntoSurface(start, triangles, actor, triangles, data, fixed)
    registration["warp"] = {int(v): list(moves[v]) for v in range(len(rest))}
    surface = cKDTree(actor)
    regions = headFit.headRegions(head)
    full = aiHead.warpedHead(start, registration, aiFace=1.0)
    none = aiHead.warpedHead(start, registration, aiFace=0.0)
    actorNormals = aiHead.vertexNormals(actor, triangles)

    def toSurface(points):
        # Point to plane: the distance to the tangent plane of the nearest point
        _, index = surface.query(points)
        return np.abs(((points - actor[index]) * actorNormals[index]).sum(1))
    print(f"face to the generated surface: {toSurface(start[:n]).mean():.3f} -> {toSurface(full[:n]).mean():.3f} cm")
    assert toSurface(full[:n]).mean() < 0.4 * toSurface(start[:n]).mean()
    assert surface.query(full[regions["ear"]])[0].mean() < 0.15
    np.testing.assert_allclose(none[:n][face.headMobility(head)[:n] == 0], photoFace[face.headMobility(head)[:n] == 0],
                               atol=1e-9)
    neck = regions["neck"]
    base = neck & (rest[:, 1] < rest[neck, 1].min() + 0.3)
    assert np.linalg.norm(moves[base], axis=1).max() < 0.05
    # Smooth: no triangle turned over

    def normals(points):
        a, b, c = (points[triangles[:, i]] for i in range(3))
        return np.cross(b - a, c - a)
    assert ((normals(full) * normals(start)).sum(1) > 0).mean() > 0.995


def test_arcfaceCrop():
    """ The crop ArcFace expects: the 5 points of the face on its template. """
    image = np.zeros((300, 400, 3))
    keypoints = np.array([[150.0, 120.0], [230.0, 118.0], [190.0, 165.0], [158.0, 205.0], [224.0, 204.0]])
    for x, y in keypoints:
        image[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3] = 1.0
    crop = aiHead.arcfaceCrop(image, keypoints)
    assert crop.shape == (112, 112, 3) and crop.dtype == np.uint8
    for x, y in aiHead.ARCFACE_TEMPLATE:
        assert crop[int(round(y)), int(round(x))].max() > 100
    # MediaPipe landmarks: eyes, nose, mouth corners, image left first
    landmarks = np.zeros((478, 2))
    landmarks[473], landmarks[468], landmarks[1], landmarks[291], landmarks[61] = keypoints
    np.testing.assert_allclose(aiHead.arcfaceKeypoints(landmarks), keypoints)


def test_flameOfAChumpyPickle(monkeypatch, tmp_path):
    """ The FLAME pickle (chumpy arrays) read without chumpy. """
    import pickle
    import sys
    import types
    from meshroom.live.neural import micaRunner
    module = types.ModuleType("chumpy.ch")

    class Ch:
        def __init__(self, x):
            self.x = x
    Ch.__module__, Ch.__qualname__ = "chumpy.ch", "Ch"
    module.Ch = Ch
    monkeypatch.setitem(sys.modules, "chumpy", types.ModuleType("chumpy"))
    monkeypatch.setitem(sys.modules, "chumpy.ch", module)
    template = np.arange(12.0).reshape(4, 3)
    shapedirs = np.ones((4, 3, 400))
    path = tmp_path / "generic_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"v_template": Ch(template), "shapedirs": Ch(shapedirs), "f": np.array([[0, 1, 2], [1, 2, 3]])}, f)
    monkeypatch.delitem(sys.modules, "chumpy.ch")
    monkeypatch.delitem(sys.modules, "chumpy")
    loaded, dirs, faces = micaRunner.loadFlame(str(path))
    np.testing.assert_array_equal(loaded, template)
    assert dirs.shape == (4, 3, 300) and faces.shape == (2, 3)


def test_micaNetworks():
    """ The networks of MICA rebuilt: the module names of its weights, the shapes of its outputs. """
    import pytest
    torch = pytest.importorskip("torch")
    from meshroom.live.neural import micaRunner
    arcface = micaRunner._iresnet100(torch).eval()
    names = set(arcface.state_dict())
    assert {"conv1.weight", "layer1.0.downsample.0.weight", "layer3.29.conv2.weight", "fc.weight",
            "features.running_mean"} <= names
    with torch.no_grad():
        assert arcface(torch.zeros(1, 3, 112, 112)).shape == (1, 512)
    mapping = micaRunner._mapping(torch)
    assert set(mapping.state_dict()) >= {"network.0.weight", "network.3.bias", "output.weight"}
    assert mapping(torch.zeros(1, 512)).shape == (1, 300)


def test_mediapipeLandmarksOnFlame(tmp_path):
    """ The MediaPipe landmarks of a FLAME head from its embedding (faces, barycentric coordinates, indices). """
    from meshroom.live.neural import micaRunner
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float64)
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    path = tmp_path / "mediapipe_landmark_embedding.npz"
    np.savez(path, lmk_face_idx=np.array([0, 1]), lmk_b_coords=np.array([[1, 0, 0], [0.2, 0.3, 0.5]]),
             landmark_indices=np.array([1, 33]))
    landmarks = micaRunner.mediapipeLandmarks(vertices, faces, str(path))
    np.testing.assert_allclose(landmarks[1], (0, 0, 0))
    np.testing.assert_allclose(landmarks[33], 0.2 * vertices[1] + 0.3 * vertices[3] + 0.5 * vertices[2])
    assert np.isnan(landmarks[0]).all() and micaRunner.mediapipeLandmarks(vertices, faces, str(tmp_path / "no")) is None
    np.savez(tmp_path / "head.npz", vertices=vertices, faces=faces, landmarks=landmarks)
    np.testing.assert_allclose(aiHead.storedLandmarks(str(tmp_path / "head.npz"))[33], landmarks[33])


def _fakeFlame(seed=0):
    """ A FLAME stand-in: the canonical head in meters, the identity modes as its directions, MediaPipe landmarks on
    some of its vertices (the embedding: a triangle with the vertex as its first corner). """
    from meshroom.live import flameFit
    head = face.canonicalHead()
    triangles = head.triangles()
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(face.MESH_VERTICES, 105, replace=False))
    faces, weights = [], []
    for index in indices:
        triangle = int(np.where((triangles == index).any(1))[0][0])
        corner = int(np.where(triangles[triangle] == index)[0][0])
        w = np.zeros(3)
        w[corner] = 1.0
        faces.append(triangle)
        weights.append(w)
    modes = headFit.identityModel()
    return flameFit.FlameModel(head.vertices / 100.0, np.moveaxis(modes, 0, 2) / 100.0, triangles,
                               (np.array(faces), np.array(weights), indices))


def test_flameFittedOnTheFace():
    """ FLAME (a stand-in of the same statistics) fitted on a face of the FaceBuilder, turned, moved and scaled: the
    whole head found, in the space of the face. """
    from meshroom.live import flameFit
    model = _fakeFlame()
    n = face.MESH_VERTICES
    rest = face.canonicalHead().vertices
    actor = _actor(6, out=0.0, back=0.0)
    angle = np.radians(10.0)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    placed = 1.1 * actor @ rotation.T + np.array([1.0, -2.0, 0.5])
    fit = flameFit.fitFlame(model, placed[:n])
    regions = headFit.headRegions(face.canonicalHead())
    head = regions["ear"] | regions["skull"]
    error = np.linalg.norm(fit["vertices"] - placed, axis=1)
    predicted = headFit.identityHead(rest, actor[:n], headFit.predictIdentity(actor[:n], rest))
    errorPredicted = 1.1 * np.linalg.norm(predicted - actor, axis=1)
    print(f"FLAME fit: face {error[:n].mean():.3f} cm, head {error[head].mean():.3f} cm "
          f"(identity from the face: {errorPredicted[head].mean():.3f})")
    assert error[:n].mean() < 0.15 and error[head].mean() < 1.2 * errorPredicted[head].mean() + 0.05
    seen = np.isfinite(fit["landmarks"]).all(1)
    assert seen.sum() == 105
    np.testing.assert_allclose(fit["landmarks"][seen], fit["vertices"][np.where(seen)[0]], atol=1e-9)


def test_flameFittedOnTheViews():
    """ FLAME fitted on its own on the landmarks of the views (their reprojection), from its fit on a face that is a
    bit off and with cameras a bit off (a view turned and moved): its landmarks then land on their pixels in every view,
    the cameras corrected by its moves in the views. """
    from tests.test_faceBuilder import _headViews
    from meshroom.live import flameFit, scene3d
    model = _fakeFlame()
    n = face.MESH_VERTICES
    actor = _actor(7, out=0.0, back=0.0)
    size = (640, 480)
    views = []
    for index, camera in enumerate(_headViews((0, 50, -50), size)):
        xy, _, _ = camera.project(actor[:n], *size)
        if index == 1:
            # (the cameras estimated for the views: off by up to 10 degrees and 1.5 cm)
            camera = camera.copy(matrix=scene3d.composeTRS((1.5, 0.5, 0), (8.0, -6.0, 0)) @ camera.matrix)
        elif index == 2:
            camera = camera.copy(matrix=scene3d.composeTRS((-1.0, 0, 0.5), (-5.0, 4.0, 0)) @ camera.matrix)
        views.append({"camera": camera, "size": size, "pixels": xy, "hidden": []})
    off = actor[:n] + 0.3 * np.sin(actor[:n] / 3.0)
    start = flameFit.fitFlame(model, off)
    fit = flameFit.fitFlameToViews(model, views, start, exact=False)
    before, after = fit["pixels"]
    print(f"FLAME alone on the views: {before:.2f} -> {after:.2f} px")
    assert after < 1.0 and after < 0.2 * before and len(fit["views"]) == 3
    fit = flameFit.fitFlameToViews(model, views, start, exact=True)
    before, after = fit["pixels"]
    print(f"FLAME on the views, exact: {before:.2f} -> {after:.2f} px")
    assert after < 0.5


def test_flameJawFittedOnTheViews():
    """ A FLAME with a jaw (skinning weights below the mouth, its joint behind it): an open mouth in the views found
    by the jaw (not by the shape). """
    from tests.test_faceBuilder import _headViews
    from meshroom.live import flameFit
    base = _fakeFlame()
    template = base.template
    # (centimeters: below the upper lip, the more the lower)
    y = template[:, 1] * 100.0
    jawWeights = np.clip((-3.0 - y) / 2.0, 0.0, 1.0) * (template[:, 2] * 100.0 > -2.0)
    regressor = np.zeros(len(template))
    behind = np.argsort(np.linalg.norm(template * 100.0 - np.array([0.0, -2.5, -6.0]), axis=1))[:8]
    regressor[behind] = 1.0 / len(behind)
    model = flameFit.FlameModel(template, base.directions, base.triangles,
                                (base.embeddingFaces, base.embeddingWeights, base.embeddingIndices), jaw=(jawWeights, regressor))
    n = face.MESH_VERTICES
    truth = 100.0 * model.posed(model.template, np.array([0.15, 0.0, 0.0]))
    size = (640, 480)
    views = []
    for camera in _headViews((0, 40, -40), size):
        xy, _, _ = camera.project(truth[:n], *size)
        views.append({"camera": camera, "size": size, "pixels": xy, "hidden": []})
    start = flameFit.fitFlame(model, 100.0 * model.template[:n])
    fit = flameFit.fitFlameToViews(model, views, start, exact=False)
    before, after = fit["pixels"]
    print(f"FLAME jaw: {before:.2f} -> {after:.2f} px, jaw {fit['jaw']}")
    assert after < 0.3 * before and 0.08 < fit["jaw"][0] < 0.22


def _skinView(camera, vertices, triangles, size):
    """ A segmentation (headFit.FitView) of a head seen by a camera: its skin where it is, the background elsewhere. """
    from meshroom.live import geoRender
    width, height = size
    xy, depth, _ = camera.project(vertices, width, height)
    _, ids = geoRender.rasterize(xy[:, 0], height - xy[:, 1], depth, triangles, (height, width))
    labels = np.zeros((height, width, 6), np.float32)
    labels[:, :, headFit.FACE_SKIN] = ids >= 0
    labels[:, :, headFit.BACKGROUND] = ids < 0
    return headFit.FitView(camera, size, labels)


def test_flameEarsOnTheSegmentation():
    """ The ears of FLAME, that no landmark sees, put where the skin of the views has them (an actor with its ears
    further out and back than the model has them). """
    from tests.test_faceBuilder import _headViews
    from meshroom.live import flameFit
    model = _fakeFlame()
    n = face.MESH_VERTICES
    head = face.canonicalHead()
    actor = _actor(3, out=0.8, back=0.5)
    size = (640, 480)
    views, segmentations = [], []
    for camera in _headViews((0, 70, -70), size):
        xy, _, _ = camera.project(actor[:n], *size)
        views.append({"camera": camera, "size": size, "pixels": xy, "hidden": []})
        segmentations.append(_skinView(camera, actor, head.triangles(), size))
    start = flameFit.fitFlame(model, actor[:n])
    ears = flameFit.earVertices(model)
    assert ears.sum() > 50
    errors = []
    for seg in (None, segmentations):
        fit = flameFit.fitFlameToViews(model, views, start, segmentations=seg)
        errors.append(np.linalg.norm(fit["vertices"][ears] - actor[ears], axis=1).mean())
    print(f"FLAME ears: {errors[0]:.3f} cm without the segmentation, {errors[1]:.3f} cm with it")
    assert errors[1] < 0.7 * errors[0]
