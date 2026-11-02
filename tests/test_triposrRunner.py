""" The surface extraction of the TripoSR runner (meshroom/live/neural/triposrRunner.py), without TripoSR. """
import numpy as np

from meshroom.live.neural import triposrRunner


def test_marchingTetrahedraOfASphere():
    size = 40
    grid = np.mgrid[0:size, 0:size, 0:size].astype(np.float64)
    center, radius = np.array([19.5, 20.0, 18.7]), 12.0
    # Inside: above the level (a density)
    field = radius - np.sqrt(((grid - center[:, None, None, None]) ** 2).sum(0))
    vertices, triangles = triposrRunner.marchingTetrahedra(field, 0.0)
    distance = np.linalg.norm(vertices - center, axis=1)
    assert np.abs(distance - radius).max() < 0.1
    p0, p1, p2 = (vertices[triangles[:, i]] for i in range(3))
    normals = np.cross(p1 - p0, p2 - p0)
    area = 0.5 * np.linalg.norm(normals, axis=1).sum()
    assert abs(area - 4 * np.pi * radius ** 2) < 0.03 * 4 * np.pi * radius ** 2
    # Facing out, every one
    assert ((((p0 + p1 + p2) / 3 - center) * normals).sum(1) > 0).mean() > 0.999
    # Closed: each edge in two triangles
    edges = np.sort(np.r_[triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()
    oriented = triposrRunner.orientOutward(vertices, triangles[:, [0, 2, 1]])
    np.testing.assert_array_equal(oriented, triangles)


def test_prepareImage():
    from PIL import Image
    rgba = np.zeros((100, 60, 4), np.uint8)
    rgba[20:80, 10:40] = (200, 100, 50, 255)
    image = triposrRunner.prepareImage(Image.fromarray(rgba), ratio=0.85, size=128)
    pixels = np.asarray(image)
    assert pixels.shape == (128, 128, 3)
    assert tuple(pixels[0, 0]) == (127, 127, 127) or tuple(pixels[0, 0]) == (128, 128, 128)
    person = (np.abs(pixels.astype(int) - (200, 100, 50)).sum(2) < 10)
    rows = np.nonzero(person.any(1))[0]
    assert abs((rows.max() - rows.min() + 1) / 128.0 - 0.85) < 0.03


def test_weightsFromTheModelsFolderOnly(tmp_path, monkeypatch):
    """ Without its weights in the models folder, the runner stops (it downloads nothing) and says how to get them. """
    import pytest
    monkeypatch.setenv("MESHROOM_MODELS_PATH", str(tmp_path))
    monkeypatch.delenv("MESHROOM_TRIPOSR_MODEL", raising=False)
    with pytest.raises(SystemExit) as error:
        triposrRunner.main([str(tmp_path / "image.png"), str(tmp_path / "head.npz")])
    assert "meshroom.live.neural.download triposr" in str(error.value) and "model.ckpt" in str(error.value)


def test_checkpointOfTransformers4OnTransformers5():
    """ The ViT names of the TripoSR checkpoint (transformers 4) mapped onto those of transformers 5. """
    import pytest

    def weights(names):
        return {name: np.zeros(shape) for name, shape in names}
    old = weights([("image_tokenizer.model.embeddings.cls_token", (1, 1, 8)),
                   ("image_tokenizer.model.embeddings.patch_embeddings.projection.weight", (8, 3, 4, 4)),
                   ("image_tokenizer.model.encoder.layer.0.attention.attention.query.weight", (8, 8)),
                   ("image_tokenizer.model.encoder.layer.0.attention.attention.key.bias", (8,)),
                   ("image_tokenizer.model.encoder.layer.0.attention.output.dense.weight", (8, 8)),
                   ("image_tokenizer.model.encoder.layer.0.intermediate.dense.weight", (32, 8)),
                   ("image_tokenizer.model.encoder.layer.0.output.dense.weight", (8, 32)),
                   ("image_tokenizer.model.encoder.layer.0.layernorm_before.weight", (8,)),
                   ("image_tokenizer.model.layernorm.weight", (8,)),
                   ("decoder.layers.0.weight", (4, 4))])
    new = weights([("image_tokenizer.model.embeddings.cls_token", (1, 1, 8)),
                   ("image_tokenizer.model.embeddings.patch_embed.proj.weight", (8, 3, 4, 4)),
                   ("image_tokenizer.model.layers.0.attention.q_proj.weight", (8, 8)),
                   ("image_tokenizer.model.layers.0.attention.k_proj.bias", (8,)),
                   ("image_tokenizer.model.layers.0.attention.o_proj.weight", (8, 8)),
                   ("image_tokenizer.model.layers.0.mlp.fc1.weight", (32, 8)),
                   ("image_tokenizer.model.layers.0.mlp.fc2.weight", (8, 32)),
                   ("image_tokenizer.model.layers.0.layernorm_before.weight", (8,)),
                   ("image_tokenizer.model.norm.weight", (8,)),
                   ("decoder.layers.0.weight", (4, 4))])
    old["image_tokenizer.model.encoder.layer.0.intermediate.dense.weight"][0, 0] = 7.0
    mapped = triposrRunner.remapCheckpoint(old, new)
    assert list(mapped) == list(new)
    assert mapped["image_tokenizer.model.layers.0.mlp.fc1.weight"][0, 0] == 7.0
    assert triposrRunner.remapCheckpoint(new, new).keys() == new.keys()
    with pytest.raises(SystemExit):
        triposrRunner.remapCheckpoint({"decoder.layers.0.weight": np.zeros((4, 4))}, new)
