"""
Generate a colored 3D head from a photo with TripoSR (Stability AI & Tripo AI, MIT:
https://github.com/VAST-AI-Research/TripoSR) for the FaceBuilder ("Generate AI Head", meshroom/live/aiHead.py).
A light model (about 6 GB of GPU memory, or the CPU): it runs in Meshroom's own Python, with the copy of its code in
Meshroom (neural/triposr/tsr, MIT) and a few small packages
(einops, omegaconf, transformers), its weights in the models folder (python -m meshroom.live.neural.download triposr:
then nothing is downloaded); what TripoSR imports only for its other uses (trimesh, rembg, imageio, torchmcubes) is
not needed: the surface is extracted here (marching tetrahedra in numpy, or scikit-image / PyMCubes when installed).

    python triposrRunner.py image.png head.npz

The image is RGBA (its alpha: the person cut out by Meshroom). Writes head.npz: vertices (n, 3) float32, faces (m, 3)
int32, colors (n, 3) uint8.
"""
import argparse
import os
import sys
import types


def marchingTetrahedra(field, level=0.0):
    """
    The surface where `field` (a, b, c) crosses `level` (inside: above it), in grid coordinates: (vertices (n, 3),
    triangles (m, 3)), the triangles facing out (toward lower values). Only the cells crossed by it are processed.
    """
    import numpy as np
    f = np.asarray(field, dtype=np.float32) - level
    a, b, c = f.shape
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    # The cells crossed by the surface: corners on both sides
    low = np.full((a - 1, b - 1, c - 1), np.inf, np.float32)
    high = np.full((a - 1, b - 1, c - 1), -np.inf, np.float32)
    for dx, dy, dz in corners:
        values = f[dx:a - 1 + dx, dy:b - 1 + dy, dz:c - 1 + dz]
        low = np.minimum(low, values)
        high = np.maximum(high, values)
    cells = np.argwhere((low <= 0) & (high > 0))
    if not len(cells):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    # Each cube in 6 tetrahedra around its diagonal 0-6
    tetrahedra = np.array([[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]])
    points = cells[:, None, :] + corners[None, :, :]                     # (k, 8, 3)
    ids = (points[..., 0] * b + points[..., 1]) * c + points[..., 2]      # (k, 8)
    values = f[points[..., 0], points[..., 1], points[..., 2]]           # (k, 8)
    tetIds = ids[:, tetrahedra].reshape(-1, 4)
    tetValues = values[:, tetrahedra].reshape(-1, 4)
    inside = tetValues > 0
    cases = (inside * np.array([1, 2, 4, 8])).sum(1)
    edgesA, edgesB, outward = [], [], []
    # The triangles of each case, as edges of the tetrahedron (pairs of its corners)
    for case in range(1, 15):
        selected = np.where(cases == case)[0]
        if not len(selected):
            continue
        ins = [i for i in range(4) if case >> i & 1]
        outs = [i for i in range(4) if not case >> i & 1]
        if len(ins) == 1 or len(outs) == 1:
            lone, others = (ins[0], outs) if len(ins) == 1 else (outs[0], ins)
            triangles = [[(lone, others[0]), (lone, others[1]), (lone, others[2])]]
        # (two corners inside, two outside: a quad, two triangles)
        else:
            p, q = ins
            r, s = outs
            triangles = [[(p, r), (p, s), (q, s)], [(p, r), (q, s), (q, r)]]
        ids4 = tetIds[selected]
        # Out of the surface: from the inside corners toward the outside ones (in grid coordinates)
        coords = np.stack([ids4 // (b * c), ids4 // c % b, ids4 % c], axis=-1).astype(np.float64)
        direction = coords[:, outs].mean(1) - coords[:, ins].mean(1)
        for triangle in triangles:
            edgesA.append(np.stack([ids4[:, e[0]] for e in triangle], axis=1))
            edgesB.append(np.stack([ids4[:, e[1]] for e in triangle], axis=1))
            outward.append(direction)
    edgesA, edgesB, outward = np.concatenate(edgesA), np.concatenate(edgesB), np.concatenate(outward)
    low, high = np.minimum(edgesA, edgesB), np.maximum(edgesA, edgesB)
    keys = low.astype(np.int64) * (a * b * c) + high
    unique, inverse = np.unique(keys.ravel(), return_inverse=True)
    lowIds, highIds = unique // (a * b * c), unique % (a * b * c)
    flat = f.ravel()
    fl, fh = flat[lowIds], flat[highIds]
    t = np.clip(fl / np.where(fl - fh == 0, 1e-12, fl - fh), 0.0, 1.0)[:, None]

    def coordinates(index):
        return np.stack([index // (b * c), index // c % b, index % c], axis=-1).astype(np.float64)
    vertices = coordinates(lowIds) + t * (coordinates(highIds) - coordinates(lowIds))
    triangles = inverse.reshape(-1, 3)
    # Facing out
    p0, p1, p2 = (vertices[triangles[:, i]] for i in range(3))
    flip = (np.cross(p1 - p0, p2 - p0) * outward).sum(1) < 0
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    # No degenerate triangles (two corners on the same vertex)
    keep = (triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2]) & (triangles[:, 0] != triangles[:, 2])
    return vertices, triangles[keep]


def extractSurface(field, level):
    """ (vertices, triangles) of the level surface of a grid: scikit-image or PyMCubes when installed, else
    marchingTetrahedra. Triangles facing toward lower values. """
    import numpy as np
    try:
        from skimage import measure
        vertices, triangles, _, _ = measure.marching_cubes(np.asarray(field, np.float32), level)
        vertices, triangles = vertices.astype(np.float64), triangles.astype(np.int64)
    except ImportError:
        vertices, triangles = marchingTetrahedra(field, level)
    return vertices, orientOutward(vertices, triangles)


def orientOutward(vertices, triangles):
    """ The triangles of a closed surface turned to face out of it (away from its middle, on the whole). """
    import numpy as np
    if not len(triangles):
        return triangles
    p0, p1, p2 = (vertices[triangles[:, i]] for i in range(3))
    normals = np.cross(p1 - p0, p2 - p0)
    middle = vertices.mean(axis=0)
    if ((((p0 + p1 + p2) / 3.0) - middle) * normals).sum() < 0:
        return triangles[:, [0, 2, 1]]
    return triangles


def _stubModules():
    """ What TripoSR imports for its other uses (videos, texture baking, its own background removal and marching
    cubes), when it is not installed: empty modules. """
    for name in ("trimesh", "rembg", "imageio", "torchmcubes", "xatlas", "moderngl"):
        try:
            __import__(name)
        except Exception:
            module = types.ModuleType(name)
            if name == "torchmcubes":
                def unavailable(*args, **kwargs):
                    raise RuntimeError("torchmcubes is not installed")
                module.marching_cubes = unavailable
            if name == "trimesh":
                module.Trimesh = object
            sys.modules[name] = module


def prepareImage(image, ratio=0.85, size=512):
    """ The person of an RGBA image on the gray of TripoSR, centered in a square, `ratio` of it. """
    import numpy as np
    from PIL import Image
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[:, :, 3]
    rows, columns = np.nonzero(alpha > 0.5)
    if not len(rows):
        raise SystemExit("triposrRunner: the image has no person (empty alpha)")
    top, bottom, left, right = rows.min(), rows.max() + 1, columns.min(), columns.max() + 1
    crop = rgba[top:bottom, left:right]
    side = int(max(bottom - top, right - left) / ratio)
    canvas = np.zeros((side, side, 4), np.float32)
    y0, x0 = (side - (bottom - top)) // 2, (side - (right - left)) // 2
    canvas[y0:y0 + bottom - top, x0:x0 + right - left] = crop
    rgb = canvas[:, :, :3] * canvas[:, :, 3:4] + 0.5 * (1.0 - canvas[:, :, 3:4])
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)


def _device(torch, wanted):
    """ The GPU when PyTorch can run on it (an old GPU may be too old for its build), else the CPU. """
    if wanted == "cpu" or not torch.cuda.is_available():
        return "cpu"
    try:
        (torch.ones(8, device="cuda") * 2).sum().item()
        return "cuda"
    except Exception as e:
        print(f"[triposr] the GPU cannot be used ({str(e).splitlines()[0]}): on the CPU", flush=True)
        return "cpu"


def _load(TSR, name, torch):
    """ TSR.from_pretrained, its checkpoint read whatever the PyTorch (weights only by default since 2.6). """
    from omegaconf import OmegaConf
    if os.path.isdir(name):
        configPath, weightPath = os.path.join(name, "config.yaml"), os.path.join(name, "model.ckpt")
    else:
        from huggingface_hub import hf_hub_download
        configPath = hf_hub_download(repo_id=name, filename="config.yaml")
        weightPath = hf_hub_download(repo_id=name, filename="model.ckpt")
    config = OmegaConf.load(configPath)
    OmegaConf.resolve(config)
    model = TSR(config)
    try:
        checkpoint = torch.load(weightPath, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(weightPath, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("image", help="RGBA image of the head (PNG)")
    parser.add_argument("output", help="the .npz written")
    parser.add_argument("--root", default=os.environ.get("MESHROOM_TRIPOSR_ROOT", ""),
                        help="a clone of TripoSR to use instead of the copy of its code in Meshroom (neural/triposr/tsr)")
    parser.add_argument("--model", default=os.environ.get("MESHROOM_TRIPOSR_MODEL", ""),
                        help="the weights: a folder (config.yaml, model.ckpt) or a Hugging Face name; default: the "
                             "models folder (MESHROOM_MODELS_PATH/triposr), else stabilityai/TripoSR downloaded")
    parser.add_argument("--resolution", type=int, default=192, help="the grid of the surface")
    parser.add_argument("--threshold", type=float, default=25.0, help="the density of the surface")
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    parser.add_argument("--chunk", type=int, default=8192)
    args = parser.parse_args(argv)
    # The code: the copy in Meshroom (neural/triposr/tsr, imported as "tsr": its config names its classes so), or a clone
    sys.path.insert(0, args.root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "triposr"))
    _stubModules()
    import numpy as np
    import torch
    from PIL import Image
    try:
        from tsr.system import TSR
    except ImportError as e:
        raise SystemExit(f"triposrRunner: TripoSR cannot be imported ({e}): it needs omegaconf, einops and "
                         "transformers (pip install --target <pylibs> omegaconf einops)") from e
    # The weights: the models folder when they are there (nothing downloaded, offline)
    if not args.model:
        folder = os.path.join(os.environ.get("MESHROOM_MODELS_PATH", ""), "triposr")
        if all(os.path.isfile(os.path.join(folder, name)) for name in ("config.yaml", "model.ckpt")):
            args.model = folder
        else:
            print("[triposr] not in the models folder (python -m meshroom.live.neural.download triposr): "
                  "downloaded from Hugging Face", flush=True)
            args.model = "stabilityai/TripoSR"
    if os.path.isdir(args.model):
        dino = os.path.join(args.model, "dino-vitb16-config.json")
        if os.path.isfile(dino):
            os.environ["MESHROOM_TRIPOSR_DINO_CONFIG"] = dino
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

    device = _device(torch, args.device)
    print(f"[triposr] loading {args.model} ({device})", flush=True)
    model = _load(TSR, args.model, torch)
    model.renderer.set_chunk_size(args.chunk)
    model.to(device)
    image = prepareImage(Image.open(args.image))
    print("[triposr] generating", flush=True)
    with torch.no_grad():
        codes = model([image], device=device)
        radius = model.renderer.cfg.radius
        steps = torch.linspace(-radius, radius, args.resolution)
        print("[triposr] density", flush=True)
        densities = []
        # The grid slice by slice (memory)
        for x in steps:
            yy, zz = torch.meshgrid(steps, steps, indexing="ij")
            points = torch.stack([torch.full_like(yy, float(x)), yy, zz], dim=-1).reshape(-1, 3).to(device)
            density = model.renderer.query_triplane(model.decoder, points, codes[0])["density_act"]
            densities.append(density.reshape(args.resolution, args.resolution).float().cpu().numpy())
        field = np.stack(densities)
    print("[triposr] surface", flush=True)
    vertices, triangles = extractSurface(field, args.threshold)
    if not len(triangles):
        raise SystemExit("triposrRunner: no surface")
    vertices = vertices / (args.resolution - 1) * 2.0 * radius - radius
    with torch.no_grad():
        colors = []
        for start in range(0, len(vertices), 65536):
            chunk = torch.as_tensor(vertices[start:start + 65536], dtype=torch.float32, device=device)
            colors.append(model.renderer.query_triplane(model.decoder, chunk, codes[0])["color"].float().cpu().numpy())
        colors = np.concatenate(colors)
    np.savez_compressed(args.output, vertices=vertices.astype(np.float32), faces=triangles.astype(np.int32),
                        colors=(np.clip(colors, 0, 1) * 255).astype(np.uint8))
    print(f"[triposr] written {args.output}: {len(vertices)} vertices", flush=True)


if __name__ == "__main__":
    main()
