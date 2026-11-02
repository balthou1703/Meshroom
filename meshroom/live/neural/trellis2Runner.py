"""
Generate a textured 3D head from a photo with TRELLIS.2 (Microsoft, MIT: https://github.com/microsoft/TRELLIS.2).

Run by the FaceBuilder ("Generate AI Head", meshroom/live/aiHead.py) in the Python of a TRELLIS.2 environment
(MESHROOM_TRELLIS2_PYTHON: its conda environment, its own PyTorch and CUDA extensions), not in Meshroom's:

    python trellis2Runner.py --root <TRELLIS.2 clone> image.png head.npz

The image is RGBA: its alpha (the person cut out by Meshroom) replaces the background removal of TRELLIS.2, whose
model (RMBG-2.0) is then not loaded. Writes head.npz (vertices (n, 3) float32, faces (m, 3) int32, colors (n, 3) uint8:
the base color of the texture at each vertex) and head.glb (the textured mesh).
Only the standard library, numpy, PIL and TRELLIS.2 itself: nothing of Meshroom is imported here.
"""
import argparse
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("image", help="RGBA image of the head (PNG)")
    parser.add_argument("output", help="the .npz written (and the .glb next to it)")
    parser.add_argument("--root", default=os.environ.get("MESHROOM_TRELLIS2_ROOT", ""),
                        help="the clone of the TRELLIS.2 repository (added to the import path)")
    parser.add_argument("--model", default=os.environ.get("MESHROOM_TRELLIS2_MODEL", "microsoft/TRELLIS.2-4B"),
                        help="the weights: a Hugging Face name or a local folder")
    parser.add_argument("--pipeline", default="1024_cascade", help="512, 1024, 1024_cascade or 1536_cascade")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vertices", type=int, default=200000, help="the mesh simplified to about this many vertices")
    args = parser.parse_args(argv)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.root:
        sys.path.insert(0, args.root)
        sys.path.insert(0, os.path.join(args.root, "o-voxel"))
    import numpy as np
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg
    import o_voxel

    image = Image.open(args.image)
    if image.mode != "RGBA":
        raise SystemExit("trellis2Runner: the image must have an alpha (the person cut out)")

    # No background removal model: the alpha of the image is used (the pipeline skips it when there is one)
    class _NoRembg:
        def __init__(self, **kwargs):
            pass

        def to(self, device):
            return self

        def cpu(self):
            return self

        def cuda(self):
            return self
    rembg.BiRefNet = _NoRembg

    print("[trellis2] loading", args.model, flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    print("[trellis2] generating", flush=True)
    mesh = pipeline.run(image, seed=args.seed, pipeline_type=args.pipeline)[0]
    mesh.simplify(16777216)
    print("[trellis2] texturing", flush=True)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords, attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=args.vertices,
        texture_size=2048, remesh=True, remesh_band=1, remesh_project=0, verbose=False)
    base = os.path.splitext(args.output)[0]
    glb.export(base + ".glb")
    try:
        colors = np.asarray(glb.visual.to_color().vertex_colors)[:, :3]
    except Exception as e:  # no texture to sample: gray
        print(f"[trellis2] no vertex colors ({e})", flush=True)
        colors = np.full((len(glb.vertices), 3), 180, dtype=np.uint8)
    np.savez_compressed(args.output, vertices=np.asarray(glb.vertices, dtype=np.float32),
                        faces=np.asarray(glb.faces, dtype=np.int32), colors=np.asarray(colors, dtype=np.uint8))
    print("[trellis2] written", args.output, flush=True)


if __name__ == "__main__":
    main()
