#!/usr/bin/env bash
# Install TRELLIS.2 (image to 3D, for the FaceBuilder "Generate AI Head") in its own environment, without root, conda
# nor pip on the machine: micromamba (a single binary in ~/.local/bin) makes the environment, with Python 3.10, the
# CUDA 12.4 toolkit and a compiler to build its CUDA extensions; PyTorch 2.6 (CUDA 12.4) and the packages with pip in it.
#
#   bash setupTrellis2.sh <clone of TRELLIS.2> [environment folder (default: ~/trellis2-env)]
#
# Then, before launching Meshroom (the lines printed at the end):
#   export MESHROOM_TRELLIS2_PYTHON=<environment>/bin/python
#   export MESHROOM_TRELLIS2_ROOT=<clone of TRELLIS.2>
#
# FLASH_ATTN=1 also installs flash-attn (faster; a long build when no prebuilt wheel fits); without it, xformers is
# the attention used (trellis2Runner.py picks it when flash-attn is missing).
set -euo pipefail

ROOT=$(cd "${1:?usage: bash setupTrellis2.sh <clone of TRELLIS.2> [environment folder]}" && pwd)
ENV=${2:-$HOME/trellis2-env}
mkdir -p "$(dirname "$ENV")"
ENV=$(cd "$(dirname "$ENV")" && pwd)/$(basename "$ENV")
[ -d "$ROOT/trellis2" ] && [ -d "$ROOT/o-voxel" ] || { echo "Not a clone of TRELLIS.2: $ROOT"; exit 1; }

# Out of the Meshroom (rez) environment: its Python must not leak in
unset PYTHONPATH PYTHONHOME

echo "== GPU"
if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || nvidia-smi -L
    MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1 || echo 0)
    if [ "${MEMORY:-0}" -lt 20000 ] && [ "${FORCE:-0}" != "1" ]; then
        echo "This GPU has ${MEMORY} MiB: TRELLIS.2 needs about 24 GB (its 4 billion parameters alone take 8 GB)."
        echo "Install it on a machine with such a GPU (FORCE=1 to install here anyway)."
        exit 1
    fi
else
    echo "nvidia-smi not found: TRELLIS.2 needs an NVIDIA GPU (24 GB or more)."
    [ "${FORCE:-0}" = "1" ] || exit 1
fi

echo "== micromamba"
MAMBA=$(command -v micromamba || true)
if [ -z "$MAMBA" ]; then
    mkdir -p "$HOME/.local/bin"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME/.local" bin/micromamba
    MAMBA=$HOME/.local/bin/micromamba
fi
# Its package cache next to the environment, not in the home (often a small quota)
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$(dirname "$ENV")/.micromamba}

echo "== environment $ENV (Python 3.10, CUDA 12.4 toolkit, compilers)"
if [ ! -x "$ENV/bin/python" ]; then
    # CUDA 12.4 (that of the PyTorch wheels): its channel first, the version pinned
    "$MAMBA" create -y -p "$ENV" --override-channels -c nvidia/label/cuda-12.4.1 -c conda-forge \
        python=3.10 "cuda-version=12.4" cuda-nvcc cuda-cudart-dev cuda-cccl libcublas-dev libcusparse-dev \
        libcusolver-dev libcurand-dev gcc_linux-64=12 gxx_linux-64=12 git
fi
PY="$ENV/bin/python"
export PATH="$ENV/bin:$PATH"
export CUDA_HOME="$ENV"
export CC="$ENV/bin/x86_64-conda-linux-gnu-gcc" CXX="$ENV/bin/x86_64-conda-linux-gnu-g++"
export CPATH="$ENV/targets/x86_64-linux/include:$ENV/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$ENV/targets/x86_64-linux/lib:$ENV/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The CUDA extensions built for this GPU only (faster to build)
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] && command -v nvidia-smi > /dev/null; then
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
    [ -n "$CAP" ] && export TORCH_CUDA_ARCH_LIST="$CAP"
fi
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-(all)}"

echo "== PyTorch and packages"
"$PY" -m pip install --upgrade pip wheel setuptools
"$PY" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers zstandard \
    kornia timm pillow
"$PY" -m pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
# (the xformers built for PyTorch 2.6)
"$PY" -m pip install xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124 \
    || "$PY" -m pip install xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu124
if [ "${FLASH_ATTN:-0}" = "1" ]; then
    "$PY" -m pip install flash-attn==2.7.3 --no-build-isolation || echo "flash-attn not installed: xformers is used."
fi

echo "== CUDA extensions"
SRC="$ENV/src"
mkdir -p "$SRC"
clone() {  # url folder [git options]
    [ -d "$SRC/$2" ] || git clone "${@:3}" "$1" "$SRC/$2"
}
clone https://github.com/NVlabs/nvdiffrast.git nvdiffrast -b v0.4.0
clone https://github.com/JeffreyXiang/nvdiffrec.git nvdiffrec -b renderutils
clone https://github.com/JeffreyXiang/CuMesh.git CuMesh --recursive
clone https://github.com/JeffreyXiang/FlexGEMM.git FlexGEMM --recursive
rm -rf "$SRC/o-voxel" && cp -r "$ROOT/o-voxel" "$SRC/o-voxel"
for package in nvdiffrast nvdiffrec CuMesh FlexGEMM o-voxel; do
    echo "-- $package"
    "$PY" -m pip install "$SRC/$package" --no-build-isolation
done

echo "== check"
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "CUDA", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import nvdiffrast.torch, cumesh, flex_gemm, o_voxel  # noqa
print("extensions: ok")
EOF

echo
echo "TRELLIS.2 is installed. Before launching Meshroom:"
echo "  export MESHROOM_TRELLIS2_PYTHON=$PY"
echo "  export MESHROOM_TRELLIS2_ROOT=$ROOT"
