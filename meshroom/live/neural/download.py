"""
Fill the models folder with the weights of the networks, on a machine with internet access:

    python -m meshroom.live.neural.download --path /shared/meshroom-models [raft tapir lightglue megapose cotracker]

Without names: every model except the non-commercial ones (CoTracker), which must be asked for by name. Then set
MESHROOM_MODELS_PATH=/shared/meshroom-models where Meshroom runs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import NeuralUnavailable
from .models import MODELS, MODELS_PATH_VARIABLE


def _downloadFile(urls, path: str):
    """ Download a file from the first of `urls` (one or several) that answers (without PyTorch: urllib). """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        print(f"  {path}: already there")
        return
    errors = []
    for url in ([urls] if isinstance(urls, str) else urls):
        print(f"  {url} -> {path}")
        try:
            try:
                import torch
                torch.hub.download_url_to_file(url, path, progress=True)
            except ImportError:
                import shutil
                import urllib.request
                with urllib.request.urlopen(url) as response, open(path + ".part", "wb") as file:
                    shutil.copyfileobj(response, file)
                os.replace(path + ".part", path)
            return
        except Exception as e:
            errors.append(f"{url}: {e}")
    raise RuntimeError("; ".join(errors))


def download(name: str):
    """ Download the weights of a model into the models folder (MESHROOM_MODELS_PATH). """
    from . import models
    model = MODELS[name]
    folder = models.modelsPath()
    print(f"{model.label} ({model.license})")
    for relative, url in model.urls.items():
        _downloadFile(url, os.path.join(folder, relative))
    if name == "raft":
        # torchvision downloads into the torch hub cache, pointed to the models folder
        models.useTorchHub()
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=True)
    elif name == "lightglue":
        models.useTorchHub()
        from lightglue import ALIKED, LightGlue
        ALIKED(max_num_keypoints=2048)
        LightGlue(features="aliked")
    elif name == "megapose":
        data = os.path.join(folder, "megapose")
        os.makedirs(data, exist_ok=True)
        environment = dict(os.environ, MEGAPOSE_DATA_DIR=data, HAPPYPOSE_DATA_DIR=data)
        commands = ([sys.executable, "-m", "megapose.scripts.download", "--megapose_models"],
                    [sys.executable, "-m", "happypose.toolbox.utils.download", "--megapose_models"])
        for command in commands:
            if subprocess.call(command, env=environment) == 0:
                break
        else:
            raise NeuralUnavailable("cannot download the MegaPose models: install megapose6d (or happypose) and "
                                    "run its download script with MEGAPOSE_DATA_DIR=" + data)


def main(arguments=None) -> int:
    parser = argparse.ArgumentParser(description="Download the weights of the networks used by the compositing nodes.")
    parser.add_argument("names", nargs="*",
                        help=f"models among {', '.join(sorted(MODELS))} (default: all but the non-commercial ones)")
    parser.add_argument("--path", default=os.environ.get(MODELS_PATH_VARIABLE, ""),
                        help=f"the models folder (default: ${MODELS_PATH_VARIABLE})")
    args = parser.parse_args(arguments)
    if not args.path:
        parser.error(f"give --path or set {MODELS_PATH_VARIABLE}")
    unknown = [name for name in args.names if name not in MODELS]
    if unknown:
        parser.error(f"unknown models: {', '.join(unknown)} (known: {', '.join(sorted(MODELS))})")
    os.makedirs(args.path, exist_ok=True)
    os.environ[MODELS_PATH_VARIABLE] = args.path
    names = args.names or [name for name, model in MODELS.items() if not model.nonCommercial]
    failed = []
    for name in names:
        if MODELS[name].nonCommercial:
            print(f"Note: {MODELS[name].label} is licensed {MODELS[name].license}.")
        try:
            download(name)
        except Exception as e:
            print(f"  failed: {e}", file=sys.stderr)
            failed.append(name)
    print(f"\nModels folder: {args.path}\nSet {MODELS_PATH_VARIABLE}={args.path} where Meshroom runs.")
    if failed:
        print(f"Not downloaded: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
