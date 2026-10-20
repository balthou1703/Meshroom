"""
GPU acceleration of the compositing nodes (filters, optical flow, tracking), with PyTorch.

A node with the "Use GPU" parameter (`gpuParam`) computes on the GPU when the machine has one that PyTorch can use
(CUDA, or MPS on macOS), else on the CPU; a failure on the GPU (out of memory, an operation the device does not
support) also falls back on the CPU, with a warning in the log (once per operation). The results are the same within
floating point precision, so the parameter is not part of the node uid.

Environment:
    MESHROOM_GPU=0          never use the GPU (e.g. a farm machine whose GPU is kept for something else).
    MESHROOM_GPU_DEVICE=1   the index of the CUDA device to use (0 by default).

The GPU kernels are in gpuOps.py; the neural networks (meshroom/live/neural) use the same device.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable

_lock = threading.Lock()
_state = {}
_warned = set()


def torchModule():
    """ The torch module, or None when PyTorch is not installed (or cannot be imported). """
    try:
        import torch
        return torch
    except Exception:
        return None


def _detect():
    """ (the torch device of the GPU or None, why there is none). """
    if os.environ.get("MESHROOM_GPU", "1").strip().lower() in ("0", "false", "no", "off"):
        return None, "disabled by MESHROOM_GPU"
    torch = torchModule()
    if torch is None:
        return None, "PyTorch is not installed"
    try:
        if torch.cuda.is_available():
            try:
                index = int(os.environ.get("MESHROOM_GPU_DEVICE", "0") or 0)
            except ValueError:
                index = 0
            index = min(max(index, 0), torch.cuda.device_count() - 1)
            return torch.device("cuda", index), ""
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps"), ""
    except Exception as e:
        return None, f"the GPU cannot be used ({e})"
    return None, "no CUDA (or MPS) GPU"


def availableDevice():
    """ The torch device of the GPU, or None (no PyTorch, no compatible GPU, or MESHROOM_GPU=0). Detected once. """
    with _lock:
        if "device" not in _state:
            device, reason = _detect()
            _state["device"], _state["reason"] = device, reason
            if device is None:
                logging.info(f"[GPU] Computing on the CPU: {reason}.")
            else:
                logging.info(f"[GPU] Computing on {deviceName(device)}.")
        return _state["device"]


def deviceName(device) -> str:
    """ A readable name of a torch device. """
    torch = torchModule()
    if device is None or torch is None:
        return "CPU"
    if device.type == "cuda":
        try:
            return f"{torch.cuda.get_device_name(device)} (cuda:{device.index or 0})"
        except Exception:
            return str(device)
    return str(device)


def device(enabled: bool = True):
    """ The device to compute on: the GPU when `enabled` and available, else None (the CPU). """
    return availableDevice() if enabled else None


def status() -> str:
    """ Where the GPU computations run, for the interface ("NVIDIA ... (cuda:0)", or "CPU: <why>"). """
    current = availableDevice()
    return deviceName(current) if current is not None else f"CPU: {_state.get('reason', '')}"


def reset():
    """ Detect the GPU again at the next use (e.g. after MESHROOM_GPU changed; used by the tests). """
    with _lock:
        _state.clear()
        _warned.clear()


def _warnOnce(what: str, error: Exception):
    key = (what, type(error).__name__)
    if key not in _warned:
        _warned.add(key)
        logging.warning(f"[GPU] {what or 'Computation'} failed on the GPU, computed on the CPU instead: {error}")


def emptyCache():
    """ Give back the memory kept by PyTorch on the GPU (after a failure, or a large computation). """
    torch = torchModule()
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run(enabled: bool, gpuFunction: Callable, cpuFunction: Callable, what: str = ""):
    """
    `gpuFunction(device)` on the GPU when `enabled` and a GPU is available, else `cpuFunction()`. A failure on the GPU
    falls back on the CPU (logged once per operation); a cancellation (EvaluationCancelled) is not a failure.
    """
    from .evaluator import EvaluationCancelled
    current = device(enabled)
    if current is not None:
        try:
            return gpuFunction(current)
        except EvaluationCancelled:
            raise
        except Exception as e:
            _warnOnce(what, e)
            emptyCache()
    return cpuFunction()


def gpuParam():
    """ The "Use GPU" parameter of the nodes computing on the GPU when possible. """
    from meshroom.core import desc
    return desc.BoolParam(name="useGpu", label="Use GPU", value=True, invalidate=False,
                          description="Compute on the GPU when the machine has one that PyTorch can use (CUDA); else, "
                                      "or if it fails, on the CPU (the same result). MESHROOM_GPU=0 turns it off "
                                      "everywhere.")


def enabled(ctx) -> bool:
    """ Whether a node asks for the GPU (its "useGpu" parameter, True when it has none). """
    try:
        return bool(ctx.value("useGpu"))
    except KeyError:
        return True
