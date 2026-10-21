"""
GPU acceleration of the compositing nodes (filters, optical flow, tracking), with PyTorch.

A node with the "Use GPU" parameter (`gpuParam`) computes on the GPU when the machine has one that PyTorch can use
(CUDA, or MPS on macOS), else on the CPU; a failure on the GPU (out of memory, an operation the device does not
support) also falls back on the CPU, with a warning in the log (once per operation); when the error says the GPU cannot run PyTorch
at all (e.g. "no kernel image is available": a GPU too old for this PyTorch build), the GPU is no longer used in the
session (`disable`). The results are the same within
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
            device = torch.device("cuda", index)
            reason = _unsupportedReason(torch, index)
            if reason:
                return None, reason
            return device, ""
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps"), ""
    except Exception as e:
        return None, f"the GPU cannot be used ({e})"
    return None, "no CUDA (or MPS) GPU"


def _unsupportedReason(torch, index: int) -> str:
    """
    Why PyTorch cannot compute on the CUDA device `index` ("" if it can): a GPU too old (or too new) for the
    architectures this PyTorch was built for (it is seen, but no kernel runs on it), checked with a small computation.
    """
    try:
        major, minor = torch.cuda.get_device_capability(index)
        name = torch.cuda.get_device_name(index)
        arches = list(torch.cuda.get_arch_list() or [])
    except Exception as e:
        return f"the GPU cannot be queried ({e})"
    capability = major * 10 + minor

    def runs(arch):
        # sm_XY: compiled for that architecture (runs on the same major version); compute_XY: compiled on the fly
        kind, _, number = arch.partition("_")
        if not number.isdigit():
            return False
        value = int(number)
        if kind == "sm":
            return value // 10 == major and value % 10 <= minor
        return kind == "compute" and value <= capability
    if arches and not any(runs(arch) for arch in arches):
        return (f"{name} (CUDA capability {major}.{minor}) is not supported by this PyTorch (built for "
                f"{', '.join(arches)})")
    try:
        # A few kernels of the kinds the nodes use (elementwise, reductions, cuBLAS, cuDNN, FFT): a PyTorch built for
        # other GPUs may run some and not others ("no kernel image is available")
        cuda = torch.device("cuda", index)
        values = torch.rand(1, 2, 16, 16, device=cuda)
        (values * 2).sum().item()
        (values[0, 0] @ values[0, 1]).sum().item()
        torch.nn.functional.conv2d(values, torch.rand(2, 2, 3, 3, device=cuda), padding=1).sum().item()
        torch.fft.rfft2(values).abs().sum().item()
        torch.cuda.synchronize(cuda)
    except Exception as e:
        return f"{name} cannot run PyTorch ({_firstLine(e)})"
    return ""


def _firstLine(error: Exception) -> str:
    text = str(error).strip()
    return text.splitlines()[0] if text else type(error).__name__


# Errors telling that the GPU cannot run PyTorch (not a passing problem, as running out of memory)
_DEVICE_FAILURES = ("no kernel image", "cuda error", "cublas_status", "cudnn_status", "cudnn error", "cufft",
                    "device-side assert", "illegal memory access", "unspecified launch failure", "not compiled with cuda",
                    "is not supported on", "invalid device function", "mps backend", "unsupported gpu")


def isDeviceFailure(error: Exception) -> bool:
    """ Whether `error` says the GPU cannot run PyTorch (then it is not used again in this session). """
    text = str(error).lower()
    return "out of memory" not in text and any(failure in text for failure in _DEVICE_FAILURES)


def disable(reason: str):
    """ Stop using the GPU for the rest of the session (everything computed on the CPU), logged once. """
    with _lock:
        if "device" in _state and _state["device"] is None:
            return
        _state["device"], _state["reason"] = None, reason
    logging.warning(f"[GPU] The GPU is not used anymore in this session, everything is computed on the CPU: {reason}.")


def failed(what: str, error: Exception):
    """ A computation failed on the GPU (the caller computes it on the CPU): logged once, memory given back, and the
    GPU disabled for the session when it cannot run PyTorch (e.g. "no kernel image is available"). """
    _warnOnce(what, error)
    emptyCache()
    if isDeviceFailure(error):
        disable(f"{what or 'a computation'} failed ({_firstLine(error)})")


def usable(device):
    """ `device` when it is a GPU still in use (not disabled after a failure), else None (the CPU). """
    if device is None or getattr(device, "type", "") == "cpu":
        return device
    return device if availableDevice() is not None else None


def withFallback(device, function: Callable, what: str = ""):
    """
    `function(device)` on the GPU `device` (still usable), else, or if it fails there, `function(None)` (the CPU).
    For the computations given a device beforehand (the networks: on the CPU they are slower, not different).
    """
    from .evaluator import EvaluationCancelled
    current = usable(device)
    if current is not None:
        try:
            return function(current)
        except EvaluationCancelled:
            raise
        except Exception as e:
            if type(e).__name__ == "NeuralUnavailable":
                raise
            failed(what, e)
    return function(None)


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
            failed(what, e)
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
