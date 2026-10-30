"""
Profiling of the live compositing engine: where the time, the memory and the transfers to the GPU go, node by node.

A `Profiler` set on an Evaluator (`evaluator.profiler`) records, for each node: its renders and their own time (its
inputs excluded, also for the time nodes rendering their inputs themselves), the renders found in the cache, the size
of the images it outputs, the bytes it sent to the GPU and brought back (gpuOps, geoRender, neural networks call
`uploaded` / `downloaded`), and with `traceMemory` the peak of the memory it allocated while rendering, above what
was held before (tracemalloc: NumPy allocations; slower), and of the GPU memory (PyTorch).

See meshroom/live/benchmark.py: the command line benchmark of a project.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NodeStats:
    """ What a node cost over a profiling session. """
    name: str
    nodeType: str = ""
    renders: int = 0
    cacheHits: int = 0
    seconds: float = 0.0
    outputBytes: int = 0
    uploadedBytes: int = 0
    downloadedBytes: int = 0
    peakBytes: int = 0
    peakGpuBytes: int = 0
    frames: List[int] = field(default_factory=list)

    @property
    def meanSeconds(self) -> float:
        return self.seconds / self.renders if self.renders else 0.0

    def asDict(self) -> dict:
        return {"name": self.name, "type": self.nodeType, "renders": self.renders, "cacheHits": self.cacheHits,
                "seconds": round(self.seconds, 6), "meanSeconds": round(self.meanSeconds, 6),
                "outputBytes": self.outputBytes, "uploadedBytes": self.uploadedBytes,
                "downloadedBytes": self.downloadedBytes, "peakBytes": self.peakBytes, "peakGpuBytes": self.peakGpuBytes}


class _Running:
    """ A node being rendered: its start, and the time of the nodes rendered inside it (time nodes). """
    __slots__ = ("key", "start", "children", "memoryAtStart")

    def __init__(self, key, memoryAtStart=0):
        self.key = key
        self.start = time.perf_counter()
        self.children = 0.0
        self.memoryAtStart = memoryAtStart


class Profiler:
    """
    Records the cost of each node rendered by an Evaluator (thread safe; the transfers are given to the node being
    rendered by the calling thread).
    """

    def __init__(self, traceMemory: bool = False):
        self.traceMemory = traceMemory
        self.stats: Dict[str, NodeStats] = {}
        self._lock = threading.Lock()
        self._local = threading.local()
        self._outside = NodeStats("(outside nodes)")

    # Called by the Evaluator

    def _stack(self) -> list:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        return stack

    def _statsOf(self, key: str, name: str, nodeType: str) -> NodeStats:
        stats = self.stats.get(key)
        if stats is None:
            stats = self.stats[key] = NodeStats(name, nodeType)
        return stats

    def started(self, key: str, name: str, nodeType: str = ""):
        """ A node starts rendering (its inputs are ready). """
        with self._lock:
            self._statsOf(key, name, nodeType)
        memory = _resetPeaks() if self.traceMemory else 0
        self._stack().append(_Running(key, memory))

    def finished(self, key: str, frame: int, outputBytes: int = 0):
        """ The node started last by this thread finished (also when it failed or was cancelled: outputBytes 0). """
        stack = self._stack()
        if not stack:
            return
        running = stack.pop()
        elapsed = time.perf_counter() - running.start
        if stack:
            stack[-1].children += elapsed
        peak, gpuPeak = _peaks() if self.traceMemory else (0, 0)
        # What the node allocated on top of what was already held (the cache, its inputs)
        peak = max(peak - running.memoryAtStart, 0)
        with self._lock:
            stats = self.stats[running.key]
            stats.renders += 1
            stats.seconds += max(elapsed - running.children, 0.0)
            stats.outputBytes += int(outputBytes)
            stats.peakBytes = max(stats.peakBytes, peak)
            stats.peakGpuBytes = max(stats.peakGpuBytes, gpuPeak)
            stats.frames.append(int(frame))

    def cacheHit(self, key: str, name: str, nodeType: str = ""):
        with self._lock:
            self._statsOf(key, name, nodeType).cacheHits += 1

    def transferred(self, uploaded: int = 0, downloaded: int = 0):
        """ Bytes sent to the GPU / brought back, counted for the node being rendered by this thread. """
        stack = self._stack()
        with self._lock:
            stats = self.stats.get(stack[-1].key, self._outside) if stack else self._outside
            stats.uploadedBytes += int(uploaded)
            stats.downloadedBytes += int(downloaded)

    # Results

    def nodes(self) -> List[NodeStats]:
        """ The nodes, most expensive first. """
        with self._lock:
            result = sorted(self.stats.values(), key=lambda stats: stats.seconds, reverse=True)
            if self._outside.uploadedBytes or self._outside.downloadedBytes:
                result.append(self._outside)
            return result

    def totalSeconds(self) -> float:
        with self._lock:
            return sum(stats.seconds for stats in self.stats.values())

    def reset(self):
        with self._lock:
            self.stats.clear()
            self._outside = NodeStats("(outside nodes)")


def _resetPeaks() -> int:
    """ Start measuring the peaks again; the memory traced now. """
    import tracemalloc
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    tracemalloc.reset_peak()
    torch = _cudaTorch()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()
    return tracemalloc.get_traced_memory()[0]


def _peaks():
    """ (peak bytes allocated by Python / NumPy, peak bytes of GPU memory) since the last reset. """
    import tracemalloc
    current, peak = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
    torch = _cudaTorch()
    return peak, int(torch.cuda.max_memory_allocated()) if torch is not None else 0


def _cudaTorch():
    """ The torch module when it is imported and has CUDA (never imported here). """
    import sys
    torch = sys.modules.get("torch")
    try:
        return torch if torch is not None and torch.cuda.is_available() else None
    except Exception:
        return None


# The profiler of the session being measured (benchmark), for the transfers counted outside an Evaluator call
_ACTIVE: Optional[Profiler] = None


def activate(profiler: Optional[Profiler]):
    """ Count the GPU transfers of every thread in `profiler` (None: stop). """
    global _ACTIVE
    _ACTIVE = profiler


def uploaded(nbytes: int):
    """ Bytes sent to the GPU (a no-op when nothing is profiled). """
    if _ACTIVE is not None:
        _ACTIVE.transferred(uploaded=nbytes)


def downloaded(nbytes: int):
    """ Bytes brought back from the GPU (a no-op when nothing is profiled). """
    if _ACTIVE is not None:
        _ACTIVE.transferred(downloaded=nbytes)


def formatBytes(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(count) < 1024 or unit == "GB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024.0
    return f"{count:.1f} GB"


def formatTable(nodes: List[NodeStats], wallSeconds: Optional[float] = None) -> str:
    """ The nodes as a text table: own time (total, per render, share), cache hits, output, transfers, peaks. """
    total = sum(stats.seconds for stats in nodes) or 1e-12
    rows = [("Node", "Type", "Renders", "Cached", "Own time", "Per render", "Share", "Output", "To GPU", "From GPU",
             "Allocated", "Peak VRAM")]
    for stats in nodes:
        rows.append((stats.name, stats.nodeType, str(stats.renders), str(stats.cacheHits), f"{stats.seconds:.3f} s",
                     f"{stats.meanSeconds * 1000:.1f} ms", f"{100.0 * stats.seconds / total:.1f} %",
                     formatBytes(stats.outputBytes), formatBytes(stats.uploadedBytes), formatBytes(stats.downloadedBytes),
                     formatBytes(stats.peakBytes) if stats.peakBytes else "-",
                     formatBytes(stats.peakGpuBytes) if stats.peakGpuBytes else "-"))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(width) if i < 2 else cell.rjust(width) for i, (cell, width) in enumerate(zip(row, widths)))
             for row in rows]
    lines.insert(1, "-" * len(lines[0]))
    if wallSeconds is not None:
        lines.append("-" * len(lines[0]))
        lines.append(f"Nodes: {total:.3f} s, wall time: {wallSeconds:.3f} s (the rest: the engine, the cache, Python)")
    return "\n".join(lines)
