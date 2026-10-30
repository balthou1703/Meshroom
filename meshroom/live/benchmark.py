"""
Benchmark of the live compositing of a project: the time, the memory and the GPU transfers of each node.

    python -m meshroom.live.benchmark project.mg [--node compositing_1/Grade_1] [--frames 1001-1010]
                                      [--proxy 0.5] [--no-gpu] [--memory] [--cache-gb 8] [--json result.json]
                                      [--python-profile 30]

Without --node, the branches of the Write nodes are measured (in the subgraphs too: the compositing workspace keeps
its compositing in compositing_1), else the last compositing nodes. Each frame is rendered once with an empty cache
("cold": everything computed), then again ("warm": what the cache gives). --memory traces the memory allocated by each
node (tracemalloc, slower: the times are then less reliable). The table lists the nodes by their own time (their
inputs excluded); the JSON keeps everything (per frame times, nodes, peaks).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List, Optional


def parseFrames(text: str) -> List[int]:
    """ "1001-1010", "1001,1005,1010", "1001-1100x10" (every 10th), or a mix separated by commas. """
    frames = []
    for part in (piece.strip() for piece in text.split(",")):
        if not part:
            continue
        step = 1
        if "x" in part:
            part, stepText = part.split("x", 1)
            step = max(1, int(stepText))
        if "-" in part[1:]:
            split = part.index("-", 1)
            first, last = int(part[:split]), int(part[split + 1:])
            frames.extend(range(first, last + 1, step) if last >= first else range(first, last - 1, -step))
        else:
            frames.append(int(part))
    return frames


def _innerNodes(graph, prefix=""):
    """ (path, node) of the nodes of a graph and of its SubGraph nodes, recursively. """
    from .evaluator import _innerGraph, _isSubGraphNode
    for node in graph.nodes:
        path = f"{prefix}{node.name}"
        yield path, node
        if _isSubGraphNode(node):
            try:
                inner = _innerGraph(node)
            except Exception:
                continue
            yield from _innerNodes(inner, path + "/")


def findNode(graph, path: str):
    """ A node by its path: its name, or "subgraph/.../name" inside SubGraph nodes. """
    for nodePath, node in _innerNodes(graph):
        if nodePath == path:
            return node
    # A name alone: the first node with it, wherever it is
    for nodePath, node in _innerNodes(graph):
        if nodePath.split("/")[-1] == path:
            return node
    raise KeyError(f"No node '{path}' in the project.")


def defaultTargets(graph) -> Dict[str, object]:
    """ {path: node} measured by default: what the Write nodes write, else the compositing nodes nothing reads. """
    from .evaluator import isEvaluable, isImageOpNode
    targets = {}
    nodes = list(_innerNodes(graph))
    for path, node in nodes:
        if node.nodeType == "Write" and node.hasAttribute("image") and node.attribute("image").isLink:
            source = node.attribute("image").inputRootLink.node
            if isEvaluable(source):
                targets[path.rsplit("/", 1)[0] + "/" + source.name if "/" in path else source.name] = source
    if targets:
        return targets
    for path, node in nodes:
        if not isImageOpNode(node) or not isEvaluable(node):
            continue
        readers = [edge.dst.node for edge in node.graph.edges if edge.src.node is node]
        if not any(isImageOpNode(reader) for reader in readers):
            targets[path] = node
    return targets


def defaultFrames(graph, node) -> List[int]:
    """ Ten frames from the start of the timeline (the project's frame range), else of the image; else frame 1. """
    frameRange = getattr(graph, "frameRange", {}) or {}
    if "first" in frameRange and "last" in frameRange:
        first, last = int(frameRange["first"]), int(frameRange["last"])
    else:
        from . import Evaluator, buildPlan
        try:
            info = Evaluator().info(buildPlan(node))
        except Exception:
            info = None
        if info is None or not info.hasFrameRange:
            return [1]
        first, last = int(info.firstFrame), int(info.lastFrame)
    return list(range(first, min(last, first + 9) + 1))


def _peakRss() -> int:
    """ The peak memory of the process in bytes (0 where unknown). """
    try:
        import resource
    except ImportError:
        return 0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Kilobytes on Linux, bytes on macOS
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def benchmark(node, frames: List[int], proxy: float = 1.0, cacheBytes: Optional[int] = None, warm: bool = True,
              traceMemory: bool = False, progress=None) -> dict:
    """
    Render `frames` of a compositing node with an empty cache, then again (`warm`), profiled. `progress(pass name,
    frame, seconds, error)` after each frame.

    Returns:
        {"node", "frames", "cold": [seconds per frame], "warm": [...], "nodes": [NodeStats.asDict() of the cold pass,
        most expensive first], "warmNodes": [...], "cacheBytes", "peakRssBytes", "errors": {frame: message}}
    """
    from . import Evaluator, ImageCache, buildPlan
    from . import profiling
    from .evaluator import EvaluationError
    evaluator = Evaluator(ImageCache(cacheBytes) if cacheBytes else ImageCache())
    plan = buildPlan(node, proxy)
    result = {"node": node.name, "frames": list(frames), "proxy": proxy, "cold": [], "warm": [], "errors": {}}
    for passName in ("cold", "warm") if warm else ("cold",):
        profiler = profiling.Profiler(traceMemory=traceMemory)
        evaluator.profiler = profiler
        profiling.activate(profiler)
        try:
            for frame in frames:
                start = time.perf_counter()
                error = ""
                try:
                    evaluator.render(plan, frame)
                except EvaluationError as e:
                    error = result["errors"][frame] = f"{e.nodeName}: {e.message}"
                result[passName].append(time.perf_counter() - start)
                if progress:
                    progress(passName, frame, result[passName][-1], error)
        finally:
            profiling.activate(None)
            evaluator.profiler = None
        result["nodes" if passName == "cold" else "warmNodes"] = [stats.asDict() for stats in profiler.nodes()]
        result["_profiler" if passName == "cold" else "_warmProfiler"] = profiler
    result["cacheBytes"] = int(evaluator.cache.usedBytes)
    result["peakRssBytes"] = _peakRss()
    return result


def report(result: dict) -> str:
    """ The result of `benchmark` as text. """
    from .profiling import formatBytes, formatTable
    lines = [f"== {result['node']}: {len(result['frames'])} frames ({result['frames'][0]}..{result['frames'][-1]}), "
             f"proxy {result['proxy']:g}"]
    cold = result["cold"]
    if cold:
        total = sum(cold)
        lines.append(f"Cold (empty cache): {total:.3f} s, {1000.0 * total / len(cold):.1f} ms per frame, "
                     f"{len(cold) / total if total > 0 else 0:.2f} frames/s; slowest frame {max(cold):.3f} s")
    if result.get("warm"):
        total = sum(result["warm"])
        lines.append(f"Warm (cached): {total:.3f} s, {1000.0 * total / len(result['warm']):.1f} ms per frame")
    lines.append("")
    lines.append(formatTable(result["_profiler"].nodes(), sum(cold)))
    lines.append("")
    if result.get("cacheBytes") is not None:
        lines.append(f"Cache: {formatBytes(result['cacheBytes'])}")
    if result.get("peakRssBytes"):
        lines.append(f"Peak memory of the process: {formatBytes(result['peakRssBytes'])}")
    for frame, message in sorted(result["errors"].items()):
        lines.append(f"Frame {frame} failed: {message}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("project", help="Meshroom project (.mg) or template (.mgt).")
    parser.add_argument("--node", action="append", default=[],
                        help="Node to measure (its branch): its name, or a path through subgraphs "
                             "(compositing_1/Merge_1). Repeatable. Default: what the Write nodes write.")
    parser.add_argument("--frames", default="", help="Frames: 1001-1010, 1001,1005, 1001-1100x10. Default: ten "
                                                     "frames from the start of the timeline.")
    parser.add_argument("--proxy", type=float, default=1.0, help="Proxy scale (1, 0.5, 0.25).")
    parser.add_argument("--no-gpu", action="store_true", help="Everything on the CPU.")
    parser.add_argument("--memory", action="store_true", help="Trace the memory of each node (slower).")
    parser.add_argument("--no-warm", action="store_true", help="Only the cold pass.")
    parser.add_argument("--cache-gb", type=float, default=0.0, help="Size of the image cache (default: the session's).")
    parser.add_argument("--json", default="", help="Write the results to this JSON file.")
    parser.add_argument("--python-profile", type=int, default=0, metavar="N",
                        help="Also profile the Python functions (cProfile) and list the N most expensive ones.")
    args = parser.parse_args(argv)

    import meshroom.core
    from meshroom.core.graph import loadGraph
    from . import gpu
    from meshroom.core import pluginManager
    if not pluginManager.isNodeDescRegistered("Read"):
        meshroom.core.initNodes()
    if args.no_gpu:
        gpu.disable("benchmark --no-gpu")
    graph = loadGraph(args.project)
    targets = {path: findNode(graph, path) for path in args.node} if args.node else defaultTargets(graph)
    if not targets:
        print("No compositing node to measure in the project.", file=sys.stderr)
        return 1
    print(f"GPU: {gpu.status()}")
    print(f"Measured: {', '.join(targets)}", flush=True)
    results = []
    for path, node in targets.items():
        frames = parseFrames(args.frames) if args.frames else defaultFrames(graph, node)
        pythonProfile = None
        if args.python_profile:
            import cProfile
            pythonProfile = cProfile.Profile()
            pythonProfile.enable()
        print(f"\n{path}: {len(frames)} frames ({frames[0]}..{frames[-1]})", flush=True)

        def progress(passName, frame, seconds, error):
            print(f"  {passName} frame {frame}: {seconds:.2f} s{'  FAILED: ' + error if error else ''}", flush=True)
        result = benchmark(node, frames, proxy=args.proxy, cacheBytes=int(args.cache_gb * 1024 ** 3) or None,
                           warm=not args.no_warm, traceMemory=args.memory, progress=progress)
        if pythonProfile is not None:
            import io
            import pstats
            pythonProfile.disable()
            text = io.StringIO()
            pstats.Stats(pythonProfile, stream=text).sort_stats("tottime").print_stats(args.python_profile)
            result["_pythonProfile"] = text.getvalue()
        result["path"] = path
        print()
        print(report(result))
        if result.get("_pythonProfile"):
            print("\nPython functions (own time):")
            print(result["_pythonProfile"])
        results.append(result)
    if args.json:
        with open(args.json, "w") as f:
            json.dump([{key: value for key, value in result.items() if not key.startswith("_")} for result in results],
                      f, indent=2)
        print(f"\nResults written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
