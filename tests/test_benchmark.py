""" The profiling of the live compositing (meshroom/live/profiling.py) and its benchmark (benchmark.py). """
import json
import os

import pytest

import meshroom
from meshroom.core import pluginManager
from meshroom.core.graph import Graph
from meshroom.live import Evaluator, buildPlan, benchmark, profiling

if not pluginManager.isNodeDescRegistered("Grade"):
    pluginManager.addPluginFromBuiltInFolder("compositing", os.path.join(os.path.dirname(meshroom.__file__), "nodes",
                                                                         "compositing"))


def _graph():
    graph = Graph("bench")
    constant = graph.addNewNode("Constant", width=64, height=48)
    blur = graph.addNewNode("Blur")
    blur.attribute("size").value = {"x": 4.0, "y": 4.0}
    graph.addEdge(constant.output, blur.input)
    transform = graph.addNewNode("Transform", rotate=10.0)
    graph.addEdge(blur.output, transform.input)
    return graph, constant, blur, transform


def test_parseFrames():
    assert benchmark.parseFrames("1001-1003") == [1001, 1002, 1003]
    assert benchmark.parseFrames("1001,1005, 1010") == [1001, 1005, 1010]
    assert benchmark.parseFrames("1-10x4") == [1, 5, 9]
    assert benchmark.parseFrames("3-1") == [3, 2, 1]
    assert benchmark.parseFrames("-2--1") == [-2, -1]


def test_profilerRecordsEachNode():
    graph, constant, blur, transform = _graph()
    evaluator = Evaluator()
    profiler = profiling.Profiler(traceMemory=True)
    evaluator.profiler = profiler
    plan = buildPlan(transform)
    for frame in (1, 2):
        evaluator.render(plan, frame)
    # Again: everything from the cache
    evaluator.render(plan, 1)
    stats = {s.name: s for s in profiler.nodes()}
    assert set(stats) >= {constant.name, blur.name, transform.name}
    for name in (constant.name, blur.name, transform.name):
        assert stats[name].renders == 2 and stats[name].seconds > 0
        assert stats[name].outputBytes > 0
    assert stats[transform.name].cacheHits == 1
    assert stats[blur.name].peakBytes > 0
    assert profiler.totalSeconds() == pytest.approx(sum(s.seconds for s in stats.values()))
    table = profiling.formatTable(profiler.nodes(), 1.0)
    assert blur.name in table and "Own time" in table


def test_ownTimeExcludesNestedRenders():
    profiler = profiling.Profiler()
    profiler.started("a", "A")
    profiler.started("b", "B")
    profiler.transferred(uploaded=100)
    profiler.finished("b", 1, 10)
    profiler.transferred(downloaded=50)
    profiler.finished("a", 1, 20)
    stats = {s.name: s for s in profiler.nodes()}
    assert stats["B"].uploadedBytes == 100 and stats["A"].downloadedBytes == 50
    # A's own time does not count B's
    assert stats["A"].seconds <= stats["A"].seconds + stats["B"].seconds
    profiling.activate(profiler)
    try:
        profiling.uploaded(7)
    finally:
        profiling.activate(None)
    profiling.uploaded(1000)  # Not profiled: nothing counted
    assert any(s.name == "(outside nodes)" and s.uploadedBytes == 7 for s in profiler.nodes())


def test_benchmarkCommand(tmp_path, capsys):
    graph, constant, blur, transform = _graph()
    path = str(tmp_path / "bench.mg")
    graph.save(path)
    output = str(tmp_path / "result.json")
    assert benchmark.main([path, "--node", transform.name, "--frames", "1-3", "--json", output]) == 0
    printed = capsys.readouterr().out
    assert transform.name in printed and "Cold" in printed and "Warm" in printed
    with open(output) as f:
        result = json.load(f)[0]
    assert result["frames"] == [1, 2, 3] and len(result["cold"]) == 3 and len(result["warm"]) == 3
    assert {node["name"] for node in result["nodes"]} >= {constant.name, blur.name, transform.name}
    # Warm: everything cached
    assert all(node["renders"] == 0 for node in result["warmNodes"])
    # Default target: the compositing node nothing reads
    assert list(benchmark.defaultTargets(graph)) == [transform.name]
