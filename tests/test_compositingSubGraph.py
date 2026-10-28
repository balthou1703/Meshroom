"""Compositing nodes in subgraphs: the live evaluation goes through SubGraph nodes and their GraphInput."""

import json
import numpy as np

from meshroom.core.subgraph.editing import SubGraphStack, collapseIntoSubGraph
from meshroom.live import Evaluator, buildPlan
from meshroom.nodes.subgraph.GraphInput import GraphInput
from meshroom.nodes.subgraph.GraphOutput import GraphOutput
from meshroom.nodes.subgraph.SubGraph import SubGraph

from .test_liveCompositing import constantGraph
from .utils import registerNodeDesc, unregisterNodeDesc

_NODE_DESCS = [GraphInput, GraphOutput, SubGraph]


def setup_module():
    for nodeDesc in _NODE_DESCS:
        registerNodeDesc(nodeDesc)


def teardown_module():
    for nodeDesc in _NODE_DESCS:
        unregisterNodeDesc(nodeDesc)


def render(node, frame=1):
    return Evaluator().render(buildPlan(node), frame).rgba()


def test_subGraphOfCompositingNodesIsEvaluatedLikeItsNodes():
    graph, red, blue, grade, merge = constantGraph()
    expected = render(merge)
    subGraphNode, _ = collapseIntoSubGraph(graph, [grade])
    graph.update()
    assert type(list(subGraphNode.ins.value)[0].desc).__name__ == "Image"
    assert merge.A.inputRootLink.node is subGraphNode
    plan = buildPlan(merge)
    assert plan.keyOf(f"{subGraphNode.name}/{grade.name}") is not None
    np.testing.assert_allclose(Evaluator().render(plan, 1).rgba(), expected)


def test_parameterExposedOnTheSubGraphNode():
    graph, red, blue, grade, merge = constantGraph()
    subGraphNode, _ = collapseIntoSubGraph(graph, [red, grade])
    stack = SubGraphStack()
    inner = stack.enter(graph, subGraphNode)
    innerRed = next(node for node in inner.nodes if node.nodeType == "Constant")
    child = inner.node("GraphInput_1").ports.duplicateAttribute(innerRed.width, isOutput=True)
    inner.addEdge(child, innerRed.width)
    stack.exit(inner)
    after = graph.addNewNode("Grade")
    graph.addEdge(list(subGraphNode.outs.value)[0], after.input)
    evaluator = Evaluator()
    subGraphNode.ins.value.get(child.name).value = 6
    graph.update()
    assert evaluator.render(buildPlan(after), 1).rgba().shape[1] == 6
    subGraphNode.ins.value.get(child.name).value = 5
    graph.update()
    # A new value of the input is a new key: not the cached image
    assert evaluator.render(buildPlan(after), 1).rgba().shape[1] == 5


def test_nodeInsideAnEnteredSubGraphReachesTheNodesOutside():
    graph, red, blue, grade, merge = constantGraph()
    flat = render(grade)
    subGraphNode, _ = collapseIntoSubGraph(graph, [grade])
    inner = SubGraphStack().enter(graph, subGraphNode)
    innerGrade = next(node for node in inner.nodes if node.nodeType == "Grade")
    plan = buildPlan(innerGrade)
    assert plan.keyOf(f"../{red.name}") is not None
    np.testing.assert_allclose(Evaluator().render(plan, 1).rgba(), flat)


def test_changeOutsideReachesTheNodesInside():
    """ The image of a node in a subgraph depends on the nodes outside connected to it (not cached alone). """
    from .test_liveCompositing import setColor
    graph, red, blue, grade, merge = constantGraph()
    subGraphNode, _ = collapseIntoSubGraph(graph, [grade])
    evaluator = Evaluator()
    first = evaluator.render(buildPlan(merge), 1).rgba()[0, 0].copy()
    setColor(red, r=0.25)
    graph.update()
    second = evaluator.render(buildPlan(merge), 1).rgba()[0, 0]
    assert not np.allclose(first, second)


def test_subGraphNodeIsEvaluatedThroughItsImageOutput():
    from meshroom.live import isEvaluable
    graph, red, blue, grade, merge = constantGraph()
    flat = render(grade)
    subGraphNode, _ = collapseIntoSubGraph(graph, [grade])
    assert isEvaluable(subGraphNode)
    plan = buildPlan(subGraphNode)
    assert plan.targetNode.name == f"{subGraphNode.name}/{grade.name}"
    np.testing.assert_allclose(Evaluator().render(plan, 1).rgba(), flat)


def test_compositingProjectAsTemplateBringsItsCompositing(tmp_path):
    from meshroom.core.subgraph.executor import readGraphData
    # A project of the compositing workspace: its compositing (nodes, chapters, ports) in its compositing_1 SubGraph
    inner = {"header": {"fileVersion": "2.1", "template": True}, "graph": {
        "Constant_1": {"nodeType": "Constant", "position": [0, 0], "inputs": {}},
        "Chapter_1": {"nodeType": "Chapter", "position": [0, -100], "inputs": {}},
        "GraphOutput_1": {"nodeType": "GraphOutput", "position": [300, 0], "inputs": {}},
    }}
    project = {"header": {"workspaceMode": "compositing", "fileVersion": "2.1"}, "graph": {
        "Write_1": {"nodeType": "Write", "position": [250, 0], "inputs": {"image": "{compositing_1.outs.output}"}},
        "compositing_1": {"nodeType": "SubGraph", "position": [0, 0], "inputs": {"subGraph": json.dumps(inner)}},
    }}
    path = tmp_path / "shot.mg"
    path.write_text(json.dumps(project))
    assert sorted(readGraphData(str(path))) == ["Chapter_1", "Constant_1", "GraphOutput_1"]
    # Another project (default workspace): all its nodes, as before
    project["header"].pop("workspaceMode")
    path.write_text(json.dumps(project))
    assert sorted(readGraphData(str(path))) == ["Write_1", "compositing_1"]


def test_importCompositingOfAProject(tmp_path):
    """ File > Import Compositing: the content of the compositing SubGraph of a project, without its ports. """
    from meshroom.core.graph import Graph
    from meshroom.core.subgraph.editing import readCompositing
    inner = {"header": {"fileVersion": "2.1", "template": True}, "graph": {
        "Grade_1": {"nodeType": "Grade", "position": [0, 0], "inputs": {}},
        "Blur_1": {"nodeType": "Blur", "position": [200, 0], "inputs": {"input": "{Grade_1.output}"}},
        "Chapter_1": {"nodeType": "Chapter", "position": [-50, -50], "inputs": {}},
        "GraphOutput_1": {"nodeType": "GraphOutput", "position": [400, 0], "inputs": {}},
    }}
    project = {"header": {"workspaceMode": "compositing", "fileVersion": "2.1"}, "graph": {
        "compositing_1": {"nodeType": "SubGraph", "position": [0, 0], "inputs": {"subGraph": json.dumps(inner)}},
    }}
    path = tmp_path / "comp.mg"
    path.write_text(json.dumps(project))
    imported = readCompositing(path)
    assert sorted(node.nodeType for node in imported.nodes) == ["Blur", "Chapter", "Grade"]
    target = Graph("target")
    target.addNewNode("Grade")
    nodes = target.importGraphContent(imported)
    assert len(nodes) == 3 and len(target.nodes) == 4
    blur = next(node for node in nodes if node.nodeType == "Blur")
    assert blur.input.inputRootLink.node.nodeType == "Grade" and blur.input.inputRootLink.node in nodes


def test_writeInsideTheCompositingSubGraph(tmp_path):
    """
    New Compositing: the Write is inside the compositing SubGraph, whose port is the written file. Running the
    subgraph writes the frames and gives the path of the file.
    """
    from meshroom.core.graph import Graph
    from meshroom.core.subgraph import run
    from meshroom.core.subgraph.ports import formatPortSpec
    from meshroom.live import imageIO
    from .test_liveCompositing import setColor
    graph = Graph("")
    constant = graph.addNewNode("Constant", width=4, height=2)
    setColor(constant, r=0.25, a=1.0)
    write = graph.addNewNode("Write", file=str(tmp_path / "render.tif"), dataType="float")
    graph.addEdge(constant.output, write.image)
    graphOutput = graph.addNewNode("GraphOutput", portSpec=formatPortSpec([("output", "File")]))
    graph.addEdge(write.output, graphOutput.ports.value.get("output"))
    result = run(graph.toDict())
    assert result == {"output": str(tmp_path / "render.tif")}
    image = imageIO.readImage(result["output"], "linear")
    assert abs(float(image.rgba()[0, 0, 0]) - 0.25) < 1e-6
