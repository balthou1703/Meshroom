"""Tests of the tree layout of the graph editor (meshroom.core.graphLayout)."""
import os

import meshroom
from meshroom.core import graphLayout, pluginManager
from meshroom.core.graph import Graph

_COMPOSITING_FOLDER = os.path.join(os.path.dirname(meshroom.__file__), "nodes", "compositing")
if not pluginManager.isNodeDescRegistered("Merge"):
    pluginManager.addPluginFromBuiltInFolder("compositing", _COMPOSITING_FOLDER)


def _layout(graph, nodes):
    edges = graphLayout.layoutEdges(graph, nodes)
    return graphLayout.treeLayout(nodes, edges, lambda node: (100.0, 50.0), (0.0, 0.0),
                                  lambda node: (node.x, node.y), columnSpacing=20.0, rowSpacing=10.0)


def test_mergeInputsBOnTopAOnBottom():
    graph = Graph("layout")
    plate = graph.addNewNode("Constant")
    grade = graph.addNewNode("Grade")
    mask = graph.addNewNode("Constant")
    background = graph.addNewNode("CheckerBoard")
    merge = graph.addNewNode("Merge")
    graph.addEdge(plate.output, grade.input)
    graph.addEdge(mask.output, grade.mask)
    graph.addEdge(background.output, merge.B)
    graph.addEdge(grade.output, merge.A)
    positions = _layout(graph, [plate, grade, mask, background, merge])
    # Left to right: each input just left of the node using it
    assert positions[merge][0] == 240.0
    assert positions[grade][0] == positions[background][0] == 120.0
    assert positions[plate][0] == positions[mask][0] == 0.0
    # B on the row of the Merge, A below; the Grade inputs: image on its row, mask below
    assert positions[background][1] == positions[merge][1] == 0.0
    assert positions[grade][1] == 60.0
    assert positions[plate][1] == 60.0 and positions[mask][1] == 120.0


def test_sharedInputAndSeparateTrees():
    graph = Graph("layout")
    read = graph.addNewNode("Constant")
    blurA = graph.addNewNode("Blur")
    blurB = graph.addNewNode("Blur")
    other = graph.addNewNode("Constant")
    graph.addEdge(read.output, blurA.input)
    graph.addEdge(read.output, blurB.input)
    blurB.position = meshroom.core.node.Position(0, 500)
    other.position = meshroom.core.node.Position(0, 900)
    positions = _layout(graph, [read, blurA, blurB, other])
    # The shared input is placed once, with the first output (the highest one)
    assert positions[read] == (0.0, 0.0) and positions[blurA] == (120.0, 0.0)
    assert positions[blurB][1] > positions[blurA][1]
    assert positions[other][1] > positions[blurB][1]
    # Links between parameters do not count
    edges = graphLayout.layoutEdges(graph, [read, blurA, blurB, other])
    assert len(edges) == 2
