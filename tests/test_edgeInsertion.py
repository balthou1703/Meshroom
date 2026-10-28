"""A node inserted on an edge of the graph editor (dropped on it with Ctrl held): meshroom/core/edgeInsertion.py."""

import os

import meshroom
from meshroom.core import pluginManager
from meshroom.core.edgeInsertion import insertionAttributes
from meshroom.core.graph import Graph

_COMPOSITING_FOLDER = os.path.join(os.path.dirname(meshroom.__file__), "nodes", "compositing")
if not pluginManager.isNodeDescRegistered("Grade"):
    pluginManager.addPluginFromBuiltInFolder("compositing", _COMPOSITING_FOLDER)


def test_mainInputAndOutput():
    graph = Graph("")
    source, target = graph.addNewNode("Constant"), graph.addNewNode("Merge")
    graph.addEdge(source.output, target.B)
    blur = graph.addNewNode("Blur")
    nodeInput, nodeOutput = insertionAttributes(blur, source.output, target.B)
    assert (nodeInput.name, nodeOutput.name) == ("input", "output")


def test_firstCompatibleInputWhenNoMainOne():
    graph = Graph("")
    source, target = graph.addNewNode("Constant"), graph.addNewNode("Grade")
    graph.addEdge(source.output, target.input)
    merge = graph.addNewNode("Merge")
    nodeInput, nodeOutput = insertionAttributes(merge, source.output, target.input)
    # The first image input of the Merge (B, as in Nuke), a free one first
    assert nodeInput.name == "B" and nodeOutput.name == "output"
    other = graph.addNewNode("Constant")
    graph.addEdge(other.output, merge.B)
    assert insertionAttributes(merge, source.output, target.input)[0].name == "A"


def test_notOnItsOwnEdgesNorMakingACycle():
    graph = Graph("")
    a, b, c = graph.addNewNode("Constant"), graph.addNewNode("Grade"), graph.addNewNode("Grade")
    graph.addEdge(a.output, b.input)
    graph.addEdge(b.output, c.input)
    assert insertionAttributes(b, a.output, b.input) is None
    # c depends on the edge a -> b: inserting it there would make a cycle
    assert insertionAttributes(c, a.output, b.input) is None
    # No compatible input: a camera stream cannot go through a Grade
    camera = graph.addNewNode("Camera")
    resection = graph.addNewNode("Resection")
    graph.addEdge(camera.output, resection.camera)
    assert insertionAttributes(graph.addNewNode("Grade"), camera.output, resection.camera) is None
