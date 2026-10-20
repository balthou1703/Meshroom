"""
Tests of the Houdini, Maya and Nuke contexts and nodes of the plugin, with fake hou / maya / nuke modules.
Run from the root of the plugin, with Meshroom importable: PYTHONPATH=/path/to/Meshroom pytest tests
"""
import json
import os

import pytest

from meshroom.core import desc, pluginManager
from meshroom.core.graph import Graph
from meshroom.core.plugins.base import NodeDescProvider
from meshroom.core.subgraph import Loop, run

PLUGIN_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def registerNodeDesc(nodeDesc):
    if not pluginManager.isNodeDescRegistered(nodeDesc.__name__):
        pluginManager._nodeDescProviders[nodeDesc.__name__] = NodeDescProvider(nodeDesc)


def unregisterNodeDesc(nodeDesc):
    pluginManager._nodeDescProviders.pop(nodeDesc.__name__, None)


def setup_module():
    import meshroom.core
    # The built-in nodes (SubGraph, GraphInput...) and this plugin, loaded as Meshroom loads it
    if not pluginManager.isNodeDescRegistered("SubGraph"):
        meshroom.core.initNodes()
    if not pluginManager.isNodeDescRegistered("HoudiniScene"):
        pluginManager.addPluginFromPath("mrDccPlugin", PLUGIN_FOLDER)
    for nodeDesc in (LiveAdd, TaskShotScene):
        registerNodeDesc(nodeDesc)


def test_pluginNodesAndContexts():
    from meshroom.core.subgraph import contextNames
    for nodeType in ("HoudiniScene", "HoudiniSave", "HoudiniSetParms", "HoudiniRopRender", "MayaScene", "MayaSave",
                     "NukeScene", "NukeSave"):
        assert pluginManager.isNodeDescRegistered(nodeType), nodeType
    assert {"houdini", "maya", "nuke"} <= set(contextNames())


class LiveAdd(desc.LiveNode):
    """ sum = a + b """
    __test__ = False
    inputs = [
        desc.LiveValue(name="a", label="A", description="", value=0),
        desc.LiveValue(name="b", label="B", description="", value=0),
    ]
    outputs = [
        desc.LiveValue(name="sum", label="Sum", description=""),
    ]

    def execute(self, ctx, inputs):
        return {"sum": inputs["a"] + inputs["b"]}


class TaskShotScene(desc.Node):
    """ A task node computing a value (like a pipeline node giving the scene of a shot). """
    __test__ = False
    runs = 0
    inputs = [
        desc.StringParam(name="shot", label="Shot", description="", value="sh010"),
    ]
    outputs = [
        desc.File(name="scene", label="Scene", description="", value=None),
    ]

    def process(self, node):
        TaskShotScene.runs += 1
        node.scene.value = f"/shots/{node.shot.value}.hip"


def port(node, anySetName, name):
    return node.attribute(anySetName).value.get(name)


class FakeParm:
    def __init__(self, value=None):
        self.value, self.expression, self.locked = value, None, False

    def set(self, value):
        self.value = value

    def setExpression(self, expression, language=None):
        self.expression = expression

    def isLocked(self):
        return self.locked

    def lock(self, locked):
        self.locked = locked

    def deleteAllKeyframes(self):
        self.expression = None

    def eval(self):
        return self.value

    def evalAtFrame(self, frame):
        return self.value.replace("$F4", f"{frame:04d}")


class FakeRop:
    def __init__(self, hou):
        self.hou, self.renders = hou, []
        self.parms = {"vm_picture": FakeParm("/render/beauty.$F4.exr")}

    def path(self):
        return "/out/mantra1"

    def parm(self, name):
        return self.parms.get(name)

    def render(self, **kwargs):
        self.renders.append(kwargs.get("frame_range"))

    def errors(self):
        return ()


class FakeHou:
    """ The parts of the hou module used by the Houdini live nodes. """

    class exprLanguage:
        Hscript = "hscript"

    def __init__(self):
        hou = self
        self.frames, self.loaded, self.saved = [], [], []
        self.parms = {"/obj/geo1/tx": FakeParm(0), "/obj/geo1/ry": FakeParm(0)}
        self.tuples = {}
        self.rop = FakeRop(self)

        class HipFile:
            current = "/scenes/untitled.hip"

            def path(self):
                return self.current

            def load(self, path, **kwargs):
                hou.loaded.append(path)
                self.current = path

            def save(self, path=None):
                self.current = path or self.current
                hou.saved.append(self.current)

            def clear(self, **kwargs):
                hou.loaded.append("clear")

        self.hipFile = HipFile()

    def parm(self, path):
        return self.parms.get(path)

    def parmTuple(self, path):
        return self.tuples.setdefault(path, FakeParm())

    def node(self, path):
        return self.rop if path == "/out/mantra1" else None

    def setFrame(self, frame):
        self.frames.append(frame)


@pytest.fixture
def fakeHou(monkeypatch):
    import sys
    nodeDescs = [BatchSetParm]
    for nodeDesc in nodeDescs:
        registerNodeDesc(nodeDesc)
    hou = FakeHou()
    monkeypatch.setitem(sys.modules, "hou", hou)
    yield hou
    for nodeDesc in nodeDescs:
        unregisterNodeDesc(nodeDesc)


def test_houdiniChain(fakeHou):
    graph = Graph("")
    graphInput = graph.addNewNode("GraphInput", portSpec="hip, offset")
    openScene = graph.addNewNode("HoudiniScene")
    setParms = graph.addNewNode("HoudiniSetParms", parms="/obj/geo1/ry := $F * 2\n/obj/geo1/t = (1, 2, 3)",
                                parmPath="/obj/geo1/tx")
    render = graph.addNewNode("HoudiniRopRender")
    graphOutput = graph.addNewNode("GraphOutput", portSpec="image")
    graph.addEdge(port(graphInput, "ports", "hip"), openScene.hipFile)
    graph.addEdge(port(graphInput, "ports", "offset"), setParms.value)
    graph.addEdge(openScene.scene, setParms.scene)
    graph.addEdge(setParms.sceneOut, render.scene)
    graph.addEdge(render.outputFile, port(graphOutput, "ports", "image"))

    # A Meshroom task: the scene is opened once (then it is the current scene), the frame follows the loop
    outputs = run(graph.toDict(), {"hip": "/scenes/shot.hip", "offset": 5}, loop=Loop("frames", first=1, last=3),
                  ownsSession=True)
    assert outputs == {"image": "/render/beauty.0003.exr"}
    assert fakeHou.loaded == ["/scenes/shot.hip", "clear"]
    assert fakeHou.frames == [1, 2, 3]
    assert fakeHou.rop.renders == [(1, 1), (2, 2), (3, 3)]
    assert fakeHou.parms["/obj/geo1/tx"].value == 5
    assert fakeHou.parms["/obj/geo1/ry"].expression == "$F * 2"
    assert fakeHou.tuples["/obj/geo1/t"].value == (1, 2, 3)


class BatchSetParm(desc.Node):
    """ A Houdini task node written as usual: it loads the scene, changes it, saves it. """
    __test__ = False
    pythonExecutable = "hython"
    inputs = [
        desc.File(name="scene", label="Scene", description="", value=""),
        desc.StringParam(name="parmPath", label="Parm", description="", value="/obj/geo1/tx"),
        desc.FloatParam(name="value", label="Value", description="", value=1.0),
    ]
    outputs = [
        desc.File(name="output", label="Output", description="", value=None),
    ]

    def process(self, node):
        import hou
        hou.hipFile.load(node.scene.value, suppress_save_prompt=True)
        hou.parm(node.parmPath.value).set(node.value.value)
        hou.hipFile.save(node.scene.value)
        node.output.value = node.scene.value


def houdiniSceneGraph():
    """ TaskShotScene -> HoudiniScene -> BatchSetParm (tx) -> BatchSetParm (ry) -> HoudiniSave """
    TaskShotScene.runs = 0
    graph = Graph("")
    shotScene = graph.addNewNode("TaskShotScene", shot="sh010")
    scene = graph.addNewNode("HoudiniScene")
    tx = graph.addNewNode("BatchSetParm", parmPath="/obj/geo1/tx", value=2.0)
    ry = graph.addNewNode("BatchSetParm", parmPath="/obj/geo1/ry", value=45.0)
    save = graph.addNewNode("HoudiniSave")
    graph.addEdge(shotScene.scene, scene.hipFile)
    graph.addEdge(scene.scene, tx.scene)
    graph.addEdge(tx.output, ry.scene)
    graph.addEdge(ry.output, save.scene)
    return graph


def test_houdiniMeshroomTask(fakeHou):
    """ In a Meshroom task: the pipeline node gives the scene, opened once, saved once at the end. """
    assert run(houdiniSceneGraph().toDict(), ownsSession=True) == {}
    assert TaskShotScene.runs == 1
    assert fakeHou.loaded == ["/shots/sh010.hip", "clear"]
    assert fakeHou.saved == ["/shots/sh010.hip"]
    assert fakeHou.parms["/obj/geo1/tx"].value == 2.0 and fakeHou.parms["/obj/geo1/ry"].value == 45.0
    # The real load / save are back after the run
    assert fakeHou.hipFile.load.__name__ == "load" and "HipFile" in type(fakeHou.hipFile).__name__


def test_houdiniSession(fakeHou):
    """ In a Houdini session: the open scene prevails, the pipeline node is not run, nothing is saved. """
    import meshroom.subgraph
    fakeHou.hipFile.current = "/shots/current.hip"
    meshroom.subgraph.run(houdiniSceneGraph().toDict())
    assert TaskShotScene.runs == 0
    assert fakeHou.loaded == [] and fakeHou.saved == []
    assert fakeHou.parms["/obj/geo1/tx"].value == 2.0 and fakeHou.parms["/obj/geo1/ry"].value == 45.0

    # Save In Houdini Session: saved once
    graph = houdiniSceneGraph()
    graph.node("HoudiniSave_1").saveInSession.value = True
    meshroom.subgraph.run(graph.toDict())
    assert fakeHou.saved == ["/shots/current.hip"]


def test_houdiniSessionReadsInputOfSkippedNode(fakeHou):
    """ A value typed in the pipeline node (the shot name) is read in memory, without running the node. """
    import meshroom.subgraph
    graph = houdiniSceneGraph()
    shotScene = graph.node("TaskShotScene_1")
    script = graph.addNewNode("LiveScript", inSpec="shot", outSpec="shot", code="outs['shot'] = ins['shot']")
    graphOutput = graph.addNewNode("GraphOutput", portSpec="shot")
    graph.addEdge(shotScene.shot, port(script, "ins", "shot"))
    graph.addEdge(port(script, "outs", "shot"), port(graphOutput, "ports", "shot"))
    assert meshroom.subgraph.run(graph.toDict()) == {"shot": "sh010"}
    assert TaskShotScene.runs == 0
    assert meshroom.subgraph.run(graph.toDict(), set={"TaskShotScene_1.shot": "sh020"}) == {"shot": "sh020"}
    assert TaskShotScene.runs == 0


def test_houdiniSessionSceneFile(fakeHou):
    """ "In Houdini Session: sceneFile": the pipeline node runs in the session and its scene is opened. """
    import meshroom.subgraph
    fakeHou.hipFile.current = "/shots/current.hip"
    graph = houdiniSceneGraph()
    graph.node("HoudiniScene_1").inSession.value = "sceneFile"
    meshroom.subgraph.run(graph.toDict())
    assert TaskShotScene.runs == 1
    assert fakeHou.loaded == ["/shots/sh010.hip"] and fakeHou.saved == []


def test_subGraphStartedLikeItsTaskNodes(fakeHou):
    """ A Houdini subgraph task is started with the interpreter and environment of its Houdini task nodes. """
    from types import SimpleNamespace
    project = Graph("project")
    node = project.addNewNode("SubGraph", subGraph=json.dumps(houdiniSceneGraph().serialize(asTemplate=True)))
    chunk = SimpleNamespace(node=node)
    assert node.nodeDesc.getPythonExecutable(chunk) == "hython"
    from meshroom.core import pluginManager
    assert node.nodeDesc.getProcessProvider(chunk) is pluginManager.getNodeDescProvider("BatchSetParm")


def test_loopRunsOnceWhatDoesNotDependOnIt(fakeHou):
    """ In a frame loop, the pipeline node and the scene run once; the nodes reading the frame at each frame. """
    graph = houdiniSceneGraph()
    graphInput = graph.addNewNode("GraphInput")
    graph.addEdge(graphInput.frame, graph.node("BatchSetParm_2").value)
    run(graph.toDict(), loop=Loop("frameRange", first=1001, last=1003), ownsSession=True)
    assert TaskShotScene.runs == 1
    assert fakeHou.loaded == ["/shots/sh010.hip", "clear"]
    assert fakeHou.parms["/obj/geo1/ry"].value == 1003
    # The save at the end of each frame (HoudiniSave, live node)
    assert fakeHou.saved == ["/shots/sh010.hip"] * 3


class FakeCmds:
    """ The parts of maya.cmds used by the Maya context. """

    def __init__(self):
        self.scene, self.opened, self.saved, self.times = "", [], [], []

    def about(self, **kwargs):
        return "2025"

    def file(self, *args, **kwargs):
        if kwargs.get("query"):
            return self.scene
        if kwargs.get("open"):
            self.opened.append(args[0])
            self.scene = args[0]
        elif kwargs.get("rename"):
            self.scene = kwargs["rename"]
        elif kwargs.get("save"):
            self.saved.append(self.scene)
        elif kwargs.get("new"):
            self.opened.append("new")
        return self.scene

    def currentTime(self, frame, edit=True):
        self.times.append(frame)


class FakeNuke:
    """ The parts of the nuke module used by the Nuke context. """

    def __init__(self):
        self.script, self.opened, self.saved, self.frames = "", [], [], []

    def scriptName(self):
        if not self.script:
            raise RuntimeError("no script")
        return self.script

    def scriptOpen(self, filename):
        self.opened.append(filename)
        self.script = filename

    def scriptSave(self, filename=None):
        self.saved.append(filename or self.script)

    def scriptSaveAs(self, filename=None, overwrite=0):
        self.script = filename
        self.saved.append(filename)

    def scriptClear(self):
        self.opened.append("clear")

    def frame(self, frame):
        self.frames.append(frame)


class MayaBatch(desc.Node):
    """ A Maya task node written as usual: opens the scene, changes it, saves it. """
    __test__ = False
    pythonExecutable = "mayapy"
    inputs = [desc.File(name="scene", label="Scene", description="", value="")]
    outputs = [desc.File(name="output", label="Output", description="", value=None)]

    def process(self, node):
        # Written for mayapy, like the nodes of the pipeline plugins
        import maya.standalone
        maya.standalone.initialize(name="python")
        from maya import cmds
        cmds.file(node.scene.value, open=True, force=True)
        cmds.file(save=True, force=True)
        maya.standalone.uninitialize()
        node.output.value = node.scene.value


class NukeBatch(desc.Node):
    """ A Nuke task node written as usual: opens the script, changes it, saves it. """
    __test__ = False
    pythonExecutable = "nuke -t"
    inputs = [desc.File(name="scene", label="Scene", description="", value="")]
    outputs = [desc.File(name="output", label="Output", description="", value=None)]

    def process(self, node):
        import nuke
        nuke.scriptOpen(node.scene.value)
        nuke.scriptSave()
        node.output.value = node.scene.value


@pytest.fixture
def fakeMayaNuke(monkeypatch):
    import sys
    import types
    nodeDescs = [MayaBatch, NukeBatch]
    for nodeDesc in nodeDescs:
        registerNodeDesc(nodeDesc)
    cmds, nuke = FakeCmds(), FakeNuke()
    maya = types.ModuleType("maya")
    maya.cmds = cmds
    standalone = types.ModuleType("maya.standalone")

    def initialize(name="python"):
        raise RuntimeError("maya.standalone.initialize() can not be call inside of Maya.")

    def uninitialize():
        raise RuntimeError("Maya uninitialized during the subgraph")

    standalone.initialize, standalone.uninitialize = initialize, uninitialize
    maya.standalone = standalone
    monkeypatch.setitem(sys.modules, "maya", maya)
    monkeypatch.setitem(sys.modules, "maya.cmds", cmds)
    monkeypatch.setitem(sys.modules, "maya.standalone", standalone)
    monkeypatch.setitem(sys.modules, "nuke", nuke)
    yield cmds, nuke
    for nodeDesc in nodeDescs:
        unregisterNodeDesc(nodeDesc)


def softwareGraph(software: str):
    """ TaskShotScene -> <Software>Scene -> batch node x2 -> <Software>Save """
    TaskShotScene.runs = 0
    graph = Graph("")
    shotScene = graph.addNewNode("TaskShotScene", shot="sh010")
    scene = graph.addNewNode(f"{software}Scene")
    first = graph.addNewNode(f"{software}Batch")
    second = graph.addNewNode(f"{software}Batch")
    save = graph.addNewNode(f"{software}Save")
    graph.addEdge(shotScene.scene, scene.hipFile)
    graph.addEdge(scene.scene, first.scene)
    graph.addEdge(first.output, second.scene)
    graph.addEdge(second.output, save.scene)
    return graph


def test_mayaAndNukeTasks(fakeMayaNuke):
    cmds, nuke = fakeMayaNuke
    run(softwareGraph("Maya").toDict(), ownsSession=True)
    assert cmds.opened == ["/shots/sh010.hip", "new"] and cmds.saved == ["/shots/sh010.hip"]
    run(softwareGraph("Nuke").toDict(), ownsSession=True)
    assert nuke.opened == ["clear", "/shots/sh010.hip", "clear"] and nuke.saved == ["/shots/sh010.hip"]
    assert TaskShotScene.runs == 1


def test_mayaAndNukeSessions(fakeMayaNuke):
    import meshroom.subgraph
    cmds, nuke = fakeMayaNuke
    cmds.scene, nuke.script = "/shots/open.ma", "/shots/open.nk"
    meshroom.subgraph.run(softwareGraph("Maya").toDict())
    meshroom.subgraph.run(softwareGraph("Nuke").toDict())
    assert TaskShotScene.runs == 0
    assert cmds.opened == [] and cmds.saved == [] and nuke.opened == [] and nuke.saved == []
    # The nodes initializing maya.standalone themselves ran (restored after the run)
    import maya.standalone
    assert maya.standalone.initialize.__name__ == "initialize"
    with pytest.raises(RuntimeError):
        maya.standalone.initialize()


def test_sessionSceneIsNeverReplacedByANode(fakeMayaNuke):
    """ In a Maya session, a task node opening another scene (e.g. the last version on disk) keeps the open one. """
    import meshroom.subgraph
    cmds, _ = fakeMayaNuke
    cmds.scene = "/shots/open.ma"
    graph = Graph("")
    graph.addNewNode("MayaBatch", scene="/shots/sh010-v002.ma")
    meshroom.subgraph.run(graph.toDict())
    assert cmds.opened == [] and cmds.scene == "/shots/open.ma"
    # In a Meshroom task the node opens it
    run(graph.toDict(), ownsSession=True)
    assert cmds.opened[0] == "/shots/sh010-v002.ma"
