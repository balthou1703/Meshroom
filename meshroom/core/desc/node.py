# desc/node.py

import enum
from inspect import getfile, getattr_static
from pathlib import Path
from typing import List
import logging
import shlex
import shutil
import sys
import signal
import subprocess
from collections import OrderedDict
from typing import Any, ClassVar, Mapping, Optional, Sequence, TYPE_CHECKING
import psutil

from meshroom import _MESHROOM_ROOT
from meshroom.common import BaseObject, Property
from meshroom.core import cgroup
from meshroom.core.desc.anySet import AnySet
from meshroom.core.utils import VERBOSE_LEVEL

from .computation import Level, StaticNodeSize
from .attribute import Attribute, BoolParam, ChoiceParam, ColorParam, File, Flow, IntParam, StringParam, ListAttribute

if TYPE_CHECKING:
    from meshroom.core.attribute import Attribute as CoreAttribute
    from meshroom.core.node import Node as CoreNode

if getattr(sys, "frozen", False):
    # In release mode, the "bin" path is one folder higher than the "_MESHROOM_ROOT".
    _MESHROOM_COMPUTE = (Path(_MESHROOM_ROOT).parent / "bin" / "meshroom_compute").as_posix()
else:
    _MESHROOM_COMPUTE = (Path(_MESHROOM_ROOT) / "bin" / "meshroom_compute").as_posix()

_MESHROOM_COMPUTE_DEPS = ["psutil"]


# Handle cleanup
class ExitCleanup:
    """
    Make sure we kill child subprocesses when the main process exits receive SIGTERM.
    """

    def __init__(self):
        self._subprocesses = []
        signal.signal(signal.SIGTERM, self.exit)

    def addSubprocess(self, process):
        logging.debug(f"[ExitCleanup] Register subprocess {process}")
        self._subprocesses.append(process)

    def exit(self, signum, frame):
        for proc in self._subprocesses:
            logging.debug(f"[ExitCleanup] Kill subprocess {proc}")
            try:
                if proc.is_running():
                    proc.terminate()
                    proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(0)

exitCleanup = ExitCleanup()


class MrNodeType(enum.Enum):
    NONE = enum.auto()
    BASENODE = enum.auto()
    NODE = enum.auto()
    COMMANDLINE = enum.auto()
    INIT = enum.auto()
    BACKDROP = enum.auto()
    IMAGEOP = enum.auto()
    LIVE = enum.auto()


class InternalAttributesFactory:
    BASIC = [
        StringParam(
            name="comment",
            label="Comments",
            description="User comments describing this specific node instance.\n"
                        "It is displayed in regular font in the invalidation/comment messages "
                        "tooltip.",
            value="",
            semantic="multiline",
            invalidate=False,
        ),
        StringParam(
            name="label",
            label="Node's Label",
            description="Customize the default label (to replace the technical name of the node "
                        "instance).",
            value="",
            invalidate=False,
        ),
        ChoiceParam(
            name="nodeDefaultLogLevel",
            label="Default Logging Level",
            description="Default logging level for the node (critical, error, warning, info, debug).",
            value="info",
            values=VERBOSE_LEVEL,
            invalidate=False,
        ),
        ColorParam(
            name="color",
            label="Color",
            description="Custom color for the node (SVG name or hexadecimal code).",
            value=lambda node: getattr(node.nodeDesc, "color", ""),
            invalidate=False,
        )
    ]

    INVALIDATION = [
        StringParam(
            name="invalidation",
            label="Invalidation Message",
            description="A message that will invalidate the node's output folder.\n"
                        "This is useful for development, we can invalidate the output of the node "
                        "when we modify the code.\n"
                        "It is displayed in bold font in the invalidation/comment messages "
                        "tooltip.",
            value="",
            semantic="multiline",
            advanced=True,
            uidIgnoreValue="",  # If the invalidation string is empty, it does not participate to the node's UID
        ),
    ]

    RESIZABLE = [
        IntParam(
            name="fontSize",
            label="Font Size",
            description="Size of the font used to display the comments.",
            value=12,
            range=(6, 100, 1),
            invalidate=False,
        ),
        ColorParam(
            name="fontColor",
            label="Font Color",
            description="Color of the font used to display the comments (SVG name or hexadecimal code).",
            value="",
            invalidate=False,
        ),
        IntParam(
            name="nodeWidth",
            label="Node Width",
            description="Width of the node in the graph editor.",
            value=600,
            range=None,
            invalidate=False,
            enabled=False,  # Hidden
        ),
        IntParam(
            name="nodeHeight",
            label="Node Height",
            description="Height of the node in the graph editor.",
            value=400,
            range=None,
            invalidate=False,
            enabled=False,  # Hidden
        ),
    ]

    # Live compositing nodes (ImageOpNode)
    COMPOSITING = [
        BoolParam(
            name="disable",
            label="Disable",
            description="Disable the node: its first image input is passed through unchanged.",
            value=False,
            invalidate=True,
        ),
        StringParam(
            name="animation",
            label="Animation",
            description="Keys of the animated parameters (JSON, see meshroom/live/animation.py).",
            value="",
            invalidate=True,
            # Not shown in the attribute editor (edited with the key buttons of the parameters).
            # Enabled: the keys are part of the uid (nothing animated: the uid is unchanged).
            uidIgnoreValue="",
            visible=False,
        ),
        StringParam(
            name="hiddenInputs",
            label="Hidden Inputs",
            description="Names of the inputs whose edges are hidden in the graph editor (UI state, key H).",
            value="",
            invalidate=False,
            # Not shown in the attribute editor
            enabled=False,
            visible=False,
        ),
        StringParam(
            name="splitVec2",
            label="Split 2D Values",
            description="Names of the linkable 2D values edited as separate x / y fields (UI state).",
            value="",
            invalidate=False,
            # Not shown in the attribute editor
            enabled=False,
            visible=False,
        ),
    ]

    FLOW_IN = [
        ListAttribute(
            name="flowInputs",
            label="Flow Inputs",
            description="Incoming flow connections to express node dependencies.",
            exposed=True,
            commandLineGroup="",
            elementDesc=Flow(
                name="flowInput",
                label="Flow Input",
                description="Incoming flow connection to express a dependency from another node.",
                exposed=True,
            ),
            uidIgnoreValueIfEmpty=True
        )
    ]

    FLOW_OUT = [
        Flow(
            name="flowOutput",
            label="Flow Output",
            description="Outgoing flow connection to express a dependency to another node.",
        )
    ]

    @classmethod
    def getInternalAttributes(cls, mrNodeType: MrNodeType) -> List[Attribute]:
        paramMap = {
            MrNodeType.NONE: cls.BASIC,
            MrNodeType.BASENODE: cls.INVALIDATION + cls.BASIC,
            MrNodeType.NODE: cls.INVALIDATION + cls.BASIC,
            MrNodeType.COMMANDLINE: cls.INVALIDATION + cls.BASIC,
            MrNodeType.INIT: cls.BASIC,
            MrNodeType.BACKDROP: cls.BASIC + cls.RESIZABLE,
            MrNodeType.IMAGEOP: cls.INVALIDATION + cls.BASIC + cls.COMPOSITING,
            MrNodeType.LIVE: cls.BASIC,
        }

        return paramMap.get(mrNodeType)

    @classmethod
    def getInternalFlowInputs(cls, mrNodeType: MrNodeType) -> List[Attribute]:
        """
        Return the list of internal input Flow attributes for a given node type.

        These are added to internal attributes so they appear as connection pins in
        the graph editor header while remaining separate from the regular attribute list.
        """
        if mrNodeType in (MrNodeType.BACKDROP, MrNodeType.IMAGEOP):
            return []
        return cls.FLOW_IN

    @classmethod
    def getInternalFlowOutputs(cls, mrNodeType: MrNodeType) -> List[Attribute]:
        """
        Return the list of internal output Flow attributes for a given node type.

        These are added to internal attributes so they appear as connection pins in
        the graph editor header while remaining separate from the regular attribute list.
        """
        if mrNodeType in (MrNodeType.BACKDROP, MrNodeType.IMAGEOP):
            return []
        return cls.FLOW_OUT


class NodeVersionType(enum.IntEnum):
    UNKNOWN = enum.auto()
    RELEASED = enum.auto()
    BETA = enum.auto()
    USER = enum.auto()


NodeVersionTypeEnum = type(
    "NodeVersionTypeEnum",
    (BaseObject,),
    {
        "__init__": lambda self, parent=None: BaseObject.__init__(self, parent),
        **{name: Property(int, lambda self, v=int(member): v, constant=True) for name, member in NodeVersionType.__members__.items()},
    },
)


class BaseNode(object):
    """
    """
    cpu = Level.NORMAL
    gpu = Level.NONE
    ram = Level.NORMAL
    packageName = ""
    color = ""
    _mrNodeType: MrNodeType = MrNodeType.BASENODE

    internalInputs = InternalAttributesFactory.getInternalAttributes(_mrNodeType)
    internalFlowInputs = InternalAttributesFactory.getInternalFlowInputs(_mrNodeType)
    internalFlowOutputs = InternalAttributesFactory.getInternalFlowOutputs(_mrNodeType)

    inputs = []
    outputs = []
    size = StaticNodeSize(1)
    parallelization = None
    documentation = ""
    category = "Other"
    nodeVersionType: NodeVersionType = NodeVersionType.UNKNOWN
    # The plugin NodeDescProvider that supplied this node
    provider = None
    # The plugin that contains this node
    plugin = None
    # Licenses required to run the plugin
    # Only used to select machines on the farm when the node is submitted
    _licenses = []

    def __init__(self):
        super(BaseNode, self).__init__()
        self.hasDynamicOutputAttribute = any(output.isDynamicValue or isinstance(output, AnySet) for output in self.outputs)
        self.sourceCodeFolder = Path(getfile(self.__class__)).parent.resolve().as_posix()

        mod = sys.modules[self.__class__.__module__]
        version = getattr(mod, "__version__", None) if mod else None

        if self.plugin and self.plugin.isUserPlugin:
            self.nodeVersionType = NodeVersionType.USER

        # If "version" is not defined, the node version type remains "UNKNOWN"
        if version and self.nodeVersionType == NodeVersionType.UNKNOWN:
            try:
                major = int(version.split(".")[0])
                if major < 1:
                    self.nodeVersionType = NodeVersionType.BETA
                else:
                    self.nodeVersionType = NodeVersionType.RELEASED
            except ValueError:
                pass

    def getMrNodeType(self):
        return self._mrNodeType

    @classmethod
    def getNodeInfo(cls):
        info = OrderedDict([
            ("module", cls.__module__),
            ("modulePath", cls.provider.path if cls.provider else ""),
        ])
        # > Info from the plugin module
        plugin_module = sys.modules.get(cls.__module__)
        if getattr(plugin_module, "__author__", None):
            info["author"] = plugin_module.__author__
        if getattr(plugin_module, "__license__", None):
            info["license"] = plugin_module.__license__
        if getattr(plugin_module, "__version__", None):
            info["version"] = plugin_module.__version__
        # > Overrides at the node-level
        if getattr(cls, "author", None):
            info["author"] = cls.author
        if getattr(cls, "version", None):
            info["version"] = cls.version
        # > Additional node information stored in a __nodeInfo__ parameter
        additionalNodeInfo = getattr(cls, "__nodeInfo__", None)
        if additionalNodeInfo:
            if isinstance(additionalNodeInfo, dict):
                additionalNodeInfo = list(additionalNodeInfo.items())
            for key, value in additionalNodeInfo:
                info[key] = value
        return info

    @classmethod
    def resolvedCpu(cls, node):
        """ Return the resolved CPU level for the given node instance.

        If `cpu` is a callable, it is called with the node instance as parameter.
        Otherwise, the static value is returned.
        """
        return cls.cpu(node) if callable(cls.cpu) else cls.cpu

    @classmethod
    def resolvedGpu(cls, node):
        """ Return the resolved GPU level for the given node instance.

        If `gpu` is a callable, it is called with the node instance as parameter.
        Otherwise, the static value is returned.
        """
        return cls.gpu(node) if callable(cls.gpu) else cls.gpu

    @classmethod
    def resolvedRam(cls, node):
        """ Return the resolved RAM level for the given node instance.

        If `ram` is a callable, it is called with the node instance as parameter.
        Otherwise, the static value is returned.
        """
        return cls.ram(node) if callable(cls.ram) else cls.ram

    @classmethod
    def resolvedSize(cls, node):
        """ Return the resolved size for the given node instance.

        If `size` is a callable, it is called with the node instance as parameter.
        If `size` is an integer, it is returned as-is.
        Objects with a `computeSize` method are supported for backward compatibility.
        """
        if callable(cls.size):
            return cls.size(node)
        if isinstance(cls.size, int):
            return cls.size
        # Backward compatibility with external size classes using computeSize instead of __call__
        if hasattr(cls.size, 'computeSize'):
            logging.warning(f"The plugin '{node.nodeType}' should use a callable instead of the deprecated method 'computeSize'.")
            return cls.size.computeSize(node)
        raise ValueError(f"{node.name} size attribute is invalid")

    def upgradeAttributeValues(self, attrValues, fromVersion):
        return attrValues

    @classmethod
    def onNodeCreated(cls, node):
        """
        Called after a node instance created from this node descriptor has been added to a Graph.
        """
        pass

    @classmethod
    def ignoredInputs(cls, node, ctx) -> tuple:
        """
        The inputs the node does not read when it runs in a subgraph with the run context `ctx` (see
        meshroom.core.subgraph): the nodes only feeding them are not run. E.g. a node giving the scene of
        a software ignores the path computed by Meshroom when it runs in an open session of that software.
        """
        return ()

    @classmethod
    def initDynamicAttributes(cls, node):
        """
        Called when a node instance has created its attributes from this descriptor, before its links are
        resolved (also while a graph is loaded): add the children of its AnySet attributes that depend on
        its input values, so that links to them can be resolved.
        """
        pass

    @classmethod
    def update(cls, node):
        """ Method call before node's internal update on invalidation.

        Args:
            node: the BaseNode instance being updated
        See Also:
            BaseNode.updateInternals
        """
        pass

    @classmethod
    def postUpdate(cls, node):
        """ Method call after node's internal update on invalidation.

        Args:
            node: the BaseNode instance being updated
        See Also:
            NodeBase.updateInternals
        """
        pass

    def preprocess(self, node):
        """ Gets invoked just before the processChunk method for the node.

        Args:
            node: The BaseNode instance about to be processed.
        """
        pass

    @property
    def hasPreprocess(self):
        """ Returns True if the class has a preprocess """
        return type(self).preprocess is not BaseNode.preprocess

    def postprocess(self, node):
        """ Gets invoked after the processChunk method for the node.

        Args:
            node: The BaseNode instance which is processed.
        """
        pass

    @property
    def hasPostprocess(self):
        """ Returns True if the class has a postprocess """
        return type(self).postprocess is not BaseNode.postprocess

    def process(self, node):
        raise NotImplementedError(f'No process implementation on node: "{node.name}"')

    def processChunk(self, chunk):
        if self.parallelization is None:
            self.process(chunk.node)
        else:
            raise NotImplementedError(f'No process implementation on node: "{chunk.node.name}"')

    def executeChunkCommandLine(self, chunk, cmd, env=None):
        try:
            with open(chunk.getLogFile(), 'a') as logF:
                chunk.status.commandLine = cmd
                chunk.saveStatusFile()
                cmdList = shlex.split(cmd)
                # Resolve executable to full path
                prog = shutil.which(cmdList[0], path=env.get("PATH") if env else None)

                print(f"Starting Process for '{chunk.node.name}'")
                print(f" - commandLine: {cmd}")
                print(f" - logFile: {chunk.getLogFile()}")
                if prog:
                    cmdList[0] = Path(prog).as_posix()
                    print(f" - command full path: {cmdList[0]}")

                # Change the process group to avoid Meshroom main process being killed if the
                # subprocess gets terminated by the user or an Out Of Memory (OOM kill).
                if sys.platform == "win32":
                    from subprocess import CREATE_NEW_PROCESS_GROUP
                    platformArgs = {"creationflags": CREATE_NEW_PROCESS_GROUP}
                    # Note: DETACHED_PROCESS means fully detached process.
                    # We do not want a fully detached process to ensure that if Meshroom is killed,
                    # the subprocesses are killed too.
                else:
                    platformArgs = {"start_new_session": True}
                    # Note: "preexec_fn"=os.setsid is the old way before python-3.2

                chunk.subprocess = psutil.Popen(
                    cmdList,
                    stdout=logF,
                    stderr=logF,
                    cwd=chunk.node.internalFolder,
                    env=env,
                    text=True,
                    **platformArgs,
                )
                exitCleanup.addSubprocess(chunk.subprocess)

                if hasattr(chunk, "statThread"):
                    # We only have a statThread if the node is running in the current process
                    # and not in a dedicated environment/process.
                    chunk.statThread.proc = chunk.subprocess

                stdout, stderr = chunk.subprocess.communicate()

                chunk.status.returnCode = chunk.subprocess.returncode

                if chunk.subprocess.returncode and chunk.subprocess.returncode < 0:
                    signal_num = -chunk.subprocess.returncode
                    logF.write(f"Process was killed by signal: {signal_num}")
                    try:
                        status = chunk.subprocess.status()
                        logF.write(f"Process status: {status}")
                    except Exception:
                        pass

            if chunk.subprocess.returncode != 0:
                with open(chunk.getLogFile(), "r") as logF:
                    logContent = "".join(logF.readlines())
                raise RuntimeError(f'Error on node "{chunk.name}":\nLog:\n{logContent}')
        finally:
            chunk.subprocess = None

    def stopProcess(self, chunk):
        # The same node could exists several times in the graph and
        # only one would have the running subprocess; ignore all others
        if not chunk.subprocess:
            logging.warning(f"[{chunk.node.name}] stopProcess: no subprocess")
            return

        # Retrieve process tree
        processes = chunk.subprocess.children(recursive=True) + [chunk.subprocess]
        logging.debug(f"[{chunk.node.name}] Processes to stop: {len(processes)}")
        for process in processes:
            try:
                # With terminate, the process has a chance to handle cleanup
                process.terminate()
            except psutil.NoSuchProcess:
                pass

        # If it is still running, force kill it
        for process in processes:
            try:
                # Use is_running() instead of poll() as we use a psutil.Process object
                if process.is_running():  # Check if process is still alive
                    process.kill()  # Forcefully kill it
            except psutil.NoSuchProcess:
                logging.info(f"[{chunk.node.name}] Process already terminated.")
            except psutil.AccessDenied:
                logging.info(f"[{chunk.node.name}] Permission denied to kill the process.")


class InitNode(BaseNode):
    """
    Node that does not need to be processed, it is just a placeholder for initialization.
    """
    _mrNodeType: MrNodeType = MrNodeType.INIT
    internalInputs = InternalAttributesFactory.getInternalAttributes(_mrNodeType)
    internalFlowInputs = InternalAttributesFactory.getInternalFlowInputs(_mrNodeType)
    internalFlowOutputs = InternalAttributesFactory.getInternalFlowOutputs(_mrNodeType)

    def __init__(self):
        super(InitNode, self).__init__()

    def getMrNodeType(self):
        return self._mrNodeType

    def processChunk(self, chunk):
        pass

    def process(self, node):
        pass

class BackdropNode(BaseNode):
    """
    Node that does not need to be processed, it is just a placeholder for grouping other nodes.
    """
    _mrNodeType: MrNodeType = MrNodeType.BACKDROP
    internalInputs = InternalAttributesFactory.getInternalAttributes(_mrNodeType)
    internalFlowInputs = InternalAttributesFactory.getInternalFlowInputs(_mrNodeType)
    internalFlowOutputs = InternalAttributesFactory.getInternalFlowOutputs(_mrNodeType)

    def __init__(self):
        super(BackdropNode, self).__init__()

    def getMrNodeType(self):
        return self._mrNodeType

    def processChunk(self, chunk):
        pass

    def process(self, node):
        pass


class LiveNode(BaseNode):
    """
    Node run in memory inside a subgraph (see meshroom.core.subgraph): no chunks, no cache folder, no
    status. Its `execute` receives the values of its inputs, any Python object for those connected to
    other live nodes, and returns the values of its outputs. A chain of live nodes runs in one process and
    one context (Python, a software session...), as fast as running their scripts one after the other.

    Live nodes are only run inside a SubGraph node (or by meshroom.core.subgraph.run): they are not
    computed on their own.
    """
    _mrNodeType: MrNodeType = MrNodeType.LIVE
    internalInputs = InternalAttributesFactory.getInternalAttributes(_mrNodeType)
    internalFlowInputs = InternalAttributesFactory.getInternalFlowInputs(_mrNodeType)
    internalFlowOutputs = InternalAttributesFactory.getInternalFlowOutputs(_mrNodeType)

    # Contexts the node can run in (see meshroom.core.subgraph.contexts), "any" for all of them
    context = "any"
    # Whether the node runs again at each step of a loop of its subgraph (it can read ctx.frame); False for a
    # node only depending on its inputs (it then runs again only if one of them depends on the loop)
    runsEachStep = True

    def __init__(self):
        super(LiveNode, self).__init__()

    def getMrNodeType(self):
        return self._mrNodeType

    @classmethod
    def contexts(cls):
        """ The names of the contexts the node can run in. """
        return (cls.context,) if isinstance(cls.context, str) else tuple(cls.context)

    def execute(self, ctx, inputs: dict) -> dict:
        """
        Run the node.

        Args:
            ctx: the meshroom.core.subgraph.RunContext: `ctx.context` (the session of the software),
                 `ctx.iteration`, `ctx.frame`, `ctx.loop`, `ctx.variables` (shared by the nodes of the run),
                 `ctx.logger`, `ctx.checkCancelled()`.
            inputs: {input name: value}; a connected input gets the value returned by the upstream node.

        Returns:
            {output name: value}
        """
        return {}

    def processChunk(self, chunk):
        pass

    def process(self, node):
        pass


class ImageOpNode(BaseNode):
    """
    Live compositing node, evaluated in memory on demand (pull model) by
    `meshroom.live.Evaluator`, frame by frame, instead of being computed in batch.

    It has no chunks, no cache folder and no flow pins: it is not computable in batch mode.
    Its image inputs and output are `desc.Image` attributes; the other inputs are its parameters.

    Subclasses implement `render` and, when the output format or frame range differs from
    the first input, `getInfo`. See docs/live-compositing.md.
    """
    _mrNodeType: MrNodeType = MrNodeType.IMAGEOP
    internalInputs = InternalAttributesFactory.getInternalAttributes(_mrNodeType)
    internalFlowInputs = InternalAttributesFactory.getInternalFlowInputs(_mrNodeType)
    internalFlowOutputs = InternalAttributesFactory.getInternalFlowOutputs(_mrNodeType)
    category = "Compositing"

    def __init__(self):
        super(ImageOpNode, self).__init__()
        # Stream outputs (images, cameras...) hold no value: they are not dynamic outputs to load from a values file
        from .attribute import Stream
        self.hasDynamicOutputAttribute = any(
            (output.isDynamicValue and not isinstance(output, Stream)) or isinstance(output, AnySet)
            for output in self.outputs)

    def __init_subclass__(cls, **kwargs):
        """ Every compositing node gets a "matrix" output, not exposed: the transform it applies (see transformMatrix). """
        super().__init_subclass__(**kwargs)
        from .attribute import Matrix
        outputs = list(cls.__dict__.get("outputs", cls.outputs) or [])
        if not any(output.name == "matrix" for output in outputs):
            outputs.append(Matrix(
                name="matrix", label="Matrix", exposed=False,
                description="Transform applied by the node at the current frame (4x4 matrix): the transform of a camera or a "
                            "geometry, or the 3x3 transform of an image in pixels (Nuke coordinates), embedded in 4x4; "
                            "identity for the nodes that do not move the image."))
            cls.outputs = outputs

    def getMrNodeType(self):
        return self._mrNodeType

    @classmethod
    def imageInputs(cls):
        """ Return the descriptions of the image inputs, in declaration order. """
        from .attribute import Image
        return [attr for attr in cls.inputs if isinstance(attr, Image)]

    # A time-based node (TimeOffset, FrameHold, Retime...) renders the frames of its inputs it needs
    # itself, with ctx.renderInput(name, frame): its inputs are not rendered at the current frame.
    timeBased = False
    # A node only passing its first input through (Dot, Clone): ctx.inputValues() looks through it
    passThroughValues = False
    # Image inputs only read for the values of their node (ctx.inputValues, e.g. the tracks of a
    # Tracker) or by a UI operation: neither rendered nor asked for their ImageInfo
    valuesOnlyInputs = ()

    # Each output pixel only depends on the input pixels at the same place (color nodes, merges...):
    # a region of interest is passed as is to the inputs (see inputRegions)
    pointwise = False

    def transformMatrix(self, ctx):
        """
        The transform applied by the node at the frame of `ctx`, given by its "matrix" output: a 3x3 matrix (2D
        transform of the image in pixels, Nuke coordinates), a 4x4 matrix (3D), or None (identity).
        """
        return None

    def computeData(self, ctx, outputName: str):
        """
        Compute the value of a stream output other than an image at the frame of `ctx` (a camera, a geometry, a
        4x4 matrix, see meshroom/live/scene3d.py). The default gives the "matrix" output from transformMatrix.
        The inputs of these types are read with ctx.inputData(name).
        """
        if outputName == "matrix":
            from meshroom.live import scene3d
            return scene3d.toMatrix4(self.transformMatrix(ctx))
        raise NotImplementedError(f"No value for the output '{outputName}' of {self.__class__.__name__}.")

    def disabledData(self, ctx, outputName: str):
        """
        The value of a stream output (other than an image) of the node when it is disabled: an input of the same
        type passed through, or the identity for a matrix. Raises RuntimeError if there is none.
        """
        from .attribute import Stream
        streamType = next((output.streamType for output in self.outputs if output.name == outputName), "")
        if streamType == "matrix":
            from meshroom.live import scene3d
            return scene3d.identity()
        for attr in self.inputs:
            if isinstance(attr, Stream) and attr.streamType == streamType and ctx.hasInput(attr.name):
                return ctx.inputData(attr.name)
        raise RuntimeError("The node is disabled.")

    def bakeStream(self, ctx, inputName: str, values: dict):
        """
        "Disconnect and Bake" of a stream input (e.g. a matrix): the parameters reproducing the values the input had
        at each frame (`values`: {frame: value}). Returns {parameter name: [[frame, value], ...] keys, or a value}
        (see meshroom/live/animation.py), or None when the input cannot be baked into parameters.
        """
        return None

    def planContext(self, node):
        """
        Values outside of the node's own attributes that its result depends on (e.g. the parameters of
        other nodes read by a Switch expression), snapshotted with the branch: a JSON-like dict, or None.
        Called on the thread owning the graph; the render reads it with `ctx.planContext`. It is part of
        the cache key of the node and of the nodes downstream. None by default.
        """
        return None

    @staticmethod
    def hiddenInputsAfterRemoval(hiddenInputs: str, listName: str, index: int) -> str:
        """
        The "hiddenInputs" value (comma-separated input names relative to the node, e.g. "inputs[2]")
        once the element `index` of the list attribute `listName` is removed: its own hidden edge is
        forgotten and the hidden elements after it move down by one, like the elements themselves.
        """
        import re
        pattern = re.compile(r"^" + re.escape(listName) + r"\[(\d+)\](.*)$")
        names = []
        for name in (hiddenInputs or "").split(","):
            match = pattern.match(name)
            if match:
                position = int(match.group(1))
                if position == index:
                    continue
                if position > index:
                    name = f"{listName}[{position - 1}]{match.group(2)}"
            if name:
                names.append(name)
        return ",".join(names)

    def inputRegions(self, ctx, region):
        """
        Return the regions of the image inputs needed to render `region` of the output (region of
        interest, Nuke coordinates): {input name: BBox} (an input not listed, or None: the whole input).
        By default, pointwise nodes need the same region of their inputs, the others their whole inputs.
        """
        from .attribute import Image
        if not self.pointwise:
            return None
        return {attr.name: region for attr in self.inputs if isinstance(attr, Image)}

    def infoOnlyInputs(self, ctx):
        """
        Return the names of the image inputs only used for their information (format, frame range,
        read with ctx.inputInfo), which do not need to be rendered.
        """
        from .attribute import Image
        return [attr.name for attr in self.inputs if isinstance(attr, Image) and attr.infoOnly]

    def isIdentity(self, ctx):
        """
        Return the name of the image input to pass through unchanged when the parameters make the
        node a no-op (e.g. a Blur of size 0), or None. A pass-through costs no computation and no memory.
        """
        return None

    def getInfo(self, ctx, inputsInfo):
        """
        Return the `meshroom.live.ImageInfo` (format, frame range) of the node output.

        Args:
            ctx: the `meshroom.live.RenderContext` of the request.
            inputsInfo: dict {image input name: ImageInfo or None if disconnected}.

        Returns:
            The info of the first connected image input by default.
        """
        for attr in self.imageInputs():
            info = inputsInfo.get(attr.name)
            if info is not None:
                return info
        raise RuntimeError("No connected image input: the node cannot determine its format.")

    def render(self, ctx, inputs):
        """
        Compute one frame.

        Args:
            ctx: the `meshroom.live.RenderContext` of the request (frame, parameter values).
            inputs: dict {image input name: ImageBuffer or None if disconnected}.

        Returns:
            meshroom.live.ImageBuffer: the output image.
        """
        raise NotImplementedError(f"No render implementation on node type: {self.__class__.__name__}")

    def processChunk(self, chunk):
        pass

    def process(self, node):
        pass


class Node(BaseNode):
    pythonExecutable = "python"
    _mrNodeType: MrNodeType = MrNodeType.NODE

    def __init__(self):
        super(Node, self).__init__()

    def getMrNodeType(self):
        return self._mrNodeType

    def getPythonExecutable(self, chunk) -> str:
        """ The Python interpreter running meshroom_compute for a chunk (pythonExecutable by default). """
        return self.pythonExecutable

    def getProcessProvider(self, chunk):
        """
        The node descriptor provider whose process environment (runtime environment, command prefix and
        suffix) runs meshroom_compute for a chunk: the one of the node by default.
        """
        return chunk.node.nodeDesc.provider

    def processChunkInEnvironment(self, chunk):
        meshroomComputeCmd = f"{self.getPythonExecutable(chunk)} {_MESHROOM_COMPUTE}" + \
                             f" \"{chunk.node.graph.filepath}\" --node {chunk.node.name}" + \
                              " --extern --inCurrentEnv"
        if chunk.isPreprocess:
            meshroomComputeCmd += f" --preprocess"
        elif chunk.isPostprocess:
            meshroomComputeCmd += f" --postprocess"
        elif len(chunk.node.getChunks()) >= 1:
            meshroomComputeCmd += f" --iteration {chunk.range.iteration}"

        provider = self.getProcessProvider(chunk)
        runtimeEnv = provider.runtimeEnv
        cmdPrefix = provider.commandPrefix
        cmdSuffix = provider.commandSuffix
        self.executeChunkCommandLine(chunk, cmdPrefix + meshroomComputeCmd + cmdSuffix,
                                     env=runtimeEnv)


class CommandLineNode(BaseNode):
    """
    """
    commandLine = ""  # need to be defined on the node
    parallelization = None
    commandLineRange = ""
    _mrNodeType: MrNodeType = MrNodeType.COMMANDLINE

    def __init__(self):
        super(CommandLineNode, self).__init__()

    def getMrNodeType(self):
        return self._mrNodeType

    def buildCommandLine(self, chunk) -> str:
        cmdLineVars = chunk.node.createCmdLineVars()
        cmdPrefix = ""
        cmdSuffix = ""
        if chunk.node.nodeDesc.provider:
            cmdPrefix = chunk.node.nodeDesc.provider.commandPrefix
            cmdSuffix = chunk.node.nodeDesc.provider.commandSuffix
        if chunk.node.isParallelized and chunk.node.size > 1:
            cmdSuffix = " " + self.commandLineRange.format(**chunk.range.toDict()) + " " + cmdSuffix

        # In the case of a lambda, we want a single "node" argument and not the node descriptor "self".
        # Therefore, we use getattr_static to retrieve the raw lambda instead of a bound method, which
        # would impose "self" as the first argument if we accessed "self.commandLine".
        commandLineValue = getattr_static(self, 'commandLine')
        if callable(commandLineValue):
            cmd = commandLineValue(chunk.node)
        else:
            cmd = commandLineValue.format(**chunk.node._expVars, **chunk.node._staticExpVars, **cmdLineVars)
        return cmdPrefix + cmd + cmdSuffix

    def processChunk(self, chunk):
        cmd = self.buildCommandLine(chunk)
        runtimeEnv = chunk.node.nodeDesc.provider.runtimeEnv
        self.executeChunkCommandLine(chunk, cmd, env=runtimeEnv)


# Specific command line node for AliceVision apps
class AVCommandLineNode(CommandLineNode):

    cgroupParsed = False
    cmdMem = ""
    cmdCore = ""

    def __init__(self):
        super(AVCommandLineNode, self).__init__()

        if AVCommandLineNode.cgroupParsed is False:

            AVCommandLineNode.cmdMem = ""
            memSize = cgroup.getCgroupMemorySize()
            if memSize > 0:
                AVCommandLineNode.cmdMem = f" --maxMemory={memSize}"

            AVCommandLineNode.cmdCore = ""
            coresCount = cgroup.getCgroupCpuCount()
            if coresCount > 0:
                AVCommandLineNode.cmdCore = f" --maxCores={coresCount}"

            AVCommandLineNode.cgroupParsed = True

    def buildCommandLine(self, chunk) -> str:
        commandLineString = super(AVCommandLineNode, self).buildCommandLine(chunk)

        return commandLineString + AVCommandLineNode.cmdMem + AVCommandLineNode.cmdCore


class IONode(object):
    def __init__(self):
        super(IONode, self).__init__()

    def resetAttributes(self, node: "CoreNode", attributeNames: Sequence[str]) -> None:
        """
        Reset the values of the provided attributes for a node.

        Args:
            node (Node): the node whose attributes are to be reset
            attributeNames (list): the list containing the names of the attributes to reset
        """
        for attrName in attributeNames:
            if node.hasAttribute(attrName):
                node.attribute(attrName).resetToDefaultValue()

    def extendAttributes(self, node: "CoreNode", attributesDict: Mapping[str, Any]) -> None:
        """
        Extend the values of the provided attributes for a node.

        Args:
            node (Node): the node whose attributes are to be extended
            attributesDict (dict): the dictionary containing the attributes' names (as keys) and the
                                   values to extend with
        """
        for attr in attributesDict.keys():
            if node.hasAttribute(attr):
                node.attribute(attr).extend(attributesDict[attr])

    def setAttributes(self, node: "CoreNode", attributesDict: Mapping[str, Any]) -> None:
        """
        Set the values of the provided attributes for a node.

        Args:
            node (Node): the node whose attributes are to be extended
            attributesDict (dict): the dictionary containing the attributes' names (as keys) and the
                                   values to set
        """
        for attr in attributesDict:
            if node.hasAttribute(attr):
                node.attribute(attr).value = attributesDict[attr]


class InputNode(IONode):
    def __init__(self):
        super(InputNode, self).__init__()

    def initialize(self, node: "CoreNode", inputs: Sequence[str], recursiveInputs: Sequence[str]) -> None:
        """
        Initialize the attributes that are needed for a node to start running.

        Args:
            node (Node): the node whose attributes must be initialized
            inputs (list): the user-provided list of input files/directories
            recursiveInputs (list): the user-provided list of input directories to search
                                    recursively for images
        """
        pass


class OutputNode(IONode):
    outputAttributes: ClassVar[Optional[Sequence[str]]] = None
    outputAttribute: ClassVar[str] = "output"

    def __init__(self):
        super(OutputNode, self).__init__()

    def getOutputAttributes(self, node: "CoreNode") -> List["CoreAttribute"]:
        """
        Return the attributes explicitly exposed by this output node.

        If "outputAttributes" is not set, "outputAttribute" is used as a
        backward-compatible fallback.
        """
        if self.outputAttributes is None:
            attributeNames: Sequence[str] = [self.outputAttribute] if self.outputAttribute else []
            return [node.attribute(attrName) for attrName in attributeNames if node.hasAttribute(attrName)]
        else:
            attributeNames = self.outputAttributes

        attributes: List["CoreAttribute"] = []
        for attrName in attributeNames:
            if not node.hasAttribute(attrName):
                raise AttributeError(f"'{node.name}.{attrName}' is not an output node attribute.")
            attributes.append(node.attribute(attrName))
        return attributes

    def getOutputFolderAttributes(self, node: "CoreNode") -> List["CoreAttribute"]:
        """
        Return the exposed File attributes managed as output folders.
        """
        return [attr for attr in self.getOutputAttributes(node) if isinstance(attr.desc, File)]

    def setOutputAttribute(self, node: "CoreNode", attributeName: str, value: Any) -> "CoreAttribute":
        """
        Set an explicitly exposed output attribute.

        Args:
            node (Node): the node whose exposed output attribute must be initialized
            attributeName (str): the exposed output attribute name
            value: the value to set on the exposed output attribute
        """
        for attr in self.getOutputAttributes(node):
            if attr.name == attributeName:
                self.setAttributes(node, {attributeName: value})
                return attr
        raise AttributeError(f"'{node.name}.{attributeName}' is not an exposed output attribute.")

    def setOutputFolder(self, node: "CoreNode", outputFolder: str) -> None:
        """
        Initialize the exposed File output attributes of a node.

        Args:
            node (Node): the node whose output folder must be initialized
            outputFolder (str): the output folder to set
        """
        outputFolderAttributes = self.getOutputFolderAttributes(node)
        if not outputFolderAttributes:
            raise AttributeError(f"'{node.name}' has no exposed File output attribute.")

        for attr in outputFolderAttributes:
            self.setAttributes(node, {attr.name: outputFolder})
