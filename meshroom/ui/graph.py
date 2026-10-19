#!/usr/bin/env python
from collections.abc import Iterable
import logging
import os
import re
import json
from enum import Enum
from threading import Thread, Event, Lock
from multiprocessing.pool import ThreadPool
from typing import Optional, Union
from collections.abc import Iterator
from collections import OrderedDict

from PySide6.QtCore import (
    Qt,
    Slot,
    QJsonValue,
    QObject,
    QUrl,
    Property,
    Signal,
    QPoint,
    QItemSelectionModel,
    QItemSelection,
)

from meshroom.core import sessionUid
from meshroom.common import deprecated
from meshroom.common.qt import QObjectListModel
from meshroom.core.attribute import Attribute, AnySet, ListAttribute, ShapeAttribute
from meshroom.core.files import MESHROOM_PROJECT_EXTENSION, MESHROOM_TEMPLATE_EXTENSION, withExtension
from meshroom.core.graph import Graph, Edge, generateTempProjectFilepath
from meshroom.core.graphIO import GraphIO

from meshroom.core.taskManager import TaskManager
from meshroom.core.submitter import jobManager

from meshroom.core.node import NodeChunk, Node, Status, ExecMode, CompatibilityNode, BackdropNode, Position
from meshroom.core import desc, submitters, MrNodeType
from meshroom.core.subgraph import SubGraphError
from meshroom.core.subgraph.editing import SubGraphStack, isSubGraphNode
from meshroom.ui import commands
from meshroom.ui.utils import makeProperty
from meshroom.ui.live import LiveController
from meshroom.core import chapters


class PollerRefreshStatus(Enum):
    AUTO_ENABLED = 0  # The file watcher polls every single status file periodically
    DISABLED = 1  # The file watcher is disabled and never polls any file
    MINIMAL_ENABLED = 2  # The file watcher only polls status files for chunks that are either submitted or running externally


class WatchedFile:
    def __init__(self, obj):
        self.isChunk = isinstance(obj, NodeChunk)
        self.chunk: Optional[NodeChunk] = obj if self.isChunk else None
        self.node: Node = obj.node if self.isChunk else obj

    @property
    def monitoredObject(self):
        return self.chunk if self.isChunk else self.node

    @property
    def monitoredObjectName(self):
        if self.isChunk:
            return self.chunk.getChunkIndexName()
        return self.node.name

    @property
    def path(self):
        if self.isChunk:
            return self.chunk.getStatusFile()
        return self.node.nodeStatusFile

    @property
    def lastModTime(self):
        if self.isChunk:
            return self.chunk.statusFileLastModTime
        return self.node.nodeStatusFileLastModTime

    def shouldMonitorChanges(self):
        return self.monitoredObject.shouldMonitorChanges()

    def updateStatusFromCache(self):
        """
        Update status from cache and return the node if it should load output attributes,
        or None otherwise.
        """
        self.monitoredObject.updateStatusFromCache()
        if self.isChunk:
            if self.chunk._status.status == Status.SUCCESS:
                return True
        else:
            if self.node.getGlobalStatus() == Status.SUCCESS:
                return True
        return False


class FilesModTimePollerThread(QObject):
    """
    Thread responsible for non-blocking polling of last modification times of a list of files.
    Uses a Python ThreadPool internally to split tasks on multiple threads.
    """
    timesAvailable = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._mutex = Lock()
        self._threadPool = ThreadPool(4)
        self._stopFlag = Event()
        self._refreshInterval = 5  # refresh interval in seconds
        self._files = []
        if submitters:
            self._filePollerRefresh = PollerRefreshStatus.MINIMAL_ENABLED
        else:
            self._filePollerRefresh = PollerRefreshStatus.DISABLED

    def __del__(self):
        self._threadPool.terminate()
        self._threadPool.join()

    def start(self, files=None):
        """
        Start polling thread.

        Args:
            files: the list of files to monitor
        """
        if self._filePollerRefresh is PollerRefreshStatus.DISABLED:
            return
        if self._thread:
            # thread already running, return
            return
        self._stopFlag.clear()
        self._files = files or []
        self._thread = Thread(target=self.run)
        self._thread.start()

    def setFiles(self, files):
        """
        Set the list of files to monitor.

        Args:
            files: the list of files to monitor
        """
        logging.debug(f"FilesModTimePollerThread: Watch files {files}")
        with self._mutex:
            self._files = files

    def stop(self):
        """ Request polling thread to stop. """
        if not self._thread:
            return
        self._stopFlag.set()
        self._thread.join()
        self._thread = None

    @staticmethod
    def getFileLastModTime(f):
        """ Return 'mtime' of the file if it exists, -1 otherwise. """
        try:
            return os.path.getmtime(f)
        except OSError:
            return -1

    def run(self):
        """ Poll watched files for last modification time. """
        while not self._stopFlag.wait(self._refreshInterval):
            with self._mutex:
                files = list(self._files)
            times = self._threadPool.map(FilesModTimePollerThread.getFileLastModTime, files)
            with self._mutex:
                if files == self._files:
                    self.timesAvailable.emit(times)

    def onFilePollerRefreshChanged(self, value):
        """ Stop or start the file poller depending on the new refresh status. """
        self._filePollerRefresh = PollerRefreshStatus(value)
        if self._filePollerRefresh is PollerRefreshStatus.DISABLED:
            self.stop()
        else:
            self.start()
        self.filePollerRefreshReady.emit()

    filePollerRefresh = Property(int, lambda self: self._filePollerRefresh.value, constant=True)
    filePollerRefreshReady = Signal()  # The refresh status has been updated and is ready to be used


class NodeStatusMonitor(QObject):
    """
    NodeStatusMonitor regularly check status files for modification and trigger their update on change.

    When working locally, status changes are reflected through the emission of 'statusChanged' signals.
    But when a graph is being computed externally - either via a Submitter or on another machine,
    Status files are modified by another instance, potentially outside this machine file system scope.
    Same goes when status files are deleted/modified manually.
    Thus, for genericity, monitoring is based on regular polling and not file system watching.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.monitorableNodes = []
        self.monitoredItems: OrderedDict[str, WatchedFile] = {}  # Dict {filepath: WatchedFile}
        self._filesTimePoller = FilesModTimePollerThread(parent=self)
        self._filesTimePoller.timesAvailable.connect(self.compareFilesTimes, Qt.QueuedConnection)
        self._filesTimePoller.start()
        self.setMonitored([])
        self.filePollerRefreshChanged.connect(self._filesTimePoller.onFilePollerRefreshChanged)
        self._filesTimePoller.filePollerRefreshReady.connect(self.onFilePollerRefreshUpdated)

    def setWatchedFiles(self):
        self.monitoredItems = self.getMonitoredFiles()
        self._filesTimePoller.setFiles(list(self.monitoredItems.keys()))

    def setMonitored(self, nodes):
        self.monitorableNodes = nodes
        self.setWatchedFiles()

    def stop(self):
        """ Stop the status files monitoring. """
        self._filesTimePoller.stop()

    def getMonitoredFiles(self) -> OrderedDict:
        """ Get files to monitor. """
        monitoredItems = OrderedDict()
        for node in self.monitorableNodes:
            # Gather all potential files
            if node._chunksCreated:
                watchedFiles = [WatchedFile(c) for c in node.getAllChunks()]
            else:
                watchedFiles = [WatchedFile(node)]
                if node.hasPreprocessChunk:
                    watchedFiles.append(WatchedFile(node._preprocessChunk))
                if node.hasPostprocessChunk:
                    watchedFiles.append(WatchedFile(node._postprocessChunk))
            # Filter depending on PollerRefreshStatus
            for wf in watchedFiles:
                if self.filePollerRefresh == PollerRefreshStatus.AUTO_ENABLED.value:
                    monitoredItems[wf.path] = wf
                # Minimal mode will not watch nodes that have succeeded for example
                elif self.filePollerRefresh == PollerRefreshStatus.MINIMAL_ENABLED.value:
                    if wf.shouldMonitorChanges():
                        monitoredItems[wf.path] = wf
        return monitoredItems

    def compareFilesTimes(self, times):
        """
        Compare previous file modification times with results from last poll.
        Trigger chunk status update if file was modified since.

        Args:
            times: the last modification times for currently monitored files.
        """
        nodesToUpdate = set()
        needsRebuild = False
        for wf, fileModTime in zip(self.monitoredItems.values(), times):
            # File has not changed : nothing to do
            if fileModTime == wf.lastModTime:
                continue
            success = wf.updateStatusFromCache()
            if success:
                needsRebuild = True  # There are files to stop watching
                nodesToUpdate.add(wf.node)
            # Chunks have been created, rebuild the watch list
            if not wf.isChunk and wf.node._chunksCreated:
                needsRebuild = True

        for node in nodesToUpdate:
            node.loadOutputAttr()

        if needsRebuild:
            self.setWatchedFiles()

    def onFilePollerRefreshUpdated(self):
        """
        Upon an update of the file poller status, retrigger the generation of the list of status files for
        the chunks that are to be watched.
        In auto-refresh mode, this includes all the chunks' status files.
        In minimal auto-refresh mode, this includes only the chunks that are submitted or running.
        """
        if self.filePollerRefresh is not PollerRefreshStatus.DISABLED.value:
            self.setWatchedFiles()

    def onComputeStatusChanged(self):
        """
        When a chunk's status is updated, update the list of watched files with submitted and running chunks if the
        file poller status is minimal auto-refresh.
        """
        if self.filePollerRefresh is PollerRefreshStatus.MINIMAL_ENABLED.value:
            self.setWatchedFiles()

    filePollerRefreshChanged = Signal(int)
    filePollerRefresh = Property(int, lambda self: self._filesTimePoller.filePollerRefresh, notify=filePollerRefreshChanged)


class GraphLayout(QObject):
    """
    GraphLayout provides auto-layout features to a UIGraph.
    """

    class DepthMode(Enum):
        """ Defines available node depth mode to layout the graph automatically. """
        MinDepth = 0  # use node minimal depth
        MaxDepth = 1  # use node maximal depth

    # map between DepthMode and corresponding node depth attribute name
    _depthAttribute = {
        DepthMode.MinDepth: 'minDepth',
        DepthMode.MaxDepth: 'depth'
    }

    def __init__(self, graph):
        super().__init__(graph)
        self.graph = graph
        self._depthMode = GraphLayout.DepthMode.MaxDepth
        self._nodeWidth = 160  # implicit node width
        self._nodeHeight = 120   # implicit node height
        self._gridSpacing = 40  # column/line spacing between nodes

    @Slot(Node, Node, int, int)
    def autoLayout(self, fromNode=None, toNode=None, startX=0, startY=0):
        """
        Perform auto-layout from 'fromNode' to 'toNode', starting from (startX, startY) position.

        Args:
            fromNode (BaseNode): where to start the auto layout from
            toNode (BaseNode): up to where to perform the layout
            startX (int): start position x coordinate
            startY (int): start position y coordinate
        """
        if not self.graph.nodes:
            return
        fromIndex = self.graph.nodes.indexOf(fromNode) if fromNode else 0
        toIndex = self.graph.nodes.indexOf(toNode) if toNode else self.graph.nodes.count - 1

        def getDepth(n):
            return getattr(n, self._depthAttribute[self._depthMode])

        maxDepth = max([getDepth(n) for n in self.graph.nodes.values()])
        grid = [[] for _ in range(maxDepth + 1)]

        # Retrieve reference depth from start node
        zeroDepth = getDepth(self.graph.nodes.at(fromIndex)) if fromIndex > 0 else 0
        for i in range(fromIndex, toIndex + 1):
            n = self.graph.nodes.at(i)
            grid[getDepth(n) - zeroDepth].append(n)

        with self.graph.groupedGraphModification("Graph Auto-Layout"):
            for x, line in enumerate(grid):
                for y, node in enumerate(line):
                    px = startX + x * (self._nodeWidth + self._gridSpacing)
                    py = startY + y * (self._nodeHeight + self._gridSpacing)
                    self.graph.moveNode(node, Position(px, py))

    @Slot()
    def reset(self):
        """ Perform auto-layout on the whole graph. """
        self.autoLayout()

    def positionBoundingBox(self, nodes=None):
        """
        Return bounding box for a set of nodes as (x, y, width, height).

        Args:
            nodes (list of Node): the list of nodes or the whole graph if None

        Returns:
            list of int: the resulting bounding box (x, y, width, height)
        """
        if nodes is None:
            nodes = self.graph.nodes.values()
        if not nodes:
            return [0, 0, 0, 0]
        first = nodes[0]
        bbox = [first.x, first.y, first.x, first.y]
        for n in nodes:
            bbox[0] = min(bbox[0], n.x)
            bbox[1] = min(bbox[1], n.y)
            bbox[2] = max(bbox[2], n.x)
            bbox[3] = max(bbox[3], n.y)

        bbox[2] -= bbox[0]
        bbox[3] -= bbox[1]
        return bbox

    def boundingBox(self, nodes=None):
        """
        Return bounding box for a set of nodes as (x, y, width, height).

        Args:
            nodes (list of Node): the list of nodes or the whole graph if None

        Returns:
            list of int: the resulting bounding box (x, y, width, height)
        """
        bbox = self.positionBoundingBox(nodes)
        bbox[2] += self._nodeWidth
        bbox[3] += self._nodeHeight
        return bbox

    def setDepthMode(self, mode):
        """ Set node depth mode to use. """
        if isinstance(mode, int):
            mode = GraphLayout.DepthMode(mode)
        if self._depthMode.value == mode.value:
            return
        self._depthMode = mode

    depthModeChanged = Signal()
    depthMode = Property(int, lambda self: self._depthMode.value, setDepthMode, notify=depthModeChanged)
    nodeHeightChanged = Signal()
    nodeHeight = makeProperty(int, "_nodeHeight", notify=nodeHeightChanged)
    nodeWidthChanged = Signal()
    nodeWidth = makeProperty(int, "_nodeWidth", notify=nodeWidthChanged)
    gridSpacingChanged = Signal()
    gridSpacing = makeProperty(int, "_gridSpacing", notify=gridSpacingChanged)


class UIGraph(QObject):
    """
    High level wrapper over core.Graph, with additional features dedicated to UI integration.

    UIGraph exposes undoable methods on its graph and computation in a separate thread.
    It also provides a monitoring of all its computation units (NodeChunks).
    """
    def __init__(self, undoStack: commands.UndoStack, taskManager: TaskManager, parent: QObject = None):
        super().__init__(parent)
        self._undoStack = undoStack
        self._taskManager = taskManager
        self._graph: Graph = Graph('', self)

        self._modificationCount = 0
        self._chunksMonitor: NodeStatusMonitor = NodeStatusMonitor(parent=self)
        self._computeThread: Thread = Thread()
        self._computingLocally = self._submitted = False
        self._sortedDFSChunks: QObjectListModel = QObjectListModel(parent=self)
        self._layout: GraphLayout = GraphLayout(self)
        self._selectedNode = None
        self._selectedChunk = None
        self._nodeSelection: QItemSelectionModel = QItemSelectionModel(self._graph.nodes, parent=self)
        self._hoveredNode = None
        # Selected node names in the order they were selected (used to connect nodes with "&")
        self._selectionOrder: list[str] = []
        self._nodeSelection.selectionChanged.connect(self._updateSelectionOrder)
        # Pin copied from a pin context menu, to connect it to another pin: (attribute full name, isOutput)
        self._copiedPin = None
        # Values of attributes being edited interactively (viewer handles), restored for the undo command
        self._interactiveEdits: dict = {}
        # Set while the values of animated parameters must not be mirrored from their keys
        self._mirroring = False
        # Uid of the Read node the frame range follows, when it was last read
        self._rangeSourceUid = None
        # 2D attributes shown as separate x / y fields (by full name), kept while the session lasts
        self._splitVec2: set = set()
        # Live compositing evaluation ("Evaluate" on a node)
        self._live: LiveController = LiveController(parent=self)
        self._live.setGraph(self._graph)
        # Animated parameters of the selected node show their value at the current frame
        self._live.currentFrameChanged.connect(self._mirrorAnimation)
        self._live.playingChanged.connect(self._mirrorAnimation)
        self.selectedNodeChanged.connect(self._mirrorAnimation)
        # Chapter lines (folding of the graph editor): folded sections and the watched Chapter nodes
        self._foldBands: list = []
        self._foldRevision = 0
        self._foldSignature = None
        self._watchedChapters: set = set()
        self._watchChapters()
        # Subgraphs entered from the project graph (see meshroom.core.subgraph.editing)
        self._subGraphStack = SubGraphStack()
        self._liveRunning = False
        self._liveStatus = ""
        self._liveSuccess = True
        # Run Live: {node name: (status, message)} of the nodes of the shown graph (see executor.NodeStatus)
        self._liveNodeStatuses = {}
        self._liveStatusRevision = 0
        self.liveRunDone.connect(self._onLiveRunDone)
        self.liveNodeEvent.connect(self._onLiveNodeEvent)

        self.submitLabel = "{projectName}"
        self.computeStatusChanged.connect(self.updateLockedUndoStack)
        self.filePollerRefreshChanged.connect(self._chunksMonitor.filePollerRefreshChanged)

    def setGraph(self, g):
        """ Set the internal graph. """
        self._dropSubGraphs()
        if self._graph:
            self.stopExecution()
            # Clear all the locally submitted nodes at once before the graph gets changed, as it will not receive further updates
            if self._computingLocally:
                self._graph.clearLocallySubmittedNodes()
            self.clear()
        oldGraph = self._graph
        self._graph = g
        if oldGraph:
            oldGraph.deleteLater()

        self._graph.updated.connect(self.onGraphUpdated)
        self._graph.statusUpdated.connect(self.updateChunkMonitor)
        self._graph.frameRangeChanged.connect(self._onFrameRangeChanged)
        self._graph.workspaceModeChanged.connect(self.workspaceModeChanged)
        self._live.setGraph(self._graph)
        self._rangeSourceUid = None
        self._updateFrameRange()
        # The range loaded with the graph (its change was emitted before this connection)
        self._onFrameRangeChanged()
        self._watchChapters()
        self._taskManager.update(self._graph)

        # Update and connect chunks when the graph is set for the first time
        self.updateChunks()

        # Perform auto-layout if graph does not provide nodes positions
        if GraphIO.Features.NodesPositions not in self._graph.fileFeatures:
            self._layout.reset()
            # Clear undo-stack after layout
            self._undoStack.clear()
        else:
            bbox = self._layout.positionBoundingBox()
            if bbox[2] == 0 and bbox[3] == 0:
                self._layout.reset()
                # Clear undo-stack after layout
                self._undoStack.clear()

        self._nodeSelection.setModel(self._graph.nodes)
        self._selectionOrder = []
        # Interactive edits of the previous graph
        self._interactiveEdits = {}
        self._copiedPin = None
        self.copiedAttributeChanged.emit()
        self.graphChanged.emit()
        self.workspaceModeChanged.emit()

    # Subgraphs

    def _showGraph(self, g: Graph):
        """ Show another graph of the project (entering / leaving a subgraph), without clearing the current one. """
        self.clearNodeHover()
        self.clearNodeSelection()
        self._graph.updated.disconnect(self.onGraphUpdated)
        self._graph.statusUpdated.disconnect(self.updateChunkMonitor)
        self._sortedDFSChunks.clear()
        # The commands of the undo stack refer to the nodes of the graph shown when they were done
        self._undoStack.clear()
        self._graph = g
        self._graph.updated.connect(self.onGraphUpdated)
        self._graph.statusUpdated.connect(self.updateChunkMonitor)
        self.updateChunks()
        bbox = self._layout.positionBoundingBox()
        if len(self._graph.nodes) and bbox[2] == 0 and bbox[3] == 0:
            self._layout.reset()
            self._undoStack.clear()
        self._nodeSelection.setModel(self._graph.nodes)
        # The Run Live statuses are those of the nodes of the graph shown before
        self._setLiveNodeStatuses({})
        # The viewer evaluates the nodes of the graph shown (its cache is kept)
        self._live.showGraph(self._graph)
        self._rangeSourceUid = None
        self.graphChanged.emit()
        self.subGraphChanged.emit()

    def _dropSubGraphs(self):
        """ Leave all the subgraphs without writing them back (another project is loaded). """
        if not len(self._subGraphStack):
            return
        root = self._subGraphStack.rootGraph
        inner = [self._graph] + [level.outerGraph for level in self._subGraphStack.levels[1:]]
        self._subGraphStack.levels.clear()
        self._showGraph(root)
        for graph in inner:
            graph.deleteLater()

    def _projectGraph(self) -> Graph:
        """ The project graph, with the entered subgraphs written back in their nodes. """
        if len(self._subGraphStack):
            return self._subGraphStack.commitAll(self._graph)
        return self._graph

    @Slot(QObject, result=bool)
    def isSubGraphNode(self, node) -> bool:
        return isSubGraphNode(node)

    @Slot(QObject, result=bool)
    def enterSubGraph(self, node) -> bool:
        """ Show the subgraph of a SubGraph node. Returns whether it was entered. """
        if not isSubGraphNode(node):
            return False
        if self.isComputingLocally():
            self.parent().showMessage("Subgraphs cannot be edited while computing.", "warning")
            return False
        try:
            inner = self._subGraphStack.enter(self._graph, node, parent=self)
        except (ValueError, SubGraphError) as e:
            self.parent().showMessage(str(e), "error")
            return False
        except Exception as e:
            logging.exception(f"Cannot enter the subgraph of {node.name}")
            self.parent().showMessage(f"Cannot enter the subgraph of {node.name}: {e}", "error")
            return False
        self._showGraph(inner)
        return True

    @Slot()
    def groupSelectedIntoSubGraph(self):
        """ Put the selected nodes in a new SubGraph node (their outside connections become its inputs / outputs). """
        nodes = [node for node in self.getSelectedNodes() if node.nodeType != "Backdrop"]
        if not nodes:
            self.parent().showMessage("Select the nodes to put in a SubGraph.", "warning")
            return
        if any(node.locked for node in nodes):
            self.parent().showMessage("Locked nodes cannot be put in a SubGraph.", "warning")
            return
        subGraphNode = self.push(commands.CollapseIntoSubGraphCommand(self._graph, nodes))
        if subGraphNode:
            self.selectNodes([subGraphNode])
            self.selectedNode = subGraphNode

    @Slot()
    def exitSubGraph(self):
        """ Leave the current subgraph, written back in its node, and select its node. """
        if not len(self._subGraphStack):
            return
        node = self._subGraphStack.levels[-1].node
        inner = self._graph
        try:
            outer = self._subGraphStack.exit(inner)
        except Exception as e:
            self.parent().showMessage(f"The subgraph could not be written back: {e}", "error")
            raise
        self._showGraph(outer)
        inner.deleteLater()
        self.selectNodes([node])
        self.selectedNode = node

    @Slot(int)
    def exitSubGraphTo(self, depth: int):
        """ Leave subgraphs until `depth` levels are entered (0: the project graph). """
        while len(self._subGraphStack) > max(0, depth):
            self.exitSubGraph()

    def _liveSource(self, node=None):
        """ (graph data, inputs, context, loop data, label) of what Run Live runs, None if nothing. """
        from meshroom.core.subgraph.executor import readGraphData
        from meshroom.core.subgraph.nodes import loopFromNode, nodeGraphSource
        if len(self._subGraphStack):
            node = self._subGraphStack.levels[-1].node
            graphData = self._graph.serialize(asTemplate=True)
        elif isSubGraphNode(node):
            source = nodeGraphSource(node)
            if source is None:
                return None
            graphData = readGraphData(source)
        else:
            return None
        loop = loopFromNode(node)
        if loop.mode in ("frames", "items"):
            # A preview: the first frame / item only
            loop = loop.subset(0, 1)
        from meshroom.core.subgraph.executor import plainValue
        inputs = {child.name: plainValue(child) for child in node.ins.value}
        return graphData, inputs, node.context.value, loop.asDict(), node.name

    @Slot()
    @Slot(QObject)
    def runLive(self, node=None):
        """
        Run the current subgraph (or the subgraph of a node) now, in the interpreter of its context, with
        the inputs of its node: the first frame of a frame loop, all the iterations of a count loop.
        """
        if self._liveRunning:
            return
        source = self._liveSource(node or self._selectedNode)
        if source is None:
            self.parent().showMessage("Run Live: enter a subgraph or select a SubGraph node.", "warning")
            return
        graphData, inputs, context, loopData, label = source
        self._liveRunning = True
        self._liveSuccess = True
        self._liveStatus = f"{label}: running..."
        self.liveStatusChanged.emit()
        # The statuses of the nodes are shown on the subgraph when it is entered, else on the SubGraph node
        inside = bool(len(self._subGraphStack))
        self._setLiveNodeStatuses({} if inside else {label: ("RUNNING", "Run Live...")})

        def onNode(nodeName, status, message):
            if inside:
                # Emitted from the worker thread: queued to the thread of the user interface
                self.liveNodeEvent.emit(nodeName, status, message)

        def work():
            from meshroom.core.subgraph.runner import runLive
            try:
                result = runLive(graphData, inputs, context, loopData, onNode=onNode)
                self.liveRunDone.emit(self._liveSummary(label, result), True)
                if not inside:
                    self.liveNodeEvent.emit(label, "SUCCESS", f"Run Live: done in {result['elapsed']:.2f} s")
            except Exception as e:
                failed = getattr(e, "nodeName", "")
                message = str(e)
                if failed and not message.startswith(f"{failed}:"):
                    message = f"{failed}: {message}"
                self.liveRunDone.emit(f"{label}: failed at {message}" if failed else f"{label}: {message}", False)
                if not inside:
                    self.liveNodeEvent.emit(label, "ERROR", message)

        Thread(target=work, daemon=True).start()

    @staticmethod
    def _liveSummary(label: str, result: dict) -> str:
        """ The message of a successful Run Live: duration, where it ran, nodes run / skipped, outputs. """
        nodes = result.get("nodes") or {}
        ran = [name for name, status in nodes.items() if status == "SUCCESS"]
        skipped = [name for name, status in nodes.items() if status == "SKIPPED"]
        parts = [f"{label}: OK in {result['elapsed']:.2f} s"]
        if result.get("interpreter"):
            parts.append(f"in {result['interpreter']}")
        text = " ".join(parts) + "."
        if nodes:
            text += f" {len(ran)} node(s) run"
            if skipped:
                text += f", {len(skipped)} not needed here ({', '.join(skipped)})"
            text += "."
        outputs = ", ".join(f"{name} = {value!r}"[:200] for name, value in result["outputs"].items())
        if outputs:
            text += f" Outputs: {outputs}"
        return text

    def _setLiveNodeStatuses(self, statuses: dict):
        self._liveNodeStatuses = statuses
        self._liveStatusRevision += 1
        self.liveNodeStatusesChanged.emit()

    def _onLiveNodeEvent(self, nodeName: str, status: str, message: str):
        self._liveNodeStatuses[nodeName] = (status, message)
        self._liveStatusRevision += 1
        self.liveNodeStatusesChanged.emit()

    @Slot(str, result=str)
    def liveNodeStatus(self, nodeName: str) -> str:
        """ The Run Live status of a node of the shown graph ("" if none): RUNNING, SUCCESS, ERROR... """
        return self._liveNodeStatuses.get(nodeName, ("", ""))[0]

    @Slot(str, result=str)
    def liveNodeMessage(self, nodeName: str) -> str:
        return self._liveNodeStatuses.get(nodeName, ("", ""))[1]

    @Slot()
    def clearLiveStatuses(self):
        self._setLiveNodeStatuses({})

    def _onLiveRunDone(self, message: str, success: bool):
        self._liveRunning = False
        self._liveSuccess = success
        self._liveStatus = message
        self.liveStatusChanged.emit()
        (logging.info if success else logging.error)(f"[Run Live] {message}")
        self.parent().showMessage(f"Run Live: {message}", "ok" if success else "error")

    def onGraphUpdated(self):
        """ Callback to any kind of attribute modification. """
        # TODO: handle this with a better granularity
        self.updateChunks()
        # Keys changed (undo / redo, key toggled): the animated values follow
        self._mirrorAnimation()
        self._updateFrameRange()

    def updateChunks(self):
        dfsNodes: list[Node] = self._graph.dfsOnFinish(None)[0]
        chunks = []
        for node in dfsNodes:
            if node._preprocessChunk:
                chunks.append(node._preprocessChunk)
            if node._chunksCreated:
                nodechunks = node.getChunks()
                chunks.extend(nodechunks)
            else:
                chunks.extend(node.chunkPlaceholder)
            if node._postprocessChunk:
                chunks.append(node._postprocessChunk)
        if self._sortedDFSChunks.objectList() == chunks:
            # Nothing has changed, return
            return
        for chunk in self._sortedDFSChunks:
            if chunk not in chunks:
                # Chunk have been already deleted
                continue
            chunk.statusChanged.disconnect(self.updateGraphComputingStatus)
            chunk.statusChanged.disconnect(self._chunksMonitor.onComputeStatusChanged)
        self._sortedDFSChunks.setObjectList(chunks)
        for chunk in self._sortedDFSChunks:
            chunk.statusChanged.connect(self.updateGraphComputingStatus)
            chunk.statusChanged.connect(self._chunksMonitor.onComputeStatusChanged)
        # provide ChunkMonitor with the update list of chunks
        self.updateChunkMonitor()
        # update graph computing status based on the new list of NodeChunks
        self.updateGraphComputingStatus()

    def updateChunkMonitor(self):
        """ Update the list of chunks for status files monitoring. """
        nodes = set()
        for node in self._graph.dfsOnFinish(None)[0]:
            if not node._chunksCreated:
                nodes.add(node)
        for chunk in self._sortedDFSChunks:
            nodes.add(chunk.node)
        self._chunksMonitor.setMonitored(list(nodes))

    def clear(self):
        if self._graph:
            self.clearNodeHover()
            self.clearNodeSelection()
            self._taskManager.clear()
            self._graph.clear()
        self._sortedDFSChunks.clear()
        self._undoStack.clear()

    def stopChildThreads(self):
        """ Stop all child threads. """
        self.stopExecution()
        self._chunksMonitor.stop()

    @Slot(str)
    def loadGraph(self, filepath):
        g = Graph("")
        if filepath:
            g.load(filepath)
            if not os.path.exists(g.cacheDir):
                os.mkdir(g.cacheDir)
        self.setGraph(g)

    @Slot(str)
    @Slot(str, bool)
    @deprecated.depreciateParam(
        "copyOutputs",
        "UIGraph.initFromTemplate 'copyOutputs' parameter is deprecated. Please use 'keepOutputNodes' instead."
    )
    def initFromTemplate(self, filepath: str, keepOutputNodes: bool = False, copyOutputs: Optional[bool] = None) -> None:
        if copyOutputs is not None:
            keepOutputNodes = copyOutputs

        graph = Graph("")
        if filepath:
            graph.initFromTemplate(filepath, keepOutputNodes=keepOutputNodes)
        self.setGraph(graph)

    @Slot(QUrl, result="QVariantList")
    @Slot(QUrl, QPoint, result="QVariantList")
    def importProject(self, filepath, position=None):
        if isinstance(filepath, (QUrl)):
            # depending how the QUrl has been initialized,
            # toLocalFile() may return the local path or an empty string
            localFile = filepath.toLocalFile()
            if not localFile:
                localFile = filepath.toString()
        else:
            localFile = filepath
        if isinstance(position, QPoint):
                position = Position(position.x(), position.y())
        yOffset = self.layout.gridSpacing + self.layout.nodeHeight
        return self.push(commands.ImportProjectCommand(self._graph, localFile, position=position, yOffset=yOffset))

    @Slot(QUrl)
    def saveAs(self, url):
        self._saveAs(url, extension=MESHROOM_PROJECT_EXTENSION)

    @Slot(QUrl)
    def saveAsTemplate(self, url):
        if len(self._subGraphStack):
            # Inside a subgraph: save the subgraph shown (e.g. to run it from a software or reference it)
            localFile = withExtension(url if isinstance(url, str) else url.toLocalFile(), MESHROOM_TEMPLATE_EXTENSION)
            self._graph.save(localFile, setupProjectFile=False, template=True)
            self.parent().showMessage(f"Subgraph saved as template: {localFile}", "ok")
            return
        self._saveAs(url, setupProjectFile=False, template=True, extension=MESHROOM_TEMPLATE_EXTENSION)

    def _saveAs(self, url, setupProjectFile=True, template=False, extension=MESHROOM_PROJECT_EXTENSION):
        """ Helper function for 'save as' features. """
        if isinstance(url, (str)):
            localFile = url
        else:
            localFile = url.toLocalFile()
        localFile = withExtension(localFile, extension)
        self._projectGraph().save(localFile, setupProjectFile=setupProjectFile, template=template)
        self._undoStack.setClean()
        # saving file on disk impacts cache folder location
        # => force re-evaluation of monitored status files paths
        self.updateChunkMonitor()

    @Slot()
    def saveAsTemp(self):
        projectPath = generateTempProjectFilepath()
        self._saveAs(projectPath)

    @Slot()
    def save(self):
        self._projectGraph().save()
        self._undoStack.setClean()

    @Slot()
    def saveAsNewVersion(self):
        self._projectGraph().saveAsNewVersion()
        self._undoStack.setClean()

    @Slot()
    def updateLockedUndoStack(self):
        if self.isComputingLocally():
            self._undoStack.lockAtThisIndex()
        else:
            self._undoStack.unlock()

    @Slot()
    @Slot(Node)
    @Slot(list)
    def execute(self, nodes: Optional[Union[list[Node], Node]] = None):
        nodes = [nodes] if not isinstance(nodes, Iterable) and nodes else nodes
        if len(self._subGraphStack):
            # Live nodes are run by the node containing them: compute it, from the project graph
            nodes = [self._subGraphStack.levels[0].node]
            self.exitSubGraphTo(0)
        self.save()  # always save the graph before computing
        self._taskManager.compute(self._graph, nodes)
        self.updateLockedUndoStack()  # explicitly call the update while it is already computing

    @Slot()
    def stopExecution(self):
        self.updateChunks()
        if not self.isComputingLocally():
            return
        self._taskManager.requestBlockRestart()
        self._graph.stopExecution()
        self._taskManager.join()

    @Slot(Node)
    def stopNodeComputation(self, node):
        """ Stop the computation of the node and update all the nodes depending on it. """
        self.updateChunks()
        if not self.isComputingLocally():
            return

        # Stop the node and wait Task Manager
        node.stopComputation()
        self._taskManager.join()

    @Slot(Node)
    def cancelNodeComputation(self, node):
        """ Cancel the computation of the node and all the nodes depending on it. """
        self.updateChunks()
        if node.getGlobalStatus() == Status.SUBMITTED:
            # Status from SUBMITTED to NONE
            # Make sure to remove the nodes from the Task Manager list
            node.clearSubmittedChunks()
            self._taskManager.removeNode(node, displayList=True, processList=True)

            for n in node.getOutputNodes(recursive=True, dependenciesOnly=True):
                n.clearSubmittedChunks()
                self._taskManager.removeNode(n, displayList=True, processList=True)

    def isChunkComputingLocally(self, chunk):
        # Update graph computing status
        computingLocally = chunk._status.execMode == ExecMode.LOCAL and \
                           (sessionUid in (chunk.node._nodeStatus.submitterSessionUid, chunk._status.computeSessionUid)) and \
                           (chunk._status.status in (Status.RUNNING, Status.SUBMITTED))
        return computingLocally

    def isChunkComputingExternally(self, chunk):
        # Note: We do not check computeSessionUid for the submitted status,
        #       as the source instance of the submit has no importance.
        return (chunk._status.execMode == ExecMode.EXTERN) and \
                chunk._status.status in (Status.RUNNING, Status.SUBMITTED)

    @Slot(NodeChunk)
    def stopTask(self, chunk: NodeChunk):
        """ Stop the selected task. """
        chunk.updateStatusFromCache()
        if not chunk.isAlreadySubmitted():
            return
        node = chunk.node
        job = jobManager.getNodeJob(node)
        if job:
            chunkIteration = chunk.range.iteration
            try:
                job.stopChunkTask(node, chunkIteration)
            except Exception as e:
                self.parent().showMessage(f"Failed to stop chunk {chunkIteration} of {node.label}", "error")
                logging.warning(f"Error on stopTask:\n{e}")
            else:
                chunk.updateStatusFromCache()
                chunk.upgradeStatusTo(Status.STOPPED)
                # TODO: Stop depending nodes?
                self.parent().showMessage(f"Stopped chunk {chunkIteration} of {node.label}")
        else:
            chunk.stopProcess()
            self._taskManager._cancelledChunks.append(chunk)
            for chunk in node._chunks:
                if chunk._status.status == Status.SUBMITTED:
                    chunk.stopProcess()
                    self._taskManager._cancelledChunks.append(chunk)
            for n in node.getOutputNodes(recursive=True, dependenciesOnly=True):
                n.clearSubmittedChunks()
                self._taskManager.removeNode(n, displayList=True, processList=True)

    @Slot(Node)
    def stopNode(self, node: Node):
        """ Stop the selected task. """
        job = jobManager.getNodeJob(node)
        if job:
            try:
                job.stopChunkTask(node, -1)
            except Exception as e:
                self.parent().showMessage(f"Failed to stop node {node.label}", "error")
                logging.warning(f"Error on stopTask:\n{e}")
            else:
                node.updateNodeStatusFromCache()
                node.upgradeStatusTo(Status.STOPPED)
                # TODO : Stop depending nodes ?
                self.parent().showMessage(f"Stopped node {node.label}")
        else:
            self.cancelNodeComputation(node)
            node.stopComputation()

    @Slot(NodeChunk)
    def restartTask(self, chunk: NodeChunk):
        """ Relaunch a stopped task. """
        node = chunk.node
        job = jobManager.getNodeJob(node)
        if job:
            chunkIteration = chunk.range.iteration
            try:
                chunk.updateStatusFromCache()
                chunk.upgradeStatusTo(Status.SUBMITTED)
                job.restartChunkTask(node, chunkIteration)
            except Exception as e:
                chunk.updateStatusFromCache()
                chunk.upgradeStatusTo(Status.ERROR)
                self.parent().showMessage(f"Failed to relaunch chunk {chunkIteration} of {node.label}", "error")
                logging.warning(f"Error on restartTask:\n{e}")
            else:
                self.parent().showMessage(f"Relaunched chunk {chunkIteration} of {node.label}")
        else:
            # For this we would need to use a pool (with either chunks or nodes)
            # instead of the list of nodes that are processed serially
            self.parent().showMessage(f"Chunks cannot be launched individually locally", "warning")
            if self.canComputeNode(node):
                self.execute([node])

    @Slot(NodeChunk)
    def skipTask(self, chunk: NodeChunk):
        """
        Skip the task: the job will continue as if the task succeeded.
        In local mode, the chunk status will be set to success.
        """
        chunk.updateStatusFromCache()
        node = chunk.node
        chunkIteration = chunk.range.iteration
        job = jobManager.getNodeJob(node)
        if job:
            try:
                job.skipChunkTask(node, chunkIteration)
            except Exception as e:
                self.parent().showMessage(f"Failed to skip chunk {chunkIteration} of {node.label}", "error")
                logging.warning(f"Error on skipTask:\n{e}")
            else:
                chunk.upgradeStatusTo(Status.SUCCESS)
                self.parent().showMessage(f"Skipped chunk {chunkIteration} of {node.label}")
        else:
            chunk.stopProcess()
            chunk.upgradeStatusTo(Status.SUCCESS)
            self._taskManager._cancelledChunks.append(chunk)
            self.parent().showMessage(f"Skipped chunk {chunkIteration} of {node.label}")

    @Slot(Node)
    def pauseJob(self, node: Node):
        """
        Pause the running job : cancel all scheduled tasks.
        Current task is not stopped but future tasks will not be launched.
        """
        job = jobManager.getNodeJob(node)
        if job:
            try:
                job.pauseJob()
            except Exception as e:
                logging.warning(f"Error on pauseJob:\n{e}")
                self.parent().showMessage(f"Failed to pause the job for node {node}", "error")
            else:
                self.parent().showMessage(f"Paused node {node.label} on farm")
        elif not node.isExtern():
            self.parent().showMessage(f"PauseJob is only available in external computation mode!", "warning")
        else:
            self.parent().showMessage(f"Cannot retrieve the job", "error")

    @Slot(Node)
    def resumeJob(self, node: Node):
        """ Resume the paused job. """
        job = jobManager.getNodeJob(node)
        if job:
            # Node is submitted to farm
            try:
                job.resumeJob()
            except Exception as e:
                self.parent().showMessage(f"Failed to resume node {node.label} on farm")
                logging.warning(f"Error on resumeJob:\n{e}")
            else:
                self.parent().showMessage(f"Resumed the job for node {node}")
        else:
            # In this case user can just relaunch the node computation
            # Could be implemented if we had a paused state on the task manager
            # Where unprocessed nodes are retained
            pass

    @Slot(Node)
    def interruptJob(self, node: Node):
        """ Interrupt the job that processes the node. """
        job = jobManager.getNodeJob(node)
        if job:
            try:
                job.interruptJob()
            except Exception as e:
                self.parent().showMessage(f"Failed to interrupt node {node.label} on farm", "error")
                logging.warning(f"Error on interruptJob:\n{e}")
            else:
                for chunk in self._sortedDFSChunks:
                    if jobManager.getNodeJob(chunk.node) == job:
                        if chunk._status.status in (Status.SUBMITTED, Status.RUNNING):
                            chunk.updateStatusFromCache()
                            chunk.upgradeStatusTo(Status.STOPPED)
                for _node in self._graph.dfsOnFinish(None)[0]:
                    if jobManager.getNodeJob(_node) == job and not _node._chunksCreated and \
                        _node._nodeStatus.status in (Status.SUBMITTED, Status.RUNNING):
                        _node.upgradeStatusTo(Status.STOPPED)
                self.parent().showMessage(f"Interrupted the job for node {node}")
        elif not node.isExtern():
            for chunk in self._sortedDFSChunks:
                if not chunk.isExtern() and chunk._status.status in (Status.SUBMITTED, Status.RUNNING):
                    chunk.updateStatusFromCache()
                    chunk.upgradeStatusTo(Status.STOPPED)
            for node in self._graph.dfsOnFinish(None)[0]:
                if not node.isExtern() and not node._chunksCreated and \
                    node._nodeStatus.status in (Status.SUBMITTED, Status.RUNNING):
                    node.upgradeStatusTo(Status.STOPPED)
            self.stopExecution()
            self.parent().showMessage(f"Stopped the local job process")
        else:
            self.parent().showMessage(f"Could not retrieve job for node {node}", "error")

    @Slot(Node)
    def restartJobErrorTasks(self, node: Node):
        """ Restart all tasks in the job that have failed. """
        job = jobManager.getNodeJob(node)
        if job:
            try:
                # Fist update status of each chunk to submitted
                for chunk in self._sortedDFSChunks:
                    if chunk._status.status not in (Status.ERROR, Status.STOPPED, Status.KILLED):
                        continue
                    if jobManager.getNodeJob(chunk.node) == job:
                        chunk.upgradeStatusTo(Status.SUBMITTED)
                for node in self._graph.dfsOnFinish(None)[0]:
                    if not node._chunksCreated and node._nodeStatus.status in (Status.ERROR, Status.STOPPED, Status.KILLED):
                        node.upgradeStatusTo(Status.SUBMITTED)
                job.restartErrorTasks()
                job.resumeJob()
            except Exception as e:
                self.parent().showMessage(f"Failed to restart error tasks for node {node.label} on farm", "error")
                logging.warning(f"Error on restartJobErrorTasks:\n{e}")
            else:
                self.parent().showMessage(f"Restarted error tasks for the node {node}")
        else:
            # In this case user can just relaunch the node computation
            # Could be implemented if we had a paused state on the task manager
            # Where error/failed nodes are retained
            pass

    @Slot()
    @Slot(Node)
    @Slot(list)
    def submit(self, nodes: Optional[Union[list[Node], Node]] = None):
        """
        Submit the graph to the default Submitter.
        If a node is specified, submit this node and its uncomputed predecessors.
        Otherwise, submit the whole

        Notes:
            Default submitter is specified using the MESHROOM_DEFAULT_SUBMITTER environment variable.
        """
        nodes = [nodes] if not isinstance(nodes, Iterable) and nodes else nodes
        if len(self._subGraphStack):
            nodes = [self._subGraphStack.levels[0].node]
            self.exitSubGraphTo(0)
        self.save()  # graph must be saved before being submitted
        self._undoStack.clear()  # the undo stack must be cleared
        mrDefaultSubmitter = os.environ.get('MESHROOM_DEFAULT_SUBMITTER', '')
        chosenSubmitter = self.parent()._defaultSubmitterName or mrDefaultSubmitter
        self.parent().showMessage(f"Submit job on farm through {chosenSubmitter}")
        self.parent().showMessage(f"Nodes to submit : {nodes}")
        self._taskManager.submit(self._graph, chosenSubmitter, nodes, submitLabel=self.submitLabel)

    def updateGraphComputingStatus(self):
        dfsNodes = self._graph.dfsOnFinish(None)[0]
        # TODO : these functions should go on the node part
        # We should do any([node.isRunning for node in dfsNodes])

        # update graph computing status
        computingLocally = any([
            ch._status.execMode == ExecMode.LOCAL and \
            (sessionUid in (ch.node._nodeStatus.submitterSessionUid, ch._status.computeSessionUid)) and \
            (ch._status.status in (Status.RUNNING, Status.SUBMITTED))
            for ch in self._sortedDFSChunks
        ])
        # Note: We do not check computeSessionUid for the submitted status,
        #       as the source instance of the submit has no importance.
        submitted = any([ch._status.execMode == ExecMode.EXTERN and ch._status.status in (Status.RUNNING, Status.SUBMITTED) for ch in self._sortedDFSChunks])

        # Handle nodes with uninitialized chunks
        for node in dfsNodes:
            if node._chunksCreated:
                continue
            if node._nodeStatus.status in (Status.RUNNING, Status.SUBMITTED):
                # TODO : save session ID in node
                if node._nodeStatus.execMode == ExecMode.LOCAL:
                    computingLocally = True
                elif node._nodeStatus.execMode == ExecMode.EXTERN:
                    submitted = True

        if self._computingLocally != computingLocally or self._submitted != submitted:
            self._computingLocally = computingLocally
            self._submitted = submitted
            self.computeStatusChanged.emit()

    def isComputing(self):
        """ Whether is graph is being computed, either locally or externally. """
        return self.isComputingLocally() or self.isComputingExternally()

    def isComputingExternally(self):
        """ Whether this graph is being computed externally. """
        return self._submitted

    def isComputingLocally(self):
        """ Whether this graph is being computed locally (i.e computation can be stopped). """
        ## One solution could be to check if the thread is still running,
        # but the latency in creating/stopping the thread can be off regarding the update signals.
        # isRunningThread = self._taskManager._thread.isRunning()
        ## Another solution is to retrieve the current status directly from all chunks status
        # isRunning = self._taskManager.hasRunningChunks()
        ## For performance reason, we use a precomputed value updated in updateGraphComputingStatus:
        return self._computingLocally

    def push(self, command):
        """
        Try and push the given command to the undo stack.

        Args:
            command (commands.UndoCommand): the command to push
        """
        return self._undoStack.tryAndPush(command)

    def groupedGraphModification(self, title, disableUpdates=True):
        """
        Get a GroupedGraphModification for this Graph.

        Args:
            title (str): the title of the macro command
            disableUpdates (bool): whether to disable graph updates

        Returns:
            GroupedGraphModification: the instantiated context manager
        """
        return commands.GroupedGraphModification(self._graph, self._undoStack, title, disableUpdates)

    @Slot(str)
    def beginModification(self, name):
        """
        Begin a Graph modification.
        Calls to beginModification and endModification may be nested, but
        every call to beginModification must have a matching call to endModification.
        """
        self._modificationCount += 1
        self._undoStack.beginMacro(name)

    @Slot()
    def endModification(self):
        """ Ends a Graph modification. Must match a call to beginModification. """
        assert self._modificationCount > 0
        self._modificationCount -= 1
        self._undoStack.endMacro()

    @Slot(str, QPoint, result=QObject)
    def addNewNode(self, nodeType, position=None, **kwargs):
        """ [Undoable]
        Create a new Node of type 'nodeType' and returns it.

        Args:
            nodeType (str): the type of the Node to create.
            position (QPoint): (optional) the initial position of the node
            **kwargs: optional node attributes values

        Returns:
            Node: the created node
        """
        if isinstance(position, QPoint):
            position = Position(position.x(), position.y())
        return self.push(commands.AddNodeCommand(self._graph, nodeType, position=position, **kwargs))

    @Slot(Node, str, result=str)
    def renameNode(self, node: Node, newName: str):
        """ Triggers the node renaming.

        In this function the last `_N` index is removed, then all special characters
        (everything except letters and numbers) are removed.
        The name uniqueness will be ensured later by adding a suffix (e.g. `_1`, `_2`, ...)

        Labels can be used to have special characters in the displayed name.

        Args:
            node (Node): Node to rename.
            newName (str): New name to set.

        Returns:
            str: The final name of the node.
        """
        newName = "_".join(newName.split("_")[:-1]) if "_" in newName else newName
        # Eliminate all characters except digits and letters
        newName = re.sub(r"[^0-9a-zA-Z]", "", newName)
        # Create unique name
        uniqueName = self._graph._createUniqueNodeName(newName, {n._name for n in self._graph._nodes if n != node})
        if not newName or uniqueName == node._name:
            return ""
        return self.push(commands.RenameNodeCommand(self._graph, node, uniqueName))

    def moveNode(self, node: Node, position: Position):
        """
        Move `node` to the given `position`.

        Args:
            node: The node to move.
            position: The target position.
        """
        self.push(commands.MoveNodeCommand(self._graph, node, position))

    @Slot(BackdropNode, int, int)
    def resizeNode(self, node, width, height):
        """
        Resize `node` to the given `width` and `height`.

        Args:
            node: The node to resize.
            width: The target width.
            height: The target height.
        """
        with self.groupedGraphModification("Resize Node"):
            if node.hasInternalAttribute("nodeWidth"):
                self.setAttribute(node.internalAttribute("nodeWidth"), width)
            if node.hasInternalAttribute("nodeHeight"):
                self.setAttribute(node.internalAttribute("nodeHeight"), height)

    @Slot(BackdropNode, int, int, QPoint)
    def resizeAndMoveNode(self, node, width, height, position=None):
        """
        Resize `node` to the given `width` and `height`, and move it to the given `position`.

        Args:
            node: The node to resize and move.
            width: The target width.
            height: The target height.
            position: The target position.
        """
        with self.groupedGraphModification("Resize and Move Node"):
            if node.hasInternalAttribute("nodeWidth"):
                self.setAttribute(node.internalAttribute("nodeWidth"), width)
            if node.hasInternalAttribute("nodeHeight"):
                self.setAttribute(node.internalAttribute("nodeHeight"), height)

            if position:
                self.moveNode(node, Position(position.x(), position.y()))

    @Slot(QPoint, int, int, result=QObject)
    def addBackdropNode(self, position, width, height):
        """[Undoable]
        Create a new Backdrop Node at the given position with the given dimensions as a single undo entry.

        Args:
            position (QPoint): the position of the backdrop node.
            width (int): the width of the backdrop node.
            height (int): the height of the backdrop node.

        Returns:
            BackdropNode: the created node.
        """
        with self.groupedGraphModification("Add Backdrop"):
            node = self.addNewNode("Backdrop", position)
            if node.hasInternalAttribute("nodeWidth"):
                self.setAttribute(node.internalAttribute("nodeWidth"), width)
            if node.hasInternalAttribute("nodeHeight"):
                self.setAttribute(node.internalAttribute("nodeHeight"), height)
        return node

    @Slot(QPoint)
    def moveSelectedNodesBy(self, offset: QPoint):
        """ Move all the selected nodes by the given `offset`. """

        with self.groupedGraphModification("Move Selected Nodes"):
            for node in self.iterSelectedNodes():
                position = Position(node.x + offset.x(), node.y + offset.y())
                self.moveNode(node, position)

    def getMeanPosition(self):
        """ Get the average Position of selected nodes. """
        # Not great, would be better if Position was a non-tuple class
        selectedNodes = self.getSelectedNodes()
        sum_pose = [0, 0]
        nb_tot = 0
        for selectedNode in selectedNodes:
            sum_pose[0] += selectedNode.x
            sum_pose[1] += selectedNode.y
            nb_tot += 1
        return Position(int(sum_pose[0] / nb_tot), int(sum_pose[1] / nb_tot))

    @Slot()
    def alignHorizontally(self):
        """ All nodes are moved horizontally to the same position, on an average position of selected nodes. """
        nodePadding = 50
        selectedNodes = self.getSelectedNodes()
        if len(selectedNodes) < 2:
            return

        # Make sure the list is correctly ordered
        selectedNodes = sorted(selectedNodes, key=lambda node:node.x)

        meanX, meanY = self.getMeanPosition()
        nodeWidth = self.layout.nodeWidth
        # Compute the first node X position
        totalWidth = len(selectedNodes) * nodeWidth + (len(selectedNodes) - 1) * nodePadding
        startX = int(meanX - totalWidth / 2 + nodeWidth / 2)
        with self.groupedGraphModification("Align nodes horizontally"):
            for i, selectedNode in enumerate(selectedNodes):
                x = startX + i * (nodeWidth + nodePadding)
                self.moveNode(selectedNode, Position(x, meanY))

    @Slot("QVariant")
    def layoutNodes(self, sizes=None):
        """
        Lay out the selected nodes (all the nodes without selection) following their connections, as one
        undoable change (see meshroom.core.graphLayout): left to right, the inputs of a node stacked in
        the order of its inputs (Merge: B on top, A below). The hidden edges and the backdrops are left out.

        Args:
            sizes: {node name: [width, height]} of the node items (an estimate is used otherwise).
        """
        from meshroom.core import graphLayout
        sizes = sizes or {}
        if hasattr(sizes, "toVariant"):
            sizes = sizes.toVariant()
        nodes = [node for node in (self.getSelectedNodes() or list(self._graph.nodes))
                 if not isinstance(node, BackdropNode) and node.nodeType not in ("Backdrop", "Chapter")]
        if len(nodes) < 2:
            return

        def size(node):
            value = sizes.get(node.name)
            if value:
                return float(value[0]), float(value[1])
            pins = sum(1 for attr in node.attributes if not attr.isOutput and attr.desc.visible
                       and (attr.desc.exposed or attr.type in ("Image", "File")))
            return 180.0, 40.0 + 16.0 * max(1, pins)

        def skip(edge):
            # Hidden edges (and the link of a Clone to its cloned node) do not pull the nodes together
            return self.isEdgeHidden(edge.dst) or (edge.dst.node.nodeType == "Clone" and edge.dst.name == "source")

        edges = graphLayout.layoutEdges(self._graph, nodes, skip)
        origin = (min(node.x for node in nodes), min(node.y for node in nodes))
        positions = graphLayout.treeLayout(nodes, edges, size, origin, lambda node: (node.x, node.y))
        with self.groupedGraphModification("Layout Nodes"):
            for node, (x, y) in positions.items():
                if (int(x), int(y)) != (node.x, node.y):
                    self.moveNode(node, Position(int(x), int(y)))

    @Slot()
    def alignVertically(self):
        """ All nodes are moved vertically to the same position, on an average position of selected nodes. """
        selectedNodes = self.getSelectedNodes()
        if len(selectedNodes) < 2:
            return

        meanX, _ = self.getMeanPosition()
        with self.groupedGraphModification("Align nodes vertically"):
            for selectedNode in selectedNodes:
                self.moveNode(selectedNode, Position(meanX, selectedNode.y))

    @Slot(list)
    def removeNodes(self, nodes: list[Node]):
        """
        Remove 'nodes' from the graph.

        Args:
            nodes: The nodes to remove.
        """
        if any(n.locked for n in nodes):
            return

        reconnections = self._imageReconnections(nodes)
        with self.groupedGraphModification("Remove Nodes"):
            for node in nodes:
                self.push(commands.RemoveNodeCommand(self._graph, node))
            # Compositing chains stay connected: the input of a removed node goes to its outputs
            for srcName, dstName in reconnections:
                src, dst = self._graph.attribute(srcName), self._graph.attribute(dstName)
                if src is not None and dst is not None and not dst.isLink:
                    self.addEdge(src, dst)

    def _imageReconnections(self, nodes):
        """
        The image links to create when `nodes` are removed, as (source, destination) attribute full
        names: each image input fed by a removed compositing node gets the image entering that node
        (its first image input, through the other removed nodes), as Nuke does.
        """
        removed = {node.name for node in nodes}

        def upstreamSource(node, seen):
            if node.name in seen:
                return None
            seen.add(node.name)
            first = next((attr for attr in node.attributes if not attr.isOutput and attr.type == "Image"
                          and attr.desc.visible and not attr.desc.infoOnly), None)
            if first is None or not first.isLink:
                return None
            source = first.inputLink
            if source.node.name in removed:
                return upstreamSource(source.node, seen) if source.node.isImageOpNode else None
            return source

        reconnections = []
        for node in nodes:
            if not node.isImageOpNode:
                continue
            for edge in list(self._graph.edges):
                if edge.src.node is not node or edge.dst.node.name in removed or edge.dst.type != "Image" \
                        or edge.dst.desc.infoOnly:
                    continue
                source = upstreamSource(node, set())
                if source is not None and edge.dst.validateIncomingConnection(source):
                    reconnections.append((source.fullName, edge.dst.fullName))
        return reconnections

    @Slot()
    def removeSelectedNodes(self):
        """ Remove selected nodes from the graph. """
        self.removeNodes(list(self.iterSelectedNodes()))

    @Slot(list)
    def removeNodesFrom(self, nodes: list[Node]):
        """
        Remove all nodes starting from 'nodes' to graph leaves.

        Args:
            nodes: the nodes to start from.
        """
        with self.groupedGraphModification("Remove Nodes From Selected Nodes"):
            nodesToRemove, _ = self._graph.dfsOnDiscover(startNodes=nodes, reverse=True, dependenciesOnly=True)
            # Filter out nodes that will be removed more than once
            uniqueNodesToRemove = list(dict.fromkeys(nodesToRemove))
            # Perform nodes removal from leaves to start node so that edges can be re-created in correct order on redo
            self.removeNodes(list(reversed(uniqueNodesToRemove)))

    @Slot(list, result=list)
    def duplicateNodes(self, nodes: list[Node]) -> list[Node]:
        """
        Duplicate 'nodes'.

        Args:
            nodes: the nodes to duplicate.

        Returns:
            The list of duplicated nodes.
        """
        nPositions = [(n.x, n.y) for n in self._graph.nodes]
        # Enable updates between duplication and layout to get correct depths during layout
        with self.groupedGraphModification("Duplicate Selected Nodes", disableUpdates=False):
            # Disable graph updates during duplication
            with self.groupedGraphModification("Node duplication", disableUpdates=True):
                duplicates = self.push(commands.DuplicateNodesCommand(self._graph, nodes))
            # Move nodes below the bounding box formed by the duplicated node(s)
            bbox = self._layout.boundingBox(nodes)

            for n in duplicates:
                yPos = n.y + self.layout.gridSpacing + bbox[3]
                if (n.x, yPos) in nPositions:
                    # Make sure the node will not be moved on top of another node
                    while (n.x, yPos) in nPositions:
                        yPos = yPos + self.layout.gridSpacing + self.layout.nodeHeight
                    self.moveNode(n, Position(n.x, yPos))
                else:
                    self.moveNode(n, Position(n.x, bbox[3] + self.layout.gridSpacing + n.y))
                nPositions.append((n.x, n.y))

        return duplicates

    @Slot(list, result=list)
    def duplicateNodesFrom(self, nodes: list[Node]) -> list[Node]:
        """
        Duplicate all nodes starting from 'nodes' to graph leaves.

        Args:
            node: The nodes to start from.
        Returns:
            The list of duplicated nodes.
        """
        with self.groupedGraphModification("Duplicate Nodes From Selected Nodes"):
            nodesToDuplicate, _ = self._graph.dfsOnDiscover(startNodes=nodes, reverse=True, dependenciesOnly=True)
            # Filter out nodes that will be duplicated more than once
            uniqueNodesToDuplicate = list(dict.fromkeys(nodesToDuplicate))
            duplicates = self.duplicateNodes(uniqueNodesToDuplicate)
        return duplicates

    @Slot(Edge, result=bool)
    def canExpandForLoop(self, currentEdge):
        """ Check if the list attribute can be expanded by looking at all the edges connected to it. """
        listAttribute = currentEdge.src.root
        # Check that the parent is indeed a ListAttribute (it could be a GroupAttribute, for example)
        if not listAttribute or not isinstance(listAttribute, ListAttribute):
            return False
        srcIndex = listAttribute.index(currentEdge.src)
        allSrc = [e.src for e in self._graph.edges.values()]
        for i in range(len(listAttribute)):
            if i == srcIndex:
                continue
            if listAttribute.at(i) in allSrc:
                return False
        return True

    @Slot(Edge, result=Edge)
    def expandForLoop(self, currentEdge):
        """ Expand 'node' by creating all its output nodes. """
        with self.groupedGraphModification("Expand For Loop Node"):
            listAttribute = currentEdge.src.root
            dst = currentEdge.dst

            for i in range(1, len(listAttribute)):
                duplicates = self.duplicateNodesFrom([dst.node])
                newNode = duplicates[0]
                previousEdge = self.graph.edge(newNode.attribute(dst.name))
                self.replaceEdge(previousEdge, listAttribute.at(i), previousEdge.dst)

            # Last, replace the edge with the first element of the list
            return self.replaceEdge(currentEdge, listAttribute.at(0), dst)

    @Slot(Edge)
    def collapseForLoop(self, currentEdge):
        """ Collapse 'node' by removing all its output nodes. """
        with self.groupedGraphModification("Collapse For Loop Node"):
            listAttribute = currentEdge.src.root
            srcIndex = listAttribute.index(currentEdge.src)
            allSrc = [e.src for e in self._graph.edges.values()]
            for i in reversed(range(len(listAttribute))):
                if i == srcIndex:
                    continue
                occurrence = allSrc.index(listAttribute.at(i)) if listAttribute.at(i) in allSrc else -1
                if occurrence != -1:
                    self.removeNodesFrom([self.graph.edges.at(occurrence).dst.node])
                    # update the edges from allSrc
                    allSrc = [e.src for e in self._graph.edges.values()]

    @Slot()
    def clearSelectedNodesData(self):
        """ Clear data from all selected nodes. """
        self.clearData(self.iterSelectedNodes())

    @Slot(list)
    def clearData(self, nodes: list[Node]):
        """ Clear data from 'nodes'. """
        for n in nodes:
            n.clearData()

    @Slot(list)
    def clearDataFrom(self, nodes: list[Node]):
        """
        Clear data from all nodes starting from 'nodes' to graph leaves.

        Args:
            nodes: The nodes to start from.
        """
        self.clearData(self._graph.dfsOnDiscover(startNodes=nodes, reverse=True, dependenciesOnly=True)[0])

    @Slot(Attribute, Attribute)
    def addEdge(self, src, dst):
        if isinstance(src, AnySet):
            self.push(commands.ConnectAnySetCommand(self._graph, src=dst, anySetAttribute=src))
        elif isinstance(dst, AnySet):
            self.push(commands.ConnectAnySetCommand(self._graph, src=src, anySetAttribute=dst))
        elif isinstance(src, ListAttribute) and not isinstance(dst, ListAttribute):
            self._addEdge(src.at(0), dst)
        elif isinstance(dst, ListAttribute) and not isinstance(src, ListAttribute):
            with self.groupedGraphModification(f"Insert and Add Edge on {dst.fullName}"):
                self.appendAttribute(dst)
                self._addEdge(src, dst.at(-1))
        else:
            self._addEdge(src, dst)

    def _addEdge(self, src, dst):
        with self.groupedGraphModification(f"Connect '{src.fullName}'->'{dst.fullName}'"):
            if dst in self._graph.edges.keys():
                self.removeEdge(self._graph.edge(dst))
            self.push(commands.AddEdgeCommand(self._graph, src, dst))

    @Slot(Edge)
    def removeEdge(self, edge):
        with self.groupedGraphModification(f"Remove Edge and Delete {edge.dst.fullName}"):
            if isinstance(edge.dst.root, ListAttribute):
                self.push(commands.RemoveEdgeCommand(self._graph, edge))
                self.removeAttribute(edge.dst)
                return
            self.push(commands.RemoveEdgeCommand(self._graph, edge))

    # Disconnect and Bake (edge menu): the value the input received stays, as keys when it changes over time

    @Slot(Edge, result=bool)
    def canBakeEdge(self, edge) -> bool:
        """ Whether "Disconnect and Bake" applies to an edge: a parameter, or a stream its node turns into parameters. """
        from meshroom.live import bake
        return edge is not None and not edge.dst.node.locked and not edge.dst.isReadOnly and bake.canBake(edge.dst)

    def _bakeFrames(self) -> list:
        """ The frames baked: the frame range of the timeline, or the current frame. """
        live = self._live
        if live.hasFrameRange and live.lastFrame >= live.firstFrame:
            return list(range(int(live.firstFrame), int(live.lastFrame) + 1))
        return [int(live.currentFrame)]

    @Slot(Edge)
    def disconnectAndBake(self, edge):
        """
        Remove an edge, keeping in its input the value it received: the value at the current frame, or keys when it
        changes over time (an animated parameter, or a camera / matrix stream turned into the parameters of the node,
        over the frame range of the timeline: see meshroom/live/bake.py). Undoable.
        """
        from meshroom.live import Evaluator, bake
        if not self.canBakeEdge(edge):
            return
        node, frame = edge.dst.node, int(self._live.currentFrame)
        try:
            baked = bake.bakedValues(edge.dst, self._bakeFrames(), frame, Evaluator(cache=self._live._evaluator.cache))
            changes, animation = bake.applyToNode(node, baked, frame)
        except Exception as e:
            logging.exception("Disconnect and Bake")
            self.parent().showMessage(f"Disconnect and Bake: {getattr(e, 'message', e)}", "error")
            return
        with self.groupedGraphModification(f"Disconnect and Bake {edge.dst.fullName}"):
            self.removeEdge(edge)
            for attribute, value in changes:
                self.push(commands.SetAttributeCommand(self._graph, attribute, value))
            if animation is not None:
                self.push(commands.SetAttributeCommand(self._graph, node.internalAttribute("animation"), animation))

    @Slot(list)
    def deleteEdgesByIndices(self, indices):
        with self.groupedGraphModification("Remove Edges"):
            copied = list(self._graph.edges)
            for index in indices:
                self.removeEdge(copied[index])

    @Slot()
    def disconnectSelectedNodes(self):
        with self.groupedGraphModification("Disconnect Nodes"):
            selectedNodes = self.getSelectedNodes()
            for edge in self._graph.edges[:]:
                # Remove only the edges which are coming or going out of the current selection
                if edge.src.node in selectedNodes and edge.dst.node in selectedNodes:
                    continue

                if edge.dst.node in selectedNodes or edge.src.node in selectedNodes:
                    self.removeEdge(edge)

    @Slot(Edge, Attribute, Attribute, result=Edge)
    def replaceEdge(self, edge, newSrc, newDst):
        with self.groupedGraphModification(f"Replace Edge '{edge.src.fullName}'->'{edge.dst.fullName}' with '{newSrc.fullName}'->'{newDst.fullName}'"):
            self.removeEdge(edge)
            self.addEdge(newSrc, newDst)
        return self._graph.edge(newDst)

    @Slot(Attribute, result=Edge)
    def getEdge(self, dst):
        return self._graph.edge(dst)

    @Slot(Attribute, "QVariant")
    def setAttribute(self, attribute, value):
        animation = self._animationOf(attribute)
        if animation is None:
            self.push(commands.SetAttributeCommand(self._graph, attribute, value))
            return
        # Animated parameter of a compositing node: the change is also a key at the current frame.
        # The values are not mirrored from the keys in between (the new value would be lost).
        self._mirroring = True
        try:
            with self.groupedGraphModification(f"Set Key '{attribute.fullName}'"):
                self.push(commands.SetAttributeCommand(self._graph, attribute, value))
                self.push(commands.SetAttributeCommand(self._graph, animation, self._keyedAnimation(attribute)))
        finally:
            self._mirroring = False

    # Frame range of the timeline (compositing), saved with the graph (Graph.frameRange)

    @property
    def rootGraph(self) -> Graph:
        """ The project graph (the graph shown, or the outermost one when a subgraph is entered). """
        return self._subGraphStack.rootGraph if len(self._subGraphStack) else self._graph

    def _updateFrameRange(self):
        """
        Keep the frame range of the timeline (entered by the user, or following the first Read of the graph shown: see
        meshroom/live/frameRange.py). It is the range of the project, saved in its header (also inside a subgraph).
        """
        from meshroom.live.frameRange import followedFrameRange
        root = self.rootGraph
        value, self._rangeSourceUid = followedFrameRange(self._graph, self._live.nodeInfo, self._rangeSourceUid,
                                                         frameRange=root.frameRange)
        if value is not None:
            root.setFrameRange(value)

    def _onFrameRangeChanged(self):
        frameRange = self.rootGraph.frameRange
        if "first" in frameRange and "last" in frameRange:
            self._live.setProjectRange(int(frameRange["first"]), int(frameRange["last"]))
        else:
            self._live.setProjectRange(None, None)
        self.frameRangeChanged.emit()

    @Slot(int, int)
    def setFrameRange(self, first, last):
        """ Set the frame range of the timeline by hand: it no longer follows a Read. Undoable. """
        first, last = int(first), int(last)
        if last < first:
            first, last = last, first
        self.push(commands.SetFrameRangeCommand(self.rootGraph, {"first": first, "last": last, "source": ""}))

    @Slot()
    def resetFrameRange(self):
        """ The frame range of the timeline follows the first Read of the graph again. Undoable. """
        from meshroom.live.frameRange import followedFrameRange
        value, self._rangeSourceUid = followedFrameRange(self._graph, self._live.nodeInfo, None, frameRange={})
        self.push(commands.SetFrameRangeCommand(self.rootGraph, value or {}))

    frameRangeChanged = Signal()
    frameRangeIsUser = Property(bool, lambda self: bool(self.rootGraph.frameRange) and
                                self.rootGraph.frameRange.get("source", "") == "",
                                notify=frameRangeChanged)

    # Animation of the parameters of compositing nodes (keys per frame, see meshroom/live/animation.py)

    @staticmethod
    def _topAttribute(attribute):
        """ The parameter an attribute belongs to (a group for its children: x of a 2D value, r of a color). """
        while attribute.root is not None:
            attribute = attribute.root
        return attribute

    def _animationOf(self, attribute):
        """ The "animation" internal attribute of the node when `attribute` is an animated parameter, else None. """
        from meshroom.live import animation as anim
        node = attribute.node
        if attribute.isOutput or not node.hasInternalAttribute(anim.ANIMATION):
            return None
        top = self._topAttribute(attribute)
        if top.node is not node or node.hasInternalAttribute(top.name) and node.internalAttribute(top.name) is top:
            return None
        animation = node.internalAttribute(anim.ANIMATION)
        return animation if top.name in anim.parseAnimation(animation.value) else None

    def _keyedAnimation(self, attribute) -> str:
        """ The animation of the node of `attribute` with a key of its current value at the current frame. """
        from meshroom.live import animation as anim
        top = self._topAttribute(attribute)
        animation = top.node.internalAttribute(anim.ANIMATION)
        keys = anim.setKey(anim.parseAnimation(animation.value), top.name, self._live.currentFrame,
                           top.getSerializedValue())
        return anim.dumpAnimation(keys)

    @Slot(Attribute, result=str)
    def keyState(self, attribute):
        """ "key" (a key at the current frame), "animated" (keys elsewhere) or "" for a parameter. """
        from meshroom.live import animation as anim
        node = attribute.node
        if not node.hasInternalAttribute(anim.ANIMATION):
            return ""
        keys = anim.parseAnimation(node.internalAttribute(anim.ANIMATION).value).get(self._topAttribute(attribute).name)
        if not keys:
            return ""
        return "key" if any(frame == self._live.currentFrame for frame, _ in keys) else "animated"

    @Slot(Attribute)
    def toggleKey(self, attribute):
        """ Set a key of a parameter at the current frame, or remove it if there is one. Undoable. """
        from meshroom.live import animation as anim
        top = self._topAttribute(attribute)
        node = top.node
        if not node.hasInternalAttribute(anim.ANIMATION):
            return
        animationAttr = node.internalAttribute(anim.ANIMATION)
        animation = anim.parseAnimation(animationAttr.value)
        frame = self._live.currentFrame
        if any(f == frame for f, _ in animation.get(top.name, [])):
            animation = anim.removeKey(animation, top.name, frame)
        else:
            animation = anim.setKey(animation, top.name, frame, top.getSerializedValue())
        self.push(commands.SetAttributeCommand(self._graph, animationAttr, anim.dumpAnimation(animation)))

    @Slot(Attribute)
    def removeAnimation(self, attribute):
        """ Remove all the keys of a parameter: it keeps its value at the current frame. Undoable. """
        from meshroom.live import animation as anim
        top = self._topAttribute(attribute)
        animationAttr = top.node.internalAttribute(anim.ANIMATION) if top.node.hasInternalAttribute(anim.ANIMATION) else None
        if animationAttr is None:
            return
        animation = anim.parseAnimation(animationAttr.value)
        animation.pop(top.name, None)
        self.push(commands.SetAttributeCommand(self._graph, animationAttr, anim.dumpAnimation(animation)))

    def _mirrorAnimation(self, *args):
        """
        Show the values of the animated parameters of the selected node at the current frame: their
        stored value mirrors their keys (it is not part of the uid, and not undoable).
        """
        from meshroom.live import animation as anim
        node = self._selectedNode
        # Not while playing (the values are mirrored when the playback stops), nor from its own changes
        if self._live.playDirection != 0 or self._mirroring:
            return
        try:
            if node is None or not node.hasInternalAttribute(anim.ANIMATION):
                return
        except RuntimeError:  # The underlying Qt object has been deleted
            return
        animation = anim.parseAnimation(node.internalAttribute(anim.ANIMATION).value)
        frame = self._live.currentFrame
        self._mirroring = True
        try:
            for name, keys in animation.items():
                if not node.hasAttribute(name) or node.attribute(name).isLink:
                    continue
                # Being edited interactively (the parameter or one of its components, e.g. "gain.r")
                fullName = node.attribute(name).fullName
                if any(edited == fullName or edited.startswith(fullName + ".") for edited in self._interactiveEdits):
                    continue
                attribute = node.attribute(name)
                value = anim.valueAt(keys, frame)
                if attribute.getSerializedValue() != value:
                    attribute.value = value
        finally:
            self._mirroring = False

    @Slot(Attribute, float, float)
    def setVec2Attribute(self, attribute, x, y):
        """ Set both components of a 2D attribute ({x, y}) in a single undoable step. """
        self.setAttribute(attribute, {"x": x, "y": y})

    @Slot(Attribute, float, float, float, float)
    def setRgbaAttribute(self, attribute, r, g, b, a):
        """ Set the 4 components of a color attribute ({r, g, b, a}) in a single undoable step. """
        self.setAttribute(attribute, {"r": r, "g": g, "b": b, "a": a})

    def _updateSelectionOrder(self, *args):
        """ Keep the selected node names in selection order: new ones are appended. """
        selected = [node.name for node in self.iterSelectedNodes()]
        selectedSet = set(selected)
        order = [name for name in self._selectionOrder if name in selectedSet]
        order += [name for name in selected if name not in order]
        self._selectionOrder = order

    @staticmethod
    def _mainOutput(node):
        """ The output used when connecting a node: its image output, or its first connectable output. """
        outputs = [attr for attr in node.attributes if attr.isOutput]
        if not outputs and node.hasInternalAttribute("flowOutput"):
            return node.internalAttribute("flowOutput")
        for attr in outputs:
            if attr.type == "Image":
                return attr
        return outputs[0] if outputs else None

    # Graph editor: edges selected, hidden (H), pulled from an output to an input

    @staticmethod
    def _relativeName(attribute):
        """ Name of an attribute within its node (e.g. "input", "inputs[0]"). """
        return attribute.fullName.split(".", 1)[1]

    def _edgesTo(self, dstNames):
        """ The edges whose destination attribute full name is in `dstNames`. """
        names = set(dstNames)
        return [edge for edge in self._graph.edges if edge.dst.fullName in names]

    @Slot("QVariantList")
    def removeEdgesTo(self, dstNames):
        """ Remove the edges going to the given destination attributes (full names), as one undoable step. """
        edges = self._edgesTo(dstNames)
        if not edges:
            return
        with self.groupedGraphModification("Remove Edges"):
            for edge in edges:
                if not edge.dst.node.locked:
                    self.removeEdge(edge)

    @Slot(QObject, result=bool)
    def isEdgeHidden(self, dstAttribute):
        """ Whether the edge going to `dstAttribute` is hidden (compositing nodes, "hiddenInputs" internal attribute). """
        node = dstAttribute.node
        if not node.hasInternalAttribute("hiddenInputs"):
            return False
        return self._relativeName(dstAttribute) in node.internalAttribute("hiddenInputs").value.split(",")

    @Slot("QVariantList")
    def toggleHiddenEdges(self, dstNames):
        """
        Hide the given edges (destination attribute full names), or show them if they are all hidden.
        Only the edges going into compositing nodes can be hidden; saved with the graph, undoable.
        """
        edges = [edge for edge in self._edgesTo(dstNames) if edge.dst.node.hasInternalAttribute("hiddenInputs")]
        if not edges:
            return
        hide = not all(self.isEdgeHidden(edge.dst) for edge in edges)
        byNode = {}
        for edge in edges:
            byNode.setdefault(edge.dst.node, []).append(self._relativeName(edge.dst))
        with self.groupedGraphModification("Hide Edges" if hide else "Show Edges"):
            for node, names in byNode.items():
                attribute = node.internalAttribute("hiddenInputs")
                current = [name for name in attribute.value.split(",") if name]
                if hide:
                    current += [name for name in names if name not in current]
                else:
                    current = [name for name in current if name not in names]
                self.setAttribute(attribute, ",".join(current))

    @Slot()
    def toggleHiddenInputEdgesOfSelectedNodes(self):
        """ Hide the input edges of the selected nodes (or show them if they are all hidden). """
        selected = set(self.getSelectedNodes())
        self.toggleHiddenEdges([edge.dst.fullName for edge in self._graph.edges if edge.dst.node in selected])

    @Slot(Attribute, Attribute, result=bool)
    def connectAttributes(self, src, dst):
        """ Connect an output to an input (an edge pulled from an output, then a click on an input), if allowed. """
        if src is None or dst is None or dst.isOutput or not src.isOutput or src.node is dst.node:
            return False
        if dst.isReadOnly or dst.node.locked or not dst.validateIncomingConnection(src) or \
                self._createsCycle(src.node, dst.node):
            return False
        with self.groupedGraphModification(f"Connect '{src.fullName}' to '{dst.fullName}'"):
            self.addEdge(src, dst)
            # A new edge into a hidden input is shown
            if self.isEdgeHidden(dst):
                self.toggleHiddenEdges([dst.fullName])
        return True

    @Slot(str, QPoint, Attribute, result=QObject)
    def addNewNodeConnectedTo(self, nodeType, position, src):
        """ Create a node (Tab while pulling an edge from an output) with `src` connected to its first compatible input. """
        with self.groupedGraphModification(f"Add Node {nodeType}"):
            node = self.addNewNode(nodeType, position)
            if node is not None and src is not None:
                dst = self._inputToConnect(node, src)
                if dst is not None:
                    self.addEdge(src, dst)
        return node

    @Slot(str, QObject, result=QObject)
    def addNewNodeAfter(self, nodeType, previous):
        """
        Create a compositing node after `previous` (nodes created one after the other, as in Nuke): placed on
        its right (below the nodes already there), its first compatible input connected to the main output
        of `previous`. Undoable.
        """
        output = self._mainOutput(previous) if previous is not None else None
        width = self._layout.nodeWidth
        x = previous.x + width + self._layout._gridSpacing * 2
        y = previous.y
        # Below the nodes already there
        occupied = [(node.x, node.y) for node in self._graph.nodes if node is not previous]
        while any(abs(nx - x) < width and abs(ny - y) < 60 for nx, ny in occupied):
            y += 90
        with self.groupedGraphModification(f"Add Node {nodeType}"):
            node = self.addNewNode(nodeType, Position(x, y))
            if node is not None and output is not None:
                dst = self._inputToConnect(node, output)
                if dst is not None:
                    self.addEdge(output, dst)
        return node

    def _inputToConnect(self, node, source):
        """ The input of `node` to connect `source` to: the first compatible free input, else the first compatible one. """
        # Hidden inputs (e.g. the source of a Clone) are not candidates
        candidates = [attr for attr in node.attributes
                      if not attr.isOutput and attr.desc.visible and attr.validateIncomingConnection(source)]
        for attr in candidates:
            if not attr.isLink:
                return attr
        return candidates[0] if candidates else None

    def _createsCycle(self, source: Node, target: Node) -> bool:
        """ Whether connecting `source` into `target` would create a cycle. """
        if source is target:
            return True
        upstream = self._graph.getInputNodes(source, recursive=True, dependenciesOnly=False)
        return target in upstream

    @Slot(QObject)
    def addLookupPoints(self, node):
        """
        Add to the curves of a ColorLookup / ColorLookupHSV node the points moving its "sourceColor"
        to its "targetColor" (picked in the viewer). Undoable.
        """
        from meshroom.live import curves
        if node is None or not node.hasAttribute("curves") or not node.hasAttribute("sourceColor"):
            return
        source = [node.attribute("sourceColor").value.get(c).value for c in "rgb"]
        target = [node.attribute("targetColor").value.get(c).value for c in "rgb"]
        try:
            values = curves.parseCurves(node.attribute("curves").value)
        except ValueError:
            values = {}
        mode = "hsv" if node.nodeType == "ColorLookupHSV" else "rgb"
        self.setAttribute(node.attribute("curves"), json.dumps(curves.lookupPoints(values, mode, source, target)))

    @Slot(QObject, str, str, result=bool)
    def renameTrack(self, tracker, oldName, newName):
        """
        Rename a track of a Tracker node, and the links of the Roto shapes reading this Tracker.
        Refused (False) if the name is empty or already used. Undoable.
        """
        from meshroom.live import tracking
        newName = newName.strip()
        try:
            tracks = tracking.parseTracks(tracker.attribute("tracks").value)
        except ValueError:
            return False
        if not newName or any(track.get("name") == newName for track in tracks):
            return False
        for track in tracks:
            if track.get("name") == oldName:
                track["name"] = newName
        from meshroom.live import roto
        with self.groupedGraphModification(f"Rename Track '{oldName}'"):
            self.setAttribute(tracker.attribute("tracks"), json.dumps(tracks))
            for node in self._graph.nodes:
                if node is tracker or not node.isImageOpNode:
                    continue
                # The image inputs of the node reading this Tracker (through Dot, Clone, disabled nodes)
                inputs = [name for name in ("tracker", "tracker2", "tracker3") if node.hasAttribute(name) and
                          self._live.upstreamValuesNode(node.attribute(name)) is tracker]
                if not inputs:
                    continue
                if node.hasAttribute("shapes"):
                    # Roto: the links of its shapes
                    try:
                        shapes = json.loads(node.attribute("shapes").value or "[]")
                    except ValueError:
                        continue
                    changed = False
                    for shape in shapes:
                        links = roto.shapeLinks(shape)
                        for link in links:
                            if link.get("input", "tracker") in inputs and link.get("track") == oldName:
                                link["track"] = newName
                                changed = True
                        if links or "link" in shape:
                            shape.pop("link", None)
                            shape["links"] = links
                    if changed:
                        self.setAttribute(node.attribute("shapes"), json.dumps(shapes))
                elif node.hasAttribute("tracks") and not node.attribute("tracks").desc.semantic:
                    # CornerPin, Stabilize, MotionBlur: track names, comma separated
                    names = [name.strip() for name in node.attribute("tracks").value.split(",")]
                    if oldName in names:
                        renamed = ",".join(newName if name == oldName else name for name in names if name)
                        self.setAttribute(node.attribute("tracks"), renamed)
        return True

    @Slot(QObject, int, str, int)
    def setRotoShapeLinks(self, node, shapeIndex, linksJson, frame):
        """
        Set the track links of a Roto shape, the shape staying where it is at `frame`: its points are
        moved by the change of the link offsets at that frame. Undoable.
        """
        from meshroom.live import roto, tracking
        try:
            shapes = json.loads(node.attribute("shapes").value or "[]")
            links = json.loads(linksJson)
        except ValueError:
            return
        if not 0 <= shapeIndex < len(shapes):
            return
        tracks = {}
        for name in ("tracker", "tracker2", "tracker3"):
            source = self._live.upstreamValuesNode(node.attribute(name)) if node.hasAttribute(name) else None
            try:
                tracks[name] = tracking.parseTracks(source.attribute("tracks").value) \
                    if source is not None and source.hasAttribute("tracks") else []
            except ValueError:
                tracks[name] = []
        # In the pixels of the Roto, as when it renders (tracks made on another resolution)
        rotoFormat = self._live.nodeFormat(node)
        if rotoFormat is not None:
            tracks = {name: tracking.scaleTracks(values, *rotoFormat) for name, values in tracks.items()}
        shape = shapes[shapeIndex]
        before = roto.shapeAtFrame(shape, frame, tracks)["points"]
        shape.pop("link", None)
        shape["links"] = links
        after = roto.shapeAtFrame(shape, frame, tracks)["points"]
        deltas = [(b.get("x", 0.0) - a.get("x", 0.0), b.get("y", 0.0) - a.get("y", 0.0)) for b, a in zip(before, after)]

        def rebase(points):
            for point, (dx, dy) in zip(points, deltas):
                point["x"] = point.get("x", 0.0) + dx
                point["y"] = point.get("y", 0.0) + dy
        rebase(shape.get("points") or [])
        for key in shape.get("keys") or []:
            rebase(key.get("points") or [])
        self.setAttribute(node.attribute("shapes"), json.dumps(shapes))

    @Slot(QObject, "QVariantList", int, result=QObject)
    def createCornerPinFromTracks(self, tracker, names, frame):
        """
        Create a CornerPin driven by 4 tracks of a Tracker (reference: `frame`), next to it, with the
        Tracker connected to its "tracker" input. Undoable.
        """
        names = [str(name) for name in names]
        if tracker is None or len(names) != 4 or not tracker.hasAttribute("output"):
            return None
        with self.groupedGraphModification(f"CornerPin from {tracker.label}"):
            cornerPin = self.addNewNode("CornerPin", Position(tracker.x + 220, tracker.y + 60))
            self.addEdge(tracker.attribute("output"), cornerPin.attribute("tracker"))
            self.setAttribute(cornerPin.attribute("tracks"), ",".join(names))
            self.setAttribute(cornerPin.attribute("referenceFrame"), int(frame))
        self.selectedNode = cornerPin
        return cornerPin

    @Slot(QObject, result=QObject)
    def cloneNode(self, node):
        """
        Create a Clone of a compositing node, next to it: a node whose link to `node` is hidden and
        which outputs the result of `node`. Undoable.
        """
        if node is None or not node.isImageOpNode or not node.hasAttribute("output"):
            return None
        # Cloning a clone: the new clone points to the same reference
        if node.nodeType == "Clone":
            source = node.attribute("source")
            if not source.isLink:
                return None
            reference = source.inputLink.node
            with self.groupedGraphModification(f"Clone {reference.label}"):
                clone = self.addNewNode("Clone", Position(node.x + 40, node.y + 120))
                self.addEdge(source.inputLink, clone.attribute("source"))
            return clone
        with self.groupedGraphModification(f"Clone {node.label}"):
            clone = self.addNewNode("Clone", Position(node.x + 40, node.y + 120))
            self.addEdge(node.attribute("output"), clone.attribute("source"))
        return clone

    @Slot()
    def connectSelectedNodes(self):
        """
        Connect the output of the first selected node to an input of each of the other selected nodes
        (the first free compatible input, as when dropping a link on a node). Undoable.
        """
        self._updateSelectionOrder()
        nodes = [self._graph.node(name) for name in self._selectionOrder]
        nodes = [node for node in nodes if node is not None]
        if len(nodes) < 2:
            return
        source, targets = nodes[0], nodes[1:]
        output = self._mainOutput(source)
        if output is None:
            return
        with self.groupedGraphModification(f"Connect {source.label}"):
            for target in targets:
                if self._createsCycle(source, target):
                    continue
                attr = self._inputToConnect(target, output)
                if attr is not None and not (attr.isLink and attr.inputLink is output):
                    self.addEdge(output, attr)

    @Slot(Attribute)
    def copyAttribute(self, attribute):
        """ Remember a pin, to connect it to another one with `connectCopiedAttribute`. """
        self._copiedPin = (attribute.fullName, attribute.isOutput)
        self.copiedAttributeChanged.emit()

    def _getCopiedAttribute(self):
        """ The copied pin, resolved by name in the current graph (None if it no longer exists). """
        if self._copiedPin is None:
            return None
        fullName, isOutput = self._copiedPin
        try:
            attribute = self._graph.attribute(fullName)
        except Exception:
            return None
        if attribute is None or attribute.isOutput != isOutput:
            return None
        return attribute

    def _copiedConnection(self, attribute):
        """ The (source, destination) edge that connecting the copied pin to `attribute` would create, or None. """
        copied = self._getCopiedAttribute()
        if copied is None or attribute is None or copied is attribute:
            return None
        if copied.isOutput and not attribute.isOutput:
            source, destination = copied, attribute
        elif not copied.isOutput and attribute.isOutput:
            source, destination = attribute, copied
        elif not copied.isOutput and not attribute.isOutput and copied.isLink:
            # Two inputs: connect to the same source as the copied input
            source, destination = copied.inputLink, attribute
        else:
            return None
        if source.node is None or destination.node is None:
            return None
        if source.node is destination.node or not destination.validateIncomingConnection(source):
            return None
        if self._createsCycle(source.node, destination.node):
            return None
        return source, destination

    @Slot(Attribute, result=bool)
    def canConnectCopiedAttribute(self, attribute):
        return self._copiedConnection(attribute) is not None

    @Slot(Attribute)
    def connectCopiedAttribute(self, attribute):
        """ Connect the copied pin to `attribute` (output to input, or input to the same source). Undoable. """
        connection = self._copiedConnection(attribute)
        if connection is not None:
            self.addEdge(*connection)

    @staticmethod
    def _splitVec2Attribute(attribute):
        """ The internal attribute of a compositing node storing its split 2D values (saved with the graph), or None. """
        node = attribute.node
        return node.internalAttribute("splitVec2") if node.hasInternalAttribute("splitVec2") else None

    @Slot(Attribute, result=bool)
    def isVec2Split(self, attribute):
        """ Whether a linkable 2D attribute was split into x / y fields by the user. """
        stored = self._splitVec2Attribute(attribute)
        if stored is not None:
            return attribute.name in [name for name in stored.value.split(",") if name]
        return attribute.fullName in self._splitVec2

    @Slot(Attribute, bool)
    def setVec2Split(self, attribute, split):
        """ Show a linkable 2D attribute as x / y fields (or a single value). Saved with compositing nodes. """
        stored = self._splitVec2Attribute(attribute)
        if stored is None:
            if split:
                self._splitVec2.add(attribute.fullName)
            else:
                self._splitVec2.discard(attribute.fullName)
            return
        names = [name for name in stored.value.split(",") if name and name != attribute.name]
        if split:
            names.append(attribute.name)
        value = ",".join(sorted(names))
        if value != stored.value:
            # Not an undoable change of the graph: a display preference
            stored.value = value

    @Slot(Attribute)
    def beginInteractiveEdit(self, attribute):
        """
        Start an interactive edit (e.g. dragging a handle in the viewer): the following changes made with
        `setValueInteractively` / `setVec2Interactively` / `setRgbaInteractively` update the graph live
        without filling the undo stack, and `endInteractiveEdit` records them as a single undoable change.
        """
        # A connected attribute takes its value from its link: it cannot be edited interactively
        if attribute.isLink:
            return
        # An animated parameter: its keys are edited with it
        animation = self._animationOf(attribute)
        if animation is not None:
            self._interactiveEdits.setdefault(animation.fullName, animation.getSerializedValue())
        self._interactiveEdits.setdefault(attribute.fullName, attribute.getSerializedValue())

    def _syncKeyInteractively(self, attribute):
        """ During an interactive edit of an animated parameter: its key at the current frame follows. """
        animation = self._animationOf(attribute)
        if animation is not None and animation.fullName in self._interactiveEdits:
            animation.value = self._keyedAnimation(attribute)

    @Slot(Attribute, "QVariant")
    def setValueInteractively(self, attribute, value):
        if attribute.fullName in self._interactiveEdits:
            attribute.value = value
            self._syncKeyInteractively(attribute)

    @Slot(Attribute, float, float)
    def setVec2Interactively(self, attribute, x, y):
        if attribute.fullName in self._interactiveEdits:
            attribute.value = {"x": x, "y": y}
            self._syncKeyInteractively(attribute)

    def cancelInteractiveEdit(self, attribute):
        """ Forget an interactive edit (the changes stay, without undo step), e.g. when its node is gone. """
        self._interactiveEdits.pop(attribute.fullName, None)
        # And the edit of its keys
        from meshroom.live import animation as anim
        try:
            animationName = f"{attribute.node.name}.{anim.ANIMATION}"
        except RuntimeError:
            return
        if not any(name.startswith(attribute.node.name + ".") for name in self._interactiveEdits if name != animationName):
            self._interactiveEdits.pop(animationName, None)

    @Slot(Attribute, float, float, float, float)
    def setRgbaInteractively(self, attribute, r, g, b, a):
        if attribute.fullName in self._interactiveEdits:
            attribute.value = {"r": r, "g": g, "b": b, "a": a}
            self._syncKeyInteractively(attribute)

    @Slot(Attribute)
    def endInteractiveEdit(self, attribute):
        """ Record the interactive changes of an attribute (and of its keys, if animated) as one undoable change. """
        if attribute.fullName not in self._interactiveEdits:
            return
        animation = self._animationOf(attribute)
        if animation is not None and animation.fullName in self._interactiveEdits:
            with self.groupedGraphModification(f"Set Key '{attribute.fullName}'"):
                self._recordInteractiveEdit(attribute)
                self._recordInteractiveEdit(animation)
            return
        self._recordInteractiveEdit(attribute)

    def _recordInteractiveEdit(self, attribute):
        """ Push the change of an attribute since beginInteractiveEdit as one undo step. """
        if attribute.fullName not in self._interactiveEdits:
            return
        previous = self._interactiveEdits.pop(attribute.fullName)
        final = attribute.getSerializedValue()
        if final == previous:
            return
        # Record the change without setting the previous value back first: the attribute keeps its
        # value (a temporary revert would reach the bindings, e.g. a shape being drawn in the viewer)
        command = commands.SetAttributeCommand(self._graph, attribute, final)
        command.oldValue = previous
        self.push(command)

    # Chapters

    def _watchChapters(self):
        """ Follow the Chapter nodes of the current graph, to update the folded sections. """
        self._watchedChapters = set()
        self._graph.nodes.countChanged.connect(self._onNodesCountChanged)
        self._onNodesCountChanged()

    def _onNodesCountChanged(self):
        current = set(self._chapterNodes())
        for node in current - self._watchedChapters:
            node.positionChanged.connect(self._updateFolds)
            node.internalAttribute("folded").valueChanged.connect(self._updateFolds)
        # Removed chapters are deleted with their connections
        self._watchedChapters = current
        self._updateFolds()

    def _chapterNodes(self):
        return [node for node in self._graph.nodes if node.nodeType == "Chapter" and node.hasInternalAttribute("folded")]

    def _updateFolds(self):
        """
        Update the folded sections. The revision changes when a chapter is added, removed, moved or
        (un)folded, and when nodes are added or removed while a chapter is folded (hidden node counts).
        """
        lines = tuple(sorted((node.y, node.name, bool(node.internalAttribute("folded").value))
                             for node in self._chapterNodes()))
        bands = chapters.foldBands((y, folded) for y, _, folded in lines)
        signature = (lines, len(self._graph.nodes) if bands else None)
        if signature == self._foldSignature:
            return
        self._foldSignature = signature
        self._foldBands = bands
        self._foldRevision += 1
        self.foldRevisionChanged.emit()

    @Slot(float, result=float)
    def unfoldY(self, y):
        """ The graph position of an item displayed at `y` in the graph editor (inverse of the folding). """
        return float(chapters.unfoldY(self._foldBands, y))

    @Slot(float, result=float)
    def foldShift(self, y):
        """ How much an item of the graph editor at `y` is moved up by the folded chapters above it. """
        return float(chapters.foldShift(self._foldBands, y))

    @Slot(float, result=bool)
    def isFoldedAway(self, y):
        """ Whether an item of the graph editor at `y` is hidden in a folded chapter. """
        return chapters.isFoldedAway(self._foldBands, y)

    @Slot(QObject, result=bool)
    def canFoldChapter(self, node):
        """ A chapter can be folded when there is another chapter line below it. """
        return any(other is not node and other.y > node.y for other in self._chapterNodes())

    @Slot(QObject, result=int)
    def foldedNodeCount(self, node):
        """ Number of nodes hidden by a folded chapter. """
        band = next((band for band in self._foldBands if band.start == node.y), None)
        if band is None:
            return 0
        return sum(1 for other in self._graph.nodes if other.nodeType != "Chapter" and band.start < other.y < band.end)

    @Slot(QObject)
    def toggleChapterFold(self, node):
        """ Fold or unfold a chapter. Undoable. """
        if node is None or not node.hasInternalAttribute("folded"):
            return
        folded = node.internalAttribute("folded")
        if not folded.value and not self.canFoldChapter(node):
            return
        self.setAttribute(folded, not folded.value)

    @Slot(QObject, QPoint)
    def moveChapter(self, node, position):
        """ Move a chapter line alone (the nodes of its section stay in place). Undoable. """
        self.moveNode(node, Position(position.x(), position.y()))

    @Slot(QPoint, result=QObject)
    def addChapter(self, position):
        """ Add a chapter line at `position`. Undoable. """
        return self.addNewNode("Chapter", position)

    @Slot("QVariantList")
    def endInteractiveEdits(self, attributes):
        """ Record the interactive changes of several attributes (edited together) as one undoable change. """
        with self.groupedGraphModification("Edit"):
            for attribute in attributes:
                self.endInteractiveEdit(attribute)

    @Slot()
    def toggleDisableSelectedNodes(self):
        """
        Disable the selected live compositing nodes, or enable them if they are all disabled (as in Nuke).
        A disabled node passes its first image input through. Undoable.
        """
        nodes = [node for node in self.getSelectedNodes()
                 if node.isImageOpNode and node.hasInternalAttribute("disable")]
        if not nodes:
            return
        disable = any(not node.internalAttribute("disable").value for node in nodes)
        with self.groupedGraphModification("Disable Nodes" if disable else "Enable Nodes"):
            for node in nodes:
                self.setAttribute(node.internalAttribute("disable"), disable)

    @Slot(str)
    def setWorkspaceMode(self, mode: str):
        """
        Switch the workspace of the project (e.g. "compositing", "" for the default one). Undoable. In the compositing
        workspace, the compositing is done in a SubGraph node (see ensureCompositingSubGraph).
        """
        if mode != self.rootGraph.workspaceMode:
            self.push(commands.SetWorkspaceModeCommand(self.rootGraph, mode))
        if mode == "compositing":
            self.ensureCompositingSubGraph()

    @Slot()
    def ensureCompositingSubGraph(self):
        """
        In the compositing workspace, the project graph shows the compositing in a SubGraph node: the first one meant
        for compositing is entered, or a new one ("compositing_1") is created and entered. A project with compositing
        nodes in its own graph (made before) is left as it is, as well as a subgraph already entered.
        """
        from meshroom.core.subgraph import editing
        if len(self._subGraphStack) or self.isComputingLocally():
            return
        graph = self._graph
        existing = editing.compositingSubGraphs(graph)
        if not existing:
            if any(getattr(node, "isImageOpNode", False) for node in graph.nodes):
                return
            bbox = self._layout.boundingBox() if len(graph.nodes) else [0, 0, 0, 0]
            position = Position(bbox[0], bbox[1] + bbox[3] + (100 if len(graph.nodes) else 0))
            name = editing.newCompositingLabel(graph)
            with self.groupedGraphModification("Create Compositing SubGraph"):
                node = self.addNewNode("SubGraph", position=position)
                # Named and labelled "compositing_1"
                if graph.node(name) is None:
                    self.push(commands.RenameNodeCommand(self._graph, node, name))
                self.setAttribute(node.internalAttribute("label"), name)
            existing = [node]
        self.enterSubGraph(existing[0])

    workspaceModeChanged = Signal()
    # The workspace of the project (also when a subgraph is entered)
    workspaceMode = Property(str, lambda self: self.rootGraph.workspaceMode, notify=workspaceModeChanged)

    @Slot(Attribute)
    def resetAttribute(self, attribute):
        """ Reset 'attribute' to its default value. """
        with self.groupedGraphModification(f"Reset Attribute '{attribute.name}'"):
            # if the attribute is a ListAttribute, remove all edges
            if isinstance(attribute, ListAttribute):
                for edge in self._graph.edges:
                    # if the edge is connected to one of the ListAttribute's elements, remove it
                    if edge.src in attribute.value:
                        self.removeEdge(edge)
            self.push(commands.SetAttributeCommand(self._graph, attribute, attribute.getDefaultValue()))

    @Slot(Attribute, str, "QVariant")
    def addAttributeKeyValue(self, attribute, key, value):
        """ Add the given (key, value) pair to the given keyable attribute. """
        self.push(commands.AddAttributeKeyValueCommand(self._graph, attribute, key, value))

    @Slot(Attribute, str)
    def addAttributeKeyDefaultValue(self, attribute, key):
        """ Add the given key with the default value to the given keyable attribute. """
        self.push(commands.AddAttributeKeyValueCommand(self._graph, attribute, key, attribute.getDefaultValue()))

    @Slot(Attribute, str)
    def removeAttributeKey(self, attribute, key):
        """ Remove the given key from the given keyable attribute. """
        self.push(commands.RemoveAttributeKeyCommand(self._graph, attribute, key))

    @Slot(str, str, "QVariant")
    def setObservationFromName(self, shapeFullName, key, observation):
        """ Set the given observation for the given shape attribute name. """
        shape = self.graph.attribute(shapeFullName)
        if shape is None:
            shape = self.graph.internalAttribute(shapeFullName)
        self.push(commands.SetObservationCommand(self._graph, shape, key, observation))

    @Slot(ShapeAttribute, str, "QVariant")
    def setObservation(self, shape, key, observation):
        """ Set the given observation for the given shape attribute. """
        self.push(commands.SetObservationCommand(self._graph, shape, key, observation))

    @Slot(ShapeAttribute, str)
    def removeObservation(self, shape, key):
        """ Remove the given observation for the given shape attribute. """
        self.push(commands.RemoveObservationCommand(self._graph, shape, key))

    @Slot(CompatibilityNode, result=Node)
    def upgradeNode(self, node):
        """ Upgrade a CompatibilityNode. """
        return self.push(commands.UpgradeNodeCommand(self._graph, node))

    @Slot()
    def upgradeAllNodes(self):
        """ Upgrade all upgradable CompatibilityNode instances in the graph. """
        with self.groupedGraphModification("Upgrade all Nodes"):
            nodes = [n for n in self._graph._compatibilityNodes.values() if n.canUpgrade]
            sortedNodes = sorted(nodes, key=lambda x: x.name)
            for node in sortedNodes:
                self.upgradeNode(node)

    @Slot()
    def forceNodesStatusUpdate(self):
        """ Force re-evaluation of graph's nodes status. """
        self._graph.updateStatusFromCache(force=True)

    @Slot(Attribute, QJsonValue)
    def appendAttribute(self, attribute, value=QJsonValue()):
        if isinstance(value, QJsonValue):
            if value.isArray():
                pyValue = value.toArray().toVariantList()
            else:
                pyValue = None if value.isNull() else value.toObject()
        else:
            pyValue = value
        self.push(commands.ListAttributeAppendCommand(self._graph, attribute, pyValue))

    @Slot(Attribute)
    def removeAttribute(self, attribute):
        node = attribute.node
        listAttribute = attribute.root
        if not (node.hasInternalAttribute("hiddenInputs") and node.internalAttribute("hiddenInputs").value
                and isinstance(listAttribute, ListAttribute) and not listAttribute.isOutput):
            self.push(commands.ListAttributeRemoveCommand(self._graph, attribute))
            return
        # Compositing node with hidden edges: the hidden edges of the next elements move down with them
        hidden = node.internalAttribute("hiddenInputs")
        newValue = desc.ImageOpNode.hiddenInputsAfterRemoval(hidden.value, self._relativeName(listAttribute),
                                                             listAttribute.index(attribute))
        with self.groupedGraphModification(f"Remove {attribute.fullName}"):
            self.push(commands.ListAttributeRemoveCommand(self._graph, attribute))
            if newValue != hidden.value:
                self.setAttribute(hidden, newValue)

    @Slot(Attribute)
    def removeAnySetAttribute(self, attribute):
        if not isinstance(attribute.root, AnySet):
            return
        self.push(commands.RemoveAnySetAttributeCommand(self._graph, attribute))

    @Slot(Attribute, str, str, str)
    def addAnySetAttribute(self, anySet, name, label, attributeType):
        """ Add a child to an AnySet attribute (name made unique), e.g. an input of a SubGraph node. """
        if not isinstance(anySet, AnySet) or not name:
            return
        self.push(commands.AddAnySetAttributeCommand(self._graph, anySet, name, label, attributeType))

    @Slot(Attribute, str, str)
    def renameAnySetAttribute(self, attribute, name, label):
        if not isinstance(attribute.root, AnySet):
            return
        self.push(commands.RenameAnySetAttributeCommand(self._graph, attribute, name, label))

    @Slot(Attribute, int)
    def moveAnySetAttribute(self, attribute, offset):
        if not isinstance(attribute.root, AnySet):
            return
        self.push(commands.MoveAnySetAttributeCommand(self._graph, attribute, offset))

    @Slot(Attribute)
    def removeImage(self, image):
        if image is None:
            return
        with self.groupedGraphModification("Remove Image"):
            # Look if the viewpoint's intrinsic is used by another viewpoint
            # If not, remove it
            intrinsicId = image.intrinsicId.value

            intrinsicUsed = False
            for intrinsic in self.cameraInit.attribute("viewpoints").getSerializedValue():
                if image.getSerializedValue() != intrinsic and intrinsic['intrinsicId'] == intrinsicId:
                    intrinsicUsed = True
                    break

            if not intrinsicUsed:
                # Find the intrinsic and remove it
                for intrinsic in self.cameraInit.attribute("intrinsics"):
                    if intrinsic.getSerializedValue()["intrinsicId"] == intrinsicId:
                        self.removeAttribute(intrinsic)
                        break

            # After every check we finally remove the attribute
            self.removeAttribute(image)
        self.imageListChanged.emit()

    @Slot(list)
    def removeImages(self, images: list):
        """ Remove a list of images as a single operation. """
        if not images:
            return
        self.push(commands.RemoveSelectedImagesCommand(self._graph, self.cameraInit, images))
        self.imageListChanged.emit()

    @Slot()
    def removeAllImages(self):
        with self.groupedGraphModification("Remove All Images"):
            self.push(commands.RemoveImagesCommand(self._graph, [self.cameraInit]))
        self.imageListChanged.emit()

    @Slot()
    def removeImagesFromAllGroups(self):
        with self.groupedGraphModification("Remove Images From All CameraInit Nodes"):
            self.push(commands.RemoveImagesCommand(self._graph, list(self.cameraInits)))
        self.imageListChanged.emit()

    @Slot(list)
    @Slot(list, int)
    def selectNodes(self, nodes, command=QItemSelectionModel.SelectionFlag.ClearAndSelect):
        """ Update selection with `nodes` using the specified `command`. """
        indices = [self._graph._nodes.indexOf(node) for node in nodes]
        self.selectNodesByIndices(indices, command)

    @Slot(Node)
    @Slot(Node, int)
    def selectFollowing(self, node: Node, command=QItemSelectionModel.SelectionFlag.ClearAndSelect):
        """ Select all the nodes that depend on `node`. """
        self.selectNodes(
            self._graph.dfsOnDiscover(startNodes=[node], reverse=True, dependenciesOnly=True)[0], command
        )
        self.selectedNode = node

    @Slot(int)
    @Slot(int, int)
    def selectNodeByIndex(self, index: int, command=QItemSelectionModel.SelectionFlag.ClearAndSelect):
        """ Update selection with node at the given `index` using the specified `command`. """
        if isinstance(command, int):
            command = QItemSelectionModel.SelectionFlag(command)

        self.selectNodesByIndices([index], command)

        if self._nodeSelection.isRowSelected(index):
            self.selectedNode = self._graph.nodes.at(index)

    @Slot(list)
    @Slot(list, int)
    def selectNodesByIndices(
        self, indices: list[int], command=QItemSelectionModel.SelectionFlag.ClearAndSelect
    ):
        """
        Update selection with node at given `indices` using the specified `command`.

        Args:
            indices: The list of indices to select.
            command: The selection command to use.
        """
        if isinstance(command, int):
            command = QItemSelectionModel.SelectionFlag(command)

        itemSelection = QItemSelection()
        for index in indices:
            itemSelection.select(
                self._graph.nodes.index(index), self._graph.nodes.index(index)
            )

        self._nodeSelection.select(itemSelection, command)

        if self.selectedNode and not self.isSelected(self.selectedNode):
            self.selectedNode = None

    def iterSelectedNodes(self) -> Iterator[Node]:
        """ Iterate over the currently selected nodes. """
        for idx in self._nodeSelection.selectedRows():
            yield self._graph.nodes.at(idx.row())

    @Slot(result=list)
    def getSelectedNodes(self) -> list[Node]:
        """ Return the list of selected Node instances. """
        return list(self.iterSelectedNodes())

    @Slot(Node, result=bool)
    def isSelected(self, node: Node) -> bool:
        """ Whether `node` is part of the current selection. """
        return self._nodeSelection.isRowSelected(self._graph.nodes.indexOf(node))

    @Slot()
    def clearNodeSelection(self):
        """ Clear all node selection. """
        self.selectedNode = None
        self._nodeSelection.clear()

    def clearNodeHover(self):
        """ Reset currently hovered node to None. """
        self.hoveredNode = None

    @Slot(str)
    def setSelectedNodesColor(self, color: str):
        """
        Sets the Provided color on the selected Nodes.

        Args:
            color (str): Hex code of the color to be set on the nodes.
        """
        # Update the color attribute of the nodes which are currently selected
        with self.groupedGraphModification("Set Nodes Color"):
            # For each of the selected nodes -> Check if the node has a color -> Apply the color if it has
            for node in self.iterSelectedNodes():
                if node.hasInternalAttribute("color"):
                    self.setAttribute(node.internalAttribute("color"), color)

    @Slot(result=str)
    def getSelectedNodesContent(self) -> str:
        """
        Serialize the current node selection and return it as JSON formatted string.

        Returns an empty string if the selection is empty.
        """
        if not self._nodeSelection.hasSelection():
            return ""
        selection = self.getSelectedNodes()
        graphData = self._graph.serializePartial(selection)
        # Clones copied without their cloned node: the partial serialization drops the link, keep it
        # aside so that the pasted Clones point to the same node (see _reconnectPastedClones)
        cloneSources = {}
        for node in selection:
            if node.nodeType == "Clone" and node.hasAttribute("source") and node.attribute("source").isLink:
                link = node.attribute("source").inputLink
                cloneSources[node.name] = [link.node.name, link.name]
        if cloneSources and isinstance(graphData, dict):
            graphData[self._CLONE_SOURCES_KEY] = cloneSources
        return json.dumps(graphData, indent=4)

    @Slot(str, QPoint, result=list)
    def pasteNodes(self, serializedData: str, position: Optional[QPoint]=None) -> list[Node]:
        """
        Import string-serialized graph content `serializedData` in the current graph, optionally at the given
        `position`.
        If the `serializedData` does not contain valid serialized graph data, nothing is done.

        This method can be used with the result of "getSelectedNodesContent".
        But it also accepts any serialized content that matches the graph data or graph content format.

        For example, it is enough to have:
        {"nodeName_1": {"nodeType":"CameraInit"}, "nodeName_2": {"nodeType":"FeatureMatching"}}
        in `serializedData` to create a default CameraInit and a default FeatureMatching nodes.

        Args:
            serializedData: The string-serialized graph data.
            position: The position where to paste the nodes. If None, the nodes are pasted at (0, 0).

        Returns:
            list: the list of Node objects that were pasted and added to the graph
        """
        try:
            graphData = json.loads(serializedData)
        except json.JSONDecodeError:
            logging.warning("Content is not a valid JSON string.")
            return []

        pos = Position(position.x(), position.y()) if position else Position(0, 0)
        with self.groupedGraphModification("Paste Nodes"):
            result = self.push(commands.PasteNodesCommand(self._graph, graphData, pos))
            if result is False:
                logging.warning("Content is not a valid graph data.")
                return []
            self._reconnectPastedClones(graphData, result)
        return result

    # Key of the copied content listing the cloned node of each copied Clone: {clone name: [node, attribute]}
    _CLONE_SOURCES_KEY = "liveCloneSources"

    def _reconnectPastedClones(self, graphData, pastedNodes):
        """ Pasted Clones whose cloned node was not pasted with them point to it again (it is still in the graph). """
        if not isinstance(graphData, dict):
            return
        nodesData = graphData.get(GraphIO.Keys.Graph, graphData)
        sources = graphData.get(self._CLONE_SOURCES_KEY) or {}
        if not isinstance(nodesData, dict) or not isinstance(sources, dict):
            return
        # The pasted nodes are created in the order of the index of their name (Graph._deserialize),
        # renamed when their name is taken: match them with their copied names in that order
        names = sorted((name for name, data in nodesData.items() if isinstance(data, dict) and "nodeType" in data),
                       key=Graph.getNodeIndexFromName)
        if len(names) != len(pastedNodes):
            return
        for name, node in zip(names, pastedNodes):
            if node.nodeType != "Clone" or not node.hasAttribute("source") or node.attribute("source").isLink:
                continue
            source = sources.get(name)
            if isinstance(source, (list, tuple)) and len(source) == 2:
                nodeName, attributeName = source
            else:
                # Link expression "{NodeName.output}" (content written by hand)
                link = nodesData[name].get("inputs", {}).get("source")
                match = re.match(r"^\{(\w+)\.(\w+)\}$", link.strip()) if isinstance(link, str) else None
                if match is None:
                    continue
                nodeName, attributeName = match.group(1), match.group(2)
            sourceNode = self._graph.node(nodeName)
            if sourceNode is not None and sourceNode is not node and sourceNode.hasAttribute(attributeName):
                self.addEdge(sourceNode.attribute(attributeName), node.attribute("source"))

    @Slot(Node, result=bool)
    def canComputeNode(self, node: Node) -> bool:
        """ Check if the node can be computed. """
        if node.isCompatibilityNode or not node.isComputableType or node.getLocked():
            return False
        if node.isComputed:
            return True
        if self._graph.canComputeTopologically(node) and self._graph.canSubmitOrCompute(node) % 2 == 1:
            return True
        return False

    @Slot(Node, result=bool)
    def canSubmitNode(self, node: Node) -> bool:
        """ Check if the node can be submitted. """
        if node.isCompatibilityNode or not node.isComputableType or node.getLocked():
            return False
        if node.isComputed:
            return True
        if self._graph.canComputeTopologically(node) and self._graph.canSubmitOrCompute(node)> 1:
            return True
        return False

    undoStack = Property(QObject, lambda self: self._undoStack, constant=True)
    graphChanged = Signal()
    graph = Property(Graph, lambda self: self._graph, notify=graphChanged)
    taskManager = Property(TaskManager, lambda self: self._taskManager, constant=True)
    nodes = Property(QObject, lambda self: self._graph.nodes, notify=graphChanged)
    layout = Property(GraphLayout, lambda self: self._layout, constant=True)
    live = Property(QObject, lambda self: self._live, constant=True)
    foldRevisionChanged = Signal()
    # Changes when chapters are folded, unfolded or moved (to refresh the positions in the graph editor)
    foldRevision = Property(int, lambda self: self._foldRevision, notify=foldRevisionChanged)
    copiedAttributeChanged = Signal()
    copiedAttribute = Property(QObject, lambda self: self._getCopiedAttribute(), notify=copiedAttributeChanged)

    subGraphChanged = Signal()
    # Names of the entered SubGraph nodes, from the project graph
    # Labels of the entered SubGraph nodes (their names without label), from the outermost one
    subGraphPath = Property("QVariantList", lambda self: [level.node.label for level in self._subGraphStack.levels],
                            notify=subGraphChanged)
    isInSubGraph = Property(bool, lambda self: len(self._subGraphStack) > 0, notify=subGraphChanged)
    liveRunDone = Signal(str, bool)
    liveNodeEvent = Signal(str, str, str)
    liveNodeStatusesChanged = Signal()
    liveStatusRevision = Property(int, lambda self: self._liveStatusRevision, notify=liveNodeStatusesChanged)
    liveStatusChanged = Signal()
    liveRunning = Property(bool, lambda self: self._liveRunning, notify=liveStatusChanged)
    liveStatus = Property(str, lambda self: self._liveStatus, notify=liveStatusChanged)
    liveSuccess = Property(bool, lambda self: self._liveSuccess, notify=liveStatusChanged)

    computeStatusChanged = Signal()
    computing = Property(bool, isComputing, notify=computeStatusChanged)
    computingExternally = Property(bool, isComputingExternally, notify=computeStatusChanged)
    computingLocally = Property(bool, isComputingLocally, notify=computeStatusChanged)
    canSubmit = Property(bool, lambda self: len(submitters), constant=True)

    sortedDFSChunks = Property(QObject, lambda self: self._sortedDFSChunks, constant=True)
    lockedChanged = Signal()

    imageListChanged = Signal()

    selectedNodeChanged = Signal()
    # Current main selected node
    selectedNode = makeProperty(QObject, "_selectedNode", selectedNodeChanged, resetOnDestroy=True)
    # Current chunk selected (used to send signals from TaskManager to ChunksListView)
    selectedChunkChanged = Signal()
    selectedChunk = makeProperty(QObject, "_selectedChunk", selectedChunkChanged, resetOnDestroy=True)

    nodeSelection = makeProperty(QObject, "_nodeSelection")

    hoveredNodeChanged = Signal()
    # Currently hovered node
    hoveredNode = makeProperty(QObject, "_hoveredNode", hoveredNodeChanged, resetOnDestroy=True)

    filePollerRefreshChanged = Signal(int)
    filePollerRefresh = Property(int, lambda self: self._chunksMonitor.filePollerRefresh, notify=filePollerRefreshChanged)
