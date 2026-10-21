"""
Qt bridge of the live compositing engine: "Evaluate" a node, render its branch in a worker
thread, report per-node evaluation statuses and display the result.

Nothing heavy runs in the UI thread: rendering happens in a render worker, and the conversion of
the result to a displayable 8-bit image (at the resolution actually visible) in a display worker.
"""
import logging
import threading
import time
from collections import OrderedDict
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Property, QTimer, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtQuick import QQuickImageProvider

from meshroom.live import (
    BBox, CancelToken, EvaluationCancelled, EvaluationError, Evaluator, ImageBuffer, RGBA, buildPlan, isEvaluable,
    isImageOpNode,
)
from meshroom.live import color
from meshroom.live.cache import systemMemory
from meshroom.live.evaluator import proxyKey
from meshroom.live.display import displayPixels
from meshroom.live.sequences import listFolder


CHANNEL_MODES = ["rgb", "r", "g", "b", "a", "luminance"]
# Reductions of the proxy mode of the viewer (the image is rendered 1 / factor times smaller)
PROXY_FACTORS = (2, 3, 4, 8)


class LiveImageProvider(QQuickImageProvider):
    """
    Serves the displayed image to QML as "image://live/<revision>", and small images (track patterns
    of the zoom window and the keyframe strip) as "image://live/patch/<key>".
    """

    def __init__(self):
        super().__init__(QQuickImageProvider.ImageType.Image)
        self._image = QImage()
        self._patches = OrderedDict()
        self._lock = threading.Lock()

    def setImage(self, image: QImage):
        with self._lock:
            self._image = image

    def setPatch(self, key: str, image: QImage):
        with self._lock:
            self._patches[key] = image
            self._patches.move_to_end(key)
            while len(self._patches) > 256:
                self._patches.popitem(last=False)

    def requestImage(self, imageId, size, requestedSize):
        with self._lock:
            if imageId.startswith("patch/"):
                image = self._patches.get(imageId[len("patch/"):], QImage())
            else:
                image = self._image
        if size is not None:
            size.setWidth(image.width())
            size.setHeight(image.height())
        return image


class _RenderRequest:
    def __init__(self, requestId, plan, frame, token, jumpToRange, signature, region=None):
        self.requestId = requestId
        self.plan = plan
        # Region of interest (pixels of the rendered image), None: the whole image
        self.region = region
        self.frame = frame
        self.token = token
        # Move to the first frame of the image if the requested frame is outside of its range
        self.jumpToRange = jumpToRange
        self.signature = signature


class _RenderResult:
    def __init__(self, requestId, frame, image=None, info=None, error=None, errorNode="", renderTime=0.0,
                 signature=None):
        self.requestId = requestId
        self.signature = signature
        self.frame = frame
        self.image = image
        self.info = info
        self.error = error
        self.errorNode = errorNode
        self.renderTime = renderTime


class _LatestWorker(threading.Thread):
    """ Thread processing the most recent submitted job; older pending jobs are dropped. """

    def __init__(self, name, process):
        super().__init__(name=name, daemon=True)
        self._process = process
        self._pending = None
        self._condition = threading.Condition()

    def submit(self, job):
        with self._condition:
            self._pending = job
            self._condition.notify()

    def run(self):
        while True:
            with self._condition:
                while self._pending is None:
                    self._condition.wait()
                job, self._pending = self._pending, None
            try:
                self._process(job)
            except Exception:
                logging.exception(f"[Live] {self.name} error")


class _DisplayJob:
    def __init__(self, displayId, image, layer, channelMode, exposure, gamma, stride, region, transform=None):
        self.displayId = displayId
        self.transform = transform
        self.image = image
        self.layer = layer
        self.channelMode = channelMode
        self.exposure = exposure
        self.gamma = gamma
        self.stride = stride
        self.region = region


class LiveController(QObject):
    """
    Evaluation of live compositing nodes for the UI.

    - `evaluate(node)` renders the branch upstream of a node and displays the result;
    - with `autoRefresh`, any change of the evaluated branch re-renders the evaluated node
      (changes elsewhere in the graph cost nothing);
    - `nodeStatus(node)` gives the evaluation status of the nodes of the evaluated branch
      at the current frame ("" for the nodes outside of it), to outline them in the graph editor.
    """

    # Emitted from the worker threads, received in the main thread
    _renderFinished = Signal(object)
    _displayReady = Signal(object)
    _statusReported = Signal()
    _displayFailed = Signal(int)
    _trackProgressed = Signal(float)
    _trackFinished = Signal(object)
    # (message, node name): a network asked for is not available (shown as an error, the tracking goes on)
    _trackWarning = Signal(str, str)
    _trackStepped = Signal(object)
    _actionProgressed = Signal(object)
    _actionFinished = Signal(object)
    _actionStepped = Signal(object)
    _matchFinished = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._graph = None
        self._evaluator = Evaluator(listener=self._onStatusFromWorker)
        self._renderWorker = _LatestWorker("LiveRenderWorker", self._render)
        self._renderWorker.start()
        self._displayWorker = _LatestWorker("LiveDisplayWorker", self._convertForDisplay)
        self._displayWorker.start()
        # Playback: the next frames are rendered ahead, while the current one is displayed
        self._prefetchWorker = _LatestWorker("LivePrefetchWorker", self._prefetch)
        self._prefetchWorker.start()
        self._prefetchToken: Optional[CancelToken] = None
        self._imageProvider = LiveImageProvider()

        self._evaluatedNode = None
        self._plan = None
        self._branchUids = set()
        self._token: Optional[CancelToken] = None
        self._requestId = 0
        self._jumpToRange = False
        # What the last render request was for, to skip renders when nothing changed
        self._requestedSignature = None

        self._currentFrame = 1
        self._firstFrame = 1
        self._lastFrame = 1
        self._hasFrameRange = False
        # Frame range of the evaluated image, and of the project (timeline in / out, saved with the graph)
        self._imageRange = (False, 1, 1)
        self._projectRange = None
        # Proxy scale of the viewer renders (1: full resolution, 0.5, 0.25)
        self._proxy = 1.0
        # Region of interest: only the visible part of the image is computed when zoomed in
        self._roiEnabled = True
        self._roi = None
        # Views: 1 = the evaluated node (Evaluate), 2 = the compare node (Evaluate Compare)
        self._mainNode = None
        self._compareNode = None
        self._activeView = 1
        self._autoRefresh = True
        self._rendering = False
        self._errorMessage = ""
        self._errorNode = ""
        self._renderTime = 0.0

        self._image: Optional[ImageBuffer] = None
        self._displayId = 0
        # A display conversion is running (playback waits for it)
        self._displayPending = False
        self._imageRevision = 0
        self._formatWidth = 0
        self._formatHeight = 0
        self._bbox = BBox(0, 0, 0, 0)
        # Visible area of the viewer (Nuke coordinates), and the displayed part of the image
        self._view: Optional[BBox] = None
        self._displayRegion = BBox(0, 0, 0, 0)
        self._shownRegion = BBox(0, 0, 0, 0)
        self._displayedStride = 0
        self._layers = []
        self._currentLayer = RGBA
        self._channelMode = "rgb"
        self._exposure = 0.0
        self._gamma = 1.0
        # Color management: the colorspaces of the files and the display transform of the viewer
        self._colorConfig = color.config()
        self._displayTransform = self._colorConfig.defaultDisplayTransform()
        self._displayStride = 1
        self._statusRevision = 0
        self._systemMemory = systemMemory()
        # Color attribute picked in the viewer with the eyedropper, or None
        self._pickAttribute = None
        # Frame range text of the Read nodes, by uid ("" when unknown)
        self._frameRanges = {}

        self._renderFinished.connect(self._onRenderFinished)
        self._displayReady.connect(self._onDisplayReady)
        self._displayFailed.connect(self._onDisplayFailed)
        self._trackProgressed.connect(self._setTrackingProgress)
        self._trackFinished.connect(self._onTrackFinished)
        self._trackWarning.connect(self._setError)
        self._trackStepped.connect(self._onTrackStepped)
        self._actionProgressed.connect(self._onActionProgressed)
        self._actionFinished.connect(self._onActionFinished)
        self._actionStepped.connect(self._onActionStepped)
        # Attributes edited live by the running action ({full name: attribute}, see ActionContext.publish)
        self._actionEdits = {}
        self._matchFinished.connect(self._onMatchFinished)
        self._statusReported.connect(self._scheduleStatusUpdate)

        # Tracking (Tracker node): progress in [0, 1], -1 when not tracking
        self._trackingProgress = -1.0
        self._trackToken: Optional[CancelToken] = None
        # Live display of the tracking: keys waiting to be written, applied at most every 40 ms
        self._trackNode = None
        self._trackPending = []
        self._trackApplyTimer = QTimer(self)
        self._trackApplyTimer.setSingleShot(True)
        self._trackApplyTimer.setInterval(40)
        self._trackApplyTimer.timeout.connect(self._applyTrackedKeys)
        # Node action running in a worker (e.g. the autotrack of the CameraTracker): progress in [0, 1], -1 if none
        self._actionToken: Optional[CancelToken] = None
        self._actionProgress = -1.0
        self._actionText = ""
        self._actionNode = None
        self._actionName = ""
        # A color match is running (ColorLookup nodes)
        self._matching = False
        # Changes with the graph: results of an older tracking / color match are dropped
        self._operationGeneration = 0

        # Playback: a frame is shown when the previous one has been rendered
        self._playDirection = 0
        self._playStep = 1
        self._fps = 24.0
        self._loop = True
        self._measuredFps = 0.0
        self._lastPlayedTime = None
        self._playTimer = QTimer(self)
        self._playTimer.timeout.connect(self._playTick)

        # Coalesce the graph updates and the status reports
        self._refreshTimer = QTimer(self)
        self._refreshTimer.setSingleShot(True)
        self._refreshTimer.setInterval(20)
        self._refreshTimer.timeout.connect(self._requestRender)
        self._statusTimer = QTimer(self)
        self._statusTimer.setSingleShot(True)
        self._statusTimer.setInterval(30)
        self._statusTimer.timeout.connect(self._bumpStatusRevision)

    @property
    def imageProvider(self) -> LiveImageProvider:
        return self._imageProvider

    # Graph

    def setGraph(self, graph):
        """ Follow a new graph: the previous evaluation and its cached images are dropped. """
        if self._graph is not None:
            try:
                self._graph.updated.disconnect(self._onGraphUpdated)
            except (RuntimeError, TypeError):
                pass
        self._graph = graph
        self._setPickAttribute(None)
        # Results of tracking / color matching still running belong to the previous graph
        self.stopTracking()
        self.stopAction()
        self._operationGeneration += 1
        self._resetOperations()
        self._projectRange = None
        self.clear()
        self._mainNode = None
        if self._compareNode is not None:
            self._compareNode = None
            self.compareNodeChanged.emit()
        self._setActiveView(1)
        self._evaluator.cache.clear()
        self._evaluator.clearStatuses()
        if graph is not None:
            graph.updated.connect(self._onGraphUpdated)

    def showGraph(self, graph):
        """
        Follow another graph of the same project (entering / leaving a subgraph): the cache is kept, the
        evaluation of a node of the previous graph stops.
        """
        if graph is self._graph:
            return
        if self._graph is not None:
            try:
                self._graph.updated.disconnect(self._onGraphUpdated)
            except (RuntimeError, TypeError):
                pass
        self._graph = graph
        self._setPickAttribute(None)
        self.stopTracking()
        self.stopAction()
        self._operationGeneration += 1
        self._resetOperations()
        self.clear()
        self._mainNode = None
        if self._compareNode is not None:
            self._compareNode = None
            self.compareNodeChanged.emit()
        self._setActiveView(1)
        if graph is not None:
            graph.updated.connect(self._onGraphUpdated)

    def _onGraphUpdated(self):
        self._checkPickAttribute()
        node = self._evaluatedNode
        if node is None:
            return
        try:
            removed = self._graph is None or self._graph.node(node.name) is not node
        except RuntimeError:  # The underlying Qt object has been deleted
            removed = True
        if self._compareNode is not None and not self._isInGraph(self._compareNode):
            self._compareNode = None
            self.compareNodeChanged.emit()
        if self._mainNode is not None and not self._isInGraph(self._mainNode):
            self._mainNode = None
        if removed:
            # The evaluated node has been removed from the graph
            self.clear()
            return
        if self._autoRefresh:
            self._refreshTimer.start()
        else:
            # Frozen: the rendered branch is kept; the modified nodes, whose uid changed, lose their outline
            self._bumpStatusRevision()

    def _checkPickAttribute(self):
        """ Stop picking a color for an attribute whose node has been removed. """
        attribute = self._pickAttribute
        if attribute is None:
            return
        try:
            node = attribute.node
            removed = self._graph is None or self._graph.node(node.name) is not node
        except RuntimeError:  # The underlying Qt object has been deleted
            removed = True
        if removed:
            self._setPickAttribute(None)

    # Part of the image rendered when the viewer is zoomed in (region of interest)
    ROI_MAX_FRACTION = 0.6

    def _wantedRegion(self):
        """
        The region of interest to render (pixels of the rendered image): the visible area extended by half
        its size on each side when it is a small part of the image, else None (the whole image).
        """
        if not self._roiEnabled or self._view is None or self._image is None:
            return None
        width, height = self._image.format
        view = self._view
        if view.isEmpty or view.width * view.height > self.ROI_MAX_FRACTION * width * height:
            return None
        return view.expanded(view.width // 2 + 1, view.height // 2 + 1)

    def _setRoiEnabled(self, value):
        if bool(value) != self._roiEnabled:
            self._roiEnabled = bool(value)
            self.roiEnabledChanged.emit()
            if self._evaluatedNode is not None:
                self._requestRender(force=True)

    def _key(self, node):
        """ The key of a node in the evaluator (its uid, at the proxy scale of the viewer). """
        plan = self._plan
        if getattr(node, "nodeType", "") == "SubGraph" and plan is not None \
                and plan.targetNode.name.startswith(f"{node.name}/"):
            # The evaluated SubGraph node: the compositing node giving its image
            return plan.target
        uid = getattr(node, "_uid", None)
        if uid is None:
            return None
        # A node of the current branch: its key in the plan (it can carry the extra key of a Switch expression)
        plan = self._plan
        if plan is not None and plan.proxy == self._proxy:
            key = plan.keyOf(node.name)
            if key is not None and key.startswith(proxyKey(uid, self._proxy)):
                return key
        return proxyKey(uid, self._proxy)

    def _setProxy(self, value):
        value = float(value)
        # 1 (full resolution) or 1 / n for the reductions of the viewer menu
        if value <= 0 or value > 1 or int(round(1.0 / value)) not in (1,) + PROXY_FACTORS or value == self._proxy:
            return
        value = 1.0 / int(round(1.0 / value))
        self._proxy = value
        self.proxyChanged.emit()
        if self._evaluatedNode is not None:
            self._requestRender(force=True)

    def _updateBranch(self):
        try:
            self._plan = buildPlan(self._evaluatedNode, proxy=self._proxy)
            self._branchUids = set(self._plan.nodes.keys())
        except Exception as e:
            logging.warning(f"[Live] Cannot snapshot the branch of '{self._evaluatedNode.name}': {e}")
            self._plan = None
            self._branchUids = set()

    # Evaluation

    @Slot(QObject, result=bool)
    def canEvaluate(self, node) -> bool:
        """ Whether a node gives an image to evaluate (not a Camera, a ReadGeo...). """
        return node is not None and isEvaluable(node)

    @Slot(QObject)
    def evaluate(self, node):
        """ Display the result of `node` (a compositing node, or a SubGraph of them), computing its branch (view 1). """
        if node is None or not isEvaluable(node):
            return
        # Re-read the file information (frame ranges) on explicit evaluations
        self._evaluator.clearInfos()
        self._frameRanges.clear()
        self._jumpToRange = True
        self._mainNode = node
        self._setActiveView(1)
        self._setEvaluatedNode(node)
        self._requestRender(force=True)

    @Slot(QObject)
    def evaluateCompare(self, node):
        """
        Make `node` the compare node (view 2) and display it: views 1 (Evaluate) and 2 (Evaluate Compare)
        are then switched with showView (keys 1 / 2), for before / after comparisons.
        """
        if node is None or not isEvaluable(node):
            return
        if node is not self._compareNode:
            self._compareNode = node
            self.compareNodeChanged.emit()
        if self._mainNode is None:
            self._mainNode = self._evaluatedNode
        self.showView(2)

    @Slot(int)
    def showView(self, view):
        """ Display view 1 (the evaluated node) or 2 (the compare node), keeping the frame. """
        node = self._mainNode if view == 1 else self._compareNode
        if node is None or not self._isInGraph(node):
            return
        self._setActiveView(view)
        if node is not self._evaluatedNode:
            self._setEvaluatedNode(node)
            self._requestRender(force=True)

    def _setActiveView(self, view):
        if view != self._activeView:
            self._activeView = view
            self.activeViewChanged.emit()

    def _isInGraph(self, node):
        try:
            return self._graph is not None and self._graph.node(node.name) is node
        except RuntimeError:  # The underlying Qt object has been deleted
            return False

    @Slot()
    def refresh(self):
        """ Re-evaluate the current node from scratch: files are read again from disk. """
        self._frameRanges.clear()
        if self._evaluatedNode is not None:
            self._evaluator.clearInfos()
            self._evaluator.cache.discardUids(self._branchUids)
            self._evaluator.clearStatuses()
            self._requestRender(force=True)

    @Slot()
    def clear(self):
        """ Stop evaluating: no node is displayed nor outlined anymore. """
        self.stop()
        if self._token:
            self._token.cancel()
        # Ignore a render or a display conversion that would finish after this point
        self._requestId += 1
        self._displayId += 1
        self._displayPending = False
        self._requestedSignature = None
        self._setEvaluatedNode(None)
        self._plan = None
        self._branchUids = set()
        self._image = None
        self._imageProvider.setImage(QImage())
        self._layers = []
        self._formatWidth = self._formatHeight = 0
        self._bbox = BBox(0, 0, 0, 0)
        self._displayRegion = BBox(0, 0, 0, 0)
        self._shownRegion = BBox(0, 0, 0, 0)
        self._displayedStride = 0
        self._setRendering(False)
        self._setError("", "")
        self._imageRange = (False, 1, 1)
        self._applyFrameRange()
        self.imageChanged.emit()
        self.layersChanged.emit()
        self._bumpStatusRevision()

    def _setEvaluatedNode(self, node):
        if node is self._evaluatedNode:
            return
        self._evaluatedNode = node
        self.evaluatedNodeChanged.emit()

    def _requestRender(self, force=False):
        if self._evaluatedNode is None:
            return
        # A frozen viewer keeps its branch (e.g. to look at other frames of the same state)
        if self._autoRefresh or force or self._plan is None:
            self._updateBranch()
        if self._plan is None:
            return
        # A change outside of the evaluated branch (a node created, moved, selected...) changes nothing
        signature = self._signature()
        if not force and signature == self._requestedSignature:
            self._bumpStatusRevision()
            return
        if self._token:
            self._token.cancel()
        if self._requestedSignature is None or signature[:2] != self._requestedSignature[:2]:
            # Another branch or other values: the frames rendered ahead are no longer the right ones
            self._cancelPrefetch()
        self._requestedSignature = signature
        self._token = CancelToken()
        self._requestId += 1
        self._setRendering(True)
        self._bumpStatusRevision()
        self._roi = self._wantedRegion()
        self._renderWorker.submit(_RenderRequest(self._requestId, self._plan, self._currentFrame, self._token,
                                                 self._jumpToRange, signature, self._roi))

    def _signature(self):
        """ What a render of the current plan depends on: the node uids, their known errors and the frame. """
        if self._plan is None:
            return None
        nodes = frozenset((uid, planNode.error) for uid, planNode in self._plan.nodes.items())
        return self._plan.target, nodes, self._currentFrame

    def _render(self, request: _RenderRequest):
        """ Render a request (render worker thread). """
        start = time.perf_counter()
        frame = request.frame
        info = None
        try:
            try:
                info = self._evaluator.info(request.plan)
            except Exception:
                # Rendering reports the error on the failing node with its status
                info = None
            if request.jumpToRange and info is not None and info.hasFrameRange \
                    and not info.firstFrame <= frame <= info.lastFrame:
                frame = info.firstFrame
            image = self._evaluator.render(request.plan, frame, request.token, request.region)
            result = _RenderResult(request.requestId, frame, image=image, info=info,
                                   renderTime=time.perf_counter() - start)
        except EvaluationCancelled:
            return
        except EvaluationError as e:
            result = _RenderResult(request.requestId, frame, info=info, error=e.message, errorNode=e.nodeName)
        except Exception as e:
            logging.exception("[Live] Unexpected evaluation error")
            result = _RenderResult(request.requestId, frame, info=info, error=str(e))
        result.signature = request.signature
        self._renderFinished.emit(result)

    @Slot(object)
    def _onRenderFinished(self, result: _RenderResult):
        if result.requestId != self._requestId:
            return  # An outdated request
        self._setRendering(False)
        self._jumpToRange = False
        if result.info is not None:
            self._setFrameRange(result.info)
        if result.frame != self._currentFrame:
            # Moved into the frame range of the image: what was rendered is the request at that frame
            self._currentFrame = result.frame
            if result.signature is not None:
                target, nodes, _ = result.signature
                self._requestedSignature = (target, nodes, result.frame)
            self.currentFrameChanged.emit()
        if result.error is not None:
            # Keep the previous image on screen, as in Nuke
            self._setError(result.error, result.errorNode)
        else:
            self._setError("", "")
            self._renderTime = result.renderTime
            if result.image is not self._image:
                self._image = result.image
                self._updateDisplay()
        self._bumpStatusRevision()

    def _setFrameRange(self, info):
        self._imageRange = (info.hasFrameRange, info.firstFrame or 1, info.lastFrame or 1)
        self._applyFrameRange()

    def setProjectRange(self, first, last):
        """ The frame range of the timeline (in / out of the project), or None, None: the range of the image. """
        self._projectRange = (int(first), int(last)) if first is not None and last is not None else None
        self._applyFrameRange()

    def _applyFrameRange(self):
        """ The timeline plays the project range when there is one, else the range of the evaluated image. """
        values = (True,) + self._projectRange if self._projectRange is not None else self._imageRange
        if values == (self._hasFrameRange, self._firstFrame, self._lastFrame):
            return
        self._hasFrameRange, self._firstFrame, self._lastFrame = values
        self.frameRangeChanged.emit()

    def _setRendering(self, value):
        if value != self._rendering:
            self._rendering = value
            self.renderingChanged.emit()

    def _setError(self, message, nodeName):
        if (message, nodeName) != (self._errorMessage, self._errorNode):
            self._errorMessage, self._errorNode = message, nodeName
            self.errorChanged.emit()

    # Tracking

    @Slot(QObject, int, "QVariantList")
    def track(self, node, direction, names):
        """
        Track the tracks `names` (all the enabled ones if empty) of a Tracker node (or the pins of a Resection) from
        the current frame, forward (1) or backward (-1), to the end of the frame range of its input, in a worker
        thread. All the tracks move together frame after frame (each frame rendered once, see tracking.trackMany);
        the pins of a Resection are linked by its pose (ThreeDNodes._PinsModel). The viewer follows the tracking
        (each tracked frame is shown with its keys); all the new keys are one undoable change.
        """
        from meshroom.live import tracking
        # Not while an action runs (e.g. the geometry tracking of a Resection: both would edit the node)
        if self._trackToken is not None or self._actionToken is not None or node is None or not node.hasAttribute("tracks"):
            return
        input = node.attribute("input")
        if not input.isLink or not isImageOpNode(input.inputRootLink.node):
            self._setError("The Tracker input is not connected.", node.name)
            return
        try:
            plan = buildPlan(input.inputRootLink.node)
            info = self._evaluator.info(plan)
            tracks = tracking.parseTracks(node.attribute("tracks").value)
        except Exception as e:
            self._setError(f"Cannot track: {e}", node.name)
            return
        selected = [(index, t) for index, t in enumerate(tracks)
                    if t.get("enabled", True) and (not names or t.get("name") in names)]
        if not selected:
            return
        step = 1 if direction > 0 else -1
        start = self._currentFrame
        if info.hasFrameRange:
            end = info.lastFrame if step > 0 else info.firstFrame
        else:
            end = start + 100 * step
        frames = list(range(start, end + step, step))
        if len(frames) < 2:
            return
        token = CancelToken()
        self._trackToken = token
        generation = self._operationGeneration
        self._setTrackingProgress(0.0)
        self._trackNode = node
        self._trackPending = []
        self._beginInteractiveEdit(node.attribute("tracks"))

        def cancel():
            if token.cancelled:
                raise EvaluationCancelled()

        # Tracks of a node linking them (the pins of a Resection, on one rigid geometry): tracked with its model
        modelContext = self._trackingModelContext(node, token)
        trackingModel = getattr(node.nodeDesc, "trackingModel", None)
        # A point tracking network (Tracker, Resection: "Point Tracker"), and the GPU ("Use GPU")
        method = node.attribute("pointTracker").value if node.hasAttribute("pointTracker") else "patches"
        nodeName = node.name
        useGpu = node.attribute("useGpu").value if node.hasAttribute("useGpu") else True

        def run():
            results = {}
            error = ""
            lastFrame = start
            try:
                # All the tracks together, frame after frame (each frame rendered once): tracked in the pixels of the
                # input image, keys written back in the format of each track (a track made on a proxy keeps its format)
                scaled, formats, stopAt = [], [], []
                for trackIndex, track in selected:
                    trackFormat = track.get("format") or [info.width, info.height]
                    formats.append((trackFormat, trackFormat[0] / float(info.width), trackFormat[1] / float(info.height)))
                    scaled.append(tracking.scaleTracks([track], info.width, info.height)[0])
                    # Stopped at the next keyframe (placed by hand) in the tracking direction, as in Nuke
                    keyframes = [f for f in track.get("keyframes") or [] if (f - start) * step > 0]
                    stopAt.append(min(keyframes, key=lambda f: abs(f - start)) if keyframes else None)
                model = None
                if modelContext is not None:
                    try:
                        model = trackingModel(modelContext, scaled)
                    except EvaluationCancelled:
                        raise
                    except Exception:
                        # The tracks are still tracked, each on its own
                        logging.warning("[Live] Tracking without the model of the node", exc_info=True)

                def progress(done, count):
                    self._trackProgressed.emit(done / float(count))

                def onKey(position, frame, x, y, score):
                    trackIndex, track = selected[position]
                    trackFormat, backX, backY = formats[position]
                    self._trackStepped.emit((generation, trackIndex, track.get("name"), frame, round(x * backX, 3),
                                             round(y * backY, 3), score, list(trackFormat)))
                from meshroom.live import gpu
                device = gpu.device(useGpu)
                pointTracker = None
                if method in ("tapir", "cotracker"):
                    from meshroom.live.neural import NeuralUnavailable, warnUnavailable
                    try:
                        from meshroom.live.neural.pointTracking import pointTracker as network
                        pointTracker = network(method, device)
                    except NeuralUnavailable as e:
                        warnUnavailable(method, e)
                        self._trackWarning.emit(f"{method} is not available, tracked with patches: {e}", nodeName)

                def imageAt(f):
                    return self._evaluator.render(plan, f, token)
                # Stopped or failing frames: the keys found so far are returned
                allKeys = None
                if pointTracker is not None:
                    try:
                        allKeys = tracking.trackManyNeural(scaled, frames, imageAt, pointTracker, cancel=cancel,
                                                           progress=progress, onKey=onKey, stopAt=stopAt)
                    except EvaluationCancelled:
                        raise
                    except Exception as e:
                        # E.g. out of memory on the GPU: tracked again with the patterns
                        gpu.failed(method, e)
                        logging.warning(f"[Live] {method} failed, tracked with patches", exc_info=True)
                        self._trackWarning.emit(f"{method} failed, tracked with patches: {e}", nodeName)
                if allKeys is None:
                    allKeys = tracking.trackMany(scaled, frames, imageAt, cancel=cancel, progress=progress, onKey=onKey,
                                                 stopAt=stopAt, model=model, device=device)
                for (trackIndex, track), keys in zip(selected, allKeys):
                    # By index: two tracks may have the same name
                    results[trackIndex] = (track.get("name"), keys)
                    if keys:
                        last = int(max(keys, key=lambda f: abs(int(f) - start)))
                        if abs(last - start) > abs(lastFrame - start):
                            lastFrame = last
            except Exception as e:
                logging.exception("[Live] Tracking error")
                error = str(e)
            self._trackFinished.emit((node, generation, results, lastFrame, error))

        threading.Thread(target=run, name="LiveTracking", daemon=True).start()

    def _trackingModelContext(self, node, token):
        """
        The render context of a node whose tracks are linked (its descriptor has `trackingModel`, e.g. a Resection), at
        the current frame, in a full resolution plan (the tracks are tracked at full resolution); None otherwise.
        """
        from meshroom.live.evaluator import RenderContext
        if not hasattr(node.nodeDesc, "trackingModel"):
            return None
        try:
            plan = buildPlan(node)
        except Exception:
            logging.debug("[Live] Tracking model", exc_info=True)
            return None
        context = RenderContext(plan.nodes[plan.target], self._currentFrame, self._evaluator, plan)
        # "Stop" also stops what the model computes (e.g. the pose of the start frame)
        context._token = token
        return context

    @Slot(QObject, "QVariantList", str)
    def deleteKeys(self, node, names, mode):
        """
        Remove keys of the tracks `names` (all if empty) of a Tracker or a Resection (and the poses of its geometry
        track), keeping the current frame (see tracking.deleteKeys): "forward", "backward", or "tracked" (the keys
        found by the tracking, the keyframes placed by hand kept). One undoable change.
        """
        import json
        from meshroom.live import geotrack, tracking
        if node is None or self._trackToken is not None or self._actionToken is not None:
            return
        changes = []
        try:
            if node.hasAttribute("tracks"):
                tracks, removed = tracking.deleteKeys(tracking.parseTracks(node.attribute("tracks").value), list(names or []),
                                                      self._currentFrame, mode)
                if removed:
                    changes.append((node.attribute("tracks"), json.dumps(tracks)))
            if node.hasAttribute("geoTrack") and node.attribute("geoTrack").value:
                text, removed = geotrack.deletePoses(node.attribute("geoTrack").value, self._currentFrame, mode)
                if removed:
                    changes.append((node.attribute("geoTrack"), text))
        except (ValueError, RuntimeError) as e:
            self._setError(f"Delete keys: {e}", node.name)
            return
        if not changes:
            return
        uiGraph = self.parent()
        label = {"forward": "Delete Keys Forward", "backward": "Delete Keys Backward"}.get(mode, "Delete Tracked Keys")
        if uiGraph is not None and hasattr(uiGraph, "groupedGraphModification"):
            with uiGraph.groupedGraphModification(label):
                for attribute, value in changes:
                    uiGraph.setAttribute(attribute, value)
        else:
            for attribute, value in changes:
                attribute.value = value

    @Slot()
    def stopTracking(self):
        """ Stop the tracking: the frames already tracked are kept. """
        if self._trackToken is not None:
            self._trackToken.cancel()

    def _resetOperations(self):
        """ Forget the running tracking / color match / action (their results will be dropped). """
        self._actionToken = None
        self._actionNode, self._actionName = None, ""
        self._actionProgress = -1.0
        self.actionProgressChanged.emit()
        self._trackToken = None
        self._matching = False
        self._setTrackingProgress(-1.0)

    def _isCurrentNode(self, node, generation) -> bool:
        """ Whether a node of an operation started at `generation` is still in the current graph. """
        if generation != self._operationGeneration or self._graph is None:
            return False
        try:
            return self._graph.node(node.name) is node
        except RuntimeError:  # The underlying Qt object has been deleted
            return False

    @Slot(object)
    def _onTrackFinished(self, payload):
        node, generation, results, lastFrame, error = payload
        if generation == self._operationGeneration:
            self._trackToken = None
            self._setTrackingProgress(-1.0)
        if not self._isCurrentNode(node, generation):
            # The node or the graph is gone: nothing more is written, the interactive edit is dropped
            self._trackApplyTimer.stop()
            self._trackPending = []
            if self._trackNode is node:
                self._trackNode = None
                uiGraph = self.parent()
                if uiGraph is not None and hasattr(uiGraph, "cancelInteractiveEdit"):
                    try:
                        uiGraph.cancelInteractiveEdit(node.attribute("tracks"))
                    except RuntimeError:
                        pass
            return
        if error:
            self._setError(f"Tracking: {error}", node.name)
        # The keys not written yet (the live display writes them as they come), then one undo step
        self._trackApplyTimer.stop()
        self._applyTrackedKeys()
        self._endInteractiveEdit(node.attribute("tracks"))
        self._trackNode = None
        self._setCurrentFrame(lastFrame)

    @Slot(object)
    def _onTrackStepped(self, payload):
        generation = payload[0]
        if generation != self._operationGeneration or self._trackNode is None:
            return
        self._trackPending.append(payload[1:])
        if not self._trackApplyTimer.isActive():
            self._trackApplyTimer.start()

    def _applyTrackedKeys(self):
        """ Write the keys tracked since the last call into the tracks of the tracked node, and show the last frame. """
        import json
        from meshroom.live import tracking
        node, pending = self._trackNode, self._trackPending
        self._trackPending = []
        if node is None or not pending or not self._isCurrentNode(node, self._operationGeneration):
            return
        try:
            tracks = tracking.parseTracks(node.attribute("tracks").value)
        except (ValueError, RuntimeError):
            return
        for index, name, frame, x, y, score, trackFormat in pending:
            # The track may have been removed or renamed meanwhile
            if index < len(tracks) and tracks[index].get("name") == name:
                tracks[index].setdefault("format", trackFormat)
                tracks[index].setdefault("keys", {})[str(frame)] = [x, y]
                tracks[index].setdefault("scores", {})[str(frame)] = round(score, 4)
        self._setInteractively(node.attribute("tracks"), json.dumps(tracks))
        self._setCurrentFrame(pending[-1][2])

    def _beginInteractiveEdit(self, attribute):
        uiGraph = self.parent()
        if uiGraph is not None and hasattr(uiGraph, "beginInteractiveEdit"):
            uiGraph.beginInteractiveEdit(attribute)

    def _setInteractively(self, attribute, value):
        uiGraph = self.parent()
        if uiGraph is not None and hasattr(uiGraph, "setValueInteractively"):
            uiGraph.setValueInteractively(attribute, value)
        else:
            attribute.value = value

    def _endInteractiveEdit(self, attribute):
        uiGraph = self.parent()
        if uiGraph is not None and hasattr(uiGraph, "endInteractiveEdit"):
            uiGraph.endInteractiveEdit(attribute)

    def _setTrackingProgress(self, value):
        if value != self._trackingProgress:
            self._trackingProgress = value
            self.trackingProgressChanged.emit()

    # Node actions

    @Slot(QObject, str)
    def runAction(self, node, name):
        """
        Run an action of a compositing node (a "liveAction" button, see meshroom/live/actions.py) in a
        worker thread, with progress; its results are set as one undoable change when it ends.
        """
        from meshroom.live.actions import ActionContext
        from meshroom.live.evaluator import _snapshotValue
        if self._actionToken is not None or node is None or not hasattr(node.nodeDesc, "runAction"):
            return
        plan = None
        if node.hasAttribute("input"):
            attr = node.attribute("input")
            if attr.isLink and isImageOpNode(attr.inputRootLink.node):
                try:
                    plan = buildPlan(attr.inputRootLink.node)
                except Exception as e:
                    self._setError(f"{name}: {e}", node.name)
                    return
        # The other connected image inputs (e.g. a mask)
        inputPlans = {}
        for attr in node.attributes:
            if attr.type == "Image" and not attr.isOutput and attr.name != "input" and attr.isLink \
                    and isImageOpNode(attr.inputRootLink.node):
                try:
                    inputPlans[attr.name] = buildPlan(attr.inputRootLink.node)
                except Exception:
                    pass
        values = {attr.name: _snapshotValue(attr) for attr in node.attributes
                  if not attr.isOutput and attr.type not in ("Image", "PushButtonParam")}
        # The node itself (its camera / geo inputs and its own outputs, see ActionContext.nodeContext)
        nodePlan = None
        if isImageOpNode(node):
            try:
                nodePlan = buildPlan(node)
            except Exception:
                logging.debug(f"[Live] Action '{name}': no plan of the node", exc_info=True)
        # Parameter values of the nodes connected to the image inputs (through Dot, Clone, disabled nodes)
        inputValues = {}
        for attr in node.attributes:
            if attr.type == "Image" and not attr.isOutput and attr.isLink:
                source = self.upstreamValuesNode(attr)
                if source is not None:
                    inputValues[attr.name] = {a.name: _snapshotValue(a) for a in source.attributes
                                              if not a.isOutput and a.type not in ("Image", "PushButtonParam")}
        token = CancelToken()
        self._actionToken = token
        self._actionNode, self._actionName = node, name
        generation = self._operationGeneration
        label = node.attribute(name).label if node.hasAttribute(name) else name
        self._onActionProgressed((0.0, label))

        def run():
            result, error = None, ""
            try:
                context = ActionContext(values, self._evaluator, plan, self._currentFrame, token, inputPlans=inputPlans,
                                        inputValues=inputValues, nodePlan=nodePlan,
                                        progress=lambda f, text="": self._actionProgressed.emit((f, text or label)),
                                        publish=lambda stepValues, frame: self._actionStepped.emit(
                                            (node, generation, stepValues, frame)))
                result = node.nodeDesc.runAction(name, context)
            except EvaluationCancelled:
                error = ""
            except EvaluationError as e:
                error = e.message
            except RuntimeError as e:
                # Expected failures (e.g. not enough tracks to solve): reported to the user
                error = str(e)
            except Exception as e:
                logging.exception(f"[Live] Action '{name}' error")
                error = str(e) or e.__class__.__name__
            self._actionFinished.emit((node, generation, label, result, error))

        threading.Thread(target=run, name="LiveAction", daemon=True).start()

    @Slot()
    def stopAction(self):
        if self._actionToken is not None:
            self._actionToken.cancel()

    @Slot(object)
    def _onActionProgressed(self, payload):
        fraction, text = payload
        if self._actionToken is None:
            return
        self._actionProgress, self._actionText = float(fraction), text
        self.actionProgressChanged.emit()

    @Slot(object)
    def _onActionStepped(self, payload):
        """ Intermediate values of a running action: set live (one undo step when it ends), viewer at `frame`. """
        node, generation, values, frame = payload
        if self._actionToken is None or not self._isCurrentNode(node, generation):
            return
        for attrName, value in values.items():
            attribute = self._actionAttribute(node, attrName)
            if attribute.fullName not in self._actionEdits:
                self._actionEdits[attribute.fullName] = attribute
                self._beginInteractiveEdit(attribute)
            self._setInteractively(attribute, value)
        if frame is not None:
            self._setCurrentFrame(int(frame))

    def _endActionEdits(self, node, generation, result):
        """
        End the live edits of an action: the final values (taken out of `result`) replace the published
        ones, then each edited attribute is recorded as one undo step.
        """
        edits, self._actionEdits = self._actionEdits, {}
        if not edits:
            return
        if not self._isCurrentNode(node, generation):
            uiGraph = self.parent()
            for attribute in edits.values():
                if uiGraph is not None and hasattr(uiGraph, "cancelInteractiveEdit"):
                    try:
                        uiGraph.cancelInteractiveEdit(attribute)
                    except RuntimeError:
                        pass
            return
        for attrName in list(result or {}):
            attribute = self._actionAttribute(node, attrName) if not attrName.startswith("_") else None
            if attribute is not None and attribute.fullName in edits:
                self._setInteractively(attribute, result.pop(attrName))
        for attribute in edits.values():
            self._endInteractiveEdit(attribute)

    @Slot(object)
    def _onActionFinished(self, payload):
        node, generation, label, result, error = payload
        if generation == self._operationGeneration:
            self._actionToken = None
            self._actionNode, self._actionName = None, ""
            self._actionProgress, self._actionText = -1.0, ""
            self.actionProgressChanged.emit()
        result = dict(result) if result else {}
        # What was published live is kept (also when the action is stopped or fails)
        self._endActionEdits(node, generation, result)
        if not self._isCurrentNode(node, generation):
            return
        if error:
            self._setError(f"{label}: {error}", node.name)
            return
        if not result:
            return
        disconnect = result.pop("_disconnect", [])
        uiGraph = self.parent()
        if uiGraph is not None and hasattr(uiGraph, "groupedGraphModification"):
            with uiGraph.groupedGraphModification(label):
                for attrName, value in result.items():
                    uiGraph.setAttribute(self._actionAttribute(node, attrName), value)
                for inputName in disconnect:
                    edge = self._graph.edge(node.attribute(inputName)) if node.hasAttribute(inputName) else None
                    if edge is not None:
                        uiGraph.removeEdge(edge)
        else:
            for attrName, value in result.items():
                self._actionAttribute(node, attrName).value = value

    @staticmethod
    def _actionAttribute(node, name):
        """ The attribute an action result sets: a parameter, or an internal attribute (e.g. "animation"). """
        if not node.hasAttribute(name) and node.hasInternalAttribute(name):
            return node.internalAttribute(name)
        return node.attribute(name)

    @Slot(QObject, result=str)
    def jsonSummary(self, attribute):
        """ One line describing the JSON value of an attribute: autotrack tracks or camera solve. """
        import json
        try:
            data = json.loads(attribute.value or "null")
        except (TypeError, ValueError):
            return "invalid"
        semantic = attribute.desc.semantic
        if semantic == "autoTracks":
            if not data:
                return "no track (Autotrack)"
            frames = sorted({int(f) for track in data for f in (track.get("keys") or {})})
            return f"{len(data)} tracks, frames {frames[0]}-{frames[-1]}" if frames else f"{len(data)} tracks"
        if semantic == "cameraSolve":
            if not data:
                return "not solved (Solve Camera)"
            interpolated = len(data.get("interpolated", []))
            return (f"{len(data.get('frames', []))} frames solved"
                    + (f" ({interpolated} interpolated)" if interpolated else "")
                    + f", {len(data.get('points', {}))} points, error {data.get('error', 0):.2f} px")
        return ""

    # Curves

    @Slot("QVariantList", float, float, int, result="QVariantList")
    def sampleCurve(self, points, x0, x1, count):
        """ Values of a curve (see meshroom.live.curves) at `count` regular x from x0 to x1, for display. """
        from meshroom.live import curves
        xs = np.linspace(x0, x1, max(2, int(count)))
        return [float(v) for v in curves.evaluate(points, xs)]

    @Slot(QObject)
    def matchColors(self, node):
        """
        Set the curves of a ColorLookup / ColorLookupHSV node so that the colors of its input match
        those of its "matchTarget" input, at the current frame (rendered in a worker thread).
        The curves are written as one undoable change.
        """
        if node is None or not node.hasAttribute("matchTarget") or self._matching:
            return
        sources = []
        for name in ("input", "matchTarget"):
            attr = node.attribute(name)
            if not attr.isLink or not isImageOpNode(attr.inputRootLink.node):
                self._setError(f"Color match: connect the '{attr.label}' input.", node.name)
                return
            sources.append(attr.inputRootLink.node)
        try:
            plans = [buildPlan(source) for source in sources]
        except Exception as e:
            self._setError(f"Color match: {e}", node.name)
            return
        frame = self._currentFrame
        hsv = node.nodeType == "ColorLookupHSV"
        self._matching = True
        generation = self._operationGeneration

        def run():
            from meshroom.live import curves
            result, error = None, ""
            try:
                source, target = (self._evaluator.render(plan, frame).rgba() for plan in plans)
                result = curves.matchCurvesHSV(source, target) if hsv else curves.matchCurvesRGB(source, target)
            except EvaluationError as e:
                error = e.message
            except Exception as e:
                logging.exception("[Live] Color match error")
                error = str(e)
            self._matchFinished.emit((node, generation, result, error))

        threading.Thread(target=run, name="LiveColorMatch", daemon=True).start()

    @Slot(object)
    def _onMatchFinished(self, payload):
        import json
        from meshroom.live import curves
        node, generation, result, error = payload
        if generation == self._operationGeneration:
            self._matching = False
        if not self._isCurrentNode(node, generation):
            return
        if error:
            self._setError(f"Color match: {error}", node.name)
        if result is None:
            return
        try:
            values = curves.parseCurves(node.attribute("curves").value)
        except ValueError:
            values = {}
        values.update(result)
        self._setAttribute(node.attribute("curves"), json.dumps(values))

    def _setAttribute(self, attribute, value):
        """ Set an attribute as an undoable change (through the UIGraph owning this controller). """
        uiGraph = self.parent()
        if uiGraph is not None and hasattr(uiGraph, "setAttribute"):
            uiGraph.setAttribute(attribute, value)
        else:
            attribute.value = value

    # Playback

    @Slot(int)
    def play(self, direction):
        """
        Play forward (1) or backward (-1) by steps of `playStep` frames, at `fps` at most: each frame is
        shown once rendered (playback slows down rather than skipping frames). Playing in the current
        direction again stops.
        """
        if direction == 0 or direction == self._playDirection or self._evaluatedNode is None:
            self.stop()
            return
        self._playDirection = 1 if direction > 0 else -1
        self._lastPlayedTime = None
        self._measuredFps = 0.0
        self._playTimer.start(max(1, int(round(1000.0 / max(self._fps, 0.1)))))
        self.playingChanged.emit()

    @Slot()
    def stop(self):
        if self._playDirection == 0:
            return
        self._playDirection = 0
        self._playTimer.stop()
        self._cancelPrefetch()
        self.playingChanged.emit()

    # Number of frames rendered ahead during playback
    PREFETCH_FRAMES = 12

    def _cancelPrefetch(self):
        if self._prefetchToken is not None:
            self._prefetchToken.cancel()
            self._prefetchToken = None

    def _schedulePrefetch(self):
        """ Render the next frames of the playback in the prefetch worker (they are then read from the cache). """
        if self._plan is None or not self._hasFrameRange or self._playDirection == 0:
            return
        frames = []
        frame = self._currentFrame
        for index in range(self.PREFETCH_FRAMES + 1):
            frame += self._playDirection * self._playStep
            if frame > self._lastFrame or frame < self._firstFrame:
                if not self._loop:
                    break
                frame = self._firstFrame if self._playDirection > 0 else self._lastFrame
            # The next frame is rendered by the render worker at the next tick: not twice
            if index > 0:
                frames.append(frame)
        if not frames:
            return
        if self._prefetchToken is None:
            self._prefetchToken = CancelToken()
        self._prefetchWorker.submit((self._plan, frames, self._prefetchToken, self._roi))

    def _prefetch(self, job):
        """ Prefetch worker thread: render frames ahead into the cache, stopping at the first error. """
        plan, frames, token, region = job
        for frame in frames:
            if token.cancelled:
                return
            if self._evaluator._cachedFor((plan.target, frame), region) is not None:
                continue
            try:
                self._evaluator.render(plan, frame, token, region)
            except (EvaluationCancelled, EvaluationError):
                return
            except Exception:
                logging.exception("[Live] Prefetch error")
                return

    @Slot(int)
    def stepFrames(self, steps):
        """ Move by `steps` times the play step (e.g. -1: previous step), within the frame range. """
        self.stop()
        frame = self._currentFrame + steps * self._playStep
        if self._hasFrameRange:
            frame = min(max(frame, self._firstFrame), self._lastFrame)
        self._setCurrentFrame(frame)

    @Slot()
    def goToFirstFrame(self):
        self.stop()
        self._setCurrentFrame(self._firstFrame)

    @Slot()
    def goToLastFrame(self):
        self.stop()
        self._setCurrentFrame(self._lastFrame)

    def _playTick(self):
        if self._evaluatedNode is None or not self._hasFrameRange:
            self.stop()
            return
        if self._rendering or self._displayPending:
            return  # Wait for the current frame to be on screen
        frame = self._currentFrame + self._playDirection * self._playStep
        if frame > self._lastFrame or frame < self._firstFrame:
            if not self._loop:
                self.stop()
                return
            frame = self._firstFrame if self._playDirection > 0 else self._lastFrame
        now = time.perf_counter()
        if self._lastPlayedTime is not None:
            instant = 1.0 / max(now - self._lastPlayedTime, 1e-3)
            self._measuredFps = instant if self._measuredFps == 0 else 0.8 * self._measuredFps + 0.2 * instant
            self.measuredFpsChanged.emit()
        self._lastPlayedTime = now
        self._setCurrentFrame(frame)
        self._schedulePrefetch()

    def _setPlayStep(self, value):
        value = max(1, int(value))
        if value != self._playStep:
            self._playStep = value
            self.playSettingsChanged.emit()

    def _setFps(self, value):
        value = max(0.1, float(value))
        if value != self._fps:
            self._fps = value
            if self._playTimer.isActive():
                self._playTimer.setInterval(max(1, int(round(1000.0 / value))))
            self.playSettingsChanged.emit()

    def _setLoop(self, value):
        if bool(value) != self._loop:
            self._loop = bool(value)
            self.playSettingsChanged.emit()

    # Display

    def _updateDisplay(self):
        """ Convert the current image for display, in the display worker. """
        image = self._image
        if image is None:
            return
        layers = image.layerNames
        if layers != self._layers:
            self._layers = layers
            self.layersChanged.emit()
        if self._currentLayer not in layers:
            self._currentLayer = RGBA if RGBA in layers else layers[0]
            self.currentLayerChanged.emit()
        self._displayId += 1
        self._displayPending = True
        self._displayWorker.submit(_DisplayJob(self._displayId, image, self._currentLayer, self._channelMode,
                                               self._exposure, self._gamma, self._displayStride,
                                               self._regionToDisplay(image), self._currentDisplayTransform()))

    def _regionToDisplay(self, image) -> BBox:
        """
        The part of the image to convert: the visible area extended by half of its size on each side
        (so that small pans do not need a new conversion), aligned on the stride.
        """
        if self._view is None:
            return image.bbox
        view = self._view
        marginX, marginY = view.width // 2 + 1, view.height // 2 + 1
        region = view.expanded(marginX, marginY).intersection(image.bbox)
        if region.isEmpty:
            return region
        # Align on the stride grid of the bbox, so that the displayed pixels do not shift while panning
        stride = self._displayStride
        bbox = image.bbox
        x0 = bbox.x0 + ((region.x0 - bbox.x0) // stride) * stride
        y1 = bbox.y1 - ((bbox.y1 - region.y1) // stride) * stride
        return BBox(x0, region.y0, region.x1, y1)

    @Slot(float, float, float, float, int)
    def setDisplayView(self, left, top, right, bottom, stride):
        """
        Set the visible area of the viewer, in canvas coordinates (pixels from the top-left corner of the
        format, y down), and the display stride. Only the visible part of the image is converted for display.
        """
        import math
        height = self._formatHeight or (self._image.height / self._proxy if self._image is not None else 0)
        # Full resolution canvas -> pixels of the rendered image (proxy)
        p = self._proxy
        view = BBox(int(math.floor(left * p)), int(math.floor((height - bottom) * p)), int(math.ceil(right * p)),
                    int(math.ceil((height - top) * p)))
        stride = max(1, int(round(int(stride) * p)))
        self._view = view
        strideChanged = stride != self._displayStride
        if strideChanged:
            self._displayStride = stride
            self.displayStrideChanged.emit()
        image = self._image
        if image is None:
            return
        # Rendered for a region of interest: render again when the view leaves it, or covers much more
        if self._roi is not None:
            inside = view.intersection(self._roi) == view
            if not inside or self._wantedRegion() is None:
                self._requestRender(force=True)
        visible = view.intersection(image.bbox)
        shown = self._shownRegion
        covered = visible.isEmpty or visible.intersection(shown) == visible
        # After zooming in, the displayed part is much larger than needed: convert a smaller one
        needed = self._regionToDisplay(image)
        displayedArea = shown.width * shown.height
        oversized = displayedArea > 2 * max(1, needed.width * needed.height)
        if strideChanged or stride != self._displayedStride or not covered or oversized:
            self._updateDisplay()

    def _convertForDisplay(self, job: _DisplayJob):
        """ Display worker thread: convert an image to an 8-bit QImage. """
        try:
            pixels = displayPixels(job.image, job.layer, job.channelMode, job.exposure, job.gamma, job.stride,
                                   job.region, job.transform)
        except Exception:
            logging.exception("[Live] Display conversion error")
            self._displayFailed.emit(job.displayId)
            return
        height, width = pixels.shape[:2]
        qimage = QImage(pixels.data, width, height, width * 4, QImage.Format.Format_RGBA8888).copy() \
            if width and height else QImage()
        self._displayReady.emit((job.displayId, job.image, qimage, job.region.intersection(job.image.bbox), job.stride))

    @Slot(object)
    def _onDisplayReady(self, payload):
        displayId, image, qimage, region, stride = payload
        if displayId != self._displayId:
            return  # Outdated
        self._displayPending = False
        self._imageProvider.setImage(qimage)
        # In the coordinates of the full resolution (a proxy image is shown enlarged)
        scale = 1.0 / self._proxy

        def full(box):
            return BBox(int(round(box.x0 * scale)), int(round(box.y0 * scale)), int(round(box.x1 * scale)),
                        int(round(box.y1 * scale)))
        self._formatWidth, self._formatHeight = int(round(image.width * scale)), int(round(image.height * scale))
        self._bbox = full(image.bbox)
        self._displayRegion = full(region)
        # The converted region in the pixels of the rendered image
        self._shownRegion = region
        self._displayedStride = stride
        self._imageRevision += 1
        self.imageChanged.emit()

    @Slot(int)
    def _onDisplayFailed(self, displayId):
        if displayId == self._displayId:
            self._displayPending = False

    @Slot(int, int, result="QVariantList")
    def pixelValues(self, x, y):
        """ Return the float values of the displayed layer at a pixel in Nuke coordinates (y up), or []. """
        image = self._image
        if image is None:
            return []
        # Viewer coordinates are those of the full resolution
        x, y = int(np.floor(x * self._proxy)), int(np.floor(y * self._proxy))
        bbox = image.bbox
        if not (bbox.x0 <= x < bbox.x1 and bbox.y0 <= y < bbox.y1):
            return []
        layer = self._currentLayer if image.hasLayer(self._currentLayer) else image.layerNames[0]
        return [float(v) for v in image.layer(layer)[bbox.y1 - 1 - y, x - bbox.x0]]

    @Slot(QObject, float, float, float, float, result="QVariantList")
    def sampleColor(self, node, x0, y0, x1, y1):
        """
        Return the average RGBA color [r, g, b, a] of a rectangle (Nuke coordinates, y up, pixel edges)
        or [] if there is no image there.

        The color is sampled in the image entering `node` ("input" image input) at the current frame
        when it is in the RAM cache, as a keyer needs the color of the screen before keying; otherwise in
        the displayed image.
        """
        image = self._inputImage(node) or self._image
        if image is None:
            return []
        p = self._proxy
        x0, y0, x1, y1 = x0 * p, y0 * p, x1 * p, y1 * p
        bbox = image.bbox
        left, right = int(np.floor(min(x0, x1))), int(np.ceil(max(x0, x1)))
        bottom, top = int(np.floor(min(y0, y1))), int(np.ceil(max(y0, y1)))
        right, top = max(right, left + 1), max(top, bottom + 1)
        region = BBox(left, bottom, right, top).intersection(bbox)
        if region.isEmpty:
            return []
        pixels = image.rgba()[bbox.y1 - region.y1:bbox.y1 - region.y0, region.x0 - bbox.x0:region.x1 - bbox.x0]
        return [float(v) for v in pixels.reshape(-1, 4).mean(axis=0)]

    def _inputImage(self, node):
        """ The cached image entering the "input" of `node` at the current frame, or None. """
        if node is None or not isImageOpNode(node):
            return None
        return self._inputImageAt(node, self._currentFrame)

    @Slot(result="QVariantList")
    def currentChannels(self):
        image = self._image
        if image is None or not image.hasLayer(self._currentLayer):
            return []
        return list(image.channels(self._currentLayer))

    # Statuses

    def _onStatusFromWorker(self, uid, frame, status, message):
        # Called from the worker thread
        self._statusReported.emit()

    @Slot()
    def _scheduleStatusUpdate(self):
        if not self._statusTimer.isActive():
            self._statusTimer.start()

    def _bumpStatusRevision(self):
        self._statusRevision += 1
        self.statusRevisionChanged.emit()

    @Slot(QObject, result=str)
    def nodeStatus(self, node):
        """
        Return the evaluation status of a node at the current frame: "RUNNING", "SUCCESS", "ERROR",
        "UPSTREAM_ERROR", or "" if it is not part of the evaluated branch or not evaluated yet.
        """
        if node is None or self._evaluatedNode is None:
            return ""
        if getattr(node, "nodeType", "") == "SubGraph":
            return self._subGraphStatus(node)[0]
        if self._key(node) not in self._branchUids:
            return ""
        status, _ = self._evaluator.status(self._key(node), self._currentFrame)
        if status.name == "NONE":
            # Rendered at another frame only (upstream of a time node): its last status
            status, _ = self._evaluator.lastStatus(self._key(node))
        return "" if status.name == "NONE" else status.name

    @Slot(QObject, str, int, int, result=str)
    def trackPatch(self, node, trackName, frame, size):
        """
        Image of the area around a track of a Tracker at a frame, for the zoom window and the keyframe
        strip of the viewer: "image://live/patch/..." (taken in the image entering the tracker, from
        the RAM cache), or "" if that frame is not in the cache.
        """
        import json
        from meshroom.live import tracking
        if node is None or not node.hasAttribute("tracks"):
            return ""
        image = self._inputImageAt(node, frame)
        if image is None:
            return ""
        try:
            track = tracking.findTrack(json.loads(node.attribute("tracks").value or "[]"), trackName)
        except ValueError:
            return ""
        if track:
            # In the pixels of the image entering the tracker (tracked on another resolution)
            track = tracking.scaleTracks([track], *image.format)[0]
        position = tracking.trackPosition(track, frame) if track else None
        if position is None:
            return ""
        size = max(8, int(size))
        bbox = image.bbox
        half = size // 2
        cx, cy = int(np.floor(position[0])), int(np.floor(position[1]))
        region = BBox(cx - half, cy - half, cx - half + size, cy - half + size)
        visible = region.intersection(bbox)
        pixels = displayPixels(image, RGBA, self._channelMode, self._exposure, self._gamma, 1, visible,
                               self._currentDisplayTransform()) if not visible.isEmpty else None
        canvas = np.zeros((size, size, 4), dtype=np.uint8)
        canvas[:, :, 3] = 255
        if pixels is not None:
            visible = region.intersection(bbox)
            top, left = region.y1 - visible.y1, visible.x0 - region.x0
            canvas[top:top + pixels.shape[0], left:left + pixels.shape[1]] = pixels
        key = f"{node.name}/{trackName}/{frame}/{size}/{self._statusRevision}"
        qimage = QImage(canvas.data, size, size, size * 4, QImage.Format.Format_RGBA8888).copy()
        self._imageProvider.setPatch(key, qimage)
        return "image://live/patch/" + key

    # Resection (meshroom/nodes/compositing/ThreeDNodes.py): pins put on the geometry in the viewer

    def _resectionContext(self, node):
        """
        (render context of a Resection node at the current frame, proxy scale): in the plan of the viewer when the node is
        in it (its solved camera is cached there), else in a full resolution plan. (None, 1) if not a Resection.
        """
        from meshroom.live.evaluator import RenderContext
        if node is None or not isImageOpNode(node) or not hasattr(node.nodeDesc, "pickPoint"):
            return None, 1.0
        plan, key = self._plan, self._key(node)
        if plan is None or key not in plan.nodes:
            plan = buildPlan(node)
            key = plan.target
        return RenderContext(plan.nodes[key], self._currentFrame, self._evaluator, plan), plan.proxy

    @Slot(QObject, float, float, result="QVariantList")
    def pickGeometryPoint(self, node, x, y):
        """
        The point [x, y, z] of the geometry of a Resection under the image position (x, y) (Nuke pixels of the
        plate), seen from its solved camera at the current frame; [] if no geometry there.
        """
        try:
            ctx, proxy = self._resectionContext(node)
            # The viewer shows the image at full resolution: (x, y) in the pixels of the rendered (proxy) image
            point = node.nodeDesc.pickPoint(ctx, float(x) * proxy, float(y) * proxy) if ctx is not None else None
        except Exception as e:
            logging.debug("[Live] Add pin", exc_info=True)
            self._setError(f"Add pin: {getattr(e, 'message', e) or e.__class__.__name__}", node.name)
            return []
        return [round(float(v), 6) for v in point] if point is not None else []

    @Slot(QObject, result=str)
    def resectionSummary(self, node):
        """
        Pins used at the current frame and their reprojection error, for the toolbar of the viewer: from the camera
        solved by the render of the viewer (nothing is solved here, in the thread of the interface).
        """
        from meshroom.live.evaluator import RenderContext
        plan, key = self._plan, self._key(node)
        if node is None or plan is None or key not in plan.nodes:
            return ""
        try:
            camera = self._evaluator.cachedData(key, "cameraOutput", self._currentFrame)
            ctx = RenderContext(plan.nodes[key], self._currentFrame, self._evaluator, plan)
            info = ctx.inputInfo("input")
            points, positions = node.nodeDesc._pinsAt(node.nodeDesc.pins(ctx, info), self._currentFrame)
            if not len(points):
                return "no pin at this frame"
            count = f"{len(points)} pin{'s' if len(points) > 1 else ''}"
            if camera is None or info is None:
                return count
            xy, _, _ = camera.project(points, info.width, info.height)
            # In the pixels of the full resolution image
            error = float(np.sqrt(np.mean(np.sum((xy - positions) ** 2, axis=1)))) / plan.proxy
            return f"{count}, error {error:.2f} px"
        except Exception as e:
            return str(getattr(e, "message", e))

    def _inputImageAt(self, node, frame):
        """ The cached image entering the "input" of `node` at a frame, or None. """
        if node is None or not isImageOpNode(node) or not node.hasAttribute("input"):
            return None
        attr = node.attribute("input")
        if not attr.isLink or not isImageOpNode(attr.inputRootLink.node):
            return None
        return self._evaluator.cache.get((self._key(attr.inputRootLink.node), frame))

    @Slot(QObject, result="QVariantList")
    def nodeKeyFrames(self, node):
        """
        The frames holding keys in a node, shown on the timeline: tracked positions of a Tracker,
        keys of the Roto shapes, points of a TimeWarp curve, keys of the animated parameters.
        """
        import json
        frames = set()
        if node is None or not isImageOpNode(node):
            return []
        for attr in node.attributes:
            semantic = getattr(attr.desc, "semantic", "")
            if semantic not in ("tracks", "rotoShapes", "timeCurve", "geoTrack") or attr.isOutput:
                continue
            try:
                data = json.loads(attr.value or "null")
            except (TypeError, ValueError):
                continue
            try:
                if semantic == "geoTrack":
                    # The poses tracked with "Track Geometry" of a Resection
                    frames.update(int(f) for f in ((data or {}).get("poses") or {}))
                elif semantic == "tracks":
                    for track in data or []:
                        frames.update(int(f) for f in (track.get("keys") or {}))
                elif semantic == "rotoShapes":
                    for shape in data or []:
                        frames.update(int(key["frame"]) for key in shape.get("keys") or [])
                else:
                    for points in (data or {}).values():
                        frames.update(int(round(point[0])) for point in points)
            except (TypeError, ValueError, KeyError, AttributeError):
                continue
        # Keys of the animated parameters
        from meshroom.live import animation as anim
        if node.hasInternalAttribute(anim.ANIMATION):
            for keys in anim.parseAnimation(node.internalAttribute(anim.ANIMATION).value).values():
                frames.update(int(round(frame)) for frame, _ in keys)
        return sorted(frames)

    @Slot(QObject, str, result="QVariantList")
    def inputLayers(self, node, inputNames):
        """
        The layers of the image entering `node` through the first connected of its image inputs
        `inputNames` (comma separated, e.g. "depth,input"): e.g. the layers of an EXR offered by Shuffle,
        from its last render; ["rgba"] if it has not been rendered yet.
        Bind to statusRevision to update the list after the renders.
        """
        try:
            if node is None:
                return ["rgba"]
            for inputName in [name.strip() for name in inputNames.split(",") if name.strip()]:
                if not node.hasAttribute(inputName) or not node.attribute(inputName).isLink:
                    continue
                names = self._evaluator.layerNames(self._key(node.attribute(inputName).inputRootLink.node))
                return names or ["rgba"]
        except RuntimeError:  # The underlying Qt object has been deleted
            pass
        return ["rgba"]

    @Slot(QObject, result=QObject)
    def upstreamValuesNode(self, attribute):
        """
        The node whose values an image input reads (ctx.inputValues): the connected node, looking
        through the disabled nodes and the pass-through ones (Dot, Clone). None if not connected.
        """
        if attribute is None or not attribute.isLink:
            return None
        node = attribute.inputRootLink.node
        seen = set()
        while node is not None and isImageOpNode(node) and node.name not in seen:
            seen.add(node.name)
            disabled = node.hasInternalAttribute("disable") and bool(node.internalAttribute("disable").value)
            if not disabled and not getattr(node.nodeDesc, "passThroughValues", False):
                return node
            first = next((attr for attr in node.attributes if not attr.isOutput and attr.type == "Image"
                          and not attr.desc.infoOnly), None)
            node = first.inputRootLink.node if first is not None and first.isLink else None
        return None

    @Slot(QObject, result=str)
    def nodeFrameRange(self, node):
        """ The frame range of a Read node, "in first  out last" (or "" for a still image or an unreadable file). """
        if node is None or getattr(node, "nodeType", "") != "Read" or not isImageOpNode(node):
            return ""
        uid = node._uid
        if uid in self._frameRanges:
            return self._frameRanges[uid]
        text = ""
        try:
            info = self._evaluator.info(buildPlan(node))
            if info.hasFrameRange:
                text = f"Frame range   in {info.firstFrame}   out {info.lastFrame}"
        except Exception:
            pass
        if len(self._frameRanges) > 1000:
            self._frameRanges.clear()
        self._frameRanges[uid] = text
        return text

    def nodeInfo(self, node):
        """ The ImageInfo (format, frame range) of the output of a compositing node, or None if unknown. """
        if node is None or not isImageOpNode(node):
            return None
        try:
            return self._evaluator.info(buildPlan(node))
        except Exception:
            return None

    def nodeFormat(self, node):
        """ The format (width, height) of the output of a compositing node, or None if unknown. """
        if node is None or not isImageOpNode(node):
            return None
        try:
            info = self._evaluator.info(buildPlan(node))
        except Exception:
            return None
        return (info.width, info.height)

    def rotoTracks(self, node) -> dict:
        """
        The tracks of the Trackers connected to the "tracker" inputs of a Roto ({input name: tracks}), in the pixels of
        the Roto, as when it renders (through Dot, Clone and disabled nodes; tracks made on another resolution scaled).
        """
        from meshroom.live import tracking
        tracks = {}
        for name in ("tracker", "tracker2", "tracker3"):
            source = self.upstreamValuesNode(node.attribute(name)) if node.hasAttribute(name) else None
            try:
                tracks[name] = tracking.parseTracks(source.attribute("tracks").value) \
                    if source is not None and source.hasAttribute("tracks") else []
            except ValueError:
                tracks[name] = []
        rotoFormat = self.nodeFormat(node)
        if rotoFormat is not None:
            tracks = {name: tracking.scaleTracks(values, *rotoFormat) for name, values in tracks.items()}
        return tracks

    @Slot(QObject, int, result="QVariantMap")
    def rotoLinkTransforms(self, node, frame):
        """
        The link transforms of a Roto at `frame`, for the viewer to show the shapes where they are and to store the
        edited points back: {"global": the transform of the Roto links, "shapes": [{"shape": the transform of the whole
        shape, "points": {point index: its own transform}}]}, each a 3x3 matrix as 9 numbers (rows); identity without
        links. A shape without entry (just drawn) takes the "global" one.
        """
        import numpy as np
        from meshroom.live import roto
        identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        empty = {"global": identity, "shapes": []}
        if node is None or not node.hasAttribute("shapes"):
            return empty
        try:
            shapes = roto.parseShapes(node.attribute("shapes").value)
            globalLinks = roto.parseShapes(node.attribute("links").value) if node.hasAttribute("links") else []
        except ValueError:
            return empty
        if not globalLinks and not any(roto.shapeLinks(shape) for shape in shapes):
            return {"global": identity, "shapes": [{"shape": identity, "points": {}} for _ in shapes]}
        tracks = self.rotoTracks(node)

        def flat(matrix):
            return [float(v) for v in np.asarray(matrix).ravel()]
        result = []
        for shape in shapes:
            whole, own = roto.linkTransforms(shape, frame, tracks, globalLinks)
            result.append({"shape": flat(whole), "points": {str(index): flat(matrix) for index, matrix in own.items()}})
        return {"global": flat(roto.linkTransforms({}, frame, tracks, globalLinks)[0]), "shapes": result}

    @Slot(QObject, result=str)
    def _subGraphStatus(self, node) -> tuple:
        """
        (status, message) of a SubGraph node in the evaluated branch: from the nodes of its subgraph (named
        "<SubGraph>/<node>" in the plan), the worst one.
        """
        plan = self._plan
        if plan is None:
            return "", ""
        prefix = f"{node.name}/"
        order = {"ERROR": 0, "UPSTREAM_ERROR": 1, "RUNNING": 2, "SUCCESS": 3}
        worst = ("", "")
        for key, planNode in plan.nodes.items():
            if not planNode.name.startswith(prefix):
                continue
            status, message = self._evaluator.status(key, self._currentFrame)
            if status.name == "NONE":
                status, message = self._evaluator.lastStatus(key)
            name = status.name
            if name in order and (worst[0] == "" or order[name] < order[worst[0]]):
                worst = (name, f"{planNode.name}: {message}" if message else "")
        return worst

    def nodeMessage(self, node):
        """ Return the evaluation error message of a node at the current frame, if any. """
        if node is not None and getattr(node, "nodeType", "") == "SubGraph":
            return self._subGraphStatus(node)[1]
        if node is None or self._key(node) not in self._branchUids:
            return ""
        return self._evaluator.status(self._key(node), self._currentFrame)[1]

    @Slot(QObject, result=float)
    def nodeRenderTime(self, node):
        """ Time in milliseconds the node itself took to render the current frame (0: pass-through), -1 if unknown. """
        if node is None or self._key(node) not in self._branchUids:
            return -1.0
        seconds = self._evaluator.renderTime(self._key(node), self._currentFrame)
        return -1.0 if seconds is None else seconds * 1000.0

    @Slot(result="QVariantList")
    def cachedFrames(self):
        """ Return the frames of the evaluated node currently in the RAM cache. """
        if self._evaluatedNode is None:
            return []
        return self._evaluator.cache.cachedFrames(self._key(self._evaluatedNode))

    @Slot()
    def clearCache(self):
        """ Free the memory of every cached image (the displayed image is kept on screen). """
        self._evaluator.cache.clear()
        self._evaluator.clearStatuses()
        self._requestedSignature = None
        self._bumpStatusRevision()

    # File browsing

    @Slot(str, bool, result="QVariantList")
    def listFolder(self, folder, imagesOnly):
        """ List a folder with the frames of image sequences grouped (see meshroom.live.sequences). """
        return listFolder(folder, imagesOnly)

    @Slot(str, result=str)
    def browseStartFolder(self, currentPath):
        """ Folder to open a file browser in: the folder of `currentPath` if it exists, the home folder otherwise. """
        import os
        folder = os.path.dirname(currentPath) if currentPath else ""
        while folder and not os.path.isdir(folder):
            parent = os.path.dirname(folder)
            folder = parent if parent != folder else ""
        return folder or os.path.expanduser("~")

    @Slot(str, result=str)
    def parentFolder(self, folder):
        import os
        return os.path.dirname(os.path.normpath(folder)) or folder

    # Properties

    def _setCurrentFrame(self, frame):
        frame = int(frame)
        if frame == self._currentFrame:
            return
        self._currentFrame = frame
        self.currentFrameChanged.emit()
        self._bumpStatusRevision()
        self._requestRender()

    def _setAutoRefresh(self, value):
        if value == self._autoRefresh:
            return
        self._autoRefresh = value
        self.autoRefreshChanged.emit()
        if value:
            self._requestRender()

    def _setCurrentLayer(self, layer):
        if layer == self._currentLayer:
            return
        self._currentLayer = layer
        self.currentLayerChanged.emit()
        self._updateDisplay()

    def _setChannelMode(self, mode):
        if mode == self._channelMode or mode not in CHANNEL_MODES:
            return
        self._channelMode = mode
        self.channelModeChanged.emit()
        self._updateDisplay()

    def _setExposure(self, value):
        if value == self._exposure:
            return
        self._exposure = float(value)
        self.exposureChanged.emit()
        self._updateDisplay()

    def _setGamma(self, value):
        if value == self._gamma:
            return
        self._gamma = float(value)
        self.gammaChanged.emit()
        self._updateDisplay()

    def _currentDisplayTransform(self):
        try:
            return self._colorConfig.displayTransform(self._displayTransform)
        except Exception:
            logging.exception(f"[Live] Display transform '{self._displayTransform}' error, sRGB used")
            return None

    def _setDisplayTransform(self, name):
        if name == self._displayTransform:
            return
        self._displayTransform = name
        self.displayTransformChanged.emit()
        self._updateDisplay()

    @Slot()
    def reloadColorConfig(self):
        """ Load the color config again (the "OCIO" environment variable): colorspaces and display transforms. """
        self._colorConfig = color.loadConfig()
        if self._displayTransform not in self._colorConfig.displayTransforms():
            self._displayTransform = self._colorConfig.defaultDisplayTransform()
            self.displayTransformChanged.emit()
        self.colorConfigChanged.emit()
        # The images read with the previous colorspaces are rendered again
        self.clearCache()
        self.refresh()
        self._updateDisplay()

    def _setDisplayStride(self, value):
        value = max(1, int(value))
        if value == self._displayStride:
            return
        self._displayStride = value
        self.displayStrideChanged.emit()
        self._updateDisplay()

    def _setPickAttribute(self, attribute):
        if attribute is self._pickAttribute:
            return
        self._pickAttribute = attribute
        self.pickAttributeChanged.emit()

    evaluatedNodeChanged = Signal()
    evaluatedNode = Property(QObject, lambda self: self._evaluatedNode, notify=evaluatedNodeChanged)
    roiEnabledChanged = Signal()
    # Region of interest: only the visible part of the image is computed when the viewer is zoomed in
    roiEnabled = Property(bool, lambda self: self._roiEnabled, _setRoiEnabled, notify=roiEnabledChanged)
    proxyChanged = Signal()
    # Proxy scale of the viewer renders: 1 (full resolution), 0.5 or 0.25 (faster)
    proxy = Property(float, lambda self: self._proxy, _setProxy, notify=proxyChanged)
    proxyFactors = Property("QVariantList", lambda self: list(PROXY_FACTORS), constant=True)
    compareNodeChanged = Signal()
    # Node of view 2 (Evaluate Compare), None if none
    compareNode = Property(QObject, lambda self: self._compareNode, notify=compareNodeChanged)
    activeViewChanged = Signal()
    # View displayed: 1 (Evaluate) or 2 (Evaluate Compare)
    activeView = Property(int, lambda self: self._activeView, notify=activeViewChanged)
    currentFrameChanged = Signal()
    currentFrame = Property(int, lambda self: self._currentFrame, _setCurrentFrame, notify=currentFrameChanged)
    frameRangeChanged = Signal()
    hasFrameRange = Property(bool, lambda self: self._hasFrameRange, notify=frameRangeChanged)
    firstFrame = Property(int, lambda self: self._firstFrame, notify=frameRangeChanged)
    lastFrame = Property(int, lambda self: self._lastFrame, notify=frameRangeChanged)
    autoRefreshChanged = Signal()
    autoRefresh = Property(bool, lambda self: self._autoRefresh, _setAutoRefresh, notify=autoRefreshChanged)
    renderingChanged = Signal()
    rendering = Property(bool, lambda self: self._rendering, notify=renderingChanged)
    errorChanged = Signal()
    errorMessage = Property(str, lambda self: self._errorMessage, notify=errorChanged)
    errorNode = Property(str, lambda self: self._errorNode, notify=errorChanged)
    imageChanged = Signal()
    imageRevision = Property(int, lambda self: self._imageRevision, notify=imageChanged)
    # Format of the displayed image
    imageWidth = Property(int, lambda self: self._formatWidth, notify=imageChanged)
    imageHeight = Property(int, lambda self: self._formatHeight, notify=imageChanged)
    # Bbox of the displayed image, in viewer coordinates (from the top-left corner of the format, y down)
    bboxX = Property(int, lambda self: self._bbox.x0, notify=imageChanged)
    bboxY = Property(int, lambda self: self._formatHeight - self._bbox.y1, notify=imageChanged)
    bboxWidth = Property(int, lambda self: self._bbox.width, notify=imageChanged)
    bboxHeight = Property(int, lambda self: self._bbox.height, notify=imageChanged)
    # Displayed part of the image (the visible area), in viewer coordinates
    displayX = Property(int, lambda self: self._displayRegion.x0, notify=imageChanged)
    displayY = Property(int, lambda self: self._formatHeight - self._displayRegion.y1, notify=imageChanged)
    displayWidth = Property(int, lambda self: self._displayRegion.width, notify=imageChanged)
    displayHeight = Property(int, lambda self: self._displayRegion.height, notify=imageChanged)
    renderTime = Property(float, lambda self: self._renderTime, notify=imageChanged)
    layersChanged = Signal()
    layers = Property("QVariantList", lambda self: self._layers, notify=layersChanged)
    currentLayerChanged = Signal()
    currentLayer = Property(str, lambda self: self._currentLayer, _setCurrentLayer, notify=currentLayerChanged)
    channelModeChanged = Signal()
    channelMode = Property(str, lambda self: self._channelMode, _setChannelMode, notify=channelModeChanged)
    channelModes = Property("QVariantList", lambda self: CHANNEL_MODES, constant=True)
    exposureChanged = Signal()
    exposure = Property(float, lambda self: self._exposure, _setExposure, notify=exposureChanged)
    gammaChanged = Signal()
    displayTransformChanged = Signal()
    displayTransform = Property(str, lambda self: self._displayTransform, _setDisplayTransform, notify=displayTransformChanged)
    colorConfigChanged = Signal()
    displayTransforms = Property("QVariantList", lambda self: self._colorConfig.displayTransforms(), notify=colorConfigChanged)
    # Colorspaces of the color config, for the colorspace parameters (Read, Write, OCIOColorSpace)
    colorspaces = Property("QVariantList", lambda self: self._colorConfig.colorspaceNames(), notify=colorConfigChanged)
    colorConfigDescription = Property(str, lambda self: self._colorConfig.description, notify=colorConfigChanged)
    gamma = Property(float, lambda self: self._gamma, _setGamma, notify=gammaChanged)
    # Keep one pixel out of `displayStride` for display (set by the viewer when zoomed out)
    displayStrideChanged = Signal()
    displayStride = Property(int, lambda self: self._displayStride, _setDisplayStride, notify=displayStrideChanged)
    actionProgressChanged = Signal()
    # Progress of the running node action in [0, 1] (-1: none), and its name
    actionProgress = Property(float, lambda self: self._actionProgress, notify=actionProgressChanged)
    actionText = Property(str, lambda self: self._actionText, notify=actionProgressChanged)
    # The node and the button of the running action
    actionNode = Property(QObject, lambda self: self._actionNode, notify=actionProgressChanged)
    actionName = Property(str, lambda self: self._actionName, notify=actionProgressChanged)
    trackingProgressChanged = Signal()
    # Progress of the tracking in [0, 1], -1 when not tracking
    trackingProgress = Property(float, lambda self: self._trackingProgress, notify=trackingProgressChanged)
    playingChanged = Signal()
    # 0: stopped, 1: playing forward, -1: playing backward
    playDirection = Property(int, lambda self: self._playDirection, notify=playingChanged)
    playSettingsChanged = Signal()
    playStep = Property(int, lambda self: self._playStep, _setPlayStep, notify=playSettingsChanged)
    fps = Property(float, lambda self: self._fps, _setFps, notify=playSettingsChanged)
    loop = Property(bool, lambda self: self._loop, _setLoop, notify=playSettingsChanged)
    measuredFpsChanged = Signal()
    # Frames per second actually played
    measuredFps = Property(float, lambda self: self._measuredFps, notify=measuredFpsChanged)
    pickAttributeChanged = Signal()
    # The color attribute the viewer eyedropper is picking for (None: no picking)
    pickAttribute = Property(QObject, lambda self: self._pickAttribute, _setPickAttribute, notify=pickAttributeChanged)
    statusRevisionChanged = Signal()
    statusRevision = Property(int, lambda self: self._statusRevision, notify=statusRevisionChanged)
    # Memory used by the cached images of the compositing, its budget and the physical memory (bytes)
    cacheUsedBytes = Property(float, lambda self: float(self._evaluator.cache.usedBytes), notify=statusRevisionChanged)
    cacheMaxBytes = Property(float, lambda self: float(self._evaluator.cache.maxBytes), constant=True)
    systemMemoryBytes = Property(float, lambda self: float(self._systemMemory), constant=True)
