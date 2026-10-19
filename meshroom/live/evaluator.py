"""
Pull-based, in-memory evaluation of live compositing nodes (`desc.ImageOpNode`).

Evaluation happens in two steps so that rendering can run in a worker thread:

1. `buildPlan(node)` (in the thread owning the graph) takes a snapshot of the branch upstream of
   a node: for each image node, its uid, descriptor, parameter values and image inputs.
2. `Evaluator.render(plan, frame)` (in any thread) renders a frame of the plan's target node,
   recursively and only for the nodes it needs, reusing the in-memory cache.
"""
from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import threading
import time
import traceback
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

from .cache import ImageCache
from .imageBuffer import BBox, ImageBuffer, ImageInfo


class EvalStatus(enum.Enum):
    """ Evaluation status of a node for a given frame. """
    NONE = 0
    RUNNING = 1
    SUCCESS = 2
    ERROR = 3
    # The node could not be computed because one of its inputs failed
    UPSTREAM_ERROR = 4


class EvaluationError(RuntimeError):
    """ Raised when a node of the branch fails. `nodeName` is the name of the failing node. """

    def __init__(self, nodeName: str, message: str):
        super().__init__(f"{nodeName}: {message}")
        self.nodeName = nodeName
        self.message = message


class EvaluationCancelled(Exception):
    """ Raised when an evaluation is cancelled because its result is no longer needed. """


class CancelToken:
    """ Shared flag used to stop an evaluation between two nodes. """

    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


# Plan

class PlanNode:
    """
    Snapshot of a live node, independent from the graph.

    Attributes:
        name: the node name.
        uid: the node uid, which identifies its output for given parameters and upstream graph.
        nodeDesc: the `desc.ImageOpNode` instance.
        values: dict {parameter name: value}.
        inputs: OrderedDict {image input name: uid of the connected PlanNode, or None}.
        dataInputs: {name of a camera / geo / matrix input: (uid of the connected PlanNode, its output name), or None}.
        optionalInputs: the names of the optional image inputs.
        error: an error known before rendering (e.g. a batch dependency not computed), or None.
        disabled: whether the node is disabled (its first image input is passed through).
    """

    def __init__(self, name, uid, nodeDesc, values, inputs, optionalInputs=(), error=None, disabled=False,
                 formatInputs=(), animation=None, context=None, dataInputs=None):
        self.name = name
        # Inputs of the other stream types (camera, geo, matrix), computed with Evaluator.data
        self.dataInputs = dataInputs or {}
        # Values outside of the node's attributes it depends on (desc.ImageOpNode.planContext)
        self.context = context
        self.uid = uid
        self.nodeDesc = nodeDesc
        self.values = values
        self.inputs = inputs
        self.optionalInputs = set(optionalInputs)
        self.error = error
        self.disabled = disabled
        # Inputs that only give a format (desc.Image infoOnly): never passed through
        self.formatInputs = set(formatInputs)
        # Keys of the animated parameters {name: [[frame, value], ...]} (see meshroom/live/animation.py)
        self.animation = animation or {}

    @property
    def firstInput(self) -> Optional[str]:
        """ The name of the first image input (passed through when the node is disabled). """
        return next((name for name in self.inputs if name not in self.formatInputs), None)

    @property
    def disabledPassThrough(self) -> Optional[str]:
        """ The input passed through by a disabled node: its first image input if connected, else None (black). """
        first = self.firstInput
        return first if first is not None and self.inputs[first] is not None else None


class RenderPlan:
    """ Snapshot of the branch upstream of a target node. """

    def __init__(self, target: str, nodes: Dict[str, PlanNode], proxy: float = 1.0):
        # uid of the target node
        self.target = target
        # {uid: PlanNode}
        self.nodes = nodes
        # Proxy scale (0.5: the images are rendered at half resolution), see buildPlan
        self.proxy = proxy

    @property
    def targetNode(self) -> PlanNode:
        return self.nodes[self.target]

    def keyOf(self, name: str) -> Optional[str]:
        """ The key of a node of the plan by its name (None if it is not in the plan). """
        for key, planNode in self.nodes.items():
            if planNode.name == name:
                return key
        return None

    def namesByUid(self) -> Dict[str, List[str]]:
        result = {}
        for planNode in self.nodes.values():
            result.setdefault(planNode.uid, []).append(planNode.name)
        return result


def _snapshotValue(attr):
    """ Return a plain Python copy of an attribute value (links and expressions resolved). """
    from meshroom.core.attribute import GroupAttribute, ListAttribute
    if isinstance(attr, ListAttribute):
        return [_snapshotValue(child) for child in attr.value]
    if isinstance(attr, GroupAttribute):
        return {child.name: _snapshotValue(child) for child in attr.value}
    value = attr.value
    if isinstance(value, str) and "{" in value:
        value = attr.evalValue
    return value


def _nodeAnimation(node) -> dict:
    """ The keys of the animated parameters of a node ({} if none). """
    from .animation import parseAnimation
    if not node.hasInternalAttribute("animation"):
        return {}
    animation = parseAnimation(node.internalAttribute("animation").value)
    # A connected parameter takes its value from its link, not from its keys
    result = {name: keys for name, keys in animation.items() if node.hasAttribute(name) and not node.attribute(name).isLink}
    # ... and from the keys of the parameter it is linked to, when that one is animated
    for attr in node.attributes:
        if attr.isInput and attr.isLink:
            linkKeys = node._animatedLinkKeys(attr) if hasattr(node, "_animatedLinkKeys") else None
            if linkKeys:
                result[attr.name] = parseAnimation(json.dumps({attr.name: linkKeys}))[attr.name]
    return result


def isImageOpNode(node) -> bool:
    from meshroom.core import desc
    return node is not None and isinstance(getattr(node, "nodeDesc", None), desc.ImageOpNode)


def upstreamImageNode(attr):
    """ Return the node connected to an image input attribute, or None if it is not connected. """
    if not attr.isLink:
        return None
    return attr.inputRootLink.node


def _batchDependencyError(node) -> Optional[str]:
    """
    Return an error message if a parameter of `node` is connected to the output of a batch node
    that has not been computed yet, None otherwise.
    """
    from meshroom.core import desc
    for attr in node.attributes:
        if attr.isOutput or isinstance(attr.desc, desc.Stream) or not attr.isLink:
            continue
        source = attr.inputRootLink.node
        if source is node or isImageOpNode(source):
            continue
        if source.isComputableType and not source.isComputed:
            return f"Batch dependency not computed: '{source.label}' (compute it first)."
    return None


# Subgraphs: a SubGraph node containing compositing nodes is evaluated through its subgraph, as if its
# nodes were in the graph (each one keeps its own cache); the nodes inside an entered subgraph reach the
# nodes connected to the SubGraph node through its GraphInput.

class _Scope:
    """
    Where a node of a plan is: the project graph (subGraphNode None), or the subgraph of `subGraphNode`.
    `prefix` makes the names of its nodes unique in the plan ("" for the graph of the evaluated node).
    """

    def __init__(self, subGraphNode=None, parent: Optional["_Scope"] = None, prefix: str = ""):
        self.subGraphNode = subGraphNode
        self.parent = parent
        self.prefix = prefix
        # Values outside of the nodes that their result depends on: the SubGraph nodes they are in
        self.signature = ""
        if subGraphNode is not None:
            # The values of its inputs (not all in its uid: the children of an AnySet). Not its uid: it changes with
            # the serialized subgraph each time the subgraph is written back (e.g. saved), while the uids of the nodes
            # inside already follow its content
            from meshroom.core.subgraph.executor import plainValue
            values = json.dumps(plainValue(subGraphNode.attribute("ins")), sort_keys=True, default=str)
            self.signature = hashlib.sha1(f"{values}|{parent.signature if parent else ''}".encode()).hexdigest()[:12]

    def child(self, subGraphNode) -> "_Scope":
        return _Scope(subGraphNode, self, f"{self.prefix}{subGraphNode.name}/")


def _scopeOf(graph, prefix: str = "") -> _Scope:
    """ The scope of the nodes of a graph: an entered subgraph knows the SubGraph node running it. """
    subGraphNode = getattr(graph, "subGraphNode", None)
    if subGraphNode is None:
        return _Scope(None, None, prefix)
    return _Scope(subGraphNode, _scopeOf(subGraphNode.graph, prefix + "../"), prefix)


_INNER_GRAPHS: "OrderedDict[tuple, object]" = OrderedDict()


def _innerGraph(subGraphNode):
    """ The Graph of the subgraph of a SubGraph node (built once per content, a few kept). """
    from meshroom.core.subgraph.editing import buildSubGraph
    content = subGraphNode.subGraph.value or subGraphNode.template.value
    key = (subGraphNode.name, hashlib.sha1(str(content).encode()).hexdigest())
    graph = _INNER_GRAPHS.get(key)
    if graph is None:
        graph = buildSubGraph(subGraphNode)
        _INNER_GRAPHS[key] = graph
        while len(_INNER_GRAPHS) > 16:
            _INNER_GRAPHS.popitem(last=False)
    _INNER_GRAPHS.move_to_end(key)
    return graph


def _isSubGraphNode(node) -> bool:
    return getattr(node, "nodeType", "") == "SubGraph" and node.hasAttribute("outs")


def _subGraphRole(node) -> str:
    return getattr(getattr(node, "nodeDesc", None), "subGraphRole", "")


def _portPath(attribute) -> list:
    """ ["port", "child"...] of an attribute under the "ports" of a GraphInput / GraphOutput node ([] if not). """
    path = attribute.fullName.split(".")[1:]
    return path[1:] if len(path) > 1 and path[0] == "ports" else []


def _child(attribute, path: list):
    """ The child of an attribute at a path of names (None if missing). """
    for name in path:
        value = attribute.value
        attribute = value.get(name) if hasattr(value, "get") and name in value.keys() else None
        if attribute is None:
            return None
    return attribute


def _resolveImageSource(attr, scope: _Scope):
    """
    The compositing node (and its scope) whose image an image input `attr` of a node of `scope` reads:
    through the GraphInput of the subgraph (the input of the SubGraph node) and through the SubGraph nodes
    (the node connected to the port of their GraphOutput). (None, None) if not connected to one.
    """
    node, scope, _ = _resolveStreamSource(attr, scope)
    return node, scope


def _resolveStreamSource(attr, scope: _Scope):
    """ As _resolveImageSource, for any stream input: (compositing node, its scope, its output name) or (None, None, None). """
    for _ in range(64):
        if attr is None or not attr.isLink:
            return None, None, None
        source = attr.inputRootLink
        node = source.node
        if _subGraphRole(node) == "input" and scope.subGraphNode is not None:
            # An input of the subgraph: what is connected to the SubGraph node, in the parent scope
            path = _portPath(source)
            attr = _child(scope.subGraphNode.attribute("ins"), path) if path else None
            scope = scope.parent
            continue
        if _isSubGraphNode(node):
            # An output of a SubGraph node: the node connected to that port of its GraphOutput
            path = source.fullName.split(".")[2:] if source.fullName.split(".")[1:2] == ["outs"] else []
            inner = _innerGraph(node)
            graphOutput = next((n for n in inner.nodes if _subGraphRole(n) == "output"), None)
            attr = _child(graphOutput.attribute("ports"), path) if graphOutput is not None and path else None
            scope = scope.child(node)
            continue
        if isImageOpNode(node):
            return node, scope, source.name
        return None, None, None
    return None, None, None


def _subGraphImageSource(subGraphNode):
    """ (compositing node, scope) giving the first image output of a SubGraph node, (None, None) if none. """
    from meshroom.core import desc
    scope = _scopeOf(subGraphNode.graph)
    for port in subGraphNode.attribute("outs").value:
        if not isinstance(port.desc, desc.Image):
            continue
        inner = _innerGraph(subGraphNode)
        graphOutput = next((n for n in inner.nodes if _subGraphRole(n) == "output"), None)
        attr = _child(graphOutput.attribute("ports"), [port.name]) if graphOutput is not None else None
        source, sourceScope = _resolveImageSource(attr, scope.child(subGraphNode))
        if source is not None:
            return source, sourceScope
    return None, None


def hasImageOutput(node) -> bool:
    from meshroom.core import desc
    return any(isinstance(output, desc.Image) for output in node.nodeDesc.outputs)


def isEvaluable(node) -> bool:
    """ Whether a node can be evaluated live: a compositing node giving an image, or a SubGraph node giving one. """
    if isImageOpNode(node):
        return hasImageOutput(node)
    try:
        return _isSubGraphNode(node) and _subGraphImageSource(node)[0] is not None
    except Exception:
        return False


def _valueInScope(attr, scope: _Scope):
    """ The value of a parameter; connected to the GraphInput of a subgraph: the value of the SubGraph input. """
    while attr.isLink and scope.subGraphNode is not None and _subGraphRole(attr.inputRootLink.node) == "input":
        path = _portPath(attr.inputRootLink)
        outer = _child(scope.subGraphNode.attribute("ins"), path) if path else None
        if outer is None:
            break
        attr, scope = outer, scope.parent
    return _snapshotValue(attr)


def proxyKey(uid: str, proxy: float = 1.0) -> str:
    """ The key of a node in a plan rendered at a proxy scale (its uid at full resolution). """
    return uid if proxy == 1.0 else f"{uid}@proxy{proxy:g}"


def buildPlan(node, proxy: float = 1.0) -> RenderPlan:
    """
    Snapshot the branch upstream of `node` (a live compositing node).
    Must be called from the thread owning the graph.

    Args:
        proxy: scale of the rendered images (1: full resolution, 0.5: half...). At a proxy scale, the
               Read nodes downscale their images, the parameters in pixels are scaled (RenderContext.value),
               and the nodes are keyed by `proxyKey` (their images are cached apart from the full ones).
    """
    from meshroom.core import desc
    target = node, _scopeOf(node.graph)
    if not isImageOpNode(node):
        # A SubGraph node: the compositing node giving its first image output
        target = _subGraphImageSource(node) if _isSubGraphNode(node) else (None, None)
        if target[0] is None:
            raise ValueError(f"'{node.name}' is not a live compositing node.")
    nodes: Dict[str, PlanNode] = {}
    # Key of each visited node: its uid at the proxy scale, with an extra key for the nodes reading
    # values outside of their own attributes (desc.ImageOpNode.planContext, e.g. a Switch expression)
    # and for every node downstream of them
    keys: Dict[str, str] = {}
    extraKeys: Dict[str, str] = {}

    def visit(current, scope: _Scope):
        name = scope.prefix + current.name
        if name in keys:
            return
        keys[name] = None  # Being visited
        inputs = OrderedDict()
        dataInputs = {}
        upstreamExtras = []
        optional = []
        formatInputs = []
        values = {}
        for attr in current.attributes:
            if attr.isOutput:
                continue
            if isinstance(attr.desc, desc.Image):
                if attr.desc.optional:
                    optional.append(attr.name)
                if attr.desc.infoOnly:
                    formatInputs.append(attr.name)
                source, sourceScope = _resolveImageSource(attr, scope)
                if source is None:
                    inputs[attr.name] = None
                else:
                    visit(source, sourceScope)
                    sourceName = sourceScope.prefix + source.name
                    inputs[attr.name] = keys[sourceName]
                    upstreamExtras.append(extraKeys.get(sourceName, ""))
                    if sourceScope is not scope:
                        # Through the boundary of a subgraph: the uid of the node does not carry the upstream one
                        upstreamExtras.append(keys[sourceName])
                continue
            if isinstance(attr.desc, desc.Stream):
                # A camera, geo or matrix input: computed on demand (RenderContext.inputData)
                source, sourceScope, outputName = _resolveStreamSource(attr, scope)
                if source is None:
                    dataInputs[attr.name] = None
                    continue
                visit(source, sourceScope)
                sourceName = sourceScope.prefix + source.name
                dataInputs[attr.name] = (keys[sourceName], outputName)
                upstreamExtras.append(extraKeys.get(sourceName, ""))
                if sourceScope is not scope:
                    upstreamExtras.append(keys[sourceName])
                continue
            values[attr.name] = _valueInScope(attr, scope)
        disabled = current.hasInternalAttribute("disable") and bool(current.internalAttribute("disable").value)
        try:
            context = current.nodeDesc.planContext(current) if hasattr(current.nodeDesc, "planContext") else None
        except Exception:
            logging.debug("[Live] planContext failed on %s:\n%s", current.name, traceback.format_exc())
            context = None
        ownExtra = json.dumps(context, sort_keys=True, default=str) if context is not None else ""
        # A node in a subgraph also depends on the SubGraph node (the values of its inputs)
        extras = [extra for extra in [ownExtra, scope.signature] + upstreamExtras if extra]
        extraKeys[name] = hashlib.sha1("|".join(extras).encode()).hexdigest()[:12] if extras else ""
        key = proxyKey(current._uid, proxy) + (f"#{extraKeys[name]}" if extras else "")
        keys[name] = key
        nodes[key] = PlanNode(
            name=name, uid=key, nodeDesc=current.nodeDesc, values=values,
            inputs=inputs, optionalInputs=optional, error=None if disabled else _batchDependencyError(current),
            disabled=disabled, formatInputs=formatInputs, animation=_nodeAnimation(current), context=context,
            dataInputs=dataInputs)

    visit(*target)
    return RenderPlan(keys[target[1].prefix + target[0].name], nodes, proxy)


# Rendering

class RenderContext:
    """ What a node needs to know to render a frame. """

    def __init__(self, planNode: PlanNode, frame: int, evaluator: "Evaluator", plan: RenderPlan,
                 region: Optional[BBox] = None):
        self.planNode = planNode
        self.frame = frame
        self.evaluator = evaluator
        self._plan = plan
        # Region of the output that is needed (region of interest), None: the whole image
        self.region = region

    @property
    def nodeName(self) -> str:
        return self.planNode.name

    @property
    def planContext(self):
        """ The values snapshotted for the node by desc.ImageOpNode.planContext (None by default). """
        return self.planNode.context

    @property
    def proxy(self) -> float:
        """ Proxy scale of the render (1: full resolution). """
        return getattr(self._plan, "proxy", 1.0) if self._plan is not None else 1.0

    def value(self, name: str):
        """
        Return the value of a parameter of the node (at the rendered frame when it is animated). At a
        proxy scale, the parameters in pixels (the node's `proxyScaled`, and width / height) are scaled.
        """
        keys = self.planNode.animation.get(name)
        if keys:
            from .animation import valueAt
            value = valueAt(keys, self.frame)
        else:
            value = self.planNode.values[name]
        proxy = self.proxy
        if proxy != 1.0 and (name in ("width", "height") or name in getattr(self.planNode.nodeDesc, "proxyScaled", ())):
            value = scaleValue(value, proxy, minimum=1 if name in ("width", "height") else None)
        return value

    def fullResolutionValue(self, name: str):
        """ The value of a parameter, not scaled to the proxy. """
        keys = self.planNode.animation.get(name)
        if keys:
            from .animation import valueAt
            return valueAt(keys, self.frame)
        return self.planNode.values[name]

    def info(self) -> ImageInfo:
        """ Return the ImageInfo of the node itself (as returned by its getInfo). """
        return self.evaluator.info(self._plan, self.planNode.uid)

    def inputInfo(self, inputName: str) -> Optional[ImageInfo]:
        """ Return the ImageInfo of an image input, or None if it is not connected. """
        uid = self.planNode.inputs.get(inputName)
        if uid is None:
            return None
        return self.evaluator.info(self._plan, uid)

    def inputValues(self, inputName: str) -> Optional[dict]:
        """
        The parameter values of the node connected to an image input (e.g. the tracks of a Tracker),
        through the disabled and pass-through nodes (Dot, Clone), or None if it is not connected.
        """
        uid = self.planNode.inputs.get(inputName)
        seen = set()
        while uid is not None and uid not in seen:
            seen.add(uid)
            planNode = self._plan.nodes[uid]
            if not planNode.disabled and not getattr(planNode.nodeDesc, "passThroughValues", False):
                from .animation import animatedValues
                return animatedValues(planNode.values, planNode.animation, self.frame)
            first = planNode.firstInput
            uid = planNode.inputs.get(first) if first else None
        return None

    def hasInput(self, inputName: str) -> bool:
        """ Whether an input (image, camera, geo or matrix) is connected. """
        if self.planNode.inputs.get(inputName) is not None:
            return True
        return self.planNode.dataInputs.get(inputName) is not None

    def inputData(self, inputName: str, frame: Optional[int] = None):
        """
        The value of a camera, geo or matrix input at the frame of the render (or `frame`): a CameraData, a GeoData,
        a 4x4 matrix (see meshroom/live/scene3d.py), or None if it is not connected.
        """
        source = self.planNode.dataInputs.get(inputName)
        if source is None:
            return None
        return self.evaluator.data(self._plan, source[0], source[1], self.frame if frame is None else frame, self._token)

    def ownData(self, outputName: str, frame: int):
        """ The value of a (non image) output of the node itself at another frame (e.g. a solve warm-started). """
        return self.evaluator.data(self._plan, self.planNode.uid, outputName, frame, self._token)

    def checkCancelled(self):
        """ Raise EvaluationCancelled if the render is no longer needed (called between bands by the ops). """
        if self._token is not None and self._token.cancelled:
            raise EvaluationCancelled()

    def renderInput(self, inputName: str, frame: int) -> Optional[ImageBuffer]:
        """
        Render another frame of an image input (for time-based nodes). None if disconnected. With a region
        of interest, only the part of the input that the node needs (its inputRegions).
        """
        uid = self.planNode.inputs.get(inputName)
        if uid is None:
            return None
        region = None
        if self.region is not None:
            regions = self.evaluator._inputRegions(self.planNode, self, self.region)
            region = regions.get(inputName) if regions else None
        return self.evaluator._renderNode(self._plan, uid, frame, self._token, region)

    _token: Optional[CancelToken] = None


def _cropped(image: ImageBuffer, region: BBox) -> ImageBuffer:
    """ An image limited to a region of interest (its bbox intersected with the region). """
    from .ops import alignToBBox
    bbox = image.bbox.intersection(region)
    if bbox == image.bbox or bbox.isEmpty:
        return image
    return image.mapLayers(lambda pixels: alignToBBox(pixels, image.bbox, bbox), bbox=bbox)


def scaleValue(value, scale: float, minimum=None):
    """ A value in pixels at a proxy scale: numbers (integers stay integers) and groups of numbers ({x, y}). """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        scaled = int(round(value * scale))
        return max(minimum, scaled) if minimum is not None else scaled
    if isinstance(value, float):
        return value * scale
    if isinstance(value, dict):
        return {key: scaleValue(item, scale, minimum) for key, item in value.items()}
    if isinstance(value, list):
        return [scaleValue(item, scale, minimum) for item in value]
    return value


StatusListener = Callable[[str, int, EvalStatus, str], None]


def _dataBytes(value) -> int:
    """ Memory used by a camera / geo / matrix value (the arrays of a geometry). """
    total = 0
    for name in ("vertices", "faceCounts", "faceIndices", "_triangles", "_edges"):
        array = getattr(value, name, None)
        total += getattr(array, "nbytes", 0)
    return total + 256

# Pass-through marker of a disabled node without image input
_BLACK = "__black__"


class Evaluator:
    """
    Renders frames of live compositing nodes, caching every node output in memory.

    Statuses are tracked per (node uid, frame) and reported to an optional listener
    `listener(uid, frame, status, message)`, called from the rendering thread.
    """

    def __init__(self, cache: Optional[ImageCache] = None, listener: Optional[StatusListener] = None):
        self.cache = cache if cache is not None else ImageCache()
        self.listener = listener
        self._statuses: Dict[Tuple[str, int], Tuple[EvalStatus, str]] = {}
        # Last status of each node uid, whatever the frame (nodes rendered at other frames by time nodes)
        self._lastStatuses: Dict[str, Tuple[EvalStatus, str]] = {}
        # Duration of the last render of each node itself (its inputs excluded), in seconds
        self._times: Dict[Tuple[str, int], float] = {}
        self._infos: Dict[str, ImageInfo] = {}
        # Layer names of the last image rendered for each node uid (e.g. the layers offered by Shuffle)
        self._layerNames: Dict[str, List[str]] = {}
        # Region of the cached images rendered for a region of interest ({(uid, frame): BBox}; absent: whole)
        self._rois: Dict[Tuple[str, int], BBox] = {}
        # Values of the camera / geo / matrix outputs: {(uid, output name, frame): value}, least recently used first
        self._data: "OrderedDict[Tuple[str, str, int], object]" = OrderedDict()
        self._dataBytes = 0
        self._lock = threading.RLock()

    # Camera / geo / matrix values kept in memory: number, and memory (MESHROOM_LIVE_DATA_CACHE_GB, 2 by default)
    DATA_CACHE_SIZE = 4096
    DATA_CACHE_BYTES = int(float(os.environ.get("MESHROOM_LIVE_DATA_CACHE_GB", "2")) * 1024 ** 3)

    def data(self, plan: RenderPlan, uid: str, outputName: str, frame: int, token: Optional[CancelToken] = None):
        """
        The value of a camera, geo or matrix output of a plan node at a frame (desc.ImageOpNode.computeData), cached.

        Raises:
            EvaluationError: if the node or one of its inputs fails.
            EvaluationCancelled: if `token` is cancelled.
        """
        key = (uid, outputName, frame)
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        if token is not None and token.cancelled:
            raise EvaluationCancelled()
        planNode = plan.nodes[uid]
        if planNode.error:
            self._setStatus(uid, frame, EvalStatus.ERROR, planNode.error)
            raise EvaluationError(planNode.name, planNode.error)
        ctx = RenderContext(planNode, frame, self, plan)
        ctx._token = token
        try:
            if planNode.disabled:
                value = planNode.nodeDesc.disabledData(ctx, outputName)
            else:
                value = planNode.nodeDesc.computeData(ctx, outputName)
        except EvaluationCancelled:
            raise
        except EvaluationError as e:
            if e.nodeName != planNode.name:
                self._setStatus(uid, frame, EvalStatus.UPSTREAM_ERROR, f"Input failed ({e.nodeName}).")
            raise
        except Exception as e:
            logging.debug("[Live] %s (%s) failed:\n%s", planNode.name, outputName, traceback.format_exc())
            message = str(e) or e.__class__.__name__
            self._setStatus(uid, frame, EvalStatus.ERROR, message)
            raise EvaluationError(planNode.name, message) from e
        self.storeData(uid, outputName, frame, value)
        if self.status(uid, frame)[0] == EvalStatus.NONE:
            # A node rendering an image too keeps the status of its image at that frame
            self._setStatus(uid, frame, EvalStatus.SUCCESS)
        return value

    def cachedData(self, uid: str, outputName: str, frame: int):
        """ The cached value of a camera / geo / matrix output, or None. """
        with self._lock:
            return self._data.get((uid, outputName, frame))

    def storeData(self, uid: str, outputName: str, frame: int, value):
        """
        Keep a camera / geo / matrix value (also one computed on the way, e.g. the poses solved before a frame). The
        values are kept within DATA_CACHE_SIZE entries and DATA_CACHE_BYTES (geometries), least recently used first.
        """
        key = (uid, outputName, frame)
        with self._lock:
            if key in self._data:
                self._dataBytes -= _dataBytes(self._data.pop(key))
            self._data[key] = value
            self._dataBytes += _dataBytes(value)
            while len(self._data) > 1 and (len(self._data) > self.DATA_CACHE_SIZE or self._dataBytes > self.DATA_CACHE_BYTES):
                _, dropped = self._data.popitem(last=False)
                self._dataBytes -= _dataBytes(dropped)

    def discardData(self, uids=None):
        """ Forget the camera / geo / matrix values (of some node uids, or all). """
        with self._lock:
            if uids is None:
                self._data.clear()
                self._dataBytes = 0
            else:
                uids = set(uids)
                for key in [key for key in self._data if key[0] in uids]:
                    self._dataBytes -= _dataBytes(self._data.pop(key))

    def layerNames(self, uid: str) -> Optional[List[str]]:
        """ The layers of the last image rendered for a node uid (None if it was never rendered). """
        with self._lock:
            names = self._layerNames.get(uid)
            return list(names) if names is not None else None

    def _store(self, key: Tuple[str, int], result: ImageBuffer, region: Optional[BBox] = None):
        """ Keep a rendered image in the cache (rendered for `region` only, or whole), and its layer names. """
        self.cache.put(key, result)
        with self._lock:
            self._layerNames[key[0]] = list(result.layerNames)
            if region is None:
                self._rois.pop(key, None)
            else:
                self._rois[key] = region

    def _cachedFor(self, key: Tuple[str, int], region: Optional[BBox]) -> Optional[ImageBuffer]:
        """ The cached image of a node at a frame when it covers `region` (None: the whole image is needed). """
        cached = self.cache.get(key)
        if cached is None:
            return None
        with self._lock:
            cachedRegion = self._rois.get(key)
        if cachedRegion is None:
            return cached
        if region is not None and region.intersection(cachedRegion) == region:
            return cached
        return None

    def _inputRegions(self, planNode: PlanNode, ctx: "RenderContext", region: BBox) -> Optional[dict]:
        """ {input name: region} a node needs to render `region` of its output, or None (whole inputs). """
        try:
            return planNode.nodeDesc.inputRegions(ctx, region)
        except Exception:
            logging.debug("[Live] inputRegions failed on %s:\n%s", planNode.name, traceback.format_exc())
            return None

    # Statuses

    def status(self, uid: str, frame: int) -> Tuple[EvalStatus, str]:
        """ Return (status, message) of a node uid at a frame. """
        with self._lock:
            return self._statuses.get((uid, frame), (EvalStatus.NONE, ""))

    def _setStatus(self, uid: str, frame: int, status: EvalStatus, message: str = ""):
        with self._lock:
            if status == EvalStatus.NONE:
                self._statuses.pop((uid, frame), None)
            else:
                self._statuses[(uid, frame)] = (status, message)
                self._lastStatuses[uid] = (status, message)
        if self.listener:
            try:
                self.listener(uid, frame, status, message)
            except Exception:
                logging.exception("[Live] Status listener error")

    def lastStatus(self, uid: str) -> Tuple[EvalStatus, str]:
        """ The last (status, message) of a node uid, whatever the frame. """
        with self._lock:
            return self._lastStatuses.get(uid, (EvalStatus.NONE, ""))

    def renderTime(self, uid: str, frame: int) -> Optional[float]:
        """ Duration in seconds of the last render of a node itself at a frame (0 for a pass-through), or None. """
        with self._lock:
            return self._times.get((uid, frame))

    def clearStatuses(self):
        with self._lock:
            self._statuses.clear()
            self._lastStatuses.clear()
            self._infos.clear()
            self._times.clear()
            self._layerNames.clear()
            self._data.clear()
            self._dataBytes = 0

    def clearInfos(self):
        """ Forget the cached ImageInfos (e.g. to detect again the frame range of sequences on disk). """
        with self._lock:
            self._infos.clear()

    # Info

    def info(self, plan: RenderPlan, uid: Optional[str] = None) -> ImageInfo:
        """ Return the ImageInfo of a plan node (the target by default). Cached per uid. """
        uid = uid or plan.target
        with self._lock:
            cached = self._infos.get(uid)
        if cached is not None:
            return cached
        planNode = plan.nodes[uid]
        if planNode.error:
            raise EvaluationError(planNode.name, planNode.error)
        if planNode.disabled and planNode.disabledPassThrough is not None:
            return self.info(plan, planNode.inputs[planNode.disabledPassThrough])
        inputsInfo = {}
        valuesOnly = set(getattr(planNode.nodeDesc, "valuesOnlyInputs", ()))
        # A node choosing one of its inputs (Switch) does not fail on the error of another one
        tolerant = getattr(planNode.nodeDesc, "tolerantInputs", False)
        for inputName, inputUid in planNode.inputs.items():
            if not inputUid or inputName in valuesOnly:
                inputsInfo[inputName] = None
                continue
            try:
                inputsInfo[inputName] = self.info(plan, inputUid)
            except EvaluationError:
                if not tolerant:
                    raise
                inputsInfo[inputName] = None
        ctx = RenderContext(planNode, 0, self, plan)
        try:
            result = planNode.nodeDesc.getInfo(ctx, inputsInfo)
        except EvaluationError:
            raise
        except Exception as e:
            raise EvaluationError(planNode.name, str(e)) from e
        with self._lock:
            self._infos[uid] = result
        return result

    # Render

    def render(self, plan: RenderPlan, frame: int, token: Optional[CancelToken] = None,
               region: Optional[BBox] = None) -> ImageBuffer:
        """
        Render a frame of the plan's target node.

        Args:
            region: region of interest (Nuke coordinates): only this part of the image is computed (the
                    nodes that can limit their work to it, see ImageOpNode.inputRegions); None: the whole image.

        Raises:
            EvaluationError: if a node of the branch fails.
            EvaluationCancelled: if `token` is cancelled before the end.
        """
        return self._renderNode(plan, plan.target, frame, token, region)

    def _markSucceeded(self, plan: RenderPlan, uid: str, frame: int):
        """
        Mark a node whose result is in cache, and its upstream nodes, as succeeded at `frame`:
        a cached result implies that its whole branch was computed successfully.
        """
        if self.status(uid, frame)[0] == EvalStatus.SUCCESS:
            return
        self._setStatus(uid, frame, EvalStatus.SUCCESS)
        planNode = plan.nodes[uid]
        if planNode.nodeDesc.timeBased and not planNode.disabled:
            # Its inputs were rendered at other frames
            return
        notRendered = self._infoOnlyInputs(plan, planNode, frame)
        for inputName, inputUid in planNode.inputs.items():
            if inputUid is not None and inputName not in notRendered:
                self._markSucceeded(plan, inputUid, frame)

    def _infoOnlyInputs(self, plan: RenderPlan, planNode: PlanNode, frame: int) -> set:
        """ The image inputs of a node that are only used for their ImageInfo (not rendered). """
        try:
            return set(planNode.nodeDesc.infoOnlyInputs(RenderContext(planNode, frame, self, plan)))
        except Exception:
            logging.debug("[Live] infoOnlyInputs failed on %s:\n%s", planNode.name, traceback.format_exc())
            return set()

    def _passThroughInput(self, plan: RenderPlan, planNode: PlanNode, frame: int) -> Optional[str]:
        """ Return the image input a node passes through (disabled or no-op node), or None. """
        if planNode.disabled:
            passThrough = planNode.disabledPassThrough
            return passThrough if passThrough is not None else _BLACK
        try:
            return planNode.nodeDesc.isIdentity(RenderContext(planNode, frame, self, plan))
        except Exception:
            logging.debug("[Live] isIdentity failed on %s:\n%s", planNode.name, traceback.format_exc())
            return None

    def _renderNode(self, plan: RenderPlan, uid: str, frame: int, token: Optional[CancelToken],
                    region: Optional[BBox] = None) -> ImageBuffer:
        key = (uid, frame)
        cached = self._cachedFor(key, region)
        if cached is not None:
            with self._lock:
                self._layerNames[uid] = list(cached.layerNames)
            self._markSucceeded(plan, uid, frame)
            return cached

        planNode = plan.nodes[uid]
        if token is not None and token.cancelled:
            raise EvaluationCancelled()
        if planNode.error:
            self._setStatus(uid, frame, EvalStatus.ERROR, planNode.error)
            raise EvaluationError(planNode.name, planNode.error)

        # A disabled or no-op node passes one of its inputs through, without computing anything
        passThrough = self._passThroughInput(plan, planNode, frame)
        if passThrough == _BLACK:
            # A disabled node without connected input (Read, Constant...) outputs transparent black
            try:
                info = self.info(plan, uid)
            except EvaluationError as e:
                self._setStatus(uid, frame, EvalStatus.ERROR, e.message)
                raise
            result = ImageBuffer.constant(info.width, info.height, (0, 0, 0, 0))
            self._store(key, result)
            self._setStatus(uid, frame, EvalStatus.SUCCESS)
            return result
        if passThrough is not None:
            inputUid = planNode.inputs.get(passThrough)
            if inputUid is None:
                message = f"Input not connected: {passThrough}."
                self._setStatus(uid, frame, EvalStatus.ERROR, message)
                raise EvaluationError(planNode.name, message)
            try:
                result = self._renderNode(plan, inputUid, frame, token, region)
            except EvaluationError as e:
                self._setStatus(uid, frame, EvalStatus.UPSTREAM_ERROR, f"Input '{passThrough}' failed ({e.nodeName}).")
                raise
            with self._lock:
                inputRegion = self._rois.get((inputUid, frame))
            self._store(key, result, inputRegion)
            with self._lock:
                self._times[key] = 0.0
            self._setStatus(uid, frame, EvalStatus.SUCCESS)
            return result

        # Render the inputs first: the node is only "running" during its own computation.
        # Time-based nodes render the frames they need themselves (ctx.renderInput), and the inputs only
        # used for their format (ctx.inputInfo) are not rendered.
        notRendered = set(planNode.inputs) if planNode.nodeDesc.timeBased else \
            self._infoOnlyInputs(plan, planNode, frame)
        missing = [name for name, inputUid in planNode.inputs.items()
                   if inputUid is None and name not in planNode.optionalInputs]
        if missing:
            message = f"Input not connected: {', '.join(missing)}."
            self._setStatus(uid, frame, EvalStatus.ERROR, message)
            raise EvaluationError(planNode.name, message)
        ctx = RenderContext(planNode, frame, self, plan, region)
        ctx._token = token
        # Region of interest: the parts of the inputs the node needs (None: whole inputs)
        regions = self._inputRegions(planNode, ctx, region) if region is not None else None
        inputs = {}
        for inputName, inputUid in planNode.inputs.items():
            if inputUid is None or inputName in notRendered:
                inputs[inputName] = None
                continue
            try:
                inputs[inputName] = self._renderNode(plan, inputUid, frame, token,
                                                     regions.get(inputName) if regions else None)
            except EvaluationError as e:
                self._setStatus(uid, frame, EvalStatus.UPSTREAM_ERROR, f"Input '{inputName}' failed ({e.nodeName}).")
                raise

        if token is not None and token.cancelled:
            raise EvaluationCancelled()

        self._setStatus(uid, frame, EvalStatus.RUNNING)
        start = time.perf_counter()
        try:
            result = planNode.nodeDesc.render(ctx, inputs)
            if not isinstance(result, ImageBuffer):
                raise TypeError(f"render() returned {type(result).__name__} instead of an ImageBuffer.")
        except EvaluationCancelled:
            self._setStatus(uid, frame, EvalStatus.NONE)
            raise
        except EvaluationError as e:
            if e.nodeName == planNode.name:
                # The node's own getInfo failed (called from render through ctx.info())
                self._setStatus(uid, frame, EvalStatus.ERROR, e.message)
            else:
                # An input failed while rendering another of its frames (time-based nodes)
                self._setStatus(uid, frame, EvalStatus.UPSTREAM_ERROR, f"Input failed ({e.nodeName}).")
            raise
        except Exception as e:
            logging.debug("[Live] %s failed:\n%s", planNode.name, traceback.format_exc())
            message = str(e) or e.__class__.__name__
            self._setStatus(uid, frame, EvalStatus.ERROR, message)
            raise EvaluationError(planNode.name, message) from e
        elapsed = time.perf_counter() - start
        logging.debug("[Live] %s frame %d rendered in %.3fs", planNode.name, frame, elapsed)
        with self._lock:
            self._times[key] = elapsed
        if region is not None:
            result = _cropped(result, region)
        self._store(key, result, region)
        self._setStatus(uid, frame, EvalStatus.SUCCESS)
        return result
