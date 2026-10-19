"""
"Disconnect and Bake" of an edge going to a compositing node: the value its input received is kept in the node,
as keys when it changes over time (meshroom/live/animation.py).

- A connected parameter keeps the value of its source, and the keys of its source when it is animated.
- A camera or matrix input is turned into the parameters of its node (desc.ImageOpNode.bakeStream), from the values it
  received over a range of frames (e.g. a Camera following a Resection gets keys of its translate / rotate).

These functions only compute the new values: {parameter name: value or keys [[frame, value], ...]}; the user interface
removes the edge and sets them (one undoable step).
"""
from __future__ import annotations

from typing import Dict, Iterable

from . import animation as anim


def isKeys(value) -> bool:
    """ Whether a baked value is a list of keys [[frame, value], ...]. """
    return isinstance(value, list) and bool(value) and all(isinstance(key, list) and len(key) == 2 for key in value)


def topAttribute(attribute):
    """ The parameter an attribute belongs to (a group for its children: x of a 2D value, r of a color). """
    while attribute.root is not None:
        attribute = attribute.root
    return attribute


def canBake(attribute) -> bool:
    """ Whether the connected input `attribute` can be baked: a parameter, or a stream its node turns into parameters. """
    from meshroom.core import desc
    if getattr(attribute, "isStream", False):
        nodeDesc = getattr(attribute.node, "nodeDesc", None)
        return attribute.streamType != "image" and nodeDesc is not None and \
            type(nodeDesc).bakeStream is not desc.ImageOpNode.bakeStream
    return not isinstance(attribute.desc, desc.Flow)


def _sourceKeys(source):
    """ The keys of the animated parameter `source` belongs to (its top parameter), or None. """
    node = source.node
    if source.isOutput or not hasattr(node, "_animationKeys"):
        return None
    return node._animationKeys().get(topAttribute(source).name)


def bakedParameter(attribute) -> Dict[str, object]:
    """
    {name of the top parameter: value or keys} keeping what the connected parameter `attribute` receives: the value of
    its source, or the keys of its source when it is animated (for a child of a group, x of a 2D value: the keys of the
    group, with this child taken from the source at each key).
    """
    from .evaluator import _snapshotValue
    source = attribute.inputRootLink
    top = topAttribute(attribute)
    keys = _sourceKeys(source) if source is not None else None
    if attribute is top:
        return {top.name: [list(key) for key in keys] if keys else _snapshotValue(attribute)}
    current = dict(_snapshotValue(top))
    current[attribute.name] = _snapshotValue(attribute)
    if not keys:
        return {top.name: current}
    node = attribute.node
    ownKeys = anim.parseAnimation(node.internalAttribute(anim.ANIMATION).value).get(top.name) \
        if node.hasInternalAttribute(anim.ANIMATION) else None
    result = []
    for frame, value in keys:
        base = dict(anim.valueAt(ownKeys, frame)) if ownKeys else dict(current)
        base[attribute.name] = value.get(source.name) if isinstance(value, dict) else value
        result.append([frame, base])
    return {top.name: result}


def bakedStream(attribute, frames: Iterable[int], currentFrame: int, evaluator=None) -> Dict[str, object]:
    """
    The parameters of the node of a camera / matrix input `attribute` reproducing what it receives at `frames`
    (desc.ImageOpNode.bakeStream). Raises RuntimeError if the node cannot bake that input.
    """
    from .evaluator import Evaluator, RenderContext, buildPlan
    node = attribute.node
    evaluator = evaluator or Evaluator()
    plan = buildPlan(node)
    values = {int(frame): RenderContext(plan.targetNode, int(frame), evaluator, plan).inputData(attribute.name)
              for frame in frames}
    result = node.nodeDesc.bakeStream(RenderContext(plan.targetNode, int(currentFrame), evaluator, plan),
                                      attribute.name, values)
    if result is None:
        raise RuntimeError(f"'{attribute.label}' of {node.label} cannot be baked into parameters.")
    return result


def bakedValues(attribute, frames: Iterable[int], currentFrame: int, evaluator=None) -> Dict[str, object]:
    """ {parameter name: value or keys} keeping what the connected input `attribute` receives (see the module). """
    if getattr(attribute, "isStream", False):
        return bakedStream(attribute, frames, currentFrame, evaluator)
    return bakedParameter(attribute)


def applyToNode(node, baked: Dict[str, object], currentFrame: int):
    """
    The changes making a node hold baked values: ([(attribute, value)...], new animation JSON or None if the node has no
    animation). Keys become the animation of their parameter (its value: the one at the current frame); a constant value
    removes the animation of its parameter.
    """
    animation = node.internalAttribute(anim.ANIMATION) if node.hasInternalAttribute(anim.ANIMATION) else None
    keys = anim.parseAnimation(animation.value) if animation is not None else {}
    changes = []
    for name, value in baked.items():
        if not node.hasAttribute(name):
            continue
        if isKeys(value):
            frameKeys = [[float(frame), key] for frame, key in value]
            if animation is not None:
                keys[name] = frameKeys
            value = anim.valueAt(frameKeys, currentFrame)
        else:
            keys.pop(name, None)
        changes.append((node.attribute(name), value))
    return changes, (anim.dumpAnimation(keys) if animation is not None else None)
