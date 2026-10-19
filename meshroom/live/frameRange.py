"""
Frame range of the timeline of a compositing graph (in / out), saved in the graph header
(Graph.frameRange): {"first", "last", "source"}.

Entered by the user ("source" is ""), it stays. Otherwise it follows a Read node: the first Read of
the graph, the one created when there was none; the Reads created after it do not change it.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple


def followedFrameRange(graph, infoOf: Callable, lastSourceUid: Optional[str],
                       frameRange: Optional[dict] = None) -> Tuple[Optional[dict], Optional[str]]:
    """
    The frame range the timeline of `graph` should have now.

    Args:
        graph: the Graph (its current frameRange and its Read nodes).
        infoOf: function giving the ImageInfo of a node (or None if it cannot be read).
        lastSourceUid: uid of the followed Read when its range was last read (the range is only read
                       again when the Read changes).
        frameRange: the current frame range (default: the one of the graph; {}: as if it was not set).

    Returns:
        (the new frame range or None if it does not change, the uid of the followed Read).
    """
    from meshroom.core.graph import Graph
    from .evaluator import isImageOpNode
    frameRange = graph.frameRange if frameRange is None else frameRange
    if frameRange and frameRange.get("source", "") == "":
        return None, lastSourceUid  # Entered by the user
    sourceName = frameRange.get("source") if frameRange else None
    source = graph.node(sourceName) if sourceName else None
    if source is None:
        if frameRange:
            return None, lastSourceUid  # The followed Read is gone: the range stays
        reads = sorted((node for node in graph.nodes if node.nodeType == "Read" and isImageOpNode(node)),
                       key=lambda node: Graph.getNodeIndexFromName(node.name))
        if not reads:
            return None, lastSourceUid
        source = reads[0]
    if frameRange and source._uid == lastSourceUid:
        return None, lastSourceUid
    info = infoOf(source)
    value = {"source": source.name}
    if info is not None and info.hasFrameRange:
        value.update({"first": int(info.firstFrame), "last": int(info.lastFrame)})
    elif frameRange.get("first") is not None:
        # The Read has no range (yet): the previous one stays
        value.update({"first": frameRange["first"], "last": frameRange["last"]})
    return (value if value != frameRange else None), source._uid
