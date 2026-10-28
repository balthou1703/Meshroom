"""
Insertion of a node on an edge of the graph editor (a node dropped on an edge with Ctrl held): the edge src -> dst
becomes src -> node input, node output -> dst.
"""
from typing import Optional, Tuple

# The attributes preferred when they fit: the main input and output of most nodes (compositing nodes, Write...)
_MAIN_INPUT = "input"
_MAIN_OUTPUT = "output"


def _usable(attribute) -> bool:
    """ A single attribute that can be connected (not a list or group, not disabled). """
    from meshroom.core.attribute import GroupAttribute, ListAttribute
    if isinstance(attribute, (ListAttribute, GroupAttribute)):
        return False
    return bool(getattr(attribute, "enabled", True))


def insertionAttributes(node, src, dst) -> Optional[Tuple[object, object]]:
    """
    (the input of `node` receiving `src`, the output of `node` going to `dst`) to insert `node` on the edge src -> dst,
    or None when it cannot be inserted there (it is one of the two nodes, the connection would make a cycle, or it has
    no compatible input and output). The input named "input" is preferred, else the first compatible one, a free one
    before a connected one; the output named "output", else the first compatible one.
    """
    if node is None or src is None or dst is None or node in (src.node, dst.node):
        return None
    graph = node.graph
    if graph is not None:
        downstream = graph.dfsOnDiscover(startNodes=[node], reverse=True, dependenciesOnly=True)[0]
        upstream = graph.dfsOnDiscover(startNodes=[node], dependenciesOnly=True)[0]
        # src would depend on the node and the node on src: a cycle
        if src.node in downstream or dst.node in upstream:
            return None

    def ordered(attributes, mainName):
        return sorted(attributes, key=lambda attribute: attribute.name != mainName)

    inputs = [attribute for attribute in ordered((a for a in node.attributes if a.isInput), _MAIN_INPUT)
              if _usable(attribute) and attribute.validateIncomingConnection(src)]
    # A free input before a connected one
    inputs.sort(key=lambda attribute: attribute.isLink)
    outputs = [attribute for attribute in ordered((a for a in node.attributes if a.isOutput), _MAIN_OUTPUT)
               if _usable(attribute) and dst.validateIncomingConnection(attribute)]
    if not inputs or not outputs:
        return None
    return inputs[0], outputs[0]
