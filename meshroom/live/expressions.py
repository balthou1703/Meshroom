"""
Python expressions of the compositing nodes (e.g. the Switch "Expression"): a small Python expression
reading the frame and the parameters of other nodes, such as

    1 if node('BG').file else 0
    nuke.toNode('BG')['file'].value() and 1 or 0      (the Python of a Nuke script works too)

The parameters of the nodes it reads are snapshotted with the plan of the branch (see snapshot): the
render threads do not read the graph, and the cached images change when those parameters change.

Names available in an expression:
- frame: the rendered frame;
- node(name): the parameters of a node, by its label or name (node('BG').file, node('BG')['file']);
  an animated parameter gives its value at the frame;
- nuke: a subset of the Nuke Python API (nuke.toNode(name)['knob'].value(), nuke.frame());
- math, and the builtins abs, min, max, int, float, round, len, bool, str.
The expression can span several lines: it then sets a variable named "result".
"""
import builtins
import math
from typing import Dict, Optional

_SAFE_BUILTINS = {name: getattr(builtins, name)
                  for name in ("abs", "min", "max", "int", "float", "round", "len", "bool", "str", "True", "False",
                               "None", "any", "all", "sum", "range")}


class ExpressionError(RuntimeError):
    pass


def _compile(text: str):
    """ The code of an expression: a single expression, or statements setting "result". """
    if "__" in text:
        raise ExpressionError("Names with '__' are not allowed in an expression.")
    try:
        return compile(text, "<expression>", "eval"), True
    except SyntaxError:
        try:
            return compile(text, "<expression>", "exec"), False
        except SyntaxError as e:
            raise ExpressionError(f"Syntax error in the expression: {e.msg} (line {e.lineno}).") from e


class NodeValues:
    """ The parameters of a node seen by an expression, at a frame. """

    def __init__(self, name: str, data: dict, frame):
        self._name = name
        self._data = data
        self._frame = frame

    def _value(self, attribute: str):
        values = self._data.get("values", {})
        if attribute not in values:
            raise ExpressionError(f"Node '{self._name}' has no parameter '{attribute}' "
                                  f"(parameters: {', '.join(sorted(values))}).")
        keys = self._data.get("animation", {}).get(attribute)
        if keys and self._frame is not None:
            from .animation import valueAt
            return valueAt(keys, self._frame)
        return values[attribute]

    def __getattr__(self, attribute):
        if attribute.startswith("_"):
            raise AttributeError(attribute)
        return self._value(attribute)

    def __getitem__(self, attribute):
        return self._value(attribute)


class _NukeKnob:
    def __init__(self, value):
        self._knobValue = value

    def value(self, *args):
        return self._knobValue

    getValue = value
    evaluate = value

    def __bool__(self):
        return bool(self._knobValue)


class _NukeNode:
    def __init__(self, values: NodeValues):
        self._values = values

    def __getitem__(self, knob):
        return _NukeKnob(self._values[_NUKE_KNOBS.get(knob, knob)])

    def knob(self, knob):
        return self[knob]


# Nuke knobs with another parameter name
_NUKE_KNOBS = {"which": "which", "mix": "mix", "white": "gain", "black": "lift", "add": "offset"}


class _NukeModule:
    def __init__(self, lookup, frame):
        self._lookup = lookup
        self._frame = frame

    def toNode(self, name):
        """ Like Nuke: None for an unknown node. """
        try:
            return _NukeNode(self._lookup(name))
        except ExpressionError:
            return None

    def frame(self):
        return self._frame


def evaluate(text: str, frame, lookup):
    """
    Evaluate an expression.

    Args:
        text: the expression (or statements setting "result").
        frame: the frame (None when unknown).
        lookup: function(name) -> NodeValues of a node (raises ExpressionError for an unknown node).
    """
    code, isExpression = _compile(text)
    scope = {"__builtins__": _SAFE_BUILTINS, "frame": frame, "node": lookup, "math": math,
             "nuke": _NukeModule(lookup, frame)}
    try:
        if isExpression:
            return eval(code, scope)
        exec(code, scope)
    except ExpressionError:
        raise
    except Exception as e:
        raise ExpressionError(f"Expression error: {type(e).__name__}: {e}") from e
    if "result" not in scope:
        raise ExpressionError("The expression has several lines: it must set a variable named 'result'.")
    return scope["result"]


def _nodeLabel(node) -> str:
    if node.hasInternalAttribute("label"):
        return node.internalAttribute("label").value or ""
    return ""


def _findNode(graph, name: str):
    """ The node of the graph with that label (first) or name. """
    for node in graph.nodes:
        if _nodeLabel(node) == name:
            return node
    return graph.node(name)


def _nodeData(node) -> dict:
    """ Snapshot of the parameters (and keys) of a live node. """
    from .evaluator import _nodeAnimation, _snapshotValue
    from meshroom.core import desc
    values = {}
    for attr in node.attributes:
        if attr.isOutput or isinstance(attr.desc, desc.Stream):
            continue
        try:
            values[attr.name] = _snapshotValue(attr)
        except Exception:
            continue
    if node.hasInternalAttribute("disable"):
        values["disable"] = bool(node.internalAttribute("disable").value)
    try:
        animation = _nodeAnimation(node)
    except Exception:
        animation = {}
    return {"values": values, "animation": animation}


def snapshot(graph, text: str, frame=None) -> Dict[str, dict]:
    """
    Snapshot of the nodes an expression reads ({name: data}), found by evaluating it on the graph
    (on the thread owning it). The nodes read only in a branch the evaluation did not take are not in it.
    """
    nodes: Dict[str, dict] = {}

    def lookup(name):
        if name not in nodes:
            node = _findNode(graph, name) if graph is not None else None
            if node is None:
                raise ExpressionError(f"No node named '{name}'.")
            nodes[name] = _nodeData(node)
        return NodeValues(name, nodes[name], frame)

    try:
        evaluate(text, frame, lookup)
    except ExpressionError:
        pass  # Reported by the render
    return nodes


def snapshotLookup(nodes: Dict[str, dict], frame):
    """ A lookup function reading a snapshot (see snapshot). """
    def lookup(name):
        data: Optional[dict] = nodes.get(name)
        if data is None:
            raise ExpressionError(f"No node named '{name}' (or it was not read when the branch was prepared).")
        return NodeValues(name, data, frame)
    return lookup


def nukePythonExpression(text: str) -> Optional[str]:
    """
    The Python code of a Nuke "[python ...]" TCL expression (escapes removed), or None if it is not one.
    e.g. '"\\[python 1\\\\ if\\\\ nuke.toNode(\\'BG\\')\\\\\\[\\'file\\'\\\\].value()\\\\ else\\\\ 0]"'
    -> "1 if nuke.toNode('BG')['file'].value() else 0".
    """
    import re
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = re.sub(r"\\(.)", r"\1", text[1:-1])
    match = re.fullmatch(r"\[python\s+(?:-exec\s+)?(.*)\]", text.strip(), re.DOTALL)
    if match is None:
        return None
    code = re.sub(r"\\(.)", r"\1", match.group(1)).strip()
    if len(code) >= 2 and code[0] == "{" and code[-1] == "}":
        code = code[1:-1].strip()
    return code
