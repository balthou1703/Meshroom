"""
Import of Nuke scripts (.nk) into a compositing graph.

A Nuke script is a list of TCL-like commands: each node is written as `Class { knob value ... }` and
takes its inputs from a stack ("inputs N": the N last entries of the stack, the top one being
input 0; for a Merge2, input 0 is B and input 1 is A; "inputs 1+1": the image then the mask), then is
pushed on it. "push 0" pushes an empty
input, "set VAR [stack 0]" names the top of the stack and "push $VAR" pushes it again; "clone $VAR {...}"
makes a clone of a named node. Groups ("Group {...}" ... "end_group") have their own stack.

The known classes become the equivalent compositing nodes (their knobs converted: names, units,
animation curves as keys, simple links to the knob of another node as attribute connections); the
others become "NukeUnsupported" nodes, passing their first input through and keeping their knobs, so
that the script keeps its structure. Positions are transposed (Nuke flows down, Meshroom to the right).
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from .animation import dumpAnimation, valueAt


# Tokenizer

class Curve:
    """ An animated value: keys [[frame, value], ...]. """

    def __init__(self, keys):
        self.keys = keys

    def at(self, frame):
        return valueAt(self.keys, frame)

    def __repr__(self):
        return f"Curve({self.keys})"


class Expression:
    """ An expression ({{...}}), e.g. a link to the knob of another node ("Blur1.size"). """

    def __init__(self, text):
        self.text = text.strip()

    def __repr__(self):
        return f"Expression({self.text!r})"


def tokenize(text: str) -> List[Tuple[str, str]]:
    """
    Tokens of a Nuke script: ("word", s), ("brace", inner text), ("quote", s), ("bracket", inner text)
    and ("nl", "") at the end of the lines (outside braces).
    """
    tokens = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\n":
            tokens.append(("nl", ""))
            i += 1
        elif c in " \t\r;":
            i += 1
        elif c == "#" and (not tokens or tokens[-1][0] == "nl"):
            # Comment line
            while i < n and text[i] != "\n":
                i += 1
        elif c in "{[":
            close = "}" if c == "{" else "]"
            depth, j = 0, i
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == c:
                    depth += 1
                elif text[j] == close:
                    depth -= 1
                    if depth == 0:
                        break
                elif text[j] == '"' and c == "{":
                    # Quoted strings inside braces may hold braces
                    j += 1
                    while j < n and text[j] != '"':
                        j += 2 if text[j] == "\\" else 1
                j += 1
            tokens.append(("brace" if c == "{" else "bracket", text[i + 1:j]))
            i = j + 1
        elif c == '"':
            j, chars = i + 1, []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    nxt = text[j + 1]
                    chars.append({"n": "\n", "t": "\t"}.get(nxt, nxt))
                    j += 2
                    continue
                chars.append(text[j])
                j += 1
            tokens.append(("quote", "".join(chars)))
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in " \t\r\n;{}[]\"":
                j += 1
            tokens.append(("word", text[i:j]))
            i = j
    return tokens


def _lines(tokens) -> List[List[Tuple[str, str]]]:
    """ Tokens grouped by line (non-empty lines). """
    lines, current = [], []
    for token in tokens:
        if token[0] == "nl":
            if current:
                lines.append(current)
            current = []
        else:
            current.append(token)
    if current:
        lines.append(current)
    return lines


def _number(word: str):
    try:
        value = float(word)
    except ValueError:
        return None
    return int(value) if re.fullmatch(r"[-+]?\d+", word) else value


def parseCurve(text: str) -> List[list]:
    """ Keys of a Nuke curve ("curve x1001 0.1 0.2 x1010 0.3", interpolation flags ignored). """
    keys = []
    frame = None
    for kind, token in tokenize(text):
        if kind != "word" or token == "curve":
            continue  # Tangents ({...}) and the like
        if re.fullmatch(r"x[-+]?\d+(\.\d*)?", token):
            frame = float(token[1:])
            continue
        number = _number(token)
        if number is None:
            continue  # Interpolation flags (K, L, C, s0, t1...)
        keys.append([1.0 if frame is None else frame, float(number)])
        frame = (1.0 if frame is None else frame) + 1.0
    return keys


def parseValue(tokens: List[Tuple[str, str]]):
    """
    The value of a knob from its tokens: a number, a bool, a string, a list (several values or a
    braced list), a Curve, an Expression, or None.
    """
    values = [_parseOne(token) for token in tokens]
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def _parseOne(token):
    kind, text = token
    if kind == "word":
        if text in ("true", "false"):
            return text == "true"
        number = _number(text)
        return number if number is not None else text
    if kind == "quote":
        return text
    if kind == "bracket":
        return Expression("[" + text + "]")
    # Braced: a curve, an expression, or a list
    inner = text.strip()
    if inner.startswith("curve"):
        return Curve(parseCurve(inner))
    if inner.startswith("{") and inner.endswith("}"):
        sub = inner[1:-1].strip()
        if sub.startswith("curve"):
            return Curve(parseCurve(sub))
        if _isSingleBraced(inner):
            return Expression(sub)
    elements = [token for token in tokenize(inner) if token[0] != "nl"]
    return [_parseOne(element) for element in elements]


def _isSingleBraced(text: str) -> bool:
    """ Whether a text is one braced group ("{...}") as a whole. """
    depth = 0
    for index, c in enumerate(text):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
    return True


def parseKnobs(body: str) -> Dict[str, object]:
    """ The knobs of a node body: {name: value}; "addUserKnob" and the like are skipped. """
    knobs = {}
    for line in _lines(tokenize(body)):
        kind, name = line[0]
        if kind not in ("word", "quote") or name.startswith("addUserKnob") or name in ("knobChanged",):
            continue
        knobs[name] = parseValue(line[1:])
    return knobs


# Scripts

class NukeNode:
    """ A node of a Nuke script: class, knobs, inputs (NukeNode or None), and the raw text of its knobs. """

    def __init__(self, nodeClass: str, knobs: dict, body: str, inputs: List[Optional["NukeNode"]], cloneOf=None):
        self.nodeClass = nodeClass
        self.knobs = knobs
        self.body = body
        self.inputs = inputs
        self.cloneOf = cloneOf
        self.children: List["NukeNode"] = []

    @property
    def name(self) -> str:
        return str(self.knobs.get("name", self.nodeClass))

    def __repr__(self):
        return f"NukeNode({self.nodeClass} {self.name})"


# Classes without input by default
_SOURCE_CLASSES = {"Read", "Constant", "CheckerBoard2", "ColorWheel", "ColorBars", "Root", "BackdropNode",
                   "StickyNote", "Camera", "Camera2", "Camera3", "Input", "DeepRead", "ReadGeo2"}


def _inputCount(nodeClass: str, knobs: dict) -> int:
    value = knobs.get("inputs")
    if value is None:
        return 0 if nodeClass in _SOURCE_CLASSES else 1
    try:
        return sum(int(part) for part in str(value).split("+"))
    except ValueError:
        return 1


def parseScript(text: str) -> Tuple[List[NukeNode], dict]:
    """
    The nodes of a Nuke script (top level; the content of the groups in their `children`), and the
    knobs of its Root node.
    """
    lines = _lines(tokenize(text))
    root = {}
    variables: Dict[str, Optional[NukeNode]] = {}
    # One (stack, node list) per group level
    levels = [([], [])]
    for line in lines:
        stack, nodes = levels[-1]
        kind, word = line[0]
        if kind != "word":
            continue
        if word == "push" and len(line) > 1:
            target = line[1][1]
            stack.append(None if target == "0" else variables.get(target.lstrip("$")))
            continue
        if word == "set" and len(line) > 1:
            variables[line[1][1]] = stack[-1] if stack else None
            continue
        if word == "end_group":
            if len(levels) > 1:
                levels.pop()
            continue
        if word in ("version", "define_window_layout_xml", "add_layer"):
            continue
        cloneOf = None
        if word == "clone" and len(line) >= 3:
            if line[1][1].startswith("$"):
                # A clone of a named node
                cloneOf = variables.get(line[1][1][1:])
                if cloneOf is None or line[2][0] != "brace":
                    continue
                nodeClass, body = cloneOf.nodeClass, line[2][1]
            elif len(line) >= 4 and line[3][0] == "brace":
                # The first instance of a cloned node: "clone node7f...|Blur|21 Blur {...}"
                nodeClass, body = line[2][1], line[3][1]
            else:
                continue
        elif len(line) >= 2 and line[1][0] == "brace":
            nodeClass, body = word, line[1][1]
        else:
            continue
        knobs = parseKnobs(body)
        if nodeClass == "Root":
            root = knobs
            continue
        if cloneOf is not None:
            knobs = dict(cloneOf.knobs, **{k: v for k, v in knobs.items() if k in ("xpos", "ypos", "inputs")})
            knobs["name"] = f"{cloneOf.name}_clone{sum(1 for node in nodes if node.cloneOf is cloneOf) + 1}"
        count = _inputCount(nodeClass, knobs)
        # The top of the stack (the last entry) is input 0, the entry below it input 1...
        inputs = stack[-count:][::-1] if count else []
        if count:
            del stack[-count:]
        inputs = list(inputs) + [None] * (count - len(inputs))
        node = NukeNode(nodeClass, knobs, body, inputs, cloneOf)
        nodes.append(node)
        stack.append(node)
        if nodeClass in ("Group", "LiveGroup"):
            levels.append(([], node.children))
    return levels[0][1], root


# Conversion of the knobs

def _first(value, index=0, default=None):
    """ The element `index` of a list value, or the value itself. """
    if isinstance(value, list):
        return value[index] if index < len(value) else (value[-1] if value else default)
    return value if value is not None else default


def _isAnimated(value) -> bool:
    return isinstance(value, Curve) or (isinstance(value, list) and any(isinstance(v, Curve) for v in value))


def _frames(values) -> List[float]:
    frames = set()
    for value in values:
        if isinstance(value, Curve):
            frames.update(frame for frame, _ in value.keys)
    return sorted(frames)


def _at(value, frame, default=0.0):
    if isinstance(value, Curve):
        return value.at(frame) if value.keys else default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


class _Converter:
    """ Values (and keys) of the parameters of a node converted from the knobs of a Nuke node. """

    def __init__(self, nukeNode: NukeNode, report: List[str]):
        self.node = nukeNode
        self.knobs = nukeNode.knobs
        self.values: Dict[str, object] = {}
        self.animation: Dict[str, List[list]] = {}
        # Knob -> parameter (for the links between knobs)
        self.knobParams: Dict[str, str] = {}
        self.expressions: List[Tuple[str, Expression]] = []
        # Image inputs to connect to another node of the script: {input name: Nuke node name}
        self.inputLinks: Dict[str, str] = {}
        self.report = report

    def has(self, knob):
        return knob in self.knobs

    def raw(self, knob, default=None):
        return self.knobs.get(knob, default)

    def _expression(self, param, knob):
        value = self.knobs.get(knob)
        if isinstance(value, Expression):
            self.expressions.append((param, value))
            return True
        return False

    def scalar(self, param, knob, transform=float, reduce="first", default=None):
        """ A number (an animated one: keys); a list knob gives its first element (or the mean of r, g, b). """
        if knob not in self.knobs:
            return
        self.knobParams[knob] = param
        if self._expression(param, knob):
            return
        value = self.knobs[knob]
        if isinstance(value, list):
            if reduce == "mean3" and len(value) >= 3:
                parts = value[:3]
                if _isAnimated(parts):
                    frames = _frames(parts)
                    self.animation[param] = [[f, transform(sum(_at(p, f) for p in parts) / 3.0)] for f in frames]
                    self.values[param] = self.animation[param][0][1]
                    return
                numbers = [float(p) for p in parts if isinstance(p, (int, float))]
                value = sum(numbers) / len(numbers) if numbers else default
            else:
                value = _first(value)
        if isinstance(value, Curve):
            if value.keys:
                self.animation[param] = [[f, transform(v)] for f, v in value.keys]
                self.values[param] = transform(value.keys[0][1])
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self.values[param] = transform(value)
        elif isinstance(value, bool):
            self.values[param] = transform(float(value))

    def boolean(self, param, knob, invert=False):
        if knob in self.knobs:
            self.knobParams[knob] = param
            value = self.knobs[knob]
            result = value if isinstance(value, bool) else str(value).lower() in ("true", "1")
            self.values[param] = (not result) if invert else result

    def choice(self, param, knob, mapping: dict, default=None):
        if knob in self.knobs:
            self.knobParams[knob] = param
            value = str(_first(self.knobs[knob]))
            key = value.lower()
            if key in mapping:
                self.values[param] = mapping[key]
            elif default is not None:
                self.values[param] = default
                self.report.append(f"{self.node.name}: {knob} '{value}' is not supported ('{default}' used).")

    def string(self, param, knob, transform=str):
        if knob in self.knobs and not isinstance(self.knobs[knob], (Expression, Curve)):
            self.knobParams[knob] = param
            self.values[param] = transform(str(_first(self.knobs[knob])) if not isinstance(self.knobs[knob], str)
                                           else self.knobs[knob])

    def components(self, param, knob, names, defaults, single=None):
        """
        A group parameter ({x, y} or {r, g, b, a}) from a list knob (a single number gives every component,
        or only those of `single`); animated components give keys of the whole group.
        """
        if knob not in self.knobs:
            return
        self.knobParams[knob] = param
        if self._expression(param, knob):
            return
        value = self.knobs[knob]
        parts = value if isinstance(value, list) else None
        if parts is None:
            singleNames = single or names
            parts = [value if name in singleNames else default for name, default in zip(names, defaults)]
        parts = list(parts) + list(defaults[len(parts):])
        parts = parts[:len(names)]
        if _isAnimated(parts):
            frames = _frames(parts)
            self.animation[param] = [[f, {n: _at(p, f, d) for n, p, d in zip(names, parts, defaults)}] for f in frames]
            self.values[param] = self.animation[param][0][1]
        else:
            self.values[param] = {n: (float(p) if isinstance(p, (int, float)) and not isinstance(p, bool) else d)
                                  for n, p, d in zip(names, parts, defaults)}

    def vec2(self, param, knob, default=(0.0, 0.0)):
        self.components(param, knob, ("x", "y"), default)

    def color(self, param, knob, default=(0.0, 0.0, 0.0, 1.0), singleAll=True):
        """ An RGBA color; a single value gives r, g, b (and a if `singleAll`). """
        self.components(param, knob, ("r", "g", "b", "a"), default, None if singleAll else ("r", "g", "b"))

    def mix(self):
        self.scalar("mix", "mix")

    def channels(self, param="channels", knob="channels", values=("rgb", "rgba", "alpha")):
        if knob in self.knobs:
            value = str(_first(self.knobs[knob])).lower()
            self.values[param] = value if value in values else ("rgba" if value == "all" and "rgba" in values else values[0])

    def maskChannel(self):
        if "maskChannelInput" in self.knobs or "maskChannelMask" in self.knobs:
            value = str(_first(self.knobs.get("maskChannelInput") or self.knobs.get("maskChannelMask")))
            channel = value.split(".")[-1]
            self.values["maskChannel"] = {"red": "red", "green": "green", "blue": "blue", "alpha": "alpha"}.get(channel, "alpha")
        self.boolean("invertMask", "invert_mask")


def _format(value, root) -> Optional[Tuple[int, int]]:
    """ (width, height) of a format knob ("1920 1080 0 0 1920 1080 1 HD_1080"), or of the root format. """
    for candidate in (value, root.get("format")):
        if candidate is None:
            continue
        words = str(candidate).split() if not isinstance(candidate, list) else [str(v) for v in candidate]
        numbers = [_number(w) for w in words[:2]]
        if len(numbers) == 2 and all(isinstance(n, int) for n in numbers):
            return numbers[0], numbers[1]
    return None


def _setFormat(c: _Converter, root, knob="format"):
    size = _format(c.raw(knob), root)
    if size is not None:
        c.values["width"], c.values["height"] = size


def _sequencePath(path: str) -> str:
    """ A Nuke sequence path in the "####" notation ("%04d" -> "####"). """
    return re.sub(r"%0?(\d*)d", lambda m: "#" * int(m.group(1) or 1), path)


def _colorspace(value, c=None, knob="colorspace") -> str:
    """ The colorspace of a Nuke colorspace knob: a name of the color config, or "auto" for "default (...)". """
    from . import color
    text = str(value or "").strip()
    if not text or text.lower().startswith("default"):
        return "auto"
    found = color.config().resolve(text)
    if found is None and " - " in text:
        # OCIO family prefix of the Nuke menus ("ACES - ACEScg", "Utility - sRGB - Texture")
        found = color.config().resolve(text.split(" - ", 1)[1])
    if found is not None:
        return found
    lower = text.lower()
    fallback = "sRGB" if "srgb" in lower and "linear" not in lower else "linear" if "linear" in lower else "auto"
    if c is not None:
        c.report.append(f"{c.node.name}: {knob} '{text}' is not in the color config ('{fallback}' used).")
    return fallback


_MERGE_OPERATIONS = {
    "over": "over", "under": "under", "plus": "plus", "minus": "minus", "from": "from", "multiply": "multiply",
    "screen": "screen", "max": "max", "min": "min", "in": "in", "out": "out", "mask": "mask", "stencil": "stencil",
    "atop": "atop", "xor": "xor", "difference": "difference", "average": "average", "copy": "copy",
}


def _convertRead(c, root):
    c.string("file", "file", _sequencePath)
    c.values["colorspace"] = _colorspace(c.raw("colorspace"), c)
    if c.raw("frame_mode") in ("offset", "start at") and isinstance(c.raw("frame"), (int, float)):
        offset = int(c.raw("frame"))
        if c.raw("frame_mode") == "start at" and isinstance(c.raw("first"), (int, float)):
            offset = offset - int(c.raw("first"))
        c.values["frameOffset"] = offset
    c.choice("outsideRange", "before", {"hold": "hold", "black": "black", "loop": "hold", "bounce": "hold"}, "hold")


def _convertWrite(c, root):
    c.string("file", "file", _sequencePath)
    c.values["colorspace"] = _colorspace(c.raw("colorspace"), c)
    datatype = str(c.raw("datatype", "")).lower()
    if "32" in datatype or "float" in datatype:
        c.values["dataType"] = "float"
    elif "half" in datatype or "16" in datatype:
        c.values["dataType"] = "half"
    if c.raw("use_limit") is True:
        c.values["frameRangeMode"] = "custom"
        c.scalar("firstFrame", "first", int)
        c.scalar("lastFrame", "last", int)


def _convertMerge(c, root):
    c.choice("operation", "operation", _MERGE_OPERATIONS, "over")
    c.choice("bbox", "bbox", {"union": "union", "b": "B", "a": "A", "intersection": "intersection"}, "union")
    c.mix()
    c.maskChannel()


def _convertGrade(c, root):
    for param, knob in (("blackpoint", "blackpoint"), ("whitepoint", "whitepoint"), ("lift", "black"),
                        ("gain", "white"), ("multiply", "multiply"), ("offset", "add"), ("gamma", "gamma")):
        default = 1.0 if param in ("whitepoint", "gain", "multiply", "gamma") else 0.0
        c.components(param, knob, ("r", "g", "b", "a"), (default,) * 4)
    c.boolean("blackClamp", "black_clamp")
    c.boolean("whiteClamp", "white_clamp")
    c.channels(values=("rgb", "rgba", "alpha", "all"))
    c.mix()
    c.maskChannel()


def _convertColorCorrect(c, root):
    c.scalar("saturation", "saturation", reduce="mean3")
    for name, default in (("contrast", 1.0), ("gamma", 1.0), ("gain", 1.0), ("offset", 0.0)):
        c.components(name, name, ("r", "g", "b", "a"), (default,) * 4)
    for zone in ("shadows", "midtones", "highlights"):
        if any(knob.startswith(zone + ".") for knob in c.knobs):
            c.report.append(f"{c.node.name}: the {zone} corrections are not imported (master only).")
    c.channels()
    c.mix()
    c.maskChannel()


def _convertBlur(c, root):
    c.vec2("size", "size")
    c.channels(values=("rgba", "all"))
    c.mix()
    c.maskChannel()


def _convertTransform(c, root):
    c.vec2("translate", "translate")
    c.scalar("rotate", "rotate")
    c.vec2("scale", "scale", (1.0, 1.0))
    c.scalar("skewX", "skewX")
    c.scalar("skewX", "skew")
    if c.has("center"):
        c.values["centerOnImage"] = False
        c.vec2("center", "center")
    c.boolean("invert", "invert_matrix")
    c.choice("filter", "filter", {"impulse": "impulse"}, "bilinear")


def _convertConstant(c, root):
    c.color("rgba", "color", (0.0, 0.0, 0.0, 0.0))
    _setFormat(c, root)


def _convertCrop(c, root):
    box = c.raw("box")
    if isinstance(box, list) and len(box) == 4:
        for name, value in zip(("left", "bottom", "right", "top"), box):
            if isinstance(value, (int, float)):
                c.values[name] = float(value)
            elif isinstance(value, Curve) and value.keys:
                c.values[name] = value.keys[0][1]
                c.animation[name] = value.keys
    c.scalar("softness", "softness")
    c.boolean("reformat", "reformat")


def _convertReformat(c, root):
    kind = str(c.raw("type", "to format"))
    if kind == "scale":
        c.values["reformatType"] = "scale"
        c.vec2("scale", "scale", (1.0, 1.0))
    else:
        c.values["reformatType"] = "to format"
        size = _format(c.raw("format"), root)
        if kind == "to box" and isinstance(c.raw("box_width"), (int, float)):
            size = (int(c.raw("box_width")), int(c.raw("box_height", c.raw("box_width"))))
        elif kind == "to box" and isinstance(c.raw("box_width"), Expression):
            # {{SRC.width}} / {{SRC.height}}: the format of another node, through the "Format" input
            widthOf = re.fullmatch(r"(\w+)\.width", c.raw("box_width").text)
            heightOf = re.fullmatch(r"(\w+)\.height", getattr(c.raw("box_height"), "text", ""))
            if widthOf and (heightOf is None or heightOf.group(1) == widthOf.group(1)):
                c.inputLinks["formatInput"] = widthOf.group(1)
            else:
                c.report.append(f"{c.node.name}: the box size expression is not imported.")
        if size:
            c.values["width"], c.values["height"] = size
    c.choice("resize", "resize", {"none": "fit", "fit": "fit", "fill": "fill", "width": "fit", "height": "fit",
                                  "distort": "distort"}, "fit")
    c.boolean("crop", "black_outside")


def _convertShuffle(c, root):
    c.string("inputLayer", "in")
    for param in ("red", "green", "blue", "alpha"):
        value = str(c.raw(param, param))
        if value in ("red", "green", "blue", "alpha", "black", "white"):
            c.values[param] = value
        elif value in ("r", "g", "b", "a"):
            c.values[param] = {"r": "red", "g": "green", "b": "blue", "a": "alpha"}[value]
        elif value in ("0",):
            c.values[param] = "black"
        elif value in ("1",):
            c.values[param] = "white"


def _convertCopy(c, root):
    for param, knob in (("fromChannel", "from0"), ("toChannel", "to0")):
        value = str(c.raw(knob, "rgba.alpha")).split(".")[-1]
        if value in ("red", "green", "blue", "alpha"):
            c.values[param] = value


def _convertMirror(c, root):
    c.boolean("horizontal", "flop" if c.node.nodeClass == "Mirror2" else "Horizontal")
    c.boolean("vertical", "flip" if c.node.nodeClass == "Mirror2" else "Vertical")


def _convertSimple(params):
    """ A converter of knobs with the same meaning: [(param, knob, kind)]. """
    def convert(c, root):
        for param, knob, kind in params:
            if kind == "float":
                c.scalar(param, knob, reduce="mean3")
            elif kind == "int":
                c.scalar(param, knob, lambda v: int(round(v)))
            elif kind == "bool":
                c.boolean(param, knob)
            elif kind == "vec2":
                c.vec2(param, knob)
            elif kind == "color":
                c.color(param, knob)
        if "mix" in c.knobs:
            c.mix()
        c.maskChannel()
        if "channels" in c.knobs:
            c.channels()
    return convert


def _convertColorspaceNode(inKnob, outKnob):
    """ OCIOColorSpace (in_colorspace / out_colorspace) and Colorspace (colorspace_in / colorspace_out). """
    def convert(c, root):
        c.values["inColorspace"] = _resolvedOrWorking(c, inKnob)
        c.values["outColorspace"] = _resolvedOrWorking(c, outKnob)
        _convertSimple([])(c, root)
    return convert


def _resolvedOrWorking(c, knob) -> str:
    """ The colorspace of a knob of a colorspace node: "auto" (default) is the working space. """
    from . import color
    name = _colorspace(c.raw(knob), c, knob)
    return color.config().workingSpace() if name == "auto" else name


def _convertSwitch(c, root):
    """ Switch: "which" (a number, keys, or a [python ...] expression, kept as a Switch expression). """
    from .expressions import nukePythonExpression
    value = c.raw("which")
    if isinstance(value, Expression):
        code = nukePythonExpression(value.text)
        if code is None and re.fullmatch(r"[\w\s.+\-*/%<>=!()&|?:]+", value.text):
            # A simple TCL expression (e.g. "frame>1010"): the same in Python
            code = value.text.replace("&&", " and ").replace("||", " or ")
        if code is None:
            c.report.append(f"{c.node.name}: the expression of 'which' ({value.text}) is not imported.")
            return
        c.values["useExpression"] = True
        c.values["expression"] = code
        return
    c.scalar("which", "which", lambda v: int(round(v)))


def _convertExposure(c, root):
    """ EXPTool: red / green / blue in stops (the mean is used). """
    stops = [c.raw(knob) for knob in ("red", "green", "blue")]
    numbers = [float(v) for v in stops if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if numbers:
        c.values["stops"] = sum(numbers) / len(numbers)
        if len(set(numbers)) > 1:
            c.report.append(f"{c.node.name}: exposure per channel (the mean is used).")
    if str(c.raw("mode", "Stops")).lower() not in ("stops", ""):
        c.report.append(f"{c.node.name}: exposure mode {c.raw('mode')} is read as stops.")
    c.mix()
    c.maskChannel()


def _convertRetime(c, root):
    for param, knob in (("inputFirst", "input.first"), ("inputLast", "input.last"),
                        ("outputFirst", "output.first"), ("outputLast", "output.last")):
        c.scalar(param, knob, lambda v: int(round(v)))
    for param, knob in (("useInputFirst", "input.first_lock"), ("useInputLast", "input.last_lock"),
                        ("useOutputFirst", "output.first_lock"), ("useOutputLast", "output.last_lock")):
        c.boolean(param, knob)
    c.scalar("speed", "speed")
    c.boolean("reverse", "reverse")
    c.choice("filter", "filter", {"none": "nearest", "nearest": "nearest", "box": "blend"}, "blend")


def _convertTimeWarp(c, root):
    lookup = c.raw("lookup")
    if isinstance(lookup, Curve):
        import json
        c.values["lookup"] = json.dumps({"time": [[float(f), float(v)] for f, v in lookup.keys]})
    c.choice("filter", "filter", {"none": "nearest", "nearest": "nearest", "box": "blend"}, "blend")


def _convertCheckerBoard(c, root):
    c.scalar("boxSize", "boxsize")
    c.color("color0", "color0", (0.1, 0.1, 0.1, 1.0))
    c.color("color1", "color1", (0.5, 0.5, 0.5, 1.0))
    c.color("lineColor", "centerlinecolor", (1.0, 1.0, 0.0, 1.0))
    c.scalar("lineWidth", "centerlinewidth")
    _setFormat(c, root)


def _convertRamp(c, root):
    c.vec2("point0", "p0")
    c.vec2("point1", "p1")
    c.color("color1", "color", (1.0, 1.0, 1.0, 1.0))
    c.choice("rampType", "type", {"linear": "linear", "smooth": "smooth", "smooth0": "smooth0", "smooth1": "smooth1"},
             "linear")


def _convertRadial(c, root):
    area = c.raw("area")
    c.values["shape"] = "circle" if c.node.nodeClass == "Radial" else "rectangle"
    if isinstance(area, list) and len(area) == 4 and all(isinstance(v, (int, float)) for v in area):
        left, bottom, right, top = (float(v) for v in area)
        c.values["centerOnImage"] = False
        c.values["center"] = {"x": (left + right) / 2.0, "y": (bottom + top) / 2.0}
        c.values["shapeSize"] = {"x": abs(right - left), "y": abs(top - bottom)}
        softness = c.raw("softness")
        if isinstance(softness, (int, float)):
            c.values["feather"] = float(softness) * min(abs(right - left), abs(top - bottom)) / 2.0
            c.values["featherMode"] = "inside"
    c.color("shapeColor", "color", (1.0, 1.0, 1.0, 1.0))
    c.boolean("invert", "invert")


def _convertCornerPin(c, root):
    for i in range(1, 5):
        c.vec2(f"to{i}", f"to{i}")
        c.vec2(f"from{i}", f"from{i}")
    c.boolean("invert", "invert")


def _convertKeyer(c, root):
    c.choice("operation", "operation", {"luminance key": "luminance", "red keyer": "red", "green keyer": "green",
                                        "blue keyer": "blue", "alpha keyer": "alpha", "saturation key": "saturation",
                                        "max keyer": "max"}, "luminance")
    keyRange = c.raw("range")
    if isinstance(keyRange, list) and len(keyRange) == 4:
        for name, value in zip("abcd", keyRange):
            if isinstance(value, (int, float)):
                c.values[name] = float(value)


def _convertKeylight(c, root):
    c.color("screenColor", "screenColour", (0.1, 0.6, 0.15, 1.0), singleAll=False)
    c.scalar("screenGain", "screenGain")
    c.scalar("screenBalance", "screenBalance")
    c.scalar("clipBlack", "clipBlack")
    c.scalar("clipWhite", "clipWhite")


def _convertZDefocus(c, root):
    c.choice("math", "math", {"depth": "depth", "far = 0": "far = 0", "far=0": "far = 0", "direct": "direct",
                              "linear": "linear", "far = 1": "far = 0"}, "depth")
    c.scalar("focalPlane", "focus_plane_C")
    c.scalar("focalPlane", "focal_point_C")
    c.scalar("depthOfField", "dof")
    c.scalar("size", "size")
    c.scalar("maxSize", "max_size")
    c.choice("shape", "filter_type", {"disc": "disc", "bladed": "blades", "image": "filter image"}, "disc")
    c.scalar("blades", "blades", lambda v: int(round(v)))
    c.scalar("rotation", "rotation")
    c.scalar("aspectRatio", "aspect")
    c.mix()


def _convertLens(c, root):
    from . import lens
    model = lens.NUKE_CLASSES.get(c.node.nodeClass)
    if model is None:
        return
    c.values["model"] = model
    names = {}
    for name, label, _ in lens.MODEL_PARAMETERS[model]:
        names[re.sub(r"[^a-z0-9]", "", label.lower())] = name
    for name, knob, _ in lens.CAMERA_PARAMETERS:
        names[re.sub(r"[^a-z0-9]", "", knob.lower())] = name
    for knob in list(c.knobs):
        param = names.get(re.sub(r"[^a-z0-9]", "", knob.lower()))
        if param is not None:
            c.scalar(param, knob)
    c.choice("direction", "direction", {"undistort": "undistort", "distort": "distort"}, "undistort")


def _lensClasses():
    from .lens import NUKE_CLASSES
    return list(NUKE_CLASSES)


# Nuke class -> (node type, converter, image inputs in the Nuke input order)
CLASSES = {
    "Read": ("Read", _convertRead, []),
    "Write": ("Write", _convertWrite, ["image"]),
    "Merge2": ("Merge", _convertMerge, ["B", "A", "mask"]),
    "Merge": ("Merge", _convertMerge, ["B", "A", "mask"]),
    "Grade": ("Grade", _convertGrade, ["input", "mask"]),
    "Switch": ("Switch", _convertSwitch, ["input", "input1", "input2", "input3"]),
    "OCIOColorSpace": ("OCIOColorSpace", _convertColorspaceNode("in_colorspace", "out_colorspace"), ["input", "mask"]),
    "Colorspace": ("OCIOColorSpace", _convertColorspaceNode("colorspace_in", "colorspace_out"), ["input", "mask"]),
    "ColorCorrect": ("ColorCorrect", _convertColorCorrect, ["input", "mask"]),
    "Saturation": ("Saturation", _convertSimple([("saturation", "saturation", "float")]), ["input", "mask"]),
    "Multiply": ("Multiply", _convertSimple([("value", "value", "float")]), ["input", "mask"]),
    "Add": ("Add", _convertSimple([("value", "value", "float")]), ["input", "mask"]),
    "Clamp": ("Clamp", _convertSimple([("minimum", "minimum", "float"), ("maximum", "maximum", "float"),
                                       ("clampMinimum", "minimum_enable", "bool"),
                                       ("clampMaximum", "maximum_enable", "bool")]), ["input", "mask"]),
    "Invert": ("Invert", _convertSimple([]), ["input", "mask"]),
    "Exposure": ("Exposure", lambda c, root: _convertExposure(c, root), ["input", "mask"]),
    "Blur": ("Blur", _convertBlur, ["input", "mask"]),
    "Sharpen": ("Sharpen", _convertSimple([("amount", "amount", "float"), ("size", "size", "float")]), ["input", "mask"]),
    "Median": ("Median", _convertSimple([("size", "size", "int")]), ["input", "mask"]),
    "FilterErode": ("Erode", _convertSimple([("size", "size", "float")]), ["input", "mask"]),
    "Erode": ("Erode", _convertSimple([("size", "size", "float")]), ["input", "mask"]),
    "Transform": ("Transform", _convertTransform, ["input"]),
    "TransformMasked": ("Transform", _convertTransform, ["input"]),
    "Constant": ("Constant", _convertConstant, []),
    "Crop": ("Crop", _convertCrop, ["input"]),
    "Reformat": ("Reformat", _convertReformat, ["input"]),
    "Shuffle": ("Shuffle", _convertShuffle, ["input"]),
    "Copy": ("Copy", _convertCopy, ["B", "A"]),
    "Premult": ("Premult", None, ["input"]),
    "Unpremult": ("Unpremult", None, ["input"]),
    "Dot": ("Dot", None, ["input"]),
    "Mirror": ("Mirror", _convertMirror, ["input"]),
    "Mirror2": ("Mirror", _convertMirror, ["input"]),
    "TimeOffset": ("TimeOffset", _convertSimple([("timeOffset", "time_offset", "int"), ("reverse", "reverse_input", "bool")]),
                   ["input"]),
    "FrameHold": ("FrameHold", _convertSimple([("firstFrame", "first_frame", "int"), ("increment", "increment", "int")]),
                  ["input"]),
    "Retime": ("Retime", _convertRetime, ["input"]),
    "TimeWarp": ("TimeWarp", _convertTimeWarp, ["input"]),
    "CheckerBoard2": ("CheckerBoard", _convertCheckerBoard, ["input"]),
    "ColorWheel": ("ColorWheel", lambda c, root: _setFormat(c, root), ["input"]),
    "Ramp": ("Ramp", _convertRamp, ["input"]),
    "Radial": ("Shape", _convertRadial, ["input"]),
    "Rectangle": ("Shape", _convertRadial, ["input"]),
    "CornerPin2D": ("CornerPin", _convertCornerPin, ["input"]),
    "Keyer": ("Keyer", _convertKeyer, ["input"]),
    "OFXuk.co.thefoundry.keylight.keylight_v201": ("Keylight", _convertKeylight, ["input", None, "core", "garbage"]),
    "VectorGenerator": ("OpticalFlow", None, ["input"]),
    "ZDefocus2": ("ZDefocus", _convertZDefocus, ["input", "filter", "mask"]),
    "Roto": ("Roto", None, ["input"]),
    "RotoPaint": ("Roto", None, ["input"]),
    "Tracker4": ("Tracker", None, ["input"]),
}
for _className in _lensClasses():
    CLASSES[_className] = ("LensDistortion", _convertLens, ["input"])

# Partially imported: what is left out
_PARTIAL = {
    "Roto": "its shapes are not imported (redraw them, or use the Roto of Meshroom)",
    "RotoPaint": "its shapes and strokes are not imported",
    "Tracker4": "its tracks are not imported",
}
# Not imported at all (not image nodes)
_SKIPPED = {"Viewer", "BackdropNode", "StickyNote", "Input", "Output"}


# Graph creation

def importScript(graph, text: str, root: Optional[dict] = None) -> Tuple[list, List[str], dict]:
    """
    Create the nodes of a Nuke script in `graph` (a core Graph): (the created nodes, a report of what
    was not (fully) imported, the knobs of the Root node).
    """
    nukeNodes, rootKnobs = parseScript(text)
    report: List[str] = []
    created: Dict[int, object] = {}
    converters: Dict[int, _Converter] = {}
    byName: Dict[str, NukeNode] = {}
    positions = {}
    xs = [n.knobs.get("xpos") for n in nukeNodes if isinstance(n.knobs.get("xpos"), (int, float))]
    ys = [n.knobs.get("ypos") for n in nukeNodes if isinstance(n.knobs.get("ypos"), (int, float))]
    minX, minY = (min(xs) if xs else 0), (min(ys) if ys else 0)

    def passThrough(node):
        """ The node a skipped node (Viewer...) stands for: its first input. """
        while node is not None and node.nodeClass in _SKIPPED:
            node = next((i for i in node.inputs if i is not None), None)
        return node

    for nukeNode in nukeNodes:
        if nukeNode.nodeClass in _SKIPPED:
            continue
        byName[nukeNode.name] = nukeNode
        if nukeNode.cloneOf is not None and id(nukeNode.cloneOf) in created:
            nodeType = "Clone"
        elif nukeNode.nodeClass in CLASSES:
            nodeType = CLASSES[nukeNode.nodeClass][0]
        else:
            nodeType = "NukeUnsupported"
        try:
            node = graph.addNewNode(nodeType)
        except Exception as e:  # pragma: no cover - a missing node type
            report.append(f"{nukeNode.name}: {nukeNode.nodeClass} could not be created ({e}).")
            continue
        created[id(nukeNode)] = node
        # Position: Nuke flows down, Meshroom to the right
        x, y = nukeNode.knobs.get("xpos", 0), nukeNode.knobs.get("ypos", 0)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            positions[id(nukeNode)] = ((y - minY) * 2.2, (x - minX) * 1.6)
        if node.hasInternalAttribute("label"):
            node.internalAttribute("label").value = nukeNode.name
        if nukeNode.knobs.get("disable") is True and node.hasInternalAttribute("disable"):
            node.internalAttribute("disable").value = True
        if nodeType == "NukeUnsupported":
            node.attribute("nukeClass").value = nukeNode.nodeClass
            node.attribute("knobs").value = nukeNode.body.strip("\n")
            report.append(f"{nukeNode.name}: {nukeNode.nodeClass} is not supported (NukeUnsupported, passes its input through).")
            continue
        if nodeType == "Clone":
            continue
        converter = _Converter(nukeNode, report)
        convert = CLASSES[nukeNode.nodeClass][1]
        if convert is not None:
            convert(converter, rootKnobs)
        converters[id(nukeNode)] = converter
        for name, value in converter.values.items():
            if not node.hasAttribute(name):
                continue
            try:
                node.attribute(name).value = value
            except Exception as e:  # pragma: no cover - an unexpected value
                report.append(f"{nukeNode.name}: {name} = {value!r} ignored ({e}).")
        animation = {name: keys for name, keys in converter.animation.items() if node.hasAttribute(name)}
        if animation and node.hasInternalAttribute("animation"):
            node.internalAttribute("animation").value = dumpAnimation(animation)
        if nukeNode.nodeClass in _PARTIAL:
            report.append(f"{nukeNode.name}: {_PARTIAL[nukeNode.nodeClass]}.")

    # Positions (the graph is laid out by the caller)
    from meshroom.core.node import Position
    for key, (px, py) in positions.items():
        graph.node(created[key].name).position = Position(int(px), int(py))

    # Connections
    for nukeNode in nukeNodes:
        node = created.get(id(nukeNode))
        if node is None:
            continue
        if node.nodeType == "Clone":
            source = created.get(id(nukeNode.cloneOf))
            if source is not None and source.hasAttribute("output"):
                graph.addEdge(source.attribute("output"), node.attribute("source"))
            continue
        if node.nodeType == "NukeUnsupported":
            inputNames = ["input", "input2", "input3", "input4"]
        else:
            inputNames = CLASSES[nukeNode.nodeClass][2]
        for index, upstream in enumerate(nukeNode.inputs):
            upstream = passThrough(upstream)
            if upstream is None or index >= len(inputNames) or inputNames[index] is None:
                if upstream is not None:
                    report.append(f"{nukeNode.name}: input {index} ({upstream.name}) is not connected.")
                continue
            src = created.get(id(upstream))
            if src is None or not src.hasAttribute("output") or not node.hasAttribute(inputNames[index]):
                continue
            try:
                graph.addEdge(src.attribute("output"), node.attribute(inputNames[index]))
            except Exception as e:  # pragma: no cover
                report.append(f"{nukeNode.name}: input {inputNames[index]} not connected ({e}).")

    # Image inputs linked to other nodes by the converters (e.g. the format of a Reformat)
    for key, converter in converters.items():
        node = created[key]
        for inputName, targetName in converter.inputLinks.items():
            target = passThrough(byName.get(targetName))
            src = created.get(id(target)) if target is not None else None
            if src is None or not src.hasAttribute("output") or not node.hasAttribute(inputName):
                report.append(f"{converter.node.name}: the link of {inputName} to {targetName} is not imported.")
                continue
            graph.addEdge(src.attribute("output"), node.attribute(inputName))

    # Links between knobs ({{Blur1.size}}): connections between the parameters
    for key, converter in converters.items():
        node = created[key]
        for param, expression in converter.expressions:
            match = re.fullmatch(r"(?:parent\.)?(\w+)\.(\w+)(?:\.\w+)?", expression.text)
            target = byName.get(match.group(1)) if match else None
            targetConverter = converters.get(id(target)) if target is not None else None
            targetParam = targetConverter.knobParams.get(match.group(2)) if targetConverter else None
            if targetParam is None:
                report.append(f"{converter.node.name}: the expression {{{{{expression.text}}}}} of {param} is not imported.")
                continue
            srcNode = created[id(target)]
            try:
                graph.addEdge(srcNode.attribute(targetParam), node.attribute(param))
            except Exception:
                report.append(f"{converter.node.name}: the link of {param} to {expression.text} is not imported.")

    for nukeNode in nukeNodes:
        if nukeNode.children:
            report.append(f"{nukeNode.name}: the content of the group ({len(nukeNode.children)} nodes) is not imported.")
    logging.debug("[Nuke import] %d nodes, %d remarks", len(created), len(report))
    return list(created.values()), report, rootKnobs


def rootFrameRange(rootKnobs: dict) -> Optional[Tuple[int, int]]:
    """ (first, last) frame of the Root of a script, or None. """
    first, last = rootKnobs.get("first_frame"), rootKnobs.get("last_frame")
    if isinstance(first, (int, float)) and isinstance(last, (int, float)):
        return int(first), int(last)
    return None
