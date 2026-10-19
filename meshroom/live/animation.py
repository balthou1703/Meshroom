"""
Animation of the parameters of compositing nodes: keys per frame, stored as JSON in the "animation"
internal attribute of the node:
    {"parameter name": [[frame, value], ...], ...}
The value of a key is the serialized value of the parameter (a number, a bool, a string, or a
dict for a group such as a 2D value or a color). Between two keys, numbers (and the numbers of a
group) are interpolated linearly; the other values hold the previous key. Before the first key and
after the last one, the value of that key is held.

An animated parameter takes its value from its keys at the rendered frame; its stored value only
mirrors the current frame of the viewer (it does not take part in the uid of the node, the keys do).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

ANIMATION = "animation"


def parseAnimation(text: str) -> Dict[str, List[list]]:
    """ The animation stored in a JSON string ({} if empty or invalid), keys sorted by frame. """
    if not text:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    result = {}
    for name, keys in data.items():
        valid = [key for key in keys or [] if isinstance(key, (list, tuple)) and len(key) == 2]
        if valid:
            result[name] = sorted(([float(frame), value] for frame, value in valid), key=lambda key: key[0])
    return result


def dumpAnimation(animation: Dict[str, List[list]]) -> str:
    """ JSON of an animation ("" when nothing is animated). """
    cleaned = {name: [[int(frame) if float(frame).is_integer() else frame, value] for frame, value in keys]
               for name, keys in animation.items() if keys}
    return json.dumps(cleaned, sort_keys=True) if cleaned else ""


def _lerp(a: Any, b: Any, t: float) -> Any:
    if isinstance(a, bool) or isinstance(b, bool):
        return a
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        value = a + (b - a) * t
        return int(round(value)) if isinstance(a, int) and isinstance(b, int) else value
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _lerp(v, b.get(k, v), t) for k, v in a.items()}
    return a


def valueAt(keys: List[list], frame: float) -> Any:
    """ The value of an animation curve at a frame. """
    if frame <= keys[0][0]:
        return keys[0][1]
    if frame >= keys[-1][0]:
        return keys[-1][1]
    for (f0, v0), (f1, v1) in zip(keys, keys[1:]):
        if f0 <= frame <= f1:
            return _lerp(v0, v1, (frame - f0) / (f1 - f0) if f1 > f0 else 0.0)
    return keys[-1][1]


def setKey(animation: Dict[str, List[list]], name: str, frame: float, value: Any) -> Dict[str, List[list]]:
    """ A copy of an animation with a key of `name` at `frame` (replacing the key of that frame). """
    result = {k: list(v) for k, v in animation.items()}
    keys = [key for key in result.get(name, []) if key[0] != frame]
    keys.append([float(frame), value])
    result[name] = sorted(keys, key=lambda key: key[0])
    return result


def removeKey(animation: Dict[str, List[list]], name: str, frame: float) -> Dict[str, List[list]]:
    """ A copy of an animation without the key of `name` at `frame` (the parameter is no longer animated without keys). """
    result = {k: list(v) for k, v in animation.items()}
    keys = [key for key in result.get(name, []) if key[0] != frame]
    if keys:
        result[name] = keys
    else:
        result.pop(name, None)
    return result


def animatedValues(values: Dict[str, Any], animation: Dict[str, List[list]], frame: float) -> Dict[str, Any]:
    """ The values of a node at a frame: the animated parameters from their keys. """
    if not animation:
        return values
    result = dict(values)
    for name, keys in animation.items():
        if name in result and keys:
            result[name] = valueAt(keys, frame)
    return result
