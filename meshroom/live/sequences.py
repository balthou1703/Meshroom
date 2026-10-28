"""
Listing of a folder where the frames of image sequences are grouped, as in the Nuke file browser.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List

IMAGE_EXTENSIONS = {
    ".exr", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dpx", ".cin", ".hdr", ".tga", ".bmp",
    ".psd", ".webp", ".sgi", ".rgb", ".iff", ".jp2", ".dds", ".pfm", ".ppm", ".pgm",
}

# A frame number: the last group of digits of the file name, just before the extension
_FRAME_NUMBER = re.compile(r"^(.*?)(\d+)(\.[^.]+)$")


def frameRanges(frames: List[int]) -> str:
    """ Compact description of sorted frames: "1001-1100", "1-10, 12, 14-20". """
    parts = []
    start = previous = None
    for frame in frames:
        if start is None:
            start = previous = frame
        elif frame == previous + 1:
            previous = frame
        else:
            parts.append(f"{start}-{previous}" if previous != start else f"{start}")
            start = previous = frame
    if start is not None:
        parts.append(f"{start}-{previous}" if previous != start else f"{start}")
    return ", ".join(parts)


def _sequencePattern(prefix: str, digits: List[str], suffix: str) -> str:
    """ Path pattern of a sequence: one "#" per digit when the numbers are padded, a single "#" otherwise. """
    lengths = {len(d) for d in digits}
    if len(lengths) == 1:
        padding = lengths.pop()
        # "1".."9" alone could be padded to 1 or unpadded: both write the same pattern
    else:
        # Different widths: unpadded numbers (1, 10, 100...), unless some have leading zeros
        padding = 1 if not any(d.startswith("0") and len(d) > 1 for d in digits) else min(len(d) for d in digits)
    return f"{prefix}{'#' * padding}{suffix}"


def listFolder(folder: str, imagesOnly: bool = True, extensions=None) -> List[Dict]:
    """
    List a folder: sub-folders first, then files, the frames of image sequences being grouped
    into one entry.

    `extensions` (e.g. [".mgt", ".mg"]): only the files of these extensions (instead of `imagesOnly`).

    Returns:
        A list of dicts with the keys: "name" (displayed name), "path" (path to use: the "####"
        pattern for sequences), "type" ("folder", "sequence" or "file"), "frames" (frame ranges
        description), "count" (number of files).
    """
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    folders, files = [], []
    for name in names:
        if name.startswith("."):
            continue
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            folders.append(name)
        elif extensions:
            # Only these kinds of files (e.g. [".mgt", ".mg"])
            if os.path.splitext(name)[1].lower() in extensions:
                files.append(name)
        elif not imagesOnly or os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            files.append(name)

    entries = [{"name": name, "path": os.path.join(folder, name), "type": "folder", "frames": "", "count": 0}
               for name in sorted(folders, key=str.lower)]

    groups: Dict[tuple, List[tuple]] = {}
    singles = []
    for name in files:
        match = _FRAME_NUMBER.match(name)
        if match:
            prefix, digits, suffix = match.groups()
            groups.setdefault((prefix, suffix), []).append((int(digits), digits, name))
        else:
            singles.append(name)

    items = []
    for (prefix, suffix), frames in groups.items():
        if len(frames) == 1:
            singles.append(frames[0][2])
            continue
        frames.sort()
        numbers = [frame for frame, _, _ in frames]
        pattern = _sequencePattern(prefix, [digits for _, digits, _ in frames], suffix)
        items.append({
            "name": pattern,
            "path": os.path.join(folder, pattern),
            "type": "sequence",
            "frames": frameRanges(numbers),
            "count": len(frames),
        })
    for name in singles:
        items.append({"name": name, "path": os.path.join(folder, name), "type": "file", "frames": "", "count": 1})
    items.sort(key=lambda item: item["name"].lower())
    return entries + items


def resolvePath(text: str, folder: str) -> Dict:
    """
    What a path typed in a file browser leads to (as the location bar of a file manager): {"type": "folder",
    "file" or "missing", "path": the absolute path}. "~" is the home folder; a relative path is relative to `folder`.
    """
    path = os.path.expanduser(os.path.expandvars((text or "").strip()))
    if not path:
        return {"type": "missing", "path": folder}
    if not os.path.isabs(path):
        path = os.path.join(folder or os.getcwd(), path)
    path = os.path.normpath(path)
    if os.path.isdir(path):
        return {"type": "folder", "path": path}
    if os.path.isfile(path):
        return {"type": "file", "path": path}
    return {"type": "missing", "path": path}
