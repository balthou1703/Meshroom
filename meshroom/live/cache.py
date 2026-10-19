"""
In-memory cache of rendered images.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Dict, Hashable, List, Optional

from .imageBuffer import ImageBuffer


def systemMemory() -> int:
    """ Physical memory in bytes (0 if it cannot be determined). """
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        return 0


def defaultCacheSize() -> int:
    """
    Default cache size: MESHROOM_LIVE_CACHE_GB if set, otherwise a quarter of the physical memory
    (2 GiB if it cannot be determined).
    """
    try:
        requested = float(os.environ.get("MESHROOM_LIVE_CACHE_GB", "0"))
    except ValueError:
        requested = 0
    if requested > 0:
        return int(requested * 1024 ** 3)
    total = systemMemory()
    return total // 4 if total else 2 * 1024 ** 3


class ImageCache:
    """
    Thread-safe LRU cache of ImageBuffers, bounded in bytes.

    Keys are `(node uid, frame)`: since the uid of a node changes whenever one of its parameters
    or anything upstream changes, outdated entries are never hit again and are evicted over time.

    Memory is counted per pixel array: images sharing arrays (a pass-through node, a node that
    only changes one layer) are counted once.
    """

    def __init__(self, maxBytes: Optional[int] = None):
        self._maxBytes = defaultCacheSize() if maxBytes is None else int(maxBytes)
        self._entries: "OrderedDict[Hashable, ImageBuffer]" = OrderedDict()
        # id(array) -> [reference count, array]; the array is kept so that its id is not reused
        self._arrays: Dict[int, list] = {}
        self._bytes = 0
        self._lock = threading.Lock()

    @property
    def maxBytes(self) -> int:
        return self._maxBytes

    @maxBytes.setter
    def maxBytes(self, value: int):
        with self._lock:
            self._maxBytes = int(value)
            self._evict()

    @property
    def usedBytes(self) -> int:
        return self._bytes

    def __len__(self):
        return len(self._entries)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._entries

    def _addArrays(self, image: ImageBuffer):
        for array in image.arrays():
            entry = self._arrays.get(id(array))
            if entry is None:
                self._arrays[id(array)] = [1, array]
                self._bytes += array.nbytes
            else:
                entry[0] += 1

    def _removeArrays(self, image: ImageBuffer):
        for array in image.arrays():
            entry = self._arrays.get(id(array))
            if entry is None:
                continue
            entry[0] -= 1
            if entry[0] <= 0:
                del self._arrays[id(array)]
                self._bytes -= array.nbytes

    def _newBytes(self, image: ImageBuffer) -> int:
        """ Memory an image would add to the cache. """
        return sum(array.nbytes for array in image.arrays() if id(array) not in self._arrays)

    def get(self, key: Hashable) -> Optional[ImageBuffer]:
        with self._lock:
            image = self._entries.get(key)
            if image is not None:
                self._entries.move_to_end(key)
            return image

    def put(self, key: Hashable, image: ImageBuffer):
        """ Store an image. An image larger than the whole cache is not stored. """
        with self._lock:
            if key in self._entries:
                self._removeArrays(self._entries.pop(key))
            if self._newBytes(image) > self._maxBytes:
                return
            self._entries[key] = image
            self._addArrays(image)
            self._evict()

    def cachedFrames(self, uid: str) -> List[int]:
        """ Return the frames of a node uid currently in cache, sorted. """
        with self._lock:
            return sorted(key[1] for key in self._entries if isinstance(key, tuple) and key[0] == uid)

    def discardUids(self, uids):
        """ Remove every cached frame of the given node uids. """
        uids = set(uids)
        with self._lock:
            for key in [k for k in self._entries if isinstance(k, tuple) and k[0] in uids]:
                self._removeArrays(self._entries.pop(key))

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._arrays.clear()
            self._bytes = 0

    def _evict(self):
        """ Remove the least recently used entries until the cache fits in its budget. """
        while self._bytes > self._maxBytes and len(self._entries) > 1:
            _, image = self._entries.popitem(last=False)
            self._removeArrays(image)
