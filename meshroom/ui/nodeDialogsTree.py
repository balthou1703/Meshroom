"""
Qt-free data structures behind the node dialogs (see meshroom/core/desc/uiRequest.py).

  - PickerTree: a tree built from item paths ("/ctx/sub/leaf"), with filtering, folding and checked
    leaves. Used by the tree picker and by the source pane of the hierarchy editor.
  - EntryHierarchy: an ordered forest of editable entries (one per element of the edited
    ListAttribute), flattened into rows with a 1-based "parent" index.

They are exposed to QML as list models by meshroom/ui/components/nodeDialogs.py.
"""


class PickerTree:
    """
    Tree of contexts and leaves built from item paths.

    Every item path is a leaf: its intermediate path parts become contexts. Only leaves can be checked;
    checking a context checks all its visible leaves.
    """

    def __init__(self, items=(), singleSelection=False):
        """
        Args:
            items: list of dicts {"path", "type"} (see desc.uiRequest.normalizeTreeItems).
            singleSelection: whether at most one leaf can be checked.
        """
        self.singleSelection = singleSelection
        self.filterText = ""
        self._nodes = []  # {"key", "name", "type", "path" (leaves only), "depth", "children", "expanded"}
        self._roots = []
        self._byKey = {}
        self._checked = set()
        for item in items:
            self._addLeaf(item["path"], item.get("type", ""))

    def _addLeaf(self, path, itemType):
        parts = [p for p in path.split("/") if p]
        if not parts:
            return
        parentIndex = None
        for depth in range(len(parts) - 1):
            parentIndex = self._getOrCreate("/" + "/".join(parts[:depth + 1]), parts[depth], depth, parentIndex)
        key = "/" + "/".join(parts)
        if key in self._byKey and self._nodes[self._byKey[key]]["path"] is not None:
            return  # Duplicated item
        if key in self._byKey:
            # A context with the same path is also an item: it becomes checkable
            node = self._nodes[self._byKey[key]]
            node["path"], node["type"] = path, itemType
            return
        index = self._getOrCreate(key, parts[-1], len(parts) - 1, parentIndex)
        self._nodes[index]["path"] = path
        self._nodes[index]["type"] = itemType

    def _getOrCreate(self, key, name, depth, parentIndex):
        if key in self._byKey:
            return self._byKey[key]
        index = len(self._nodes)
        self._nodes.append({"key": key, "name": name, "type": "", "path": None, "depth": depth,
                            "children": [], "expanded": True})
        self._byKey[key] = index
        (self._nodes[parentIndex]["children"] if parentIndex is not None else self._roots).append(index)
        return index

    # Queries

    @property
    def leafCount(self) -> int:
        return sum(1 for n in self._nodes if n["path"] is not None)

    def isLeaf(self, key) -> bool:
        index = self._byKey.get(key)
        return index is not None and self._nodes[index]["path"] is not None

    def _matches(self, node) -> bool:
        text = self.filterText
        return not text or text in node["name"].lower() or text in node["type"].lower()

    def _visibleSet(self) -> set:
        """ Indices of the nodes matching the filter, or having a matching descendant. """
        visible = set()

        def walk(index):
            node = self._nodes[index]
            childVisible = False
            for child in node["children"]:
                if walk(child):
                    childVisible = True
            if childVisible or self._matches(node):
                visible.add(index)
                return True
            return False

        for root in self._roots:
            walk(root)
        return visible

    def rows(self) -> list:
        """
        The displayed rows, in tree order: list of dicts {"key", "name", "type", "depth", "isLeaf",
        "hasChildren", "expanded", "checkState"} (checkState: 0 unchecked, 1 partially, 2 checked).
        While a filter is active, all the matching branches are unfolded.
        """
        visible = self._visibleSet()
        result = []

        def walk(index):
            if index not in visible:
                return
            node = self._nodes[index]
            expanded = node["expanded"] or bool(self.filterText)
            result.append({
                "key": node["key"],
                "name": node["name"],
                "type": node["type"],
                "depth": node["depth"],
                "isLeaf": node["path"] is not None,
                "hasChildren": bool(node["children"]),
                "expanded": expanded,
                "checkState": self._checkState(index, visible),
            })
            if expanded:
                for child in node["children"]:
                    walk(child)

        for root in self._roots:
            walk(root)
        return result

    def _leavesUnder(self, index, visible=None) -> list:
        """ Leaf indices of the subtree of `index` (itself included), restricted to `visible`. """
        result = []

        def walk(i):
            if visible is not None and i not in visible:
                return
            if self._nodes[i]["path"] is not None:
                result.append(i)
            for child in self._nodes[i]["children"]:
                walk(child)

        walk(index)
        return result

    def _checkState(self, index, visible) -> int:
        node = self._nodes[index]
        if node["path"] is not None:
            return 2 if node["key"] in self._checked else 0
        leaves = self._leavesUnder(index, visible)
        count = sum(1 for i in leaves if self._nodes[i]["key"] in self._checked)
        if count == 0:
            return 0
        return 2 if count == len(leaves) else 1

    def visibleLeafKeys(self) -> list:
        """ Keys of the leaves matching the filter, in tree order (folding is ignored). """
        visible = self._visibleSet()
        result = []
        for root in self._roots:
            result.extend(self._nodes[i]["key"] for i in self._leavesUnder(root, visible))
        return result

    def checkedPaths(self) -> list:
        """ Paths of the checked leaves, in tree order (hidden ones included). """
        result = []
        for root in self._roots:
            for i in self._leavesUnder(root):
                if self._nodes[i]["key"] in self._checked:
                    result.append(self._nodes[i]["path"])
        return result

    @property
    def checkedCount(self) -> int:
        return len(self._checked)

    # Modifications

    def setFilter(self, text):
        self.filterText = (text or "").strip().lower()

    def toggleExpanded(self, key):
        index = self._byKey.get(key)
        if index is not None:
            self._nodes[index]["expanded"] = not self._nodes[index]["expanded"]

    def setChecked(self, key, checked: bool):
        """ Check or uncheck a leaf, or all the visible leaves of a context. """
        index = self._byKey.get(key)
        if index is None:
            return
        keys = [self._nodes[i]["key"] for i in self._leavesUnder(index, self._visibleSet())]
        if self.singleSelection:
            if checked and len(keys) == 1:
                self._checked = set(keys)
            elif not checked:
                self._checked.difference_update(keys)
            return
        if checked:
            self._checked.update(keys)
        else:
            self._checked.difference_update(keys)

    def toggle(self, key):
        """ Toggle a leaf, or all the visible leaves of a context (checked unless all are checked). """
        index = self._byKey.get(key)
        if index is None:
            return
        state = self._checkState(index, self._visibleSet())
        self.setChecked(key, state != 2)

    def setRangeChecked(self, anchorKey, key, checked: bool):
        """ Check or uncheck all the visible leaves between `anchorKey` and `key` (both included). """
        leaves = self.visibleLeafKeys()
        if anchorKey not in leaves or key not in leaves or self.singleSelection:
            self.setChecked(key, checked)
            return
        lo, hi = sorted((leaves.index(anchorKey), leaves.index(key)))
        for k in leaves[lo:hi + 1]:
            self.setChecked(k, checked)

    def setAllChecked(self, checked: bool):
        """ Check or uncheck all the visible leaves. """
        keys = self.visibleLeafKeys()
        if checked and self.singleSelection:
            return
        if checked:
            self._checked.update(keys)
        else:
            self._checked.difference_update(keys)

    def checkPaths(self, paths):
        """ Check the leaves whose path, or leaf name, is in `paths`. """
        wanted = set(paths)
        for node in self._nodes:
            if node["path"] is not None and (node["path"] in wanted or node["name"] in wanted):
                self._checked.add(node["key"])
                if self.singleSelection:
                    return

    def isChecked(self, key) -> bool:
        return key in self._checked

    def pathOf(self, key):
        index = self._byKey.get(key)
        return self._nodes[index]["path"] if index is not None else None


class EntryHierarchy:
    """
    Ordered forest of editable entries.

    Each entry holds one value per column. The nesting expresses a dependency: a child is computed
    after its parent, siblings are independent.
    """

    def __init__(self, columns, pathKey):
        """
        Args:
            columns: list of column dicts (see desc.uiRequest.columnsFromElementDesc).
            pathKey: key of the column holding the path of the entry.
        """
        self.columns = columns
        self.pathKey = pathKey
        self._entries = {}  # id -> {"id", "values", "parent", "children", "expanded"}
        self._roots = []
        self._nextId = 1

    # Construction

    def _defaultValues(self) -> dict:
        return {c["key"]: c["default"] for c in self.columns}

    def _conformValues(self, values) -> dict:
        result = self._defaultValues()
        for column in self.columns:
            key = column["key"]
            value = (values or {}).get(key)
            if value is None or (isinstance(value, str) and value.strip() == "" and key != self.pathKey):
                continue
            if column["editor"] == "bool" and isinstance(value, str):
                value = value.strip().lower() == "true"
            elif column["editor"] == "choice":
                value = str(value)
            result[key] = value
        return result

    def _siblings(self, parentId) -> list:
        return self._entries[parentId]["children"] if parentId is not None else self._roots

    def addEntry(self, values=None, parentId=None, index=None) -> int:
        """ Create an entry under `parentId` (None: root) at `index` (None: last). Returns its id. """
        entryId = self._nextId
        self._nextId += 1
        self._entries[entryId] = {"id": entryId, "values": self._conformValues(values), "parent": parentId,
                                  "children": [], "expanded": True}
        siblings = self._siblings(parentId)
        siblings.insert(len(siblings) if index is None else index, entryId)
        return entryId

    def loadRows(self, rows):
        """
        Rebuild the forest from rows. The optional "parent" key of a row is the 1-based index of its
        parent row; invalid parents and cycles fall back to the root level.
        """
        self._entries.clear()
        self._roots.clear()
        rows = list(rows or [])

        def parentIndex(i):
            try:
                value = int(str(rows[i].get("parent", "") or "").strip()) - 1
            except ValueError:
                return None
            return value if 0 <= value < len(rows) and value != i else None

        ids = {}
        pending = list(range(len(rows)))
        progress = True
        while pending and progress:
            progress = False
            for i in list(pending):
                p = parentIndex(i)
                if p is not None and p not in ids:
                    continue  # Parent not created yet
                ids[i] = self.addEntry(rows[i], parentId=ids.get(p))
                pending.remove(i)
                progress = True
        for i in pending:  # Cycles
            ids[i] = self.addEntry(rows[i])

    # Queries

    def __len__(self):
        return len(self._entries)

    def __contains__(self, entryId):
        return entryId in self._entries

    def entry(self, entryId) -> dict:
        return self._entries[entryId]

    def ids(self) -> list:
        """ All the entry ids, in tree order. """
        result = []

        def walk(entryId):
            result.append(entryId)
            for child in self._entries[entryId]["children"]:
                walk(child)

        for root in self._roots:
            walk(root)
        return result

    def rows(self) -> list:
        """
        The displayed rows, in tree order (folded subtrees excluded): list of dicts
        {"id", "depth", "hasChildren", "expanded", "values"}.
        """
        result = []

        def walk(entryId, depth):
            entry = self._entries[entryId]
            result.append({"id": entryId, "depth": depth, "hasChildren": bool(entry["children"]),
                           "expanded": entry["expanded"], "values": entry["values"]})
            if entry["expanded"]:
                for child in entry["children"]:
                    walk(child, depth + 1)

        for root in self._roots:
            walk(root, 0)
        return result

    def paths(self) -> set:
        return {e["values"].get(self.pathKey) for e in self._entries.values()}

    def isAncestor(self, ancestorId, entryId) -> bool:
        """ Whether `ancestorId` is `entryId` or one of its ancestors. """
        current = entryId
        while current is not None:
            if current == ancestorId:
                return True
            current = self._entries[current]["parent"]
        return False

    def collectRows(self) -> list:
        """
        Flatten the forest into rows: one dict per entry with all the column values plus "parent", the
        1-based index of the parent row as a string ("" for a root). A parent precedes its children.
        """
        rows = []
        indexOf = {}
        for entryId in self.ids():
            entry = self._entries[entryId]
            row = dict(entry["values"])
            path = row.get(self.pathKey)
            row[self.pathKey] = path.strip() if isinstance(path, str) else path
            parent = entry["parent"]
            row["parent"] = str(indexOf[parent] + 1) if parent is not None else ""
            indexOf[entryId] = len(rows)
            rows.append(row)
        return rows

    def _topLevel(self, ids) -> list:
        """ The given ids without those having an ancestor in the list, in tree order. """
        wanted = set(i for i in ids if i in self._entries)
        result = []
        for entryId in self.ids():
            if entryId not in wanted:
                continue
            parent = self._entries[entryId]["parent"]
            covered = False
            while parent is not None:
                if parent in wanted:
                    covered = True
                    break
                parent = self._entries[parent]["parent"]
            if not covered:
                result.append(entryId)
        return result

    # Modifications

    def setValue(self, entryId, key, value):
        if entryId in self._entries:
            self._entries[entryId]["values"][key] = value

    def toggleExpanded(self, entryId):
        if entryId in self._entries:
            self._entries[entryId]["expanded"] = not self._entries[entryId]["expanded"]

    def setAllExpanded(self, expanded: bool):
        for entry in self._entries.values():
            entry["expanded"] = expanded

    def addPaths(self, paths) -> list:
        """ Add one root entry per path not already present. Returns the created ids. """
        existing = self.paths()
        created = []
        for path in paths:
            if path in existing:
                continue
            created.append(self.addEntry({self.pathKey: path}))
            existing.add(path)
        return created

    def _detach(self, entryId):
        entry = self._entries[entryId]
        self._siblings(entry["parent"]).remove(entryId)

    def _attach(self, entryId, parentId, index=None):
        siblings = self._siblings(parentId)
        siblings.insert(len(siblings) if index is None else index, entryId)
        self._entries[entryId]["parent"] = parentId

    def remove(self, ids):
        """ Remove entries; their children take their place under their parent. """
        for entryId in [i for i in self.ids() if i in set(ids)]:
            if entryId not in self._entries:
                continue
            entry = self._entries[entryId]
            siblings = self._siblings(entry["parent"])
            position = siblings.index(entryId)
            siblings.pop(position)
            for offset, child in enumerate(entry["children"]):
                siblings.insert(position + offset, child)
                self._entries[child]["parent"] = entry["parent"]
            del self._entries[entryId]

    def clear(self):
        self._entries.clear()
        self._roots.clear()

    def duplicate(self, ids) -> list:
        """ Duplicate entries (without their children) next to the originals. Returns the created ids. """
        created = []
        for entryId in self._topLevel(ids):
            entry = self._entries[entryId]
            siblings = self._siblings(entry["parent"])
            created.append(self.addEntry(dict(entry["values"]), entry["parent"], siblings.index(entryId) + 1))
        return created

    def move(self, entryId, targetId, position) -> bool:
        """
        Move an entry (with its subtree) relatively to `targetId`.

        Args:
            position: "inside" (last child of the target), "before" or "after" (sibling of the target).
                With targetId None, the entry becomes the last root.

        Returns:
            bool: False if the move is invalid (e.g. into its own subtree).
        """
        if entryId not in self._entries:
            return False
        if targetId is None:
            self._detach(entryId)
            self._attach(entryId, None)
            return True
        if targetId not in self._entries or self.isAncestor(entryId, targetId):
            return False
        self._detach(entryId)
        target = self._entries[targetId]
        if position == "inside":
            self._attach(entryId, targetId)
            target["expanded"] = True
        else:
            siblings = self._siblings(target["parent"])
            index = siblings.index(targetId) + (1 if position == "after" else 0)
            self._attach(entryId, target["parent"], index)
        return True

    def moveMany(self, ids, targetId, position) -> bool:
        """ Move several entries (see move), keeping their relative order. """
        moving = self._topLevel(ids)
        if targetId is not None and any(self.isAncestor(i, targetId) for i in moving):
            return False
        if position == "before" or position == "inside" or targetId is None:
            for entryId in moving:
                self.move(entryId, targetId, position)
        else:
            anchor = targetId
            for entryId in moving:
                self.move(entryId, anchor, "after")
                anchor = entryId
        return True

    def indent(self, ids):
        """ Make each entry the last child of its previous sibling. """
        for entryId in self._topLevel(ids):
            siblings = self._siblings(self._entries[entryId]["parent"])
            position = siblings.index(entryId)
            if position > 0:
                self.move(entryId, siblings[position - 1], "inside")

    def outdent(self, ids):
        """ Move each entry right after its parent, at the parent's level. """
        for entryId in reversed(self._topLevel(ids)):
            parent = self._entries[entryId]["parent"]
            if parent is not None:
                self.move(entryId, parent, "after")

    def unparent(self, ids):
        """ Move each entry to the root level, right after its root ancestor. """
        for entryId in reversed(self._topLevel(ids)):
            root = entryId
            while self._entries[root]["parent"] is not None:
                root = self._entries[root]["parent"]
            if root != entryId:
                self.move(entryId, root, "after")

    def shift(self, ids, offset: int):
        """ Move entries up (offset -1) or down (offset 1) among their siblings. """
        moving = self._topLevel(ids)
        if offset > 0:
            moving.reverse()
        for entryId in moving:
            siblings = self._siblings(self._entries[entryId]["parent"])
            position = siblings.index(entryId)
            newPosition = position + offset
            if 0 <= newPosition < len(siblings) and siblings[newPosition] not in moving:
                siblings[position], siblings[newPosition] = siblings[newPosition], siblings[position]
