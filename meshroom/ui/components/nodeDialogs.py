"""
Qt side of the node dialogs: the dialogs node descriptors can ask the UI to open by returning a
desc.UiRequest (see meshroom/core/desc/uiRequest.py).

NodeDialogManager, exposed to QML as `_nodeDialogs`, receives the requests and wraps each of them in a
backend object (TreePickerBackend, HierarchyEditorBackend) that the QML NodeDialogs component displays
in its own window.
"""

import logging

from PySide6.QtCore import QAbstractListModel, QByteArray, QModelIndex, QObject, Qt, Property, Signal, Slot

from meshroom.core import desc
from meshroom.ui.nodeDialogsTree import EntryHierarchy, PickerTree


def _roleNames(names):
    """ Build the role ids and the roleNames() mapping of a list model from a list of role names. """
    roles = {name: Qt.UserRole + 1 + i for i, name in enumerate(names)}
    return roles, {role: QByteArray(name.encode()) for name, role in roles.items()}


class PickerTreeModel(QAbstractListModel):
    """ List model of the displayed rows of a PickerTree (contexts and checkable leaves). """

    _ROLES, _ROLE_NAMES = _roleNames(["key", "name", "type", "depth", "isLeaf", "hasChildren", "expanded",
                                      "checkState"])

    checkedCountChanged = Signal()

    def __init__(self, tree: PickerTree, parent=None):
        super().__init__(parent)
        self._tree = tree
        self._rows = tree.rows()
        self._anchor = None  # Last clicked leaf, for range selection

    # QAbstractListModel

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def roleNames(self):
        return self._ROLE_NAMES

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        row = self._rows[index.row()]
        if role == Qt.DisplayRole:
            return row["name"]
        for name, value in self._ROLES.items():
            if value == role:
                return row[name]
        return None

    def _refresh(self, structural: bool):
        """ Update the rows: reset the model if they changed, else only notify the data change. """
        rows = self._tree.rows()
        if structural or [r["key"] for r in rows] != [r["key"] for r in self._rows]:
            self.beginResetModel()
            self._rows = rows
            self.endResetModel()
        else:
            self._rows = rows
            if rows:
                self.dataChanged.emit(self.index(0), self.index(len(rows) - 1), [self._ROLES["checkState"]])
        self.checkedCountChanged.emit()

    # Slots

    @Slot(str)
    def setFilter(self, text):
        self._tree.setFilter(text)
        self._refresh(True)

    @Slot(str)
    def toggleExpanded(self, key):
        self._tree.toggleExpanded(key)
        self._refresh(True)

    @Slot(str, bool)
    def click(self, key, shift):
        """
        Click on a row: toggle a leaf (or all the visible leaves of a context). With shift, apply the
        state of the last clicked leaf to all the visible leaves in between.
        """
        if shift and self._anchor is not None and self._anchor != key and self._tree.isLeaf(key):
            self._tree.setRangeChecked(self._anchor, key, self._tree.isChecked(self._anchor))
        else:
            self._tree.toggle(key)
            if self._tree.isLeaf(key):
                self._anchor = key
        self._refresh(False)

    @Slot(bool)
    def setAllChecked(self, checked):
        self._tree.setAllChecked(checked)
        self._refresh(False)

    @Slot(result="QVariantList")
    def checkedPaths(self):
        return self._tree.checkedPaths()

    @Slot(str, result=str)
    def pathOf(self, key):
        return self._tree.pathOf(key) or ""

    checkedCount = Property(int, lambda self: self._tree.checkedCount, notify=checkedCountChanged)
    leafCount = Property(int, lambda self: self._tree.leafCount, constant=True)


class EntryHierarchyModel(QAbstractListModel):
    """ List model of the displayed rows of an EntryHierarchy, with a selection of entries. """

    _ROLES, _ROLE_NAMES = _roleNames(["entryId", "depth", "hasChildren", "expanded", "values", "selected"])

    countChanged = Signal()
    selectionChanged = Signal()

    def __init__(self, hierarchy: EntryHierarchy, parent=None):
        super().__init__(parent)
        self._hierarchy = hierarchy
        self._rows = hierarchy.rows()
        self._selected = set()
        self._anchor = None

    # QAbstractListModel

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def roleNames(self):
        return self._ROLE_NAMES

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        row = self._rows[index.row()]
        if role == self._ROLES["entryId"]:
            return row["id"]
        if role == self._ROLES["selected"]:
            return row["id"] in self._selected
        if role == self._ROLES["values"]:
            return dict(row["values"])
        for name in ("depth", "hasChildren", "expanded"):
            if role == self._ROLES[name]:
                return row[name]
        if role == Qt.DisplayRole:
            return str(row["values"].get(self._hierarchy.pathKey, ""))
        return None

    def _refresh(self):
        """ Reset the rows after a structural change, dropping the selection of removed entries. """
        self.beginResetModel()
        self._rows = self._hierarchy.rows()
        self._selected = {i for i in self._selected if i in self._hierarchy}
        self.endResetModel()
        self.countChanged.emit()
        self.selectionChanged.emit()

    def _notifySelection(self):
        if self._rows:
            self.dataChanged.emit(self.index(0), self.index(len(self._rows) - 1), [self._ROLES["selected"]])
        self.selectionChanged.emit()

    def _rowOf(self, entryId):
        return next((i for i, r in enumerate(self._rows) if r["id"] == entryId), -1)

    # Selection

    @Slot(int, bool, bool)
    def select(self, entryId, toggle, extend):
        """
        Select an entry: alone, toggled in the selection (ctrl), or the displayed range from the last
        selected entry (shift).
        """
        if extend and self._anchor is not None and self._rowOf(self._anchor) >= 0:
            lo, hi = sorted((self._rowOf(self._anchor), self._rowOf(entryId)))
            self._selected = {r["id"] for r in self._rows[lo:hi + 1]}
        elif toggle:
            self._selected ^= {entryId}
            self._anchor = entryId
        else:
            self._selected = {entryId}
            self._anchor = entryId
        self._notifySelection()

    @Slot(int)
    def ensureSelected(self, entryId):
        """ Select only `entryId` unless it is already selected (before dragging it). """
        if entryId not in self._selected:
            self.select(entryId, False, False)

    @Slot()
    def selectAll(self):
        self._selected = set(self._hierarchy.ids())
        self._notifySelection()

    @Slot()
    def clearSelection(self):
        self._selected = set()
        self._notifySelection()

    # Edition

    @Slot(int, str, "QVariant")
    def setValue(self, entryId, key, value):
        if entryId not in self._hierarchy or self._hierarchy.entry(entryId)["values"].get(key) == value:
            return
        self._hierarchy.setValue(entryId, key, value)
        row = self._rowOf(entryId)
        if row >= 0:
            self.dataChanged.emit(self.index(row), self.index(row), [self._ROLES["values"]])

    @Slot(int, str, "QVariant")
    def setValueOnSelection(self, entryId, key, value):
        """ Set a value on `entryId`, and on all the selected entries if `entryId` is part of the selection. """
        targets = self._selected if entryId in self._selected else {entryId}
        for target in targets:
            self.setValue(target, key, value)

    @Slot(int)
    def toggleExpanded(self, entryId):
        self._hierarchy.toggleExpanded(entryId)
        self._refresh()

    @Slot(bool)
    def setAllExpanded(self, expanded):
        self._hierarchy.setAllExpanded(expanded)
        self._refresh()

    def addPaths(self, paths):
        created = self._hierarchy.addPaths(paths)
        self._refresh()
        return created

    @Slot()
    def addEmpty(self):
        entryId = self._hierarchy.addEntry()
        self._selected = {entryId}
        self._anchor = entryId
        self._refresh()

    @Slot()
    def removeSelected(self):
        self._hierarchy.remove(self._selected)
        self._refresh()

    @Slot()
    def duplicateSelected(self):
        created = self._hierarchy.duplicate(self._selected)
        if created:
            self._selected = set(created)
        self._refresh()

    @Slot()
    def indentSelected(self):
        self._hierarchy.indent(self._selected)
        self._refresh()

    @Slot()
    def outdentSelected(self):
        self._hierarchy.outdent(self._selected)
        self._refresh()

    @Slot()
    def unparentSelected(self):
        self._hierarchy.unparent(self._selected)
        self._refresh()

    @Slot(int)
    def shiftSelected(self, offset):
        self._hierarchy.shift(self._selected, offset)
        self._refresh()

    @Slot()
    def clear(self):
        self._hierarchy.clear()
        self._refresh()

    @Slot(int, str, result=bool)
    def canDropSelection(self, targetId, position):
        """ Whether the selected entries can be dropped relatively to `targetId` (-1: root level). """
        if targetId < 0:
            return True
        return not any(self._hierarchy.isAncestor(i, targetId) for i in self._selected if i in self._hierarchy)

    @Slot(int, str, result=bool)
    def dropSelection(self, targetId, position):
        """ Move the selected entries "inside", "before" or "after" `targetId` (-1: last at root level). """
        moved = self._hierarchy.moveMany(self._selected, None if targetId < 0 else targetId, position)
        if moved:
            self._refresh()
        return moved

    def collectRows(self):
        return self._hierarchy.collectRows()

    count = Property(int, lambda self: len(self._hierarchy), notify=countChanged)
    selectedCount = Property(int, lambda self: len(self._selected), notify=selectionChanged)


class NodeDialogBackend(QObject):
    """ Base class of the objects bridging a desc.UiRequest and its QML window. """

    def __init__(self, request: desc.UiRequest, parent=None):
        super().__init__(parent)
        self._request = request

    def _nodeLabel(self):
        node = self._request.node
        return getattr(node, "label", None) or getattr(node, "name", "") or ""

    kind = Property(str, lambda self: self._request.kind, constant=True)
    title = Property(str, lambda self: self._request.title, constant=True)
    header = Property(str, lambda self: self._request.header, constant=True)
    nodeLabel = Property(str, _nodeLabel, constant=True)


class TreePickerBackend(NodeDialogBackend):
    """ Backend of a desc.TreePickerRequest. """

    def __init__(self, request: desc.TreePickerRequest, parent=None):
        super().__init__(request, parent)
        tree = PickerTree(request.items, singleSelection=not request.multiSelect)
        tree.checkPaths(request.preselected)
        self._model = PickerTreeModel(tree, parent=self)

    @Slot()
    def accept(self):
        self._request.accept(self._model.checkedPaths())

    model = Property(QObject, lambda self: self._model, constant=True)
    columns = Property("QVariantList", lambda self: self._request.columns, constant=True)
    acceptLabel = Property(str, lambda self: self._request.acceptLabel, constant=True)
    multiSelect = Property(bool, lambda self: self._request.multiSelect, constant=True)
    errorMessage = Property(str, lambda self: self._request.errorMessage, constant=True)


class HierarchyEditorBackend(NodeDialogBackend):
    """ Backend of a desc.HierarchyEditorRequest. """

    def __init__(self, request: desc.HierarchyEditorRequest, parent=None):
        super().__init__(request, parent)
        hierarchy = EntryHierarchy(request.columns, request.pathKey)
        hierarchy.loadRows(request.rows)
        self._entries = EntryHierarchyModel(hierarchy, parent=self)
        self._source = None
        if request.sourceItems is not None:
            self._source = PickerTreeModel(PickerTree(request.sourceItems), parent=self)

    @Slot()
    def addCheckedSources(self):
        """ Add the checked source items as root entries (existing paths are skipped). """
        if self._source is not None:
            self._entries.addPaths(self._source.checkedPaths())

    @Slot(str)
    def addSource(self, key):
        """ Add one source item (double click). """
        if self._source is not None and self._source.pathOf(key):
            self._entries.addPaths([self._source.pathOf(key)])

    @Slot(str, result=bool)
    def trigger(self, name):
        """
        Run an action with the collected rows.

        Returns:
            bool: whether the dialog should be closed.
        """
        action = self._request.action(name)
        if action is None:
            return False
        success = self._request.trigger(name, self._entries.collectRows())
        return success and action.closeDialog

    def _actions(self):
        return [{"name": a.name, "label": a.label, "tooltip": a.tooltip, "color": a.color} for a in self._request.actions]

    entries = Property(QObject, lambda self: self._entries, constant=True)
    source = Property(QObject, lambda self: self._source, constant=True)
    hasSource = Property(bool, lambda self: self._source is not None, constant=True)
    columns = Property("QVariantList", lambda self: self._request.columns, constant=True)
    pathKey = Property(str, lambda self: self._request.pathKey, constant=True)
    actions = Property("QVariantList", _actions, constant=True)
    sourceColumns = Property("QVariantList", lambda self: self._request.sourceColumns, constant=True)
    sourceHeader = Property(str, lambda self: self._request.sourceHeader, constant=True)
    sourceErrorMessage = Property(str, lambda self: self._request.sourceErrorMessage, constant=True)
    entriesHeader = Property(str, lambda self: self._request.entriesHeader, constant=True)


class NodeDialogManager(QObject):
    """
    Receive the desc.UiRequest opened by node descriptors and expose them to QML (as `_nodeDialogs`).

    It is registered as the desc UiRequest handler: the requests can be opened from any thread, the
    backends are always created in the thread of the manager.
    """

    _BACKENDS = {
        desc.TreePickerRequest.kind: TreePickerBackend,
        desc.HierarchyEditorRequest.kind: HierarchyEditorBackend,
    }

    # Emitted with the backend of a request to display
    dialogRequested = Signal(QObject)
    _requestReceived = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._backends = []
        self._requestReceived.connect(self._createBackend, Qt.QueuedConnection)

    def handle(self, request: desc.UiRequest):
        """ desc UiRequest handler: forward the request to the thread of the manager. """
        self._requestReceived.emit(request)

    @Slot(object)
    def _createBackend(self, request):
        backendType = self._BACKENDS.get(request.kind)
        if backendType is None:
            logging.error(f"[NodeDialogs] Unsupported request kind: '{request.kind}'")
            return
        try:
            backend = backendType(request, parent=self)
        except Exception:
            logging.exception(f"[NodeDialogs] Could not open '{request.title}'")
            return
        self._backends.append(backend)
        self.dialogRequested.emit(backend)

    @Slot(QObject)
    def release(self, backend):
        """ Forget a backend once its window is closed. """
        if backend in self._backends:
            self._backends.remove(backend)
            backend.deleteLater()
