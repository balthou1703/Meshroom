"""
UI requests: dialogs that a node descriptor can ask the Meshroom UI to open.

A node descriptor never imports Qt. To get an interactive editor (e.g. when one of its PushButtonParam
is clicked), it builds a UiRequest describing the dialog and either returns it from its
``on<Name>Clicked`` callback or calls ``request.open()``. The UI application registers a handler
(``setUiRequestHandler``) that turns requests into QML windows; in headless mode no handler is
registered and opening a request is a no-op that returns False.

Two kinds of requests are available:
  - TreePickerRequest: pick one or several items in a tree built from paths ("/a/b/leaf").
  - HierarchyEditorRequest: edit a ListAttribute of GroupAttribute as a tree of entries (the
    nesting being a parent/child dependency), optionally filled from a tree of source items, and
    validated by custom actions.

Results are given back to the descriptor through Python callbacks.
"""

import logging
from collections.abc import Mapping, Sequence

_uiRequestHandler = None


def setUiRequestHandler(handler):
    """
    Register the callable opening UiRequests (called by the UI application), or None to unregister.

    Args:
        handler: callable(request: UiRequest) or None.
    """
    global _uiRequestHandler
    _uiRequestHandler = handler


def hasUiRequestHandler() -> bool:
    """ Whether a UI is available to open UiRequests. """
    return _uiRequestHandler is not None


def openUiRequest(request) -> bool:
    """
    Ask the UI to open a request.

    Returns:
        bool: True if the request has been handed to a UI, False in headless mode.
    """
    if _uiRequestHandler is None:
        logging.warning(f"[UiRequest] No UI available to open '{request.title}'.")
        return False
    _uiRequestHandler(request)
    return True


def normalizeTreeItems(items) -> list:
    """
    Normalize tree items into a list of dicts {"path": str, "type": str}.

    Accepted item formats: objects with ``name``/``type`` attributes (e.g. a NodeInfo namedtuple),
    mappings with a "path" (or "name") and an optional "type" key, (path, type) sequences, or strings.
    """
    result = []
    for item in items or []:
        if isinstance(item, str):
            path, itemType = item, ""
        elif isinstance(item, Mapping):
            path, itemType = item.get("path", item.get("name", "")), item.get("type", "")
        elif hasattr(item, "name"):
            path, itemType = item.name, getattr(item, "type", "")
        elif isinstance(item, Sequence):
            path, itemType = item[0], (item[1] if len(item) > 1 else "")
        else:
            raise TypeError(f"Unsupported tree item: {item!r}")
        path = str(path or "").strip()
        if path:
            result.append({"path": path, "type": str(itemType or "")})
    return result


class UiRequest:
    """ Base class of the dialogs a node descriptor can ask the UI to open. """

    kind = ""

    def __init__(self, node, title: str = "", header: str = ""):
        """
        Args:
            node: the Node the request is made for (given back to the callbacks).
            title: title of the window.
            header: rich text displayed at the top of the window.
        """
        self.node = node
        self.title = title
        self.header = header

    def open(self) -> bool:
        """ Ask the UI to open this request. Returns False in headless mode. """
        return openUiRequest(self)

    @staticmethod
    def _call(callback, *args):
        """ Call a descriptor callback, logging (but not propagating) its errors. """
        try:
            callback(*args)
            return True
        except Exception:
            logging.exception(f"[UiRequest] Error in callback {callback!r}")
            return False


class TreePickerRequest(UiRequest):
    """
    Pick items in a tree built from paths.

    Each item path is split on "/": the intermediate parts are displayed as (non selectable) contexts,
    the last part is the selectable leaf.
    """

    kind = "treePicker"

    def __init__(self, node, items, onAccepted, title="Select", header="", columns=("Name", "Type"),
                 preselected=(), multiSelect=True, acceptLabel="OK", errorMessage=""):
        """
        Args:
            node: the Node the request is made for.
            items: the selectable items (see normalizeTreeItems).
            onAccepted: callable(node, paths: list[str]) called with the selected paths, in tree order.
            columns: header labels of the name and type columns.
            preselected: paths, or leaf names, selected when the dialog opens.
            multiSelect: whether several items can be selected.
            acceptLabel: label of the validation button.
            errorMessage: message displayed instead of the tree (e.g. the scene could not be parsed).
        """
        super().__init__(node, title, header)
        self.items = normalizeTreeItems(items)
        self.onAccepted = onAccepted
        self.columns = list(columns)
        self.preselected = [str(p) for p in preselected or []]
        self.multiSelect = multiSelect
        self.acceptLabel = acceptLabel
        self.errorMessage = errorMessage

    def accept(self, paths) -> bool:
        """ Give the selected paths back to the descriptor. """
        return self._call(self.onAccepted, self.node, list(paths))


class Action:
    """ A validation button of a HierarchyEditorRequest. """

    def __init__(self, name: str, label: str, callback, tooltip: str = "", color: str = "", closeDialog: bool = True):
        """
        Args:
            name: identifier of the action.
            label: text of the button.
            callback: callable(node, rows) called with the collected rows (see HierarchyEditorRequest).
            tooltip: tooltip of the button.
            color: optional background color of the button (e.g. "#4A7C59").
            closeDialog: whether the dialog is closed once the action is triggered.
        """
        self.name = name
        self.label = label
        self.callback = callback
        self.tooltip = tooltip
        self.color = color
        self.closeDialog = closeDialog


def columnsFromElementDesc(elementDesc) -> list:
    """
    Build the column descriptions of a HierarchyEditorRequest from the description of a ListAttribute
    element (a GroupAttribute, or a single Param).

    Returns:
        list of dicts {"key", "label", "description", "editor", "choices", "default"} where editor is
        one of "text", "choice", "bool", "int", "float".
    """
    from meshroom.core.desc import attribute as descAttr

    items = elementDesc.items if isinstance(elementDesc, descAttr.GroupAttribute) else [elementDesc]
    columns = []
    for item in items:
        if not item.visible:
            continue
        column = {
            "key": item.name,
            "label": item.label or item.name,
            "description": item.description or "",
            "editor": "text",
            "choices": [],
            "default": item.value,
        }
        if isinstance(item, descAttr.ChoiceParam) and item.exclusive:
            column["editor"] = "choice"
            column["choices"] = [str(v) for v in item.values]
        elif isinstance(item, descAttr.BoolParam):
            column["editor"] = "bool"
        elif isinstance(item, descAttr.IntParam):
            column["editor"] = "int"
        elif isinstance(item, descAttr.FloatParam):
            column["editor"] = "float"
        elif not isinstance(item, (descAttr.StringParam, descAttr.File, descAttr.ChoiceParam)):
            # Nested groups, lists... are not editable in a table cell
            continue
        if column["default"] is None:
            column["default"] = column["choices"][0] if column["choices"] else ""
        columns.append(column)
    return columns


def rowsFromListAttribute(listAttribute, keys) -> list:
    """ Read the values of a ListAttribute of GroupAttribute as a list of dicts restricted to `keys`. """
    rows = []
    for element in listAttribute.value:
        row = {}
        for key in keys:
            child = element.childAttribute(key) if hasattr(element, "childAttribute") else None
            row[key] = child.value if child is not None else None
        rows.append(row)
    return rows


class HierarchyEditorRequest(UiRequest):
    """
    Edit the entries of a ListAttribute of GroupAttribute as a tree.

    The columns are deduced from the element description of the ListAttribute (ChoiceParam -> combo
    box, BoolParam -> check box, other params -> text field). Entries can be nested by drag and drop:
    the nesting expresses a dependency (a child is computed after its parent).

    The dialog does not modify the ListAttribute itself: the actions receive the rows, a list of dicts
    holding one value per column plus a "parent" key (1-based index of the parent entry as a string,
    "" for a root entry). A parent always comes before its children.
    """

    kind = "hierarchyEditor"

    def __init__(self, node, listAttribute, actions, title="Edit", header="", rows=None, columns=None,
                 pathKey=None, sourceItems=None, sourceColumns=("Node / Context", "Type"), sourceHeader="",
                 sourceErrorMessage="", entriesHeader=""):
        """
        Args:
            node: the Node the request is made for.
            listAttribute: the edited ListAttribute (its element description defines the columns).
            actions: list of Action (validation buttons).
            rows: initial rows (list of dicts, optional "parent" key). Read from listAttribute if None.
            columns: override of the columns (see columnsFromElementDesc).
            pathKey: key of the column filled with the source item path. Defaults to the first column.
            sourceItems: items of the source tree (see normalizeTreeItems). No source pane if None.
            sourceColumns: header labels of the source tree.
            sourceHeader: rich text displayed above the source tree.
            sourceErrorMessage: message displayed instead of the source tree.
            entriesHeader: rich text displayed above the entries.
        """
        super().__init__(node, title, header)
        self.listAttribute = listAttribute
        self.actions = list(actions)
        self.columns = columns if columns is not None else columnsFromElementDesc(listAttribute.desc.elementDesc)
        if not self.columns:
            raise ValueError("HierarchyEditorRequest: no editable column.")
        self.pathKey = pathKey or self.columns[0]["key"]
        keys = [c["key"] for c in self.columns]
        self.rows = rows if rows is not None else rowsFromListAttribute(listAttribute, keys)
        self.sourceItems = normalizeTreeItems(sourceItems) if sourceItems is not None else None
        self.sourceColumns = list(sourceColumns)
        self.sourceHeader = sourceHeader
        self.sourceErrorMessage = sourceErrorMessage
        self.entriesHeader = entriesHeader

    def action(self, name):
        """ Get an action by its name, or None. """
        return next((a for a in self.actions if a.name == name), None)

    def trigger(self, name, rows) -> bool:
        """ Run the action `name` with the collected rows. """
        action = self.action(name)
        if action is None:
            logging.warning(f"[UiRequest] Unknown action '{name}'.")
            return False
        return self._call(action.callback, self.node, rows)
