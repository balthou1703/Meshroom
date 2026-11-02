from meshroom.core import desc
from meshroom.core.graph import Graph

from .utils import registerNodeDesc, unregisterNodeDesc


class NodeWithDialogs(desc.Node):
    """ Node opening dialogs from its PushButtonParams. """

    inputs = [
        desc.StringParam(name="picked", label="Picked", description="", value=""),
        desc.ListAttribute(
            name="configs",
            label="Configs",
            description="",
            elementDesc=desc.GroupAttribute(
                name="config",
                label="Config",
                description="",
                items=[
                    desc.StringParam(name="nodePath", label="Node Path", description="", value=""),
                    desc.ChoiceParam(name="extension", label="Extension", description="", value="usd",
                                     values=["usd", "usda"], exclusive=True),
                    desc.BoolParam(name="separate", label="Separate", description="", value=False),
                    desc.IntParam(name="hidden", label="Hidden", description="", value=0, visible=False),
                ],
            ),
        ),
        desc.PushButtonParam(name="pick", label="Pick", description=""),
        desc.PushButtonParam(name="edit", label="Edit", description=""),
        desc.PushButtonParam(name="nothing", label="Nothing", description=""),
    ]

    def onPickClicked(self, node):
        return desc.TreePickerRequest(
            node, [("/stage/a", "usdrender_rop"), {"path": "/stage/b"}],
            onAccepted=lambda n, paths: setattr(n.picked, "value", ",".join(paths)),
        )

    def onEditClicked(self, node):
        return desc.HierarchyEditorRequest(node, node.configs, actions=[
            desc.Action("apply", "Apply", lambda n, rows: n.configs.resetToDefaultValue() or
                        n.configs.extend([{k: v for k, v in r.items() if k != "parent"} for r in rows])),
        ])

    def onNothingClicked(self, node):
        pass


class TestUiRequest:

    @classmethod
    def setup_class(cls):
        registerNodeDesc(NodeWithDialogs)

    @classmethod
    def teardown_class(cls):
        unregisterNodeDesc(NodeWithDialogs)

    def setup_method(self):
        self.requests = []
        desc.setUiRequestHandler(self.requests.append)

    def teardown_method(self):
        desc.setUiRequestHandler(None)

    def test_headlessOpenReturnsFalse(self):
        desc.setUiRequestHandler(None)
        node = Graph("").addNewNode(NodeWithDialogs.__name__)
        request = desc.TreePickerRequest(node, [], onAccepted=lambda n, p: None)
        assert not desc.hasUiRequestHandler()
        assert request.open() is False
        # Clicking still works without UI
        node.pick.clicked()

    def test_clickReturningRequestOpensIt(self):
        node = Graph("").addNewNode(NodeWithDialogs.__name__)
        node.pick.clicked()
        assert len(self.requests) == 1
        request = self.requests[0]
        assert request.kind == "treePicker"
        assert request.items == [{"path": "/stage/a", "type": "usdrender_rop"}, {"path": "/stage/b", "type": ""}]
        assert request.accept(["/stage/a", "/stage/b"])
        assert node.picked.value == "/stage/a,/stage/b"

    def test_clickReturningNothing(self):
        node = Graph("").addNewNode(NodeWithDialogs.__name__)
        node.nothing.clicked()
        assert self.requests == []

    def test_hierarchyEditorColumnsAndRows(self):
        node = Graph("").addNewNode(NodeWithDialogs.__name__)
        node.configs.append({"nodePath": "/stage/rop1", "extension": "usda", "separate": True})
        node.edit.clicked()
        request = self.requests[0]
        assert [c["key"] for c in request.columns] == ["nodePath", "extension", "separate"]
        assert request.columns[1]["editor"] == "choice"
        assert request.columns[1]["choices"] == ["usd", "usda"]
        assert request.columns[2]["editor"] == "bool"
        assert request.pathKey == "nodePath"
        assert request.rows == [{"nodePath": "/stage/rop1", "extension": "usda", "separate": True}]

        rows = [{"nodePath": "/stage/a", "extension": "usd", "separate": False, "parent": ""},
                {"nodePath": "/stage/b", "extension": "usda", "separate": True, "parent": "1"}]
        assert request.trigger("apply", rows)
        assert [c.nodePath.value for c in node.configs.value] == ["/stage/a", "/stage/b"]
        assert not request.trigger("unknown", rows)

    def test_callbackErrorsAreNotPropagated(self):
        node = Graph("").addNewNode(NodeWithDialogs.__name__)

        def failing(n, paths):
            raise RuntimeError("boom")

        request = desc.TreePickerRequest(node, ["/a"], onAccepted=failing)
        assert request.accept(["/a"]) is False
