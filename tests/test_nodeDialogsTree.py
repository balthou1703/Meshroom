from meshroom.ui.nodeDialogsTree import EntryHierarchy, PickerTree

ITEMS = [
    {"path": "/obj/geo/cache1", "type": "filecache::2.0"},
    {"path": "/obj/geo/cache2", "type": "filecache::2.0"},
    {"path": "/obj/fx/sim", "type": "vellumio::2.0"},
    {"path": "/out/abc", "type": "rop_alembic"},
]

COLUMNS = [
    {"key": "nodePath", "label": "Node Path", "editor": "text", "choices": [], "default": ""},
    {"key": "extension", "label": "Extension", "editor": "choice", "choices": ["bgeo.sc", "abc"], "default": "bgeo.sc"},
    {"key": "separate", "label": "Separate", "editor": "bool", "choices": [], "default": False},
]


class TestPickerTree:

    def test_structure(self):
        tree = PickerTree(ITEMS)
        rows = tree.rows()
        assert [r["key"] for r in rows] == ["/obj", "/obj/geo", "/obj/geo/cache1", "/obj/geo/cache2",
                                            "/obj/fx", "/obj/fx/sim", "/out", "/out/abc"]
        assert [r["depth"] for r in rows] == [0, 1, 2, 2, 1, 2, 0, 1]
        assert tree.leafCount == 4

    def test_folding(self):
        tree = PickerTree(ITEMS)
        tree.toggleExpanded("/obj")
        assert [r["key"] for r in tree.rows()] == ["/obj", "/out", "/out/abc"]

    def test_filter(self):
        tree = PickerTree(ITEMS)
        tree.setFilter("Vellum")
        assert [r["key"] for r in tree.rows()] == ["/obj", "/obj/fx", "/obj/fx/sim"]
        tree.setAllChecked(True)
        assert tree.checkedPaths() == ["/obj/fx/sim"]
        tree.setFilter("")
        assert tree.checkedPaths() == ["/obj/fx/sim"]

    def test_contextCheck(self):
        tree = PickerTree(ITEMS)
        tree.toggle("/obj/geo")
        assert tree.checkedPaths() == ["/obj/geo/cache1", "/obj/geo/cache2"]
        states = {r["key"]: r["checkState"] for r in tree.rows()}
        assert states["/obj/geo"] == 2 and states["/obj"] == 1 and states["/out"] == 0
        tree.toggle("/obj/geo")
        assert tree.checkedPaths() == []

    def test_range(self):
        tree = PickerTree(ITEMS)
        tree.setRangeChecked("/obj/geo/cache2", "/out/abc", True)
        assert tree.checkedPaths() == ["/obj/geo/cache2", "/obj/fx/sim", "/out/abc"]

    def test_singleSelectionAndPreselection(self):
        tree = PickerTree(ITEMS, singleSelection=True)
        tree.checkPaths(["cache2"])
        assert tree.checkedPaths() == ["/obj/geo/cache2"]
        tree.toggle("/out/abc")
        assert tree.checkedPaths() == ["/out/abc"]
        tree.setAllChecked(True)
        assert tree.checkedPaths() == ["/out/abc"]


class TestEntryHierarchy:

    def makeHierarchy(self):
        hierarchy = EntryHierarchy(COLUMNS, "nodePath")
        hierarchy.loadRows([
            {"nodePath": "a", "extension": "abc", "parent": ""},
            {"nodePath": "b", "parent": "1"},
            {"nodePath": "c", "parent": "2", "separate": "True"},
            {"nodePath": "d"},
        ])
        return hierarchy

    def paths(self, hierarchy):
        return [(r["values"]["nodePath"], r["depth"]) for r in hierarchy.rows()]

    def test_loadAndCollect(self):
        hierarchy = self.makeHierarchy()
        assert self.paths(hierarchy) == [("a", 0), ("b", 1), ("c", 2), ("d", 0)]
        rows = hierarchy.collectRows()
        assert [r["parent"] for r in rows] == ["", "1", "2", ""]
        assert rows[1]["extension"] == "bgeo.sc"  # Default value
        assert rows[2]["separate"] is True

    def test_loadInvalidParents(self):
        hierarchy = EntryHierarchy(COLUMNS, "nodePath")
        # Child before its parent, cycle, invalid index
        hierarchy.loadRows([{"nodePath": "x", "parent": "2"}, {"nodePath": "y"},
                            {"nodePath": "p", "parent": "4"}, {"nodePath": "q", "parent": "3"},
                            {"nodePath": "z", "parent": "42"}])
        assert self.paths(hierarchy) == [("y", 0), ("x", 1), ("z", 0), ("p", 0), ("q", 0)]
        rows = hierarchy.collectRows()
        assert [r["nodePath"] for r in rows][:2] == ["y", "x"] and rows[1]["parent"] == "1"

    def test_addPathsSkipsDuplicates(self):
        hierarchy = self.makeHierarchy()
        created = hierarchy.addPaths(["a", "e", "e"])
        assert len(created) == 1
        assert self.paths(hierarchy)[-1] == ("e", 0)

    def test_removeKeepsChildren(self):
        hierarchy = self.makeHierarchy()
        ids = hierarchy.ids()
        hierarchy.remove([ids[1]])
        assert self.paths(hierarchy) == [("a", 0), ("c", 1), ("d", 0)]

    def test_move(self):
        hierarchy = self.makeHierarchy()
        a, b, c, d = hierarchy.ids()
        assert not hierarchy.move(a, c, "inside")  # Into its own subtree
        assert hierarchy.move(d, a, "inside")
        assert self.paths(hierarchy) == [("a", 0), ("b", 1), ("c", 2), ("d", 1)]
        assert hierarchy.move(c, a, "before")
        assert self.paths(hierarchy) == [("c", 0), ("a", 0), ("b", 1), ("d", 1)]
        assert hierarchy.move(b, None, "after")
        assert self.paths(hierarchy) == [("c", 0), ("a", 0), ("d", 1), ("b", 0)]

    def test_moveManyKeepsOrder(self):
        hierarchy = self.makeHierarchy()
        a, b, c, d = hierarchy.ids()
        e = hierarchy.addEntry({"nodePath": "e"})
        assert hierarchy.moveMany([e, d], a, "after")
        assert self.paths(hierarchy) == [("a", 0), ("b", 1), ("c", 2), ("d", 0), ("e", 0)]
        assert not hierarchy.moveMany([a, d], c, "inside")

    def test_indentOutdentUnparent(self):
        hierarchy = self.makeHierarchy()
        a, b, c, d = hierarchy.ids()
        hierarchy.indent([d])
        assert self.paths(hierarchy) == [("a", 0), ("b", 1), ("c", 2), ("d", 1)]
        hierarchy.outdent([c])
        assert self.paths(hierarchy) == [("a", 0), ("b", 1), ("c", 1), ("d", 1)]
        hierarchy.unparent([b, c])
        assert self.paths(hierarchy) == [("a", 0), ("d", 1), ("b", 0), ("c", 0)]

    def test_shiftAndDuplicate(self):
        hierarchy = self.makeHierarchy()
        a, b, c, d = hierarchy.ids()
        hierarchy.shift([d], -1)
        assert self.paths(hierarchy) == [("d", 0), ("a", 0), ("b", 1), ("c", 2)]
        created = hierarchy.duplicate([b, c])
        assert len(created) == 1  # c is covered by b
        assert self.paths(hierarchy) == [("d", 0), ("a", 0), ("b", 1), ("c", 2), ("b", 1)]

    def test_collapse(self):
        hierarchy = self.makeHierarchy()
        a = hierarchy.ids()[0]
        hierarchy.toggleExpanded(a)
        assert self.paths(hierarchy) == [("a", 0), ("d", 0)]
        assert len(hierarchy.collectRows()) == 4
