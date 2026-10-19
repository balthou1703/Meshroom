import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Shapes

import Controls 1.0
import MaterialIcons 2.2
import Utils 1.0

/**
 * A component displaying a Graph (nodes, attributes and edges).
 */

Item {
    id: root

    property variant uigraph: null  /// Meshroom UI graph (UIGraph)
    readonly property variant graph: uigraph ? uigraph.graph : null  /// Core graph contained in the UI graph
    property variant nodeTypesModel: null  /// The list of node types that can be instantiated
    property real maxZoom: 2.0
    property real minZoom: 0.1
    property var edgeAboutToBeRemoved: undefined

    property var _attributeToDelegate: ({})

    // Signals
    signal workspaceMoved()
    signal workspaceClicked()

    signal nodeDoubleClicked(var mouse, var node)
    signal computeRequest(var nodes)
    signal submitRequest(var nodes)

    property int nbMeshroomScenes: 0
    property int nbDraggedFiles: 0
    signal filesDropped(var drop, var mousePosition)  // Files have been dropped

    // Trigger initial fit() after initialization
    // (ensure GraphEditor has its final size)
    Component.onCompleted: firstFitTimer.start()

    Timer {
        id: firstFitTimer
        running: false
        interval: 10
        onTriggered: fit()
    }

    clip: true

    SystemPalette { id: activePalette }

    /// Get node delegate for the given node object
    function nodeDelegate(node) {
        for(var i = 0; i < nodeRepeater.count; ++i) {
            if (nodeRepeater.getItemAt(i).node === node)
                return nodeRepeater.getItemAt(i)
        }
        return undefined
    }

    /// The visible pin of an attribute (or of its first visible parent group), or null
    function pinOf(attribute) {
        if (!attribute)
            return null
        let pin = root._attributeToDelegate[attribute.uuid]
        if (pin && pin.visible)
            return pin
        let group = attribute.root
        while (group) {
            let groupPin = root._attributeToDelegate[group.uuid]
            if (groupPin && groupPin.visible)
                return groupPin
            group = group.root
        }
        return pin || null
    }

    // Edges selected in the graph editor (click on an edge, or right drag a box over edges):
    // {destination attribute full name: true}. Delete removes them, H hides them (compositing nodes).
    property var selectedEdges: ({})
    readonly property bool hasSelectedEdges: Object.keys(selectedEdges).length > 0
    function selectEdges(names, mode) {
        // mode: "replace", "add" or "toggle"
        var selection = mode === "replace" ? {} : Object.assign({}, selectedEdges)
        names.forEach(function(name) {
            if (mode === "toggle" && selection[name])
                delete selection[name]
            else
                selection[name] = true
        })
        selectedEdges = selection
    }
    function clearEdgeSelection() {
        if (hasSelectedEdges)
            selectedEdges = {}
    }
    // The selected edges that still exist (an edge selected before an undo or a reconnection may be gone)
    function validSelectedEdges() {
        var names = []
        if (!root.graph)
            return names
        for (var i = 0; i < root.graph.edges.count; i++) {
            var edge = root.graph.edges.at(i)
            if (root.selectedEdges[edge.dst.fullName])
                names.push(edge.dst.fullName)
        }
        return names
    }
    // Selecting nodes, or another graph, unselects the edges (Delete then acts on the nodes)
    Connections {
        target: uigraph ? uigraph.nodeSelection : null
        // Only when nodes get selected (clearing the node selection, e.g. a click on the graph, keeps the edges)
        function onSelectionChanged() {
            if (uigraph.nodeSelection.hasSelection)
                root.clearEdgeSelection()
        }
    }
    onGraphChanged: {
        clearEdgeSelection()
        cancelPendingEdge()
    }
    // The pulled edge is dropped when its node is removed
    Connections {
        target: root.graph
        function onUpdated() {
            if (root.pendingSource) {
                var node = root.pendingSource.node
                if (!node || !node.alive)
                    root.cancelPendingEdge()
            }
        }
    }

    // Edge pulled from an output pin (a click on it): it follows the cursor; a click on an input connects
    // it, Tab creates a node connected to it, Escape or a click on an empty area cancels it
    property var pendingSource: null
    // Where the node context menu was asked (graph editor coordinates)
    property point menuPosition
    function cancelPendingEdge() { pendingSource = null }
    function handlePinClick(attribute) {
        if (root.pendingSource) {
            if (!attribute.isOutput) {
                uigraph.connectAttributes(root.pendingSource, attribute)
                root.pendingSource = null
            } else {
                // Another output: it becomes the pulled one (the same one: cancel)
                root.pendingSource = attribute === root.pendingSource ? null : attribute
            }
            return
        }
        if (attribute.isOutput && attribute.type !== "ListAttribute" && attribute.type !== "GroupAttribute") {
            root.pendingSource = attribute
            // Tab (create a connected node) and Escape go to the graph editor
            root.forceActiveFocus()
        }
    }
    // Cursor position in the graph ('draggable' coordinates)
    readonly property point cursorInGraph: {
        var p = graphHover.point.position
        draggable.x; draggable.y; draggable.scale  // Re-evaluated when the view moves
        return root.mapToItem(draggable, p.x, p.y)
    }

    // Hidden edges (H): the hidden edges leaving an output, for the tooltip of its stub
    function hiddenTargetsText(srcAttribute) {
        var lines = []
        if (!root.graph)
            return ""
        for (var i = 0; i < root.graph.edges.count; i++) {
            var edge = root.graph.edges.at(i)
            if (edge.src === srcAttribute && uigraph.isEdgeHidden(edge.dst))
                lines.push("\u2192 " + edge.dst.node.label + "." + edge.dst.label)
        }
        return lines.join("\n")
    }

    /// Get the InputNode delegate at the given position in 'draggable' coordinates, or null
    function inputNodeDelegateAt(draggableX, draggableY) {
        for (var i = 0; i < nodeRepeater.count; ++i) {
            var delegate = nodeRepeater.getItemAt(i)
            if (delegate && delegate.visible && delegate.node && delegate.node.isInputNode && !delegate.readOnly) {
                const delegateY = delegate.y - (delegate.foldShift || 0)
                if (draggableX >= delegate.x && draggableX <= delegate.x + delegate.width &&
                    draggableY >= delegateY && draggableY <= delegateY + delegate.height) {
                    return delegate
                }
            }
        }
        return null
    }

    /// Duplicate a node and optionally all the following ones
    function duplicateNode(duplicateFollowingNodes) {
        var nodes
        if (duplicateFollowingNodes) {
            nodes = uigraph.duplicateNodesFrom(uigraph.getSelectedNodes())
        } else {
            nodes = uigraph.duplicateNodes(uigraph.getSelectedNodes())
        }
        uigraph.selectedNode = nodes[0]
        uigraph.selectNodes(nodes)
    }

    /// Copy node content to clipboard
    function copyNodes() {
        var nodeContent = uigraph.getSelectedNodesContent()
        if (nodeContent !== '') {
            Clipboard.clear()
            Clipboard.setText(nodeContent)
        }
    }

    /// Paste content of clipboard to graph editor and create new node if valid
    function pasteNodes() {
        let finalPosition = undefined
        if (mouseArea.containsMouse) {
            finalPosition = toGraphPosition(mapToItem(draggable, mouseArea.mouseX, mouseArea.mouseY))
        } else {
            finalPosition = getCenterPosition()
        }

        const copiedContent = Clipboard.getText()
        const nodes = uigraph.pasteNodes(copiedContent, finalPosition)
        if (nodes.length > 0) {
            uigraph.selectedNode = nodes[0]
            uigraph.selectNodes(nodes)
        }
    }

    /// Get the coordinates of the point at the center of the GraphEditor
    function getCenterPosition() {
        return toGraphPosition(mapToItem(draggable, mouseArea.width / 2, mouseArea.height / 2))
    }

    /// Graph position of a point of the 'draggable' item: positions below folded chapters are displayed moved up
    function toGraphPosition(point) {
        return uigraph ? Qt.point(point.x, uigraph.unfoldY(point.y)) : point
    }

    // Compositing shortcuts, active in the compositing workspace while the mouse is over the graph editor,
    // whatever has the keyboard focus (as in Nuke). Text fields keep the keys typed in them.
    HoverHandler {
        id: graphHover
    }
    readonly property bool compositingShortcuts: graphHover.hovered && !!uigraph && uigraph.graph.workspaceMode === "compositing"
    Shortcut {
        // Live compositing: disable / enable the selected nodes
        sequence: "I"
        enabled: root.compositingShortcuts
        onActivated: uigraph.toggleDisableSelectedNodes()
    }
    Shortcut {
        // Live compositing: display the result of the selected node
        sequence: "E"
        enabled: root.compositingShortcuts
        onActivated: {
            if (uigraph.selectedNode && uigraph.selectedNode.isImageOpNode)
                uigraph.live.evaluate(uigraph.selectedNode)
        }
    }
    // Live compositing: view 1 (Evaluate) / view 2 (Evaluate Compare), as the inputs of the Nuke viewer
    Shortcut {
        sequences: ["1", "Num+1"]
        enabled: root.compositingShortcuts && !!uigraph.live.compareNode
        onActivated: uigraph.live.showView(1)
    }
    Shortcut {
        sequences: ["2", "Num+2"]
        enabled: root.compositingShortcuts && !!uigraph.live.compareNode
        onActivated: uigraph.live.showView(2)
    }
    Shortcut {
        // Hide the selected edges (or the input edges of the selected nodes), or show them if they are hidden:
        // a hidden edge leaves a stub on both pins, its tooltip giving the other end
        sequence: "H"
        enabled: root.compositingShortcuts
        onActivated: {
            if (root.validSelectedEdges().length > 0) {
                uigraph.toggleHiddenEdges(root.validSelectedEdges())
                root.clearEdgeSelection()
            } else {
                uigraph.toggleHiddenInputEdgesOfSelectedNodes()
            }
        }
    }
    Shortcut {
        sequence: "Escape"
        enabled: root.pendingSource !== null && graphHover.hovered
        onActivated: root.cancelPendingEdge()
    }
    Shortcut {
        // Connect the output of the first selected node to the inputs of the other selected nodes
        sequence: "&"
        enabled: root.compositingShortcuts
        onActivated: uigraph.connectSelectedNodes()
    }

    Keys.onPressed: function(event) {
        if (event.key === Qt.Key_F) {
            fit()
        } else if (event.key === Qt.Key_Escape && root.pendingSource) {
            root.cancelPendingEdge()
        } else if (event.key === Qt.Key_Delete && root.validSelectedEdges().length > 0) {
            // Selected edges first (the nodes stay)
            uigraph.removeEdgesTo(root.validSelectedEdges())
            root.clearEdgeSelection()
        } else if (event.key === Qt.Key_Delete) {
            if (event.modifiers === Qt.AltModifier) {
                uigraph.removeNodesFrom(uigraph.getSelectedNodes())
            } else {
                uigraph.removeSelectedNodes()
            }
        } else if (event.key === Qt.Key_D) {
            duplicateNode(event.modifiers === Qt.AltModifier)
        } else if (event.key === Qt.Key_X) {
            if (event.modifiers === Qt.ControlModifier) {
                copyNodes()
                uigraph.removeSelectedNodes()
            }
            else {
                uigraph.disconnectSelectedNodes()
            }
        } else if (event.key === Qt.Key_C) {
            if (event.modifiers === Qt.ControlModifier) {
                copyNodes()
            }
            else {
                colorSelector.toggle()
            }
        } else if (event.key === Qt.Key_V && event.modifiers === Qt.ControlModifier) {
            pasteNodes()
        } else if (event.key === Qt.Key_V && event.modifiers === Qt.ShiftModifier) {
            uigraph.alignVertically()
        } else if (event.key === Qt.Key_H && event.modifiers === Qt.ShiftModifier) {
            uigraph.alignHorizontally()
        } else if (event.key === Qt.Key_L && event.modifiers === Qt.NoModifier) {
            // Layout following the connections: the selected nodes, or the whole graph
            var sizes = {}
            for (var i = 0; i < nodeRepeater.count; ++i) {
                var item = nodeRepeater.getItemAt(i)
                if (item && item.node)
                    sizes[item.node.name] = [item.width, item.height]
            }
            uigraph.layoutNodes(sizes)
        } else if (event.key === Qt.Key_Tab) {
            event.accepted = true
            if (mouseArea.containsMouse) {
                newNodeMenu.spawnPosition = toGraphPosition(mouseArea.mapToItem(draggable, mouseArea.mouseX, mouseArea.mouseY))
                newNodeMenu.popup()
            }
        }
    }

    MouseArea {
        id: mouseArea
        anchors.fill: parent
        property double factor: 1.15
        property bool removingEdges: false
        // Activate multisampling for edges antialiasing
        layer.enabled: true
        layer.samples: 8

        hoverEnabled: true
        acceptedButtons: Qt.LeftButton | Qt.RightButton | Qt.MiddleButton
        drag.threshold: 0
        drag.smoothed: false
        cursorShape: drag.target == draggable ? Qt.ClosedHandCursor : removingEdges ? Qt.CrossCursor : Qt.ArrowCursor

        onWheel: function(wheel) {
            var zoomFactor = wheel.angleDelta.y > 0 ? factor : 1 / factor
            var scale = draggable.scale * zoomFactor
            scale = Math.min(Math.max(minZoom, scale), maxZoom)
            if (draggable.scale == scale)
                return
            var point = mapToItem(draggable, wheel.x, wheel.y)
            draggable.x += (1 - zoomFactor) * point.x * draggable.scale
            draggable.y += (1 - zoomFactor) * point.y * draggable.scale
            draggable.scale = scale
            workspaceMoved()
        }

        // Box selection of edges: right button drag on an empty area
        property point edgeBoxStart
        property bool edgeBoxActive: false
        // The last right press made a box (no menu on its release)
        property bool edgeBoxWasUsed: false
        function edgeBoxUsed() { return edgeBoxWasUsed }

        onPressed: function(mouse) {
            if (root.pendingSource && mouse.button == Qt.LeftButton) {
                // A click on an empty area cancels the edge pulled from an output
                root.cancelPendingEdge()
            }
            if (mouse.button != Qt.MiddleButton && mouse.modifiers == Qt.NoModifier) {
                uigraph.clearNodeSelection()
                root.clearEdgeSelection()
            }
            if (mouse.button == Qt.RightButton || mouse.button == Qt.LeftButton) {
                edgeBoxStart = Qt.point(mouse.x, mouse.y)
                edgeBoxActive = false
            }
            if (mouse.button == Qt.LeftButton && (mouse.modifiers == Qt.NoModifier || mouse.modifiers & (Qt.ControlModifier | Qt.ShiftModifier))) {
                nodeSelectionBox.startSelection(mouse)
            }
            if (mouse.button == Qt.MiddleButton || (mouse.button == Qt.LeftButton && mouse.modifiers & Qt.AltModifier)) {
                drag.target = draggable // start drag
            }
            if (mouse.button == Qt.LeftButton && (mouse.modifiers & Qt.ControlModifier) && (mouse.modifiers & Qt.AltModifier)) {
                edgeSelectionLine.startSelection(mouse)
                removingEdges = true
            }
        }

        onReleased: function(mouse) {
            edgeBoxWasUsed = edgeBoxActive
            if (edgeBoxActive) {
                selectEdgesInBox(mouse.modifiers)
                edgeBoxActive = false
            }
            removingEdges = false
            edgeSelectionLine.endSelection()
            var nodeBox = nodeSelectionBox.active && mouse.button == Qt.LeftButton
            nodeSelectionBox.endSelection()
            // A selection box (left button) around no node selects the edges it crosses
            if (nodeBox && !uigraph.nodeSelection.hasSelection &&
                Math.abs(mouse.x - edgeBoxStart.x) + Math.abs(mouse.y - edgeBoxStart.y) > 6) {
                edgeBox.setEnd(mouse.x, mouse.y)
                selectEdgesInBox(mouse.modifiers)
            }
            drag.target = null
            root.forceActiveFocus()
            workspaceClicked()
        }

        onPositionChanged: function(mouse) {
            if (drag.active)
                workspaceMoved()
            if ((mouse.buttons & Qt.RightButton) && !edgeBoxActive &&
                Math.abs(mouse.x - edgeBoxStart.x) + Math.abs(mouse.y - edgeBoxStart.y) > 6) {
                edgeBox.setEnd(mouse.x, mouse.y)
                edgeBoxActive = true
            }
            if (edgeBoxActive)
                edgeBox.setEnd(mouse.x, mouse.y)
        }

        // Select the edges crossing the box (Shift: add, Ctrl: toggle)
        function selectEdgesInBox(modifiers) {
            var a = mapToItem(draggable, Math.min(edgeBoxStart.x, edgeBox.endX), Math.min(edgeBoxStart.y, edgeBox.endY))
            var b = mapToItem(draggable, Math.max(edgeBoxStart.x, edgeBox.endX), Math.max(edgeBoxStart.y, edgeBox.endY))
            var corners = [Qt.point(a.x, a.y), Qt.point(b.x, a.y), Qt.point(b.x, b.y), Qt.point(a.x, b.y)]
            var names = []
            for (var i = 0; i < edgesRepeater.count; i++) {
                var item = edgesRepeater.itemAt(i)
                if (!item || !item.visible || !item.edge)
                    continue
                var inside = function(x, y) { return x >= a.x && x <= b.x && y >= a.y && y <= b.y }
                var hit = inside(item.point1x, item.point1y) || inside(item.point2x, item.point2y)
                for (var k = 0; k < 4 && !hit; k++)
                    hit = item.intersectsSegment(corners[k], corners[(k + 1) % 4])
                if (hit)
                    names.push(item.edge.dst.fullName)
            }
            root.selectEdges(names, (modifiers & Qt.ControlModifier) ? "toggle" : (modifiers & Qt.ShiftModifier) ? "add" : "replace")
        }

        // Box of the edge selection
        Rectangle {
            id: edgeBox
            property real endX: 0
            property real endY: 0
            function setEnd(x, y) {
                endX = x
                endY = y
            }
            visible: mouseArea.edgeBoxActive
            x: Math.min(mouseArea.edgeBoxStart.x, endX)
            y: Math.min(mouseArea.edgeBoxStart.y, endY)
            width: Math.abs(endX - mouseArea.edgeBoxStart.x)
            height: Math.abs(endY - mouseArea.edgeBoxStart.y)
            z: 10
            color: Qt.rgba(1, 0.65, 0.2, 0.08)
            border.color: "#F5A623"
            border.width: 1
        }

        onCanceled: {
            edgeBoxActive = false
            edgeBoxWasUsed = false
        }

        onClicked: function(mouse) {
            if (mouse.button == Qt.RightButton && edgeBox.visible === false && !mouseArea.edgeBoxUsed()) {
                // Store mouse click position in 'draggable' coordinates as new node spawn position
                newNodeMenu.spawnPosition = toGraphPosition(mouseArea.mapToItem(draggable, mouse.x, mouse.y))
                newNodeMenu.popup()
            }
        }

        // Contextual Menu for creating new nodes
        // TODO: add filtering + validate on 'Enter'
        Menu {
            id: newNodeMenu
            property point spawnPosition
            property variant menuKeys: Object.keys(root.nodeTypesModel).concat(Object.values(MeshroomApp.pipelineTemplateNames))
            height: searchBar.height + nodeMenuRepeater.height + instantiator.height

            function createNode(nodeType) {
                // "nodeType" might be a pipeline (artificially added in the "Pipelines" category) instead of a node
                // If it is not a pipeline to import, then it must be a node
                if (!importPipeline(nodeType)) {
                    // Add node via the proper command in uigraph (connected to the pulled edge, if any)
                    // Compositing: a node created while another one is selected comes after it (connected, on its right),
                    // so that nodes created one after the other are chained (as in Nuke)
                    var previous = uigraph.nodeSelection.hasSelection && uigraph.selectedNode && uigraph.selectedNode.isImageOpNode &&
                                   root.nodeTypesModel[nodeType] && uigraph.graph.workspaceMode === "compositing" ? uigraph.selectedNode : null
                    var node = root.pendingSource ? uigraph.addNewNodeConnectedTo(nodeType, spawnPosition, root.pendingSource)
                             : previous ? uigraph.addNewNodeAfter(nodeType, previous)
                             : uigraph.addNewNode(nodeType, spawnPosition)
                    root.cancelPendingEdge()
                    uigraph.selectedNode = node
                    uigraph.selectNodes([node])
                }
                close()
            }

            function importPipeline(pipeline) {
                if (MeshroomApp.pipelineTemplateNames.includes(pipeline)) {
                    var url = MeshroomApp.pipelineTemplateFiles[MeshroomApp.pipelineTemplateNames.indexOf(pipeline)]["path"]
                    var nodes = uigraph.importProject(Filepath.stringToUrl(url), spawnPosition)
                    uigraph.selectedNode = nodes[0]
                    uigraph.selectNodes(nodes)
                    return true
                }
                return false
            }

            function parseCategories() {
                // Organize nodes based on their category
                // {"category1": ["node1", "node2"], "category2": ["node3", "node4"]}
                let categories = {}
                for (const [name, data] of Object.entries(root.nodeTypesModel)) {
                    let category = data["category"]
                    if (categories[category] === undefined) {
                        categories[category] = []
                    }
                    categories[category].push(name)
                }

                // Add a "Pipelines" category, filled with the list of templates to create pipelines from the menu
                if (MeshroomApp.pipelineTemplateNames.length > 0) {
                    categories["Pipelines"] = MeshroomApp.pipelineTemplateNames
                }

                return categories
            }

            onVisibleChanged: {
                searchBar.clear()
                if (visible) {
                    // When menu is shown, give focus to the TextField filter
                    searchBar.forceActiveFocus()
                }
            }

            SearchBar {
                id: searchBar
                width: parent.width
            }

            // menuItemDelegate is wrapped in a component so it can be used in both the search bar and sub-menus
            Component {
                id: menuItemDelegateComponent
                MenuItem {
                    id: menuItemDelegate
                    font.pointSize: 8
                    padding: 3

                    // Hide items that does not match the filter text
                    visible: modelData.toLowerCase().indexOf(searchBar.text.toLowerCase()) > -1
                    text: modelData
                    // Forward key events to the search bar to continue typing seamlessly
                    // even if this delegate took the activeFocus due to mouse hovering
                    Keys.forwardTo: [searchBar.textField]
                    Keys.onPressed: function(event) {
                        event.accepted = false
                        switch (event.key) {
                            case Qt.Key_Return:
                            case Qt.Key_Enter:
                                // Create node on validation (Enter/Return keys)
                                newNodeMenu.createNode(modelData)
                                event.accepted = true
                                break
                            case Qt.Key_Up:
                            case Qt.Key_Down:
                            case Qt.Key_Left:
                            case Qt.Key_Right:
                                break  // Ignore if arrow key was pressed to let the menu be controlled
                            default:
                                searchBar.forceActiveFocus()
                        }
                    }
                    // Set the priority ordering of the keys to be Item's own Key Handling > ForwardTo
                    Keys.priority: Keys.AfterItem
                    // Create node on mouse click
                    onClicked: newNodeMenu.createNode(modelData)

                    states: [
                        State {
                            // Additional property setting when the MenuItem is not visible
                            when: !visible
                            name: "invisible"
                            PropertyChanges {
                                target: menuItemDelegate
                                height: 0  // Make sure the item is no visible by setting height to 0
                                focusPolicy: Qt.NoFocus  // Don't grab focus when not visible
                            }
                        }
                    ]
                }
            }

            Repeater {
                id: nodeMenuRepeater
                model: searchBar.text !== "" ? Object.values(newNodeMenu.menuKeys) : undefined

                // Create Menu items from available items
                delegate: menuItemDelegateComponent
            }

            // Dynamically add the menu categories
            Instantiator {
                id: instantiator
                model: (searchBar.text === "") ? Object.keys(newNodeMenu.parseCategories()).sort() : undefined
                onObjectAdded: function(index, object) {
                    // Add sub-menu under the search bar
                    newNodeMenu.insertMenu(index + 1, object)
                }
                onObjectRemoved: function(index, object) {
                    newNodeMenu.removeMenu(object)
                }

                delegate: Menu {
                    title: modelData
                    id: newNodeSubMenu

                    Instantiator {
                        model: newNodeMenu.visible ? newNodeMenu.parseCategories()[modelData] : undefined
                        onObjectAdded: function(index, object) {
                            newNodeSubMenu.insertItem(index, object)
                        }
                        onObjectRemoved: function(index, object) {
                            newNodeSubMenu.removeItem(object)
                        }
                        delegate: menuItemDelegateComponent
                    }
                }
            }
        }

        // Informative contextual menu when graph is read-only
        Menu {
            id: lockedMenu
            MenuItem {
                id: item
                font.pointSize: 8
                enabled: false
                text: "Computing - Graph is Locked!"
            }
        }

        Canvas {
            id: templateBackgroundStripes

            anchors.fill: parent
            visible: Boolean(root.graph && root.graph.templateFilepath)
            opacity: 0.03

            readonly property color stripeColor: activePalette.highlight
            readonly property int stripeWidth: 21
            readonly property int stripeGap: 42

            onPaint: {
                const ctx = getContext("2d")
                ctx.clearRect(0, 0, width, height)

                if (!visible) {
                    return
                }

                ctx.save()
                ctx.strokeStyle = stripeColor
                ctx.lineWidth = stripeWidth

                const diagonal = Math.sqrt(width * width + height * height)
                const step = stripeWidth + stripeGap
                ctx.translate(width * 0.5, height * 0.5)
                ctx.rotate(-Math.PI / 4)

                for (let x = -diagonal; x <= diagonal; x += step) {
                    ctx.beginPath()
                    ctx.moveTo(x, -diagonal)
                    ctx.lineTo(x, diagonal)
                    ctx.stroke()
                }

                ctx.restore()
            }

            onVisibleChanged: requestPaint()
            onWidthChanged: requestPaint()
            onHeightChanged: requestPaint()
            onStripeColorChanged: requestPaint()
        }

        Item {
            id: draggable
            transformOrigin: Item.TopLeft
            width: 1000
            height: 1000

            Popup {
                id: edgeMenu
                property var currentEdge: null
                property bool forLoop: false

                onOpened: {
                    expandButton.canExpand = uigraph.canExpandForLoop(edgeMenu.currentEdge)
                }

                contentItem: Row {
                    IntSelector {
                        id: loopIterationSelector
                        tooltipText: "Iterations"
                        visible: edgeMenu.currentEdge && edgeMenu.forLoop

                        enabled: expandButton.canExpand

                        property var listAttr: edgeMenu.currentEdge ? edgeMenu.currentEdge.src.root : null

                        Connections {
                            target: edgeMenu
                            function onCurrentEdgeChanged() {
                                if (edgeMenu.currentEdge) {
                                    loopIterationSelector.listAttr = edgeMenu.currentEdge.src.root
                                    loopIterationSelector.value = loopIterationSelector.listAttr ? loopIterationSelector.listAttr.value.indexOf(edgeMenu.currentEdge.src) + 1 : 0
                                }
                            }
                        }

                        // We add 1 to the index because of human readable index (starting at 1)
                        value: listAttr ? listAttr.value.indexOf(edgeMenu.currentEdge.src) + 1 : 0
                        range: { "min": 1, "max": listAttr ? listAttr.value.count : 0 }

                        onValueChanged: {
                            if (listAttr === null) {
                                return
                            }
                            const newSrcAttr = listAttr.value.at(value - 1)
                            const dst = edgeMenu.currentEdge.dst

                            // If the edge exists, do not replace it
                            if (newSrcAttr === edgeMenu.currentEdge.src && dst === edgeMenu.currentEdge.dst) {
                                return
                            }
                            edgeMenu.currentEdge = uigraph.replaceEdge(edgeMenu.currentEdge, newSrcAttr, dst)
                        }
                    }

                    MaterialToolButton {
                        font.pointSize: 13
                        ToolTip.text: "Remove Edge"
                        enabled: edgeMenu.currentEdge && !edgeMenu.currentEdge.dst.node.locked && !edgeMenu.currentEdge.dst.isReadOnly
                        text: MaterialIcons.delete_
                        onClicked: {
                            uigraph.removeEdge(edgeMenu.currentEdge)
                            edgeMenu.close()
                        }
                    }

                    MaterialToolButton {
                        id: expandButton

                        property bool canExpand: edgeMenu.currentEdge && edgeMenu.forLoop

                        visible: edgeMenu.currentEdge && edgeMenu.forLoop && canExpand
                        enabled: edgeMenu.currentEdge && !edgeMenu.currentEdge.dst.node.locked && !edgeMenu.currentEdge.dst.isReadOnly
                        font.pointSize: 13
                        ToolTip.text: "Expand"
                        text: MaterialIcons.open_in_full

                        onClicked: {
                            edgeMenu.currentEdge = uigraph.expandForLoop(edgeMenu.currentEdge)
                            canExpand = false
                            edgeMenu.close()
                        }
                    }

                    MaterialToolButton {
                        id: collapseButton

                        visible: edgeMenu.currentEdge && edgeMenu.forLoop && !expandButton.canExpand
                        enabled: edgeMenu.currentEdge && !edgeMenu.currentEdge.dst.node.locked && !edgeMenu.currentEdge.dst.isReadOnly
                        font.pointSize: 13
                        ToolTip.text: "Collapse"
                        text: MaterialIcons.close_fullscreen

                        onClicked: {
                            uigraph.collapseForLoop(edgeMenu.currentEdge)
                            expandButton.canExpand = true
                            edgeMenu.close()
                        }
                    }
                }
            }

            // Hidden links of the Clones: shown (dashed) while the clone or the cloned node is selected
            Repeater {
                model: nodeRepeater.loaded && root.graph ? root.graph.nodes : undefined
                delegate: Loader {
                    active: object.nodeType === "Clone" && object.hasAttribute("source")
                    sourceComponent: Shape {
                        id: cloneLink
                        readonly property var clone: object
                        readonly property var reference: {
                            var source = clone.attribute("source")
                            return source.isLink && source.inputLink ? source.inputLink.node : null
                        }
                        readonly property var cloneItem: { nodeRepeater.count; return root.nodeDelegate(clone) }
                        readonly property var referenceItem: { nodeRepeater.count; return reference ? root.nodeDelegate(reference) : null }
                        visible: !!cloneItem && !!referenceItem && cloneItem.visible && referenceItem.visible &&
                                 (uigraph.selectedNode === clone || uigraph.selectedNode === reference)
                        readonly property real x1: referenceItem ? referenceItem.x + referenceItem.width : 0
                        readonly property real y1: referenceItem ? referenceItem.y - (referenceItem.foldShift || 0) + 14 : 0
                        readonly property real x2: cloneItem ? cloneItem.x : 0
                        readonly property real y2: cloneItem ? cloneItem.y - (cloneItem.foldShift || 0) + 14 : 0
                        ShapePath {
                            strokeColor: "#C9C9C9"
                            strokeWidth: 1.5
                            strokeStyle: ShapePath.DashLine
                            dashPattern: [4, 3]
                            fillColor: "transparent"
                            startX: cloneLink.x1
                            startY: cloneLink.y1
                            PathCubic {
                                x: cloneLink.x2
                                y: cloneLink.y2
                                control1X: cloneLink.x1 + Math.max(40, Math.abs(cloneLink.x2 - cloneLink.x1) / 2)
                                control1Y: cloneLink.y1
                                control2X: cloneLink.x2 - Math.max(40, Math.abs(cloneLink.x2 - cloneLink.x1) / 2)
                                control2Y: cloneLink.y2
                            }
                        }
                    }
                }
            }

            // Menu of the stubs of hidden edges: copy the output or the input (to connect it elsewhere with the
            // "Connect" item of a pin menu), or show the edges again
            Menu {
                id: stubMenu
                property var edge: null
                property var names: []
                function openFor(edge, names, item) {
                    stubMenu.edge = edge
                    stubMenu.names = names
                    parent = item
                    popup()
                }
                MenuItem {
                    text: stubMenu.edge ? "Copy Output (" + stubMenu.edge.src.node.label + "." + stubMenu.edge.src.label + ")" : ""
                    onTriggered: uigraph.copyAttribute(stubMenu.edge.src)
                }
                MenuItem {
                    text: stubMenu.edge ? "Copy Input (" + stubMenu.edge.dst.node.label + "." + stubMenu.edge.dst.label + ")" : ""
                    onTriggered: uigraph.copyAttribute(stubMenu.edge.dst)
                }
                MenuSeparator {}
                MenuItem {
                    text: stubMenu.names.length > 1 ? "Show the Edges" : "Show the Edge"
                    onTriggered: uigraph.toggleHiddenEdges(stubMenu.names)
                }
            }

            // Stubs of the hidden edges (H): a pill next to each pin, colored as the node at the other end;
            // its tooltip gives the other end, a click selects the other node(s)
            Repeater {
                model: nodeRepeater.loaded && root.graph ? root.graph.edges : undefined
                delegate: Item {
                    id: stub
                    readonly property var edge: object
                    readonly property bool hidden: {
                        if (!edge || !edge.dst || !edge.dst.node.hasInternalAttribute("hiddenInputs"))
                            return false
                        edge.dst.node.internalAttribute("hiddenInputs").value
                        return uigraph.isEdgeHidden(edge.dst)
                    }
                    readonly property var srcPin: { root._attributeToDelegate; return hidden ? root.pinOf(edge.src) : null }
                    readonly property var dstPin: { root._attributeToDelegate; return hidden ? root.pinOf(edge.dst) : null }
                    readonly property bool selected: hidden && !!root.selectedEdges[edge.dst.fullName]
                    visible: hidden && !!srcPin && !!dstPin && srcPin.visible && dstPin.visible

                    function nodeColor(node) { return node.color !== "" ? node.color : "#8A8A8A" }
                    // The hidden edges leaving the same output (destination full names)
                    function hiddenFromSource() {
                        var names = []
                        for (var i = 0; i < root.graph.edges.count; i++) {
                            var e = root.graph.edges.at(i)
                            if (e.src === stub.edge.src && uigraph.isEdgeHidden(e.dst))
                                names.push(e.dst.fullName)
                        }
                        return names
                    }

                    // At the input: the source
                    Rectangle {
                        x: stub.dstPin ? stub.dstPin.globalX + stub.dstPin.inputAnchorPos.x - width - 6 : 0
                        y: stub.dstPin ? stub.dstPin.globalY + stub.dstPin.inputAnchorPos.y - height / 2 : 0
                        width: 16
                        height: 8
                        radius: 4
                        color: stub.edge ? stub.nodeColor(stub.edge.src.node) : "grey"
                        border.color: stub.selected ? "#F5A623" : (dstStubMouse.containsMouse ? "white" : Qt.rgba(0, 0, 0, 0.5))
                        border.width: stub.selected || dstStubMouse.containsMouse ? 2 : 1
                        MouseArea {
                            id: dstStubMouse
                            anchors.fill: parent
                            anchors.margins: -3
                            hoverEnabled: true
                            acceptedButtons: Qt.LeftButton | Qt.RightButton
                            // Double-click: the edge is shown again
                            onDoubleClicked: uigraph.toggleHiddenEdges([stub.edge.dst.fullName])
                            onPressed: function(mouse) {
                                if (mouse.button === Qt.RightButton)
                                    stubMenu.openFor(stub.edge, [stub.edge.dst.fullName], dstStubMouse)
                            }
                            ToolTip.visible: containsMouse
                            ToolTip.delay: 300
                            ToolTip.text: stub.edge ? "\u2190 " + stub.edge.src.node.label + "." + stub.edge.src.label +
                                                      "\n(hidden edge: double-click to show it, right click to copy)" : ""
                            onClicked: function(mouse) {
                                if (mouse.button !== Qt.LeftButton)
                                    return
                                root.selectEdges([stub.edge.dst.fullName], (mouse.modifiers & Qt.ShiftModifier) ? "add" : "replace")
                                root.forceActiveFocus()
                            }
                        }
                    }
                    // At the output: the destinations
                    Rectangle {
                        x: stub.srcPin ? stub.srcPin.globalX + stub.srcPin.outputAnchorPos.x + 6 : 0
                        y: stub.srcPin ? stub.srcPin.globalY + stub.srcPin.outputAnchorPos.y - height / 2 : 0
                        width: 16
                        height: 8
                        radius: 4
                        color: stub.edge ? stub.nodeColor(stub.edge.dst.node) : "grey"
                        border.color: stub.selected ? "#F5A623" : (srcStubMouse.containsMouse ? "white" : Qt.rgba(0, 0, 0, 0.5))
                        border.width: stub.selected || srcStubMouse.containsMouse ? 2 : 1
                        MouseArea {
                            id: srcStubMouse
                            anchors.fill: parent
                            anchors.margins: -3
                            hoverEnabled: true
                            acceptedButtons: Qt.LeftButton | Qt.RightButton
                            // Double-click: the hidden edges of this output are shown again
                            onDoubleClicked: uigraph.toggleHiddenEdges(stub.hiddenFromSource())
                            onPressed: function(mouse) {
                                if (mouse.button === Qt.RightButton)
                                    stubMenu.openFor(stub.edge, stub.hiddenFromSource(), srcStubMouse)
                            }
                            ToolTip.visible: containsMouse
                            ToolTip.delay: 300
                            ToolTip.text: containsMouse && stub.edge ? root.hiddenTargetsText(stub.edge.src) +
                                                                      "\n(hidden edges: double-click to show them, right click to copy)" : ""
                            onClicked: function(mouse) {
                                if (mouse.button !== Qt.LeftButton)
                                    return
                                root.selectEdges(stub.hiddenFromSource(), (mouse.modifiers & Qt.ShiftModifier) ? "add" : "replace")
                                root.forceActiveFocus()
                            }
                        }
                    }
                }
            }

            // Edge pulled from an output (click on an output pin), to the cursor
            Edge {
                id: pendingEdge
                readonly property var pin: { root._attributeToDelegate; return root.pendingSource ? root.pinOf(root.pendingSource) : null }
                visible: !!pin && pin.visible
                enabled: false  // No mouse interaction: clicks go to the pins and the graph
                z: 50
                point1x: pin ? pin.globalX + pin.outputAnchorPos.x : 0
                point1y: pin ? pin.globalY + pin.outputAnchorPos.y : 0
                point2x: root.cursorInGraph.x
                point2y: root.cursorInGraph.y
                color: activePalette.highlight
                thickness: 2
            }

            // Edges
            Repeater {
                id: edgesRepeater

                // Delay edges loading after nodes (edges needs attribute pins to be created)
                model: nodeRepeater.loaded && root.graph ? root.graph.edges : undefined

                delegate: Edge {
                    property var src: { root._attributeToDelegate; return root.pinOf(edge.src) }
                    property var dst: { root._attributeToDelegate; return root.pinOf(edge.dst) }
                    // Hidden with H (compositing nodes): replaced by stubs on its pins
                    readonly property bool hidden: {
                        if (!edge || !edge.dst || !edge.dst.node.hasInternalAttribute("hiddenInputs"))
                            return false
                        edge.dst.node.internalAttribute("hiddenInputs").value  // Re-evaluated when it changes
                        return uigraph.isEdgeHidden(edge.dst)
                    }
                    readonly property bool edgeSelected: !!edge && !!edge.dst && !!root.selectedEdges[edge.dst.fullName]
                    property bool isValidEdge: src != null && dst != null
                    visible: isValidEdge && src.visible && dst.visible && !hidden

                    property var loopSource: edge.src && edge.src.root && edge.src.root.value ? edge.src.root : null
                    property bool forLoop: src != null && dst != null &&
                                           loopSource != null &&
                                           src.attribute.type === "ListAttribute" &&
                                           dst.attribute.type != "ListAttribute"

                    property bool inFocus: containsMouse || (edgeMenu.opened && edgeMenu.currentEdge === edge)

                    edge: object
                    isForLoop: forLoop
                    loopSize: forLoop ? loopSource.value.count : 0

                    iteration: forLoop ? loopSource.value.indexOf(edge.src) : 0
                    color: edge.dst === root.edgeAboutToBeRemoved ? "red" : edgeSelected ? "#F5A623" : inFocus ? activePalette.highlight : activePalette.text
                    thickness: {
                        if (forLoop) {
                            return (inFocus) ? 4 : 3
                        }
                        return (inFocus || edgeSelected) ? 2 : 1
                    }
                    point1x: isValidEdge ? src.globalX + src.outputAnchorPos.x : 0
                    point1y: isValidEdge ? src.globalY + src.outputAnchorPos.y : 0
                    point2x: isValidEdge ? dst.globalX + dst.inputAnchorPos.x : 0
                    point2y: isValidEdge ? dst.globalY + dst.inputAnchorPos.y : 0
                    onPressed: function(event) {
                        const canEdit = !edge.dst.node.locked

                        if (event.button) {
                            if (canEdit && (event.modifiers & Qt.AltModifier)) {
                                uigraph.removeEdge(edge)
                            } else if (event.button == Qt.LeftButton) {
                                // Select the edge (Shift: add, Ctrl: toggle); Delete removes the selected edges, H hides them
                                root.selectEdges([edge.dst.fullName], (event.modifiers & Qt.ControlModifier) ? "toggle" :
                                                                       (event.modifiers & Qt.ShiftModifier) ? "add" : "replace")
                                root.forceActiveFocus()
                            } else if (event.button == Qt.RightButton) {
                                edgeMenu.currentEdge = edge
                                edgeMenu.forLoop = forLoop
                                var spawnPosition = mouseArea.mapToItem(draggable, mouseArea.mouseX, mouseArea.mouseY)
                                edgeMenu.x = spawnPosition.x
                                edgeMenu.y = spawnPosition.y
                                edgeMenu.open()
                            }
                        }
                    }
                }
            }

            Loader {
                id: nodeMenuLoader
                property var currentNode: null
                active: currentNode != null
                sourceComponent: nodeMenuComponent

                function load(node) {
                    currentNode = node
                }

                function unload() {
                    currentNode = null
                }

                function showDataDeletionDialog(deleteFollowing: bool, callback) {
                    uigraph.forceNodesStatusUpdate()
                    const dialog = deleteDataDialog.createObject(
                        root,
                        {
                            "node": currentNode,
                            "deleteFollowing": deleteFollowing
                        }
                    )
                    dialog.open()
                    if(callback)
                        dialog.dataDeleted.connect(callback)
                }
            }

            Component {
                id: nodeMenuComponent
                Menu {
                    id: nodeMenu

                    property var currentNode: nodeMenuLoader.currentNode

                    // Cache computatibility/submitability status of each selected node.
                    readonly property var nodeSubmitOrComputeStatus: {
                        var collectedStatus = ({})
                        uigraph.nodeSelection.selectedIndexes.forEach(function(idx) {
                            const node = uigraph.graph.nodes.at(idx.row)
                            collectedStatus[node] = uigraph.graph.canSubmitOrCompute(node)
                        })
                        return collectedStatus
                    }

                    readonly property bool isSelectionFullyComputed: {
                        return uigraph.nodeSelection.selectedIndexes.every(function(idx) {
                            const node = uigraph.graph.nodes.at(idx.row)
                            return node.isComputed
                        })
                    }

                    // Selection contains only compatibility nodes
                    readonly property bool isSelectionFullyCompatibility: {
                        return uigraph.nodeSelection.selectedIndexes.every(function(idx) {
                            const node = uigraph.graph.nodes.at(idx.row)
                            return node.isCompatibilityNode
                        })
                    }

                    // Selection contains at least one computable node type
                    readonly property bool selectionContainsComputableNodeType: {
                        return uigraph.nodeSelection.selectedIndexes.some(function(idx) {
                            const node = uigraph.graph.nodes.at(idx.row)
                            return node.isComputableType
                        })
                    }

                    readonly property bool canSelectionBeComputed: {
                        if(!selectionContainsComputableNodeType)
                            return false
                        if(isSelectionFullyCompatibility)
                            return false
                        if(isSelectionFullyComputed)
                            return true
                        var b = uigraph.nodeSelection.selectedIndexes.every(function(idx) {
                            const node = uigraph.graph.nodes.at(idx.row)
                            return (
                                node.isComputed ||
                                (uigraph.graph.canComputeTopologically(node) &&
                                // canCompute if canSubmitOrCompute == 1(can compute) or 3(can compute & submit)
                                nodeSubmitOrComputeStatus[node] % 2 == 1)
                            )
                        })
                        return b
                    }

                    readonly property bool isSelectionSubmitable: uigraph.canSubmit && selectionContainsComputableNodeType

                    readonly property bool canSelectionBeSubmitted: {
                        if(!selectionContainsComputableNodeType)
                            return false
                        if(isSelectionFullyCompatibility)
                            return false
                        if(isSelectionFullyComputed)
                            return true
                        return uigraph.nodeSelection.selectedIndexes.every(function(idx) {
                            const node = uigraph.graph.nodes.at(idx.row)
                            return (
                                node.isComputed ||
                                (uigraph.graph.canComputeTopologically(node) &&
                                 // canSubmit if canSubmitOrCompute == 2(can submit) or 3(can compute & submit)
                                 nodeSubmitOrComputeStatus[node] > 1)
                            )
                        })
                    }

                    width: 220

                    Component.onCompleted: {
                        // Compositing node: the menu opens with "Evaluate" under the cursor (right click, then click).
                        // Placed in the graph editor (not scaled by the zoom), at the cursor
                        if (nodeMenu.currentNode.isImageOpNode) {
                            parent = root
                            var cursor = root.menuPosition
                            popup(cursor.x - 40, cursor.y - topPadding - evaluateMenuItem.implicitHeight / 2)
                        } else {
                            popup()
                        }
                    }
                    onClosed: nodeMenuLoader.unload()

                    MenuItem {
                        id: evaluateMenuItem
                        text: "Evaluate"
                        visible: nodeMenu.currentNode.isImageOpNode
                        height: visible ? implicitHeight : 0
                        ToolTip.text: "Compute the branch of this node in memory and display the result: view 1 (E)"
                        ToolTip.visible: hovered
                        ToolTip.delay: 500
                        onTriggered: uigraph.live.evaluate(nodeMenu.currentNode)
                    }
                    MenuItem {
                        text: "Evaluate Compare"
                        visible: nodeMenu.currentNode.isImageOpNode
                        height: visible ? implicitHeight : 0
                        ToolTip.text: "Display this node as view 2: keys 1 / 2 switch between Evaluate and Evaluate Compare (before / after)"
                        ToolTip.visible: hovered
                        ToolTip.delay: 500
                        onTriggered: uigraph.live.evaluateCompare(nodeMenu.currentNode)
                    }
                    MenuItem {
                        id: disableMenuItem
                        // Same rule as toggleDisableSelectedNodes: enable only when every selected node is disabled
                        readonly property bool isDisabled: uigraph.nodeSelection.selectedIndexes.every(function(idx) {
                            const node = uigraph.graph.nodes.at(idx.row)
                            return !node.isImageOpNode || node.internalAttribute("disable").value === true
                        })
                        text: isDisabled ? "Enable" : "Disable"
                        visible: nodeMenu.currentNode.isImageOpNode
                        height: visible ? implicitHeight : 0
                        ToolTip.text: "Disabled nodes pass their first image input through (I)"
                        ToolTip.visible: hovered
                        ToolTip.delay: 500
                        onTriggered: uigraph.toggleDisableSelectedNodes()
                    }
                    MenuItem {
                        text: "Clone"
                        visible: nodeMenu.currentNode.isImageOpNode && nodeMenu.currentNode.hasAttribute("output")
                        height: visible ? implicitHeight : 0
                        ToolTip.text: "Create a node that looks disconnected but outputs the result of this one (hidden link)"
                        ToolTip.visible: hovered
                        ToolTip.delay: 500
                        onTriggered: {
                            var clone = uigraph.cloneNode(nodeMenu.currentNode)
                            if (clone) {
                                uigraph.selectedNode = clone
                                uigraph.selectNodes([clone])
                            }
                        }
                    }
                    MenuItem {
                        id: computeMenuItem
                        text: nodeMenu.isSelectionFullyComputed ? "Re-Compute" : "Compute"
                        visible: nodeMenu.selectionContainsComputableNodeType
                        height: visible ? implicitHeight : 0
                        enabled: nodeMenu.canSelectionBeComputed

                        onTriggered: {
                            if (nodeMenu.isSelectionFullyComputed) {
                                nodeMenuLoader.showDataDeletionDialog(
                                    false,
                                    function(request, uigraph) {
                                        request(uigraph.getSelectedNodes())
                                    }.bind(null, computeRequest, uigraph)
                                )
                            } else {
                                computeRequest(uigraph.getSelectedNodes())
                            }
                        }
                    }
                    MenuItem {
                        id: submitMenuItem

                        text: nodeMenu.isSelectionFullyComputed ? "Re-Submit" : "Submit"
                        visible: nodeMenu.isSelectionSubmitable
                        height: visible ? implicitHeight : 0
                        enabled: nodeMenu.canSelectionBeSubmitted

                        onTriggered: {
                            if (nodeMenu.isSelectionFullyComputed) {
                                nodeMenuLoader.showDataDeletionDialog(
                                    false,
                                    function(request, uigraph) {
                                        request(uigraph.getSelectedNodes())
                                    }.bind(null, submitRequest, uigraph)
                                )
                            } else {
                                submitRequest(uigraph.getSelectedNodes())
                            }
                        }
                    }
                    MenuItem {
                        text: "Stop Computation"
                        enabled: nodeMenu.currentNode.canBeStopped() && nodeMenu.currentNode.globalExecMode == "LOCAL"
                        visible: enabled
                        height: visible ? implicitHeight : 0
                        onTriggered: uigraph.stopNodeComputation(nodeMenu.currentNode)
                    }
                    MenuItem {
                        text: "Cancel Computation"
                        enabled: nodeMenu.currentNode.canBeCanceled() && nodeMenu.currentNode.globalExecMode == "LOCAL"
                        visible: enabled
                        height: visible ? implicitHeight : 0
                        onTriggered: uigraph.cancelNodeComputation(nodeMenu.currentNode)
                    }
                    MenuItem {
                        text: "Interrupt Job"
                        enabled: nodeMenu.currentNode.canBeStopped() && nodeMenu.currentNode.globalExecMode == "EXTERN"
                        visible: enabled
                        height: visible ? implicitHeight : 0
                        onTriggered: uigraph.stopNode(nodeMenu.currentNode)
                    }
                    MenuItem {
                        text: "Cancel Job"
                        enabled: nodeMenu.currentNode.canBeCanceled() && nodeMenu.currentNode.globalExecMode == "EXTERN"
                        visible: enabled
                        height: visible ? implicitHeight : 0
                        onTriggered: uigraph.stopNode(nodeMenu.currentNode)
                    }
                    MenuItem {
                        text: "Retry Error Tasks"
                        enabled: nodeMenu.currentNode.globalExecMode == "EXTERN" && ["ERROR", "STOPPED", "KILLED"].includes(nodeMenu.currentNode.globalStatus)
                        visible: enabled
                        height: visible ? implicitHeight : 0
                        onTriggered: uigraph.restartJobErrorTasks(nodeMenu.currentNode)
                    }
                    MenuItem {
                        text: "Open Folder"
                        visible: nodeMenu.currentNode.isComputableType
                        height: visible ? implicitHeight : 0
                        onTriggered: Qt.openUrlExternally(Filepath.stringToUrl(nodeMenu.currentNode.internalFolder))
                    }
                    MenuSeparator {
                        visible: nodeMenu.currentNode.isComputableType
                    }
                    MenuItem {
                        text: "Cut Node(s)"
                        enabled: true
                        ToolTip.text: "Copy selection to the clipboard and remove it"
                        ToolTip.visible: hovered
                        onTriggered: {
                            copyNodes()
                            uigraph.removeSelectedNodes()
                        }
                    }
                    MenuItem {
                        text: "Copy Node(s)"
                        enabled: true
                        ToolTip.text: "Copy selection to the clipboard"
                        ToolTip.visible: hovered
                        onTriggered: copyNodes()
                    }
                    MenuItem {
                        text: "Paste Node(s)"
                        enabled: true
                        ToolTip.text: "Copy selection to the clipboard and immediately paste it"
                        ToolTip.visible: hovered
                        onTriggered: {
                            copyNodes()
                            pasteNodes()
                        }
                    }
                    MenuItem {
                        text: "Disconnect Node(s)"
                        enabled: true
                        ToolTip.text: "Disconnect all edges from the selected Node(s)"
                        ToolTip.visible: hovered
                        onTriggered: uigraph.disconnectSelectedNodes()
                    }
                    MenuItem {
                        text: "Duplicate Node(s)" + (duplicateFollowingButton.hovered ? " From Here" : "")
                        enabled: true
                        onTriggered: duplicateNode(false)
                        MaterialToolButton {
                            id: duplicateFollowingButton
                            height: parent.height
                            anchors {
                                right: parent.right
                                rightMargin: parent.padding
                            }
                            text: MaterialIcons.fast_forward
                            onClicked: {
                                duplicateNode(true)
                                nodeMenu.close()
                            }
                        }
                    }
                    MenuItem {
                        text: "Remove Node(s)" + (removeFollowingButton.hovered ? " From Here" : "")
                        enabled: !nodeMenu.currentNode.locked
                        onTriggered: uigraph.removeSelectedNodes()
                        MaterialToolButton {
                            id: removeFollowingButton
                            height: parent.height
                            anchors {
                                right: parent.right
                                rightMargin: parent.padding
                            }
                            text: MaterialIcons.fast_forward
                            onClicked: {
                                uigraph.removeNodesFrom(uigraph.getSelectedNodes())
                                nodeMenu.close()
                            }
                        }
                    }
                    MenuSeparator {
                        visible: nodeMenu.currentNode.isComputableType
                    }
                    MenuItem {
                        id: deleteDataMenuItem
                        text: "Delete Data" + (deleteFollowingButton.hovered ? " From Here" : "" ) + "..."
                        visible: nodeMenu.currentNode.isComputableType
                        height: visible ? implicitHeight : 0
                        enabled: {
                            if (!nodeMenu.currentNode)
                                return false
                            // Check if the current node is locked (needed because it does not belong to its own duplicates list)
                            if (nodeMenu.currentNode.locked)
                                return false
                            // Check if at least one of the duplicate nodes is locked
                            for (let i = 0; i < nodeMenu.currentNode.duplicates.count; ++i) {
                                if (nodeMenu.currentNode.duplicates.at(i).locked)
                                    return false
                            }
                            return true
                        }

                        onTriggered: nodeMenuLoader.showDataDeletionDialog(false)

                        MaterialToolButton {
                            id: deleteFollowingButton
                            anchors {
                                right: parent.right
                                rightMargin: parent.padding
                            }
                            height: parent.height
                            text: MaterialIcons.fast_forward
                            onClicked: {
                                nodeMenuLoader.showDataDeletionDialog(true)
                                nodeMenu.close()
                            }
                        }
                    }
                }
            }

            // Confirmation dialog for node cache deletion
            Component {
                id: deleteDataDialog
                MessageDialog  {
                    property var node
                    property bool deleteFollowing: false

                    signal dataDeleted()

                    focus: true
                    modal: false
                    header.visible: false

                    text: "Delete Data of '" + node.label + "'" + (uigraph.nodeSelection.selectedIndexes.length > 1 ? " and other selected Nodes" : "") + (deleteFollowing ?  " and following Nodes?" : "?")
                    helperText: "Warning: This operation cannot be undone."
                    standardButtons: Dialog.Yes | Dialog.Cancel

                    onAccepted: {
                        if (deleteFollowing)
                            uigraph.clearDataFrom(uigraph.getSelectedNodes())
                        else
                            uigraph.clearSelectedNodesData()
                        dataDeleted()
                    }
                    onClosed: destroy()
                }
            }

            // Nodes
            Repeater {
                id: nodeRepeater

                model: root.graph ? root.graph.nodes : undefined

                property bool loaded: model ? count === model.count : false
                property bool ongoingDrag: false
                property bool updateSelectionOnClick: false
                property var temporaryEdgeAboutToBeRemoved: undefined

                function getItemAt(index) {
                    const loader = itemAt(index)
                    if (loader && loader.item)
                        return loader.item
                    return null
                }

                delegate: Loader {
                    id: nodeLoader
                    Component {
                        id: nodeComponent
                        Node {
                            id: nodeDelegate

                            node: object
                            width: uigraph.layout.nodeWidth

                            mainSelected: uigraph.selectedNode === node
                            hovered: uigraph.hoveredNode === node
                            hasWarnings: node.hasInvalidAttribute
                            liveController: uigraph.live
                            // Chapter folding: hidden in a folded chapter, moved up below folded ones
                            foldShift: { uigraph.foldRevision; return uigraph.foldShift(node.y) }
                            visible: { uigraph.foldRevision; return !uigraph.isFoldedAway(node.y) }

                            // ItemSelectionModel.hasSelection triggers updates anytime the selectionChanged() signal is emitted.
                            selected: uigraph.nodeSelection.hasSelection ? uigraph.nodeSelection.isRowSelected(index) : false

                            onAttributePinCreated: function(attribute, pin) { registerAttributePin(attribute, pin) }
                            onPinClicked: function(attribute) { root.handlePinClick(attribute) }
                            onAttributePinDeleted: function(attribute, pin) { unregisterAttributePin(attribute, pin) }

                            onShaked: {
                                uigraph.disconnectSelectedNodes()
                            }

                            onPressed: function(mouse) {
                                nodeRepeater.updateSelectionOnClick = true
                                root.clearEdgeSelection()
                                nodeRepeater.ongoingDrag = true

                                let selectionMode = ItemSelectionModel.NoUpdate

                                if (!selected) {
                                    selectionMode = ItemSelectionModel.ClearAndSelect
                                }

                                if (mouse.button === Qt.LeftButton) {
                                    if (mouse.modifiers & Qt.ShiftModifier) {
                                        selectionMode = ItemSelectionModel.Select
                                    }
                                    if (mouse.modifiers & Qt.ControlModifier) {
                                        selectionMode = ItemSelectionModel.Toggle
                                    }
                                    if (mouse.modifiers & Qt.AltModifier) {
                                        let selectFollowingMode = ItemSelectionModel.ClearAndSelect
                                        if (mouse.modifiers & Qt.ShiftModifier) {
                                            selectFollowingMode = ItemSelectionModel.Select
                                        }
                                        uigraph.selectFollowing(node, selectFollowingMode)
                                        // Indicate selection has been dealt with by setting conservative Select mode.
                                        selectionMode = ItemSelectionModel.Select
                                    }
                                }
                                else if (mouse.button === Qt.RightButton) {
                                    if (selected) {
                                        // Keep the full selection when right-clicking on an already selected node.
                                        nodeRepeater.updateSelectionOnClick = false
                                    }
                                }

                                if (selectionMode != ItemSelectionModel.NoUpdate) {
                                    nodeRepeater.updateSelectionOnClick = false
                                    uigraph.selectNodeByIndex(index, selectionMode)
                                }

                                // If the node is selected after this, make it the active selected node.
                                if (selected) {
                                    uigraph.selectedNode = node
                                }

                                // Open the node context menu once selection has been updated, where the node was pressed
                                if (mouse.button == Qt.RightButton) {
                                    root.menuPosition = nodeDelegate.mapToItem(root, nodeDelegate.mousePosition.x,
                                                                               nodeDelegate.mousePosition.y)
                                    nodeMenuLoader.load(node)
                                }
                            }

                            onReleased: function(mouse, wasDragged) {
                                nodeRepeater.ongoingDrag = false
                            }

                            // Only called when the node has not been dragged.
                            onClicked: function(mouse) {
                                if (!nodeRepeater.updateSelectionOnClick) {
                                    return
                                }
                                uigraph.selectNodeByIndex(index)
                            }

                            onDoubleClicked: function(mouse) { root.nodeDoubleClicked(mouse, node) }

                            onEntered: uigraph.hoveredNode = node
                            onExited: uigraph.hoveredNode = null

                            onEdgeAboutToBeRemoved: function(input) {
                                /*
                                 * Sometimes the signals are not in the right order because of weird Qt/QML update order
                                 * (next DropArea entered signal before previous DropArea exited signal) so edgeAboutToBeRemoved
                                 * must be set to undefined before it can be set to another attribute object.
                                 */
                                if (input === undefined) {
                                    if (nodeRepeater.temporaryEdgeAboutToBeRemoved === undefined) {
                                        root.edgeAboutToBeRemoved = input
                                    } else {
                                        root.edgeAboutToBeRemoved = nodeRepeater.temporaryEdgeAboutToBeRemoved
                                        nodeRepeater.temporaryEdgeAboutToBeRemoved = undefined
                                    }
                                } else {
                                    if (root.edgeAboutToBeRemoved === undefined) {
                                        root.edgeAboutToBeRemoved = input
                                    } else {
                                        nodeRepeater.temporaryEdgeAboutToBeRemoved = input
                                    }
                                }
                            }

                            // Interactive dragging: move the visual delegates
                            onPositionChanged: {
                                if (!selected || !dragging) {
                                    return
                                }

                                // Check for shake on the node
                                checkForShake()

                                // Compute offset between the delegate and the stored node position.
                                const offset = Qt.point(x - node.x, y - node.y)

                                uigraph.nodeSelection.selectedIndexes.forEach(function(idx) {
                                    if (idx != index) {
                                        const delegate = nodeRepeater.getItemAt(idx.row)
                                        delegate.x = delegate.node.x + offset.x
                                        delegate.y = delegate.node.y + offset.y
                                    }
                                })
                            }

                            // After drag: apply the final offset to all selected nodes
                            onMoved: function(position) {
                                // Dropped where it is displayed: below folded chapters, the display is moved up
                                const offset = Qt.point(position.x - node.x, uigraph.unfoldY(position.y - foldShift) - node.y)
                                uigraph.moveSelectedNodesBy(offset)
                            }

                            Behavior on x {
                                enabled: !nodeRepeater.ongoingDrag
                                NumberAnimation { duration: 100 }
                            }
                            Behavior on y {
                                enabled: !nodeRepeater.ongoingDrag
                                NumberAnimation { duration: 100 }
                            }
                        }
                    }

                    Component {
                        id: backdropComponent
                        Backdrop {
                            id: backdropDelegate
                            node: object
                            modelInstantiator: nodeRepeater
                            foldShift: { uigraph.foldRevision; return uigraph.foldShift(node.y) }
                            visible: { uigraph.foldRevision; return !uigraph.isFoldedAway(node.y) }

                            mainSelected: uigraph.selectedNode === node
                            hovered: uigraph.hoveredNode === node

                            // ItemSelectionModel.hasSelection triggers updates anytime the selectionChanged() signal is emitted.
                            selected: uigraph.nodeSelection.hasSelection ? uigraph.nodeSelection.isRowSelected(index) : false

                            onPressed: function(mouse) {
                                nodeRepeater.updateSelectionOnClick = true
                                root.clearEdgeSelection()
                                nodeRepeater.ongoingDrag = true

                                ctrlHeld = (mouse.modifiers & Qt.ControlModifier) !== 0

                                let selectionMode = ItemSelectionModel.NoUpdate

                                if (!selected) {
                                    selectionMode = ItemSelectionModel.ClearAndSelect
                                }

                                if (mouse.button === Qt.LeftButton) {
                                    if (mouse.modifiers & Qt.ShiftModifier) {
                                        selectionMode = ItemSelectionModel.Select
                                    }
                                    if (mouse.modifiers & Qt.ControlModifier) {
                                        selectionMode = ItemSelectionModel.Clear
                                    }
                                    if (mouse.modifiers & Qt.AltModifier) {
                                        let selectFollowingMode = ItemSelectionModel.ClearAndSelect
                                        if (mouse.modifiers & Qt.ShiftModifier) {
                                            selectFollowingMode = ItemSelectionModel.Select
                                        }
                                        uigraph.selectFollowing(node, selectFollowingMode)
                                        // Indicate selection has been dealt with by setting conservative Select mode.
                                        selectionMode = ItemSelectionModel.Select
                                    }
                                } else if (mouse.button === Qt.RightButton) {
                                    if (selected) {
                                        // Keep the full selection when right-clicking on an already selected node.
                                        nodeRepeater.updateSelectionOnClick = false
                                    }
                                }

                                if (selectionMode != ItemSelectionModel.NoUpdate) {
                                    nodeRepeater.updateSelectionOnClick = false
                                    uigraph.selectNodeByIndex(index, selectionMode)
                                }

                                // If the node is selected after this, make it the active selected node.
                                if (selected) {
                                    uigraph.selectedNode = node
                                }

                                if (!(mouse.modifiers & Qt.AltModifier)) {
                                    uigraph.selectNodesByIndices(childrenIndices, ItemSelectionModel.Select)
                                }

                                // Open the node context menu once selection has been updated.
                                if (mouse.button == Qt.RightButton) {
                                    nodeMenuLoader.load(node)
                                }
                            }

                            onReleased: function(mouse, wasDragged) {
                                ctrlHeld = false
                                nodeRepeater.ongoingDrag = false
                            }

                            // Only called when the node has not been dragged.
                            onClicked: function(mouse) {
                                if (!nodeRepeater.updateSelectionOnClick) {
                                    return
                                }
                                uigraph.selectNodeByIndex(index)

                                if (!(mouse.modifiers & Qt.AltModifier)) {
                                    uigraph.selectNodesByIndices(childrenIndices, ItemSelectionModel.Select)
                                }
                            }

                            onDoubleClicked: function(mouse) { root.nodeDoubleClicked(mouse, node) }

                            onResized: function(width, height) {
                                uigraph.resizeNode(node, width, height)
                            }
                            onResizedAndMoved: function(width, height, position) {
                                uigraph.resizeAndMoveNode(node, width, height, position)
                            }

                            onEntered: uigraph.hoveredNode = node
                            onExited: uigraph.hoveredNode = null

                            // Interactive dragging: move the visual delegates
                            onPositionChanged: {
                                if (!selected || !dragging) {
                                    return
                                }

                                // Compute offset between the delegate and the stored node position.
                                const offset = Qt.point(x - node.x, y - node.y)

                                uigraph.nodeSelection.selectedIndexes.forEach(function(idx) {
                                    if (idx != index) {
                                        const delegate = nodeRepeater.getItemAt(idx.row)
                                        delegate.x = delegate.node.x + offset.x
                                        delegate.y = delegate.node.y + offset.y
                                    }
                                })
                            }

                            // After drag: apply the final offset to all selected nodes
                            onMoved: function(position) {
                                // Dropped where it is displayed: below folded chapters, the display is moved up
                                const offset = Qt.point(position.x - node.x, uigraph.unfoldY(position.y - foldShift) - node.y)
                                uigraph.moveSelectedNodesBy(offset)
                            }

                            Behavior on x {
                                enabled: !nodeRepeater.ongoingDrag && !resizing && !uigraph.animationsDisabled
                                NumberAnimation { duration: 100 }
                            }
                            Behavior on y {
                                enabled: !nodeRepeater.ongoingDrag && !resizing && !uigraph.animationsDisabled
                                NumberAnimation { duration: 100 }
                            }
                        }
                    }

                    Component {
                        id: chapterComponent
                        ChapterLine {
                            node: object
                            uigraph: root.uigraph
                            foldShift: { uigraph.foldRevision; return uigraph.foldShift(node.y) }
                            mainSelected: uigraph.selectedNode === node
                            hovered: uigraph.hoveredNode === node
                            selected: uigraph.nodeSelection.hasSelection ? uigraph.nodeSelection.isRowSelected(index) : false

                            onPressed: function(mouse) {
                                nodeRepeater.ongoingDrag = true
                                if (mouse.modifiers & Qt.ShiftModifier)
                                    uigraph.selectNodeByIndex(index, ItemSelectionModel.Select)
                                else if (!selected)
                                    uigraph.selectNodeByIndex(index, ItemSelectionModel.ClearAndSelect)
                                if (selected)
                                    uigraph.selectedNode = node
                                if (mouse.button === Qt.RightButton)
                                    nodeMenuLoader.load(node)
                            }
                            onReleased: nodeRepeater.ongoingDrag = false
                            onClicked: function(mouse) {
                                if (mouse.button === Qt.LeftButton && !(mouse.modifiers & Qt.ShiftModifier))
                                    uigraph.selectNodeByIndex(index)
                            }
                            onEntered: uigraph.hoveredNode = node
                            onExited: uigraph.hoveredNode = null
                            // A chapter line moves alone (the nodes of its section stay in place)
                            onMoved: function(position) {
                                uigraph.moveChapter(node, Qt.point(Math.round(position.x), Math.round(uigraph.unfoldY(position.y - foldShift))))
                            }
                        }
                    }

                    sourceComponent: object.nodeType === "Chapter" ? chapterComponent :
                                     (object.isBackdropNode ? backdropComponent : nodeComponent)

                    onLoaded: {
                        nodeLoader.z = Qt.binding(function() { return nodeLoader.item ? nodeLoader.item.z : 0 })
                    }
                }
            }
        }

        DelegateSelectionBox {
            id: nodeSelectionBox
            mouseArea: mouseArea
            modelInstantiator: nodeRepeater
            container: draggable
            onDelegateSelectionEnded: function(selectedIndices, modifiers) {
                let selectionMode = ItemSelectionModel.ClearAndSelect
                if(modifiers & Qt.ShiftModifier) {
                    selectionMode = ItemSelectionModel.Select
                } else if(modifiers & Qt.ControlModifier) {
                    selectionMode = ItemSelectionModel.Deselect
                }
                uigraph.selectNodesByIndices(selectedIndices, selectionMode)
            }
        }

        DelegateSelectionLine {
            id: edgeSelectionLine
            mouseArea: mouseArea
            modelInstantiator: edgesRepeater
            container: draggable
            onDelegateSelectionEnded: function(selectedIndices, modifiers) {
                uigraph.deleteEdgesByIndices(selectedIndices)
            }
        }

        DropArea {
            id: dropArea
            anchors.fill: parent
            keys: ["text/uri-list"]

            /// The InputNode delegate currently being hovered during a file drag, if any
            property var hoveredInputNodeDelegate: null

            onEntered: function(drag) {
                nbMeshroomScenes = 0
                nbDraggedFiles = drag.urls.length

                drag.urls.forEach(function(file) {
                    const extension = Filepath.extension(file)
                    if (extension === ".mg" || extension === ".mgt") {
                        nbMeshroomScenes++
                    }
                })

                // Check if the drag enters directly over an InputNode
                var draggablePos = mapToItem(draggable, drag.x, drag.y)
                hoveredInputNodeDelegate = inputNodeDelegateAt(draggablePos.x, draggablePos.y)
                if (hoveredInputNodeDelegate) {
                    hoveredInputNodeDelegate.inputNodeDragHover = true
                }
            }

            onPositionChanged: function(drag) {
                // Update which InputNode (if any) the drag is currently over
                var draggablePos = mapToItem(draggable, drag.x, drag.y)
                var newDelegate = inputNodeDelegateAt(draggablePos.x, draggablePos.y)
                if (newDelegate !== hoveredInputNodeDelegate) {
                    if (hoveredInputNodeDelegate) {
                        hoveredInputNodeDelegate.inputNodeDragHover = false
                    }
                    hoveredInputNodeDelegate = newDelegate
                    if (hoveredInputNodeDelegate) {
                        hoveredInputNodeDelegate.inputNodeDragHover = true
                    }
                }
            }

            onExited: {
                if (hoveredInputNodeDelegate) {
                    hoveredInputNodeDelegate.inputNodeDragHover = false
                    hoveredInputNodeDelegate = null
                }
            }

            onDropped: function(drop) {
                // Clear visual feedback
                if (hoveredInputNodeDelegate) {
                    hoveredInputNodeDelegate.inputNodeDragHover = false
                }

                // If dropped on an InputNode, forward the files to its initialize() method
                if (hoveredInputNodeDelegate) {
                    _currentScene.initializeNode(hoveredInputNodeDelegate.node, drop.urls)
                    hoveredInputNodeDelegate = null
                    return
                }

                if (nbMeshroomScenes == nbDraggedFiles || nbMeshroomScenes == 0) {
                    // Retrieve mouse position and convert coordinate system
                    // from pixel values to graph reference system
                    var mousePosition = mapToItem(draggable, drag.x, drag.y)
                    // Send the list of files,
                    // to create the corresponding nodes or open another scene
                    filesDropped(drop, mousePosition)
                } else {
                    errorDialog.open()
                }
            }
        }
    }

    NodeActions {
        id: nodeActions
        uigraph: root.uigraph
        draggable: draggable
        nodeRepeater: nodeRepeater
        anchors.fill: parent

        onComputeRequest: function(node) {
            root.computeRequest([node])
        }

        onStopComputeRequest: function(node) {
            if (node.canBeStopped()) {
                uigraph.stopNodeComputation(node)
            } else if (node.canBeCanceled()) {
                uigraph.cancelNodeComputation(node)
            }
        }

        onDeleteDataRequest: function(node) {
            if (nodeActionsSettings.confirmBeforeDelete) {
                uigraph.forceNodesStatusUpdate();
                const dialog = deleteDataDialog.createObject(
                    root,
                    {
                        "node": node,
                        "deleteFollowing": false
                    }
                );
                dialog.open()
            }
            else {
                uigraph.clearSelectedNodesData()
            }
        }

        onSubmitRequest: function(node) {
            root.submitRequest([node])
        }

        onStopSubmitRequest: function(node) {
            if (node.canBeStopped() || node.canBeCanceled()) {
                uigraph.stopNode(node)
            }
        }

        onRetrySubmitRequest: function(node) {
            uigraph.restartJobErrorTasks(node)
        }
    }

    MessageDialog {
        id: errorDialog

        icon.text: MaterialIcons.error
        icon.color: "#F44336"

        title: "Different File Types"
        text: "Do not mix Meshroom project/template files and other types of files."
        standardButtons: Dialog.Ok

        parent: Overlay.overlay

        onAccepted: close()
    }

    // Toolbar
    FloatingPane {
        padding: 2
        anchors.bottom: parent.bottom
        RowLayout {
            id: navigation
            spacing: 4

            // Default index for search
            property int currentIndex: -1
            // Fit
            MaterialToolButton {
                text: MaterialIcons.fullscreen
                ToolTip.text: "Fit"
                onClicked: root.fit()
            }
            // Auto-Layout
            MaterialToolButton {
                text: MaterialIcons.linear_scale
                ToolTip.text: "Auto-Layout"
                onClicked: uigraph.layout.reset()
            }
            // Add Chapter line
            MaterialToolButton {
                text: MaterialIcons.horizontal_rule
                ToolTip.text: "Add Chapter Line (above the selected nodes)\nA chapter followed by another one can be folded with its arrow"
                onClicked: {
                    var selectedNodes = uigraph.getSelectedNodes()
                    var position = getCenterPosition()
                    if (selectedNodes.length > 0) {
                        var minX = Number.MAX_VALUE
                        var minY = Number.MAX_VALUE
                        for (var i = 0; i < selectedNodes.length; i++) {
                            minX = Math.min(minX, selectedNodes[i].x)
                            minY = Math.min(minY, selectedNodes[i].y)
                        }
                        position = Qt.point(minX, minY - uigraph.layout.gridSpacing - 28)
                    }
                    var chapter = uigraph.addChapter(position)
                    uigraph.selectedNode = chapter
                    uigraph.selectNodes([chapter])
                }
            }
            // Add Backdrop
            MaterialToolButton {
                text: MaterialIcons.sticky_note_2
                ToolTip.text: "Add Backdrop"
                onClicked: {
                    var selectedNodes = uigraph.getSelectedNodes()
                    var backdrop
                    if (selectedNodes.length > 0) {
                        // Calculate bounding box of selected nodes
                        var padding = uigraph.layout.gridSpacing * 0.5
                        var minX = Number.MAX_VALUE
                        var minY = Number.MAX_VALUE
                        var maxX = -Number.MAX_VALUE
                        var maxY = -Number.MAX_VALUE
                        for (var i = 0; i < selectedNodes.length; i++) {
                            var n = selectedNodes[i]
                            var nw = n.nodeWidth > 0 ? n.nodeWidth : uigraph.layout.nodeWidth
                            var nh = n.nodeHeight > 0 ? n.nodeHeight : uigraph.layout.nodeHeight
                            minX = Math.min(minX, n.x)
                            minY = Math.min(minY, n.y)
                            maxX = Math.max(maxX, n.x + nw)
                            maxY = Math.max(maxY, n.y + nh)
                        }
                        var bboxX = minX - padding
                        var bboxY = minY - 2 * padding  // minus padding and title bar height
                        var bboxW = Math.round(maxX - minX + 2 * padding)
                        var bboxH = Math.round(maxY - minY + 2 * padding)
                        backdrop = uigraph.addBackdropNode(Qt.point(bboxX, bboxY), bboxW, bboxH)
                    } else {
                        backdrop = uigraph.addNewNode("Backdrop", getCenterPosition())
                    }
                    uigraph.selectedNode = backdrop
                    uigraph.selectNodes([backdrop])
                }
            }

            // Separator
            Rectangle {
                Layout.fillHeight: true
                Layout.margins: 2
                implicitWidth: 1
                color: activePalette.window
            }

            ColorSelector {
                id: colorSelector
                Layout.minimumWidth: colorSelector.width

                // When a Color is selected
                onColorSelected: (color)=> {
                    uigraph.setSelectedNodesColor(color)
                }
            }

            // Separator
            Rectangle {
                Layout.fillHeight: true
                Layout.margins: 2
                implicitWidth: 1
                color: activePalette.window
            }

            // Search nodes in the graph
            SearchBar {
                id: graphSearchBar

                toggle: true // enable toggling the actual text field by the search button
                Layout.minimumWidth: graphSearchBar.width
                maxWidth: 150

                textField.background.opacity: 0.5
                textField.onTextChanged: navigation.currentIndex = -1

                onAccepted: {
                    navigation.navigateForward()
                }
            }

            MaterialToolButton {
                text: MaterialIcons.arrow_left
                padding: 0
                visible: graphSearchBar.text !== ""
                onClicked: navigation.navigateBackward()
            }

            MaterialToolButton {
                id: nextArrow
                text: MaterialIcons.arrow_right
                padding: 0
                visible: graphSearchBar.text !== ""
                onClicked: navigation.navigateForward()
            }

            Label {
                id: currentSearchLabel
                text: " " + (navigation.currentIndex + 1) + "/" + filteredNodes.count
                padding: 0
                visible: graphSearchBar.text !== ""
            }

            Repeater {
                id: filteredNodes
                model: SortFilterDelegateModel {
                    model: root.graph ? root.graph.nodes : undefined
                    sortRole: "label"
                    filters: [{role: "label", value: graphSearchBar.text}]
                    delegate: Item {
                        visible: false  // Hide the items to not affect the layout as the nodes model gets changes
                        property var index_: index
                    }
                    function modelData(item, roleName_) {
                        return item.model.object[roleName_]
                    }
                }
            }

            function navigateForward() {
                /**
                 * Moves the navigation index forwards and focuses on the next node as per index.
                 */
                if (!filteredNodes.count)
                    return

                navigation.currentIndex++
                if (navigation.currentIndex === filteredNodes.count)
                    navigation.currentIndex = 0
                navigation.nextItem()
            }

            function navigateBackward() {
                /**
                 * Moves the navigation index backwards and focuses on the previous node as per index.
                 */
                if (!filteredNodes.count)
                    return

                navigation.currentIndex--
                if (navigation.currentIndex === -1)
                    navigation.currentIndex = filteredNodes.count - 1
                navigation.nextItem()
            }

            function nextItem() {
                // Compute bounding box
                var node = nodeRepeater.getItemAt(filteredNodes.itemAt(navigation.currentIndex).index_)
                var bbox = Qt.rect(node.x, node.y, node.width, node.height)
                // Rescale to fit the bounding box in the view, zoom is limited to prevent huge text
                draggable.scale = Math.min(Math.min(root.width / bbox.width, root.height / bbox.height),maxZoom)
                // Recenter
                draggable.x = bbox.x * draggable.scale * -1 + (root.width - bbox.width * draggable.scale) * 0.5
                draggable.y = bbox.y * draggable.scale * -1 + (root.height - bbox.height * draggable.scale) * 0.5
            }
        }
    }

    function registerAttributePin(attribute, pin) {
        root._attributeToDelegate[attribute.uuid] = pin
        root._attributeToDelegateChanged()
    }

    function unregisterAttributePin(attribute, pin) {
        if (!attribute || !Object.prototype.hasOwnProperty.call(attribute, 'uid')) { return }

        if (attribute.uuid in root._attributeToDelegate) {
            if (root._attributeToDelegate[attribute.uuid] !== pin) { return }
            delete root._attributeToDelegate[attribute.uuid]
            root._attributeToDelegateChanged()
        }
    }

    function boundingBox() {
        var first = nodeRepeater.getItemAt(0)
        if (first === null) {
            return Qt.rect(0, 0, 0, 0)
        }
        // Displayed items only: hidden in folded chapters, or moved up below them (foldShift)
        var bbox = null
        for (var i = 0; i < root.graph.nodes.count; ++i) {
            var item = nodeRepeater.getItemAt(i)
            if (!item || !item.visible)
                continue
            var itemY = item.y - (item.foldShift || 0)
            if (bbox === null) {
                bbox = Qt.rect(item.x, itemY, item.x + item.width, itemY + item.height)
                continue
            }
            bbox.x = Math.min(bbox.x, item.x)
            bbox.y = Math.min(bbox.y, itemY)
            bbox.width = Math.max(bbox.width, item.x + item.width)
            bbox.height = Math.max(bbox.height, itemY + item.height)
        }
        if (bbox === null) {
            return Qt.rect(0, 0, 0, 0)
        }
        bbox.width -= bbox.x
        bbox.height -= bbox.y
        return bbox
    }

    function selectionBoundingBox() {
        /**
         * Returns the bounding box considering the nodes which are selected.
         * The returned bounding box starts from the Minumum x,y position to the
         * Maximum x,y postion of the selected nodes.
         */
        var firstIdx = uigraph.nodeSelection.selectedIndexes[0]
        const first = nodeRepeater.getItemAt(firstIdx.row)
        // Bounding box of the first selected item
        var bbox = Qt.rect(first.x, first.y, first.x + first.width, first.y + first.height)
        // Iterate over the remaining items in the selection
        uigraph.nodeSelection.selectedIndexes.forEach(function(idx) {
            if(idx != firstIdx) {
                const item = nodeRepeater.getItemAt(idx.row)
                bbox.x = Math.min(bbox.x, item.x)
                bbox.y = Math.min(bbox.y, item.y)
                bbox.width = Math.max(bbox.width, item.x + item.width)
                bbox.height = Math.max(bbox.height, item.y + item.height)
            }
        })

        bbox.width -= bbox.x
        bbox.height -= bbox.y
        return bbox
    }

    // Fit graph to fill root
    function fit() {
        var bbox
        // Compute bounding box
        if (uigraph.nodeSelection.hasSelection) {
            bbox = selectionBoundingBox()
        }
        else {
            bbox = boundingBox()
        }
        // Rescale to fit the bounding box in the view, zoom is limited to prevent huge text
        draggable.scale = Math.min(Math.min(root.width / bbox.width, root.height / bbox.height), maxZoom)
        // Recenter
        draggable.x = bbox.x * draggable.scale * -1 + (root.width - bbox.width * draggable.scale) * 0.5
        draggable.y = bbox.y * draggable.scale * -1 + (root.height - bbox.height * draggable.scale) * 0.5
    }
}
