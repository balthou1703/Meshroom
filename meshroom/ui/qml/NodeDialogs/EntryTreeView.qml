import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MaterialIcons 2.2

/**
 * EntryTreeView displays and edits an EntryHierarchyModel (see meshroom/ui/components/nodeDialogs.py):
 * a tree of entries with one editable cell per column.
 *
 *  - Click / Ctrl + click / Shift + click on a row: select, toggle, extend the selection.
 *  - Drag a row by its handle and drop it ON another row to make it a child, or on the top / bottom
 *    edge of a row to make it a sibling. Dropping below the last row moves it to the root level.
 *  - Changing a combo box or a check box of a selected row applies it to all the selected rows.
 */

Item {
    id: root

    property var model: null
    /// Columns: list of {key, label, description, editor, choices, default}
    property var columns: []
    property string pathKey: columns.length > 0 ? columns[0].key : ""
    property int indentation: 18
    readonly property real rowHeight: Math.round(metricsLabel.implicitHeight * 2.0)

    // Width of each column, the path column takes the remaining space
    property var columnWidths: []
    readonly property real fixedColumnsWidth: {
        let total = 0
        for (let i = 1; i < columnWidths.length; ++i)
            total += columnWidths[i]
        return total
    }
    readonly property real pathColumnWidth: Math.max(columnWidths.length ? columnWidths[0] : 300,
                                                     listView.width - fixedColumnsWidth)
    readonly property real contentWidth: pathColumnWidth + fixedColumnsWidth

    // Drag and drop state: dropTargetId is -2 when there is no target, -1 for the root level
    property bool dragging: false
    property int dropTargetId: -2
    property string dropPosition: "inside"
    property bool dropValid: false

    Label { id: metricsLabel; visible: false; text: "Ag" }
    TextMetrics { id: textMetrics; font: metricsLabel.font }

    function textWidth(text) {
        textMetrics.text = text
        return textMetrics.advanceWidth
    }

    /// Compute the columns' widths from their header and the longest of their choices
    function computeColumnWidths() {
        const widths = []
        for (let i = 0; i < columns.length; ++i) {
            const column = columns[i]
            let w = textWidth(column.label) + 24
            if (i === 0) {
                w = Math.max(w, textWidth("/obj/example/some_node_name") + 80)
            } else if (column.editor === "choice") {
                for (let c = 0; c < column.choices.length; ++c)
                    w = Math.max(w, textWidth(column.choices[c]) + 56)
            } else if (column.editor === "bool") {
                w = Math.max(w, 48)
            } else {
                w = Math.max(w, textWidth("0000000000") + 24)
            }
            widths.push(Math.ceil(w))
        }
        columnWidths = widths
    }

    Component.onCompleted: computeColumnWidths()
    onColumnsChanged: computeColumnWidths()
    Connections {
        target: metricsLabel
        function onFontChanged() { root.computeColumnWidths() }
    }

    function updateDropTarget(entryId, y, height) {
        if (y < height * 0.25)
            dropPosition = "before"
        else if (y > height * 0.75)
            dropPosition = "after"
        else
            dropPosition = "inside"
        dropTargetId = entryId
        dropValid = model.canDropSelection(entryId, dropPosition)
    }

    function finishDrag() {
        const target = dropTargetId
        const position = dropPosition
        const valid = dropValid
        dragging = false
        dropTargetId = -2
        if (target !== -2 && valid)
            Qt.callLater(function() { root.model.dropSelection(target, position) })
    }

    // Drop below the last row: move to the root level
    DropArea {
        anchors.fill: listView
        onEntered: function(drag) { root.dropTargetId = -1; root.dropPosition = "after"; root.dropValid = true }
        onExited: if (root.dropTargetId === -1) root.dropTargetId = -2
    }

    ListView {
        id: listView
        anchors.fill: parent
        clip: true
        model: root.model
        boundsBehavior: Flickable.StopAtBounds
        contentWidth: root.contentWidth
        flickableDirection: Flickable.AutoFlickIfNeeded
        interactive: !root.dragging
        ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }
        ScrollBar.horizontal: ScrollBar { policy: ScrollBar.AsNeeded }
        headerPositioning: ListView.OverlayHeader

        header: Rectangle {
            z: 2
            width: root.contentWidth
            height: root.rowHeight * 0.8
            color: Qt.darker(palette.window, 1.15)
            Row {
                anchors.fill: parent
                Repeater {
                    model: root.columns
                    delegate: Label {
                        required property int index
                        required property var modelData
                        width: index === 0 ? root.pathColumnWidth : (root.columnWidths[index] || 0)
                        height: parent.height
                        leftPadding: index === 0 ? 8 : 0
                        horizontalAlignment: index === 0 ? Text.AlignLeft : Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        text: modelData.label
                        font.bold: true
                        elide: Text.ElideRight
                        ToolTip.text: modelData.description
                        ToolTip.visible: !!modelData.description && headerMouseArea.containsMouse
                        ToolTip.delay: 500
                        MouseArea { id: headerMouseArea; anchors.fill: parent; hoverEnabled: true; acceptedButtons: Qt.NoButton }
                    }
                }
            }
        }

        delegate: Item {
            id: rowItem

            required property int index
            required property int entryId
            required property int depth
            required property bool hasChildren
            required property bool expanded
            required property var values
            required property bool selected

            readonly property bool isDropTarget: root.dragging && root.dropTargetId === entryId

            width: root.contentWidth
            height: root.rowHeight

            Rectangle {
                anchors.fill: parent
                color: {
                    if (rowItem.selected)
                        return Qt.rgba(palette.highlight.r, palette.highlight.g, palette.highlight.b, 0.45)
                    return rowItem.index % 2 ? palette.alternateBase : palette.base
                }
                border.width: rowItem.isDropTarget && root.dropPosition === "inside" ? 2 : 0
                border.color: root.dropValid ? palette.highlight : "#E05050"
            }
            // Drop indicator between rows
            Rectangle {
                visible: rowItem.isDropTarget && root.dropPosition !== "inside"
                x: (rowItem.depth + 1) * root.indentation
                width: parent.width - x
                height: 3
                y: root.dropPosition === "before" ? 0 : parent.height - height
                color: root.dropValid ? palette.highlight : "#E05050"
            }

            // Selection (under the editors, which handle their own clicks)
            MouseArea {
                anchors.fill: parent
                onClicked: function(mouse) {
                    root.model.select(rowItem.entryId, (mouse.modifiers & Qt.ControlModifier) !== 0,
                                      (mouse.modifiers & Qt.ShiftModifier) !== 0)
                    root.forceActiveFocus()
                }
            }

            DropArea {
                anchors.fill: parent
                onEntered: function(drag) { root.updateDropTarget(rowItem.entryId, drag.y, height) }
                onPositionChanged: function(drag) { root.updateDropTarget(rowItem.entryId, drag.y, height) }
                onExited: if (root.dropTargetId === rowItem.entryId) root.dropTargetId = -2
            }

            Row {
                height: parent.height

                // Path column: indentation, drag handle, fold arrow, text field
                Item {
                    width: root.pathColumnWidth
                    height: parent.height
                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 2 + rowItem.depth * root.indentation
                        anchors.rightMargin: 4
                        spacing: 2

                        MaterialLabel {
                            id: dragHandle
                            text: MaterialIcons.drag_indicator
                            font.pointSize: metricsLabel.font.pointSize
                            opacity: handleArea.containsMouse || handleArea.drag.active ? 1.0 : 0.5
                            MouseArea {
                                id: handleArea
                                anchors.fill: parent
                                anchors.margins: -2
                                hoverEnabled: true
                                cursorShape: drag.active ? Qt.ClosedHandCursor : Qt.OpenHandCursor
                                drag.target: dragProxy
                                drag.threshold: 4
                                onPressed: function(mouse) {
                                    root.model.ensureSelected(rowItem.entryId)
                                    const p = mapToItem(root, mouse.x, mouse.y)
                                    dragProxy.x = p.x - dragProxy.Drag.hotSpot.x
                                    dragProxy.y = p.y - dragProxy.Drag.hotSpot.y
                                    dragProxy.path = String(rowItem.values[root.pathKey] || "(empty)")
                                }
                            }
                            Connections {
                                target: handleArea.drag
                                function onActiveChanged() {
                                    if (handleArea.drag.active) {
                                        root.dropTargetId = -2
                                        root.dragging = true
                                        dragProxy.Drag.active = true
                                    } else {
                                        dragProxy.Drag.active = false
                                        root.finishDrag()
                                    }
                                }
                            }
                        }
                        MaterialLabel {
                            Layout.preferredWidth: root.indentation
                            horizontalAlignment: Text.AlignHCenter
                            font.pointSize: metricsLabel.font.pointSize
                            opacity: rowItem.hasChildren ? 1.0 : 0.0
                            text: rowItem.expanded ? MaterialIcons.keyboard_arrow_down : MaterialIcons.keyboard_arrow_right
                            MouseArea {
                                anchors.fill: parent
                                enabled: rowItem.hasChildren
                                onClicked: {
                                    const entryId = rowItem.entryId
                                    Qt.callLater(function() { root.model.toggleExpanded(entryId) })
                                }
                            }
                        }
                        TextField {
                            Layout.fillWidth: true
                            selectByMouse: true
                            placeholderText: "(node path)"
                            text: rowItem.values[root.pathKey] !== undefined ? String(rowItem.values[root.pathKey]) : ""
                            onActiveFocusChanged: if (activeFocus && !rowItem.selected) root.model.select(rowItem.entryId, false, false)
                            onEditingFinished: root.model.setValue(rowItem.entryId, root.pathKey, text)
                        }
                    }
                }

                // Other columns
                Repeater {
                    model: root.columns.length - 1
                    delegate: Item {
                        id: cell
                        required property int index
                        readonly property var column: root.columns[index + 1]
                        width: root.columnWidths[index + 1] || 0
                        height: rowItem.height

                        Loader {
                            anchors.fill: parent
                            anchors.margins: 2
                            sourceComponent: {
                                switch (cell.column.editor) {
                                    case "choice": return choiceEditor
                                    case "bool": return boolEditor
                                    case "int":
                                    case "float": return numberEditor
                                    default: return textEditor
                                }
                            }
                        }

                        Component {
                            id: choiceEditor
                            ComboBox {
                                id: combo
                                readonly property var value: rowItem.values[cell.column.key]
                                model: cell.column.choices
                                currentIndex: cell.column.choices.indexOf(String(value))
                                displayText: currentIndex < 0 ? String(value) : currentText
                                focusPolicy: Qt.NoFocus
                                ToolTip.text: displayText
                                ToolTip.visible: hovered && displayText.length > 20
                                ToolTip.delay: 500
                                onActivated: function(i) {
                                    root.model.setValueOnSelection(rowItem.entryId, cell.column.key, cell.column.choices[i])
                                    // Restore the binding broken by the user interaction
                                    currentIndex = Qt.binding(function() { return cell.column.choices.indexOf(String(combo.value)) })
                                }
                            }
                        }
                        Component {
                            id: boolEditor
                            Item {
                                CheckBox {
                                    anchors.centerIn: parent
                                    focusPolicy: Qt.NoFocus
                                    checked: !!rowItem.values[cell.column.key]
                                    // Keep the binding: the model is the source of truth
                                    nextCheckState: function() { return checkState }
                                    onClicked: root.model.setValueOnSelection(rowItem.entryId, cell.column.key, !checked)
                                }
                            }
                        }
                        Component {
                            id: textEditor
                            TextField {
                                selectByMouse: true
                                horizontalAlignment: Text.AlignHCenter
                                text: rowItem.values[cell.column.key] !== undefined && rowItem.values[cell.column.key] !== null ? String(rowItem.values[cell.column.key]) : ""
                                onEditingFinished: root.model.setValue(rowItem.entryId, cell.column.key, text)
                            }
                        }
                        Component {
                            id: numberEditor
                            TextField {
                                selectByMouse: true
                                horizontalAlignment: Text.AlignHCenter
                                validator: cell.column.editor === "int" ? intValidator : doubleValidator
                                text: rowItem.values[cell.column.key] !== undefined ? String(rowItem.values[cell.column.key]) : ""
                                onEditingFinished: root.model.setValue(rowItem.entryId, cell.column.key,
                                                                       cell.column.editor === "int" ? parseInt(text) : parseFloat(text))
                                IntValidator { id: intValidator }
                                DoubleValidator { id: doubleValidator }
                            }
                        }
                    }
                }
            }
        }

        Label {
            anchors.centerIn: parent
            visible: listView.count === 0
            width: parent.width - 40
            horizontalAlignment: Text.AlignHCenter
            wrapMode: Text.WordWrap
            text: "No entry. Add nodes from the scene, or create an empty entry."
            opacity: 0.6
        }
    }

    // Item following the mouse while dragging rows
    Rectangle {
        id: dragProxy
        property string path: ""
        visible: root.dragging
        z: 10
        width: dragLabel.implicitWidth + 16
        height: root.rowHeight * 0.8
        radius: 3
        color: palette.highlight
        opacity: 0.85
        Drag.hotSpot.x: 8
        Drag.hotSpot.y: height / 2
        Label {
            id: dragLabel
            anchors.centerIn: parent
            color: palette.highlightedText
            text: {
                const n = root.model ? root.model.selectedCount : 0
                return n > 1 ? n + " entries" : dragProxy.path
            }
        }
    }

    // Keyboard shortcuts on the selection
    Keys.onPressed: function(event) {
        if (!root.model)
            return
        if (event.key === Qt.Key_Delete) {
            root.model.removeSelected()
        } else if (event.key === Qt.Key_A && (event.modifiers & Qt.ControlModifier)) {
            root.model.selectAll()
        } else if (event.key === Qt.Key_Up && (event.modifiers & Qt.ControlModifier)) {
            root.model.shiftSelected(-1)
        } else if (event.key === Qt.Key_Down && (event.modifiers & Qt.ControlModifier)) {
            root.model.shiftSelected(1)
        } else if (event.key === Qt.Key_Right && (event.modifiers & Qt.ControlModifier)) {
            root.model.indentSelected()
        } else if (event.key === Qt.Key_Left && (event.modifiers & Qt.ControlModifier)) {
            root.model.outdentSelected()
        } else {
            return
        }
        event.accepted = true
    }
}
