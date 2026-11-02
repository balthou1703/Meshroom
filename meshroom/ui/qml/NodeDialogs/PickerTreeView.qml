import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MaterialIcons 2.2

/**
 * PickerTreeView displays a PickerTreeModel (see meshroom/ui/components/nodeDialogs.py): a tree of
 * contexts and checkable leaves, with a name and a type column.
 *
 *  - Click on a row: check / uncheck the leaf (or all the visible leaves of a context).
 *  - Shift + click: apply the state of the last clicked leaf to all the leaves in between.
 *  - Click on the arrow: fold / unfold a context.
 *  - Double click on a leaf: leafDoubleClicked(key).
 */

ColumnLayout {
    id: root

    property var model: null
    property var columns: ["Name", "Type"]
    property real typeColumnWidth: Math.max(140, typeMetrics.advanceWidth + 24)
    readonly property real rowHeight: Math.round(metricsLabel.implicitHeight * 1.7)
    property int indentation: 18

    signal leafDoubleClicked(string key)

    spacing: 0

    // Hidden label used to follow the font of the window
    Label { id: metricsLabel; visible: false; text: "Ag" }
    TextMetrics { id: typeMetrics; font: metricsLabel.font; text: "usdrender_rop::3.0" }

    // Header
    Rectangle {
        Layout.fillWidth: true
        implicitHeight: root.rowHeight
        color: Qt.darker(palette.window, 1.15)
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 8
            anchors.rightMargin: 8
            Label {
                Layout.fillWidth: true
                text: root.columns.length > 0 ? root.columns[0] : ""
                font.bold: true
                elide: Text.ElideRight
            }
            Label {
                Layout.preferredWidth: root.typeColumnWidth
                text: root.columns.length > 1 ? root.columns[1] : ""
                font.bold: true
                elide: Text.ElideRight
            }
        }
    }

    ListView {
        id: listView
        Layout.fillWidth: true
        Layout.fillHeight: true
        clip: true
        model: root.model
        boundsBehavior: Flickable.StopAtBounds
        ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

        delegate: Rectangle {
            id: rowItem

            required property int index
            required property string key
            required property string name
            required property string type
            required property int depth
            required property bool isLeaf
            required property bool hasChildren
            required property bool expanded
            required property int checkState

            width: ListView.view.width
            height: root.rowHeight
            color: {
                if (rowMouseArea.containsMouse)
                    return Qt.lighter(palette.base, 1.4)
                return index % 2 ? palette.alternateBase : palette.base
            }

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 4 + rowItem.depth * root.indentation
                anchors.rightMargin: 8
                spacing: 4

                MaterialLabel {
                    id: expander
                    Layout.preferredWidth: root.indentation
                    horizontalAlignment: Text.AlignHCenter
                    font.pointSize: metricsLabel.font.pointSize
                    opacity: rowItem.hasChildren ? 1.0 : 0.0
                    text: rowItem.expanded ? MaterialIcons.keyboard_arrow_down : MaterialIcons.keyboard_arrow_right
                }
                CheckBox {
                    // Display only: the clicks are handled by the row
                    padding: 0
                    focusPolicy: Qt.NoFocus
                    tristate: !rowItem.isLeaf
                    checkState: rowItem.checkState === 2 ? Qt.Checked : (rowItem.checkState === 1 ? Qt.PartiallyChecked : Qt.Unchecked)
                    opacity: rowItem.isLeaf ? 1.0 : 0.7
                }
                Label {
                    Layout.fillWidth: true
                    text: rowItem.name
                    font.bold: !rowItem.isLeaf
                    elide: Text.ElideMiddle
                }
                Label {
                    Layout.preferredWidth: root.typeColumnWidth
                    text: rowItem.type
                    opacity: 0.7
                    elide: Text.ElideRight
                }
            }

            MouseArea {
                id: rowMouseArea
                anchors.fill: parent
                hoverEnabled: true
                onClicked: function(mouse) {
                    const key = rowItem.key
                    const p = mapToItem(expander, mouse.x, mouse.y)
                    if (rowItem.hasChildren && p.x >= -4 && p.x <= expander.width + 4) {
                        // The model is reset: let the delegate finish its handler first
                        Qt.callLater(function() { root.model.toggleExpanded(key) })
                        return
                    }
                    root.model.click(key, (mouse.modifiers & Qt.ShiftModifier) !== 0)
                }
                onDoubleClicked: function(mouse) {
                    if (rowItem.isLeaf)
                        root.leafDoubleClicked(rowItem.key)
                }
            }
        }

        Label {
            anchors.centerIn: parent
            visible: listView.count === 0
            text: "No item"
            opacity: 0.6
        }
    }
}
