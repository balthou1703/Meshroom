import QtQuick
import QtQuick.Controls

import MaterialIcons 2.2

/**
 * Visual representation of a Chapter node: an infinite horizontal line with a title (the node
 * label) and a fold arrow, splitting the graph into sections.
 *
 * The arrow points right when the chapter is unfolded and down when it is folded: the nodes between
 * this line and the next one are then hidden and the graph below moves up (UIGraph.foldShift).
 * The title can be dragged to move the line; double-click it to rename the chapter.
 */
Item {
    id: root

    property variant node
    property var uigraph: null

    property bool mainSelected: false
    property bool selected: false
    property bool hovered: false
    property bool dragging: titleMouseArea.drag.active
    // Vertical offset of the folded chapters above (the delegate is drawn moved up by this amount)
    property real foldShift: 0

    readonly property bool folded: !!node && node.hasInternalAttribute("folded") && node.internalAttribute("folded").value === true
    readonly property bool canFold: {
        if (!uigraph || !node)
            return false
        uigraph.foldRevision
        node.y
        return uigraph.canFoldChapter(node)
    }
    readonly property int hiddenCount: {
        if (!uigraph || !node || !folded)
            return 0
        uigraph.foldRevision
        return uigraph.foldedNodeCount(node)
    }
    readonly property color lineColor: node && node.color !== "" ? node.color : "#7C93AB"
    // Not a backdrop: selected by its title only (see DelegateSelectionBox)
    readonly property bool isBackdropNode: false

    signal pressed(var mouse)
    signal released(var mouse)
    signal clicked(var mouse)
    signal moved(var position)
    signal entered()
    signal exited()

    objectName: node ? node.name : ""
    x: root.node ? root.node.x : 0
    y: root.node ? root.node.y : 0
    width: title.width
    height: 28
    // Above the backdrops, below the nodes
    z: -1

    transform: Translate { y: -root.foldShift }

    Connections {
        target: root.node
        function onPositionChanged() {
            root.x = root.node.x
            root.y = root.node.y
        }
    }

    // The line, "infinite" on both sides
    Rectangle {
        x: -100000
        width: 200000
        y: root.height - height
        height: root.selected ? 3 : 2
        color: root.lineColor
        opacity: root.folded ? 1.0 : 0.8
    }

    Row {
        id: title
        height: root.height - 4
        spacing: 4

        // Fold arrow: right when unfolded, down when folded
        Label {
            anchors.verticalCenter: parent.verticalCenter
            font.family: MaterialIcons.fontFamily
            font.pointSize: 16
            text: root.folded ? MaterialIcons.arrow_drop_down : MaterialIcons.arrow_right
            color: root.lineColor
            opacity: root.canFold || root.folded ? 1.0 : 0.35
            MouseArea {
                anchors.fill: parent
                anchors.margins: -4
                cursorShape: root.canFold || root.folded ? Qt.PointingHandCursor : Qt.ArrowCursor
                enabled: !!root.uigraph
                onClicked: root.uigraph.toggleChapterFold(root.node)
                ToolTip.visible: containsMouse
                ToolTip.text: root.folded ? "Unfold the chapter" :
                              (root.canFold ? "Fold the chapter (hide the nodes down to the next line)" :
                                              "Add a chapter line below to fold this one")
                hoverEnabled: true
            }
        }

        Item {
            width: Math.max(80, nameLabel.implicitWidth)
            height: parent.height

            Label {
                id: nameLabel
                visible: !nameField.visible
                anchors.verticalCenter: parent.verticalCenter
                text: root.node ? root.node.label : ""
                font.pointSize: 13
                font.bold: true
                color: root.mainSelected ? activePalette.highlight : root.lineColor
            }
            TextField {
                id: nameField
                visible: false
                anchors.verticalCenter: parent.verticalCenter
                width: Math.max(160, implicitWidth)
                selectByMouse: true
                onEditingFinished: {
                    if (visible && root.uigraph && root.node && text !== root.node.label)
                        root.uigraph.setAttribute(root.node.internalAttribute("label"), text)
                    visible = false
                }
                Keys.onEscapePressed: visible = false
            }

            MouseArea {
                id: titleMouseArea
                anchors.fill: parent
                enabled: !nameField.visible
                hoverEnabled: true
                acceptedButtons: Qt.LeftButton | Qt.RightButton
                drag.target: root
                drag.threshold: 2
                cursorShape: drag.active ? Qt.ClosedHandCursor : Qt.OpenHandCursor
                onPressed: (mouse) => root.pressed(mouse)
                onReleased: (mouse) => root.released(mouse)
                onClicked: (mouse) => root.clicked(mouse)
                onDoubleClicked: {
                    nameField.text = root.node.label
                    nameField.visible = true
                    nameField.forceActiveFocus()
                    nameField.selectAll()
                }
                onEntered: root.entered()
                onExited: root.exited()
                drag.onActiveChanged: {
                    if (!drag.active)
                        root.moved(Qt.point(root.x, root.y))
                }
            }
        }

        Label {
            visible: root.hiddenCount > 0
            anchors.verticalCenter: parent.verticalCenter
            text: root.hiddenCount + (root.hiddenCount > 1 ? " nodes" : " node")
            font.pointSize: 8
            opacity: 0.6
        }
    }

    SystemPalette { id: activePalette }
}
