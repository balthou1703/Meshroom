import QtQuick
import QtQuick.Shapes

/**
 * Handles of a selected Ramp node, drawn over the viewer (screen coordinates): the two points of the
 * ramp, joined by a line, with a bar across each end showing the orientation of the fade.
 * Dragging a point updates the image live; each drag is a single undoable change.
 * Dragging the line moves both points.
 */
Item {
    id: root

    property var node: null
    property var scene: null
    // Canvas placement in the viewer: canvas pixel (u, v) is at screen (canvasX + u * zoom, canvasY + v * zoom)
    property real canvasX: 0
    property real canvasY: 0
    property real zoom: 1
    property int formatHeight: 0

    readonly property color handleColor: "#E0B062"
    readonly property real barLength: 40

    function attr(name) { return node ? node.attribute(name) : null }
    function component(name, key) {
        var a = attr(name)
        return a ? a.value.get(key).value : 0
    }

    readonly property real x0: component("point0", "x")
    readonly property real y0: component("point0", "y")
    readonly property real x1: component("point1", "x")
    readonly property real y1: component("point1", "y")
    // On screen (y down)
    readonly property real sx0: canvasX + x0 * zoom
    readonly property real sy0: canvasY + (formatHeight - y0) * zoom
    readonly property real sx1: canvasX + x1 * zoom
    readonly property real sy1: canvasY + (formatHeight - y1) * zoom
    // Unit vector across the ramp on screen
    readonly property real length: Math.max(1e-6, Math.hypot(sx1 - sx0, sy1 - sy0))
    readonly property real nx: -(sy1 - sy0) / length
    readonly property real ny: (sx1 - sx0) / length

    visible: !!node && formatHeight > 0

    Shape {
        anchors.fill: parent
        ShapePath {
            strokeColor: root.handleColor
            strokeWidth: 1.5
            strokeStyle: ShapePath.DashLine
            dashPattern: [5, 4]
            fillColor: "transparent"
            startX: root.sx0; startY: root.sy0
            PathLine { x: root.sx1; y: root.sy1 }
        }
        ShapePath {
            strokeColor: root.handleColor
            strokeWidth: 2
            fillColor: "transparent"
            startX: root.sx0 - root.nx * root.barLength; startY: root.sy0 - root.ny * root.barLength
            PathLine { x: root.sx0 + root.nx * root.barLength; y: root.sy0 + root.ny * root.barLength }
            PathMove { x: root.sx1 - root.nx * root.barLength; y: root.sy1 - root.ny * root.barLength }
            PathLine { x: root.sx1 + root.nx * root.barLength; y: root.sy1 + root.ny * root.barLength }
        }
    }

    // Drag the middle of the line: move both points
    Rectangle {
        width: 10; height: 10
        rotation: 45
        x: (root.sx0 + root.sx1) / 2 - width / 2
        y: (root.sy0 + root.sy1) / 2 - height / 2
        color: moveMA.containsMouse || moveMA.pressed ? root.handleColor : Qt.rgba(0, 0, 0, 0.4)
        border.color: root.handleColor
        border.width: 2
        MouseArea {
            id: moveMA
            anchors.fill: parent
            anchors.margins: -4
            hoverEnabled: true
            cursorShape: Qt.SizeAllCursor
            property point start
            property var startValues: []
            onPressed: function(mouse) {
                start = mapToItem(root, mouse.x, mouse.y)
                startValues = [root.x0, root.y0, root.x1, root.y1]
                root.scene.beginInteractiveEdit(root.attr("point0"))
                root.scene.beginInteractiveEdit(root.attr("point1"))
            }
            onPositionChanged: function(mouse) {
                if (!pressed)
                    return
                var p = mapToItem(root, mouse.x, mouse.y)
                var dx = (p.x - start.x) / root.zoom, dy = -(p.y - start.y) / root.zoom
                root.scene.setVec2Interactively(root.attr("point0"), startValues[0] + dx, startValues[1] + dy)
                root.scene.setVec2Interactively(root.attr("point1"), startValues[2] + dx, startValues[3] + dy)
            }
            onReleased: root.scene.endInteractiveEdits([root.attr("point0"), root.attr("point1")])
            onCanceled: root.scene.endInteractiveEdits([root.attr("point0"), root.attr("point1")])
        }
    }

    Repeater {
        model: ["point0", "point1"]
        Rectangle {
            id: handle
            required property string modelData
            required property int index
            readonly property real sx: index === 0 ? root.sx0 : root.sx1
            readonly property real sy: index === 0 ? root.sy0 : root.sy1
            width: 14; height: 14; radius: 7
            x: sx - width / 2
            y: sy - height / 2
            color: pointMA.containsMouse || pointMA.pressed ? root.handleColor : (index === 0 ? Qt.rgba(0, 0, 0, 0.6) : Qt.rgba(1, 1, 1, 0.6))
            border.color: root.handleColor
            border.width: 2

            MouseArea {
                id: pointMA
                anchors.fill: parent
                anchors.margins: -4
                hoverEnabled: true
                cursorShape: Qt.SizeAllCursor
                property point offset
                onPressed: function(mouse) {
                    var p = mapToItem(root, mouse.x, mouse.y)
                    offset = Qt.point(p.x - handle.sx, p.y - handle.sy)
                    root.scene.beginInteractiveEdit(root.attr(handle.modelData))
                }
                onPositionChanged: function(mouse) {
                    if (!pressed)
                        return
                    var p = mapToItem(root, mouse.x, mouse.y)
                    root.scene.setVec2Interactively(root.attr(handle.modelData),
                                                    (p.x - offset.x - root.canvasX) / root.zoom,
                                                    root.formatHeight - (p.y - offset.y - root.canvasY) / root.zoom)
                }
                onReleased: root.scene.endInteractiveEdit(root.attr(handle.modelData))
                onCanceled: root.scene.endInteractiveEdit(root.attr(handle.modelData))
            }
        }
    }
}
