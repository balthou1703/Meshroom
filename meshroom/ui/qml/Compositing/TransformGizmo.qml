import QtQuick

/**
 * Handles of a selected Transform node, drawn over the viewer (screen coordinates):
 * - the center point moves the image (translate);
 * - the end of the horizontal arm scales it (uniformly);
 * - the end of the vertical arm rotates it.
 * Dragging updates the image live; each drag is a single undoable change.
 */
Item {
    id: root

    property var node: null
    property var scene: null
    // Canvas placement in the viewer: canvas pixel (u, v) is at screen (canvasX + u * zoom, canvasY + v * zoom)
    property real canvasX: 0
    property real canvasY: 0
    property real zoom: 1
    property int formatWidth: 0
    property int formatHeight: 0

    readonly property color handleColor: "#62B5E0"
    readonly property real armLength: 90

    function attr(name) { return node ? node.attribute(name) : null }
    function vec2Component(name, key) {
        var a = attr(name)
        return a ? a.value.get(key).value : 0
    }

    readonly property real translateX: node ? vec2Component("translate", "x") : 0
    readonly property real translateY: node ? vec2Component("translate", "y") : 0
    readonly property real scaleX: node ? vec2Component("scale", "x") : 1
    readonly property real scaleY: node ? vec2Component("scale", "y") : 1
    readonly property real rotateValue: node && attr("rotate") ? attr("rotate").value : 0
    readonly property bool centerOnImage: node && attr("centerOnImage") ? attr("centerOnImage").value : true
    readonly property real centerX: centerOnImage ? formatWidth / 2 : vec2Component("center", "x")
    readonly property real centerY: centerOnImage ? formatHeight / 2 : vec2Component("center", "y")

    // Gizmo origin in Nuke coordinates (y up), then on screen
    readonly property real originX: centerX + translateX
    readonly property real originY: centerY + translateY
    readonly property real screenX: canvasX + originX * zoom
    readonly property real screenY: canvasY + (formatHeight - originY) * zoom
    // Screen direction of the rotated x axis (screen y goes down)
    readonly property real angleRadians: rotateValue * Math.PI / 180

    visible: !!node && formatWidth > 0

    function toNuke(screenPoint) {
        return Qt.point((screenPoint.x - canvasX) / zoom, formatHeight - (screenPoint.y - canvasY) / zoom)
    }

    // Arms
    Rectangle {
        x: root.screenX
        y: root.screenY - 1
        width: root.armLength
        height: 2
        color: root.handleColor
        transformOrigin: Item.Left
        rotation: -root.rotateValue
        antialiasing: true
    }
    Rectangle {
        x: root.screenX - 1
        y: root.screenY - root.armLength * 0.7
        width: 2
        height: root.armLength * 0.7
        color: root.handleColor
        transformOrigin: Item.Bottom
        rotation: -root.rotateValue
        antialiasing: true
    }

    // Translate: center point
    Rectangle {
        id: centerHandle
        width: 14
        height: 14
        radius: 7
        x: root.screenX - width / 2
        y: root.screenY - height / 2
        color: centerMA.containsMouse || centerMA.pressed ? root.handleColor : "transparent"
        border.color: root.handleColor
        border.width: 2

        MouseArea {
            id: centerMA
            anchors.fill: parent
            anchors.margins: -4
            hoverEnabled: true
            cursorShape: Qt.SizeAllCursor
            property point start
            property real startX
            property real startY
            onPressed: function(mouse) {
                start = mapToItem(root, mouse.x, mouse.y)
                startX = root.translateX
                startY = root.translateY
                root.scene.beginInteractiveEdit(root.attr("translate"))
            }
            onPositionChanged: function(mouse) {
                if (!pressed)
                    return
                var p = mapToItem(root, mouse.x, mouse.y)
                root.scene.setVec2Interactively(root.attr("translate"),
                                                startX + (p.x - start.x) / root.zoom,
                                                startY - (p.y - start.y) / root.zoom)
            }
            onReleased: root.scene.endInteractiveEdit(root.attr("translate"))
            onCanceled: root.scene.endInteractiveEdit(root.attr("translate"))
        }
    }

    // Scale: end of the x arm
    Rectangle {
        width: 10
        height: 10
        x: root.screenX + Math.cos(root.angleRadians) * root.armLength - width / 2
        y: root.screenY - Math.sin(root.angleRadians) * root.armLength - height / 2
        color: scaleMA.containsMouse || scaleMA.pressed ? root.handleColor : Qt.rgba(0, 0, 0, 0.4)
        border.color: root.handleColor
        border.width: 2
        rotation: -root.rotateValue

        MouseArea {
            id: scaleMA
            anchors.fill: parent
            anchors.margins: -4
            hoverEnabled: true
            cursorShape: Qt.SizeHorCursor
            property real startDistance: 1
            property real startScaleX: 1
            property real startScaleY: 1
            onPressed: function(mouse) {
                var p = mapToItem(root, mouse.x, mouse.y)
                startDistance = Math.max(1, Math.hypot(p.x - root.screenX, p.y - root.screenY))
                startScaleX = root.scaleX
                startScaleY = root.scaleY
                root.scene.beginInteractiveEdit(root.attr("scale"))
            }
            onPositionChanged: function(mouse) {
                if (!pressed)
                    return
                var p = mapToItem(root, mouse.x, mouse.y)
                var ratio = Math.hypot(p.x - root.screenX, p.y - root.screenY) / startDistance
                root.scene.setVec2Interactively(root.attr("scale"), startScaleX * ratio, startScaleY * ratio)
            }
            onReleased: root.scene.endInteractiveEdit(root.attr("scale"))
            onCanceled: root.scene.endInteractiveEdit(root.attr("scale"))
        }
    }

    // Rotate: end of the y arm
    Rectangle {
        width: 12
        height: 12
        radius: 6
        x: root.screenX - Math.sin(root.angleRadians) * root.armLength * 0.7 - width / 2
        y: root.screenY - Math.cos(root.angleRadians) * root.armLength * 0.7 - height / 2
        color: rotateMA.containsMouse || rotateMA.pressed ? root.handleColor : Qt.rgba(0, 0, 0, 0.4)
        border.color: root.handleColor
        border.width: 2

        MouseArea {
            id: rotateMA
            anchors.fill: parent
            anchors.margins: -4
            hoverEnabled: true
            cursorShape: Qt.CrossCursor
            property real startAngle: 0
            property real startRotation: 0
            function angleAt(mouse) {
                var p = mapToItem(root, mouse.x, mouse.y)
                // Counter-clockwise angle in degrees (screen y goes down)
                return Math.atan2(root.screenY - p.y, p.x - root.screenX) * 180 / Math.PI
            }
            onPressed: function(mouse) {
                startAngle = angleAt(mouse)
                startRotation = root.rotateValue
                root.scene.beginInteractiveEdit(root.attr("rotate"))
            }
            onPositionChanged: function(mouse) {
                if (!pressed)
                    return
                var value = startRotation + angleAt(mouse) - startAngle
                // Keep the angle in [-180, 180]
                value = ((value + 180) % 360 + 360) % 360 - 180
                root.scene.setValueInteractively(root.attr("rotate"), value)
            }
            onReleased: root.scene.endInteractiveEdit(root.attr("rotate"))
            onCanceled: root.scene.endInteractiveEdit(root.attr("rotate"))
        }
    }
}
