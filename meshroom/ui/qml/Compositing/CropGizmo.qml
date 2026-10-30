import QtQuick
import QtQuick.Controls
import QtQuick.Shapes

/**
 * Box of a selected Crop node, drawn over the viewer (screen coordinates): the outside darkened, the box size and
 * aspect ratio shown, handles on its edges and corners and in its middle (move). The box is in the pixels of the
 * input (from the bottom-left corner); "Right" / "Top" at 0 mean the format width / height. Values are whole pixels.
 * Shift while dragging a corner keeps the aspect ratio. The buttons above the box frame common aspect ratios in the
 * format (full width, centered). Each drag is a single undoable change.
 * When the Crop itself is viewed with "Reformat", the image shown is the box: the handles are then on its borders.
 */
Item {
    id: root

    property var node: null
    property var scene: null
    property var live: null
    // Canvas placement in the viewer: canvas pixel (u, v) is at screen (canvasX + u * zoom, canvasY + v * zoom)
    property real canvasX: 0
    property real canvasY: 0
    property real zoom: 1
    property int formatWidth: 0
    property int formatHeight: 0

    readonly property color handleColor: "#E0B062"
    readonly property var names: ["left", "bottom", "right", "top"]

    function attr(name) { return node ? node.attribute(name) : null }
    function value(name) { var a = attr(name); return a ? a.value : 0 }

    // The Crop is viewed with "Reformat": the image shown is the box, its bottom-left corner at the origin
    readonly property bool reformatViewed: !!node && !!live && live.evaluatedNode === node && value("reformat")
    readonly property real left: value("left")
    readonly property real bottom: value("bottom")
    // The width / height of the input format
    readonly property real inputWidth: reformatViewed ? left + formatWidth : formatWidth
    readonly property real inputHeight: reformatViewed ? bottom + formatHeight : formatHeight
    readonly property real right: value("right") > 0 ? value("right") : inputWidth
    readonly property real top: value("top") > 0 ? value("top") : inputHeight
    // Where the input pixel (0, 0) is on the canvas
    readonly property real originX: reformatViewed ? -left : 0
    readonly property real originY: reformatViewed ? -bottom : 0

    function screenX(x) { return canvasX + (x + originX) * zoom }
    function screenY(y) { return canvasY + (formatHeight - (y + originY)) * zoom }

    readonly property real sLeft: screenX(left)
    readonly property real sRight: screenX(right)
    readonly property real sTop: screenY(top)
    readonly property real sBottom: screenY(bottom)

    visible: !!node && formatWidth > 0 && formatHeight > 0

    // Drag state: the box when the drag started, and which of its sides follow the mouse
    property var startBox: [0, 0, 0, 0]
    property point startMouse

    function beginDrag(point) {
        startBox = [left, bottom, right, top]
        startMouse = point
        for (var i = 0; i < names.length; ++i)
            scene.beginInteractiveEdit(attr(names[i]))
    }

    function endDrag() {
        scene.endInteractiveEdits(names.map(function(name) { return attr(name) }))
    }

    function setBox(l, b, r, t) {
        // Whole pixels, at least one pixel wide and high
        l = Math.round(l); b = Math.round(b); r = Math.round(r); t = Math.round(t)
        if (r < l + 1) r = l + 1
        if (t < b + 1) t = b + 1
        scene.setValueInteractively(attr("left"), l)
        scene.setValueInteractively(attr("bottom"), b)
        scene.setValueInteractively(attr("right"), r)
        scene.setValueInteractively(attr("top"), t)
    }

    /** Drag `sides` ({left, bottom, right, top}: which sides move) by the mouse moved to `point`. */
    function dragTo(point, sides, keepAspect) {
        var dx = (point.x - startMouse.x) / zoom
        var dy = -(point.y - startMouse.y) / zoom
        var l = startBox[0] + (sides.left ? dx : 0)
        var b = startBox[1] + (sides.bottom ? dy : 0)
        var r = startBox[2] + (sides.right ? dx : 0)
        var t = startBox[3] + (sides.top ? dy : 0)
        if (keepAspect && (sides.left || sides.right) && (sides.bottom || sides.top)) {
            // The height follows the width (the aspect ratio of the box when the drag started)
            var aspect = (startBox[2] - startBox[0]) / Math.max(1, startBox[3] - startBox[1])
            var height = Math.max(1, (r - l) / aspect)
            if (sides.top)
                t = b + height
            else
                b = t - height
        }
        setBox(l, b, r, t)
    }

    /** Frame an aspect ratio in the input format: its full width (or height when narrower), centered. */
    function frameAspect(aspect) {
        beginDrag(Qt.point(0, 0))
        var w = inputWidth, h = inputHeight
        if (aspect <= 0) {
            setBox(0, 0, w, h)
        } else if (w / h > aspect) {
            var width = h * aspect
            setBox((w - width) / 2, 0, (w + width) / 2, h)
        } else {
            var height = w / aspect
            setBox(0, (h - height) / 2, w, (h + height) / 2)
        }
        endDrag()
    }

    // The outside of the box, darkened (to judge the framing)
    Shape {
        anchors.fill: parent
        ShapePath {
            fillColor: Qt.rgba(0, 0, 0, 0.45)
            strokeColor: "transparent"
            fillRule: ShapePath.OddEvenFill
            startX: root.screenX(0); startY: root.screenY(0)
            PathLine { x: root.screenX(root.inputWidth); y: root.screenY(0) }
            PathLine { x: root.screenX(root.inputWidth); y: root.screenY(root.inputHeight) }
            PathLine { x: root.screenX(0); y: root.screenY(root.inputHeight) }
            PathLine { x: root.screenX(0); y: root.screenY(0) }
            PathMove { x: root.sLeft; y: root.sBottom }
            PathLine { x: root.sRight; y: root.sBottom }
            PathLine { x: root.sRight; y: root.sTop }
            PathLine { x: root.sLeft; y: root.sTop }
            PathLine { x: root.sLeft; y: root.sBottom }
        }
        ShapePath {
            fillColor: "transparent"
            strokeColor: root.handleColor
            strokeWidth: 1.5
            startX: root.sLeft; startY: root.sBottom
            PathLine { x: root.sRight; y: root.sBottom }
            PathLine { x: root.sRight; y: root.sTop }
            PathLine { x: root.sLeft; y: root.sTop }
            PathLine { x: root.sLeft; y: root.sBottom }
        }
    }

    // Size and aspect ratio of the box, and aspect ratio presets, above it
    Row {
        x: Math.max(4, Math.min(root.width - width - 4, root.sLeft))
        y: Math.max(4, root.sTop - height - 6)
        spacing: 4

        Label {
            anchors.verticalCenter: parent.verticalCenter
            readonly property int boxWidth: Math.round(root.right - root.left)
            readonly property int boxHeight: Math.round(root.top - root.bottom)
            text: boxWidth + " x " + boxHeight + "  (" + (boxWidth / Math.max(1, boxHeight)).toFixed(2) + ")"
            color: root.handleColor
            font.pointSize: 9
            padding: 2
            background: Rectangle { color: Qt.rgba(0, 0, 0, 0.6); radius: 2 }
        }
        Repeater {
            model: [{"label": "Full", "aspect": 0}, {"label": "1:1", "aspect": 1}, {"label": "16:9", "aspect": 16 / 9},
                    {"label": "1.85", "aspect": 1.85}, {"label": "2.39", "aspect": 2.39}]
            delegate: Rectangle {
                required property var modelData
                width: presetLabel.implicitWidth + 8
                height: presetLabel.implicitHeight + 4
                radius: 2
                color: presetMA.containsMouse ? root.handleColor : Qt.rgba(0, 0, 0, 0.6)
                border.color: root.handleColor
                Label {
                    id: presetLabel
                    anchors.centerIn: parent
                    text: modelData.label
                    font.pointSize: 8
                    color: presetMA.containsMouse ? "black" : root.handleColor
                }
                MouseArea {
                    id: presetMA
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: root.frameAspect(modelData.aspect)
                    ToolTip.visible: containsMouse
                    ToolTip.delay: 500
                    ToolTip.text: modelData.aspect > 0 ? "Frame a " + modelData.label + " box in the format (centered)"
                                                       : "The whole format"
                }
            }
        }
    }

    // Move the whole box
    Rectangle {
        width: 12; height: 12
        rotation: 45
        x: (root.sLeft + root.sRight) / 2 - width / 2
        y: (root.sTop + root.sBottom) / 2 - height / 2
        color: moveMA.containsMouse || moveMA.pressed ? root.handleColor : Qt.rgba(0, 0, 0, 0.4)
        border.color: root.handleColor
        border.width: 2
        MouseArea {
            id: moveMA
            anchors.fill: parent
            anchors.margins: -4
            hoverEnabled: true
            cursorShape: Qt.SizeAllCursor
            onPressed: function(mouse) { root.beginDrag(mapToItem(root, mouse.x, mouse.y)) }
            onPositionChanged: function(mouse) {
                if (pressed)
                    root.dragTo(mapToItem(root, mouse.x, mouse.y), {"left": true, "bottom": true, "right": true, "top": true}, false)
            }
            onReleased: root.endDrag()
            onCanceled: root.endDrag()
        }
    }

    // Edges and corners: (x, y) on the box in [0, 1] from its left / bottom, the sides they move
    Repeater {
        model: [
            {"u": 0, "v": 0.5, "sides": {"left": true}, "cursor": Qt.SizeHorCursor},
            {"u": 1, "v": 0.5, "sides": {"right": true}, "cursor": Qt.SizeHorCursor},
            {"u": 0.5, "v": 0, "sides": {"bottom": true}, "cursor": Qt.SizeVerCursor},
            {"u": 0.5, "v": 1, "sides": {"top": true}, "cursor": Qt.SizeVerCursor},
            {"u": 0, "v": 0, "sides": {"left": true, "bottom": true}, "cursor": Qt.SizeBDiagCursor},
            {"u": 1, "v": 0, "sides": {"right": true, "bottom": true}, "cursor": Qt.SizeFDiagCursor},
            {"u": 0, "v": 1, "sides": {"left": true, "top": true}, "cursor": Qt.SizeFDiagCursor},
            {"u": 1, "v": 1, "sides": {"right": true, "top": true}, "cursor": Qt.SizeBDiagCursor}
        ]
        delegate: Rectangle {
            id: handle
            required property var modelData
            readonly property bool corner: modelData.u !== 0.5 && modelData.v !== 0.5
            width: corner ? 12 : 10
            height: width
            x: root.sLeft + modelData.u * (root.sRight - root.sLeft) - width / 2
            y: root.sBottom + modelData.v * (root.sTop - root.sBottom) - height / 2
            color: handleMA.containsMouse || handleMA.pressed ? root.handleColor : Qt.rgba(0, 0, 0, 0.6)
            border.color: root.handleColor
            border.width: 2

            MouseArea {
                id: handleMA
                anchors.fill: parent
                anchors.margins: -5
                hoverEnabled: true
                cursorShape: handle.modelData.cursor
                onPressed: function(mouse) { root.beginDrag(mapToItem(root, mouse.x, mouse.y)) }
                onPositionChanged: function(mouse) {
                    if (pressed)
                        root.dragTo(mapToItem(root, mouse.x, mouse.y), handle.modelData.sides,
                                    handle.corner && (mouse.modifiers & Qt.ShiftModifier))
                }
                onReleased: root.endDrag()
                onCanceled: root.endDrag()
            }
        }
    }
}
