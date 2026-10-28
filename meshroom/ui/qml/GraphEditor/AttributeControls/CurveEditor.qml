import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window

import MaterialIcons 2.2

/**
 * Editor of a set of curves stored as JSON {name: [[x, y], ...]} (see meshroom/live/curves.py):
 * the color curves of ColorLookup / ColorLookupHSV ("colorCurves" semantic, x and y from 0 to 1) or
 * the time curve of TimeWarp ("timeCurve" semantic, frames).
 *
 * - the buttons above choose the edited curve (the others are drawn faintly);
 * - click to add a point (and drag it); click a point to select it, Shift + click to add it to the selection,
 *   Ctrl + drag on an empty area to select the points in a box (Shift: added), Ctrl + A: all the points;
 *   drag a selected point to move the whole selection; Delete (or the delete button, or double-click) removes;
 * - x / y fields: the value of the selected points; "+100", "-=5", "*2", "/2" offset or scale them all
 *   (a plain number sets it);
 * - interpolation of the selected points: smooth (a slope), linear (a corner) or constant (a step);
 * - Ctrl + wheel (in the attribute editor; the wheel alone scrolls it) or wheel (in the curve window):
 *   zoom around the cursor (+ Shift: horizontally only, + Alt: vertically only); middle drag or
 *   Alt + drag: pan; "fit" button: show all the points again;
 * - the bar under the editor resizes it (kept for the session); the "open in window" button shows the editor
 *   in its own window (with a full screen button);
 * - lookup nodes: "Add Point" moves the picked Source Color to the Target Color on the curves,
 *   "Match" (with "Match Source") matches the colors of the second input.
 * Each press and release is one undoable change.
 */
ColumnLayout {
    id: root

    required property var attribute
    property bool editable: true
    // Editor shown in its own window: the plot fills the height, no pop-out button
    property bool inWindow: false

    readonly property bool timeCurve: attribute.desc.semantic === "timeCurve"
    readonly property var node: attribute.node
    readonly property var live: _currentScene ? _currentScene.live : null
    readonly property var curves: {
        try {
            var parsed = JSON.parse(attribute.value || "{}")
            return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {}
        } catch (e) {
            return {}
        }
    }
    readonly property var names: Object.keys(curves)
    property string activeName: ""
    // Height of the plot (drag the bar under the editor to resize), kept for the session per attribute
    readonly property string sizeKey: (node ? node.name : "") + "." + attribute.name
    property real plotHeight: live ? live.editorSize(sizeKey, timeCurve ? 220 : 200) : 200
    // Indices of the selected points of the active curve (in x order)
    property var selection: []
    readonly property var selectedPoints: {
        var points = sortedPoints(active)
        return selection.filter(function(i) { return i < points.length }).map(function(i) { return points[i] })
    }
    readonly property string active: names.indexOf(activeName) >= 0 ? activeName : (names.length > 0 ? names[0] : "")
    readonly property var colors: ({
        "master": "#E0E0E0", "red": "#E57373", "green": "#81C784", "blue": "#64B5F6", "alpha": "#9E9E9E",
        "hue": "#CE93D8", "saturation": "#FFB74D", "value": "#F5F5F5", "time": "#D4C27A"
    })
    // View range {x0, x1, y0, y1}: zoomed / panned by the user, else fitted to the points; frozen while dragging
    property var view: null
    property var frozenBounds: null
    readonly property var bounds: view ? view : (frozenBounds ? frozenBounds : computeBounds())
    readonly property bool hasPickers: !!node && node.hasAttribute("sourceColor") && node.hasAttribute("targetColor")
    readonly property bool canMatch: !!node && node.hasAttribute("matchSource") && node.attribute("matchSource").value

    spacing: 2

    function copy(value) { return JSON.parse(JSON.stringify(value)) }
    function modeOf(p) { return p.length > 2 ? p[2] : "smooth" }
    // A point moved to (x, y), its interpolation kept
    function withXY(p, x, y) { return p.length > 2 ? [x, y, p[2]] : [x, y] }
    function isSelected(i) { return selection.indexOf(i) >= 0 }

    function computeBounds() {
        var x0, x1, y0, y1
        if (timeCurve) {
            var first = live && live.hasFrameRange ? live.firstFrame : 1
            var last = live && live.hasFrameRange ? live.lastFrame : 100
            x0 = y0 = first
            x1 = y1 = Math.max(first + 1, last)
        } else {
            x0 = y0 = 0
            x1 = y1 = 1
        }
        names.forEach(function(name) {
            (curves[name] || []).forEach(function(p) {
                x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0])
                y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1])
            })
        })
        var padX = (x1 - x0) * 0.05, padY = (y1 - y0) * 0.05
        return { "x0": x0 - padX, "x1": x1 + padX, "y0": y0 - padY, "y1": y1 + padY }
    }

    // Zoom by `factor` (< 1: in) around the curve point (cx, cy)
    function zoom(factor, cx, cy, horizontal, vertical) {
        var b = bounds
        var x0 = b.x0, x1 = b.x1, y0 = b.y0, y1 = b.y1
        if (horizontal) {
            x0 = cx + (x0 - cx) * factor
            x1 = cx + (x1 - cx) * factor
        }
        if (vertical) {
            y0 = cy + (y0 - cy) * factor
            y1 = cy + (y1 - cy) * factor
        }
        if (x1 - x0 < 1e-4 || y1 - y0 < 1e-4 || x1 - x0 > 1e7 || y1 - y0 > 1e7)
            return
        view = { "x0": x0, "x1": x1, "y0": y0, "y1": y1 }
    }

    function pan(dxCurve, dyCurve) {
        var b = bounds
        view = { "x0": b.x0 - dxCurve, "x1": b.x1 - dxCurve, "y0": b.y0 - dyCurve, "y1": b.y1 - dyCurve }
    }

    function toScreenX(x) { return (x - bounds.x0) / (bounds.x1 - bounds.x0) * plot.width }
    function toScreenY(y) { return plot.height - (y - bounds.y0) / (bounds.y1 - bounds.y0) * plot.height }
    function toCurveX(px) { return bounds.x0 + px / plot.width * (bounds.x1 - bounds.x0) }
    function toCurveY(py) { return bounds.y0 + (plot.height - py) / plot.height * (bounds.y1 - bounds.y0) }

    function sortedPoints(name) {
        return copy(curves[name] || []).sort(function(a, b) { return a[0] - b[0] })
    }

    function begin() { _currentScene.beginInteractiveEdit(root.attribute) }
    function apply(newCurves) { _currentScene.setValueInteractively(root.attribute, JSON.stringify(newCurves)) }
    function end() { _currentScene.endInteractiveEdit(root.attribute) }

    // One undoable change of the active curve
    function commitPoints(points) {
        var newCurves = copy(curves)
        newCurves[active] = points
        begin()
        apply(newCurves)
        end()
    }

    // Picked colors (markers on the plot)
    readonly property string pickedColors: hasPickers ? JSON.stringify([
        node.attribute("sourceColor").value.get("r").value, node.attribute("sourceColor").value.get("g").value,
        node.attribute("sourceColor").value.get("b").value, node.attribute("targetColor").value.get("r").value,
        node.attribute("targetColor").value.get("g").value, node.attribute("targetColor").value.get("b").value]) : ""
    onPickedColorsChanged: plot.requestPaint()
    onCurvesChanged: plot.requestPaint()
    onActiveChanged: {
        selection = []
        plot.requestPaint()
    }
    onSelectionChanged: plot.requestPaint()
    onBoundsChanged: plot.requestPaint()

    /**
     * The new value of `value` for a typed text: "+N", "+=N", "-=N", "*N", "*=N", "/N", "/=N" offset or scale it,
     * a plain number (negative included) replaces it. NaN if the text is not valid.
     */
    function applyExpression(text, value) {
        var t = String(text).trim()
        var match = t.match(/^([+\-*\/])=\s*(.+)$/) || t.match(/^([+*\/])\s*(.+)$/)
        if (!match) {
            var number = Number(t)
            return t === "" ? NaN : number
        }
        var operand = Number(match[2])
        if (!isFinite(operand))
            return NaN
        switch (match[1]) {
            case "+": return value + operand
            case "-": return value - operand
            case "*": return value * operand
            case "/": return operand !== 0 ? value / operand : NaN
        }
        return NaN
    }

    // The x (axis 0) or y (axis 1) of the selected points set from a typed text (see applyExpression)
    function setSelectedValue(axis, text) {
        var points = sortedPoints(active)
        var chosen = selection.filter(function(i) { return i < points.length })
        if (!chosen.length)
            return
        var tagged = points.map(function(p, i) { return { "point": p, "index": i } })
        for (var k = 0; k < chosen.length; k++) {
            var p = points[chosen[k]]
            var value = applyExpression(text, p[axis])
            if (!isFinite(value))
                return
            if (root.timeCurve && axis === 0)
                value = Math.round(value)
            tagged[chosen[k]].point = axis === 0 ? withXY(p, value, p[1]) : withXY(p, p[0], value)
        }
        // Back in x order, one point per x (a moved point replaces an unselected one at the same x)
        tagged.sort(function(a, b) { return a.point[0] - b.point[0] || (isSelected(b.index) ? 1 : 0) - (isSelected(a.index) ? 1 : 0) })
        var kept = []
        tagged.forEach(function(item) {
            if (kept.length && Math.abs(kept[kept.length - 1].point[0] - item.point[0]) < 1e-9)
                return
            kept.push(item)
        })
        var newSelection = []
        kept.forEach(function(item, i) {
            if (chosen.indexOf(item.index) >= 0)
                newSelection.push(i)
        })
        commitPoints(kept.map(function(item) { return item.point }))
        selection = newSelection
    }

    // The interpolation of the selected points: "smooth", "linear" or "constant"
    function setSelectedMode(mode) {
        var points = sortedPoints(active)
        selection.forEach(function(i) {
            if (i < points.length)
                points[i] = mode === "smooth" ? [points[i][0], points[i][1]] : [points[i][0], points[i][1], mode]
        })
        commitPoints(points)
    }

    function deleteSelection() {
        if (!editable || !selection.length)
            return
        var points = sortedPoints(active).filter(function(p, i) { return !root.isSelected(i) })
        selection = []
        commitPoints(points)
    }

    // The common value of the selected points on an axis ("" when they differ)
    function commonValue(axis) {
        var points = selectedPoints
        if (!points.length)
            return ""
        var first = points[0][axis]
        for (var i = 1; i < points.length; i++) {
            if (Math.abs(points[i][axis] - first) > 1e-9)
                return ""
        }
        return String(Math.round(first * 10000) / 10000)
    }
    readonly property string commonMode: {
        var points = selectedPoints
        if (!points.length)
            return ""
        var first = modeOf(points[0])
        return points.every(function(p) { return root.modeOf(p) === first }) ? first : ""
    }

    // Curve choice
    RowLayout {
        Layout.fillWidth: true
        spacing: 2
        Repeater {
            model: root.names
            ToolButton {
                required property string modelData
                text: modelData
                checked: root.active === modelData
                font.pointSize: 8
                padding: 3
                contentItem: Label {
                    text: parent.text
                    font: parent.font
                    color: root.colors[parent.modelData] || "white"
                    opacity: parent.checked ? 1.0 : 0.6
                    horizontalAlignment: Text.AlignHCenter
                }
                onClicked: root.activeName = modelData
            }
        }
        Item { Layout.fillWidth: true }
        MaterialToolButton {
            text: MaterialIcons.fit_screen
            font.pointSize: 9
            padding: 2
            enabled: root.view !== null
            ToolTip.text: "Fit the view to the curves.\nZoom: Ctrl + wheel (wheel in the curve window), Shift: horizontally, " +
                          "Alt: vertically. Pan: middle drag or Alt + drag."
            onClicked: root.view = null
        }
        MaterialToolButton {
            visible: !root.inWindow
            text: MaterialIcons.open_in_new
            font.pointSize: 9
            padding: 2
            checked: curveWindowLoader.active
            ToolTip.text: curveWindowLoader.active ? "Close the curve window" : "Open the curves in a window"
            onClicked: curveWindowLoader.active = !curveWindowLoader.active
        }
        MaterialToolButton {
            visible: root.inWindow
            text: root.Window.window && root.Window.window.visibility === Window.FullScreen ? MaterialIcons.fullscreen_exit
                                                                                         : MaterialIcons.fullscreen
            font.pointSize: 9
            padding: 2
            ToolTip.text: "Full screen"
            onClicked: {
                var w = root.Window.window
                if (w)
                    w.visibility = w.visibility === Window.FullScreen ? Window.Windowed : Window.FullScreen
            }
        }
        MaterialToolButton {
            text: MaterialIcons.restart_alt
            font.pointSize: 9
            padding: 2
            enabled: root.editable && root.active !== ""
            ToolTip.text: "Reset the " + root.active + " curve"
            onClicked: {
                root.selection = []
                root.commitPoints(root.timeCurve ? [] : [[0, 0], [1, 1]])
            }
        }
    }

    // Selected points: their values (offsets / scales), interpolation, removal
    RowLayout {
        Layout.fillWidth: true
        spacing: 4
        Label {
            text: root.timeCurve ? "Output frame" : "x"
            font.pointSize: 8
            opacity: 0.7
        }
        TextField {
            id: pointXField
            Layout.fillWidth: true
            font.pointSize: 8
            enabled: root.editable && root.selection.length > 0
            text: root.commonValue(0)
            placeholderText: root.selection.length > 1 ? root.selection.length + " points" : ""
            selectByMouse: true
            ToolTip.visible: hovered
            ToolTip.text: "The x of the selected points. \"+100\", \"-=5\", \"*2\", \"/2\": offset or scale them all; " +
                          "a number sets it."
            onEditingFinished: {
                // Unchanged (e.g. the field left): no edit
                if (text !== root.commonValue(0))
                    root.setSelectedValue(0, text)
                text = Qt.binding(function() { return root.commonValue(0) })
            }
        }
        Label {
            text: root.timeCurve ? "Input frame" : "y"
            font.pointSize: 8
            opacity: 0.7
        }
        TextField {
            id: pointYField
            Layout.fillWidth: true
            font.pointSize: 8
            enabled: root.editable && root.selection.length > 0
            text: root.commonValue(1)
            placeholderText: root.selection.length > 1 ? root.selection.length + " points" : ""
            selectByMouse: true
            ToolTip.visible: hovered
            ToolTip.text: "The y of the selected points. \"+100\", \"-=5\", \"*2\", \"/2\": offset or scale them all; " +
                          "a number sets it."
            onEditingFinished: {
                // Unchanged (e.g. the field left): no edit
                if (text !== root.commonValue(1))
                    root.setSelectedValue(1, text)
                text = Qt.binding(function() { return root.commonValue(1) })
            }
        }
        Label {
            visible: root.selection.length === 0
            text: "(select points)"
            font.pointSize: 7
            opacity: 0.5
        }
        // Interpolation of the selected points
        Repeater {
            model: [["smooth", MaterialIcons.timeline, "Smooth: the curve goes through the point with a slope"],
                    ["linear", MaterialIcons.show_chart, "Linear: no slope, a corner (straight lines to the neighbors)"],
                    ["constant", MaterialIcons.stairs, "Constant: the value is held until the next point (a step)"]]
            MaterialToolButton {
                required property var modelData
                visible: root.selection.length > 0
                text: modelData[1]
                font.pointSize: 9
                padding: 2
                checkable: true
                checked: root.commonMode === modelData[0]
                enabled: root.editable
                ToolTip.text: modelData[2]
                onClicked: root.setSelectedMode(modelData[0])
            }
        }
        MaterialToolButton {
            visible: root.selection.length > 0
            text: MaterialIcons.delete_outline
            font.pointSize: 9
            padding: 2
            enabled: root.editable
            ToolTip.text: "Delete the selected points (Delete)"
            onClicked: root.deleteSelection()
        }
    }

    // Plot
    Rectangle {
        Layout.fillWidth: true
        Layout.fillHeight: root.inWindow
        Layout.preferredHeight: root.inWindow ? -1 : root.plotHeight
        Layout.minimumHeight: 100
        color: Qt.rgba(0, 0, 0, 0.25)
        border.color: plotMouse.activeFocus ? Qt.rgba(1, 1, 1, 0.25) : Qt.rgba(1, 1, 1, 0.1)

        Canvas {
            id: plot
            anchors.fill: parent
            anchors.margins: 1
            onWidthChanged: requestPaint()
            onHeightChanged: requestPaint()

            onPaint: {
                var ctx = getContext("2d")
                ctx.reset()
                if (width <= 0 || height <= 0 || !root.live)
                    return
                var b = root.bounds
                // Grid: 0.25 steps for the colors, about 10 lines for the frames
                ctx.strokeStyle = Qt.rgba(1, 1, 1, 0.07)
                ctx.lineWidth = 1
                var step = root.timeCurve ? Math.max(1, Math.pow(10, Math.floor(Math.log10((b.x1 - b.x0) / 2)))) : 0.25
                for (var gx = Math.ceil(b.x0 / step) * step; gx <= b.x1; gx += step) {
                    ctx.beginPath(); ctx.moveTo(root.toScreenX(gx), 0); ctx.lineTo(root.toScreenX(gx), height); ctx.stroke()
                }
                for (var gy = Math.ceil(b.y0 / step) * step; gy <= b.y1; gy += step) {
                    ctx.beginPath(); ctx.moveTo(0, root.toScreenY(gy)); ctx.lineTo(width, root.toScreenY(gy)); ctx.stroke()
                }
                // Identity
                ctx.strokeStyle = Qt.rgba(1, 1, 1, 0.15)
                ctx.setLineDash([3, 3])
                ctx.beginPath()
                ctx.moveTo(root.toScreenX(b.x0), root.toScreenY(b.x0))
                ctx.lineTo(root.toScreenX(b.x1), root.toScreenY(b.x1))
                ctx.stroke()
                ctx.setLineDash([])

                // Picked source and target colors, on the active color curve
                if (root.hasPickers) {
                    var channel = {"red": "r", "green": "g", "blue": "b", "alpha": "a"}[root.active]
                    if (channel) {
                        var sx = root.toScreenX(root.node.attribute("sourceColor").value.get(channel).value)
                        var ty = root.toScreenY(root.node.attribute("targetColor").value.get(channel).value)
                        ctx.strokeStyle = Qt.rgba(1, 1, 1, 0.35)
                        ctx.setLineDash([2, 4])
                        ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, height); ctx.stroke()
                        ctx.beginPath(); ctx.moveTo(0, ty); ctx.lineTo(width, ty); ctx.stroke()
                        ctx.setLineDash([])
                    }
                }

                // Curves: the active one last, on top
                var order = root.names.filter(function(n) { return n !== root.active })
                if (root.active)
                    order.push(root.active)
                var count = Math.max(2, Math.round(width / 2))
                order.forEach(function(name) {
                    var isActive = name === root.active
                    var values = root.live.sampleCurve(root.sortedPoints(name), b.x0, b.x1, count)
                    ctx.strokeStyle = root.colors[name] || "white"
                    ctx.globalAlpha = isActive ? 1.0 : 0.35
                    ctx.lineWidth = isActive ? 1.6 : 1
                    ctx.beginPath()
                    for (var i = 0; i < values.length; i++) {
                        var px = i / (values.length - 1) * width
                        var py = root.toScreenY(values[i])
                        if (i === 0) ctx.moveTo(px, py)
                        else ctx.lineTo(px, py)
                    }
                    ctx.stroke()
                })
                ctx.globalAlpha = 1.0
                // Points of the active curve: a square (smooth), a diamond (linear), a hollow square (constant);
                // the selected ones white and larger
                var curveColor = root.colors[root.active] || "white"
                root.sortedPoints(root.active).forEach(function(p, index) {
                    var selected = root.isSelected(index)
                    var size = selected ? 4.5 : 3.5
                    var x = root.toScreenX(p[0]), y = root.toScreenY(p[1])
                    var mode = root.modeOf(p)
                    ctx.fillStyle = selected ? "white" : curveColor
                    ctx.strokeStyle = selected ? "white" : curveColor
                    ctx.lineWidth = 1.5
                    if (mode === "linear") {
                        ctx.beginPath()
                        ctx.moveTo(x, y - size - 1); ctx.lineTo(x + size + 1, y); ctx.lineTo(x, y + size + 1); ctx.lineTo(x - size - 1, y)
                        ctx.closePath()
                        ctx.fill()
                    } else if (mode === "constant") {
                        ctx.strokeRect(x - size, y - size, 2 * size, 2 * size)
                    } else {
                        ctx.fillRect(x - size, y - size, 2 * size, 2 * size)
                    }
                })
                // Box selection
                if (plotMouse.boxStart) {
                    ctx.fillStyle = Qt.rgba(1, 1, 1, 0.08)
                    ctx.strokeStyle = Qt.rgba(1, 1, 1, 0.6)
                    ctx.lineWidth = 1
                    var bx = Math.min(plotMouse.boxStart.x, plotMouse.boxEnd.x), by = Math.min(plotMouse.boxStart.y, plotMouse.boxEnd.y)
                    var bw = Math.abs(plotMouse.boxEnd.x - plotMouse.boxStart.x), bh = Math.abs(plotMouse.boxEnd.y - plotMouse.boxStart.y)
                    ctx.fillRect(bx, by, bw, bh)
                    ctx.strokeRect(bx, by, bw, bh)
                }
            }
        }

        MouseArea {
            id: plotMouse
            anchors.fill: plot
            enabled: root.active !== ""
            hoverEnabled: true
            acceptedButtons: Qt.LeftButton | Qt.MiddleButton
            // Vertical drags move the points, not the attribute list around
            preventStealing: true
            // The keys (Delete, Ctrl + A) go to the plot once clicked
            activeFocusOnTab: true
            // Points when the drag started, and where it started
            property var startPoints: []
            property real startX
            property real startY
            property bool dragging: false
            // Panning (middle button or Alt + left): last position
            property bool panning: false
            property real lastX
            property real lastY
            property bool editing: false
            // Box selection (Ctrl + drag on an empty area): its corners on screen, null when not selecting
            property var boxStart: null
            property point boxEnd
            property bool boxAdds: false

            onWheel: function(wheel) {
                // In the attribute editor, the wheel alone scrolls the editor: Ctrl + wheel zooms
                if (!root.inWindow && !(wheel.modifiers & Qt.ControlModifier)) {
                    wheel.accepted = false
                    return
                }
                var delta = wheel.angleDelta.y !== 0 ? wheel.angleDelta.y : wheel.angleDelta.x
                var factor = delta > 0 ? 0.8 : 1.25
                var horizontal = !(wheel.modifiers & Qt.AltModifier)
                var vertical = !(wheel.modifiers & Qt.ShiftModifier)
                root.zoom(factor, root.toCurveX(wheel.x), root.toCurveY(wheel.y), horizontal, vertical)
            }

            function hit(mx, my) {
                var points = root.sortedPoints(root.active)
                for (var i = 0; i < points.length; i++) {
                    if (Math.abs(root.toScreenX(points[i][0]) - mx) < 6 && Math.abs(root.toScreenY(points[i][1]) - my) < 6)
                        return i
                }
                return -1
            }

            // The selected points of `startPoints` moved by (dx, dy) curve units, each kept between its unselected
            // neighbors (the x order does not change)
            function moved(dx, dy) {
                var points = root.copy(startPoints)
                var low = -Infinity, high = Infinity
                for (var i = 0; i < points.length; i++) {
                    if (!root.isSelected(i))
                        continue
                    var previous = i - 1
                    while (previous >= 0 && root.isSelected(previous))
                        previous--
                    var next = i + 1
                    while (next < points.length && root.isSelected(next))
                        next++
                    if (previous >= 0)
                        low = Math.max(low, points[previous][0] + 1e-3 - points[i][0])
                    if (next < points.length)
                        high = Math.min(high, points[next][0] - 1e-3 - points[i][0])
                }
                dx = Math.max(low, Math.min(high, dx))
                if (root.timeCurve) {
                    dx = Math.round(dx)
                    dx = Math.max(Math.ceil(low), Math.min(Math.floor(high), dx))
                    dy = Math.round(dy * 100) / 100
                }
                for (var j = 0; j < points.length; j++) {
                    if (root.isSelected(j))
                        points[j] = root.withXY(points[j], points[j][0] + dx, points[j][1] + dy)
                }
                return points
            }

            onPressed: function(mouse) {
                forceActiveFocus()
                panning = mouse.button === Qt.MiddleButton || (mouse.modifiers & Qt.AltModifier)
                editing = false
                if (panning) {
                    lastX = mouse.x
                    lastY = mouse.y
                    return
                }
                var index = hit(mouse.x, mouse.y)
                if (index < 0 && (mouse.modifiers & Qt.ControlModifier)) {
                    // Box selection
                    boxAdds = !!(mouse.modifiers & Qt.ShiftModifier)
                    if (!boxAdds)
                        root.selection = []
                    boxStart = Qt.point(mouse.x, mouse.y)
                    boxEnd = boxStart
                    return
                }
                if (index >= 0) {
                    if (mouse.modifiers & Qt.ShiftModifier)
                        root.selection = root.isSelected(index) ? root.selection.filter(function(i) { return i !== index })
                                                                : root.selection.concat([index])
                    else if (!root.isSelected(index))
                        root.selection = [index]
                    if (!root.editable || !root.isSelected(index))
                        return
                    editing = true
                    root.frozenBounds = root.computeBounds()
                    startPoints = root.sortedPoints(root.active)
                    startX = mouse.x
                    startY = mouse.y
                    dragging = false
                    root.begin()
                    return
                }
                if (mouse.modifiers & Qt.ShiftModifier)
                    return
                if (!root.editable) {
                    root.selection = []
                    return
                }
                // New point, then dragged
                editing = true
                root.frozenBounds = root.computeBounds()
                var x = root.toCurveX(mouse.x), y = root.toCurveY(mouse.y)
                if (root.timeCurve) {
                    x = Math.round(x)
                    y = Math.round(y * 100) / 100
                }
                var points = root.sortedPoints(root.active).filter(function(p) { return Math.abs(p[0] - x) > 1e-6 })
                points.push([x, y])
                points.sort(function(a, b) { return a[0] - b[0] })
                root.selection = [points.findIndex(function(p) { return p[0] === x })]
                root.begin()
                var newCurves = root.copy(root.curves)
                newCurves[root.active] = points
                root.apply(newCurves)
                startPoints = points
                startX = mouse.x
                startY = mouse.y
                dragging = false
            }
            onPositionChanged: function(mouse) {
                if (pressed && panning) {
                    var b = root.bounds
                    root.pan((mouse.x - lastX) / plot.width * (b.x1 - b.x0), -(mouse.y - lastY) / plot.height * (b.y1 - b.y0))
                    lastX = mouse.x
                    lastY = mouse.y
                    return
                }
                if (pressed && boxStart) {
                    boxEnd = Qt.point(mouse.x, mouse.y)
                    plot.requestPaint()
                    return
                }
                if (!pressed || !editing)
                    return
                if (!dragging && Math.abs(mouse.x - startX) < 2 && Math.abs(mouse.y - startY) < 2)
                    return
                dragging = true
                var b2 = root.bounds
                var dx = (mouse.x - startX) / plot.width * (b2.x1 - b2.x0)
                var dy = -(mouse.y - startY) / plot.height * (b2.y1 - b2.y0)
                var newCurves = root.copy(root.curves)
                newCurves[root.active] = moved(dx, dy)
                root.apply(newCurves)
            }
            function finish() {
                panning = false
                if (boxStart) {
                    var x0 = Math.min(boxStart.x, boxEnd.x), x1 = Math.max(boxStart.x, boxEnd.x)
                    var y0 = Math.min(boxStart.y, boxEnd.y), y1 = Math.max(boxStart.y, boxEnd.y)
                    var inside = []
                    root.sortedPoints(root.active).forEach(function(p, i) {
                        var sx = root.toScreenX(p[0]), sy = root.toScreenY(p[1])
                        if (sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1 && !root.isSelected(i))
                            inside.push(i)
                    })
                    root.selection = root.selection.concat(inside)
                    boxStart = null
                    plot.requestPaint()
                    return
                }
                if (!editing)
                    return
                editing = false
                dragging = false
                root.end()
                root.frozenBounds = null
            }
            onReleased: finish()
            onCanceled: finish()
            onDoubleClicked: function(mouse) {
                if (!root.editable || mouse.button !== Qt.LeftButton || (mouse.modifiers & (Qt.AltModifier | Qt.ControlModifier)))
                    return
                var index = hit(mouse.x, mouse.y)
                if (index < 0)
                    return
                root.selection = [index]
                root.deleteSelection()
            }
            Keys.onPressed: function(event) {
                if ((event.key === Qt.Key_Delete || event.key === Qt.Key_Backspace) && root.selection.length) {
                    root.deleteSelection()
                    event.accepted = true
                } else if (event.key === Qt.Key_A && (event.modifiers & Qt.ControlModifier)) {
                    root.selection = root.sortedPoints(root.active).map(function(p, i) { return i })
                    event.accepted = true
                } else if (event.key === Qt.Key_Escape && root.selection.length) {
                    root.selection = []
                    event.accepted = true
                }
            }
        }

        // Coordinates under the cursor
        Label {
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.margins: 3
            visible: plotMouse.containsMouse
            font.pointSize: 7
            opacity: 0.7
            text: {
                var decimals = root.timeCurve ? 1 : 3
                return root.toCurveX(plotMouse.mouseX).toFixed(decimals) + ", " + root.toCurveY(plotMouse.mouseY).toFixed(decimals)
            }
        }
    }

    // Pickers and color matching (lookup nodes)
    RowLayout {
        visible: root.hasPickers
        Layout.fillWidth: true
        spacing: 4
        Button {
            text: root.node && root.node.nodeType === "ColorLookupHSV" ? "Add H / S / V points: Source → Target"
                                                                       : "Add R / G / B points: Source → Target"
            font.pointSize: 8
            enabled: root.editable
            ToolTip.text: "Add on the curves the points moving the Source Color to the Target Color.\n" +
                          "Pick them in the viewer with their eyedroppers (Source: in the image entering the node, " +
                          "Target: in the displayed image), then click; repeat with other colors to refine."
            ToolTip.visible: hovered
            onClicked: _currentScene.addLookupPoints(root.node)
        }
        Button {
            text: "Match"
            font.pointSize: 8
            visible: root.canMatch
            enabled: root.editable
            ToolTip.text: "Set the curves so that the colors of the input match those of the 'Match Target' input (current frame)"
            ToolTip.visible: hovered
            onClicked: root.live.matchColors(root.node)
        }
        Item { Layout.fillWidth: true }
    }

    // Resize bar: drag it to make the editor taller or shorter (kept for the session)
    Rectangle {
        visible: !root.inWindow
        Layout.fillWidth: true
        Layout.preferredHeight: 8
        radius: 2
        color: resizeMouse.containsMouse || resizeMouse.pressed ? Qt.rgba(1, 1, 1, 0.18) : Qt.rgba(1, 1, 1, 0.05)
        Label {
            anchors.centerIn: parent
            text: MaterialIcons.drag_handle
            font.family: MaterialIcons.fontFamily
            font.pointSize: 8
            opacity: resizeMouse.containsMouse || resizeMouse.pressed ? 0.9 : 0.4
        }
        MouseArea {
            id: resizeMouse
            anchors.fill: parent
            hoverEnabled: true
            preventStealing: true
            cursorShape: Qt.SizeVerCursor
            property real startY
            property real startHeight
            ToolTip.visible: containsMouse && !pressed
            ToolTip.delay: 600
            ToolTip.text: "Drag to resize the curve editor"
            onPressed: function(mouse) {
                startY = mapToItem(null, mouse.x, mouse.y).y
                startHeight = root.plotHeight
            }
            onPositionChanged: function(mouse) {
                if (pressed)
                    root.plotHeight = Math.max(100, Math.min(1600, startHeight + mapToItem(null, mouse.x, mouse.y).y - startY))
            }
            onReleased: {
                if (root.live)
                    root.live.setEditorSize(root.sizeKey, root.plotHeight)
            }
        }
    }

    // The curves in their own window (same attribute: both editors stay in sync), with the colors of Meshroom
    Loader {
        id: curveWindowLoader
        active: false
        sourceComponent: ApplicationWindow {
            id: curveWindow
            title: root.attribute.node.label + " - " + root.attribute.label
            width: 900
            height: 700
            minimumWidth: 300
            minimumHeight: 240
            visible: true
            color: palette.window
            onClosing: curveWindowLoader.active = false

            // The palette of the application (QPalette is not convertible to a QML palette: copied, as main.qml)
            Component.onCompleted: {
                if (typeof _PaletteManager === "undefined")
                    return
                palette.alternateBase = _PaletteManager.alternateBase
                palette.base = _PaletteManager.base
                palette.button = _PaletteManager.button
                palette.buttonText = _PaletteManager.buttonText
                palette.highlight = _PaletteManager.highlight
                palette.highlightedText = _PaletteManager.highlightedText
                palette.link = _PaletteManager.link
                palette.mid = _PaletteManager.mid
                palette.shadow = _PaletteManager.shadow
                palette.text = _PaletteManager.text
                palette.toolTipBase = _PaletteManager.toolTipBase
                palette.toolTipText = _PaletteManager.toolTipText
                palette.window = _PaletteManager.window
                palette.windowText = _PaletteManager.windowText
                palette.disabled.buttonText = _PaletteManager.disabledButtonText
                palette.disabled.text = _PaletteManager.disabledText
                palette.disabled.windowText = _PaletteManager.disabledWindowText
            }

            Pane {
                anchors.fill: parent
                padding: 8
                Loader {
                    anchors.fill: parent
                    Component.onCompleted: setSource("CurveEditor.qml", {
                        "attribute": root.attribute,
                        "editable": Qt.binding(function() { return root.editable }),
                        "inWindow": true
                    })
                }
            }
            Shortcut {
                sequence: "Escape"
                enabled: curveWindow.visibility === Window.FullScreen
                onActivated: curveWindow.visibility = Window.Windowed
            }
        }
    }
}
