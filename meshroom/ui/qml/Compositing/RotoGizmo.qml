import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MaterialIcons 2.2

/**
 * Shape editing of a selected Roto node, drawn over the viewer (screen coordinates).
 *
 * Tools (toolbar on the left):
 * - Select: click a point or a curve, or drag a selection box on an empty area, to select points
 *   (Shift: add to the selection); drag the selected points, a tangent handle (Shift: move it
 *   alone, a broken tangent) or a curve (moves its shape); double-click a point to make it smooth /
 *   a corner; Delete removes the selected points (a shape left with less than 3 points is removed,
 *   an open curve with less than 2). With 2 selected points or more, a transform box (as in Nuke):
 *   drag inside it to move the selection, its square handles to scale it (from the opposite side;
 *   Ctrl: from the center; Shift on a corner: uniformly), its round handle to rotate it.
 * - Bézier: click to add a corner point, click and drag to add a smooth point. Click the first point
 *   to close the shape; Enter or another tool finishes it as it is set (closed by default, the
 *   "open / closed" button makes it an open curve, drawn as a stroke of its width).
 * - Rectangle / Ellipse: drag a box.
 * Alt + drag or the middle button pans the viewer.
 *
 * Animation: with Auto Key (on by default), a new shape gets a key at the current frame, and any
 * change made at a frame adds or updates the key of that frame for the edited shapes; the points
 * are interpolated between the keys (meshroom/live/roto.py). Shapes linked to a track are shown
 * where the track moves them.
 *
 * The shapes are the JSON value of the "shapes" attribute; each press and release is one
 * undoable change.
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
    // Width of the displayed image format: tracks made on another resolution are scaled to it
    property int formatWidth: 0
    property int frame: 1

    property string tool: "select"
    property bool autoKey: true
    // Selected points: {shapeIndex: [pointIndex, ...]}
    property var selection: ({})
    // Shape showing its tangents (the last one clicked), -1 if none
    property int activeShape: -1
    // Index of the shape being drawn with the Bézier tool, -1 if none
    property int drawingShape: -1
    // The mouse is over a point, a tangent or a curve (Select tool)
    property bool overShape: false

    readonly property var shapesAttribute: node ? node.attribute("shapes") : null
    readonly property var shapes: parseJson(shapesAttribute ? shapesAttribute.value : "[]")
    // Tracks of the Tracker connected to the "tracker" input (to show the linked shapes where they are)
    // Tracks of the Trackers connected to the "tracker" inputs: {input name: [track, ...]}
    readonly property var tracks: {
        var result = {}
        if (!node || !scene || !scene.live)
            return result
        ;["tracker", "tracker2", "tracker3"].forEach(function(name) {
            if (!node.hasAttribute(name))
                return
            var input = node.attribute(name)
            input.isLink  // Re-evaluated when the link changes
            // The node whose values the Roto reads: through Dot, Clone and disabled nodes (as the render)
            var tracker = scene.live.upstreamValuesNode(input)
            result[name] = (tracker && tracker.hasAttribute("tracks") ? parseJson(tracker.attribute("tracks").value) : [])
                .map(scaledTrack)
        })
        return result
    }
    // The shapes as displayed at the current frame: [{shape, points}]
    readonly property var displayed: shapes.map(function(shape) { return displayedPoints(shape) })

    // Transform box of the selected points (Nuke coordinates), when 2 points or more are selected
    readonly property var selectionBox: {
        if (tool !== "select")
            return null
        var count = 0, x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
        Object.keys(selection).forEach(function(key) {
            var points = displayed[Number(key)] || []
            selection[key].forEach(function(i) {
                if (i >= points.length)
                    return
                count++
                x0 = Math.min(x0, points[i].x); x1 = Math.max(x1, points[i].x)
                y0 = Math.min(y0, points[i].y); y1 = Math.max(y1, points[i].y)
            })
        })
        return count >= 2 ? { "x0": x0, "y0": y0, "x1": x1, "y1": y1, "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2 } : null
    }
    // Screen margin around the transform box, and distance of its rotation handle above it
    readonly property real boxMargin: 8
    readonly property real rotateDistance: 26

    readonly property color curveColor: "#E07B62"
    readonly property color selectedColor: "#F5C26B"
    readonly property real hitRadius: 8

    visible: !!node && formatHeight > 0

    onNodeChanged: {
        // Forget the shape being drawn first: it belongs to the previous node
        drawingShape = -1
        selection = {}
        activeShape = -1
        tool = "select"
    }
    onShapesChanged: {
        if (activeShape >= shapes.length)
            activeShape = -1
        if (drawingShape >= shapes.length)
            drawingShape = -1
        overlay.requestPaint()
    }
    onDisplayedChanged: overlay.requestPaint()
    onCanvasXChanged: overlay.requestPaint()
    onCanvasYChanged: overlay.requestPaint()
    onZoomChanged: overlay.requestPaint()
    onSelectionChanged: overlay.requestPaint()
    onSelectionBoxChanged: overlay.requestPaint()
    onActiveShapeChanged: overlay.requestPaint()
    onToolChanged: {
        if (tool !== "bezier")
            finishDrawing()
    }

    // A track in the pixels of the displayed image (as meshroom/live/tracking.py scaleTracks)
    function scaledTrack(track) {
        if (!track.format || !formatWidth || !formatHeight)
            return track
        var sx = formatWidth / track.format[0], sy = formatHeight / track.format[1]
        if (sx === 1 && sy === 1)
            return track
        var scaled = JSON.parse(JSON.stringify(track))
        Object.keys(scaled.keys || {}).forEach(function(f) {
            scaled.keys[f] = [scaled.keys[f][0] * sx, scaled.keys[f][1] * sy]
        })
        return scaled
    }

    function parseJson(text) {
        try {
            var parsed = JSON.parse(text || "[]")
            return Array.isArray(parsed) ? parsed : []
        } catch (e) {
            return []
        }
    }

    // Coordinates

    function sx(x) { return canvasX + x * zoom }
    function sy(y) { return canvasY + (formatHeight - y) * zoom }
    function nukeX(screenX) { return (screenX - canvasX) / zoom }
    function nukeY(screenY) { return formatHeight - (screenY - canvasY) / zoom }

    // Animation (same rules as meshroom/live/roto.py)

    function copy(value) { return JSON.parse(JSON.stringify(value)) }

    function sortedKeys(shape) {
        return (shape.keys || []).filter(function(key) { return key.points && key.points.length })
                                 .sort(function(a, b) { return a.frame - b.frame })
    }

    function pointsAt(shape, frame) {
        var keys = sortedKeys(shape)
        if (keys.length === 0)
            return shape.points || []
        if (frame <= keys[0].frame)
            return keys[0].points
        if (frame >= keys[keys.length - 1].frame)
            return keys[keys.length - 1].points
        for (var i = 0; i + 1 < keys.length; i++) {
            var a = keys[i], b = keys[i + 1]
            if (a.frame <= frame && frame <= b.frame) {
                if (a.points.length !== b.points.length)
                    return a.points
                var t = (frame - a.frame) / Math.max(1e-9, b.frame - a.frame)
                return a.points.map(function(p, index) {
                    var q = b.points[index], r = {}
                    ;["x", "y", "inX", "inY", "outX", "outY"].forEach(function(k) {
                        r[k] = (1 - t) * (p[k] || 0) + t * (q[k] || 0)
                    })
                    return r
                })
            }
        }
        return keys[keys.length - 1].points
    }

    function trackPosition(track, frame) {
        var keys = Object.keys(track.keys || {}).map(function(f) { return [Number(f), track.keys[f]] })
                                               .sort(function(a, b) { return a[0] - b[0] })
        if (keys.length === 0)
            return null
        if (frame <= keys[0][0])
            return keys[0][1]
        if (frame >= keys[keys.length - 1][0])
            return keys[keys.length - 1][1]
        for (var i = 0; i + 1 < keys.length; i++) {
            if (keys[i][0] <= frame && frame <= keys[i + 1][0]) {
                var t = (frame - keys[i][0]) / (keys[i + 1][0] - keys[i][0])
                return [keys[i][1][0] + t * (keys[i + 1][1][0] - keys[i][1][0]),
                        keys[i][1][1] + t * (keys[i + 1][1][1] - keys[i][1][1])]
            }
        }
        return keys[keys.length - 1][1]
    }

    // Offset of a linked shape at the current frame: {"point": index or -1, "dx", "dy"}, or null
    // Track links of a shape (the former single "link" goes to the "tracker" input)
    function shapeLinks(shape) {
        var links = (shape.links || []).slice()
        if (shape.link)
            links.push(Object.assign({ "input": "tracker" }, shape.link))
        return links.filter(function(link) { return link && link.track })
    }

    // Offset [dx, dy] of each point of a shape at the current frame, from its track links (as roto.py):
    // a point with its own link follows it instead of the links of the whole shape
    function pointOffsets(shape, count) {
        var whole = [0, 0]
        var own = {}
        shapeLinks(shape).forEach(function(link) {
            var track = (tracks[link.input || "tracker"] || []).find(function(t) { return t.name === link.track })
            if (!track || track.enabled === false)
                return
            var start = trackPosition(track, link.frame !== undefined ? link.frame : frame)
            var current = trackPosition(track, frame)
            if (!start || !current)
                return
            var index = link.point === undefined || link.point === null ? -1 : link.point
            var target = index < 0 ? whole : (own[index] = own[index] || [0, 0])
            target[0] += current[0] - start[0]
            target[1] += current[1] - start[1]
        })
        var offsets = []
        for (var i = 0; i < count; i++)
            offsets.push(own[i] || whole)
        return offsets
    }

    function displayedPoints(shape) {
        var points = copy(pointsAt(shape, frame))
        var offsets = pointOffsets(shape, points.length)
        points.forEach(function(p, index) {
            p.x += offsets[index][0]
            p.y += offsets[index][1]
        })
        return points
    }

    // Store points edited on screen into a shape (inverse of the link offsets; as a key with Auto Key or keys)
    function storePoints(shape, points) {
        var stored = copy(points)
        var offsets = pointOffsets(shape, stored.length)
        stored.forEach(function(p, index) {
            p.x -= offsets[index][0]
            p.y -= offsets[index][1]
        })
        var keys = shape.keys || []
        if (keys.length > 0 || autoKey) {
            var key = keys.find(function(k) { return k.frame === frame })
            if (key)
                key.points = stored
            else
                keys.push({ "frame": frame, "points": stored })
            shape.keys = keys.sort(function(a, b) { return a.frame - b.frame })
        }
        shape.points = stored
    }

    // Apply a function to the points of a shape in every key (adding / removing points)
    function forEachPointList(shape, func) {
        func(shape.points || [])
        ;(shape.keys || []).forEach(function(key) { func(key.points) })
    }

    // Edition: every change of a press is applied live, and recorded as one undo step on release

    function begin() { scene.beginInteractiveEdit(shapesAttribute) }
    function apply(newShapes) { scene.setValueInteractively(shapesAttribute, JSON.stringify(newShapes)) }
    function end() { scene.endInteractiveEdit(shapesAttribute) }
    function commit(newShapes) {
        begin()
        apply(newShapes)
        end()
    }

    function newShape(prefix, points) {
        var count = 1
        var names = shapes.map(function(s) { return s.name })
        while (names.indexOf(prefix + count) >= 0)
            count++
        var shape = { "name": prefix + count, "closed": true, "visible": true, "invert": false, "opacity": 1.0,
                      "feather": 0.0, "points": points }
        if (autoKey)
            shape.keys = [{ "frame": frame, "points": copy(points) }]
        return shape
    }
    function point(x, y, inX, inY, outX, outY) {
        return { "x": x, "y": y, "inX": inX || 0, "inY": inY || 0, "outX": outX || 0, "outY": outY || 0 }
    }
    // Box shapes (Nuke coordinates), as Bézier curves
    function boxPoints(kind, x0, y0, x1, y1) {
        var left = Math.min(x0, x1), right = Math.max(x0, x1)
        var bottom = Math.min(y0, y1), top = Math.max(y0, y1)
        if (kind === "rectangle")
            return [point(left, bottom), point(right, bottom), point(right, top), point(left, top)]
        // Ellipse: 4 points with tangents of 0.5523 times the radii
        var cx = (left + right) / 2, cy = (bottom + top) / 2
        var kx = (right - left) / 2 * 0.5523, ky = (top - bottom) / 2 * 0.5523
        return [point(right, cy, 0, -ky, 0, ky), point(cx, top, kx, 0, -kx, 0),
                point(left, cy, 0, ky, 0, -ky), point(cx, bottom, -kx, 0, kx, 0)]
    }

    function isSelected(shapeIndex, pointIndex) {
        var points = selection[shapeIndex]
        return !!points && points.indexOf(pointIndex) >= 0
    }
    function select(shapeIndex, pointIndices, add) {
        var newSelection = add ? copy(selection) : {}
        var current = newSelection[shapeIndex] || []
        pointIndices.forEach(function(i) {
            if (current.indexOf(i) < 0)
                current.push(i)
        })
        newSelection[shapeIndex] = current
        selection = newSelection
        activeShape = shapeIndex
    }
    function hasSelection() { return Object.keys(selection).length > 0 }

    // Finish the shape being drawn: dropped if it has less than 3 points
    function finishDrawing() {
        if (drawingShape < 0 || drawingShape >= shapes.length) {
            drawingShape = -1
            return
        }
        var newShapes = copy(shapes)
        var index = drawingShape
        drawingShape = -1
        if (displayed[index].length < (shapes[index].closed === false ? 2 : 3)) {
            newShapes.splice(index, 1)
            selection = {}
            activeShape = -1
            commit(newShapes)
        }
    }

    function deleteSelection() {
        if (!hasSelection())
            return
        var newShapes = copy(shapes)
        var removed = []
        Object.keys(selection).forEach(function(key) {
            var shapeIndex = Number(key)
            var shape = newShapes[shapeIndex]
            var indices = selection[key].slice().sort(function(a, b) { return b - a })
            forEachPointList(shape, function(points) {
                indices.forEach(function(i) { if (i < points.length) points.splice(i, 1) })
            })
            if ((shape.points || []).length < (shape.closed === false ? 2 : 3))
                removed.push(shapeIndex)
        })
        removed.sort(function(a, b) { return b - a }).forEach(function(i) { newShapes.splice(i, 1) })
        selection = {}
        activeShape = -1
        commit(newShapes)
    }

    // Toggle a point between a corner (no tangents) and a smooth point (tangents along its neighbors)
    function toggleSmooth(shapeIndex, pointIndex) {
        var newShapes = copy(shapes)
        var points = copy(displayed[shapeIndex])
        var p = points[pointIndex]
        if (Math.abs(p.inX) + Math.abs(p.inY) + Math.abs(p.outX) + Math.abs(p.outY) > 1e-6) {
            p.inX = p.inY = p.outX = p.outY = 0
        } else if (points.length >= 2) {
            var previous = points[(pointIndex - 1 + points.length) % points.length]
            var next = points[(pointIndex + 1) % points.length]
            var dx = next.x - previous.x, dy = next.y - previous.y
            var length = Math.max(1e-6, Math.hypot(dx, dy))
            var before = Math.hypot(p.x - previous.x, p.y - previous.y) / 3
            var after = Math.hypot(next.x - p.x, next.y - p.y) / 3
            p.inX = -dx / length * before; p.inY = -dy / length * before
            p.outX = dx / length * after; p.outY = dy / length * after
        }
        storePoints(newShapes[shapeIndex], points)
        commit(newShapes)
    }

    // Keys of the shapes with selected points (toolbar)
    function setKeyOnSelection() {
        var newShapes = copy(shapes)
        Object.keys(selection).forEach(function(key) {
            var shape = newShapes[Number(key)]
            var points = copy(displayed[Number(key)])
            var keys = shape.keys || []
            if (keys.length === 0)
                shape.keys = [{ "frame": frame, "points": copy(pointsAt(shape, frame)) }]
            else
                storePoints(shape, points)
        })
        commit(newShapes)
    }
    function deleteKeyOnSelection() {
        var newShapes = copy(shapes)
        Object.keys(selection).forEach(function(key) {
            var shape = newShapes[Number(key)]
            var current = copy(pointsAt(shape, frame))
            shape.keys = (shape.keys || []).filter(function(k) { return k.frame !== frame })
            if (shape.keys.length === 0) {
                delete shape.keys
                shape.points = current
            }
        })
        commit(newShapes)
    }

    // Open (curve) or closed (shape) state of the shapes with selected points, or of the shape being drawn
    readonly property bool selectionOpen: {
        var indices = drawingShape >= 0 ? [drawingShape] : Object.keys(selection).map(Number)
        return indices.length > 0 && indices.every(function(i) { return i < shapes.length && shapes[i].closed === false })
    }
    function setClosed(closed) {
        var indices = drawingShape >= 0 ? [drawingShape] : Object.keys(selection).map(Number)
        if (indices.length === 0)
            return
        var newShapes = copy(shapes)
        indices.forEach(function(i) {
            if (i < newShapes.length)
                newShapes[i].closed = closed
        })
        commit(newShapes)
    }
    function isClosed(shapeIndex) {
        return shapeIndex >= shapes.length || shapes[shapeIndex].closed !== false
    }
    // Number of curve segments of a shape: the closing one only for a closed shape
    function segmentCount(shapeIndex, pointCount) {
        return isClosed(shapeIndex) ? pointCount : Math.max(0, pointCount - 1)
    }

    // Transform box: screen rectangle, handles and their hit test
    function boxScreen() {
        var b = selectionBox
        if (!b)
            return null
        return { "left": sx(b.x0) - boxMargin, "right": sx(b.x1) + boxMargin,
                 "top": sy(b.y1) - boxMargin, "bottom": sy(b.y0) + boxMargin }
    }
    function boxHandles() {
        var r = boxScreen()
        if (!r)
            return []
        var mx = (r.left + r.right) / 2, my = (r.top + r.bottom) / 2
        return [
            { "name": "tl", "x": r.left, "y": r.top }, { "name": "t", "x": mx, "y": r.top },
            { "name": "tr", "x": r.right, "y": r.top }, { "name": "r", "x": r.right, "y": my },
            { "name": "br", "x": r.right, "y": r.bottom }, { "name": "b", "x": mx, "y": r.bottom },
            { "name": "bl", "x": r.left, "y": r.bottom }, { "name": "l", "x": r.left, "y": my },
            { "name": "rotate", "x": mx, "y": r.top - rotateDistance }
        ]
    }
    // "rotate", a scale handle ("tl", "t"...), "inside" or "" (not on the box)
    function hitBox(mx, my) {
        var r = boxScreen()
        if (!r)
            return ""
        var handles = boxHandles()
        for (var i = 0; i < handles.length; i++) {
            if (Math.abs(handles[i].x - mx) < 6 && Math.abs(handles[i].y - my) < 6)
                return handles[i].name
        }
        return mx > r.left && mx < r.right && my > r.top && my < r.bottom ? "inside" : ""
    }
    // Transform a point list (Nuke coordinates) with the linear part [a, b, c, d] around an origin, plus a translation
    function transformPoints(points, indices, ox, oy, a, b, c, d, tx, ty) {
        indices.forEach(function(i) {
            if (i >= points.length)
                return
            var p = points[i]
            var x = p.x - ox, y = p.y - oy
            p.x = ox + a * x + b * y + tx
            p.y = oy + c * x + d * y + ty
            var inX = p.inX, inY = p.inY, outX = p.outX, outY = p.outY
            p.inX = a * inX + b * inY; p.inY = c * inX + d * inY
            p.outX = a * outX + b * outY; p.outY = c * outX + d * outY
        })
    }
    // The transform of a drag of the box: {ox, oy, a, b, c, d, tx, ty}
    function boxTransform(edit, mouse) {
        var b = edit.box
        var x = nukeX(mouse.x), y = nukeY(mouse.y)
        var x0 = nukeX(edit.startX), y0 = nukeY(edit.startY)
        if (edit.handle === "inside")
            return { "ox": 0, "oy": 0, "a": 1, "b": 0, "c": 0, "d": 1, "tx": x - x0, "ty": y - y0 }
        if (edit.handle === "rotate") {
            var angle = Math.atan2(y - b.cy, x - b.cx) - Math.atan2(y0 - b.cy, x0 - b.cx)
            var cos = Math.cos(angle), sin = Math.sin(angle)
            return { "ox": b.cx, "oy": b.cy, "a": cos, "b": -sin, "c": sin, "d": cos, "tx": 0, "ty": 0 }
        }
        var h = edit.handle
        var usesX = h.indexOf("l") >= 0 || h.indexOf("r") >= 0
        var usesY = h.indexOf("t") >= 0 || h.indexOf("b") >= 0
        // Anchor: the opposite side (Ctrl: the center)
        var ox = edit.fromCenter ? b.cx : (h.indexOf("l") >= 0 ? b.x1 : b.x0)
        var oy = edit.fromCenter ? b.cy : (h.indexOf("t") >= 0 ? b.y0 : b.y1)
        var scaleX = usesX && Math.abs(x0 - ox) > 1e-6 ? (x - ox) / (x0 - ox) : 1
        var scaleY = usesY && Math.abs(y0 - oy) > 1e-6 ? (y - oy) / (y0 - oy) : 1
        if (edit.uniform && usesX && usesY) {
            // Projection of the mouse on the diagonal of the drag start
            var vx = x0 - ox, vy = y0 - oy
            var scale = ((x - ox) * vx + (y - oy) * vy) / Math.max(1e-9, vx * vx + vy * vy)
            scaleX = scaleY = scale
        }
        return { "ox": ox, "oy": oy, "a": scaleX, "b": 0, "c": 0, "d": scaleY, "tx": 0, "ty": 0 }
    }

    // Hit tests (screen coordinates)

    function hitHandle(mx, my) {
        // Tangent handles of the active shape first, then the points of every shape
        if (activeShape >= 0 && activeShape < displayed.length) {
            var active = displayed[activeShape]
            for (var i = 0; i < active.length; i++) {
                var p = active[i]
                if ((p.outX || p.outY) && Math.hypot(sx(p.x + p.outX) - mx, sy(p.y + p.outY) - my) < hitRadius)
                    return { "shape": activeShape, "point": i, "part": "out" }
                if ((p.inX || p.inY) && Math.hypot(sx(p.x + p.inX) - mx, sy(p.y + p.inY) - my) < hitRadius)
                    return { "shape": activeShape, "point": i, "part": "in" }
            }
        }
        for (var s = displayed.length - 1; s >= 0; s--) {
            var points = displayed[s]
            for (var j = 0; j < points.length; j++) {
                if (Math.hypot(sx(points[j].x) - mx, sy(points[j].y) - my) < hitRadius)
                    return { "shape": s, "point": j, "part": "point" }
            }
        }
        return null
    }

    function bezierAt(a, b, t) {
        var u = 1 - t
        var x = u * u * u * a.x + 3 * u * u * t * (a.x + a.outX) + 3 * u * t * t * (b.x + b.inX) + t * t * t * b.x
        var y = u * u * u * a.y + 3 * u * u * t * (a.y + a.outY) + 3 * u * t * t * (b.y + b.inY) + t * t * t * b.y
        return Qt.point(x, y)
    }

    function hitCurve(mx, my) {
        for (var s = displayed.length - 1; s >= 0; s--) {
            var points = displayed[s]
            for (var i = 0; i < segmentCount(s, points.length); i++) {
                var a = points[i], b = points[(i + 1) % points.length]
                for (var k = 0; k <= 16; k++) {
                    var q = bezierAt(a, b, k / 16)
                    if (Math.hypot(sx(q.x) - mx, sy(q.y) - my) < hitRadius * 0.75)
                        return s
                }
            }
        }
        return -1
    }

    // Drawing of the shapes

    Canvas {
        id: overlay
        anchors.fill: parent
        renderStrategy: Canvas.Cooperative

        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            for (var s = 0; s < root.displayed.length; s++) {
                var shape = root.shapes[s]
                var points = root.displayed[s]
                if (points.length === 0)
                    continue
                var active = s === root.activeShape
                var animated = (shape.keys || []).length > 0
                ctx.strokeStyle = active ? root.selectedColor : root.curveColor
                ctx.globalAlpha = shape.visible === false ? 0.35 : 1.0
                ctx.lineWidth = active ? 1.6 : 1.2
                // Closed path: the last point is joined to the first one (not for an open curve, nor while drawing)
                ctx.beginPath()
                ctx.moveTo(root.sx(points[0].x), root.sy(points[0].y))
                var segments = s === root.drawingShape ? points.length - 1 : root.segmentCount(s, points.length)
                for (var i = 0; i < segments; i++) {
                    var a = points[i], b = points[(i + 1) % points.length]
                    ctx.bezierCurveTo(root.sx(a.x + a.outX), root.sy(a.y + a.outY),
                                      root.sx(b.x + b.inX), root.sy(b.y + b.inY), root.sx(b.x), root.sy(b.y))
                }
                ctx.stroke()

                // Tangents of the active shape
                if (active) {
                    ctx.lineWidth = 1
                    ctx.strokeStyle = Qt.rgba(1, 1, 1, 0.6)
                    for (var t = 0; t < points.length; t++) {
                        var p = points[t]
                        var parts = [[p.inX, p.inY], [p.outX, p.outY]]
                        for (var h = 0; h < 2; h++) {
                            if (!parts[h][0] && !parts[h][1])
                                continue
                            var hx = root.sx(p.x + parts[h][0]), hy = root.sy(p.y + parts[h][1])
                            ctx.beginPath()
                            ctx.moveTo(root.sx(p.x), root.sy(p.y))
                            ctx.lineTo(hx, hy)
                            ctx.stroke()
                            ctx.fillStyle = "white"
                            ctx.beginPath()
                            ctx.arc(hx, hy, 3, 0, 2 * Math.PI)
                            ctx.fill()
                        }
                    }
                }
                // Points (diamonds on a key frame of an animated shape)
                var onKey = animated && (shape.keys || []).some(function(k) { return k.frame === root.frame })
                for (var j = 0; j < points.length; j++) {
                    var selected = root.isSelected(s, j)
                    var first = s === root.drawingShape && j === 0
                    ctx.fillStyle = selected ? root.selectedColor : (first ? "white" : root.curveColor)
                    var size = selected || first ? 7 : 5
                    var px = root.sx(points[j].x), py = root.sy(points[j].y)
                    if (onKey) {
                        ctx.beginPath()
                        ctx.moveTo(px, py - size * 0.7); ctx.lineTo(px + size * 0.7, py)
                        ctx.lineTo(px, py + size * 0.7); ctx.lineTo(px - size * 0.7, py)
                        ctx.closePath()
                        ctx.fill()
                    } else {
                        ctx.fillRect(px - size / 2, py - size / 2, size, size)
                    }
                }
            }
            // Transform box of the selected points
            var box = root.boxScreen()
            if (box) {
                ctx.globalAlpha = 1.0
                ctx.lineWidth = 1
                ctx.strokeStyle = Qt.rgba(1, 1, 1, 0.7)
                ctx.setLineDash([5, 4])
                ctx.strokeRect(box.left, box.top, box.right - box.left, box.bottom - box.top)
                ctx.setLineDash([])
                var handles = root.boxHandles()
                handles.forEach(function(handle) {
                    if (handle.name === "rotate") {
                        ctx.beginPath()
                        ctx.moveTo(handle.x, box.top)
                        ctx.lineTo(handle.x, handle.y + 5)
                        ctx.stroke()
                        ctx.beginPath()
                        ctx.arc(handle.x, handle.y, 5, 0, 2 * Math.PI)
                        ctx.stroke()
                    } else {
                        ctx.fillStyle = "white"
                        ctx.fillRect(handle.x - 3.5, handle.y - 3.5, 7, 7)
                    }
                })
                // Center
                var bcx = (box.left + box.right) / 2, bcy = (box.top + box.bottom) / 2
                ctx.beginPath()
                ctx.moveTo(bcx - 6, bcy); ctx.lineTo(bcx + 6, bcy)
                ctx.moveTo(bcx, bcy - 6); ctx.lineTo(bcx, bcy + 6)
                ctx.stroke()
            }
            // Selection box
            if (editArea.marquee) {
                var m = editArea.marquee
                ctx.globalAlpha = 1.0
                ctx.lineWidth = 1
                ctx.strokeStyle = root.selectedColor
                ctx.setLineDash([4, 3])
                ctx.strokeRect(Math.min(m.x0, m.x1), Math.min(m.y0, m.y1), Math.abs(m.x1 - m.x0), Math.abs(m.y1 - m.y0))
            }
        }
    }

    // Editing

    MouseArea {
        id: editArea
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton
        // No hover: the viewer under it keeps reading the pixel values (see the HoverHandler below)
        hoverEnabled: false
        cursorShape: root.tool === "select" ? (root.overShape ? Qt.PointingHandCursor : Qt.ArrowCursor) : Qt.CrossCursor
        // What the current drag edits: {"kind": "points" | "in" | "out" | "shape" | "newPoint" | "box", ...}
        property var edit: null
        // Selection box in screen coordinates {x0, y0, x1, y1}, or null
        property var marquee: null
        property var startShapes: []
        property var startDisplayed: []
        property point startMouse

        onWheel: function(wheel) { wheel.accepted = false }

        onPositionChanged: function(mouse) {
            if (!pressed)
                return
            if (marquee) {
                marquee = { "x0": marquee.x0, "y0": marquee.y0, "x1": mouse.x, "y1": mouse.y }
                overlay.requestPaint()
                return
            }
            if (!edit)
                return
            var shapes = root.copy(startShapes)
            var dx = (mouse.x - startMouse.x) / root.zoom, dy = -(mouse.y - startMouse.y) / root.zoom
            var x = root.nukeX(mouse.x), y = root.nukeY(mouse.y)
            if (edit.kind === "transform") {
                var m = root.boxTransform(edit, mouse)
                Object.keys(edit.moved).forEach(function(key) {
                    var shapeIndex = Number(key)
                    var points = root.copy(startDisplayed[shapeIndex])
                    root.transformPoints(points, edit.moved[key], m.ox, m.oy, m.a, m.b, m.c, m.d, m.tx, m.ty)
                    root.storePoints(shapes[shapeIndex], points)
                })
            } else if (edit.kind === "points" || edit.kind === "shape") {
                // Move the selected points (or every point of the shape)
                Object.keys(edit.moved).forEach(function(key) {
                    var shapeIndex = Number(key)
                    var points = root.copy(startDisplayed[shapeIndex])
                    edit.moved[key].forEach(function(i) {
                        if (i < points.length) {
                            points[i].x += dx
                            points[i].y += dy
                        }
                    })
                    root.storePoints(shapes[shapeIndex], points)
                })
            } else if (edit.kind === "in" || edit.kind === "out" || edit.kind === "newPoint") {
                var points = root.copy(startDisplayed[edit.shape])
                var q = points[edit.point]
                var tx = x - q.x, ty = y - q.y
                var own = edit.kind === "in" ? "in" : "out"
                var other = own === "in" ? "out" : "in"
                q[own + "X"] = tx; q[own + "Y"] = ty
                if (edit.kind === "newPoint" || !(mouse.modifiers & Qt.ShiftModifier)) {
                    // Smooth point: the opposite tangent is aligned, its length kept (mirrored for a new point)
                    var length = edit.kind === "newPoint" ? Math.hypot(tx, ty) : Math.hypot(q[other + "X"], q[other + "Y"])
                    var norm = Math.max(1e-6, Math.hypot(tx, ty))
                    q[other + "X"] = -tx / norm * length; q[other + "Y"] = -ty / norm * length
                }
                root.storePoints(shapes[edit.shape], points)
            } else if (edit.kind === "box") {
                root.storePoints(shapes[edit.shape], root.boxPoints(edit.box, edit.x0, edit.y0, x, y))
            }
            root.apply(shapes)
        }

        onPressed: function(mouse) {
            // Alt + drag pans the viewer, as the middle button
            if (mouse.modifiers & Qt.AltModifier) {
                mouse.accepted = false
                return
            }
            startMouse = Qt.point(mouse.x, mouse.y)
            startShapes = root.copy(root.shapes)
            startDisplayed = root.copy(root.displayed)
            edit = null
            marquee = null
            var x = root.nukeX(mouse.x), y = root.nukeY(mouse.y)
            var add = (mouse.modifiers & Qt.ShiftModifier) !== 0

            if (root.tool === "select") {
                var hit = root.hitHandle(mouse.x, mouse.y)
                // Transform box: its handles; inside it, a drag moves the selection when it starts on a selected
                // point, on a curve of a selected shape or on an empty area (a click there unselects); a point
                // that is not selected, a tangent, another shape or Shift (add to the selection) come first
                var boxHit = root.hitBox(mouse.x, mouse.y)
                var curveHit = boxHit === "inside" && !hit ? root.hitCurve(mouse.x, mouse.y) : -1
                var onSelected = !!hit && hit.part === "point" && root.isSelected(hit.shape, hit.point)
                var emptyInside = boxHit === "inside" && !hit && curveHit < 0
                var insideMove = boxHit === "inside" && !add &&
                                 (onSelected || emptyInside || (curveHit >= 0 && root.selection[curveHit] !== undefined))
                if ((boxHit !== "" && boxHit !== "inside") || insideMove) {
                    edit = { "kind": "transform", "handle": boxHit, "box": root.selectionBox, "moved": root.copy(root.selection),
                             "startX": mouse.x, "startY": mouse.y, "unselectOnClick": emptyInside,
                             "fromCenter": (mouse.modifiers & Qt.ControlModifier) !== 0,
                             "uniform": (mouse.modifiers & Qt.ShiftModifier) !== 0 }
                    root.begin()
                    return
                }
                if (hit && hit.part === "point") {
                    if (!root.isSelected(hit.shape, hit.point))
                        root.select(hit.shape, [hit.point], add)
                    root.activeShape = hit.shape
                    edit = { "kind": "points", "moved": root.copy(root.selection) }
                } else if (hit) {
                    edit = { "kind": hit.part, "shape": hit.shape, "point": hit.point }
                } else {
                    var curve = root.hitCurve(mouse.x, mouse.y)
                    if (curve < 0) {
                        // Empty area: selection box
                        if (!add)
                            root.selection = {}
                        marquee = { "x0": mouse.x, "y0": mouse.y, "x1": mouse.x, "y1": mouse.y }
                        return
                    }
                    var all = root.displayed[curve].map(function(p, i) { return i })
                    root.select(curve, all, add)
                    var moved = {}
                    moved[curve] = all
                    edit = { "kind": "shape", "moved": moved }
                }
                root.begin()
                return
            }

            root.begin()
            var shapes = root.copy(root.shapes)
            if (root.tool === "bezier") {
                if (root.drawingShape >= 0 && root.drawingShape < shapes.length) {
                    var drawn = root.displayed[root.drawingShape]
                    var first = drawn[0]
                    if (drawn.length >= 3 && Math.hypot(root.sx(first.x) - mouse.x, root.sy(first.y) - mouse.y) < root.hitRadius) {
                        // Click on the first point: the shape is closed and finished
                        if (shapes[root.drawingShape].closed === false) {
                            shapes[root.drawingShape].closed = true
                            root.apply(shapes)
                        }
                        root.drawingShape = -1
                        root.tool = "select"
                        return
                    }
                    var points = root.copy(drawn)
                    points.push(root.point(x, y))
                    root.storePoints(shapes[root.drawingShape], points)
                    // Every key gets the new point (the shape is drawn at one frame)
                    ;(shapes[root.drawingShape].keys || []).forEach(function(key) {
                        if (key.points.length < points.length)
                            key.points.push(root.point(x, y))
                    })
                } else {
                    shapes.push(root.newShape("Bezier", [root.point(x, y)]))
                    root.drawingShape = shapes.length - 1
                }
                var pointIndex = root.displayedPoints(shapes[root.drawingShape]).length - 1
                root.select(root.drawingShape, [pointIndex], false)
                edit = { "kind": "newPoint", "shape": root.drawingShape, "point": pointIndex }
            } else {
                // Rectangle / ellipse
                shapes.push(root.newShape(root.tool === "rectangle" ? "Rectangle" : "Ellipse",
                                          root.boxPoints(root.tool, x, y, x, y)))
                root.selection = {}
                root.activeShape = shapes.length - 1
                edit = { "kind": "box", "box": root.tool, "shape": shapes.length - 1, "x0": x, "y0": y }
            }
            root.apply(shapes)
            // The following moves start from this state
            startShapes = shapes
            startDisplayed = shapes.map(function(shape) { return root.displayedPoints(shape) })
        }

        onReleased: function(mouse) {
            if (marquee) {
                // Select the points inside the box
                var left = Math.min(marquee.x0, marquee.x1), right = Math.max(marquee.x0, marquee.x1)
                var top = Math.min(marquee.y0, marquee.y1), bottom = Math.max(marquee.y0, marquee.y1)
                var newSelection = (mouse.modifiers & Qt.ShiftModifier) ? root.copy(root.selection) : {}
                var lastShape = -1
                root.displayed.forEach(function(points, s) {
                    points.forEach(function(p, i) {
                        var px = root.sx(p.x), py = root.sy(p.y)
                        if (px >= left && px <= right && py >= top && py <= bottom) {
                            var list = newSelection[s] || []
                            if (list.indexOf(i) < 0)
                                list.push(i)
                            newSelection[s] = list
                            lastShape = s
                        }
                    })
                })
                root.selection = newSelection
                if (lastShape >= 0)
                    root.activeShape = lastShape
                marquee = null
                overlay.requestPaint()
                return
            }
            if (edit && edit.kind === "transform" && edit.unselectOnClick &&
                Math.hypot(mouse.x - startMouse.x, mouse.y - startMouse.y) < 3) {
                // A click (no drag) on an empty area of the box: unselect
                root.selection = {}
            }
            if (edit && edit.kind === "box") {
                // Drop the boxes that are too small (a click)
                if (Math.hypot(mouse.x - startMouse.x, mouse.y - startMouse.y) < 3) {
                    var shapes = root.copy(root.shapes)
                    shapes.splice(edit.shape, 1)
                    root.activeShape = -1
                    root.apply(shapes)
                } else {
                    var index = edit.shape
                    root.tool = "select"
                    root.select(index, root.displayed[index].map(function(p, i) { return i }), false)
                }
            }
            edit = null
            root.end()
        }

        // The press was taken by another item: keep what was done, as one undo step
        onCanceled: {
            edit = null
            marquee = null
            root.end()
        }

        onDoubleClicked: function(mouse) {
            if (root.tool !== "select")
                return
            var hit = root.hitHandle(mouse.x, mouse.y)
            if (hit && hit.part === "point")
                root.toggleSmooth(hit.shape, hit.point)
        }
    }

    HoverHandler {
        id: gizmoHover
        onPointChanged: root.overShape = root.tool === "select" && hovered &&
                        (root.hitHandle(point.position.x, point.position.y) !== null ||
                         root.hitBox(point.position.x, point.position.y) !== "" ||
                         root.hitCurve(point.position.x, point.position.y) >= 0)
    }
    Shortcut {
        sequences: ["Delete", "Backspace"]
        enabled: root.visible && gizmoHover.hovered && root.hasSelection()
        onActivated: root.deleteSelection()
    }
    Shortcut {
        sequences: ["Return", "Enter"]
        enabled: root.visible && root.drawingShape >= 0
        onActivated: root.tool = "select"
    }

    // Toolbar, above the timeline
    Rectangle {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 6
        width: toolRow.implicitWidth + 8
        height: toolRow.implicitHeight + 6
        radius: 4
        color: Qt.rgba(0, 0, 0, 0.6)

        RowLayout {
            id: toolRow
            anchors.centerIn: parent
            spacing: 2

            Label { text: "Roto"; font.bold: true; leftPadding: 4; rightPadding: 6 }

            Repeater {
                model: [
                    { "tool": "select", "icon": MaterialIcons.near_me, "tip": "Select: click, or drag a box (Shift: add), then drag the points" },
                    { "tool": "bezier", "icon": MaterialIcons.gesture, "tip": "Bézier: click for corners, drag for smooth points; Enter or the first point to finish" },
                    { "tool": "rectangle", "icon": MaterialIcons.crop_square, "tip": "Rectangle: drag a box" },
                    { "tool": "ellipse", "icon": MaterialIcons.panorama_fish_eye, "tip": "Ellipse: drag a box" }
                ]
                MaterialToolButton {
                    required property var modelData
                    font.pointSize: 16
                    text: modelData.icon
                    checked: root.tool === modelData.tool
                    onClicked: root.tool = modelData.tool
                    ToolTip.text: modelData.tip
                }
            }
            MaterialToolButton {
                font.pointSize: 16
                text: root.selectionOpen ? MaterialIcons.timeline : MaterialIcons.pentagon
                enabled: root.drawingShape >= 0 || root.hasSelection()
                onClicked: root.setClosed(root.selectionOpen)
                ToolTip.text: root.selectionOpen ? "Open curve (drawn as a stroke of its width): click to close it"
                                                 : "Closed shape (filled): click to make it an open curve"
            }
            Rectangle { Layout.fillHeight: true; Layout.preferredWidth: 1; color: Qt.rgba(1, 1, 1, 0.2) }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.key
                checked: root.autoKey
                onClicked: root.autoKey = !root.autoKey
                ToolTip.text: root.autoKey ? "Auto Key: on (changes are keyed at the current frame)" :
                                             "Auto Key: off (shapes without keys are not animated)"
            }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.add_location_alt
                enabled: root.hasSelection()
                onClicked: root.setKeyOnSelection()
                ToolTip.text: "Set a key at the current frame on the selected shapes"
            }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.key_off
                enabled: root.hasSelection()
                onClicked: root.deleteKeyOnSelection()
                ToolTip.text: "Delete the key of the current frame on the selected shapes"
            }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.delete_outline
                enabled: root.hasSelection()
                onClicked: root.deleteSelection()
                ToolTip.text: "Delete the selected points (Delete)"
            }
        }
    }
}
