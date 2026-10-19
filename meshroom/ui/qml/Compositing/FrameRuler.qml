import QtQuick

/**
 * Frame ruler of the timeline, as in Nuke: graduations with the frame numbers, the current frame
 * (orange), the keys of the selected node (yellow marks) and the frames in the RAM cache (green bar).
 * Click or drag to change the frame.
 */
Item {
    id: root

    property int first: 1
    property int last: 100
    property int current: 1
    property var cachedFrames: []
    property var keyFrames: []

    signal frameRequested(int frame)

    implicitHeight: 34

    readonly property real margin: 12
    readonly property real span: Math.max(1, last - first)

    function xOf(frame) { return margin + (frame - first) / span * (width - 2 * margin) }
    function frameAt(x) { return Math.round(first + (x - margin) / Math.max(1, width - 2 * margin) * span) }

    onFirstChanged: ruler.requestPaint()
    onLastChanged: ruler.requestPaint()
    onCurrentChanged: ruler.requestPaint()
    onCachedFramesChanged: ruler.requestPaint()
    onKeyFramesChanged: ruler.requestPaint()
    onWidthChanged: ruler.requestPaint()

    Canvas {
        id: ruler
        anchors.fill: parent

        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            var w = width, h = height
            ctx.fillStyle = Qt.rgba(0, 0, 0, 0.25)
            ctx.fillRect(0, 0, w, h)

            // Graduations: labels at least 60 pixels apart
            var pixelsPerFrame = (w - 2 * root.margin) / root.span
            var steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000]
            var step = steps[steps.length - 1]
            for (var i = 0; i < steps.length; i++) {
                if (steps[i] * pixelsPerFrame >= 60) {
                    step = steps[i]
                    break
                }
            }
            var minor = step >= 10 ? step / 5 : (step >= 2 ? 1 : 0)
            ctx.strokeStyle = Qt.rgba(1, 1, 1, 0.25)
            ctx.lineWidth = 1
            if (minor > 0 && minor * pixelsPerFrame >= 4) {
                for (var f = Math.ceil(root.first / minor) * minor; f <= root.last; f += minor) {
                    var mx = Math.round(root.xOf(f)) + 0.5
                    ctx.beginPath(); ctx.moveTo(mx, h - 6); ctx.lineTo(mx, h - 2); ctx.stroke()
                }
            }
            ctx.font = "10px sans-serif"
            ctx.fillStyle = Qt.rgba(1, 1, 1, 0.65)
            ctx.strokeStyle = Qt.rgba(1, 1, 1, 0.45)
            var labels = []
            for (var g = Math.ceil(root.first / step) * step; g <= root.last; g += step)
                labels.push(g)
            // The first and last frames are always written
            if (labels.indexOf(root.first) < 0)
                labels.unshift(root.first)
            if (labels.indexOf(root.last) < 0)
                labels.push(root.last)
            labels.forEach(function(frame) {
                var x = Math.round(root.xOf(frame)) + 0.5
                ctx.beginPath(); ctx.moveTo(x, h - 12); ctx.lineTo(x, h - 2); ctx.stroke()
                var text = String(frame)
                var tw = ctx.measureText(text).width
                ctx.fillText(text, Math.max(0, Math.min(w - tw, x - tw / 2)), 12)
            })

            // Frames in the RAM cache
            ctx.fillStyle = "#5FA36A"
            root.cachedFrames.forEach(function(frame) {
                if (frame < root.first || frame > root.last)
                    return
                var x0 = root.xOf(frame - 0.5), x1 = root.xOf(frame + 0.5)
                ctx.fillRect(x0, h - 3, Math.max(1, x1 - x0), 2)
            })

            // Keys of the selected node
            ctx.fillStyle = "#E8C547"
            root.keyFrames.forEach(function(frame) {
                if (frame < root.first || frame > root.last)
                    return
                var x = root.xOf(frame)
                ctx.fillRect(Math.round(x) - 1, 15, 3, h - 21)
            })

            // Current frame
            var cx = Math.round(root.xOf(root.current)) + 0.5
            ctx.strokeStyle = "#F0883E"
            ctx.lineWidth = 2
            ctx.beginPath(); ctx.moveTo(cx, 14); ctx.lineTo(cx, h); ctx.stroke()
            var label = String(root.current)
            var lw = ctx.measureText(label).width + 8
            var lx = Math.max(0, Math.min(w - lw, cx - lw / 2))
            ctx.fillStyle = "#F0883E"
            ctx.fillRect(lx, 0, lw, 14)
            ctx.fillStyle = "black"
            ctx.fillText(label, lx + 4, 11)
        }
    }

    MouseArea {
        anchors.fill: parent
        preventStealing: true
        onPressed: function(mouse) { root.frameRequested(Math.max(root.first, Math.min(root.last, root.frameAt(mouse.x)))) }
        onPositionChanged: function(mouse) {
            if (pressed)
                root.frameRequested(Math.max(root.first, Math.min(root.last, root.frameAt(mouse.x))))
        }
    }
}
