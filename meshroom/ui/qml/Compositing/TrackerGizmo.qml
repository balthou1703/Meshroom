import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MaterialIcons 2.2

/**
 * Tracks of a selected Tracker node, drawn over the viewer (screen coordinates): position at the
 * current frame (cross), pattern box, search area of the selected tracks (dashed) and path.
 *
 * - toolbar "Add track", then click on a feature: a new track with a key at the current frame;
 * - click a track to select it (Shift: add to the selection), drag it to set its key at the
 *   current frame;
 * - track the selected tracks (all the enabled ones if none is selected) backward or forward from
 *   the current frame; stop keeps what was tracked;
 * - Delete removes the selected tracks.
 * The frames where a track was placed by hand are its keyframes (strip at the top of the viewer);
 * the zoom window (top left) magnifies the selected track at the current frame. "Error" colors
 * the paths by the match score of each tracked frame (green: good, red: poor).
 * Each press and release is one undoable change of the "tracks" attribute (JSON, see
 * meshroom/live/tracking.py).
 */
Item {
    id: root

    property var node: null
    property var scene: null
    property var live: null
    property real canvasX: 0
    property real canvasY: 0
    property real zoom: 1
    property int formatHeight: 0
    property int frame: 1

    // Tracks shown but not edited (CameraTracker: the autotrack)
    property bool readOnly: false
    // Pins of a Resection or a FaceTracker: tracks stuck to a point of its geometry ("point"), added by clicking on it
    readonly property bool pinMode: !!node && (node.nodeType === "Resection" || node.nodeType === "FaceTracker")
    // Resection only: the geometry track (Track Geometry, Align, Refine)
    readonly property bool geoMode: !!node && node.nodeType === "Resection"
    // FaceTracker only: the face tracked by MediaPipe / the optical flow
    readonly property bool faceMode: !!node && node.nodeType === "FaceTracker"
    readonly property string itemName: pinMode ? "pin" : "track"
    property bool adding: false
    property var selectedNames: []
    // Color the paths by the match score of the tracked frames
    property bool showError: false
    readonly property var selectedTrack: {
        if (selectedNames.length === 0)
            return null
        var name = selectedNames[selectedNames.length - 1]
        return tracks.find(function(t) { return t.name === name }) || null
    }

    // The edited tracks: the pins of a FaceTracker ("tracks" are its landmarks), else "tracks"
    readonly property var tracksAttribute: node ? node.attribute(node.hasAttribute("pins") ? "pins" : "tracks") : null
    // Width of the displayed image format: tracks made on another resolution are scaled to it
    property int formatWidth: 0
    // The tracks as stored (each in the format of the image it was tracked on)
    readonly property var rawTracks: {
        try {
            var parsed = JSON.parse(tracksAttribute ? tracksAttribute.value || "[]" : "[]")
            return Array.isArray(parsed) ? parsed : []
        } catch (e) {
            return []
        }
    }
    // The tracks in the pixels of the displayed image (as meshroom/live/tracking.py scaleTracks)
    readonly property var tracks: rawTracks.map(function(track) {
        var s = scaleOf(track)
        if (s[0] === 1 && s[1] === 1)
            return track
        var scaled = JSON.parse(JSON.stringify(track))
        Object.keys(scaled.keys || {}).forEach(function(f) {
            scaled.keys[f] = [scaled.keys[f][0] * s[0], scaled.keys[f][1] * s[1]]
        })
        scaled.pattern = Math.max(5, Math.round((track.pattern || 21) * s[0]))
        scaled.search = Math.max(7, Math.round((track.search || 61) * s[0]))
        return scaled
    })
    // Scale (x, y) from the format of a track to the displayed image
    function scaleOf(track) {
        if (!track.format || !formatWidth || !formatHeight)
            return [1, 1]
        return [formatWidth / track.format[0], formatHeight / track.format[1]]
    }
    readonly property bool tracking: !!live && live.trackingProgress >= 0
    readonly property color trackColor: "#6FD0E0"
    readonly property color selectedColor: "#F5C26B"

    visible: !!node && formatHeight > 0

    onNodeChanged: {
        adding = false
        selectedNames = []
    }
    onTracksChanged: overlay.requestPaint()
    onPinProjectionsChanged: overlay.requestPaint()
    onFrameChanged: overlay.requestPaint()
    onCanvasXChanged: overlay.requestPaint()
    onCanvasYChanged: overlay.requestPaint()
    onZoomChanged: overlay.requestPaint()
    onSelectedNamesChanged: overlay.requestPaint()
    onShowErrorChanged: overlay.requestPaint()

    // Color of a match score: red (0.6 and less) to green (0.98 and more)
    function scoreColor(score) {
        var t = Math.max(0, Math.min(1, (score - 0.6) / 0.38))
        return Qt.rgba(1 - t, 0.25 + 0.6 * t, 0.2, 1)
    }

    // A frame placed by hand becomes a keyframe of the track
    function addKeyframe(track, frame) {
        var keyframes = track.keyframes || []
        if (keyframes.indexOf(frame) < 0)
            keyframes.push(frame)
        track.keyframes = keyframes.sort(function(a, b) { return a - b })
    }

    function sx(x) { return canvasX + x * zoom }
    function sy(y) { return canvasY + (formatHeight - y) * zoom }
    function nukeX(screenX) { return (screenX - canvasX) / zoom }
    function nukeY(screenY) { return formatHeight - (screenY - canvasY) / zoom }
    function copy(value) { return JSON.parse(JSON.stringify(value)) }

    function sortedKeys(track) {
        return Object.keys(track.keys || {}).map(function(f) { return [Number(f), track.keys[f]] })
                                           .sort(function(a, b) { return a[0] - b[0] })
    }
    // Pins (as in GeoTracker): a pin constrains only the frames where it has a key; at the other frames it is shown
    // where the solved geometry puts its point (dragging it there adds a key)
    readonly property var pinProjections: {
        if (!pinMode || !live)
            return ({})
        live.statusRevision  // Refreshed with the renders
        frame
        rawTracks
        var projections = live.pinProjections(node)
        // In the displayed format (computed at the full resolution of the plate)
        var s = live.imageWidth && formatWidth ? formatWidth / live.imageWidth : 1
        var result = {}
        for (var name in projections)
            result[name] = [projections[name][0] * s, projections[name][1] * s]
        return result
    }
    function hasKey(track, frame) { return !!track.keys && track.keys[String(frame)] !== undefined }

    function position(track, frame) {
        if (pinMode && frame === root.frame && !hasKey(track, frame)) {
            // FaceTracker (as in KeenTools FaceTracker): the pins belong to their keyframes, hidden elsewhere
            if (faceMode)
                return null
            if (pinProjections[track.name])
                return pinProjections[track.name]
        }
        var keys = sortedKeys(track)
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
    function isSelected(name) { return selectedNames.indexOf(name) >= 0 }

    function begin() { scene.beginInteractiveEdit(tracksAttribute) }
    function apply(newTracks) { scene.setValueInteractively(tracksAttribute, JSON.stringify(newTracks)) }
    function end() { scene.endInteractiveEdit(tracksAttribute) }
    function commit(newTracks) {
        begin()
        apply(newTracks)
        end()
    }

    function hitTrack(mx, my) {
        for (var i = tracks.length - 1; i >= 0; i--) {
            var p = position(tracks[i], frame)
            if (!p)
                continue
            var half = Math.max(8, (tracks[i].pattern || 21) * zoom / 2)
            if (Math.abs(sx(p[0]) - mx) <= half && Math.abs(sy(p[1]) - my) <= half)
                return i
        }
        return -1
    }

    function deleteSelection() {
        commit(rawTracks.filter(function(t) { return !root.isSelected(t.name) }))
        selectedNames = []
    }

    function deleteKeyOnSelection() {
        var newTracks = copy(rawTracks)
        newTracks.forEach(function(t) {
            if (root.isSelected(t.name) && t.keys) {
                delete t.keys[String(root.frame)]
                if (t.scores)
                    delete t.scores[String(root.frame)]
                t.keyframes = (t.keyframes || []).filter(function(f) { return f !== root.frame })
            }
        })
        commit(newTracks)
    }

    Canvas {
        id: overlay
        anchors.fill: parent
        renderStrategy: Canvas.Cooperative

        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            root.tracks.forEach(function(track) {
                var selected = root.isSelected(track.name)
                var color = track.enabled === false ? "#808080" : (selected ? root.selectedColor : root.trackColor)
                ctx.strokeStyle = color
                ctx.fillStyle = color
                // Path (colored by the match scores with "Error")
                var keys = root.sortedKeys(track)
                // The pins of a FaceTracker belong to their keyframes: no path between them
                if (keys.length > 1 && !root.faceMode) {
                    ctx.globalAlpha = root.showError ? 0.9 : 0.5
                    ctx.lineWidth = root.showError ? 2 : 1
                    var scores = track.scores || {}
                    for (var k = 1; k < keys.length; k++) {
                        var score = scores[String(keys[k][0])]
                        ctx.strokeStyle = root.showError && score !== undefined ? root.scoreColor(score) : color
                        ctx.beginPath()
                        ctx.moveTo(root.sx(keys[k - 1][1][0]), root.sy(keys[k - 1][1][1]))
                        ctx.lineTo(root.sx(keys[k][1][0]), root.sy(keys[k][1][1]))
                        ctx.stroke()
                    }
                    ctx.strokeStyle = color
                    // Keyframes (placed by hand)
                    ;(track.keyframes || []).forEach(function(frame) {
                        var p = root.position(track, frame)
                        if (p)
                            ctx.fillRect(root.sx(p[0]) - 2.5, root.sy(p[1]) - 2.5, 5, 5)
                    })
                }
                ctx.globalAlpha = 1.0
                var p = root.position(track, root.frame)
                if (!p)
                    return
                var x = root.sx(p[0]), y = root.sy(p[1])
                var onKey = (track.keys || {})[String(root.frame)] !== undefined
                // A pin without a key at this frame (it does not constrain it): faint, where the geometry puts it
                if (root.pinMode && !onKey)
                    ctx.globalAlpha = 0.45
                // Pattern
                var pattern = (track.pattern || 21) * root.zoom
                ctx.lineWidth = onKey ? 1.5 : 1
                ctx.setLineDash([])
                ctx.strokeRect(x - pattern / 2, y - pattern / 2, pattern, pattern)
                // Search area
                if (selected) {
                    var search = (track.search || 61) * root.zoom
                    ctx.setLineDash([4, 3])
                    ctx.strokeRect(x - search / 2, y - search / 2, search, search)
                    ctx.setLineDash([])
                }
                // Center
                ctx.beginPath()
                ctx.moveTo(x - 5, y); ctx.lineTo(x + 5, y)
                ctx.moveTo(x, y - 5); ctx.lineTo(x, y + 5)
                ctx.stroke()
                if (!root.readOnly) {
                    ctx.font = "11px sans-serif"
                    ctx.fillText(track.name, x + pattern / 2 + 3, y - pattern / 2)
                }
            })
        }
    }

    MouseArea {
        id: trackMouse
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton
        hoverEnabled: false
        enabled: !root.tracking && !root.readOnly
        cursorShape: root.adding ? Qt.CrossCursor : Qt.ArrowCursor
        property int dragged: -1
        property point startMouse
        property var startPosition: null
        // Box selection (Ctrl + drag on an empty area): its corners on screen, null when not selecting
        property var boxStart: null
        property point boxEnd

        onWheel: function(wheel) { wheel.accepted = false }

        onPressed: function(mouse) {
            if (mouse.modifiers & Qt.AltModifier) {
                mouse.accepted = false
                return
            }
            var x = root.nukeX(mouse.x), y = root.nukeY(mouse.y)
            if (root.adding) {
                var point = null
                if (root.pinMode) {
                    // The point of the geometry under the click, seen from the solved camera
                    point = root.live.pickGeometryPoint(root.node, x, y)
                    if (point.length !== 3)
                        return  // Not on the geometry: still adding
                }
                var count = 1
                var names = root.tracks.map(function(t) { return t.name })
                while (names.indexOf(root.itemName + count) >= 0)
                    count++
                var keys = {}
                keys[String(root.frame)] = [x, y]
                var newTracks = root.copy(root.rawTracks)
                // Tracked in the format of the displayed image (keys scaled when used on another resolution)
                var newTrack = { "name": root.itemName + count, "enabled": true, "pattern": 21, "search": 61, "keys": keys,
                                 "keyframes": [root.frame], "format": [root.formatWidth, root.formatHeight] }
                if (point)
                    newTrack["point"] = point
                newTracks.push(newTrack)
                root.begin()
                root.apply(newTracks)
                root.selectedNames = [root.itemName + count]
                dragged = newTracks.length - 1
                startMouse = Qt.point(mouse.x, mouse.y)
                startPosition = [x, y]
                root.adding = false
                return
            }
            var hit = root.hitTrack(mouse.x, mouse.y)
            if (hit < 0) {
                if (mouse.modifiers & Qt.ControlModifier) {
                    // Box selection (added to the selection with Shift)
                    if (!(mouse.modifiers & Qt.ShiftModifier))
                        root.selectedNames = []
                    boxStart = Qt.point(mouse.x, mouse.y)
                    boxEnd = boxStart
                    return
                }
                if (!(mouse.modifiers & Qt.ShiftModifier))
                    root.selectedNames = []
                // Let the viewer pan
                mouse.accepted = false
                return
            }
            var name = root.tracks[hit].name
            if (mouse.modifiers & Qt.ShiftModifier)
                root.selectedNames = root.isSelected(name) ? root.selectedNames.filter(function(n) { return n !== name })
                                                           : root.selectedNames.concat([name])
            else if (!root.isSelected(name))
                root.selectedNames = [name]
            dragged = hit
            startMouse = Qt.point(mouse.x, mouse.y)
            startPosition = root.position(root.tracks[hit], root.frame)
            root.begin()
        }
        onPositionChanged: function(mouse) {
            if (pressed && boxStart) {
                boxEnd = Qt.point(mouse.x, mouse.y)
                return
            }
            if (!pressed || dragged < 0 || !startPosition)
                return
            var newTracks = root.copy(root.rawTracks)
            var track = newTracks[dragged]
            if (!track)
                return
            track.keys = track.keys || {}
            // Displayed position, written in the format of the track
            var s = root.scaleOf(track)
            track.keys[String(root.frame)] = [(startPosition[0] + (mouse.x - startMouse.x) / root.zoom) / s[0],
                                              (startPosition[1] - (mouse.y - startMouse.y) / root.zoom) / s[1]]
            root.addKeyframe(track, root.frame)
            // Placed by hand: no match score
            if (track.scores)
                delete track.scores[String(root.frame)]
            root.apply(newTracks)
        }
        onReleased: {
            if (boxStart) {
                var x0 = Math.min(boxStart.x, boxEnd.x), x1 = Math.max(boxStart.x, boxEnd.x)
                var y0 = Math.min(boxStart.y, boxEnd.y), y1 = Math.max(boxStart.y, boxEnd.y)
                var inside = []
                root.tracks.forEach(function(t) {
                    var p = root.position(t, root.frame)
                    if (p && root.sx(p[0]) >= x0 && root.sx(p[0]) <= x1 && root.sy(p[1]) >= y0 && root.sy(p[1]) <= y1
                            && !root.isSelected(t.name))
                        inside.push(t.name)
                })
                root.selectedNames = root.selectedNames.concat(inside)
                boxStart = null
                return
            }
            dragged = -1
            root.end()
        }
        onCanceled: {
            if (boxStart) {
                boxStart = null
                return
            }
            dragged = -1
            root.end()
        }
    }

    // The box being drawn (Ctrl + drag)
    Rectangle {
        visible: !!trackMouse.boxStart
        x: trackMouse.boxStart ? Math.min(trackMouse.boxStart.x, trackMouse.boxEnd.x) : 0
        y: trackMouse.boxStart ? Math.min(trackMouse.boxStart.y, trackMouse.boxEnd.y) : 0
        width: trackMouse.boxStart ? Math.abs(trackMouse.boxEnd.x - trackMouse.boxStart.x) : 0
        height: trackMouse.boxStart ? Math.abs(trackMouse.boxEnd.y - trackMouse.boxStart.y) : 0
        color: Qt.rgba(0.96, 0.76, 0.42, 0.12)
        border.color: root.selectedColor
    }

    function selectAll() { selectedNames = tracks.map(function(t) { return t.name }) }

    HoverHandler { id: trackerHover }
    Shortcut {
        sequence: StandardKey.SelectAll
        enabled: root.visible && trackerHover.hovered && !root.tracking
        onActivated: root.selectAll()
    }
    Shortcut {
        sequences: ["Delete", "Backspace"]
        enabled: root.visible && trackerHover.hovered && root.selectedNames.length > 0 && !root.tracking
        onActivated: root.deleteSelection()
    }

    // Zoom window (top left): the selected track magnified at the current frame
    Rectangle {
        id: zoomWindow
        x: 6
        y: keyframeStrip.visible ? keyframeStrip.y + keyframeStrip.height + 6 : 40
        width: 180
        height: 180
        visible: !!root.selectedTrack && !!root.live
        color: "black"
        border.color: root.selectedColor
        readonly property int patchSize: root.selectedTrack ? (root.selectedTrack.search || 61) : 61
        Image {
            anchors.fill: parent
            anchors.margins: 1
            smooth: false
            cache: false
            source: {
                if (!zoomWindow.visible)
                    return ""
                root.live.statusRevision  // Refreshed when frames are rendered
                return root.live.trackPatch(root.node, root.selectedTrack.name, root.frame, zoomWindow.patchSize)
            }
        }
        // Pattern box and center
        Rectangle {
            readonly property real ratio: (zoomWindow.width - 2) / zoomWindow.patchSize
            width: (root.selectedTrack ? (root.selectedTrack.pattern || 21) : 21) * ratio
            height: width
            anchors.centerIn: parent
            color: "transparent"
            border.color: root.selectedColor
        }
        Rectangle { width: 9; height: 1; color: root.selectedColor; anchors.centerIn: parent }
        Rectangle { width: 1; height: 9; color: root.selectedColor; anchors.centerIn: parent }
        Label {
            anchors.left: parent.left
            anchors.bottom: parent.bottom
            anchors.margins: 3
            text: root.selectedTrack ? root.selectedTrack.name + "  " + root.frame : ""
            font.pointSize: 8
            background: Rectangle { color: Qt.rgba(0, 0, 0, 0.5) }
        }
    }

    // Keyframes of the selected track (frames placed by hand), as in Nuke: click to go there
    Row {
        id: keyframeStrip
        x: 6
        y: 40
        spacing: 3
        visible: !!root.selectedTrack && (root.selectedTrack.keyframes || []).length > 0
        Repeater {
            model: root.selectedTrack ? (root.selectedTrack.keyframes || []) : []
            Rectangle {
                required property int modelData
                width: 64
                height: 64
                color: "black"
                border.color: modelData === root.frame ? root.selectedColor : Qt.rgba(1, 1, 1, 0.3)
                border.width: modelData === root.frame ? 2 : 1
                Image {
                    anchors.fill: parent
                    anchors.margins: 2
                    smooth: false
                    cache: false
                    source: {
                        root.live.statusRevision
                        return root.live.trackPatch(root.node, root.selectedTrack.name, parent.modelData,
                                                    (root.selectedTrack.pattern || 21) * 2)
                    }
                }
                Label {
                    anchors.bottom: parent.bottom
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: parent.modelData
                    font.pointSize: 7
                    background: Rectangle { color: Qt.rgba(0, 0, 0, 0.5) }
                }
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: {
                        root.live.stop()
                        root.live.currentFrame = parent.modelData
                    }
                }
            }
        }
    }

    // Toolbar, above the timeline
    Rectangle {
        visible: !root.readOnly
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 6
        width: trackRow.implicitWidth + 8
        height: trackRow.implicitHeight + 6
        radius: 4
        color: Qt.rgba(0, 0, 0, 0.6)

        RowLayout {
            id: trackRow
            anchors.centerIn: parent
            spacing: 2

            Label { text: root.pinMode ? root.node.nodeType : "Tracker"; font.bold: true; leftPadding: 4; rightPadding: 6 }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.add_location_alt
                checked: root.adding
                enabled: !root.tracking
                onClicked: root.adding = !root.adding
                ToolTip.text: root.pinMode ? "Add a pin: click on the " + (root.geoMode ? "geometry" : "face") + ", then drag the pin onto its place in the plate"
                                           : "Add a track: click on a feature to track"
            }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.select_all
                enabled: root.tracks.length > 0
                checked: root.tracks.length > 0 && root.selectedNames.length === root.tracks.length
                onClicked: checked ? (root.selectedNames = []) : root.selectAll()
                ToolTip.text: "Select all the " + root.itemName + "s (Ctrl+A), or none. Ctrl + drag: select in a box, " +
                              "Shift + click: add to the selection. The Track buttons track the selected ones together " +
                              "(all of them when none is selected)"
            }
            Label {
                text: root.selectedNames.length > 0 ? root.selectedNames.length + "/" + root.tracks.length
                                                    : (root.tracks.length > 0 ? "all" : "")
                opacity: 0.7
            }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.fast_rewind
                enabled: !root.tracking && root.tracks.length > 0
                         && !(root.live && root.live.actionProgress >= 0)
                visible: !root.faceMode  // The pins of a FaceTracker are not tracked: the face is
                onClicked: root.live.track(root.node, -1, root.selectedNames)
                ToolTip.text: "Track backward from the current frame (selected " + root.itemName + "s, or all)"
            }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.chevron_left
                enabled: !root.tracking && root.tracks.length > 0
                         && !(root.live && root.live.actionProgress >= 0)
                visible: !root.faceMode  // The pins of a FaceTracker are not tracked: the face is
                onClicked: root.live.trackStep(root.node, -1, root.selectedNames)
                ToolTip.text: "Track one frame backward (selected " + root.itemName + "s, or all)"
            }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.stop_circle
                enabled: root.tracking
                onClicked: root.live.stopTracking()
                ToolTip.text: "Stop tracking (the tracked frames are kept)"
            }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.chevron_right
                enabled: !root.tracking && root.tracks.length > 0
                         && !(root.live && root.live.actionProgress >= 0)
                visible: !root.faceMode  // The pins of a FaceTracker are not tracked: the face is
                onClicked: root.live.trackStep(root.node, 1, root.selectedNames)
                ToolTip.text: "Track one frame forward (selected " + root.itemName + "s, or all)"
            }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.fast_forward
                enabled: !root.tracking && root.tracks.length > 0
                         && !(root.live && root.live.actionProgress >= 0)
                visible: !root.faceMode  // The pins of a FaceTracker are not tracked: the face is
                onClicked: root.live.track(root.node, 1, root.selectedNames)
                ToolTip.text: "Track forward from the current frame (selected " + root.itemName + "s, or all)"
            }
            ProgressBar {
                Layout.preferredWidth: 60
                visible: root.tracking
                from: 0
                to: 1
                value: root.live ? Math.max(0, root.live.trackingProgress) : 0
            }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.key_off
                enabled: !root.tracking && root.selectedNames.length > 0
                onClicked: root.deleteKeyOnSelection()
                ToolTip.text: "Delete the key of the current frame of the selected tracks"
            }
            // Keys removed after / before the current frame (its key kept), or the tracked ones (keyframes kept)
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.first_page
                enabled: !root.tracking && !!root.live && root.live.actionProgress < 0
                onClicked: root.live.deleteKeys(root.node, root.selectedNames, "backward")
                ToolTip.text: "Delete the keys before the current frame (its key kept), of the selected tracks or all" +
                              (root.geoMode ? ", and the geometry track before it" : "")
            }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.last_page
                enabled: !root.tracking && !!root.live && root.live.actionProgress < 0
                onClicked: root.live.deleteKeys(root.node, root.selectedNames, "forward")
                ToolTip.text: "Delete the keys after the current frame (its key kept), of the selected tracks or all" +
                              (root.geoMode ? ", and the geometry track after it" : "")
            }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.auto_delete
                enabled: !root.tracking && !!root.live && root.live.actionProgress < 0
                onClicked: root.live.deleteKeys(root.node, root.selectedNames, "tracked")
                ToolTip.text: "Delete the tracked keys, keeping the keyframes placed by hand, of the selected tracks or all" +
                              (root.geoMode ? " (and the tracked poses of the geometry, its keyframes kept)" : "")
            }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.delete_outline
                enabled: !root.tracking && root.selectedNames.length > 0
                onClicked: root.deleteSelection()
                ToolTip.text: root.pinMode ? "Delete the selected pins (Delete)" : "Delete the selected tracks (Delete)"
            }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.insights
                checked: root.showError
                onClicked: root.showError = !root.showError
                ToolTip.text: "Error: color the paths by match quality (green: good, red: poor)"
            }
            // Resection: start from the camera stepped back to see the whole geometry, centered
            MaterialToolButton {
                readonly property var attribute: root.geoMode && root.node.hasAttribute("centerGeometry")
                                                 ? root.node.attribute("centerGeometry") : null
                visible: !!attribute
                font.pointSize: 16
                text: MaterialIcons.center_focus_strong
                checkable: true
                checked: !!attribute && attribute.value
                enabled: !root.tracking
                onClicked: _currentScene.setAttribute(attribute, !attribute.value)
                ToolTip.text: "Center Geometry: step the starting camera back along its view axis until the whole geometry " +
                              "is centered in the view (large geometry around the camera)"
            }
            // FaceTracker: the face itself (Track With: MediaPipe, flow, or both), backward / forward, image by image
            Rectangle { visible: root.faceMode; width: 1; Layout.preferredHeight: 20; color: Qt.rgba(1, 1, 1, 0.3) }
            Label { visible: root.faceMode; text: MaterialIcons.face; font.family: MaterialIcons.fontFamily; font.pointSize: 14 }
            Repeater {
                model: root.faceMode ? [["trackFaceBackward", MaterialIcons.skip_previous, "Track the face backward, to the previous keyframe or the start of the track range"],
                                        ["trackFaceStepBackward", MaterialIcons.navigate_before, "Track the face one frame backward"],
                                        ["trackFaceStepForward", MaterialIcons.navigate_next, "Track the face one frame forward"],
                                        ["trackFaceForward", MaterialIcons.skip_next, "Track the face forward, to the next keyframe or the end of the track range"],
                                        ["refineFace", MaterialIcons.auto_fix_high, "Refine: track the face again between each pair of keyframes, from both ends, and blend"],
                                        ["fitShape", MaterialIcons.face_retouching_natural, "Fit Shape: the shape of the actor's face from the tracked frames (one rest face for the shot, the rig then only changes its expression)"],
                                        ["setReference", MaterialIcons.star, "Reference frame: the current frame (where the face is seen at its best) becomes the base of the fit of the other frames"]] : []
                delegate: MaterialToolButton {
                    font.pointSize: 16
                    text: modelData[1]
                    enabled: !root.tracking && !!root.live && root.live.actionProgress < 0
                    onClicked: root.live.runAction(root.node, modelData[0])
                    ToolTip.text: modelData[2] + " (Track With: " + root.node.attribute("trackWith").value + ")"
                }
            }
            // Resection: the geometry itself tracked by many points of its surface (optical flow / patches)
            Rectangle { visible: root.geoMode; width: 1; Layout.preferredHeight: 20; color: Qt.rgba(1, 1, 1, 0.3) }
            MaterialToolButton {
                visible: root.geoMode
                font.pointSize: 16
                text: MaterialIcons.skip_previous
                enabled: !root.tracking && !!root.live && root.live.actionProgress < 0
                onClicked: root.live.runAction(root.node, "trackGeoBackward")
                ToolTip.text: "Track Geometry backward: many points of the geometry follow the plate, from the current " +
                              "frame back to the previous keyframe"
            }
            MaterialToolButton {
                visible: root.geoMode
                font.pointSize: 16
                text: MaterialIcons.skip_next
                enabled: !root.tracking && !!root.live && root.live.actionProgress < 0
                onClicked: root.live.runAction(root.node, "trackGeoForward")
                ToolTip.text: "Track Geometry forward: many points of the geometry follow the plate, from the current " +
                              "frame to the next keyframe"
            }
            MaterialToolButton {
                visible: root.geoMode
                font.pointSize: 16
                text: MaterialIcons.my_location
                enabled: !root.tracking && !!root.live && root.live.actionProgress < 0
                onClicked: root.live.runAction(root.node, "alignGeo")
                ToolTip.text: "Align: find the right position of the geometry at this frame (from a rough one), against " +
                              "the nearest keyframe; the frame becomes a keyframe"
            }
            MaterialToolButton {
                visible: root.geoMode
                font.pointSize: 16
                text: MaterialIcons.auto_fix_high
                enabled: !root.tracking && !!root.live && root.live.actionProgress < 0
                onClicked: root.live.runAction(root.node, "refineGeo")
                ToolTip.text: "Refine: track the geometry again between each pair of keyframes, from both ends, and blend"
            }
            MaterialToolButton {
                visible: root.geoMode
                font.pointSize: 16
                text: MaterialIcons.layers_clear
                enabled: !root.tracking && !!root.live && root.live.actionProgress < 0 && root.node.hasAttribute("geoTrack")
                         && root.node.attribute("geoTrack").value !== ""
                onClicked: root.live.runAction(root.node, "clearGeoTrack")
                ToolTip.text: "Clear the geometry track (the pins solve again)"
            }
            // Resection: the geometry deformed by the optical flow (non rigid)
            Rectangle { visible: root.geoMode; width: 1; Layout.preferredHeight: 20; color: Qt.rgba(1, 1, 1, 0.3) }
            MaterialToolButton {
                visible: root.geoMode
                font.pointSize: 16
                text: MaterialIcons.west
                enabled: !root.tracking && !!root.live && root.live.actionProgress < 0
                onClicked: root.live.runAction(root.node, "deformBackward")
                ToolTip.text: "Deform With Flow backward: the geometry follows the optical flow of the plate (non rigid), " +
                              "from the current frame back to the start of the track range"
            }
            MaterialToolButton {
                visible: root.geoMode
                font.pointSize: 16
                text: MaterialIcons.waves
                enabled: !root.tracking && !!root.live && root.live.actionProgress < 0
                onClicked: root.live.runAction(root.node, "deformForward")
                ToolTip.text: "Deform With Flow forward: the geometry follows the optical flow of the plate (non rigid), " +
                              "from the current frame to the end of the track range"
            }
            MaterialToolButton {
                visible: root.geoMode
                font.pointSize: 16
                text: MaterialIcons.blur_off
                enabled: !root.tracking && !!root.live && root.live.actionProgress < 0 && root.node.hasAttribute("geoDeform")
                         && root.node.attribute("geoDeform").value !== ""
                onClicked: root.live.runAction(root.node, "clearDeform")
                ToolTip.text: "Clear the deformation (the rigid geometry)"
            }
            // Pins of the Resection at this frame and their reprojection error
            Label {
                visible: root.geoMode
                leftPadding: 6
                rightPadding: 4
                text: {
                    if (!root.geoMode || !root.live)
                        return ""
                    root.live.statusRevision  // Refreshed with the renders
                    root.frame
                    root.tracks
                    return root.live.resectionSummary(root.node)
                }
            }
            MaterialToolButton {
                font.pointSize: 16
                text: MaterialIcons.crop_free
                visible: !root.pinMode
                enabled: !root.tracking && root.selectedNames.length === 4
                onClicked: _currentScene.createCornerPinFromTracks(root.node, root.selectedNames, root.frame)
                ToolTip.text: "Create a CornerPin driven by the 4 selected tracks (reference: the current frame)"
            }
        }
    }
}
