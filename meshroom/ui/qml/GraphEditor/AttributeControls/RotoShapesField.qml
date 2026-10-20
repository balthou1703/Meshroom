import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MaterialIcons 2.2

/**
 * List of the shapes of a Roto node (JSON "shapes" attribute, see meshroom/live/roto.py): name,
 * visibility, invert, feather, opacity, delete; animation keys; track links (RotoLinkRow): all the
 * tracks of a connected Tracker or some of them, and the transform fitted on them, moving the whole
 * shape or one of its points from the frame where the link is made ("Roto Links": RotoLinksField, all
 * the shapes). The shapes are drawn in the viewer.
 */
ColumnLayout {
    id: root

    required property var attribute
    property bool editable: true
    readonly property var shapes: {
        try {
            var parsed = JSON.parse(attribute.value || "[]")
            return Array.isArray(parsed) ? parsed : []
        } catch (e) {
            return []
        }
    }

    // Track names of the Trackers connected to the Roto, by input: {"tracker": [...], "tracker2": [...], ...}
    readonly property var inputLabels: ({ "tracker": "Tracker 1", "tracker2": "Tracker 2", "tracker3": "Tracker 3" })
    readonly property var trackNames: {
        var result = {}
        var node = attribute.node
        if (!node || !_currentScene || !_currentScene.live)
            return result
        ;["tracker", "tracker2", "tracker3"].forEach(function(name) {
            if (!node.hasAttribute(name))
                return
            var input = node.attribute(name)
            input.isLink  // Re-evaluated when the link changes
            // The node whose values the Roto reads: through Dot, Clone and disabled nodes (as the render)
            var tracker = _currentScene.live.upstreamValuesNode(input)
            if (!tracker || !tracker.hasAttribute("tracks"))
                return
            try {
                result[name] = JSON.parse(tracker.attribute("tracks").value || "[]").map(function(track) { return track.name })
            } catch (e) {
            }
        })
        return result
    }
    readonly property var connectedInputs: Object.keys(trackNames)

    // Links of a shape as a list (the former single "link" goes to the "tracker" input)
    function shapeLinks(shape) {
        var links = (shape.links || []).slice()
        if (shape.link)
            links.push(Object.assign({ "input": "tracker" }, shape.link))
        return links
    }
    // The shape stays where it is at the current frame (its points absorb the change of the link transforms)
    function setLinks(index, links) {
        _currentScene.setRotoShapeLinks(root.attribute.node, index, JSON.stringify(links), root.currentFrame)
    }
    function replaced(links, index, newLink) {
        var result = JSON.parse(JSON.stringify(links))
        // Re-linked from the current frame: the shapes do not jump
        newLink.frame = root.currentFrame
        delete newLink.track
        result[index] = newLink
        return result
    }
    readonly property int currentFrame: _currentScene && _currentScene.live ? _currentScene.live.currentFrame : 1

    spacing: 1

    // Points of a shape at a frame, interpolated between its keys (as meshroom/live/roto.py)
    function pointsAt(shape, frame) {
        var keys = (shape.keys || []).filter(function(k) { return k.points && k.points.length })
                                     .sort(function(a, b) { return a.frame - b.frame })
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
                    var q = b.points[index]
                    var r = {}
                    var names = ["x", "y", "inX", "inY", "outX", "outY"]
                    for (var n = 0; n < names.length; n++)
                        r[names[n]] = (1 - t) * (p[names[n]] || 0) + t * (q[names[n]] || 0)
                    return r
                })
            }
        }
        return keys[keys.length - 1].points
    }

    function updateShape(index, key, value) {
        var shapes = JSON.parse(JSON.stringify(root.shapes))
        if (key === null)
            shapes.splice(index, 1)
        else
            shapes[index][key] = value
        _currentScene.setAttribute(root.attribute, JSON.stringify(shapes))
    }

    Label {
        visible: root.shapes.length === 0
        text: "No shape: select the node and draw in the viewer"
        opacity: 0.6
        font.pointSize: 8
    }

    Repeater {
        model: root.shapes

        ColumnLayout {
            id: shapeRow
            required property var modelData
            required property int index
            readonly property var links: root.shapeLinks(modelData)
            readonly property int pointCount: (modelData.points || []).length
            readonly property int keyCount: (modelData.keys || []).length
            Layout.fillWidth: true
            spacing: 0

        RowLayout {
            Layout.fillWidth: true
            spacing: 2

            MaterialToolButton {
                text: shapeRow.modelData.visible === false ? MaterialIcons.visibility_off : MaterialIcons.visibility
                font.pointSize: 8
                padding: 2
                enabled: root.editable
                onClicked: root.updateShape(shapeRow.index, "visible", shapeRow.modelData.visible === false)
                ToolTip.text: "Show / hide the shape"
            }
            TextField {
                Layout.fillWidth: true
                text: shapeRow.modelData.name || ("Shape" + (shapeRow.index + 1))
                enabled: root.editable
                selectByMouse: true
                font.pointSize: 8
                onEditingFinished: {
                    if (text !== shapeRow.modelData.name)
                        root.updateShape(shapeRow.index, "name", text)
                }
            }
            Label { text: "F"; opacity: 0.6; font.pointSize: 8 }
            TextField {
                Layout.preferredWidth: 42
                text: Number(shapeRow.modelData.feather || 0).toString()
                enabled: root.editable
                validator: DoubleValidator { bottom: 0; locale: 'C' }
                font.pointSize: 8
                onEditingFinished: {
                    var value = Number(text)
                    if (isFinite(value))
                        root.updateShape(shapeRow.index, "feather", Math.max(0, value))
                }
                ToolTip.text: "Feather (pixels)"
                ToolTip.visible: hovered
            }
            Label { text: "O"; opacity: 0.6; font.pointSize: 8 }
            TextField {
                Layout.preferredWidth: 38
                text: Number(shapeRow.modelData.opacity === undefined ? 1 : shapeRow.modelData.opacity).toString()
                enabled: root.editable
                validator: DoubleValidator { bottom: 0; top: 1; locale: 'C' }
                font.pointSize: 8
                onEditingFinished: {
                    var value = Number(text)
                    if (isFinite(value))
                        root.updateShape(shapeRow.index, "opacity", Math.max(0, Math.min(1, value)))
                }
                ToolTip.text: "Opacity"
                ToolTip.visible: hovered
            }
            // Moved along the optical flow by "Track Shapes Forward / Backward" (all the shapes when none is on)
            MaterialToolButton {
                text: MaterialIcons.air
                font.pointSize: 8
                padding: 2
                checked: !!shapeRow.modelData.flow
                enabled: root.editable
                onClicked: root.updateShape(shapeRow.index, "flow", !shapeRow.modelData.flow)
                ToolTip.text: "Track this shape with the optical flow (Track Shapes Forward / Backward; all the shapes when none is on)"
            }
            // Closed shape (filled) or open curve (a stroke of its width)
            MaterialToolButton {
                readonly property bool open: shapeRow.modelData.closed === false
                text: open ? MaterialIcons.timeline : MaterialIcons.pentagon
                font.pointSize: 8
                padding: 2
                enabled: root.editable
                onClicked: root.updateShape(shapeRow.index, "closed", open)
                ToolTip.text: open ? "Open curve (a stroke): click to close it" : "Closed shape: click to make it an open curve"
            }
            Label { visible: shapeRow.modelData.closed === false; text: "W"; opacity: 0.6; font.pointSize: 8 }
            TextField {
                visible: shapeRow.modelData.closed === false
                Layout.preferredWidth: 36
                text: Number(shapeRow.modelData.width === undefined ? 4 : shapeRow.modelData.width).toString()
                enabled: root.editable
                validator: DoubleValidator { bottom: 0; locale: 'C' }
                font.pointSize: 8
                onEditingFinished: {
                    var value = Number(text)
                    if (isFinite(value))
                        root.updateShape(shapeRow.index, "width", Math.max(0, value))
                }
                ToolTip.text: "Width of the stroke of the open curve (pixels)"
                ToolTip.visible: hovered
            }
            MaterialToolButton {
                text: MaterialIcons.invert_colors
                font.pointSize: 8
                padding: 2
                checked: !!shapeRow.modelData.invert
                enabled: root.editable
                onClicked: root.updateShape(shapeRow.index, "invert", !shapeRow.modelData.invert)
                ToolTip.text: "Invert the shape"
            }
            MaterialToolButton {
                text: MaterialIcons.delete_outline
                font.pointSize: 8
                padding: 2
                enabled: root.editable
                onClicked: root.updateShape(shapeRow.index, null, null)
                ToolTip.text: "Delete the shape"
            }
        }

        // Animation and tracking
        RowLayout {
            Layout.fillWidth: true
            Layout.leftMargin: 22
            spacing: 4

            Label {
                text: shapeRow.keyCount > 0 ? shapeRow.keyCount + (shapeRow.keyCount > 1 ? " keys" : " key") : "not animated"
                font.pointSize: 7
                opacity: 0.7
            }
            MaterialToolButton {
                visible: shapeRow.keyCount > 0
                text: MaterialIcons.key_off
                font.pointSize: 8
                padding: 2
                enabled: root.editable
                ToolTip.text: "Remove the animation (keeps the shape of the current frame)"
                onClicked: {
                    var shapes = JSON.parse(JSON.stringify(root.shapes))
                    var shape = shapes[shapeRow.index]
                    // The shape of the current frame (interpolated between its keys)
                    shape.points = root.pointsAt(shape, root.currentFrame)
                    delete shape.keys
                    _currentScene.setAttribute(root.attribute, JSON.stringify(shapes))
                }
            }
            Item { Layout.fillWidth: true }
            MaterialToolButton {
                text: MaterialIcons.add_link
                font.pointSize: 8
                padding: 2
                enabled: root.editable && root.connectedInputs.length > 0
                ToolTip.text: root.connectedInputs.length > 0 ?
                                  "Link the shape (or one of its points) to tracks, from the current frame" :
                                  "Connect a Tracker to a 'Tracker' input of the Roto to link the shape to its tracks"
                onClicked: {
                    var input = root.connectedInputs[0]
                    var names = root.trackNames[input]
                    var links = JSON.parse(JSON.stringify(shapeRow.links))
                    links.push({ "input": input, "tracks": names.length ? [names[0]] : [], "transform": "translate",
                                 "point": -1, "frame": root.currentFrame })
                    root.setLinks(shapeRow.index, links)
                }
            }
        }

        // Track links: [Tracker input] [tracks] [transform] [whole shape / point N] [remove]
        Repeater {
            model: shapeRow.links
            RotoLinkRow {
                required property var modelData
                required property int index
                Layout.fillWidth: true
                Layout.leftMargin: 22
                link: modelData
                pointCount: shapeRow.pointCount
                trackNames: root.trackNames
                inputLabels: root.inputLabels
                editable: root.editable
                onEdited: function(newLink) { root.setLinks(shapeRow.index, root.replaced(shapeRow.links, index, newLink)) }
                onRemoved: {
                    var links = JSON.parse(JSON.stringify(shapeRow.links))
                    links.splice(index, 1)
                    root.setLinks(shapeRow.index, links)
                }
            }
        }
        }
    }
}
