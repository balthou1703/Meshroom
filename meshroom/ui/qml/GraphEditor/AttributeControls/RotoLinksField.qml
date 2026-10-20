import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MaterialIcons 2.2

/**
 * The links of a Roto warping all its shapes ("links" attribute, see meshroom/live/roto.py): one row per link
 * (RotoLinkRow: the Tracker, its tracks checked in a list or all of them, the transform), and a button adding
 * one. Changing them keeps the shapes where they are at the current frame.
 */
ColumnLayout {
    id: root

    required property var attribute
    property bool editable: true

    readonly property var links: {
        try {
            var parsed = JSON.parse(attribute.value || "[]")
            return Array.isArray(parsed) ? parsed : []
        } catch (e) {
            return []
        }
    }
    readonly property int currentFrame: _currentScene && _currentScene.live ? _currentScene.live.currentFrame : 1
    readonly property var inputLabels: ({ "tracker": "Tracker 1", "tracker2": "Tracker 2", "tracker3": "Tracker 3" })
    // Track names of the Trackers connected to the Roto, by input
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

    function setLinks(links) {
        _currentScene.setRotoLinks(root.attribute.node, JSON.stringify(links), root.currentFrame)
    }

    spacing: 1

    RowLayout {
        Layout.fillWidth: true
        spacing: 4
        Label {
            text: root.links.length ? "" : (root.connectedInputs.length ? "No link: the shapes are not warped as a whole"
                                                                        : "Connect a Tracker to a 'Tracker' input to warp all the shapes")
            font.pointSize: 7
            opacity: 0.6
        }
        Item { Layout.fillWidth: true }
        MaterialToolButton {
            text: MaterialIcons.add_link
            font.pointSize: 8
            padding: 2
            enabled: root.editable && root.connectedInputs.length > 0
            ToolTip.text: "Warp all the shapes with the tracks of a Tracker (all of them, or the ones checked), from the current frame"
            onClicked: {
                var links = JSON.parse(JSON.stringify(root.links))
                links.push({ "input": root.connectedInputs[0], "tracks": [], "transform": "translate rotate scale",
                             "frame": root.currentFrame })
                root.setLinks(links)
            }
        }
    }
    Repeater {
        model: root.links
        RotoLinkRow {
            required property var modelData
            required property int index
            Layout.fillWidth: true
            link: modelData
            pointCount: -1
            trackNames: root.trackNames
            inputLabels: root.inputLabels
            editable: root.editable
            onEdited: function(newLink) {
                var links = JSON.parse(JSON.stringify(root.links))
                // Re-linked from the current frame: the shapes do not jump
                newLink.frame = root.currentFrame
                delete newLink.track
                links[index] = newLink
                root.setLinks(links)
            }
            onRemoved: {
                var links = JSON.parse(JSON.stringify(root.links))
                links.splice(index, 1)
                root.setLinks(links)
            }
        }
    }
}
