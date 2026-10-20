import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MaterialIcons 2.2

/**
 * Choice of tracks of the Tracker connected to the "tracker" input of a node (Stabilize, MotionBlur,
 * CornerPin): a button showing the choice, opening a list of the tracks to check. The value of the
 * attribute is the names, comma separated ("" : all the tracks). With `count` (CornerPin: 4), exactly
 * that many tracks in the order they are checked (shown by their number), no "all".
 */
RowLayout {
    id: root

    required property var attribute
    property bool editable: true
    // Number of tracks needed (0: any)
    property int count: 0

    readonly property var chosen: (attribute.value || "").split(",").map(function(name) { return name.trim() })
                                                                 .filter(function(name) { return name !== "" })
    // Names of the tracks of the Tracker connected to "tracker" (through Dot, Clone, disabled nodes)
    readonly property var available: {
        var node = attribute.node
        if (!node || !node.hasAttribute("tracker") || !_currentScene || !_currentScene.live)
            return []
        var input = node.attribute("tracker")
        input.isLink  // Re-evaluated when the link changes
        var tracker = _currentScene.live.upstreamValuesNode(input)
        if (!tracker || !tracker.hasAttribute("tracks"))
            return []
        try {
            return JSON.parse(tracker.attribute("tracks").value || "[]").map(function(track) { return track.name })
        } catch (e) {
            return []
        }
    }

    function setNames(names) {
        var value = names.join(",")
        if (value !== root.attribute.value)
            _currentScene.setAttribute(root.attribute, value)
    }

    spacing: 2

    Button {
        id: pickButton
        Layout.fillWidth: true
        enabled: root.editable
        font.pointSize: 8
        text: {
            if (root.chosen.length === 0)
                return root.count > 0 ? "choose " + root.count + " tracks" : "all tracks (" + root.available.length + ")"
            if (root.chosen.length <= 3)
                return root.chosen.join(", ")
            return root.chosen.length + " tracks"
        }
        ToolTip.text: root.available.length === 0 ? "Connect a Tracker to the 'Tracker' input to choose its tracks" :
                      (root.chosen.length ? root.chosen.join(", ") : (root.count > 0 ? "No track: the values above" :
                                                                                       "All the enabled tracks"))
        ToolTip.visible: hovered
        onClicked: popup.open()

        Popup {
            id: popup
            y: pickButton.height
            width: Math.max(pickButton.width, 200)
            height: Math.min(360, column.implicitHeight + 16)
            padding: 6
            ColumnLayout {
                id: column
                anchors.fill: parent
                spacing: 0
                Label {
                    visible: root.count > 0
                    text: "Check " + root.count + " tracks, in the order of the corners (bottom-left, bottom-right, " +
                          "top-right, top-left)"
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                    font.pointSize: 7
                    opacity: 0.7
                }
                CheckBox {
                    visible: root.count === 0
                    text: "All tracks"
                    font.pointSize: 8
                    checked: root.chosen.length === 0
                    onClicked: root.setNames(checked ? [] : root.available.slice(0, 1))
                }
                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.preferredHeight: contentHeight
                    clip: true
                    model: root.available
                    ScrollBar.vertical: ScrollBar {}
                    delegate: CheckBox {
                        required property string modelData
                        readonly property int order: root.chosen.indexOf(modelData)
                        text: modelData + (root.count > 0 && order >= 0 ? "  (" + (order + 1) + ")" : "")
                        font.pointSize: 8
                        checked: (root.count === 0 && root.chosen.length === 0) || order >= 0
                        onClicked: {
                            var names = (root.count === 0 && root.chosen.length === 0) ? root.available.slice()
                                                                                      : root.chosen.slice()
                            var at = names.indexOf(modelData)
                            if (checked && at < 0) {
                                if (root.count > 0 && names.length >= root.count)
                                    names.shift()  // The oldest choice gives way
                                names.push(modelData)
                            }
                            if (!checked && at >= 0)
                                names.splice(at, 1)
                            // Every track checked: all of them (also the tracks added later)
                            if (root.count === 0 && names.length === root.available.length)
                                names = []
                            root.setNames(names)
                        }
                    }
                }
                Button {
                    visible: root.chosen.length > 0
                    text: root.count > 0 ? "Clear" : "Use all the tracks"
                    font.pointSize: 8
                    Layout.alignment: Qt.AlignRight
                    onClicked: root.setNames([])
                }
            }
        }
    }
}
