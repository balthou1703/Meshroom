import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MaterialIcons 2.2

/**
 * One track link of a Roto (meshroom/live/roto.py): [Tracker input] [tracks: all or some] [transform]
 * [whole shape / point N] [remove]. The tracks of the link are chosen in a list (all of them, or some);
 * the transform is fitted on them from the link frame. `pointCount` < 0: a link of the whole Roto (no
 * point choice). Every change is emitted as a new link (`edited`), re-linked from the current frame.
 */
RowLayout {
    id: root

    property var link: ({})
    property int pointCount: -1
    property var trackNames: ({})
    property var inputLabels: ({})
    property bool editable: true
    signal edited(var newLink)
    signal removed()

    readonly property var inputs: ["tracker", "tracker2", "tracker3"]
    readonly property string input: link.input || "tracker"
    readonly property var available: trackNames[input] || []
    // The tracks of the link ([]: all of them; the former single "track")
    readonly property var linkTracks: link.tracks !== undefined ? (link.tracks || []) : (link.track ? [link.track] : [])
    readonly property var transforms: ["translate", "rotate", "scale", "translate rotate", "translate rotate scale",
                                       "perspective"]

    spacing: 2

    function update(changes) {
        var newLink = { "input": root.input, "tracks": root.linkTracks.slice(),
                        "transform": root.link.transform || "translate", "frame": root.link.frame }
        if (root.pointCount >= 0)
            newLink.point = root.link.point === undefined ? -1 : root.link.point
        for (var key in changes)
            newLink[key] = changes[key]
        root.edited(newLink)
    }

    Label { text: "↳"; opacity: 0.6; font.pointSize: 8 }
    ComboBox {
        Layout.preferredWidth: 80
        font.pointSize: 8
        enabled: root.editable
        model: root.inputs.map(function(name) { return root.inputLabels[name] || name })
        currentIndex: Math.max(0, root.inputs.indexOf(root.input))
        // Another Tracker: all its tracks
        onActivated: function(index) { root.update({ "input": root.inputs[index], "tracks": [] }) }
    }
    // The tracks: all, or the ones checked in the list
    Button {
        id: tracksButton
        Layout.fillWidth: true
        font.pointSize: 8
        enabled: root.editable
        text: {
            if (root.linkTracks.length === 0)
                return "all tracks (" + root.available.length + ")"
            if (root.linkTracks.length === 1)
                return root.linkTracks[0]
            return root.linkTracks.length + " tracks"
        }
        ToolTip.text: root.linkTracks.length ? root.linkTracks.join(", ") : "All the tracks of the Tracker"
        ToolTip.visible: hovered
        onClicked: tracksPopup.open()

        Popup {
            id: tracksPopup
            y: tracksButton.height
            width: Math.max(tracksButton.width, 180)
            height: Math.min(320, tracksColumn.implicitHeight + 16)
            padding: 6
            ColumnLayout {
                id: tracksColumn
                anchors.fill: parent
                spacing: 0
                CheckBox {
                    text: "All tracks"
                    font.pointSize: 8
                    checked: root.linkTracks.length === 0
                    onClicked: root.update({ "tracks": checked ? [] : root.available.slice(0, 1) })
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
                        text: modelData
                        font.pointSize: 8
                        // With "all tracks", every track is in the link
                        checked: root.linkTracks.length === 0 || root.linkTracks.indexOf(modelData) >= 0
                        onClicked: {
                            var names = root.linkTracks.length === 0 ? root.available.slice() : root.linkTracks.slice()
                            var at = names.indexOf(modelData)
                            if (checked && at < 0)
                                names.push(modelData)
                            if (!checked && at >= 0)
                                names.splice(at, 1)
                            if (names.length === 0)
                                return  // At least one track
                            // Every track checked: all of them (also the tracks added later)
                            root.update({ "tracks": names.length === root.available.length ? [] : names })
                        }
                    }
                }
            }
        }
    }
    ComboBox {
        Layout.preferredWidth: 128
        font.pointSize: 8
        enabled: root.editable
        model: root.transforms
        currentIndex: Math.max(0, root.transforms.indexOf(root.link.transform || "translate"))
        ToolTip.text: "Transform fitted on the tracks from the link frame (rotate / scale: 2 tracks, perspective: 4; " +
                      "with fewer tracks known at a frame, their mean motion)"
        ToolTip.visible: hovered
        onActivated: function(index) { root.update({ "transform": root.transforms[index] }) }
    }
    ComboBox {
        visible: root.pointCount >= 0
        Layout.preferredWidth: 90
        font.pointSize: 8
        enabled: root.editable
        model: {
            var items = ["whole shape"]
            for (var i = 0; i < root.pointCount; i++)
                items.push("point " + (i + 1))
            return items
        }
        currentIndex: root.link.point >= 0 ? root.link.point + 1 : 0
        ToolTip.text: "What follows the tracks (a point with its own link follows it instead of the shape links)"
        ToolTip.visible: hovered
        onActivated: function(index) { root.update({ "point": index - 1 }) }
    }
    MaterialToolButton {
        text: MaterialIcons.link_off
        font.pointSize: 8
        padding: 2
        enabled: root.editable
        ToolTip.text: "Remove the link"
        onClicked: root.removed()
    }
}
