import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MaterialIcons 2.2

/**
 * List of the tracks of a Tracker node (JSON "tracks" attribute, see meshroom/live/tracking.py):
 * enabled, name, pattern and search sizes, number of keys, delete. The tracks are placed and
 * tracked in the viewer.
 */
ColumnLayout {
    id: root

    required property var attribute
    property bool editable: true
    readonly property var tracks: {
        try {
            var parsed = JSON.parse(attribute.value || "[]")
            return Array.isArray(parsed) ? parsed : []
        } catch (e) {
            return []
        }
    }

    spacing: 1

    function updateTrack(index, key, value) {
        var tracks = JSON.parse(JSON.stringify(root.tracks))
        if (key === null)
            tracks.splice(index, 1)
        else
            tracks[index][key] = value
        _currentScene.setAttribute(root.attribute, JSON.stringify(tracks))
    }

    Label {
        visible: root.tracks.length === 0
        text: "No track: select the node, then 'Add track' in the viewer toolbar"
        opacity: 0.6
        font.pointSize: 8
    }

    Repeater {
        model: root.tracks

        RowLayout {
            id: trackRow
            required property var modelData
            required property int index
            Layout.fillWidth: true
            spacing: 2

            CheckBox {
                checked: trackRow.modelData.enabled !== false
                enabled: root.editable
                padding: 0
                onClicked: root.updateTrack(trackRow.index, "enabled", checked)
                ToolTip.text: "Enabled (used by the transform and the linked shapes)"
                ToolTip.visible: hovered
            }
            TextField {
                Layout.fillWidth: true
                text: trackRow.modelData.name || ""
                enabled: root.editable
                selectByMouse: true
                font.pointSize: 8
                onEditingFinished: {
                    if (text === trackRow.modelData.name)
                        return
                    // Unique names: the Roto shapes are linked to a track by its name (renamed with it)
                    if (!_currentScene.renameTrack(root.attribute.node, trackRow.modelData.name, text))
                        text = trackRow.modelData.name || ""
                }
            }
            Label {
                text: Object.keys(trackRow.modelData.keys || {}).length + " keys"
                font.pointSize: 7
                opacity: 0.7
            }
            Label { text: "P"; opacity: 0.6; font.pointSize: 8 }
            TextField {
                Layout.preferredWidth: 34
                text: String(trackRow.modelData.pattern || 21)
                enabled: root.editable
                validator: IntValidator { bottom: 5; top: 301 }
                font.pointSize: 8
                onEditingFinished: root.updateTrack(trackRow.index, "pattern", Number(text))
                ToolTip.text: "Pattern size (pixels)"
                ToolTip.visible: hovered
            }
            Label { text: "S"; opacity: 0.6; font.pointSize: 8 }
            TextField {
                Layout.preferredWidth: 34
                text: String(trackRow.modelData.search || 61)
                enabled: root.editable
                validator: IntValidator { bottom: 7; top: 601 }
                font.pointSize: 8
                onEditingFinished: root.updateTrack(trackRow.index, "search", Number(text))
                ToolTip.text: "Search area size (pixels)"
                ToolTip.visible: hovered
            }
            MaterialToolButton {
                text: MaterialIcons.delete_outline
                font.pointSize: 8
                padding: 2
                enabled: root.editable
                onClicked: root.updateTrack(trackRow.index, null, null)
                ToolTip.text: "Delete the track"
            }
        }
    }
}
