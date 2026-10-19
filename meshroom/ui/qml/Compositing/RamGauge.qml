import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MaterialIcons 2.2
import Utils 1.0

/**
 * RamGauge shows the memory used by the cached images of the compositing, against its budget,
 * with a button to purge the cache (the displayed image is kept on screen).
 */
RowLayout {
    id: root

    property var live: null
    readonly property real used: live ? live.cacheUsedBytes : 0
    readonly property real budget: live && live.cacheMaxBytes > 0 ? live.cacheMaxBytes : 1
    readonly property real ratio: Math.min(1, used / budget)

    spacing: 6

    function gigabytes(bytes) {
        return (bytes / (1024 * 1024 * 1024)).toFixed(1)
    }

    Item {
        implicitWidth: gaugeLayout.implicitWidth
        implicitHeight: gaugeLayout.implicitHeight

        RowLayout {
            id: gaugeLayout
            anchors.verticalCenter: parent.verticalCenter
            spacing: 6

            // Track
            Rectangle {
                Layout.preferredWidth: 96
                Layout.preferredHeight: 5
                radius: height / 2
                color: Qt.darker(Colors.sysPalette.window, 1.5)
                border.color: Qt.rgba(1, 1, 1, 0.06)

                Rectangle {
                    width: Math.max(root.ratio > 0 ? height : 0, parent.width * root.ratio)
                    height: parent.height
                    radius: parent.radius
                    // Calm blue-grey, muted amber when the budget is nearly reached
                    color: root.ratio < 0.75 ? "#6F8FA6" : (root.ratio < 0.92 ? "#C2A06A" : "#C0716A")
                    Behavior on width { NumberAnimation { duration: 250; easing.type: Easing.OutCubic } }
                    Behavior on color { ColorAnimation { duration: 250 } }
                }
            }

            Label {
                text: root.gigabytes(root.used) + " / " + root.gigabytes(root.budget) + " GB"
                font.pointSize: 8
                color: Colors.lightgrey
            }
        }

        MouseArea {
            id: gaugeMouse
            anchors.fill: parent
            hoverEnabled: true
            acceptedButtons: Qt.NoButton
        }

        ToolTip.visible: gaugeMouse.containsMouse
        ToolTip.delay: 400
        ToolTip.text: "Compositing cache: " + root.gigabytes(root.used) + " GB used, budget " + root.gigabytes(root.budget) + " GB" +
                      (root.live && root.live.systemMemoryBytes > 0 ? " (physical memory " + root.gigabytes(root.live.systemMemoryBytes) + " GB)" : "") +
                      "\nThe budget can be set with MESHROOM_LIVE_CACHE_GB."
    }

    MaterialToolButton {
        text: MaterialIcons.delete_
        font.pointSize: 10
        padding: 2
        enabled: root.used > 0
        ToolTip.text: "Purge the cache: frees the memory of every computed image (the displayed one stays on screen)"
        onClicked: root.live.clearCache()
    }
}
