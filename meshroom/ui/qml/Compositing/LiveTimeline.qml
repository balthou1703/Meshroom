import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MaterialIcons 2.2
import Utils 1.0

/**
 * Timeline of the live compositing evaluation: frame ruler of the evaluated node (frame numbers,
 * current frame, keys of the selected node, frames in the RAM cache), and playback.
 *
 * Playback (as in Nuke): play backward / forward, previous / next step, first / last frame, with a
 * step in frames, a frame rate and a loop. Each frame is shown once rendered: playback slows down
 * rather than skipping frames; the frame rate actually played is shown while playing.
 * Shortcuts over the compositing view: J / K / L (backward, stop, forward), Left / Right (step).
 */
Pane {
    id: root

    property var live: null
    readonly property bool active: !!live && !!live.evaluatedNode
    readonly property int firstFrame: live && live.hasFrameRange ? live.firstFrame : 1
    readonly property int lastFrame: live && live.hasFrameRange ? live.lastFrame : 1
    readonly property int playDirection: live ? live.playDirection : 0

    padding: 4
    // The in / out can be set before evaluating a node
    enabled: true
    background: Rectangle { color: Qt.darker(palette.window, 1.15) }

    // Keys of the selected node, shown on the ruler
    readonly property var selectedNode: _currentScene ? _currentScene.selectedNode : null
    readonly property var keyFrames: {
        if (!live || !selectedNode || !selectedNode.isImageOpNode)
            return []
        live.statusRevision  // Refreshed with the graph changes
        return live.nodeKeyFrames(selectedNode)
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 2

    // In / out of the timeline on both sides of the ruler: they follow the first Read of the graph,
    // until they are entered by hand (saved with the graph; the link button follows the Read again)
    RowLayout {
        Layout.fillWidth: true
        spacing: 4

        SpinBox {
            id: inField
            Layout.preferredWidth: 84
            from: -100000
            to: 100000
            editable: true
            value: root.firstFrame
            enabled: !!_currentScene
            onValueModified: _currentScene.setFrameRange(value, Math.max(value, root.lastFrame))
            ToolTip.text: "In: first frame of the timeline"
            ToolTip.visible: hovered
            ToolTip.delay: 600
        }
        FrameRuler {
            Layout.fillWidth: true
            first: root.firstFrame
            last: Math.max(root.firstFrame + 1, root.lastFrame)
            current: root.live ? root.live.currentFrame : 1
            cachedFrames: {
                if (!root.active)
                    return []
                root.live.statusRevision  // Refresh with the evaluation statuses
                return root.live.cachedFrames()
            }
            keyFrames: root.keyFrames
            onFrameRequested: function(frame) {
                root.live.stop()
                root.live.currentFrame = frame
            }
        }
        SpinBox {
            id: outField
            Layout.preferredWidth: 84
            from: -100000
            to: 100000
            editable: true
            value: root.lastFrame
            enabled: !!_currentScene
            onValueModified: _currentScene.setFrameRange(Math.min(value, root.firstFrame), value)
            ToolTip.text: "Out: last frame of the timeline"
            ToolTip.visible: hovered
            ToolTip.delay: 600
        }
        MaterialToolButton {
            visible: !!_currentScene && _currentScene.frameRangeIsUser
            text: MaterialIcons.link
            font.pointSize: 11
            onClicked: _currentScene.resetFrameRange()
            ToolTip.text: "In / out entered by hand: click to follow the first Read again"
        }
    }

    RowLayout {
        Layout.fillWidth: true
        spacing: 6

        // Left: the current frame; center: the transport; right: the playback settings
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredWidth: 1
            SpinBox {
                from: -100000
                to: 100000
                editable: true
                value: root.live ? root.live.currentFrame : 1
                onValueModified: {
                    root.live.stop()
                    root.live.currentFrame = value
                }
            }
            Item { Layout.fillWidth: true }
        }

        // Transport
        RowLayout {
            Layout.alignment: Qt.AlignHCenter
            spacing: 2

            MaterialToolButton {
                font.pointSize: 17
                text: MaterialIcons.skip_previous
                onClicked: root.live.goToFirstFrame()
                ToolTip.text: "First frame"
            }
            MaterialToolButton {
                font.pointSize: 17
                text: MaterialIcons.navigate_before
                onClicked: root.live.stepFrames(-1)
                ToolTip.text: "Previous step (Left)"
            }
            MaterialToolButton {
                font.pointSize: 17
                text: MaterialIcons.play_arrow
                rotation: 180
                checked: root.playDirection < 0
                onClicked: root.live.play(-1)
                ToolTip.text: "Play backward (J)"
            }
            MaterialToolButton {
                font.pointSize: 17
                text: MaterialIcons.stop
                enabled: root.playDirection !== 0
                onClicked: root.live.stop()
                ToolTip.text: "Stop (K)"
            }
            MaterialToolButton {
                font.pointSize: 17
                text: MaterialIcons.play_arrow
                checked: root.playDirection > 0
                onClicked: root.live.play(1)
                ToolTip.text: "Play forward (L)"
            }
            MaterialToolButton {
                font.pointSize: 17
                text: MaterialIcons.navigate_next
                onClicked: root.live.stepFrames(1)
                ToolTip.text: "Next step (Right)"
            }
            MaterialToolButton {
                font.pointSize: 17
                text: MaterialIcons.skip_next
                onClicked: root.live.goToLastFrame()
                ToolTip.text: "Last frame"
            }
            MaterialToolButton {
                font.pointSize: 17
                text: MaterialIcons.repeat
                checked: root.live ? root.live.loop : true
                onClicked: root.live.loop = !root.live.loop
                ToolTip.text: checked ? "Loop" : "Play once"
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.preferredWidth: 1
            Item { Layout.fillWidth: true }
        Label { text: "Step"; color: Colors.lightgrey }
        SpinBox {
            Layout.preferredWidth: 70
            from: 1
            to: 1000
            editable: true
            value: root.live ? root.live.playStep : 1
            onValueModified: root.live.playStep = value
            ToolTip.text: "Frames between two played frames (and the previous / next step)"
            ToolTip.visible: hovered
        }

        Label { text: "fps"; color: Colors.lightgrey }
        SpinBox {
            Layout.preferredWidth: 70
            from: 1
            to: 120
            editable: true
            value: root.live ? Math.round(root.live.fps) : 24
            onValueModified: root.live.fps = value
            ToolTip.text: "Maximum playback frame rate"
            ToolTip.visible: hovered
        }

        // Frame rate actually played (rendering included)
        Label {
            Layout.preferredWidth: 46
            visible: root.playDirection !== 0
            text: root.live ? root.live.measuredFps.toFixed(1) : ""
            color: root.live && root.live.measuredFps < root.live.fps * 0.9 ? "#C2A06A" : Colors.lightgrey
            ToolTip.text: "Frames per second actually played"
        }
        }
    }
    }
}
