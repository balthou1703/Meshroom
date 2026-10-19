import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

/**
 * Upper part of the compositing workspace: the viewer of the evaluated node and its timeline.
 * The graph editor and the node editor stay below, as in the default workspace.
 */
Item {
    id: root

    property var currentScene: null
    readonly property var live: currentScene ? currentScene.live : null

    // Playback shortcuts while the mouse is over the viewer or the timeline, as in Nuke
    HoverHandler {
        id: hover
    }
    readonly property bool shortcutsEnabled: hover.hovered && !!root.live && !!root.live.evaluatedNode
    Shortcut { sequence: "L"; enabled: root.shortcutsEnabled; onActivated: root.live.play(1) }
    Shortcut { sequence: "J"; enabled: root.shortcutsEnabled; onActivated: root.live.play(-1) }
    Shortcut { sequence: "K"; enabled: root.shortcutsEnabled; onActivated: root.live.stop() }
    Shortcut { sequence: "Right"; enabled: root.shortcutsEnabled; onActivated: root.live.stepFrames(1) }
    Shortcut { sequence: "Left"; enabled: root.shortcutsEnabled; onActivated: root.live.stepFrames(-1) }
    // View 1 (Evaluate) / view 2 (Evaluate Compare)
    Shortcut { sequences: ["1", "Num+1"]; enabled: root.shortcutsEnabled && !!root.live.compareNode; onActivated: root.live.showView(1) }
    Shortcut { sequences: ["2", "Num+2"]; enabled: root.shortcutsEnabled && !!root.live.compareNode; onActivated: root.live.showView(2) }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        LiveViewer {
            Layout.fillWidth: true
            Layout.fillHeight: true
            live: root.live
        }

        LiveTimeline {
            Layout.fillWidth: true
            live: root.live
        }
    }
}
