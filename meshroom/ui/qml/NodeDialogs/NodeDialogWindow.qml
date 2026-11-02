import QtCore
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

/**
 * NodeDialogWindow is the base window of the node dialogs: a non-modal window, transient for the main
 * window, displaying a header above its content and a row of buttons below.
 *
 *  - Ctrl + "+" / Ctrl + "-" / Ctrl + 0: change the size of the text (persisted for all the dialogs).
 *  - F11 or double click on the title bar: maximize / restore. Escape: close.
 *  - The size of the window is persisted per kind of dialog.
 */

ApplicationWindow {
    id: root

    /// Backend of the dialog (see meshroom/ui/components/nodeDialogs.py)
    required property var backend
    /// Rich text displayed at the top of the window
    property string headerText: backend ? backend.header : ""
    /// Content of the window, laid out in a ColumnLayout
    property alias content: contentLayout.data
    /// Buttons displayed at the bottom right of the window
    property alias buttons: buttonsLayout.data

    signal finished()

    title: backend ? (backend.nodeLabel ? backend.title + " - " + backend.nodeLabel : backend.title) : ""
    minimumWidth: 480
    minimumHeight: 360
    color: palette.window
    font.pointSize: Math.max(6, Qt.application.font.pointSize + settings.fontDelta)

    Settings {
        id: settings
        category: "NodeDialogs"
        property int fontDelta: 2
        property string windowSizes: "{}"
    }

    function sizeKey() { return backend ? backend.kind : "" }

    Component.onCompleted: {
        let sizes = {}
        try { sizes = JSON.parse(settings.windowSizes) } catch (e) {}
        const size = sizes[sizeKey()]
        if (size) {
            width = Math.max(minimumWidth, size[0])
            height = Math.max(minimumHeight, size[1])
        }
    }

    onClosing: function(close) {
        if (visibility !== Window.Maximized && visibility !== Window.FullScreen) {
            let sizes = {}
            try { sizes = JSON.parse(settings.windowSizes) } catch (e) {}
            sizes[sizeKey()] = [width, height]
            settings.windowSizes = JSON.stringify(sizes)
        }
        root.finished()
    }

    Shortcut { sequences: [StandardKey.ZoomIn, "Ctrl+="]; onActivated: settings.fontDelta = Math.min(settings.fontDelta + 1, 16) }
    Shortcut { sequence: StandardKey.ZoomOut; onActivated: settings.fontDelta = Math.max(settings.fontDelta - 1, -4) }
    Shortcut { sequence: "Ctrl+0"; onActivated: settings.fontDelta = 0 }
    Shortcut { sequence: "F11"; onActivated: root.visibility = (root.visibility === Window.Maximized ? Window.Windowed : Window.Maximized) }
    Shortcut { sequence: "Escape"; onActivated: root.close() }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 8

        Label {
            Layout.fillWidth: true
            visible: text.length > 0
            text: root.headerText
            textFormat: Text.StyledText
            wrapMode: Text.WordWrap
        }

        ColumnLayout {
            id: contentLayout
            Layout.fillWidth: true
            Layout.fillHeight: true
        }

        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 1
            color: Qt.darker(palette.window, 1.4)
        }

        RowLayout {
            Layout.fillWidth: true
            Label {
                text: "Text size: Ctrl +/-"
                opacity: 0.5
                font.pointSize: Math.max(6, root.font.pointSize - 2)
            }
            Item { Layout.fillWidth: true }
            RowLayout {
                id: buttonsLayout
                spacing: 6
            }
        }
    }
}
