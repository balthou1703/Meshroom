import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MaterialIcons 2.2
import Utils 1.0

/**
 * SequenceBrowser is a file browser that groups the frames of image sequences into one entry,
 * as in Nuke ("render.####.exr  1001-1100"). Choosing an entry emits `selected` with its path
 * (the "####" pattern for a sequence).
 */
Dialog {
    id: root

    property var live: null
    property string folder: ""
    property bool imagesOnly: true
    property var entries: []

    signal selected(string path)

    title: "Read Image or Sequence"
    modal: true
    parent: Overlay.overlay
    anchors.centerIn: parent
    width: Math.min(760, parent ? parent.width * 0.9 : 760)
    height: Math.min(520, parent ? parent.height * 0.85 : 520)
    standardButtons: Dialog.Cancel | Dialog.Open

    function refresh() {
        list.currentIndex = -1
        entries = live ? live.listFolder(folder, imagesOnly) : []
    }

    function openAt(path) {
        folder = live.browseStartFolder(path)
        refresh()
        open()
    }

    function choose(entry) {
        if (!entry)
            return
        if (entry.type === "folder") {
            folder = entry.path
            refresh()
            return
        }
        selected(entry.path)
        close()
    }

    onImagesOnlyChanged: refresh()
    onAccepted: {
        var entry = entries[list.currentIndex]
        if (list.currentIndex >= 0 && entry && entry.type !== "folder")
            selected(entry.path)
    }
    Component.onCompleted: {
        // "Open" is only available once a file or a sequence is selected
        var button = standardButton(Dialog.Open)
        if (button)
            button.enabled = Qt.binding(function() {
                var entry = root.entries[list.currentIndex]
                return list.currentIndex >= 0 && !!entry && entry.type !== "folder"
            })
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 4

        RowLayout {
            Layout.fillWidth: true
            MaterialToolButton {
                text: MaterialIcons.arrow_upward
                ToolTip.text: "Parent Folder"
                onClicked: {
                    root.folder = root.live.parentFolder(root.folder)
                    root.refresh()
                }
            }
            TextField {
                Layout.fillWidth: true
                text: root.folder
                selectByMouse: true
                onAccepted: {
                    root.folder = text
                    root.refresh()
                }
            }
            CheckBox {
                text: "Images Only"
                checked: root.imagesOnly
                onToggled: root.imagesOnly = checked
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: Qt.darker(Colors.sysPalette.window, 1.25)
            radius: 2

            ListView {
                id: list
                anchors.fill: parent
                anchors.margins: 2
                clip: true
                model: root.entries
                currentIndex: -1
                ScrollBar.vertical: ScrollBar {}

                delegate: ItemDelegate {
                    width: ListView.view.width
                    height: 26
                    highlighted: ListView.isCurrentItem
                    onClicked: list.currentIndex = index
                    onDoubleClicked: root.choose(modelData)

                    contentItem: RowLayout {
                        spacing: 8
                        Label {
                            font.family: MaterialIcons.fontFamily
                            font.pointSize: 12
                            color: modelData.type === "sequence" ? "#8FB3C9" : Colors.lightgrey
                            text: modelData.type === "folder" ? MaterialIcons.folder
                                : (modelData.type === "sequence" ? MaterialIcons.burst_mode : MaterialIcons.image)
                        }
                        Label {
                            Layout.fillWidth: true
                            text: modelData.name
                            elide: Text.ElideMiddle
                        }
                        Label {
                            visible: modelData.type === "sequence"
                            text: modelData.frames + "  (" + modelData.count + ")"
                            color: Colors.lightgrey
                            elide: Text.ElideRight
                            Layout.maximumWidth: 260
                        }
                    }
                }
            }

            Label {
                anchors.centerIn: parent
                visible: list.count === 0
                text: "Empty folder"
                color: Colors.lightgrey
            }
        }
    }
}
