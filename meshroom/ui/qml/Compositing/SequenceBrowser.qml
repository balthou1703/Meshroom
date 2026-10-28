import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MaterialIcons 2.2
import Utils 1.0

/**
 * SequenceBrowser is a file browser that groups the frames of image sequences into one entry,
 * as in Nuke ("render.####.exr  1001-1100"). Choosing an entry emits `selected` with its path
 * (the "####" pattern for a sequence). With `extensions` (e.g. [".mgt", ".mg"]) it browses those files only.
 *
 * The location bar takes a path, as in a file manager (Ctrl + L): a folder is opened, a file is chosen; "~" is the
 * home folder, a relative path is relative to the current folder. Alt + Up: the parent folder.
 */
Dialog {
    id: root

    property var live: null
    property string folder: ""
    property bool imagesOnly: true
    // Only these kinds of files when set (e.g. [".mgt", ".mg"]): no image sequences
    property var extensions: []
    property string browseTitle: "Read Image or Sequence"
    property var entries: []
    // The location bar shows a path that does not exist
    property bool missing: false

    signal selected(string path)

    title: browseTitle
    modal: true
    parent: Overlay.overlay
    anchors.centerIn: parent
    width: Math.min(760, parent ? parent.width * 0.9 : 760)
    height: Math.min(520, parent ? parent.height * 0.85 : 520)
    standardButtons: Dialog.Cancel | Dialog.Open

    function refresh() {
        list.currentIndex = -1
        missing = false
        if (!live)
            entries = []
        else
            entries = extensions.length ? live.listFiles(folder, extensions) : live.listFolder(folder, imagesOnly)
    }

    // A path typed in the location bar: a folder is opened, a file chosen (of the kinds browsed)
    function goTo(text) {
        var target = live.resolvePath(text, folder)
        if (target.type === "folder") {
            folder = target.path
            refresh()
        } else if (target.type === "file" && (!extensions.length || extensions.some(function(e) {
                       return target.path.toLowerCase().endsWith(e) }))) {
            selected(target.path)
            close()
        } else {
            missing = true
        }
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

    Shortcut {
        sequence: "Ctrl+L"
        enabled: root.visible
        onActivated: {
            locationField.forceActiveFocus()
            locationField.selectAll()
        }
    }
    Shortcut {
        sequence: "Alt+Up"
        enabled: root.visible
        onActivated: {
            root.folder = root.live.parentFolder(root.folder)
            root.refresh()
        }
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
                id: locationField
                Layout.fillWidth: true
                text: root.folder
                selectByMouse: true
                color: root.missing ? "#E57373" : palette.text
                placeholderText: "Type a path (Ctrl + L)"
                ToolTip.visible: hovered
                ToolTip.text: "Type a folder to open it, or a file to choose it (~: home folder, relative to the " +
                              "current folder)"
                onTextEdited: root.missing = false
                onAccepted: root.goTo(text)
            }
            CheckBox {
                visible: root.extensions.length === 0
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
                                : (modelData.type === "sequence" ? MaterialIcons.burst_mode
                                   : (root.extensions.length ? MaterialIcons.description : MaterialIcons.image))
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
