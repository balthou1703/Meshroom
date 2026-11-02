import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Controls 1.0
import MaterialIcons 2.2

/**
 * HierarchyEditorWindow displays a HierarchyEditorBackend:
 *  - left: a filterable tree of source items to check, sent to the right with "Add";
 *  - right: the editable tree of entries, organized by drag and drop (a child is computed after its
 *    parent, siblings are independent);
 *  - bottom: the actions of the request, called with the rows of the entries.
 */

NodeDialogWindow {
    id: root

    width: 1280
    height: 720
    minimumWidth: 800

    readonly property var entries: backend.entries
    readonly property var source: backend.source

    content: [
        MSplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal

            // Source pane
            ColumnLayout {
                visible: root.backend.hasSource
                SplitView.preferredWidth: root.width * 0.38
                SplitView.minimumWidth: 220
                spacing: 6

                Label {
                    Layout.fillWidth: true
                    visible: text.length > 0
                    text: root.backend.sourceHeader
                    textFormat: Text.StyledText
                    wrapMode: Text.WordWrap
                }
                RowLayout {
                    Layout.fillWidth: true
                    visible: !root.backend.sourceErrorMessage
                    MaterialLabel { text: MaterialIcons.filter_list; font.pointSize: root.font.pointSize }
                    TextField {
                        Layout.fillWidth: true
                        placeholderText: "Filter by name or type..."
                        selectByMouse: true
                        onTextChanged: if (root.source) root.source.setFilter(text)
                    }
                }
                PickerTreeView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    visible: !root.backend.sourceErrorMessage
                    model: root.source
                    columns: root.backend.sourceColumns
                    onLeafDoubleClicked: function(key) { root.backend.addSource(key) }
                    ToolTip.text: "Click: check / uncheck. Shift + click: check the range from the last clicked node.\nDouble click: add to the entries."
                    ToolTip.visible: hoverHandler.hovered
                    ToolTip.delay: 1500
                    HoverHandler { id: hoverHandler }
                }
                Label {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    visible: root.backend.sourceErrorMessage.length > 0
                    text: root.backend.sourceErrorMessage
                    textFormat: Text.StyledText
                    wrapMode: Text.WordWrap
                    color: "#E05050"
                }
                RowLayout {
                    Layout.fillWidth: true
                    visible: !root.backend.sourceErrorMessage
                    Button { text: "Select All"; onClicked: root.source.setAllChecked(true) }
                    Button { text: "Deselect All"; onClicked: root.source.setAllChecked(false) }
                    Item { Layout.fillWidth: true }
                    Label {
                        text: root.source ? root.source.checkedCount + " checked" : ""
                        opacity: 0.8
                    }
                }
            }

            // Transfer buttons
            ColumnLayout {
                visible: root.backend.hasSource
                SplitView.minimumWidth: implicitWidth
                SplitView.maximumWidth: implicitWidth
                spacing: 8
                Item { Layout.fillHeight: true }
                Button {
                    Layout.fillWidth: true
                    text: "Add  " + "→"
                    enabled: root.source && root.source.checkedCount > 0
                    ToolTip.text: "Add the checked nodes to the entries (at the root level)."
                    ToolTip.visible: hovered
                    onClicked: root.backend.addCheckedSources()
                }
                Button {
                    Layout.fillWidth: true
                    text: "←" + "  Remove"
                    enabled: root.entries.selectedCount > 0
                    ToolTip.text: "Remove the selected entries."
                    ToolTip.visible: hovered
                    onClicked: root.entries.removeSelected()
                }
                Item { Layout.fillHeight: true }
            }

            // Entries pane
            ColumnLayout {
                SplitView.fillWidth: true
                SplitView.minimumWidth: 360
                spacing: 6

                Label {
                    Layout.fillWidth: true
                    visible: text.length > 0
                    text: root.backend.entriesHeader
                    textFormat: Text.StyledText
                    wrapMode: Text.WordWrap
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: palette.base
                    border.color: entryTree.activeFocus ? palette.highlight : Qt.darker(palette.window, 1.4)
                    EntryTreeView {
                        id: entryTree
                        anchors.fill: parent
                        anchors.margins: 1
                        model: root.entries
                        columns: root.backend.columns
                        pathKey: root.backend.pathKey
                    }
                }

                // Edition toolbar
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 0
                    MaterialToolButton {
                        text: MaterialIcons.add
                        ToolTip.text: "Add an empty entry"
                        onClicked: root.entries.addEmpty()
                    }
                    MaterialToolButton {
                        text: MaterialIcons.content_copy
                        enabled: root.entries.selectedCount > 0
                        ToolTip.text: "Duplicate the selected entries"
                        onClicked: root.entries.duplicateSelected()
                    }
                    MaterialToolButton {
                        text: MaterialIcons.delete_sweep
                        enabled: root.entries.selectedCount > 0
                        ToolTip.text: "Remove the selected entries (Del). Their children are kept."
                        onClicked: root.entries.removeSelected()
                    }
                    ToolSeparator {}
                    MaterialToolButton {
                        text: MaterialIcons.arrow_upward
                        enabled: root.entries.selectedCount > 0
                        ToolTip.text: "Move up (Ctrl+Up)"
                        onClicked: root.entries.shiftSelected(-1)
                    }
                    MaterialToolButton {
                        text: MaterialIcons.arrow_downward
                        enabled: root.entries.selectedCount > 0
                        ToolTip.text: "Move down (Ctrl+Down)"
                        onClicked: root.entries.shiftSelected(1)
                    }
                    MaterialToolButton {
                        text: MaterialIcons.format_indent_increase
                        enabled: root.entries.selectedCount > 0
                        ToolTip.text: "Make a child of the previous entry (Ctrl+Right): computed after it"
                        onClicked: root.entries.indentSelected()
                    }
                    MaterialToolButton {
                        text: MaterialIcons.format_indent_decrease
                        enabled: root.entries.selectedCount > 0
                        ToolTip.text: "Move one level up (Ctrl+Left)"
                        onClicked: root.entries.outdentSelected()
                    }
                    MaterialToolButton {
                        text: MaterialIcons.subdirectory_arrow_left
                        enabled: root.entries.selectedCount > 0
                        ToolTip.text: "Unparent: move to the root level"
                        onClicked: root.entries.unparentSelected()
                    }
                    ToolSeparator {}
                    MaterialToolButton {
                        text: MaterialIcons.unfold_more
                        ToolTip.text: "Expand all"
                        onClicked: root.entries.setAllExpanded(true)
                    }
                    MaterialToolButton {
                        text: MaterialIcons.unfold_less
                        ToolTip.text: "Collapse all"
                        onClicked: root.entries.setAllExpanded(false)
                    }
                    ToolSeparator {}
                    MaterialToolButton {
                        text: MaterialIcons.clear_all
                        enabled: root.entries.count > 0
                        ToolTip.text: "Clear all the entries"
                        onClicked: clearDialog.open()
                    }
                    Item { Layout.fillWidth: true }
                    Label {
                        text: {
                            const n = root.entries.count
                            let t = n + (n === 1 ? " entry" : " entries")
                            if (root.entries.selectedCount > 0)
                                t += ", " + root.entries.selectedCount + " selected"
                            return t
                        }
                        opacity: 0.8
                    }
                }
            }
        },

        Dialog {
            id: clearDialog
            anchors.centerIn: parent
            modal: true
            title: "Clear All"
            standardButtons: Dialog.Yes | Dialog.No
            Label { text: "Remove all the entries?" }
            onAccepted: root.entries.clear()
        }
    ]

    buttons: [
        ActionButton {
            text: "Cancel"
            onClicked: root.close()
        },
        Repeater {
            model: root.backend.actions
            delegate: ActionButton {
                required property var modelData
                text: modelData.label
                toolTipText: modelData.tooltip
                fillColor: modelData.color
                onClicked: {
                    if (root.backend.trigger(modelData.name))
                        root.close()
                }
            }
        }
    ]
}
