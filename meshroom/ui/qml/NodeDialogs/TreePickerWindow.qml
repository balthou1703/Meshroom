import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MaterialIcons 2.2

/**
 * TreePickerWindow displays a TreePickerBackend: a filterable tree of checkable items, validated
 * by a single button.
 */

NodeDialogWindow {
    id: root

    width: 560
    height: 640

    readonly property var model: backend ? backend.model : null

    content: [
        RowLayout {
            Layout.fillWidth: true
            visible: !root.backend.errorMessage
            MaterialLabel { text: MaterialIcons.filter_list; font.pointSize: root.font.pointSize }
            TextField {
                id: filterField
                Layout.fillWidth: true
                placeholderText: "Filter by name or type..."
                selectByMouse: true
                onTextChanged: root.model.setFilter(text)
                Component.onCompleted: forceActiveFocus()
            }
        },

        PickerTreeView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: !root.backend.errorMessage
            model: root.model
            columns: root.backend.columns
            onLeafDoubleClicked: function(key) {
                // Quick pick: in single selection, a double click validates the dialog
                if (!root.backend.multiSelect) {
                    root.backend.accept()
                    root.close()
                }
            }
        },

        Label {
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: root.backend.errorMessage.length > 0
            text: root.backend.errorMessage
            textFormat: Text.StyledText
            wrapMode: Text.WordWrap
            color: "#E05050"
        },

        RowLayout {
            Layout.fillWidth: true
            visible: root.backend.multiSelect && !root.backend.errorMessage
            Button { text: "Select All"; onClicked: root.model.setAllChecked(true) }
            Button { text: "Deselect All"; onClicked: root.model.setAllChecked(false) }
            Item { Layout.fillWidth: true }
            Label {
                text: root.model ? root.model.checkedCount + " / " + root.model.leafCount + " selected" : ""
                opacity: 0.8
            }
        }
    ]

    buttons: [
        ActionButton {
            text: "Cancel"
            onClicked: root.close()
        },
        ActionButton {
            text: root.backend.acceptLabel
            fillColor: "#4A7C59"
            enabled: !root.backend.errorMessage
            onClicked: {
                root.backend.accept()
                root.close()
            }
        }
    ]
}
