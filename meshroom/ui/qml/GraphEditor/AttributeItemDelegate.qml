import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Dialogs

import MaterialIcons 2.2
import Utils 1.0
import Controls 1.0
import Compositing 1.0
import "AttributeControls" as AttributeControls
import "AnySetUtils.js" as AnySetUtils

/**
 * Instantiate a control to visualize and edit an Attribute based on its type.
 */

RowLayout {
    id: root

    property variant attribute: null
    property bool readOnly: false  // Whether the attribute's value can be modified
    property bool objectsHideable: true
    property string filterText: ""

    property alias label: parameterLabel  // Accessor to the internal Label (attribute's name)
    property int labelWidth               // Shortcut to set the fixed size of the Label

    readonly property bool editable: !attribute.isOutput && !attribute.isLink &&
                                     !readOnly && !(attribute.keyable && _currentScene.selectedViewId === "-1")
    readonly property bool isAnySetChild: !!attribute && !!attribute.root && attribute.root.type === "AnySet"
    property var errorMessages: attribute.errorMessages

    signal doubleClicked(var mouse, var attr)
    signal inAttributeClicked(var srcItem, var mouse, var inAttributes)
    signal outAttributeClicked(var srcItem, var mouse, var outAttributes)
    signal showInViewer(var attr)

    Connections {
        target: attribute
        function onValueChanged() {
            root.errorMessages = attribute.errorMessages
        }
    }

    Component {
        id: removeAnySetAttributeMenuComp
        Menu {
            id: anySetMenu

            property real preferredX: 0
            property real preferredY: 0

            MenuItem {
                text: "Move Up"
                enabled: AnySetUtils.canMoveBy(attribute, -1)
                onTriggered: _currentScene.moveAnySetAttribute(attribute, -1)
            }
            MenuItem {
                text: "Move Down"
                enabled: AnySetUtils.canMoveBy(attribute, 1)
                onTriggered: _currentScene.moveAnySetAttribute(attribute, 1)
            }
            MenuSeparator {}
            MenuItem {
                text: "Rename Attribute"
                enabled: root.isAnySetChild
                onTriggered: {
                    var dialog = renameAnySetAttributeDialogComp.createObject(Overlay.overlay, {
                        "targetAttribute": attribute,
                        "preferredX": anySetMenu.preferredX,
                        "preferredY": anySetMenu.preferredY
                    })
                    dialog.open()
                }
            }
            MenuItem {
                text: "Remove Attribute"
                enabled: root.isAnySetChild
                onTriggered: _currentScene.removeAnySetAttribute(attribute)
            }
        }
    }

    Component {
        id: renameAnySetAttributeDialogComp
        Dialog {
            id: renameDialog

            property var targetAttribute: null
            property real preferredX: 0
            property real preferredY: 0

            title: "Rename Attribute"
            modal: true
            parent: Overlay.overlay
            contentWidth: 280
            x: parent ? Math.max(0, Math.min(preferredX, parent.width - width)) : 0
            y: parent ? Math.max(0, Math.min(preferredY, parent.height - height)) : 0
            standardButtons: Dialog.Ok | Dialog.Cancel
            closePolicy: Popup.CloseOnEscape

            GridLayout {
                columns: 2
                columnSpacing: 8
                rowSpacing: 8
                width: renameDialog.contentWidth

                Label {
                    text: "Name"
                    Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                }
                TextField {
                    id: nameField
                    Layout.fillWidth: true
                    Layout.minimumWidth: 0
                    text: renameDialog.targetAttribute ? renameDialog.targetAttribute.name : ""
                    selectByMouse: true
                    validator: RegularExpressionValidator { regularExpression: /^[A-Za-z_][A-Za-z0-9_]*$/ }
                    onAccepted: renameDialog.accept()
                }

                Label {
                    text: "Label"
                    Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                }
                TextField {
                    id: labelField
                    Layout.fillWidth: true
                    Layout.minimumWidth: 0
                    text: renameDialog.targetAttribute ? renameDialog.targetAttribute.label : ""
                    selectByMouse: true
                    onAccepted: renameDialog.accept()
                }
            }

            onOpened: {
                nameField.forceActiveFocus()
                nameField.selectAll()
            }

            onAccepted: {
                if (targetAttribute && nameField.acceptableInput && labelField.text.trim() !== "") {
                    _currentScene.renameAnySetAttribute(targetAttribute, nameField.text.trim(), labelField.text.trim())
                }
                destroy()
            }

            onRejected: destroy()
        }
    }

    Component {
        id: addAnySetAttributeDialogComp
        Dialog {
            id: addDialog

            property var targetAnySet: null

            title: "Add Attribute"
            modal: true
            parent: Overlay.overlay
            contentWidth: 280
            anchors.centerIn: parent
            standardButtons: Dialog.Ok | Dialog.Cancel
            closePolicy: Popup.CloseOnEscape

            GridLayout {
                columns: 2
                columnSpacing: 8
                rowSpacing: 8
                width: addDialog.contentWidth

                Label { text: "Name"; Layout.alignment: Qt.AlignRight | Qt.AlignVCenter }
                TextField {
                    id: addNameField
                    Layout.fillWidth: true
                    Layout.minimumWidth: 0
                    selectByMouse: true
                    placeholderText: "shotName"
                    validator: RegularExpressionValidator { regularExpression: /^[A-Za-z_][A-Za-z0-9_]*$/ }
                    onAccepted: addDialog.accept()
                }

                Label { text: "Label"; Layout.alignment: Qt.AlignRight | Qt.AlignVCenter }
                TextField {
                    id: addLabelField
                    Layout.fillWidth: true
                    Layout.minimumWidth: 0
                    selectByMouse: true
                    placeholderText: addNameField.text
                    onAccepted: addDialog.accept()
                }

                Label { text: "Type"; Layout.alignment: Qt.AlignRight | Qt.AlignVCenter }
                ComboBox {
                    id: addTypeField
                    Layout.fillWidth: true
                    // Any value (LiveValue) or a typed value, shown with its usual editor
                    model: ["LiveValue", "StringParam", "File", "IntParam", "FloatParam", "BoolParam"]
                }
            }

            onOpened: addNameField.forceActiveFocus()

            onAccepted: {
                if (targetAnySet && addNameField.acceptableInput) {
                    var label = addLabelField.text.trim() !== "" ? addLabelField.text.trim() : addNameField.text.trim()
                    _currentScene.addAnySetAttribute(targetAnySet, addNameField.text.trim(), label, addTypeField.currentText)
                }
                destroy()
            }

            onRejected: destroy()
        }
    }

    spacing: 2

    // Groups show their own header, except the colors (a single line editor)
    // Colors ("rgbaColor") and values per channel ("linkedColor"): edited by RgbaColorField
    readonly property bool isRgbaColor: attribute.type === "GroupAttribute" && !attribute.keyable &&
                                        (attribute.desc.semantic === "rgbaColor" || attribute.desc.semantic === "linkedColor")

    // Parameter of a compositing node that can be animated (keys per frame)
    readonly property bool animatable: {
        if (!attribute || attribute.isOutput || attribute.root || !attribute.node || !attribute.node.isImageOpNode)
            return false
        var node = attribute.node
        if (!node.hasInternalAttribute("animation") ||
            (node.hasInternalAttribute(attribute.name) && node.internalAttribute(attribute.name) === attribute))
            return false
        if (attribute.desc.semantic === "liveAction")
            return false
        return ["IntParam", "FloatParam", "BoolParam", "ChoiceParam", "Vec2d"].indexOf(attribute.type) >= 0 ||
               root.isRgbaColor
    }

    Pane {
        visible: attribute.type !== "GroupAttribute" || root.isRgbaColor
        background: Rectangle {
            id: background
            color: Qt.darker(parent.palette.window, 1.1)
        }
        padding: 0
        Layout.preferredWidth: labelWidth || implicitWidth
        Layout.fillHeight: true

        RowLayout {
            spacing: 0
            width: parent.width
            height: parent.height

            // In connection
            MaterialToolButton {
                id: navButtonIn

                property bool shouldBeVisible: (object != undefined && object.hasAnyInputLinks)

                text: MaterialIcons.login
                enabled: shouldBeVisible
                font.pointSize: 8
                Layout.fillHeight: true
                visible: shouldBeVisible

                MouseArea {
                    anchors.fill: parent
                    acceptedButtons: Qt.LeftButton | Qt.MiddleButton | Qt.RightButton

                    onClicked: function(mouse) {
                        root.inAttributeClicked(navButtonIn, mouse, object.allInputLinks)
                    }
                }

            }

            Label {
                id: parameterLabel

                Layout.fillHeight: true
                Layout.fillWidth: true
                horizontalAlignment: attribute.isOutput ? Qt.AlignRight : Qt.AlignLeft
                verticalAlignment: Text.AlignVCenter
                elide: Label.ElideRight
                padding: 5
                wrapMode: Label.WrapAtWordBoundaryOrAnywhere

                text: attribute.isMandatory && attribute.isDefault ? `\* ${object.label}` : object.label

                color: {
                    if (object != undefined && (object.hasAnyOutputLinks || object.isLink) && !object.enabled)
                        return Colors.lightgrey
                    else
                        return palette.text
                }

                // Tooltip hint with attribute's description
                ToolTip {
                    id: parameterTooltip

                    // Position in y at mouse position
                    y: parameterMA.mouseY + 10

                    text: {
                        return `<b>${object.desc.name}:</b> ${attribute.type}<br>${Format.plainToHtml(object.desc.description)}`
                    }
                    visible: parameterMA.containsMouse
                    delay: 800
                }

                // Make label bold if attribute's value is not the default one
                font.bold: !object.isOutput && !object.isDefault

                // Make label italic if attribute is a link
                font.italic: object.isLink

                MouseArea {
                    id: parameterMA
                    anchors.fill: parent
                    hoverEnabled: true
                    acceptedButtons: Qt.AllButtons
                    onDoubleClicked: function(mouse) { root.doubleClicked(mouse, root.attribute) }

                    property Component menuComp: Menu {
                        id: paramMenu

                        property real preferredX: 0
                        property real preferredY: 0
                        property bool isFileAttribute: attribute.type === "File"
                        property bool isFilepath: isFileAttribute && Filepath.isFile(attribute.evalValue)

                        MenuItem {
                            text: "Reset To Default Value"
                            enabled: root.editable && !attribute.isDefault
                            onTriggered: {
                                _currentScene.resetAttribute(attribute)
                            }
                        }
                        MenuItem {
                            text: "Copy"
                            enabled: !attribute.keyable && attribute.value != ""
                            onTriggered: {
                                Clipboard.clear()
                                Clipboard.setText(attribute.value)
                            }
                        }
                        MenuItem {
                            text: "Paste"
                            enabled: Clipboard.getText() != "" && !attribute.keyable && root.editable
                            onTriggered: {
                                _currentScene.setAttribute(attribute, Clipboard.getText())
                            }
                        }

                        MenuSeparator {
                            visible: paramMenu.isFileAttribute
                            height: visible ? implicitHeight : 0
                        }

                        MenuItem {
                            visible: paramMenu.isFileAttribute
                            height: visible ? implicitHeight : 0
                            text: paramMenu.isFilepath ? "Open Containing Folder" : "Open Folder"
                            onClicked: paramMenu.isFilepath ? Qt.openUrlExternally(Filepath.dirname(attribute.evalValue)) :
                                                              Qt.openUrlExternally(Filepath.stringToUrl(attribute.evalValue))
                        }

                        MenuItem {
                            visible: paramMenu.isFilepath
                            height: visible ? implicitHeight : 0
                            text: "Open File"
                            onClicked: Qt.openUrlExternally(Filepath.stringToUrl(attribute.evalValue))
                        }

                        MenuItem {
                            visible: attribute.isOutput && (attribute.is2dDisplayable || attribute.is3dDisplayable || attribute.isTextDisplayable)
                            height: visible ? implicitHeight : 0
                            text: {
                                if (attribute.is2dDisplayable)
                                    return "Show in 2D Viewer"
                                if (attribute.isTextDisplayable)
                                    return "Show in Text Viewer"
                                return "Show in 3D Viewer"
                            }
                            onClicked: root.showInViewer(attribute)
                        }

                        MenuSeparator {
                            visible: root.isAnySetChild
                            height: visible ? implicitHeight : 0
                        }

                        MenuItem {
                            visible: root.isAnySetChild
                            height: visible ? implicitHeight : 0
                            text: "Move Up"
                            enabled: AnySetUtils.canMoveBy(attribute, -1)
                            onTriggered: _currentScene.moveAnySetAttribute(attribute, -1)
                        }

                        MenuItem {
                            visible: root.isAnySetChild
                            height: visible ? implicitHeight : 0
                            text: "Move Down"
                            enabled: AnySetUtils.canMoveBy(attribute, 1)
                            onTriggered: _currentScene.moveAnySetAttribute(attribute, 1)
                        }

                        MenuSeparator {
                            visible: root.isAnySetChild
                            height: visible ? implicitHeight : 0
                        }

                        MenuItem {
                            visible: root.isAnySetChild
                            height: visible ? implicitHeight : 0
                            text: "Rename Attribute"
                            onTriggered: {
                                var dialog = renameAnySetAttributeDialogComp.createObject(Overlay.overlay, {
                                    "targetAttribute": attribute,
                                    "preferredX": paramMenu.preferredX,
                                    "preferredY": paramMenu.preferredY
                                })
                                dialog.open()
                            }
                        }

                        MenuItem {
                            visible: root.isAnySetChild
                            height: visible ? implicitHeight : 0
                            text: "Remove Attribute"
                            onTriggered: _currentScene.removeAnySetAttribute(attribute)
                        }

                    }

                    onClicked: function(mouse) {
                        forceActiveFocus()
                        if (mouse.button == Qt.RightButton) {
                            var menu = menuComp.createObject(parameterLabel)
                            var position = parameterMA.mapToItem(Overlay.overlay, mouse.x, mouse.y)
                            menu.preferredX = position.x
                            menu.preferredY = position.y
                            menu.parent = parameterLabel
                            menu.popup()
                        }
                    }
                }
            }

            // Key of an animated parameter of a compositing node at the current frame (see meshroom/live/animation.py)
            Loader {
                active: root.animatable
                visible: active
                Layout.fillHeight: true
                sourceComponent: MaterialToolButton {
                    readonly property string keyState: {
                        // Re-evaluated when the frame or the keys change
                        var frame = _currentScene.live.currentFrame
                        var keys = attribute.node.internalAttribute("animation").value
                        return _currentScene.keyState(attribute)
                    }
                    text: MaterialIcons.diamond
                    font.pointSize: 8
                    padding: 3
                    enabled: !attribute.isLink && !root.readOnly
                    opacity: keyState === "" ? 0.35 : 1.0
                    textColor: keyState === "key" ? "#E8A33D" : keyState === "animated" ? "#5DA5E8" : palette.text
                    ToolTip.visible: hovered
                    ToolTip.delay: 600
                    ToolTip.text: keyState === "key" ? "Remove the key at frame " + _currentScene.live.currentFrame +
                                                    " (right click: remove the animation)"
                                : keyState === "animated" ? "Animated: set a key at frame " + _currentScene.live.currentFrame +
                                                         " (right click: remove the animation)"
                                : "Set a key at frame " + _currentScene.live.currentFrame + " (animate this value)"
                    onClicked: _currentScene.toggleKey(attribute)
                    MouseArea {
                        anchors.fill: parent
                        acceptedButtons: Qt.RightButton
                        onClicked: _currentScene.removeAnimation(attribute)
                    }
                }
            }

            MaterialLabel {
                property bool isDisplayable: attribute.isOutput && (attribute.is2dDisplayable || attribute.is3dDisplayable || attribute.isTextDisplayable)
                property bool isDisplayed: attribute === _currentScene.displayedAttr2D || _currentScene.displayedAttrs3D.count && _currentScene.displayedAttrs3D.contains(attribute)
                text: isDisplayed ? MaterialIcons.visibility : MaterialIcons.visibility_off
                enabled: isDisplayed
                visible: isDisplayable
                ToolTip.text: {
                    if (attribute.is2dDisplayable)
                        return "This attribute is displayable in the 2D viewer."
                    if (attribute.isTextDisplayable)
                        return "This attribute is displayable in the Text viewer."
                    return "This attribute is displayable in the 3D viewer."
                }

                padding: 4
                font.pointSize: 8
            }

            MaterialToolButton {
                id: navButtonOut

                property bool shouldBeVisible: (attribute != undefined && attribute.hasAnyOutputLinks)

                text: MaterialIcons.logout
                font.pointSize: 8
                enabled: shouldBeVisible
                Layout.fillHeight: true
                visible: shouldBeVisible

                MouseArea {
                    anchors.fill: parent
                    acceptedButtons: Qt.LeftButton | Qt.MiddleButton | Qt.RightButton

                    onClicked: function(mouse) {
                        root.outAttributeClicked(navButtonOut, mouse, attribute.allOutputLinks)
                    }
                }


            }

            MaterialLabel {
                visible: attribute.desc.advanced
                text: MaterialIcons.build
                color: palette.mid
                font.pointSize: 8
                padding: 4
            }
        }
    }

    function setTextFieldAttribute(value) {
        // editingFinished called even when TextField is readonly
        if (!editable)
            return
        switch (attribute.type) {
            case "IntParam":
            case "FloatParam":
                // We do not set a number because we want to keep the invalid expression
                if(attribute.keyable)
                    _currentScene.addAttributeKeyValue(root.attribute, _currentScene.selectedViewId, Number(value))
                else
                    _currentScene.setAttribute(root.attribute, Number(value))
                break
            case "File":
                _currentScene.setAttribute(root.attribute, value)
                break
            default:
                _currentScene.setAttribute(root.attribute, value.trim())
                break
        }
    }


    Loader {
        Layout.fillWidth: true
        id: inputField

        sourceComponent: {
            // PushButtonParam, Flow and the streams of the compositing nodes (Image, Camera, Geo, Matrix) always have
            // value == undefined: exclude them from this check
            if (attribute.type != "PushButtonParam" && attribute.type != "Flow" && !attribute.isStream &&
                !attribute.keyable && attribute.value === undefined) {
                return notComputedComponent
            }
            switch (attribute.type) {
                case "PushButtonParam":
                    return pushButtonComponent
                case "ChoiceParam":
                    return attribute.desc.exclusive ? choiceComponent : choiceMultiComponent
                case "IntParam": return sliderComponent
                case "FloatParam":
                    if (attribute.desc.semantic === 'color/hue')
                        return colorHueComponent
                    return sliderComponent
                case "BoolParam":
                    return checkboxComponent
                case "ListAttribute":
                    return listAttributeComponent
                case "GroupAttribute":
                    return root.isRgbaColor ? rgbaColorComponent : groupAttributeComponent
                case "Vec2d":
                    return root.attribute.keyable ? groupAttributeComponent : vec2Component
                case "AnySet":
                    return groupAttributeComponent
                case "StringParam":
                    if (attribute.desc.semantic === "rotoShapes" && !attribute.isLink)
                        return rotoShapesComponent
                    if (attribute.desc.semantic === "tracks" && !attribute.isLink)
                        return tracksComponent
                    if (attribute.desc.semantic.startsWith("trackNames") && !attribute.isLink && !!_currentScene && !!_currentScene.live)
                        return trackPickerComponent
                    if (attribute.desc.semantic === "autoTracks" || attribute.desc.semantic === "cameraSolve")
                        return jsonSummaryComponent
                    if ((attribute.desc.semantic === "colorCurves" || attribute.desc.semantic === "timeCurve") && !attribute.isLink)
                        return curvesComponent
                    if (attribute.desc.semantic.startsWith("imageLayer") && !attribute.isLink && !!_currentScene && !!_currentScene.live)
                        return imageLayerComponent
                    if (attribute.desc.semantic === "colorspace" && !attribute.isLink && !!_currentScene && !!_currentScene.live)
                        return colorspaceComponent
                    if (attribute.desc.semantic.includes('multiline'))
                        return textAreaComponent
                    return textFieldComponent
                case "ColorParam":
                    return colorComponent
                case "Flow":
                case "Image":
                case "Camera":
                case "Geo":
                case "Matrix":
                    return flowComponent
                case "File":
                    if (attribute.desc.semantic === "imageSequence" && !attribute.isOutput)
                        return sequenceFieldComponent
                    if ((attribute.desc.semantic.startsWith("saveFile") || attribute.desc.semantic.startsWith("openFile")) &&
                        !attribute.isOutput && !attribute.isLink)
                        return saveFileComponent
                    return textFieldComponent
                default:
                    return textFieldComponent
            }
        }

        Component {
            id: notComputedComponent
            AttributeControls.NotComputed {}
        }

        Component {
            id: pushButtonComponent
            RowLayout {
                // Compositing actions ("liveAction"): run in the background by the live controller, with progress
                readonly property bool liveAction: attribute.desc.semantic === "liveAction" && !!_currentScene && !!_currentScene.live
                // Any action running: the buttons wait; this one running: its progress
                readonly property bool busy: liveAction && _currentScene.live.actionProgress >= 0
                readonly property bool running: busy && _currentScene.live.actionNode === attribute.node &&
                                                _currentScene.live.actionName === attribute.name
                Button {
                    text: attribute.label
                    enabled: root.editable && !parent.busy
                    onClicked: {
                        if (parent.liveAction)
                            _currentScene.live.runAction(attribute.node, attribute.name)
                        else
                            attribute.clicked()
                    }
                }
                ProgressBar {
                    Layout.fillWidth: true
                    visible: parent.running
                    from: 0
                    to: 1
                    value: parent.running ? _currentScene.live.actionProgress : 0
                }
                MaterialToolButton {
                    visible: parent.running
                    text: MaterialIcons.stop_circle
                    font.pointSize: 10
                    ToolTip.text: "Stop " + _currentScene.live.actionText
                    onClicked: _currentScene.live.stopAction()
                }
            }
        }

        Component {
            id: jsonSummaryComponent
            // Results of compositing actions (autotrack tracks, camera solve): a summary line
            Label {
                text: {
                    attribute.value  // Refreshed when the value changes
                    return _currentScene && _currentScene.live ? _currentScene.live.jsonSummary(attribute) : ""
                }
                opacity: 0.8
                elide: Text.ElideRight
            }
        }

        Component {
            id: flowComponent
            // Flow has no data to display; connection state is shown via the attribute pin
            Item {}
        }

        Component {
            id: textFieldComponent
            AttributeControls.TextFieldRow {
                text: attribute.value
                mandatory: attribute.isMandatory
                attribute: root.attribute
                editable: root.editable
                onEditingFinished: (text) => setTextFieldAttribute(text)
                onAccepted: (parameterLabel, text) => {
                    setTextFieldAttribute(text)
                    parameterLabel.forceActiveFocus()
                }
                onDestruction: (activeFocus, text) => {
                    if (activeFocus)
                        setTextFieldAttribute(text)
                }
                onDropped: (hasUrls, hasText, urlText, text) => {
                    if (hasUrls)
                        setTextFieldAttribute(urlText)
                    else if (hasText)
                        setTextFieldAttribute(text)
                }
                onTriggered: (text, start, end, length, clipboard) => {
                    const before = text.substr(0, start)
                    const after = text.substr(end, length)
                    const updatedValue = before + clipboardText + after
                    setTextFieldAttribute(updatedValue)
                    // Set the cursor at the end of the added text
                    textField.cursorPosition = before.length + clipboard.length
                }
            }
        }

        // File written by a node (e.g. the USD exports of a CameraTracker): text field and a "save as" dialog;
        // "saveFile:Name (*.ext)" gives the name filter. "openFile:Name (*.ext)": a file read by the node (open dialog)
        Component {
            id: saveFileComponent
            RowLayout {
                spacing: 2
                TextField {
                    Layout.fillWidth: true
                    text: attribute.value
                    readOnly: !root.editable
                    selectByMouse: true
                    readonly property bool opening: attribute.desc.semantic.startsWith("openFile")
                    placeholderText: opening ? "Path of the file to read" : "Path of the file to write"
                    onEditingFinished: setTextFieldAttribute(text.trim())
                }
                MaterialToolButton {
                    text: MaterialIcons.folder
                    enabled: root.editable
                    font.pointSize: 11
                    ToolTip.text: attribute.desc.semantic.startsWith("openFile") ? "Choose the file" : "Choose where to write the file"
                    onClicked: saveFileDialog.open()
                }
                FileDialog {
                    id: saveFileDialog
                    title: (attribute.desc.semantic.startsWith("openFile") ? "Open " : "Save ") + attribute.label
                    fileMode: attribute.desc.semantic.startsWith("openFile") ? FileDialog.OpenFile : FileDialog.SaveFile
                    readonly property string filter: attribute.desc.semantic.indexOf(":") >= 0
                                                     ? attribute.desc.semantic.split(":").slice(1).join(":") : ""
                    nameFilters: filter ? [filter, "All files (*)"] : ["All files (*)"]
                    currentFolder: attribute.value ? Filepath.stringToUrl(Filepath.dirname(attribute.value)) : ""
                    onAccepted: setTextFieldAttribute(Filepath.urlToString(selectedFile))
                }
            }
        }

        Component {
            id: sequenceFieldComponent
            // File of a compositing Read: text field and a browser grouping image sequences, frame range below
            ColumnLayout {
                spacing: 1
            RowLayout {
                Layout.fillWidth: true
                spacing: 2
                Item {
                    Layout.fillWidth: true
                    implicitHeight: sequenceTextField.implicitHeight
                    AttributeControls.TextFieldRow {
                        id: sequenceTextField
                        text: attribute.value
                        mandatory: attribute.isMandatory
                        attribute: root.attribute
                        editable: root.editable
                        onEditingFinished: (text) => setTextFieldAttribute(text)
                        onAccepted: (parameterLabel, text) => {
                            setTextFieldAttribute(text)
                            parameterLabel.forceActiveFocus()
                        }
                        onDestruction: (activeFocus, text) => {
                            if (activeFocus)
                                setTextFieldAttribute(text)
                        }
                        onDropped: (hasUrls, hasText, urlText, text) => {
                            if (hasUrls)
                                setTextFieldAttribute(urlText)
                            else if (hasText)
                                setTextFieldAttribute(text)
                        }
                        onTriggered: (text, start, end, length, clipboard) => {
                            setTextFieldAttribute(text.substr(0, start) + clipboard + text.substr(end, length))
                        }
                    }
                }
                MaterialToolButton {
                    text: MaterialIcons.folder
                    enabled: root.editable
                    font.pointSize: 11
                    ToolTip.text: "Browse Images and Sequences"
                    onClicked: {
                        sequenceBrowserLoader.active = true
                        sequenceBrowserLoader.item.openAt(attribute.value)
                    }
                }
                Loader {
                    id: sequenceBrowserLoader
                    active: false
                    sourceComponent: SequenceBrowser {
                        live: _currentScene.live
                        onSelected: (path) => setTextFieldAttribute(path)
                        onClosed: Qt.callLater(function() { sequenceBrowserLoader.active = false })
                    }
                }
            }
                // Frame range of the sequence (with the frame offset of the Read)
                Label {
                    Layout.leftMargin: 4
                    readonly property string frameRange: {
                        var node = root.attribute.node
                        if (!node || !_currentScene || !_currentScene.live)
                            return ""
                        // Re-evaluated when the file or the offset changes
                        root.attribute.value
                        node.attribute("frameOffset") ? node.attribute("frameOffset").value : 0
                        return _currentScene.live.nodeFrameRange(node)
                    }
                    visible: frameRange !== ""
                    text: frameRange
                    font.pointSize: 8
                    opacity: 0.75
                    ToolTip.text: "Frame range of the sequence, with the frame offset"
                    ToolTip.visible: frameRangeMouse.containsMouse
                    MouseArea {
                        id: frameRangeMouse
                        anchors.fill: parent
                        hoverEnabled: true
                        acceptedButtons: Qt.NoButton
                    }
                }
            }
        }

        Component {
            id: textAreaComponent
            AttributeControls.TextAreaFlick {
                label: attribute.value
                isLarge: attribute.desc.semantic.includes("large")
                editable: root.editable
                onEditingFinished: (text) => setTextFieldAttribute(text)
                onDestruction: (activeFocus, text) => {
                    if (activeFocus)
                        setTextFieldAttribute(text)
                }
                onDropped: (hasUrls, hasText, urlText, text) => {
                    if (hasUrls)
                        setTextFieldAttribute(urlText)
                    else if (hasText)
                        setTextFieldAttribute(text)
                }
            }
        }

        Component {
            id: colorComponent
            AttributeControls.Color {
                id: colorControl
                value: root.attribute.value
                editable: root.editable
                onClicked: (checked, previousColor, colorTextValue) =>{
                    if (checked) {
                        if (colorTextValue == "") {
                            if (previousColor !== "")
                                _currentScene.setAttribute(root.attribute, previousColor)
                            else
                                _currentScene.setAttribute(root.attribute, "#0000FF")
                        }
                        else
                            _currentScene.setAttribute(root.attribute, colorTextValue)
                    } else {
                        colorControl.previousColor = root.attribute.value
                        _currentScene.setAttribute(root.attribute, "")
                    }
                }
                onEditingFinished: (text) => setTextFieldAttribute(text)
                onAccepted: (text) => setTextFieldAttribute(text)
                onDestruction: (activeFocus, text) => {
                    if (activeFocus)
                        setTextFieldAttribute(text)
                }
            }
        }

        Component {
            id: choiceComponent

            AttributeControls.Choice {
                value: root.attribute.value
                values: root.attribute.values
                enabled: root.editable

                onEditingFinished: (value) => {
                    _currentScene.setAttribute(root.attribute, value)
                }
            }
        }

        // Tracks of the connected Tracker (Stabilize, MotionBlur, CornerPin), checked in a list
        Component {
            id: trackPickerComponent
            AttributeControls.TrackPicker {
                attribute: root.attribute
                editable: root.editable
                // "trackNames:4": exactly 4 tracks, in order
                count: root.attribute.desc.semantic.indexOf(":") >= 0 ? Number(root.attribute.desc.semantic.split(":")[1]) : 0
            }
        }

        // Layer of the image entering the node (e.g. Shuffle): the layers of its last render, or any name
        Component {
            id: imageLayerComponent

            ComboBox {
                id: layerCombo
                property var layers: []
                // Updated after the renders, but not while the name is being typed or the list is open
                function refresh() {
                    // "imageLayer" reads the "input" image, "imageLayer:a,b" the first connected of a and b
                    var semantic = root.attribute.desc.semantic
                    var inputs = semantic.indexOf(":") >= 0 ? semantic.split(":")[1] : "input"
                    var names = _currentScene.live.inputLayers(root.attribute.node, inputs)
                    var value = root.attribute.value
                    if (value && names.indexOf(value) < 0)
                        names = names.concat([value])
                    if (JSON.stringify(names) !== JSON.stringify(layers))
                        layers = names
                }
                Component.onCompleted: refresh()
                Connections {
                    target: _currentScene.live
                    function onStatusRevisionChanged() {
                        if (!layerCombo.activeFocus && !layerCombo.contentItem.activeFocus && !layerCombo.popup.visible)
                            layerCombo.refresh()
                    }
                }
                Connections {
                    target: root.attribute
                    function onValueChanged() { layerCombo.refresh() }
                }
                model: layers
                editable: true
                enabled: root.editable
                currentIndex: layers.indexOf(root.attribute.value)
                ToolTip.text: "Layers of the image entering the node (after a render); a name can also be typed."
                ToolTip.visible: hovered
                ToolTip.delay: 800
                onActivated: function(index) {
                    if (layers[index] !== root.attribute.value)
                        _currentScene.setAttribute(root.attribute, layers[index])
                }
                onAccepted: {
                    var text = editText.trim()
                    if (text !== "" && text !== root.attribute.value)
                        _currentScene.setAttribute(root.attribute, text)
                }
            }
        }

        // Colorspace of the color config (OCIO or builtin): its colorspaces, "auto" for the files, or any name
        Component {
            id: colorspaceComponent

            ComboBox {
                id: colorspaceCombo
                readonly property bool isFile: root.attribute.desc.name === "colorspace"
                property var names: {
                    var list = _currentScene.live.colorspaces.slice()
                    if (isFile)
                        list.unshift("auto")
                    var value = root.attribute.value
                    if (value && list.indexOf(value) < 0)
                        list.push(value)
                    return list
                }
                model: names
                editable: true
                enabled: root.editable
                currentIndex: names.indexOf(root.attribute.value)
                ToolTip.text: _currentScene.live.colorConfigDescription + (isFile ?
                              "\n'auto': sRGB for 8/16-bit integer files, the working space for float files." : "")
                ToolTip.visible: hovered
                ToolTip.delay: 800
                onActivated: function(index) {
                    if (names[index] !== root.attribute.value)
                        _currentScene.setAttribute(root.attribute, names[index])
                }
                onAccepted: {
                    var text = editText.trim()
                    if (text !== "" && text !== root.attribute.value)
                        _currentScene.setAttribute(root.attribute, text)
                }
            }
        }

        Component {
            id: choiceMultiComponent

            AttributeControls.ChoiceMulti {
                value: root.attribute.value
                values: root.attribute.values
                enabled: root.editable
                customValueColor: Colors.orange

                onToggled: (value, checked) => {
                    var currentValue = root.attribute.value;
                    if (!checked) {
                        currentValue.splice(currentValue.indexOf(value), 1);
                    } else {
                        currentValue.push(value);
                    }
                    _currentScene.setAttribute(attribute, currentValue);
                }
            }
        }

        Component {
            id: sliderComponent
            AttributeControls.SliderField {
                checked: root.attribute.keyable
                                ? root.attribute.keyValues.getValueAtKeyOrDefault(_currentScene.selectedViewId)
                                : root.attribute.value
                type: root.attribute.type
                length: (root.attribute.desc.range && root.attribute.desc.range.length) || 0
                start: (root.attribute.desc.range && root.attribute.desc.range[0]) || 0
                end: (root.attribute.desc.range && root.attribute.desc.range[1]) || 0
                step: (root.attribute.desc.range && root.attribute.desc.range[2]) || 0
                editable: root.editable
                onEditingFinished: (hasExprError, evaluatedValue, text, displayValue) => {
                    if (!hasExprError) {
                        setTextFieldAttribute(evaluatedValue)
                        // Restore binding
                        text = Qt.binding(function() { return String(displayValue); })
                    }
                }
                onAccepted: (hasExprError, evaluatedValue, text, displayValue) => {
                    if (!hasExprError) {
                        setTextFieldAttribute(evaluatedValue)
                        // Restore binding
                        text = Qt.binding(function() { return String(displayValue); })
                    }
                }
                Component.onDestruction: (activeFocus, hasExprError, displayValue) =>  {
                    if (activeFocus) {
                        if (!hasExprError)
                            setTextFieldAttribute(evaluatedValue)
                    }
                }
                // Compositing nodes update the image while the slider is dragged, recorded as one undo step
                readonly property bool live: !root.attribute.keyable && !root.attribute.isLink && !!root.attribute.node &&
                                             !!root.attribute.node.isImageOpNode
                onMoved: (formattedValue) => {
                    if (live)
                        _currentScene.setValueInteractively(root.attribute, formattedValue)
                }
                onPressedChanged: (pressed, formattedValue) => {
                    if (live) {
                        if (pressed) {
                            _currentScene.beginInteractiveEdit(root.attribute)
                        } else {
                            _currentScene.setValueInteractively(root.attribute, formattedValue)
                            _currentScene.endInteractiveEdit(root.attribute)
                        }
                        return
                    }
                    if (!pressed) {
                        if (root.attribute.keyable)
                            _currentScene.addAttributeKeyValue(root.attribute, _currentScene.selectedViewId, formattedValue)
                        else
                            _currentScene.setAttribute(root.attribute, formattedValue)
                    }
                }
            }
        }

        Component {
            id: tracksComponent
            AttributeControls.TracksField {
                attribute: root.attribute
                editable: root.editable
            }
        }

        Component {
            id: curvesComponent
            AttributeControls.CurveEditor {
                attribute: root.attribute
                editable: root.editable
            }
        }

        Component {
            id: rotoShapesComponent
            AttributeControls.RotoShapesField {
                attribute: root.attribute
                editable: root.editable
            }
        }

        Component {
            id: rgbaColorComponent
            AttributeControls.RgbaColorField {
                attribute: root.attribute
                editable: root.editable
            }
        }

        Component {
            id: vec2Component
            AttributeControls.Vec2Field {
                attribute: root.attribute
                editable: root.editable
            }
        }

        Component {
            id: checkboxComponent
            AttributeControls.CheckBoxRow {
                editable: root.editable
                checked: root.attribute.keyable
                                ? root.attribute.keyValues.getValueAtKeyOrDefault(_currentScene.selectedViewId)
                                : root.attribute.value
                onToggled: {
                    if(root.attribute.keyable)
                    {
                        const value = root.attribute.keyValues.getValueAtKeyOrDefault(_currentScene.selectedViewId)
                        _currentScene.addAttributeKeyValue(root.attribute, _currentScene.selectedViewId, !value)
                    }
                    else
                    {
                        _currentScene.setAttribute(root.attribute, !root.attribute.value)
                    }
                }
            }
        }

        Component {
            id: listAttributeComponent
            ColumnLayout {
                id: listAttributeLayout
                width: parent.width
                property bool expanded: false
                RowLayout {
                    spacing: 4
                    ToolButton {
                        text: listAttributeLayout.expanded  ? MaterialIcons.keyboard_arrow_down : MaterialIcons.keyboard_arrow_right
                        font.family: MaterialIcons.fontFamily
                        onClicked: listAttributeLayout.expanded = !listAttributeLayout.expanded
                    }
                    Label {
                        Layout.alignment: Qt.AlignVCenter
                        text: attribute.value.count + " elements"
                    }
                    ToolButton {
                        text: MaterialIcons.add_circle_outline
                        font.family: MaterialIcons.fontFamily
                        font.pointSize: 11
                        padding: 2
                        enabled: root.editable
                        onClicked: _currentScene.appendAttribute(attribute, undefined)
                    }
                }
                ListView {
                    id: lv
                    model: attribute.value
                    visible: listAttributeLayout.expanded && count > 0
                    implicitHeight: visible ? Math.min(contentHeight, 300) : 0
                    Layout.fillWidth: true
                    Layout.margins: 4
                    clip: true
                    spacing: 4

                    ScrollBar.vertical: MScrollBar { id: sb }

                    delegate: Loader {
                        active: !objectsHideable
                            || ((object.isDefault && GraphEditorSettings.showDefaultAttributes || !object.isDefault && GraphEditorSettings.showModifiedAttributes)
                            && (object.hasAnyInputLinks && GraphEditorSettings.showLinkAttributes || !object.hasAnyInputLinks && GraphEditorSettings.showNotLinkAttributes))
                        visible: active
                        sourceComponent: RowLayout {
                            id: item
                            property var childAttrib: object
                            layoutDirection: Qt.RightToLeft
                            width: lv.width - sb.width
                            Component.onCompleted: {
                                var cpt = Qt.createComponent("AttributeItemDelegate.qml")
                                var obj = cpt.createObject(item,
                                                        {
                                                            'attribute': Qt.binding(function() { return item.childAttrib }),
                                                            'readOnly': Qt.binding(function() { return !root.editable })
                                                        })
                                obj.Layout.fillWidth = true
                                obj.label.text = index
                                obj.label.horizontalAlignment = Text.AlignHCenter
                                obj.label.verticalAlignment = Text.AlignVCenter
                                obj.doubleClicked.connect(function(attr) { root.doubleClicked(attr) })
                                obj.inAttributeClicked.connect(function(srcItem, mouse, inAttributes) { root.inAttributeClicked(srcItem, mouse, inAttributes) })
                                obj.outAttributeClicked.connect(function(srcItem, mouse, outAttributes) { root.outAttributeClicked(srcItem, mouse, outAttributes) })
                            }
                            ToolButton {
                                enabled: root.editable
                                text: MaterialIcons.remove_circle_outline
                                font.family: MaterialIcons.fontFamily
                                font.pointSize: 11
                                padding: 2
                                ToolTip.text: "Remove Element"
                                ToolTip.visible: hovered
                                onClicked: _currentScene.removeAttribute(item.childAttrib)
                            }
                        }
                    }
                }
            }
        }

        Component {
            id: groupAttributeComponent
            ColumnLayout {
                id: groupItem
                spacing: 0
                property bool expanded: true

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 0

                    ToolButton {
                        text: groupItem.expanded ? MaterialIcons.expand_more : MaterialIcons.chevron_right
                        font.family: MaterialIcons.fontFamily
                        font.pointSize: 10
                        padding: 2
                        onClicked: groupItem.expanded = !groupItem.expanded
                    }

                    Label {
                        text: attribute.label
                        font.bold: true
                        font.pointSize: 8
                        Layout.fillWidth: true
                        elide: Text.ElideRight
                        padding: 3

                        ToolTip {
                            text: {
                                var tooltip = ""
                                if (attribute.desc)
                                    tooltip += "<b>" + attribute.desc.name + ":</b> " + attribute.type + "<br>" + Format.plainToHtml(attribute.desc.description)
                                return tooltip
                            }
                            visible: labelMA.containsMouse
                            delay: 800
                        }

                        MouseArea {
                            id: labelMA
                            anchors.fill: parent
                            acceptedButtons: Qt.LeftButton | Qt.RightButton
                            hoverEnabled: true
                            onClicked: function(mouse) {
                                if (mouse.button == Qt.RightButton && root.isAnySetChild) {
                                    var menu = removeAnySetAttributeMenuComp.createObject(labelMA)
                                    var position = labelMA.mapToItem(Overlay.overlay, mouse.x, mouse.y)
                                    menu.preferredX = position.x
                                    menu.preferredY = position.y
                                    menu.parent = labelMA
                                    menu.popup()
                                    return
                                }
                                groupItem.expanded = !groupItem.expanded
                            }
                            onDoubleClicked: function(mouse) { root.doubleClicked(mouse, root.attribute) }
                        }
                    }

                    // Add a child by hand to an AnySet (e.g. an input of a SubGraph node)
                    MaterialToolButton {
                        id: addAnySetButton
                        visible: attribute.type === "AnySet" && !root.readOnly
                        text: MaterialIcons.add
                        font.pointSize: 9
                        padding: 2
                        ToolTip.text: "Add an attribute"
                        ToolTip.visible: hovered
                        onClicked: {
                            var dialog = addAnySetAttributeDialogComp.createObject(Overlay.overlay, { "targetAnySet": attribute })
                            dialog.open()
                        }
                    }
                }

                Component.onCompleted: {
                    var cpt = Qt.createComponent("AttributeEditor.qml");
                    var obj = cpt.createObject(groupItem,
                                               {
                                                   'model': Qt.binding(function() { return attribute.value }),
                                                   'readOnly': Qt.binding(function() { return root.readOnly }),
                                                   'labelWidth': Qt.binding(function() { return root.labelWidth }),
                                                   'objectsHideable': Qt.binding(function() { return root.objectsHideable }),
                                                   'filterText': Qt.binding(function() { return root.filterText }),
                                                   'visible': Qt.binding(function() { return groupItem.expanded }),
                                               })
                    obj.Layout.fillWidth = true;
                    obj.Layout.leftMargin = 8;
                    obj.attributeDoubleClicked.connect(
                        function(mouse, attr) {
                            root.doubleClicked(mouse, attr)
                        }
                    )
                    obj.inAttributeClicked.connect(
                        function(srcItem, mouse, inAttributes) {
                            root.inAttributeClicked(srcItem, mouse, inAttributes)
                        }
                    )
                    obj.outAttributeClicked.connect(
                        function(srcItem, mouse, outAttributes) {
                            root.outAttributeClicked(srcItem, mouse, outAttributes)
                        }
                    )
                    obj.showInViewer.connect(
                        function(attr) {
                            root.showInViewer(attr)
                        }
                    )
                }
            }
        }

        Component {
            id: colorHueComponent
            AttributeControls.ColorHue {
                value: root.attribute.value
                editable: root.editable
                onEditingFinished: (text) => setTextFieldAttribute(text)
                onAccepted: (text) => setTextFieldAttribute(text)
                onDestruction: (activeFocus, text) => {
                    if (activeFocus)
                        setTextFieldAttribute(text)
                }
                onPressedChanged: (pressed, formattedValue) => {
                    if (!pressed)
                        _currentScene.setAttribute(root.attribute, formattedValue)
                }
            }
        }
    }

    MaterialLabel {
        visible: !attribute.isOutput && root.errorMessages.length
        text: MaterialIcons.fmd_bad
        ToolTip.text: root.errorMessages.join("\n")
        color: "orange"
    }

    // Add or remove key button for keyable attribute
    Loader {
        active: attribute.keyable
        sourceComponent: MaterialToolButton {
            font.pointSize: 5
            padding: 6
            text: MaterialIcons.circle
            checkable: true
            checked: attribute.keyable && attribute.keyValues.hasKey(_currentScene.selectedViewId)
            enabled: root.editable
            onClicked: {
                if (attribute.keyValues.hasKey(_currentScene.selectedViewId))
                    _currentScene.removeAttributeKey(attribute, _currentScene.selectedViewId)
                else
                    _currentScene.addAttributeKeyDefaultValue(attribute, _currentScene.selectedViewId)
            }
        }
    }
}
