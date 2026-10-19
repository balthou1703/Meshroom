import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MaterialIcons 2.2

/**
 * Editor of a 2D attribute (Vec2d: {x, y}).
 *
 * A "linkable" attribute (semantic "linkedVec2") has a "split" checkbox: unchecked, a single slider
 * sets x and y; checked, x and y have their own fields. The checkbox is saved with the node (its
 * "splitVec2" internal attribute); unchecking it sets y = x. Values that differ (e.g. loaded from a
 * file) are shown split. Other 2D attributes always show x and y.
 */
RowLayout {
    id: root

    required property var attribute
    property bool editable: true
    readonly property bool linkable: attribute.desc.semantic === "linkedVec2"
    readonly property var xAttribute: attribute.value ? attribute.value.get("x") : null
    readonly property var yAttribute: attribute.value ? attribute.value.get("y") : null
    readonly property real xValue: xAttribute ? xAttribute.value : 0
    readonly property real yValue: yAttribute ? yAttribute.value : 0
    readonly property var range: xAttribute && xAttribute.desc.range ? xAttribute.desc.range : []
    readonly property var splitStore: attribute.node && attribute.node.hasInternalAttribute("splitVec2") ?
                                      attribute.node.internalAttribute("splitVec2") : null
    // The checkbox state: stored with compositing nodes, remembered by the scene for the others
    property bool sessionSplit: _currentScene ? _currentScene.isVec2Split(attribute) : false
    readonly property bool splitChecked: splitStore ? splitStore.value.split(",").indexOf(attribute.name) >= 0 : sessionSplit
    readonly property bool valuesDiffer: Math.abs(xValue - yValue) > 1e-9
    // Shown split: when checked, and always when x and y differ (they are never lost).
    // Frozen while the values are being changed: x and y are set one after the other, and a transient
    // difference must not hide the single slider being dragged (it would lose the mouse).
    property bool editing: false
    property bool frozenSplit: false
    readonly property bool split: editing ? frozenSplit : (!linkable || splitChecked || valuesDiffer)

    function freezeSplit(frozen) {
        if (frozen)
            frozenSplit = split
        editing = frozen
    }

    spacing: 2

    function setValues(x, y) {
        x = Number(x)
        y = Number(y)
        if (!isFinite(x) || !isFinite(y) || (Math.abs(x - xValue) < 1e-12 && Math.abs(y - yValue) < 1e-12))
            return
        freezeSplit(true)
        _currentScene.setVec2Attribute(root.attribute, x, y)
        freezeSplit(false)
    }

    function setSplit(split) {
        if (!split && valuesDiffer)
            setValues(xValue, xValue)
        _currentScene.setVec2Split(root.attribute, split)
        if (!splitStore)
            sessionSplit = split
    }

    // Values that differ when the field is created (e.g. a loaded graph): the checkbox follows
    Component.onCompleted: {
        if (linkable && valuesDiffer && !splitChecked)
            _currentScene.setVec2Split(root.attribute, true)
    }

    // Linked: a single slider for both values
    SliderField {
        id: linkedField
        visible: !root.split
        Layout.fillWidth: true
        checked: root.xValue
        type: "FloatParam"
        length: root.range.length || 0
        start: root.range[0] || 0
        end: root.range[1] || 0
        step: root.range[2] || 0
        editable: root.editable
        // A hidden field never writes (e.g. when it loses the focus after a split)
        onEditingFinished: (hasExprError, evaluatedValue, text, displayValue) => {
            if (!hasExprError && linkedField.visible)
                root.setValues(evaluatedValue, evaluatedValue)
        }
        onAccepted: (hasExprError, evaluatedValue, text, displayValue) => {
            if (!hasExprError && linkedField.visible)
                root.setValues(evaluatedValue, evaluatedValue)
        }
        // Live update while dragging, recorded as one undo step
        onMoved: (formattedValue) => _currentScene.setVec2Interactively(root.attribute, Number(formattedValue), Number(formattedValue))
        onPressedChanged: (pressed, formattedValue) => {
            if (pressed) {
                root.freezeSplit(true)
                _currentScene.beginInteractiveEdit(root.attribute)
            } else {
                _currentScene.setVec2Interactively(root.attribute, Number(formattedValue), Number(formattedValue))
                _currentScene.endInteractiveEdit(root.attribute)
                root.freezeSplit(false)
            }
        }
    }

    // Split: x and y fields
    Label { visible: root.split; text: "x"; opacity: 0.6 }
    SliderField {
        id: xField
        visible: root.split
        Layout.fillWidth: true
        checked: root.xValue
        type: "FloatParam"
        length: 0
        start: 0
        end: 0
        step: 0
        editable: root.editable
        onEditingFinished: (hasExprError, evaluatedValue, text, displayValue) => {
            if (!hasExprError && xField.visible)
                root.setValues(evaluatedValue, root.yValue)
        }
        onAccepted: (hasExprError, evaluatedValue, text, displayValue) => {
            if (!hasExprError && xField.visible)
                root.setValues(evaluatedValue, root.yValue)
        }
    }
    Label { visible: root.split; text: "y"; opacity: 0.6 }
    SliderField {
        id: yField
        visible: root.split
        Layout.fillWidth: true
        checked: root.yValue
        type: "FloatParam"
        length: 0
        start: 0
        end: 0
        step: 0
        editable: root.editable
        onEditingFinished: (hasExprError, evaluatedValue, text, displayValue) => {
            if (!hasExprError && yField.visible)
                root.setValues(root.xValue, evaluatedValue)
        }
        onAccepted: (hasExprError, evaluatedValue, text, displayValue) => {
            if (!hasExprError && yField.visible)
                root.setValues(root.xValue, evaluatedValue)
        }
    }

    CheckBox {
        visible: root.linkable
        enabled: root.editable
        text: "split"
        font.pointSize: 8
        padding: 0
        checked: root.split
        ToolTip.text: checked ? "Uncheck: a single value (y = x)" : "Check: edit x and y separately"
        ToolTip.visible: hovered
        onClicked: {
            root.setSplit(!root.split)
            // The click toggled the box: it follows the stored state again
            checked = Qt.binding(function() { return root.split })
        }
    }
}
