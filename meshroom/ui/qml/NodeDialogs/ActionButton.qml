import QtQuick
import QtQuick.Controls

/**
 * ActionButton is a validation button of a node dialog, optionally filled with a custom color.
 */

Button {
    id: control

    /// Background color ("" for the colors of the palette)
    property string fillColor: ""
    property string toolTipText: ""

    readonly property color baseColor: fillColor.length > 0 ? fillColor : palette.button

    font.bold: fillColor.length > 0
    leftPadding: 16
    rightPadding: 16
    topPadding: 6
    bottomPadding: 6
    ToolTip.text: toolTipText
    ToolTip.visible: toolTipText.length > 0 && hovered
    ToolTip.delay: 500

    contentItem: Label {
        text: control.text
        font: control.font
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        color: control.fillColor.length > 0 ? "white" : control.palette.buttonText
        opacity: control.enabled ? 1.0 : 0.5
    }

    background: Rectangle {
        implicitWidth: 80
        implicitHeight: 28
        radius: 4
        color: {
            if (!control.enabled)
                return Qt.darker(control.baseColor, 1.6)
            if (control.down)
                return Qt.darker(control.baseColor, 1.25)
            return control.hovered ? Qt.lighter(control.baseColor, 1.2) : control.baseColor
        }
        border.width: control.visualFocus ? 2 : 0
        border.color: control.palette.highlight
    }
}
