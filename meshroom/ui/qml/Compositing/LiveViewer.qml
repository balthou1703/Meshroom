import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Controls 1.0
import MaterialIcons 2.2
import Utils 1.0

/**
 * LiveViewer displays the result of the evaluated live compositing node (LiveController).
 *
 * Toolbar: layer (AOV), channels, exposure, gamma, display transform, auto-refresh, refresh, fit, cache memory.
 * Mouse: wheel to zoom, drag to pan, double-click or "F" to fit. The footer shows the float
 * values of the displayed layer under the cursor.
 *
 * The canvas is the image format; the image covers its bbox, which can be larger (overscan,
 * dashed outline). When zoomed out, the image is converted for display at a reduced resolution.
 */
Panel {
    id: root

    property var live: null
    // Show the image inside the format only (the bbox stays outlined)
    property bool hideOverscan: true
    readonly property var evaluatedNode: live ? live.evaluatedNode : null
    readonly property bool hasImage: !!live && live.imageRevision > 0 && live.imageWidth > 0

    // With a compare node (Evaluate Compare): the view shown, 1 or 2 (keys 1 / 2)
    title: {
        if (!evaluatedNode)
            return "Viewer"
        var proxyText = live.proxy < 1 ? "  [proxy 1/" + Math.round(1 / live.proxy) + "]" : ""
        if (live.compareNode)
            return "Viewer " + live.activeView + " / 2 - " + evaluatedNode.label + proxyText
        return "Viewer - " + evaluatedNode.label + proxyText
    }
    loading: !!live && live.rendering
    focus: true

    Keys.onPressed: function(event) {
        if (event.key === Qt.Key_F) {
            canvas.fit()
            event.accepted = true
        }
    }

    // Channel shortcuts while the mouse is over the viewer, as in Nuke:
    // R, G, B, A (and Y for the luminance) show one channel, the same key again goes back to RGB
    HoverHandler {
        id: viewerHover
    }
    function toggleChannel(mode) {
        if (root.live)
            root.live.channelMode = root.live.channelMode === mode ? "rgb" : mode
    }
    Shortcut { sequence: "R"; enabled: viewerHover.hovered; onActivated: root.toggleChannel("r") }
    Shortcut { sequence: "G"; enabled: viewerHover.hovered; onActivated: root.toggleChannel("g") }
    Shortcut { sequence: "B"; enabled: viewerHover.hovered; onActivated: root.toggleChannel("b") }
    Shortcut { sequence: "A"; enabled: viewerHover.hovered; onActivated: root.toggleChannel("a") }
    Shortcut { sequence: "Y"; enabled: viewerHover.hovered; onActivated: root.toggleChannel("luminance") }
    Shortcut { sequence: "F"; enabled: viewerHover.hovered; onActivated: canvas.fit() }

    headerBar: RowLayout {
        spacing: 2

        ComboBox {
            id: layerCombo
            Layout.preferredWidth: 140
            model: root.live ? root.live.layers : []
            enabled: count > 1
            currentIndex: root.live ? Math.max(0, model.indexOf(root.live.currentLayer)) : 0
            onActivated: function(index) { root.live.currentLayer = model[index] }
            ToolTip.text: "Layer (AOV)"
            ToolTip.visible: hovered
        }

        Repeater {
            model: root.live ? root.live.channelModes : []
            delegate: ToolButton {
                text: modelData === "luminance" ? "Y" : modelData.toUpperCase()
                checkable: true
                checked: root.live && root.live.channelMode === modelData
                padding: 4
                font.pointSize: 9
                onClicked: root.live.channelMode = modelData
                ToolTip.text: modelData === "luminance" ? "Luminance" : "Channels: " + modelData.toUpperCase()
                ToolTip.visible: hovered
            }
        }

        // Region of interest: when zoomed in, only the visible part is computed
        ToolButton {
            text: "ROI"
            checkable: true
            checked: root.live ? root.live.roiEnabled : true
            padding: 4
            font.pointSize: 9
            onClicked: root.live.roiEnabled = checked
            ToolTip.text: checked ? "Region of interest: when zoomed in, only the visible part of the image is computed"
                                  : "The whole image is computed"
            ToolTip.visible: hovered
        }

        // Proxy mode: the viewer renders at a reduced resolution (faster, cached apart), shown at the size
        // of the full one; the Writes always render at full resolution
        MaterialToolButton {
            id: proxyButton
            text: "Proxy"
            font.pointSize: 8
            checkable: true
            checked: !!root.live && root.live.proxy < 1
            onClicked: root.live.proxy = checked ? 1.0 / proxyFactorCombo.factor : 1.0
            ToolTip.text: checked ? "Proxy mode on (1/" + proxyFactorCombo.factor + "): click to render at full resolution"
                                  : "Proxy mode: render the viewer images reduced (the Reads are downscaled, the values in "
                                    + "pixels follow; the cache keeps them apart; the Writes stay at full resolution)"
        }
        ComboBox {
            id: proxyFactorCombo
            Layout.preferredWidth: 58
            readonly property var factors: root.live ? root.live.proxyFactors : [2, 3, 4, 8]
            // The reduction used when the proxy mode is turned on
            property int factor: 2
            model: factors.map(function(f) { return "1/" + f })
            currentIndex: Math.max(0, factors.indexOf(factor))
            onActivated: function(index) {
                factor = factors[index]
                root.live.proxy = 1.0 / factor
            }
            Connections {
                target: root.live
                function onProxyChanged() {
                    if (root.live.proxy < 1)
                        proxyFactorCombo.factor = Math.round(1 / root.live.proxy)
                }
            }
            ToolTip.text: "Proxy reduction (choosing one turns the proxy mode on)"
            ToolTip.visible: hovered
        }

        // Click the name of a display value to reset it
        Label {
            text: "Exp"
            leftPadding: 6
            color: root.live && root.live.exposure !== 0 ? "#F0B04E" : palette.text
            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: root.live.exposure = 0
            }
            ToolTip.text: "Exposure: click to reset (0)"
            ToolTip.visible: expHover.hovered
            HoverHandler { id: expHover }
        }
        Slider {
            id: exposureSlider
            Layout.preferredWidth: 90
            from: -6; to: 6; stepSize: 0.1
            value: root.live ? root.live.exposure : 0
            onMoved: root.live.exposure = value
            ToolTip.text: "Exposure (f-stops)"
            ToolTip.visible: hovered
        }
        Label { text: root.live ? root.live.exposure.toFixed(1) : ""; Layout.preferredWidth: 28 }

        Label {
            text: "Gamma"
            leftPadding: 4
            color: root.live && root.live.gamma !== 1 ? "#F0B04E" : palette.text
            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: root.live.gamma = 1
            }
            ToolTip.text: "Gamma: click to reset (1)"
            ToolTip.visible: gammaHover.hovered
            HoverHandler { id: gammaHover }
        }
        Slider {
            Layout.preferredWidth: 70
            from: 0.2; to: 4; stepSize: 0.05
            value: root.live ? root.live.gamma : 1
            onMoved: root.live.gamma = value
        }
        Label { text: root.live ? root.live.gamma.toFixed(2) : ""; Layout.preferredWidth: 30 }

        // Display transform (OCIO display / view, or the builtin ones): from the working space to the screen
        ComboBox {
            id: displayTransformCombo
            Layout.preferredWidth: 130
            model: root.live ? root.live.displayTransforms : []
            currentIndex: root.live ? model.indexOf(root.live.displayTransform) : -1
            onActivated: function(index) { root.live.displayTransform = model[index] }
            ToolTip.text: "Display transform of the viewer (display / view)\n" +
                          (root.live ? root.live.colorConfigDescription : "") + "\nRight-click: reload the color config"
            ToolTip.visible: hovered
            MouseArea {
                anchors.fill: parent
                acceptedButtons: Qt.RightButton
                onClicked: root.live.reloadColorConfig()
            }
        }

        MaterialToolButton {
            text: root.live && root.live.autoRefresh ? MaterialIcons.lock_open : MaterialIcons.lock
            checkable: true
            checked: root.live ? !root.live.autoRefresh : false
            onClicked: root.live.autoRefresh = !checked
            ToolTip.text: checked ? "Image frozen: click to update it on each change" : "Freeze the image (stop updating it on each change)"
        }
        MaterialToolButton {
            text: MaterialIcons.refresh
            enabled: !!root.evaluatedNode
            onClicked: root.live.refresh()
            ToolTip.text: "Re-evaluate (re-reads the files)"
        }
        MaterialToolButton {
            text: MaterialIcons.crop_free
            checked: root.hideOverscan
            onClicked: root.hideOverscan = !root.hideOverscan
            ToolTip.text: root.hideOverscan ? "Overscan hidden: click to show the image outside of the format" :
                                              "Hide the overscan (the image outside of the format, the bbox stays outlined)"
        }
        MaterialToolButton {
            text: MaterialIcons.fit_screen
            enabled: root.hasImage
            onClicked: canvas.fit()
            ToolTip.text: "Fit (F)"
        }

        RamGauge {
            Layout.leftMargin: 10
            Layout.rightMargin: 4
            Layout.alignment: Qt.AlignVCenter
            live: root.live
        }
    }

    // Image area
    Rectangle {
        anchors.fill: parent
        color: Qt.darker(palette.window, 1.6)
        clip: true

        // Placement of the image format in the viewer: canvas pixel (u, v) is at screen
        // (canvas.x + u * zoom, canvas.y + v * zoom). The canvas itself draws nothing: the image and
        // the outlines are drawn in screen coordinates (no scaled items, whatever the zoom).
        Item {
            id: canvas
            property real zoom: 1.0
            width: root.live ? root.live.imageWidth : 0
            height: root.live ? root.live.imageHeight : 0

            function toCanvas(px, py) {
                return Qt.point((px - x) / zoom, (py - y) / zoom)
            }

            function fit() {
                if (!root.hasImage)
                    return
                zoom = Math.min(parent.width / width, parent.height / height) * 0.95
                x = (parent.width - width * zoom) / 2
                y = (parent.height - height * zoom) / 2
            }

            // Fit when the image format changes
            onWidthChanged: Qt.callLater(fit)
            onHeightChanged: Qt.callLater(fit)

            // Only the visible part of the image is converted for display, at one pixel out of 2^n
            // when zoomed out: the displayed texture stays small whatever the zoom
            onZoomChanged: viewTimer.restart()
            onXChanged: viewTimer.restart()
            onYChanged: viewTimer.restart()
            Timer {
                id: viewTimer
                interval: 60
                onTriggered: canvas.updateView()
            }
            function updateView() {
                if (!root.live || !parent)
                    return
                var stride = 1
                while (stride * 2 <= 1 / canvas.zoom && stride < 16)
                    stride *= 2
                var left = -canvas.x / canvas.zoom
                var top = -canvas.y / canvas.zoom
                root.live.setDisplayView(left, top, left + parent.width / canvas.zoom, top + parent.height / canvas.zoom, stride)
            }
            Connections {
                target: canvas.parent
                function onWidthChanged() { viewTimer.restart() }
                function onHeightChanged() { viewTimer.restart() }
            }
        }

        // The image, in screen coordinates; clipped to the format when the overscan is hidden
        Item {
            id: imageClip
            x: root.hideOverscan ? canvas.x : 0
            y: root.hideOverscan ? canvas.y : 0
            width: root.hideOverscan ? canvas.width * canvas.zoom : parent.width
            height: root.hideOverscan ? canvas.height * canvas.zoom : parent.height
            clip: true

            Image {
                id: image
                x: canvas.x + (root.live ? root.live.displayX : 0) * canvas.zoom - imageClip.x
                y: canvas.y + (root.live ? root.live.displayY : 0) * canvas.zoom - imageClip.y
                width: (root.live ? root.live.displayWidth : 0) * canvas.zoom
                height: (root.live ? root.live.displayHeight : 0) * canvas.zoom
                cache: false
                asynchronous: false
                smooth: canvas.zoom < 2
                source: root.hasImage ? "image://live/" + root.live.imageRevision : ""
            }
        }

        // Format border and bbox (data window) outline, in screen coordinates
        Canvas {
            id: outlines
            anchors.fill: parent
            visible: root.hasImage
            readonly property var geometry: [canvas.x, canvas.y, canvas.zoom, canvas.width, canvas.height,
                                             root.live ? root.live.bboxX : 0, root.live ? root.live.bboxY : 0,
                                             root.live ? root.live.bboxWidth : 0, root.live ? root.live.bboxHeight : 0]
            onGeometryChanged: requestPaint()
            onWidthChanged: requestPaint()
            onHeightChanged: requestPaint()
            onPaint: {
                var ctx = getContext("2d")
                ctx.reset()
                if (!root.live)
                    return
                var z = canvas.zoom
                ctx.lineWidth = 1
                ctx.strokeStyle = Qt.rgba(1, 1, 1, 0.25)
                ctx.strokeRect(Math.round(canvas.x) + 0.5, Math.round(canvas.y) + 0.5,
                               Math.round(canvas.width * z), Math.round(canvas.height * z))
                var live = root.live
                if (live.bboxX !== 0 || live.bboxY !== 0 || live.bboxWidth !== canvas.width || live.bboxHeight !== canvas.height) {
                    ctx.strokeStyle = Qt.rgba(1, 1, 1, 0.4)
                    ctx.setLineDash([4, 4])
                    ctx.strokeRect(Math.round(canvas.x + live.bboxX * z) + 0.5, Math.round(canvas.y + live.bboxY * z) + 0.5,
                                   Math.round(live.bboxWidth * z), Math.round(live.bboxHeight * z))
                }
            }
        }

        MouseArea {
            id: viewerMouse
            anchors.fill: parent
            hoverEnabled: true
            acceptedButtons: Qt.LeftButton | Qt.MiddleButton
            property point lastPosition
            property var pixel: null

            function updatePixel(mouse) {
                if (!root.hasImage) {
                    pixel = null
                    return
                }
                var p = canvas.toCanvas(mouse.x, mouse.y)
                // Nuke coordinates: from the bottom-left corner of the format, y up (overscan included)
                var px = Math.floor(p.x)
                var py = root.live.imageHeight - 1 - Math.floor(p.y)
                var values = root.live.pixelValues(px, py)
                pixel = values.length > 0 ? { "x": px, "y": py, "values": values } : null
            }

            onPressed: function(mouse) {
                lastPosition = Qt.point(mouse.x, mouse.y)
                root.forceActiveFocus()
            }
            onPositionChanged: function(mouse) {
                if (pressed) {
                    canvas.x += mouse.x - lastPosition.x
                    canvas.y += mouse.y - lastPosition.y
                    lastPosition = Qt.point(mouse.x, mouse.y)
                }
                updatePixel(mouse)
            }
            onExited: pixel = null
            onDoubleClicked: canvas.fit()
            onWheel: function(wheel) {
                var factor = wheel.angleDelta.y > 0 ? 1.15 : 1 / 1.15
                // Zoom around the cursor
                var before = canvas.toCanvas(wheel.x, wheel.y)
                canvas.zoom = Math.max(0.02, Math.min(64, canvas.zoom * factor))
                canvas.x = wheel.x - before.x * canvas.zoom
                canvas.y = wheel.y - before.y * canvas.zoom
            }
        }

        // Handles of the selected Transform node
        TransformGizmo {
            anchors.fill: parent
            readonly property var selected: _currentScene ? _currentScene.selectedNode : null
            node: selected && selected.nodeType === "Transform" && root.hasImage ? selected : null
            scene: _currentScene
            canvasX: canvas.x
            canvasY: canvas.y
            zoom: canvas.zoom
            formatWidth: root.live ? root.live.imageWidth : 0
            formatHeight: root.live ? root.live.imageHeight : 0
        }

        // Box of the selected Crop node
        CropGizmo {
            anchors.fill: parent
            readonly property var selected: _currentScene ? _currentScene.selectedNode : null
            node: selected && selected.nodeType === "Crop" && root.hasImage ? selected : null
            scene: _currentScene
            live: root.live
            canvasX: canvas.x
            canvasY: canvas.y
            zoom: canvas.zoom
            formatWidth: root.live ? root.live.imageWidth : 0
            formatHeight: root.live ? root.live.imageHeight : 0
        }

        // Handles of the selected Ramp node
        RampGizmo {
            anchors.fill: parent
            readonly property var selected: _currentScene ? _currentScene.selectedNode : null
            node: selected && selected.nodeType === "Ramp" && root.hasImage ? selected : null
            scene: _currentScene
            canvasX: canvas.x
            canvasY: canvas.y
            zoom: canvas.zoom
            formatHeight: root.live ? root.live.imageHeight : 0
        }

        // Shapes of the selected Roto node
        RotoGizmo {
            anchors.fill: parent
            readonly property var selected: _currentScene ? _currentScene.selectedNode : null
            node: selected && selected.nodeType === "Roto" && root.hasImage ? selected : null
            formatWidth: root.live ? root.live.imageWidth : 0
            scene: _currentScene
            canvasX: canvas.x
            canvasY: canvas.y
            zoom: canvas.zoom
            formatHeight: root.live ? root.live.imageHeight : 0
            frame: root.live ? root.live.currentFrame : 1
        }

        // Tracks of the selected Tracker node
        TrackerGizmo {
            anchors.fill: parent
            readonly property var selected: _currentScene ? _currentScene.selectedNode : null
            node: selected && ["Tracker", "CameraTracker", "Resection", "FaceTracker", "FaceBuilder"].indexOf(selected.nodeType) >= 0 && root.hasImage ? selected : null
            // The tracks of the autotrack are shown, not edited
            readOnly: !!node && node.nodeType === "CameraTracker"
            formatWidth: root.live ? root.live.imageWidth : 0
            scene: _currentScene
            live: root.live
            canvasX: canvas.x
            canvasY: canvas.y
            zoom: canvas.zoom
            formatHeight: root.live ? root.live.imageHeight : 0
            frame: root.live ? root.live.currentFrame : 1
        }

        // Color picking for a color attribute (eyedropper of the attribute editor):
        // click a pixel, or drag a rectangle to average it; one undo step per pick
        MouseArea {
            id: pickArea
            readonly property var attribute: root.live ? root.live.pickAttribute : null
            anchors.fill: parent
            enabled: !!attribute && root.hasImage
            visible: enabled
            cursorShape: Qt.CrossCursor
            acceptedButtons: Qt.LeftButton
            property point start
            property point current

            function toNuke(point) {
                var p = canvas.toCanvas(point.x, point.y)
                return Qt.point(p.x, root.live.imageHeight - p.y)
            }
            function pick() {
                var a = toNuke(start), b = toNuke(current)
                // A target color is taken in the displayed image (e.g. a reference plate), the others in
                // the image entering the node
                var node = attribute.name === "targetColor" ? null : attribute.node
                var color = root.live.sampleColor(node, a.x, a.y, b.x, b.y)
                if (color.length === 4)
                    _currentScene.setRgbaInteractively(attribute, color[0], color[1], color[2], color[3])
            }
            onPressed: function(mouse) {
                start = current = Qt.point(mouse.x, mouse.y)
                _currentScene.beginInteractiveEdit(attribute)
                pick()
            }
            onPositionChanged: function(mouse) {
                if (!pressed)
                    return
                current = Qt.point(mouse.x, mouse.y)
                pick()
            }
            onReleased: _currentScene.endInteractiveEdit(attribute)
            onCanceled: _currentScene.endInteractiveEdit(attribute)
            onWheel: function(wheel) { wheel.accepted = false }

            // Sampled rectangle
            Rectangle {
                visible: pickArea.pressed
                x: Math.min(pickArea.start.x, pickArea.current.x)
                y: Math.min(pickArea.start.y, pickArea.current.y)
                width: Math.abs(pickArea.current.x - pickArea.start.x)
                height: Math.abs(pickArea.current.y - pickArea.start.y)
                color: "transparent"
                border.color: "#FF5555"
                border.width: 1
            }
            Label {
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 6
                padding: 4
                text: pickArea.attribute ? "Picking " + pickArea.attribute.node.label + " > " + pickArea.attribute.label + "   (Esc to stop)" : ""
                background: Rectangle { color: Qt.rgba(0, 0, 0, 0.6); radius: 3 }
            }
        }
        Shortcut {
            sequence: "Esc"
            enabled: pickArea.enabled
            onActivated: root.live.pickAttribute = null
        }

        // Placeholder
        Label {
            anchors.centerIn: parent
            visible: !root.evaluatedNode
            horizontalAlignment: Text.AlignHCenter
            color: Colors.lightgrey
            text: "Right-click a compositing node > Evaluate\n(or select it and press E)"
        }

        // Node action running in the background (autotrack, solve...)
        Rectangle {
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.margins: 6
            // Below the error banner
            anchors.topMargin: 40
            width: actionRow.implicitWidth + 12
            height: actionRow.implicitHeight + 8
            radius: 4
            visible: !!root.live && root.live.actionProgress >= 0
            color: Qt.rgba(0, 0, 0, 0.65)
            RowLayout {
                id: actionRow
                anchors.centerIn: parent
                Label { text: root.live ? root.live.actionText : "" }
                ProgressBar {
                    Layout.preferredWidth: 140
                    from: 0
                    to: 1
                    value: root.live ? Math.max(0, root.live.actionProgress) : 0
                }
                MaterialToolButton {
                    text: MaterialIcons.stop_circle
                    onClicked: root.live.stopAction()
                    ToolTip.text: "Stop"
                }
            }
        }

        // Error banner: the previous image stays displayed
        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            height: errorLabel.implicitHeight + 8
            visible: !!root.live && root.live.errorMessage !== ""
            color: Qt.rgba(0.6, 0.1, 0.1, 0.85)
            Label {
                id: errorLabel
                anchors.fill: parent
                anchors.margins: 4
                wrapMode: Text.Wrap
                text: root.live ? (root.live.errorNode ? root.live.errorNode + ": " : "") + root.live.errorMessage : ""
            }
        }
    }

    footerContent: RowLayout {
        Layout.fillWidth: true
        Label {
            Layout.fillWidth: true
            elide: Text.ElideRight
            font.family: "monospace"
            text: {
                var p = viewerMouse.pixel
                if (!p)
                    return ""
                var channels = root.live.currentChannels()
                var parts = []
                for (var i = 0; i < p.values.length; i++)
                    parts.push((channels[i] || i) + " " + p.values[i].toFixed(4))
                return "x " + p.x + "  y " + p.y + "   " + parts.join("  ")
            }
        }
        Label {
            visible: root.hasImage
            color: Colors.lightgrey
            text: {
                if (!root.hasImage)
                    return ""
                var live = root.live
                var text = live.imageWidth + "x" + live.imageHeight
                if (live.bboxX !== 0 || live.bboxY !== 0 || live.bboxWidth !== live.imageWidth || live.bboxHeight !== live.imageHeight) {
                    var y0 = live.imageHeight - live.bboxY - live.bboxHeight
                    text += "   bbox " + live.bboxX + " " + y0 + " " + (live.bboxX + live.bboxWidth) + " " + (y0 + live.bboxHeight)
                }
                return text + "   " + (live.renderTime * 1000).toFixed(0) + " ms"
            }
        }
    }
}
