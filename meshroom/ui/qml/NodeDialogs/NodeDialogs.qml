import QtQuick

/**
 * NodeDialogs opens a window for each dialog requested by a node descriptor (a desc.UiRequest,
 * received through `_nodeDialogs`, see meshroom/ui/components/nodeDialogs.py).
 *
 * The windows are created as children of this item, which makes them transient for the main window.
 */

Item {
    id: root

    readonly property var windowComponents: ({
        "treePicker": treePickerComponent,
        "hierarchyEditor": hierarchyEditorComponent
    })

    Component { id: treePickerComponent; TreePickerWindow {} }
    Component { id: hierarchyEditorComponent; HierarchyEditorWindow {} }

    Connections {
        target: _nodeDialogs
        function onDialogRequested(backend) {
            const component = root.windowComponents[backend.kind]
            if (!component) {
                console.warn("NodeDialogs: unsupported dialog kind '" + backend.kind + "'")
                _nodeDialogs.release(backend)
                return
            }
            const window = component.createObject(root, { "backend": backend, "transientParent": root.Window.window })
            if (!window) {
                console.warn("NodeDialogs: could not create the '" + backend.kind + "' window: " + component.errorString())
                _nodeDialogs.release(backend)
                return
            }
            window.finished.connect(function() {
                // Destroy the window once its closing handlers are done
                Qt.callLater(function() {
                    window.destroy()
                    _nodeDialogs.release(backend)
                })
            })
            window.show()
            window.raise()
            window.requestActivate()
        }
    }
}
