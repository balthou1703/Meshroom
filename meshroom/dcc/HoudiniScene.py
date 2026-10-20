__version__ = "1.0"

from .sceneNodes import SceneNode, sceneNodeInputs, sceneNodeOutputs


class HoudiniScene(SceneNode):
    __doc__ = SceneNode.__doc__
    category = "Houdini"
    context = "houdini"

    inputs = sceneNodeInputs("Houdini")
    outputs = sceneNodeOutputs()
