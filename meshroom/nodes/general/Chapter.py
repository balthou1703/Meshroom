__version__ = "1.0"

from meshroom.core import desc


class Chapter(desc.BackdropNode):
    """
    Chapter line: an infinite horizontal line, named and colored, splitting the graph into sections
    (e.g. to show the steps of a compositing script to a newcomer). A chapter followed by another one
    can be folded with its arrow: the nodes between the two lines are hidden and the rest of the
    graph moves up.
    """

    category = "Utils"

    internalInputs = desc.BackdropNode.internalInputs + [
        desc.BoolParam(
            name="folded",
            label="Folded",
            description="Hide the nodes between this line and the next one.",
            value=False,
            invalidate=False,
        ),
    ]
