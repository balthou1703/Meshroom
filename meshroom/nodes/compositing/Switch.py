__version__ = "0.1"

from meshroom.core import desc
from meshroom.live.params import imageOutput

_INPUTS = ("input", "input1", "input2", "input3")


class Switch(desc.ImageOpNode):
    category = "Comp Merge"
    color = "#9A9A9A"
    # The inputs that are not chosen can fail (e.g. a Read without file): their info is then ignored
    tolerantInputs = True
    documentation = """
Outputs one of its inputs, chosen by "Which" (0 to 3), which can be animated to switch over time.
An input that is not connected gives a transparent black image.

With "Use Expression", the input is chosen by a Python expression instead, e.g. to use a background
only when its file is set:

    1 if node('BG').file else 0

- frame: the rendered frame; node(name): the parameters of a node, by its label or name
  (node('BG').file, node('BG')['file']); nuke.toNode(name)['knob'].value() also works (scripts
  imported from Nuke); math, abs, min, max, int, float, round, len, bool, str;
- on several lines, the expression sets a variable named "result".
The parameters read by the expression are part of the cache: the image changes when they change.
"""
    inputs = [
        desc.Image(name=name, label=str(index), optional=True,
                   description=f"Input {index}: output when 'Which' is {index}.")
        for index, name in enumerate(_INPUTS)
    ] + [
        desc.IntParam(name="which", label="Which", value=0, range=(0, len(_INPUTS) - 1, 1),
                      description="The input to output (0 to 3); it can be animated.",
                      enabled=lambda node: not node.useExpression.value),
        desc.BoolParam(name="useExpression", label="Use Expression", value=False,
                       description="Choose the input with the Python expression below instead of 'Which'."),
        desc.StringParam(name="expression", label="Expression", value="", semantic="multiline",
                         description="Python expression giving the input number, e.g. "
                                     "\"1 if node('BG').file else 0\" (see the documentation).",
                         enabled=lambda node: node.useExpression.value),
    ]
    outputs = imageOutput("The chosen input.")

    def planContext(self, node):
        """ The parameters of the nodes read by the expression (snapshotted with the branch). """
        if not node.useExpression.value or not node.expression.value.strip():
            return None
        from meshroom.live import expressions
        return {"nodes": expressions.snapshot(node.graph, node.expression.value)}

    def _which(self, ctx) -> int:
        """ The chosen input number (clamped to the inputs); raises for an expression error. """
        if ctx.value("useExpression") and ctx.value("expression").strip():
            from meshroom.live import expressions
            context = ctx.planContext or {}
            lookup = expressions.snapshotLookup(context.get("nodes", {}), ctx.frame)
            result = expressions.evaluate(ctx.value("expression"), ctx.frame, lookup)
            try:
                which = int(round(float(result)))
            except (TypeError, ValueError):
                raise RuntimeError(f"The expression gives {result!r}, not an input number.")
        else:
            which = int(ctx.value("which"))
        return min(max(which, 0), len(_INPUTS) - 1)

    def _connected(self, ctx):
        return [name for name in _INPUTS if ctx.planNode.inputs.get(name) is not None]

    def isIdentity(self, ctx):
        name = _INPUTS[self._which(ctx)]
        return name if ctx.planNode.inputs.get(name) is not None else None

    def infoOnlyInputs(self, ctx):
        # Only the chosen input is rendered (as a pass-through): the others give the format at most
        return list(_INPUTS)

    def getInfo(self, ctx, inputsInfo):
        name = _INPUTS[self._which(ctx)]
        if inputsInfo.get(name) is not None:
            return inputsInfo[name]
        if ctx.planNode.inputs.get(name) is not None:
            # The chosen input is connected but fails: its error
            return ctx.evaluator.info(ctx._plan, ctx.planNode.inputs[name])
        return super().getInfo(ctx, inputsInfo)

    def render(self, ctx, inputs):
        # The chosen input is not connected: transparent black, in the format of the node
        from meshroom.live import ImageBuffer
        info = ctx.info()
        return ImageBuffer.constant(info.width, info.height, (0.0, 0.0, 0.0, 0.0))
