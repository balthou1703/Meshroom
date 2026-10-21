"""
Actions of compositing nodes: long operations started by a button of the node (a `desc.PushButtonParam`
with the "liveAction" semantic), run in a worker thread by the UI (e.g. the autotrack and the camera
solve of the CameraTracker).

The descriptor implements `runAction(name, context)` and returns the new values of its attributes
{name: value}, set as one undoable change when the action ends; "_disconnect": [input names] also
disconnects these inputs in that change. While it runs, `context.publish(values, frame)` shows
intermediate values (e.g. the shapes tracked so far) and moves the viewer to `frame`: the graph is
updated live, and the whole action stays one undoable change (a cancelled action keeps what it published).
"""
from __future__ import annotations

from typing import Callable, Optional

from .evaluator import CancelToken, EvaluationCancelled, Evaluator, RenderPlan
from .imageBuffer import ImageBuffer, ImageInfo


class ActionContext:
    """
    What an action needs: the values of the node, the images entering it (rendered on demand), the
    progress report and the cancellation.

    Attributes:
        values: {parameter name: value} of the node when the action started.
        info: the ImageInfo of the input (None if the node has no connected input).
        currentFrame: the current frame of the viewer.
    """

    def __init__(self, values: dict, evaluator: Optional[Evaluator] = None, plan: Optional[RenderPlan] = None,
                 currentFrame: int = 1, token: Optional[CancelToken] = None,
                 progress: Optional[Callable[[float, str], None]] = None, inputPlans: Optional[dict] = None,
                 inputValues: Optional[dict] = None, publish: Optional[Callable[[dict, int], None]] = None,
                 nodePlan: Optional[RenderPlan] = None):
        self.values = values
        # Plan of the node itself (its own outputs, e.g. the camera solved by a Resection, see nodeContext)
        self._nodePlan = nodePlan
        self._publish = publish
        self._evaluator = evaluator
        self._plan = plan
        # Plans of the other connected image inputs ({input name: plan}), e.g. a mask
        self._inputPlans = dict(inputPlans or {})
        # Parameter values of the nodes connected to the image inputs ({input name: {name: value}})
        self._inputValues = dict(inputValues or {})
        self.currentFrame = currentFrame
        self._token = token or CancelToken()
        self._progress = progress
        self.info: Optional[ImageInfo] = evaluator.info(plan) if evaluator is not None and plan is not None else None

    def value(self, name: str):
        return self.values[name]

    @property
    def frames(self):
        """ The frames of the input (the current frame for a still image), within the track range of the node
        ("Limit Track Range", params.trackRangeParams) when it has one. """
        from .params import trackRange
        info = self.info
        if info is None or not info.hasFrameRange:
            return [self.currentFrame]
        return trackRange(self.values, range(info.firstFrame, info.lastFrame + 1))

    def inputFrames(self, name: str):
        """ The frames of the image input `name` (the current frame for a still image or a disconnected input). """
        if name == "input":
            return self.frames
        if self._evaluator is None or name not in self._inputPlans:
            return [self.currentFrame]
        info = self._evaluator.info(self._inputPlans[name])
        if info is None or not info.hasFrameRange:
            return [self.currentFrame]
        return list(range(info.firstFrame, info.lastFrame + 1))

    def render(self, frame: int) -> ImageBuffer:
        """ The image entering the node at a frame. """
        if self._evaluator is None or self._plan is None:
            raise RuntimeError("The input of the node is not connected.")
        return self._evaluator.render(self._plan, frame, self._token)

    def inputValues(self, name: str) -> Optional[dict]:
        """ The parameter values of the node connected to an image input (e.g. its tracks), or None. """
        return self._inputValues.get(name)

    def hasInput(self, name: str) -> bool:
        """ Whether the image input `name` is connected ("input": the main one). """
        return (self._plan is not None) if name == "input" else name in self._inputPlans

    def renderInput(self, name: str, frame: int) -> ImageBuffer:
        """ The image entering the node through the image input `name` at a frame. """
        if name == "input":
            return self.render(frame)
        if self._evaluator is None or name not in self._inputPlans:
            raise RuntimeError(f"The input '{name}' of the node is not connected.")
        return self._evaluator.render(self._inputPlans[name], frame, self._token)

    def nodeContext(self, frame: Optional[int] = None):
        """
        The render context of the node itself at a frame (the current one by default): its camera, geo and matrix
        inputs (`inputData`) and its own outputs at any frame (`ownData`), with the values it had when the action started.
        """
        from .evaluator import RenderContext
        if self._evaluator is None or self._nodePlan is None:
            raise RuntimeError("The node cannot be evaluated.")
        plan = self._nodePlan
        context = RenderContext(plan.nodes[plan.target], self.currentFrame if frame is None else frame, self._evaluator, plan)
        context._token = self._token
        return context

    def checkCancelled(self):
        if self._token.cancelled:
            raise EvaluationCancelled()

    @property
    def cancelled(self) -> bool:
        return self._token.cancelled

    def publish(self, values: dict, frame: Optional[int] = None):
        """ Show intermediate values of the action (see the module documentation), at `frame`. """
        if self._publish is not None:
            self._publish(dict(values), frame)

    def progress(self, fraction: float, message: str = ""):
        if self._progress:
            self._progress(max(0.0, min(1.0, fraction)), message)
