"""
Live compositing engine: in-memory, pull-based evaluation of image nodes (`desc.ImageOpNode`).

See docs/live-compositing.md.
"""
from .imageBuffer import BBox, ImageBuffer, ImageInfo, RGBA, RGBA_CHANNELS
from .cache import ImageCache
from .evaluator import (
    CancelToken,
    EvalStatus,
    EvaluationCancelled,
    EvaluationError,
    Evaluator,
    PlanNode,
    RenderContext,
    RenderPlan,
    buildPlan,
    isImageOpNode,
)

__all__ = [
    "BBox", "ImageBuffer", "ImageInfo", "RGBA", "RGBA_CHANNELS", "ImageCache",
    "CancelToken", "EvalStatus", "EvaluationCancelled", "EvaluationError", "Evaluator",
    "PlanNode", "RenderContext", "RenderPlan", "buildPlan", "isImageOpNode",
]
