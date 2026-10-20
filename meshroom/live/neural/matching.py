"""
Feature matching with LightGlue and ALIKED features (the lightglue package): corresponding points of two images, even
far apart in time or seen from very different places (the CameraTracker autotrack, the "Align" of the Resection).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from . import displayRGB, requireTorch, torchDevice
from .models import cachedModel, importPackage, loadError, useTorchHub


class Matcher:
    """ ALIKED features matched by LightGlue. """

    def __init__(self, device, maxKeypoints: int = 2048):
        requireTorch()
        importPackage("lightglue")
        self.device = torchDevice(device)

        def load():
            from lightglue import ALIKED, LightGlue
            useTorchHub()
            try:
                extractor = ALIKED(max_num_keypoints=maxKeypoints).eval().to(self.device)
                matcher = LightGlue(features="aliked").eval().to(self.device)
            except Exception as e:
                raise loadError("lightglue", e) from e
            return extractor, matcher
        self.extractor, self.matcher = cachedModel(("lightglue", maxKeypoints, str(self.device)), load)

    def _tensor(self, image):
        """ The image over its format as a (3, h, w) tensor in [0, 1], and the height of its format. """
        torch = requireTorch()
        from ..imageBuffer import BBox
        rgb = displayRGB(image, BBox.fromFormat(*image.format))
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(self.device), image.format[1]

    def features(self, image):
        """ The ALIKED features of an image (to match it with several others without extracting them again). """
        torch = requireTorch()
        tensor, height = self._tensor(image)
        with torch.no_grad():
            return self.extractor.extract(tensor), height

    def match(self, imageA, imageB, featuresA=None, featuresB=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The points of A (k, 2) and their matches in B (k, 2), in Nuke pixels of their formats (y up), with the match
        scores (k,) in [0, 1].
        """
        torch = requireTorch()
        from lightglue.utils import rbd
        featuresA, heightA = featuresA or self.features(imageA)
        featuresB, heightB = featuresB or self.features(imageB)
        with torch.no_grad():
            matches = self.matcher({"image0": featuresA, "image1": featuresB})
        a, b, matches = rbd(featuresA), rbd(featuresB), rbd(matches)
        pairs = matches["matches"]
        pointsA = a["keypoints"][pairs[:, 0]].to("cpu").numpy().astype(np.float64)
        pointsB = b["keypoints"][pairs[:, 1]].to("cpu").numpy().astype(np.float64)
        scores = matches["scores"].to("cpu").numpy().astype(np.float64)
        # Keypoints: pixel indices from the top left corner -> Nuke pixels (centers at .5, y up)
        return _toNuke(pointsA, heightA), _toNuke(pointsB, heightB), scores


def _toNuke(points: np.ndarray, height: int) -> np.ndarray:
    return np.c_[points[:, 0] + 0.5, height - (points[:, 1] + 0.5)] if len(points) else np.zeros((0, 2))


def matcher(device, maxKeypoints: int = 2048) -> Matcher:
    """ The LightGlue matcher on `device` (None: the CPU), or NeuralUnavailable. """
    from . import NeuralUnavailable
    try:
        return Matcher(device, maxKeypoints)
    except NeuralUnavailable:
        raise
    except Exception as e:
        raise loadError("lightglue", e) from e
