"""
SuperPoint keypoints (Magic Leap's architecture, the weights retrained by FaceCapture, Technicolor:
<models>/facecapture/keypoints/superpoints_retrain/superpoint_retrain.pth) for the FaceTracker: points of the skin
detected and described in two frames, matched by their descriptors; their moves carry the landmarks ("Optical Flow":
superpoint). The network is rebuilt here (the module names of the weights).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

WEIGHTS = "facecapture/keypoints/superpoints_retrain/superpoint_retrain.pth"


def _network(torch):
    nn = torch.nn
    F = torch.nn.functional

    class SuperPointNet(nn.Module):
        def __init__(self):
            super().__init__()
            c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
            self.vgg_conv1_1 = nn.Conv2d(1, c1, 3, 1, 1)
            self.vgg_conv1_2 = nn.Conv2d(c1, c1, 3, 1, 1)
            self.vgg_conv2_1 = nn.Conv2d(c1, c2, 3, 1, 1)
            self.vgg_conv2_2 = nn.Conv2d(c2, c2, 3, 1, 1)
            self.vgg_conv3_1 = nn.Conv2d(c2, c3, 3, 1, 1)
            self.vgg_conv3_2 = nn.Conv2d(c3, c3, 3, 1, 1)
            self.vgg_conv4_1 = nn.Conv2d(c3, c4, 3, 1, 1)
            self.vgg_conv4_2 = nn.Conv2d(c4, c4, 3, 1, 1)
            self.convDet_1 = nn.Conv2d(c4, c5, 3, 1, 1)
            self.convDet_2 = nn.Conv2d(c5, 65, 1, 1, 0)
            self.convDesc_1 = nn.Conv2d(c4, c5, 3, 1, 1)
            self.convDesc_2 = nn.Conv2d(c5, d1, 1, 1, 0)

        def forward(self, x):
            for a, b, pool in (("vgg_conv1_1", "vgg_conv1_2", True), ("vgg_conv2_1", "vgg_conv2_2", True),
                               ("vgg_conv3_1", "vgg_conv3_2", True), ("vgg_conv4_1", "vgg_conv4_2", False)):
                x = F.relu(getattr(self, b)(F.relu(getattr(self, a)(x))))
                if pool:
                    x = F.max_pool2d(x, 2, 2)
            semi = self.convDet_2(F.relu(self.convDet_1(x)))
            desc = self.convDesc_2(F.relu(self.convDesc_1(x)))
            return semi, F.normalize(desc, dim=1)
    return SuperPointNet()


class SuperPoint:
    """ Keypoints and descriptors of a grayscale image (CPU, or `device`). """

    def __init__(self, path: str, device=None, threshold: float = 0.015, nms: int = 4, border: int = 4):
        from . import requireTorch
        torch = requireTorch()
        self._torch = torch
        self.device = device or torch.device("cpu")
        self.net = _network(torch)
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        self.net.eval().to(self.device)
        self.threshold, self.nms, self.border = threshold, nms, border

    def detect(self, gray: np.ndarray, maxPoints: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """ (points (k, 2): x, y in pixels from the top left corner, descriptors (k, 256)) of an image (h, w) in
        [0, 1]: the most confident ones first, one per `nms` pixels. """
        torch = self._torch
        F = torch.nn.functional
        h, w = gray.shape
        # (a multiple of 8)
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        image = np.pad(np.asarray(gray, dtype=np.float32), ((0, ph), (0, pw)), mode="edge")
        x = torch.from_numpy(image)[None, None].to(self.device)
        with torch.no_grad():
            semi, coarse = self.net(x)
            probabilities = torch.softmax(semi, dim=1)[:, :-1]
            heat = F.pixel_shuffle(probabilities, 8)[0, 0]
            # Non-maximum suppression: the local maxima over (2 nms + 1) pixels
            pooled = F.max_pool2d(heat[None, None], 2 * self.nms + 1, 1, self.nms)[0, 0]
            keep = (heat >= self.threshold) & (heat == pooled)
            rows, columns = torch.nonzero(keep, as_tuple=True)
            scores = heat[rows, columns]
            inside = (rows >= self.border) & (rows < h - self.border) & (columns >= self.border) & \
                (columns < w - self.border)
            rows, columns, scores = rows[inside], columns[inside], scores[inside]
            order = torch.argsort(-scores)[:maxPoints]
            rows, columns = rows[order], columns[order]
            if not len(rows):
                return np.zeros((0, 2)), np.zeros((0, coarse.shape[1]))
            # The descriptors interpolated at the points (the coarse map is 1/8 of the image)
            grid = torch.stack([(columns.float() + 0.5) / (image.shape[1] / 2.0) - 1.0,
                                (rows.float() + 0.5) / (image.shape[0] / 2.0) - 1.0], dim=1)[None, None]
            descriptors = F.grid_sample(coarse, grid, align_corners=False)[0, :, 0].t()
            descriptors = F.normalize(descriptors, dim=1)
        points = np.stack([columns.cpu().numpy() + 0.5, rows.cpu().numpy() + 0.5], axis=1).astype(np.float64)
        return points, descriptors.cpu().numpy().astype(np.float32)


def match(descriptorsA: np.ndarray, descriptorsB: np.ndarray, ratio: float = 0.9,
          minScore: float = 0.7) -> np.ndarray:
    """ (k, 2) the indices of the mutual nearest neighbors of two sets of unit descriptors, the second nearest
    clearly further (`ratio` of the distances), similar enough (cosine above `minScore`). """
    if not len(descriptorsA) or not len(descriptorsB):
        return np.zeros((0, 2), np.int64)
    similarity = descriptorsA @ descriptorsB.T
    best = np.argmax(similarity, axis=1)
    back = np.argmax(similarity, axis=0)
    mutual = back[best] == np.arange(len(descriptorsA))
    distance = np.sqrt(np.maximum(2.0 - 2.0 * similarity, 0.0))
    sortedDistance = np.sort(distance, axis=1)
    second = sortedDistance[:, 1] if distance.shape[1] > 1 else np.full(len(distance), np.inf)
    distinct = sortedDistance[:, 0] < ratio * second
    good = mutual & distinct & (similarity[np.arange(len(best)), best] > minScore)
    return np.stack([np.where(good)[0], best[good]], axis=1)


def carry(points: np.ndarray, sources: np.ndarray, targets: np.ndarray, sigma: float,
          minimum: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """ (moves (n, 2), support (n,)) of points carried by the moves of matched keypoints (sources -> targets): at each
    point the gaussian-weighted (`sigma` pixels) median-filtered mean of the moves near it (a local, robust
    interpolation of a sparse flow); its support, the sum of the weights. Fewer than `minimum` matches: no move. """
    moves = np.zeros((len(points), 2))
    if len(sources) < minimum:
        return moves, np.zeros(len(points))
    displacement = targets - sources
    # Robust: the matches far from the median move of their neighbors are dropped
    from scipy.spatial import cKDTree
    tree = cKDTree(sources)
    _, near = tree.query(sources, k=min(8, len(sources)))
    local = np.median(displacement[near], axis=1)
    keep = np.linalg.norm(displacement - local, axis=1) < max(2.0, 0.25 * sigma)
    sources, displacement = sources[keep], displacement[keep]
    if len(sources) < minimum:
        return moves, np.zeros(len(points))
    weights = np.exp(-((points[:, None, :] - sources[None, :, :]) ** 2).sum(-1) / (2.0 * sigma ** 2))
    support = weights.sum(1)
    moves = (weights @ displacement) / np.maximum(support, 1e-12)[:, None]
    return moves, support


_INSTANCE = {}


def superPoint(device=None) -> SuperPoint:
    """ The SuperPoint network with its weights from the models folder (kept), or NeuralUnavailable. """
    from .models import modelFile
    path = modelFile("superpoint", WEIGHTS)
    key = (path, str(device))
    if key not in _INSTANCE:
        _INSTANCE.clear()
        _INSTANCE[key] = SuperPoint(path, device)
    return _INSTANCE[key]


def landmarkMoves(detector: SuperPoint, grayA: np.ndarray, grayB: np.ndarray, points: np.ndarray,
                  margin: float = 0.3) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """ (moves (n, 2), support (n,)) in pixels (from the top left corner) of points (n, 2) of image A in image B: the
    keypoints detected around them (their box with `margin`) in both, matched, carried; None with too few matches. """
    h, w = grayA.shape
    x0, y0 = points.min(0)
    x1, y1 = points.max(0)
    size = max(x1 - x0, y1 - y0)
    left, top = int(max(0, x0 - margin * size)), int(max(0, y0 - margin * size))
    right, bottom = int(min(w, x1 + margin * size)), int(min(h, y1 + margin * size))
    if right - left < 16 or bottom - top < 16:
        return None
    pointsA, descriptorsA = detector.detect(grayA[top:bottom, left:right])
    pointsB, descriptorsB = detector.detect(grayB[top:bottom, left:right])
    matches = match(descriptorsA, descriptorsB)
    if len(matches) < 8:
        return None
    offset = np.array([left, top], dtype=np.float64)
    return carry(points, pointsA[matches[:, 0]] + offset, pointsB[matches[:, 1]] + offset, sigma=0.08 * size)
