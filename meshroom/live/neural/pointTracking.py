"""
Point trackers: many points tracked together through a sequence by a network that sees all its frames at once, and
tells when a point is hidden. BootsTAPIR (tapnet, Apache-2.0 / CC-BY-4.0) by default; CoTracker3 (CC-BY-NC-4.0,
non-commercial use only) as an option.

`pointTracker(name, device).track(video, queries)` takes a video (T, H, W, 3) (display RGB in [0, 1], rows from the top)
and the positions (N, 2) of the points at its first frame (x right, y down, pixels, 0 at the top left corner of the
image) and returns their positions (T, N, 2) and whether they are visible (T, N). tracking.trackManyNeural uses it
through the sequence, a few frames at a time.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from . import NeuralUnavailable, requireTorch, torchDevice
from .models import cachedModel, importPackage, loadError, modelFile

TRACKERS = ("tapir", "cotracker")


class PointTracker:
    """ A point tracking network. """

    # Size (height, width) the video is resized to before the network, None: as it is
    resolution = None
    # Frames given to the network at once (memory)
    chunk = 24

    def __init__(self, device):
        self.device = device

    def track(self, video: np.ndarray, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _resized(self, video: np.ndarray):
        """ (the video as a (T, H', W', 3) float tensor in [0, 1] at the resolution of the network, scale x, scale y). """
        torch = requireTorch()
        import torch.nn.functional as F
        if self.resolution is None:
            return torch.from_numpy(np.ascontiguousarray(video, dtype=np.float32)).to(self.device), 1.0, 1.0
        height, width = self.resolution
        # One frame at a time on the device (a whole full resolution video would not fit in its memory)
        frames = []
        for frame in video:
            tensor = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32)).to(self.device).permute(2, 0, 1)[None]
            frames.append(F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False,
                                        antialias=True)[0].permute(1, 2, 0))
        return torch.stack(frames), width / float(video.shape[2]), height / float(video.shape[1])


class TapirTracker(PointTracker):
    """ BootsTAPIR (tapnet's PyTorch model). """
    resolution = (256, 256)
    chunk = 32

    def __init__(self, device):
        super().__init__(device)
        torch = requireTorch()
        importPackage("tapir")
        path = modelFile("tapir", "tapnet/bootstapir_checkpoint_v2.pt")

        def load():
            from tapnet.torch import tapir_model
            state = torch.load(path, map_location="cpu")
            errors = []
            # BootsTAPIR: one more pyramid level than TAPIR (the arguments changed between tapnet versions)
            for arguments in ({"pyramid_level": 1}, {"pyramid_level": 1, "extra_convs": True}, {}):
                try:
                    model = tapir_model.TAPIR(**arguments)
                    model.load_state_dict(state)
                    return model.eval().to(device)
                except Exception as e:
                    errors.append(str(e))
            raise loadError("tapir", RuntimeError("; ".join(errors[-1:])))
        self.model = cachedModel(("tapir", str(device)), load)

    def track(self, video, queries):
        torch = requireTorch()
        frames, sx, sy = self._resized(video)
        with torch.no_grad():
            # Frames in [-1, 1]; query points (t, y, x) at the resolution of the network
            points = torch.tensor([[0.0, y * sy, x * sx] for x, y in queries], dtype=torch.float32, device=self.device)
            outputs = self.model((frames * 2.0 - 1.0)[None], points[None])
            tracks = outputs["tracks"][0]                              # (N, T, 2): x, y
            occluded = torch.sigmoid(outputs["occlusion"][0])
            uncertain = torch.sigmoid(outputs["expected_dist"][0])
            visible = (1.0 - occluded) * (1.0 - uncertain) > 0.5      # (N, T)
        tracks = tracks.permute(1, 0, 2).to("cpu").numpy().astype(np.float64)
        tracks[:, :, 0] /= sx
        tracks[:, :, 1] /= sy
        return tracks, visible.permute(1, 0).to("cpu").numpy()


class CoTracker(PointTracker):
    """ CoTracker3 offline (non-commercial license). """
    # Its own working resolution
    resolution = (384, 512)
    chunk = 48

    def __init__(self, device):
        super().__init__(device)
        requireTorch()
        importPackage("cotracker")
        path = modelFile("cotracker", "cotracker/scaled_offline.pth")

        def load():
            from cotracker.predictor import CoTrackerPredictor
            errors = []
            # CoTracker3 (offline) then CoTracker2 constructors
            for arguments in ({"offline": True, "v2": False, "window_len": 60}, {}):
                try:
                    return CoTrackerPredictor(checkpoint=path, **arguments).eval().to(device)
                except TypeError as e:
                    errors.append(str(e))
                except Exception as e:
                    raise loadError("cotracker", e) from e
            raise loadError("cotracker", RuntimeError("; ".join(errors)))
        self.model = cachedModel(("cotracker", str(device)), load)

    def track(self, video, queries):
        torch = requireTorch()
        frames, sx, sy = self._resized(video)
        with torch.no_grad():
            # Video (1, T, 3, H, W) in [0, 255]; queries (t, x, y)
            clip = (frames * 255.0).permute(0, 3, 1, 2)[None]
            points = torch.tensor([[0.0, x * sx, y * sy] for x, y in queries], dtype=torch.float32, device=self.device)
            tracks, visible = self.model(clip, queries=points[None])
        tracks = tracks[0].to("cpu").numpy().astype(np.float64)
        tracks[:, :, 0] /= sx
        tracks[:, :, 1] /= sy
        return tracks, visible[0].to("cpu").numpy().astype(bool)


def pointTracker(name: str, device) -> PointTracker:
    """ The point tracker `name` ("tapir" or "cotracker") on `device` (None: the CPU), or NeuralUnavailable. """
    device = torchDevice(device)
    if name == "tapir":
        return TapirTracker(device)
    if name == "cotracker":
        return CoTracker(device)
    raise NeuralUnavailable(f"unknown point tracker '{name}'")
