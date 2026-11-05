"""
Face parsing (BiSeNet, face-parsing.PyTorch by zllrunning, MIT; its 19 classes of CelebAMask-HQ): the parts of a
face in an image, its ears apart (FaceBuilder: the ears of FLAME on their outline in the views). Brought from
FaceCapture (Technicolor), whose weights are used: <models>/facecapture/face_segmentor/bisenet/pretrain79999.pth.

The network is rebuilt here (ResNet-18 context path, attention refinement and feature fusion modules; the module
names those of the weights), nothing of its code is needed.
"""
from __future__ import annotations

import numpy as np

# The classes of the network
BACKGROUND, SKIN, LEFT_BROW, RIGHT_BROW, LEFT_EYE, RIGHT_EYE, GLASSES, LEFT_EAR, RIGHT_EAR, EARRING, NOSE, MOUTH, \
    UPPER_LIP, LOWER_LIP, NECK, NECKLACE, CLOTH, HAIR, HAT = range(19)
EARS = (LEFT_EAR, RIGHT_EAR)
# Into the categories of the MediaPipe multiclass segmenter (headFit: BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES,
# OTHERS)
TO_SELFIE = {BACKGROUND: 0, HAIR: 1, NECK: 2, CLOTH: 4, NECKLACE: 4, HAT: 5, GLASSES: 5, EARRING: 5}
WEIGHTS = "facecapture/face_segmentor/bisenet/pretrain79999.pth"


def _network(torch, classes: int = 19):
    nn = torch.nn
    F = torch.nn.functional

    class ConvBNReLU(nn.Module):
        def __init__(self, inputs, outputs, kernel=3, stride=1, padding=1):
            super().__init__()
            self.conv = nn.Conv2d(inputs, outputs, kernel, stride, padding, bias=False)
            self.bn = nn.BatchNorm2d(outputs)

        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))

    class BasicBlock(nn.Module):
        def __init__(self, inputs, outputs, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(outputs)
            self.conv2 = nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(outputs)
            self.downsample = None
            if inputs != outputs or stride != 1:
                self.downsample = nn.Sequential(nn.Conv2d(inputs, outputs, 1, stride, bias=False), nn.BatchNorm2d(outputs))

        def forward(self, x):
            residual = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
            shortcut = x if self.downsample is None else self.downsample(x)
            return F.relu(shortcut + residual)

    def layer(inputs, outputs, stride):
        return nn.Sequential(BasicBlock(inputs, outputs, stride), BasicBlock(outputs, outputs, 1))

    class Resnet18(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(3, 2, 1)
            self.layer1 = layer(64, 64, 1)
            self.layer2 = layer(64, 128, 2)
            self.layer3 = layer(128, 256, 2)
            self.layer4 = layer(256, 512, 2)

        def forward(self, x):
            x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
            feat8 = self.layer2(self.layer1(x))
            feat16 = self.layer3(feat8)
            return feat8, feat16, self.layer4(feat16)

    class AttentionRefinement(nn.Module):
        def __init__(self, inputs, outputs):
            super().__init__()
            self.conv = ConvBNReLU(inputs, outputs)
            self.conv_atten = nn.Conv2d(outputs, outputs, 1, bias=False)
            self.bn_atten = nn.BatchNorm2d(outputs)

        def forward(self, x):
            feat = self.conv(x)
            attention = torch.sigmoid(self.bn_atten(self.conv_atten(F.avg_pool2d(feat, feat.shape[2:]))))
            return feat * attention

    class ContextPath(nn.Module):
        def __init__(self):
            super().__init__()
            self.resnet = Resnet18()
            self.arm16 = AttentionRefinement(256, 128)
            self.arm32 = AttentionRefinement(512, 128)
            self.conv_head32 = ConvBNReLU(128, 128)
            self.conv_head16 = ConvBNReLU(128, 128)
            self.conv_avg = ConvBNReLU(512, 128, 1, 1, 0)

        def forward(self, x):
            feat8, feat16, feat32 = self.resnet(x)
            average = self.conv_avg(F.avg_pool2d(feat32, feat32.shape[2:]))
            feat32 = self.arm32(feat32) + F.interpolate(average, feat32.shape[2:], mode="nearest")
            feat32 = self.conv_head32(F.interpolate(feat32, feat16.shape[2:], mode="nearest"))
            feat16 = self.arm16(feat16) + feat32
            feat16 = self.conv_head16(F.interpolate(feat16, feat8.shape[2:], mode="nearest"))
            return feat8, feat16, feat32

    class FeatureFusion(nn.Module):
        def __init__(self, inputs, outputs):
            super().__init__()
            self.convblk = ConvBNReLU(inputs, outputs, 1, 1, 0)
            self.conv1 = nn.Conv2d(outputs, outputs // 4, 1, bias=False)
            self.conv2 = nn.Conv2d(outputs // 4, outputs, 1, bias=False)

        def forward(self, spatial, context):
            feat = self.convblk(torch.cat([spatial, context], dim=1))
            attention = torch.sigmoid(self.conv2(F.relu(self.conv1(F.avg_pool2d(feat, feat.shape[2:])))))
            return feat * attention + feat

    class Output(nn.Module):
        def __init__(self, inputs, middle, classes):
            super().__init__()
            self.conv = ConvBNReLU(inputs, middle)
            self.conv_out = nn.Conv2d(middle, classes, 1, bias=False)

        def forward(self, x):
            return self.conv_out(self.conv(x))

    class BiSeNet(nn.Module):
        def __init__(self, classes):
            super().__init__()
            self.cp = ContextPath()
            self.ffm = FeatureFusion(256, 256)
            self.conv_out = Output(256, 256, classes)
            self.conv_out16 = Output(128, 64, classes)
            self.conv_out32 = Output(128, 64, classes)

        def forward(self, x):
            feat8, context8, _ = self.cp(x)
            out = self.conv_out(self.ffm(feat8, context8))
            return F.interpolate(out, x.shape[2:], mode="bilinear", align_corners=True)
    return BiSeNet(classes)


class FaceParser:
    """ BiSeNet face parsing on the CPU (or `device`): segment() gives the confidences of the 19 classes. """

    def __init__(self, path: str, device=None):
        from . import requireTorch
        torch = requireTorch()
        self._torch = torch
        self.device = device or torch.device("cpu")
        weights = torch.load(path, map_location="cpu", weights_only=False)
        weights = weights.get("state_dict", weights) if isinstance(weights, dict) else weights
        classes = int(weights["conv_out.conv_out.weight"].shape[0])
        self.net = _network(torch, classes)
        self.net.load_state_dict(weights)
        self.net.eval().to(self.device)

    def segment(self, rgb: np.ndarray, size: int = 512) -> np.ndarray:
        """ (h, w, classes) confidences (softmax) of an image (h, w, 3) uint8 or float in [0, 1], rows from the top: the
        network on it resized to `size` (it was trained on 512 x 512 faces), back to its size. """
        torch = self._torch
        F = torch.nn.functional
        image = np.asarray(rgb)[:, :, :3]
        image = image.astype(np.float32) / (255.0 if image.dtype == np.uint8 else 1.0)
        h, w = image.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None]
        x = F.interpolate(x, (size, size), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        with torch.no_grad():
            out = self.net(((x - mean) / std).to(self.device))
            out = F.interpolate(torch.softmax(out, 1), (h, w), mode="bilinear", align_corners=False)
        return out[0].permute(1, 2, 0).cpu().numpy()

    def close(self):
        pass


def toSelfie(confidences: np.ndarray) -> np.ndarray:
    """ (h, w, 6): the confidences of the 19 classes gathered into the categories of the MediaPipe multiclass segmenter
    (headFit.FitView labels: the face, its ears among it, is FACE_SKIN). """
    selfie = np.zeros(confidences.shape[:2] + (6,), np.float32)
    for index in range(confidences.shape[2]):
        selfie[:, :, TO_SELFIE.get(index, 3)] += confidences[:, :, index]
    return selfie


def ears(confidences: np.ndarray) -> np.ndarray:
    """ (h, w) the confidence of an ear (either). """
    return confidences[:, :, list(EARS)].sum(2)


def faceParser(device=None) -> FaceParser:
    """ The face parser with its weights from the models folder, or NeuralUnavailable. """
    from .models import modelFile
    return FaceParser(modelFile("faceparsing", WEIGHTS), device)
