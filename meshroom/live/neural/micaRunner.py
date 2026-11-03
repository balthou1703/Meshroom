"""
The shape of a whole head (FLAME 2020: face, skull, ears, neck) from photos of the face with MICA (Zielonka et al.,
"Towards Metrical Reconstruction of Human Faces", MPI-IS), for the FaceBuilder ("Generate AI Head", "AI Model": mica).
NON-COMMERCIAL: MICA and FLAME 2020 are licensed for non-commercial scientific research only; a reference to compare
the other models with, not for production.

Its networks are rebuilt here (an ArcFace iresnet100 and MICA's mapping network to the 300 FLAME shape coefficients),
nothing of MICA's code is needed; its files, in the models folder (MESHROOM_MODELS_PATH/mica), come from its authors
with their licenses:
    mica.tar           the trained weights (https://github.com/Zielon/MICA: install.sh)
    generic_model.pkl  FLAME 2020 (https://flame.is.tue.mpg.de, an account: FLAME2020.zip)
    mediapipe_landmark_embedding.npz  (optional, the FLAME site) the MediaPipe landmarks on FLAME: the head placed on
                       the face by them (else its face is found on renders of it)

    python micaRunner.py head.npz crop1.png [crop2.png ...]

The crops are 112x112, aligned as ArcFace expects them (eyes, nose, mouth corners on its template: aiHead.arcfaceCrop);
the identity is averaged over them. Writes head.npz: vertices (n, 3) float32 in centimeters, faces (m, 3) int32,
colors (n, 3) uint8 (gray).
"""
import argparse
import os
import pickle
import sys


def _iresnet100(torch):
    """ The ArcFace iresnet100 of insightface (its module names: those of the weights of MICA). """
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
            self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
            self.prelu = nn.PReLU(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
            self.downsample = downsample

        def forward(self, x):
            out = self.bn3(self.conv2(self.prelu(self.bn2(self.conv1(self.bn1(x))))))
            return out + (self.downsample(x) if self.downsample is not None else x)

    class IResNet(nn.Module):
        def __init__(self, layers=(3, 13, 30, 3)):
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
            self.prelu = nn.PReLU(64)
            self.layer1 = self._layer(64, layers[0])
            self.layer2 = self._layer(128, layers[1])
            self.layer3 = self._layer(256, layers[2])
            self.layer4 = self._layer(512, layers[3])
            self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
            self.dropout = nn.Dropout(p=0.0)
            self.fc = nn.Linear(512 * 7 * 7, 512)
            self.features = nn.BatchNorm1d(512, eps=1e-05)

        def _layer(self, planes, blocks):
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, 2, bias=False),
                                       nn.BatchNorm2d(planes, eps=1e-05))
            layers = [Block(self.inplanes, planes, 2, downsample)]
            self.inplanes = planes
            layers += [Block(planes, planes) for _ in range(1, blocks)]
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.prelu(self.bn1(self.conv1(x)))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            x = torch.flatten(self.dropout(self.bn2(x)), 1)
            return self.features(self.fc(x))
    return IResNet()


def _mapping(torch, inputs=512, hidden=300, outputs=300, layers=3):
    """ MICA's mapping network: the ArcFace identity to the FLAME shape coefficients. """
    nn = torch.nn

    class Mapping(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.ModuleList([nn.Linear(inputs, hidden)] + [nn.Linear(hidden, hidden) for _ in range(layers)])
            self.output = nn.Linear(hidden, outputs)

        def forward(self, z):
            for layer in self.network:
                z = torch.nn.functional.leaky_relu(layer(z), negative_slope=0.2)
            return self.output(z)
    return Mapping()


class _ChumpyStub:
    """ A chumpy array of the FLAME pickle, without chumpy: its data kept. """

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state

    def array(self):
        import numpy as np
        state = self.state if isinstance(self.state, dict) else {}
        for key in ("x", "a", "r"):
            if key in state and state[key] is not None:
                return np.asarray(state[key])
        for value in state.values():
            if isinstance(value, np.ndarray):
                return value
        raise ValueError("a chumpy array of the FLAME model has no data")


class _FlameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChumpyStub
        return super().find_class(module, name)


def loadFlame(path):
    """ (v_template (n, 3), shapedirs (n, 3, 300), faces (m, 3)) of the FLAME pickle, in meters. """
    import numpy as np
    with open(path, "rb") as f:
        data = _FlameUnpickler(f, encoding="latin1").load()

    def array(value):
        if isinstance(value, _ChumpyStub):
            return value.array()
        if hasattr(value, "toarray"):  # a scipy sparse matrix
            return value.toarray()
        return np.asarray(value)
    return (array(data["v_template"]).astype(np.float64), array(data["shapedirs"]).astype(np.float64)[:, :, :300],
            array(data["f"]).astype(np.int64))


def mediapipeLandmarks(vertices, faces, path):
    """ (468, 3): the MediaPipe landmarks on a FLAME head from its MediaPipe landmark embedding (the .npz of the FLAME
    site: the face and barycentric coordinates of some of them), NaN for the others; None without the file. """
    import numpy as np
    if not os.path.isfile(path):
        return None
    data = np.load(path, allow_pickle=True)

    def find(*parts):
        for key in data.files:
            if all(part in key.lower() for part in parts):
                return np.asarray(data[key])
        raise SystemExit(f"micaRunner: {path} has no {'/'.join(parts)} ({data.files})")
    faceIndices = find("face", "idx").astype(np.int64).ravel()
    coordinates = find("b_coord").astype(np.float64).reshape(len(faceIndices), 3)
    indices = find("indices").astype(np.int64).ravel()
    points = (vertices[faces[faceIndices]] * coordinates[:, :, None]).sum(1)
    landmarks = np.full((468, 3), np.nan)
    keep = indices < 468
    landmarks[indices[keep]] = points[keep]
    return landmarks


def main(argv=None):
    parser = argparse.ArgumentParser(description="MICA: a FLAME head from aligned face crops (non-commercial)")
    parser.add_argument("output", help="the .npz written")
    parser.add_argument("crops", nargs="+", help="112x112 ArcFace-aligned crops of the face (PNG)")
    parser.add_argument("--models", default=os.path.join(os.environ.get("MESHROOM_MODELS_PATH", ""), "mica"),
                        help="the folder of mica.tar and generic_model.pkl")
    args = parser.parse_args(argv)
    weights, flame = os.path.join(args.models, "mica.tar"), os.path.join(args.models, "generic_model.pkl")
    missing = [path for path in (weights, flame) if not os.path.isfile(path)]
    if missing:
        raise SystemExit("micaRunner: missing " + ", ".join(missing) + " (MICA: mica.tar from its install.sh; FLAME 2020: "
                         "generic_model.pkl from FLAME2020.zip, https://flame.is.tue.mpg.de). Non-commercial licenses.")
    import numpy as np
    import torch
    from PIL import Image
    print("[mica] loading", flush=True)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    arcface = _iresnet100(torch)
    arcface.load_state_dict(checkpoint["arcface"])
    mapping = _mapping(torch)
    regressor = {key[len("regressor."):]: value for key, value in checkpoint["flameModel"].items()
                 if key.startswith("regressor.")}
    mapping.load_state_dict(regressor)
    arcface.eval()
    mapping.eval()
    template, shapedirs, faces = loadFlame(flame)
    crops = []
    for path in args.crops:
        rgb = np.asarray(Image.open(path).convert("RGB").resize((112, 112)), dtype=np.float32)
        crops.append(((rgb - 127.5) / 127.5).transpose(2, 0, 1))
    print(f"[mica] identity of {len(crops)} views", flush=True)
    with torch.no_grad():
        identities = torch.nn.functional.normalize(arcface(torch.as_tensor(np.stack(crops))), dim=1)
        # All the views together: the mean identity (on the sphere)
        identity = torch.nn.functional.normalize(identities.mean(0, keepdim=True), dim=1)
        shape = mapping(identity)[0].numpy().astype(np.float64)
    vertices = (template + shapedirs @ shape) * 100.0
    extra = {}
    landmarks = mediapipeLandmarks(vertices, faces, os.path.join(args.models, "mediapipe_landmark_embedding.npz"))
    if landmarks is not None:
        extra["landmarks"] = landmarks.astype(np.float32)
        print(f"[mica] {int(np.isfinite(landmarks).all(1).sum())} MediaPipe landmarks on the head", flush=True)
    np.savez_compressed(args.output, vertices=vertices.astype(np.float32), faces=faces.astype(np.int32),
                        colors=np.full((len(vertices), 3), 190, np.uint8), shape=shape.astype(np.float32), **extra)
    print(f"[mica] written {args.output}: {len(vertices)} vertices", flush=True)


if __name__ == "__main__":
    sys.exit(main())
