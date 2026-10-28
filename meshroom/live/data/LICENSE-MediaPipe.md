The canonical face mesh in faceHead.npz ("faceVertices", "faceTriangles") comes from MediaPipe
(mediapipe/modules/face_geometry/data/canonical_face_model.obj, https://github.com/google-ai-edge/mediapipe),
Copyright The MediaPipe Authors, licensed under the Apache License, Version 2.0:
https://www.apache.org/licenses/LICENSE-2.0

The ears, skull and neck ("vertices", "triangles") come from ICT-FaceKit (generic_neutral_mesh.obj), MIT license:
see LICENSE-ICT-FaceKit.md. Both were aligned, cut and joined by buildFaceHead.py.
