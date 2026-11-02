TripoSR (Stability AI & Tripo AI, MIT: LICENSE), the `tsr` package of https://github.com/VAST-AI-Research/TripoSR
at commit 107cefdc244c39106fa830359024f6a2f1c78871, for the FaceBuilder "Generate AI Head" (../triposrRunner.py).
Changes for Meshroom: bake_texture.py removed; trimesh, rembg, imageio, torchmcubes, huggingface_hub imported only
when used (they are not); the config of the DINO encoder read from the models folder (MESHROOM_TRIPOSR_DINO_CONFIG).
