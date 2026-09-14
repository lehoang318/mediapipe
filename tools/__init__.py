"""Model bootstrap and verification tools.

These scripts are only needed when the ONNX files are not already present in
``pretrained/`` (or to re-verify them); the runtime itself does not import
them except for the lazy auto-download in ``faceframework.model_store``.
"""
