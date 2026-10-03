"""The prompt-injection classifier: a pinned ONNX model, verified before it is loaded.

`manifest` pins the model (repository, revision, SHA-256 of every file), `classifier` runs it
(ONNX Runtime + ``tokenizers``, no torch, no pickle), and `fetch` downloads exactly the pinned
files (the ``models-init`` compose profile runs it).
"""
