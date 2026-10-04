"""The prompt-injection classifier: a pinned ONNX model, verified before it is loaded.

`manifest` pins the model (repository, revision, SHA-256 of every file), `classifier` runs it
(ONNX Runtime + ``tokenizers``, no torch, no pickle), and `fetch` downloads exactly the pinned
files (the ``models-init`` compose profile runs it).

**No phoning home.** ONNX Runtime 1.30 on Linux and macOS ships a 1DS telemetry uploader
(``core/platform/posix/telemetry.cc``) that posts to
``https://mobile.events.data.microsoft.com/OneCollector/1.0``. It starts when ``onnxruntime``
is imported (it also persists a device id and an offline event queue), and its HTTP callback
crashed the interpreter at exit (``recursive_mutex lock failed``). ``disable_telemetry_events()``
only mutes events after the uploader exists; the one full opt-out is the environment variable
``ORT_DISABLE_TELEMETRY`` (``core/platform/telemetry_environment.h``), read once when ORT
initializes. Every import of ``onnxruntime`` in the gateway goes through this package, so it
is set here, before any submodule can import ORT, and forced on (an inherited ``0`` does not
turn the uploader back on). ``huggingface_hub`` (installed with ``tokenizers``) is never
called, but its telemetry and network access are switched off the same way.
"""

import os
import sys

if "onnxruntime" in sys.modules and os.environ.get("ORT_DISABLE_TELEMETRY") != "1":
    msg = (
        "onnxruntime was imported before gateway.injection set ORT_DISABLE_TELEMETRY=1: "
        "its telemetry uploader may already be running"
    )
    raise ImportError(msg)

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
