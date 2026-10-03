"""Download exactly the pinned classifier files and verify them.

Run as ``python -m gateway.injection.fetch [--dest DIR]``.

The ``models-init`` compose profile runs this once, on the ``bootstrap`` network only, into
the volume the gateway later mounts read-only. Locally, ``--dest models/cache`` fills the
gitignored cache the real-model tests read.

Each file is fetched from ``<endpoint>/<repo>/resolve/<revision>/<path>`` (the full commit,
never a branch), streamed to a ``.part`` file while it is hashed, and moved into place only
when its size and SHA-256 match the manifest. A file already in place that verifies is not
downloaded again. Exit status: 0 when every file verifies, 1 otherwise.
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Final

import httpx

from gateway.injection.manifest import (
    ManifestFile,
    ModelManifest,
    ModelVerificationError,
    check_file,
)

logger = logging.getLogger("gateway.injection.fetch")

DEFAULT_ENDPOINT: Final = "https://huggingface.co"
_TIMEOUT: Final = httpx.Timeout(30.0, read=120.0)
_CHUNK: Final = 1 << 20


def fetch_file(
    client: httpx.Client, manifest: ModelManifest, file: ManifestFile, dest: Path, endpoint: str
) -> None:
    """Download one pinned file into ``dest`` unless a verified copy is already there."""
    target = manifest.local_path(dest, file)
    try:
        check_file(target, file)
    except ModelVerificationError:
        pass
    else:
        logger.info("%s: already verified", file.path)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest, size = hashlib.sha256(), 0
    with (
        client.stream("GET", manifest.download_url(file, endpoint)) as response,
        partial.open("wb") as handle,
    ):
        response.raise_for_status()
        for chunk in response.iter_bytes(_CHUNK):
            size += len(chunk)
            if size > file.size:
                break
            digest.update(chunk)
            handle.write(chunk)
    if size != file.size or digest.hexdigest() != file.sha256:
        partial.unlink(missing_ok=True)
        msg = f"{file.path}: downloaded bytes do not match the manifest (size {size})"
        raise ModelVerificationError(msg)
    os.replace(partial, target)
    logger.info("%s: %d bytes, sha256 verified", file.path, size)


def fetch(manifest: ModelManifest, dest: Path, endpoint: str = DEFAULT_ENDPOINT) -> None:
    """Fetch and verify every pinned file; raises on the first failure."""
    with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client:
        for file in manifest.all_files():
            fetch_file(client, manifest, file, dest, endpoint)
    manifest.verify(dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--dest", type=Path, default=Path(os.environ.get("ACL_MODELS_DIR", "models/cache"))
    )
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # no signed CDN URLs in the log
    manifest = ModelManifest.load()
    logger.info("fetching %s@%s into %s", manifest.repo_id, manifest.revision, args.dest)
    try:
        fetch(manifest, args.dest, args.endpoint)
    except (ModelVerificationError, httpx.HTTPError, OSError) as exc:
        logger.error("model fetch failed: %s", exc)  # noqa: TRY400 -- the reason is the message
        return 1
    logger.info("every pinned file is in place and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
