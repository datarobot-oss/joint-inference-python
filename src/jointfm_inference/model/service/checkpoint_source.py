"""Make the configured checkpoint file available before the service loads it.

The image ships no weights. A checkpoint is either already at the configured
path (mounted into the container, or a local file) or downloaded there once
from the configured URL and verified against its SHA-256 digest. A file that
is already present is reused only when its digest matches, so a changed
download source can never serve stale weights.

``urlopen`` is imported at module scope so tests can patch the symbol at
``jointfm_inference.model.service.checkpoint_source.urlopen``.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from .config import InferenceServiceConfig

DOWNLOAD_CHUNK_BYTES = 1 << 20
DOWNLOAD_TIMEOUT_SECONDS = 60.0
_PARTIAL_SUFFIX = ".partial"
# The temporary file is owner-only; the checkpoint stays readable to every
# process that shares the models directory.
CHECKPOINT_FILE_MODE = 0o644


def resolve_checkpoint(config: InferenceServiceConfig) -> Path:
    """Return the path of the configured checkpoint, downloading it when absent.

    Without ``config.checkpoint_url`` the file must already exist. With one,
    a present file must match ``config.checkpoint_sha256``; an absent file is
    downloaded next to its final path, verified, and only then moved into
    place, so an interrupted or corrupted download never leaves a file that
    a later start would load.

    Raises:
        FileNotFoundError: No URL is configured and no file is at the path.
        RuntimeError: A present or downloaded file does not match the digest.
        OSError: The download or a file operation fails.
    """
    checkpoint_path = Path(config.checkpoint_path)
    if config.checkpoint_url is None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found at configured path: {checkpoint_path}"
            )
        return checkpoint_path
    if config.checkpoint_sha256 is None:
        raise ValueError("a checkpoint URL requires its SHA-256 digest")
    if checkpoint_path.is_file():
        present_digest = file_sha256(checkpoint_path)
        if present_digest != config.checkpoint_sha256:
            raise RuntimeError(
                f"Checkpoint at {checkpoint_path} has SHA-256 {present_digest}, "
                f"but the configured download has {config.checkpoint_sha256}; "
                "remove the file or configure another checkpoint path"
            )
        return checkpoint_path
    download_checkpoint(
        config.checkpoint_url,
        checkpoint_path,
        sha256=config.checkpoint_sha256,
        token=config.checkpoint_token,
    )
    return checkpoint_path


def file_sha256(path: Path) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(
    url: str,
    checkpoint_path: Path,
    *,
    sha256: str,
    token: str | None,
) -> None:
    """Download ``url`` to ``checkpoint_path`` and verify its SHA-256 digest.

    The body streams into a temporary file in the target directory, so the
    final rename is atomic on one file system. ``token``, when given, is sent
    as an ``Authorization: Bearer`` header. Progress lines name the target
    path, never the URL, which may carry credentials in its query string.

    Raises:
        RuntimeError: The downloaded body does not match ``sha256``.
        OSError: The request or a file operation fails.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {} if token is None else {"Authorization": f"Bearer {token}"}
    print(f"Downloading checkpoint to {checkpoint_path}", file=sys.stderr)
    descriptor, partial_name = tempfile.mkstemp(
        dir=checkpoint_path.parent,
        prefix=f".{checkpoint_path.name}.",
        suffix=_PARTIAL_SUFFIX,
    )
    partial_path = Path(partial_name)
    try:
        digest = hashlib.sha256()
        size = 0
        with (
            os.fdopen(descriptor, "wb") as partial,
            urlopen(
                Request(url, headers=headers), timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response,
        ):
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                digest.update(chunk)
                partial.write(chunk)
                size += len(chunk)
        if digest.hexdigest() != sha256:
            raise RuntimeError(
                f"Downloaded checkpoint has SHA-256 {digest.hexdigest()}, "
                f"expected {sha256}"
            )
        partial_path.chmod(CHECKPOINT_FILE_MODE)
        partial_path.replace(checkpoint_path)
    finally:
        partial_path.unlink(missing_ok=True)
    print(
        f"Downloaded checkpoint to {checkpoint_path} ({size} bytes, SHA-256 verified)",
        file=sys.stderr,
    )
