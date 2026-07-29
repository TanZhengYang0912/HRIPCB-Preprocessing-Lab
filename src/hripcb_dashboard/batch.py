"""Safe normalization of individual and ZIP image uploads."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable
from pathlib import PurePosixPath


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _is_safe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _append_image(target: list[tuple[str, bytes]], skipped: list[str], name: str, payload: bytes, max_file_bytes: int) -> None:
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        skipped.append(f"Unsupported file skipped: {name}")
        return
    if len(payload) > max_file_bytes:
        skipped.append(f"File exceeds size limit and was skipped: {name}")
        return
    target.append((name, payload))


def extract_image_entries(
    uploaded_files: Iterable[tuple[str, bytes]],
    max_file_bytes: int = 200_000_000,
) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Return image payloads from direct uploads and ZIP archives.

    ZIP members are never written to disk. Absolute paths and traversal members
    are skipped before reading, preventing archive path traversal.
    """

    images: list[tuple[str, bytes]] = []
    skipped: list[str] = []
    for name, payload in uploaded_files:
        suffix = PurePosixPath(name).suffix.lower()
        if suffix != ".zip":
            _append_image(images, skipped, name, payload, max_file_bytes)
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    if not _is_safe_archive_name(member.filename):
                        skipped.append(f"Unsafe archive path skipped: {member.filename}")
                        continue
                    if member.file_size > max_file_bytes:
                        skipped.append(f"File exceeds size limit and was skipped: {member.filename}")
                        continue
                    _append_image(
                        images,
                        skipped,
                        member.filename,
                        archive.read(member),
                        max_file_bytes,
                    )
        except zipfile.BadZipFile:
            skipped.append(f"Invalid ZIP archive skipped: {name}")
    return images, skipped
