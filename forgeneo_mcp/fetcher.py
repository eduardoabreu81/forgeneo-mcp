"""Fetching a missing module, only when told to.

Everything else in this bridge is read-only. This is the one place that writes,
and it writes multi-gigabyte files into someone's models folder — often across a
network share, sometimes onto a disk with no room. So the default is to report
what would happen and stop: the caller has to pass an explicit confirmation for
anything to be downloaded.

Downloads land in a .part file and are renamed on completion, so an interrupted
transfer never leaves something that looks like a usable model.
"""

from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass

CHUNK = 1024 * 1024
TIMEOUT = 60.0
USER_AGENT = "forgeneo-mcp"
# Refuse to start a download that would leave the disk under this much room.
HEADROOM_BYTES = 2 * 1024 ** 3


@dataclass(frozen=True)
class Plan:
    url: str
    destination: str
    size_bytes: int | None
    free_bytes: int | None
    already_present: bool
    writable: bool
    blocked: str | None = None

    def as_dict(self) -> dict:
        def gb(value):
            return None if value is None else round(value / 1024 ** 3, 2)

        return {
            "url": self.url,
            "destination": self.destination,
            "size_gb": gb(self.size_bytes),
            "free_gb": gb(self.free_bytes),
            "already_present": self.already_present,
            "writable": self.writable,
            "blocked": self.blocked,
        }


def remote_size(url: str) -> int | None:
    """Ask how large a file is without downloading it."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, ValueError, TimeoutError):
        return None


def plan(url: str, folder: str, filename: str) -> Plan:
    """Work out what fetching this would involve, without doing any of it."""
    destination = os.path.join(folder, filename)
    present = os.path.isfile(destination)
    writable = os.path.isdir(folder) and os.access(folder, os.W_OK)

    free = None
    try:
        free = shutil.disk_usage(folder).free
    except OSError:
        pass

    size = None if present else remote_size(url)

    blocked = None
    if present:
        blocked = "file already exists; nothing to do"
    elif not os.path.isdir(folder):
        blocked = f"target folder does not exist or is unreachable: {folder}"
    elif not writable:
        blocked = f"target folder is not writable from here: {folder}"
    elif size and free is not None and free - size < HEADROOM_BYTES:
        blocked = (
            f"not enough room: needs {size / 1024 ** 3:.1f} GB and only "
            f"{free / 1024 ** 3:.1f} GB is free"
        )

    return Plan(url, destination, size, free, present, writable, blocked)


def fetch(url: str, folder: str, filename: str) -> dict:
    """Download the file. Callers are expected to have confirmed with a human."""
    intent = plan(url, folder, filename)
    if intent.blocked:
        return {"ok": False, "error": intent.blocked, "plan": intent.as_dict()}

    partial = intent.destination + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    written = 0
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response, open(partial, "wb") as handle:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                written += len(block)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        _discard(partial)
        return {"ok": False, "error": f"download failed after {written} bytes: {str(exc)[:120]}"}

    if intent.size_bytes and written != intent.size_bytes:
        _discard(partial)
        return {
            "ok": False,
            "error": f"incomplete: expected {intent.size_bytes} bytes, received {written}",
        }

    try:
        os.replace(partial, intent.destination)
    except OSError as exc:
        _discard(partial)
        return {"ok": False, "error": f"could not finalise the file: {exc}"}

    return {
        "ok": True,
        "saved": intent.destination,
        "size_gb": round(written / 1024 ** 3, 2),
        "next_step": "refresh the module list in Forge, then select it for this preset",
    }


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
