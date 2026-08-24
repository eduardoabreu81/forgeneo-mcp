"""Optional lineage lookup by file hash.

Off by default: it is the only part of the bridge that leaves the machine. When
enabled it answers the one question nothing local can — whether an `xl`
checkpoint is Pony, Illustrious or stock SDXL — using the SHA256 Forge already
computes and caches.

No account and no API key: the by-hash endpoint is public and read-only. Nothing
is uploaded but the hash itself.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

ENDPOINT = "https://civitai.com/api/v1/model-versions/by-hash/"
TIMEOUT = 20.0
USER_AGENT = "forgeneo-mcp"


def enabled() -> bool:
    return os.environ.get("FORGE_CIVITAI_LOOKUP", "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class HashLookup:
    ok: bool
    base_model: str | None = None
    name: str | None = None
    trained_words: tuple[str, ...] = ()
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "base_model": self.base_model,
            "name": self.name,
            "trained_words": list(self.trained_words),
            "error": self.error,
        }


def lookup(sha256: str) -> HashLookup:
    """Ask what a file is. Requires FORGE_CIVITAI_LOOKUP to be enabled."""
    if not enabled():
        return HashLookup(False, error="lookup disabled (set FORGE_CIVITAI_LOOKUP=1 to allow)")
    if not sha256 or len(sha256) < 12:
        return HashLookup(False, error="no usable hash for this file")

    request = urllib.request.Request(
        ENDPOINT + sha256,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return HashLookup(False, error="hash not found on CivitAI")
        return HashLookup(False, error=f"HTTP {exc.code} from CivitAI")
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return HashLookup(False, error=f"lookup failed: {str(exc)[:80]}")

    if not isinstance(payload, dict):
        return HashLookup(False, error="unexpected response shape")

    words = payload.get("trainedWords") or []
    return HashLookup(
        True,
        base_model=payload.get("baseModel"),
        name=(payload.get("model") or {}).get("name") if isinstance(payload.get("model"), dict) else None,
        trained_words=tuple(str(word) for word in words[:8]),
    )
