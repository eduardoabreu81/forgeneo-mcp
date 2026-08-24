"""Runtime configuration, read from the environment.

The bridge never requires filesystem access: every setting below degrades to a
working default. When FORGE_PATH_MAP is set the server can translate the paths
reported by Forge into paths reachable from this machine, which lets tools
return file locations instead of megabytes of base64.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_URL = "http://127.0.0.1:7860"
DEFAULT_TIMEOUT = 600.0


def _parse_path_map(raw: str) -> tuple[tuple[str, str], ...]:
    """Parse "REMOTE=LOCAL;REMOTE=LOCAL" into normalised prefix pairs.

    Remote prefixes are compared case-insensitively with forward slashes, since
    Forge reports Windows paths and we may be reading them over SMB.
    """
    pairs: list[tuple[str, str]] = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        remote, local = entry.split("=", 1)
        remote = remote.strip().replace("\\", "/").rstrip("/")
        local = local.strip().replace("\\", "/").rstrip("/")
        if remote and local:
            pairs.append((remote.lower(), local))
    return tuple(pairs)


@dataclass(frozen=True)
class Config:
    url: str = DEFAULT_URL
    auth: tuple[str, str] | None = None
    timeout: float = DEFAULT_TIMEOUT
    path_map: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    output_dir: str | None = None
    history_limit: int = 600

    @classmethod
    def from_env(cls) -> "Config":
        raw_auth = os.environ.get("FORGE_AUTH", "").strip()
        auth = None
        if ":" in raw_auth:
            user, _, password = raw_auth.partition(":")
            auth = (user, password)

        raw_timeout = os.environ.get("FORGE_TIMEOUT", "").strip()
        try:
            timeout = float(raw_timeout) if raw_timeout else DEFAULT_TIMEOUT
        except ValueError:
            timeout = DEFAULT_TIMEOUT

        raw_limit = os.environ.get("FORGE_HISTORY_LIMIT", "").strip()
        try:
            history_limit = int(raw_limit) if raw_limit else 600
        except ValueError:
            history_limit = 600

        return cls(
            url=os.environ.get("FORGE_URL", DEFAULT_URL).rstrip("/"),
            auth=auth,
            timeout=timeout,
            path_map=_parse_path_map(os.environ.get("FORGE_PATH_MAP", "")),
            output_dir=os.environ.get("FORGE_OUTPUT_DIR") or None,
            history_limit=history_limit,
        )

    def localise(self, remote_path: str) -> str | None:
        """Translate a path reported by Forge into one reachable from here.

        Returns None when no mapping applies, which callers treat as "this file
        exists, but not for us" rather than as an error.
        """
        if not remote_path:
            return None
        candidate = remote_path.replace("\\", "/")
        lowered = candidate.lower()
        for remote_prefix, local_prefix in self.path_map:
            if lowered.startswith(remote_prefix):
                return local_prefix + candidate[len(remote_prefix):]
        return None
