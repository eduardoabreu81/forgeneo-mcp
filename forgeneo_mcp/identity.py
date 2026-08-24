"""Working out which prompt dialect a checkpoint expects.

On a fresh install there is no history, no sidecar, and the file name carries
the lineage 0.5% of the time. Pony, Illustrious and stock SDXL share an
architecture and a tensor layout, so nothing in the file separates them.

Rather than guess, the resolver reports what it knows and how it knows it. When
that is not enough it says so, and the caller asks the operator once — the
answer is cached by file hash and never asked again.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import dialects
from .dialects import Dialect

CACHE_DIR = Path(os.environ.get("FORGENEO_CACHE_DIR") or (Path.home() / ".forgeneo-mcp"))
CACHE_FILE = CACHE_DIR / "dialects.json"
# Below this share, the installed LoRAs are too mixed to hint at anything.
LORA_HINT_THRESHOLD = 0.6
MIN_LORA_SAMPLE = 5


@dataclass(frozen=True)
class DialectResolution:
    dialect: Dialect | None
    source: str
    confidence: str  # "confirmed" | "high" | "medium" | "low" | "unknown"
    detail: str = ""
    alternatives: tuple[str, ...] = ()

    @property
    def known(self) -> bool:
        return self.dialect is not None

    def as_dict(self) -> dict:
        payload = {
            "source": self.source,
            "confidence": self.confidence,
            "detail": self.detail,
        }
        if self.dialect:
            payload.update(self.dialect.as_dict())
        else:
            payload["dialect"] = None
            payload["alternatives"] = list(self.alternatives)
        return payload


def _load_cache() -> dict:
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def remember(identifier: str, dialect_key: str) -> bool:
    """Persist an operator's answer so the question is asked only once.

    Stored under the MCP's own directory, never inside the Forge installation.
    """
    if dialect_key not in dialects.BY_KEY or not identifier:
        return False
    cache = _load_cache()
    cache[identifier.lower()] = dialect_key
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=1, sort_keys=True)
    except OSError:
        return False
    return True


def recall(identifier: str) -> Dialect | None:
    if not identifier:
        return None
    return dialects.BY_KEY.get(_load_cache().get(identifier.lower(), ""))


def lora_ecosystem(entries: list) -> tuple[Dialect | None, str]:
    """What the installed LoRAs suggest about this collection's lineage.

    Never decisive on its own: it describes the library, not the checkpoint. It
    turns an open question ("which dialect?") into one an operator can confirm
    with a word ("Illustrious, right?").
    """
    counts: dict[str, int] = {}
    for entry in entries:
        dialect = dialects.from_declared_base(getattr(entry, "base_model", None))
        if dialect:
            counts[dialect.key] = counts.get(dialect.key, 0) + 1
    total = sum(counts.values())
    if total < MIN_LORA_SAMPLE:
        return None, ""
    key, count = max(counts.items(), key=lambda item: item[1])
    share = count / total
    if share < LORA_HINT_THRESHOLD:
        return None, f"installed LoRAs are mixed across lineages ({total} declaring a base model)"
    return dialects.BY_KEY[key], f"{count} of {total} installed LoRAs declare {key}"


def resolve(
    *,
    identifier: str | None,
    architecture: str | None,
    declared_base: str | None = None,
    observed_prompts: list[str] | None = None,
    lora_entries: list | None = None,
) -> DialectResolution:
    """Best available answer, with its provenance."""
    if identifier:
        cached = recall(identifier)
        if cached:
            return DialectResolution(cached, "cache", "confirmed", "answered previously for this checkpoint")

    declared = dialects.from_declared_base(declared_base)
    if declared:
        return DialectResolution(declared, "declared base model", "high", f"baseModel: {declared_base}")

    observed = dialects.from_observed_prompts(observed_prompts or [])
    if observed and len(observed_prompts or []) >= MIN_LORA_SAMPLE:
        return DialectResolution(
            observed,
            "observed prompts",
            "high",
            f"inferred from {len(observed_prompts or [])} past generations with this checkpoint",
        )

    implied = dialects.for_architecture(architecture)
    if implied:
        return DialectResolution(implied, "architecture", "medium", f"{architecture} implies this dialect")

    hinted, detail = lora_ecosystem(lora_entries or [])
    if hinted:
        return DialectResolution(hinted, "installed LoRAs", "low", detail)

    return DialectResolution(
        None,
        "none",
        "unknown",
        (
            f"'{architecture}' covers several lineages that share tensors and preset, so the "
            "dialect cannot be read from the file. Ask the operator once; the answer is cached."
        ),
        alternatives=("pony", "illustrious", "animagine", "sdxl_base"),
    )
