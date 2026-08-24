"""LoRA index assembled from every source that happens to be available.

Ordered by how much the source can be trusted and how universally it exists:

1. /sdapi/v1/loras          always present; carries the parsed safetensors header
2. <name>.json sidecar      read natively by Forge; populated by whoever wrote it
3. generation history       what the operator actually ran, with real weights

Nothing here is required. On the development instance the header alone gave
base model for 87% of 362 LoRAs, a title for 71% and training tags for 48%; the
sidecars pushed tags and descriptions to 99%. A clean Forge install keeps the
first tier and simply reports less.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .client import ForgeClient
from .history import HistoryIndex
from .presets import looks_like_accelerator

MAX_TRIGGERS = 6
MIN_TAG_COUNT = 2
MAX_DESCRIPTION = 600


def _shorten(text: str | None, limit: int) -> str | None:
    if not text:
        return None
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


@dataclass(frozen=True)
class LoraEntry:
    name: str
    title: str | None = None
    base_model: str | None = None
    triggers: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    description: str | None = None
    kind: str = "content"  # "content" or "accelerator"
    network_dim: int | None = None
    uses: int = 0
    typical_weight: float | None = None
    sources: tuple[str, ...] = field(default_factory=tuple)

    @property
    def suggested_weight(self) -> float:
        if self.typical_weight is not None:
            return self.typical_weight
        return 1.0 if self.kind == "accelerator" else 0.8

    def prompt_fragment(self, weight: float | None = None) -> str:
        chosen = weight if weight is not None else self.suggested_weight
        fragment = f"<lora:{self.name}:{chosen:g}>"
        if self.triggers:
            fragment += " " + ", ".join(self.triggers)
        return fragment

    def as_dict(self, verbose: bool = False) -> dict:
        data = {
            "name": self.name,
            "title": self.title,
            "base_model": self.base_model,
            "kind": self.kind,
            "triggers": list(self.triggers),
            "suggested_weight": self.suggested_weight,
            "uses": self.uses,
        }
        if verbose:
            data["tags"] = list(self.tags)
            # Descriptions can run to several thousand characters of author
            # notes; keep the useful head and spare the caller's context.
            data["description"] = _shorten(self.description, MAX_DESCRIPTION)
            data["network_dim"] = self.network_dim
            data["sources"] = list(self.sources)
            data["prompt_fragment"] = self.prompt_fragment()
        return data

    def search_blob(self) -> str:
        parts = [self.name, self.title or "", " ".join(self.tags), " ".join(self.triggers)]
        if self.description:
            parts.append(self.description[:400])
        return " ".join(parts).lower()


class LoraIndex:
    def __init__(self, client: ForgeClient, history: HistoryIndex) -> None:
        self._client = client
        self._history = history
        self._entries: list[LoraEntry] = []
        self._built = False
        self._error: str | None = None
        self._sidecars_found = 0

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def sidecars_found(self) -> int:
        return self._sidecars_found

    def build(self, force: bool = False) -> None:
        if self._built and not force:
            return
        self._built = True

        result = self._client.loras()
        if not result.ok:
            self._error = result.error
            return

        entries: list[LoraEntry] = []
        for raw in result.value or []:
            entry = self._build_entry(raw)
            if entry:
                entries.append(entry)
        self._entries = entries

    def _build_entry(self, raw: dict) -> LoraEntry | None:
        name = (raw.get("name") or "").strip()
        if not name:
            return None

        metadata = raw.get("metadata") or {}
        sources = ["api"]

        title = _clean(metadata.get("modelspec.title"))
        base_model = _clean(metadata.get("ss_base_model_version")) or _architecture(metadata)
        triggers = _triggers_from_tag_frequency(metadata)
        dim = _as_int(metadata.get("ss_network_dim"))

        sidecar = self._read_sidecar(raw.get("path"))
        tags: tuple[str, ...] = ()
        description = None
        if sidecar:
            sources.append("sidecar")
            self._sidecars_found += 1
            tags = tuple(str(tag) for tag in (sidecar.get("modelTags") or []))
            description = _clean(sidecar.get("description"))
            activation = _clean(sidecar.get("activation text"))
            if activation and not triggers:
                triggers = tuple(part.strip() for part in activation.split(",") if part.strip())[:MAX_TRIGGERS]
            base_model = base_model or _clean(sidecar.get("baseModel")) or _clean(sidecar.get("sd version"))
            title = title or _clean((sidecar.get("model") or {}).get("name") if isinstance(sidecar.get("model"), dict) else None)

        uses, typical_weight = self._history.lora_usage(name)
        if uses:
            sources.append("history")

        return LoraEntry(
            name=name,
            title=title,
            base_model=base_model,
            triggers=triggers,
            tags=tags,
            description=description,
            kind="accelerator" if looks_like_accelerator(name, tags) else "content",
            network_dim=dim,
            uses=uses,
            typical_weight=typical_weight,
            sources=tuple(sources),
        )

    def _read_sidecar(self, remote_path: str | None) -> dict | None:
        if not remote_path:
            return None
        local = self._client.config.localise(remote_path)
        if not local:
            return None
        candidate = os.path.splitext(local)[0] + ".json"
        if not os.path.isfile(candidate):
            return None
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, ValueError):
            return None
        return loaded if isinstance(loaded, dict) else None

    # -- queries --------------------------------------------------------------

    def all(self) -> list[LoraEntry]:
        self.build()
        return list(self._entries)

    def search(
        self,
        query: str = "",
        base_model: str | None = None,
        kind: str | None = None,
        limit: int = 20,
    ) -> list[LoraEntry]:
        self.build()
        terms = [term for term in query.lower().split() if term]
        results = []
        for entry in self._entries:
            if kind and entry.kind != kind:
                continue
            if base_model and (entry.base_model or "").lower() != base_model.lower():
                continue
            if terms:
                blob = entry.search_blob()
                score = sum(1 for term in terms if term in blob)
                if score == 0:
                    continue
            else:
                score = 0
            results.append((score, entry.uses, entry))

        results.sort(key=lambda item: (-item[0], -item[1], item[2].name.lower()))
        return [entry for _, _, entry in results[:limit]]

    def get(self, name: str) -> LoraEntry | None:
        self.build()
        lowered = name.lower()
        for entry in self._entries:
            if entry.name.lower() == lowered:
                return entry
        return None

    def summary(self) -> dict:
        """Compact overview: enough for an agent to know what exists, cheaply."""
        self.build()
        by_base: dict[str, int] = {}
        by_tag: dict[str, int] = {}
        accelerators = 0
        for entry in self._entries:
            # Base model casing varies by source ("anima" from the header,
            # "Anima" from the sidecar); fold them so counts do not split.
            key = (entry.base_model or "unknown").lower()
            by_base[key] = by_base.get(key, 0) + 1
            if entry.kind == "accelerator":
                accelerators += 1
            for tag in entry.tags[:1]:
                by_tag[tag] = by_tag.get(tag, 0) + 1
        return {
            "total": len(self._entries),
            "accelerators": accelerators,
            "by_base_model": dict(sorted(by_base.items(), key=lambda kv: -kv[1])[:6]),
            "categories": dict(sorted(by_tag.items(), key=lambda kv: -kv[1])[:8]),
            "with_triggers": sum(1 for entry in self._entries if entry.triggers),
            "sidecars_found": self._sidecars_found,
        }


def _clean(value) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _architecture(metadata: dict) -> str | None:
    raw = _clean(metadata.get("modelspec.architecture"))
    if not raw:
        return None
    return raw.split("/")[0]


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _triggers_from_tag_frequency(metadata: dict) -> tuple[str, ...]:
    """Pull likely trigger words from kohya-style training tag counts."""
    raw = metadata.get("ss_tag_frequency")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return ()
    if not isinstance(raw, dict):
        return ()

    counts: dict[str, int] = {}
    for group in raw.values():
        if not isinstance(group, dict):
            continue
        for tag, count in group.items():
            tag = str(tag).strip()
            if not tag or not isinstance(count, int):
                continue
            counts[tag] = counts.get(tag, 0) + count

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return tuple(tag for tag, count in ranked[:MAX_TRIGGERS] if count >= MIN_TAG_COUNT)
