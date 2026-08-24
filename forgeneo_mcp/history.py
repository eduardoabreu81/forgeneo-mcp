"""Aggregation of past generations into per-combination sampling regimes.

This is the highest-confidence source the bridge has. Forge's own preset table
states what an architecture *should* use; the output folder states what the
operator actually ran and kept. Where they disagree, this wins.

Regimes are keyed by (checkpoint, accelerators) rather than by checkpoint
alone, because a turbo LoRA changes the viable step count as much as a turbo
checkpoint does.
"""

from __future__ import annotations

import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .infotext import parse_infotext, read_generation_metadata
from .presets import looks_like_accelerator

# JPEG/WebP never carry a PNG text chunk, but they still get a .txt sibling when
# save_txt is on, so they are worth scanning.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".avif", ".jxl")
DANBOORU_MARKERS = ("score_9", "masterpiece", "best quality")
CHECKPOINT_SUFFIXES = (".safetensors", ".ckpt", ".gguf")


def normalise_checkpoint(name: str) -> str:
    """Reduce a checkpoint reference to something two sources can agree on.

    /sdapi/v1/options reports a path with subfolder and extension
    ("Anima\\animeMix_v10.safetensors") while infotext records the bare
    name ("animeMix_v10"). Without this the history lookup never matches
    and every profile silently falls back to preset defaults.
    """
    cleaned = (name or "").replace("\\", "/").split("/")[-1].strip()
    lowered = cleaned.lower()
    for suffix in CHECKPOINT_SUFFIXES:
        if lowered.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    # Forge appends a hash in brackets in some infotext variants.
    if cleaned.endswith("]") and "[" in cleaned:
        cleaned = cleaned[: cleaned.rindex("[")].strip()
    return cleaned.lower()


@dataclass(frozen=True)
class Regime:
    """What actually worked for one checkpoint + accelerator combination."""

    checkpoint: str
    accelerators: tuple[str, ...]
    samples: int
    steps: float | None
    cfg: float | None
    sampler: str | None
    scheduler: str | None
    danbooru_ratio: float
    # None when nothing was observed. Forge only records Shift / Distilled CFG
    # Scale for engines that consume it, so its presence in past infotext is
    # direct evidence of whether this architecture uses the parameter at all.
    uses_shift: bool | None = None
    shift: float | None = None
    common_loras: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "accelerators": list(self.accelerators),
            "samples": self.samples,
            "steps": self.steps,
            "cfg": self.cfg,
            "sampler": self.sampler,
            "scheduler": self.scheduler,
            "danbooru_ratio": round(self.danbooru_ratio, 2),
            "uses_shift": self.uses_shift,
            "shift": self.shift,
            "common_loras": [{"name": n, "weight": w} for n, w in self.common_loras],
        }


class HistoryIndex:
    """Lazily built index over the Forge output directory."""

    def __init__(self, output_dir: str | None, limit: int = 600) -> None:
        self._output_dir = output_dir
        self._limit = limit
        self._regimes: dict[tuple[str, tuple[str, ...]], Regime] = {}
        self._by_checkpoint: dict[str, list[Regime]] = defaultdict(list)
        self._lora_usage: Counter[str] = Counter()
        self._lora_weights: dict[str, list[float]] = defaultdict(list)
        self._scanned = 0
        self._files_without_metadata = 0
        self._sources: Counter[str] = Counter()
        self._prompts_by_checkpoint: dict[str, list[str]] = defaultdict(list)
        self._built = False

    @property
    def available(self) -> bool:
        return bool(self._output_dir) and os.path.isdir(self._output_dir or "")

    def set_output_dir(self, output_dir: str | None) -> None:
        """Point the index at a directory discovered after construction.

        The output folder is only knowable once the instance answers, so the
        server resolves it lazily and hands it over here. Changing it discards
        anything already indexed.
        """
        if output_dir == self._output_dir:
            return
        self._output_dir = output_dir
        self._built = False
        self._scanned = 0
        self._files_without_metadata = 0
        self._sources.clear()
        self._prompts_by_checkpoint.clear()
        self._regimes.clear()
        self._by_checkpoint.clear()
        self._lora_usage.clear()
        self._lora_weights.clear()

    @property
    def scanned(self) -> int:
        return self._scanned

    def diagnostics(self) -> dict:
        """Where the indexed metadata came from, and how much was unreadable.

        Lets the caller distinguish "you have no outputs yet" from "your outputs
        carry no parameters because both enable_pnginfo and save_txt are off".
        """
        self.build()
        return {
            "indexed": self._scanned,
            "files_without_metadata": self._files_without_metadata,
            "sources": dict(self._sources),
        }

    def build(self, force: bool = False) -> None:
        if self._built and not force:
            return
        self._built = True
        if not self.available:
            return

        buckets: dict[tuple[str, tuple[str, ...]], list] = defaultdict(list)
        for path in self._recent_files():
            raw, source = read_generation_metadata(path)
            if not raw:
                self._files_without_metadata += 1
                continue
            self._sources[source] += 1
            info = parse_infotext(raw)
            checkpoint = info.checkpoint
            if not checkpoint:
                continue
            self._scanned += 1
            accelerators = tuple(
                sorted(name for name, _ in info.loras if looks_like_accelerator(name))
            )
            key = normalise_checkpoint(checkpoint)
            buckets[(key, accelerators)].append(info)
            if info.prompt and len(self._prompts_by_checkpoint[key]) < 60:
                self._prompts_by_checkpoint[key].append(info.prompt)
            for name, weight in info.loras:
                self._lora_usage[name] += 1
                self._lora_weights[name].append(weight)

        for (checkpoint, accelerators), entries in buckets.items():
            regime = _summarise(checkpoint, accelerators, entries)
            self._regimes[(checkpoint, accelerators)] = regime
            self._by_checkpoint[checkpoint].append(regime)

    def _recent_files(self) -> list[str]:
        found: list[tuple[float, str]] = []
        for root, _, files in os.walk(self._output_dir or ""):
            for name in files:
                if name.lower().endswith(IMAGE_SUFFIXES):
                    full = os.path.join(root, name)
                    try:
                        found.append((os.path.getmtime(full), full))
                    except OSError:
                        continue
        found.sort(reverse=True)
        return [path for _, path in found[: self._limit]]

    def regime_for(self, checkpoint: str, accelerators: tuple[str, ...] | None = None) -> Regime | None:
        """The sampling regime for a checkpoint, optionally for a known setup.

        `accelerators=None` means "I have not decided what to load yet", and
        answers with the most-observed regime. Passing an explicit tuple asks
        about that precise combination, and an empty tuple genuinely means
        "with no accelerator".

        The distinction matters: the development instance had one stray
        generation of a checkpoint without an accelerator against seven with
        one. Treating a bare call as an exact request for "no accelerator"
        picked that single outlier over the representative sample.
        """
        self.build()
        key = normalise_checkpoint(checkpoint)
        candidates = self._by_checkpoint.get(key)
        if not candidates:
            return None
        if accelerators is not None:
            exact = self._regimes.get((key, tuple(sorted(accelerators))))
            if exact:
                return exact
        return max(candidates, key=lambda regime: regime.samples)

    def regimes_for(self, checkpoint: str) -> list[Regime]:
        """Every observed regime for a checkpoint, most-observed first."""
        self.build()
        candidates = self._by_checkpoint.get(normalise_checkpoint(checkpoint), [])
        return sorted(candidates, key=lambda regime: -regime.samples)

    def prompts_for(self, checkpoint: str, limit: int = 40) -> list[str]:
        """Positive prompts previously used with a checkpoint.

        Feeds dialect inference: the vocabulary an operator actually reached for
        with a model is stronger evidence than anything the file declares.
        """
        self.build()
        key = normalise_checkpoint(checkpoint)
        return list(self._prompts_by_checkpoint.get(key, [])[:limit])

    def lora_usage(self, name: str) -> tuple[int, float | None]:
        """How many times a LoRA was used and its typical weight."""
        self.build()
        weights = self._lora_weights.get(name) or []
        median = round(statistics.median(weights), 2) if weights else None
        return self._lora_usage.get(name, 0), median

    def top_loras(self, limit: int = 10) -> list[tuple[str, int]]:
        self.build()
        return self._lora_usage.most_common(limit)

    def accelerator_habit(self) -> dict:
        """Whether this operator works with accelerator LoRAs, and which.

        A distilled workflow is a standing choice, not a per-image one: someone
        who always loads a turbo LoRA expects low steps at CFG ~1 by default,
        and someone who never does expects the architecture's own numbers. The
        agent needs to know which world it is in before writing the first
        prompt.
        """
        self.build()
        total = sum(regime.samples for regime in self._regimes.values())
        if not total:
            return {"known": False, "rate": None, "common": []}

        with_accelerator = sum(
            regime.samples for regime in self._regimes.values() if regime.accelerators
        )
        counts: Counter[str] = Counter()
        for regime in self._regimes.values():
            for name in regime.accelerators:
                counts[name] += regime.samples

        common = []
        for name, uses in counts.most_common(3):
            weights = self._lora_weights.get(name) or []
            common.append(
                {
                    "name": name,
                    "uses": uses,
                    "typical_weight": round(statistics.median(weights), 2) if weights else None,
                }
            )
        return {
            "known": True,
            "rate": round(with_accelerator / total, 2),
            "generations": total,
            "common": common,
        }


def _summarise(key: str, accelerators: tuple[str, ...], entries: list) -> Regime:
    # Report the name as the operator sees it, not the normalised lookup key.
    checkpoint = next((entry.checkpoint for entry in entries if entry.checkpoint), key)
    steps = [entry.steps for entry in entries if entry.steps is not None]
    cfgs = [entry.cfg for entry in entries if entry.cfg is not None]
    samplers = Counter(entry.sampler for entry in entries if entry.sampler)
    schedulers = Counter(entry.scheduler for entry in entries if entry.scheduler)
    danbooru = sum(1 for entry in entries if _looks_danbooru(entry.prompt))
    lora_weights: dict[str, list[float]] = defaultdict(list)
    for entry in entries:
        for name, weight in entry.loras:
            lora_weights[name].append(weight)

    common = tuple(
        (name, round(statistics.median(weights), 2))
        for name, weights in sorted(lora_weights.items(), key=lambda kv: -len(kv[1]))[:5]
    )
    shifts = [
        value
        for value in (entry.get_float("Shift") or entry.get_float("Distilled CFG Scale") for entry in entries)
        if value is not None
    ]
    return Regime(
        checkpoint=checkpoint,
        accelerators=accelerators,
        samples=len(entries),
        steps=round(statistics.median(steps), 1) if steps else None,
        cfg=round(statistics.median(cfgs), 2) if cfgs else None,
        sampler=samplers.most_common(1)[0][0] if samplers else None,
        scheduler=schedulers.most_common(1)[0][0] if schedulers else None,
        danbooru_ratio=danbooru / len(entries) if entries else 0.0,
        uses_shift=bool(shifts) if entries else None,
        shift=round(statistics.median(shifts), 2) if shifts else None,
        common_loras=common,
    )


def _looks_danbooru(prompt: str) -> bool:
    lowered = prompt.lower()
    if any(marker in lowered for marker in DANBOORU_MARKERS):
        return True
    # Underscore-joined tags are the other strong signal of a booru dialect.
    return lowered.count("_") >= 2 and "," in lowered
