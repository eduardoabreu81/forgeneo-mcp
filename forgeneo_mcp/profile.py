"""What the agent is currently working with, and how to prompt it.

The goal is that an agent never has to be told "this is a turbo anima model, use
11 steps and danbooru tags" — it asks once and the answer is derived from the
live instance plus observed usage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .client import ForgeClient
from .history import HistoryIndex
from .presets import LINEAGE_PROMPT_STYLE, PROMPT_STYLE, defaults_for, detect_lineage

TURBO_NAME_HINTS = ("turbo", "lightning", "hyper", "lcm", "dmd", "schnell", "flash", "distill")
# Fewer observations than this and the "measured" regime is really an anecdote.
MIN_CONFIDENT_SAMPLES = 5
# A preset that guides at CFG 1.0 describes a CFG-distilled architecture. This
# is a property of the family, not evidence about the individual checkpoint.
DISTILLED_ARCH_CFG = 1.0
# Only meaningful against a preset that expects many steps: running far below
# what the architecture asks for is a sign of a distilled variant.
TURBO_STEP_RATIO = 0.5
# Above this share of generations, loading an accelerator is a standing habit
# rather than an occasional experiment (and below its complement, the reverse).
ACCELERATOR_HABIT_THRESHOLD = 0.7


@dataclass(frozen=True)
class TurboAssessment:
    """Whether the loaded checkpoint is a distilled/turbo variant.

    Tri-state on purpose. A clean install has no history, most checkpoints carry
    no usable metadata, and the name is only a hint — so "unknown" is the honest
    answer far more often than either yes or no, and the agent needs to be able
    to tell the difference between "no" and "cannot tell".
    """

    state: str  # "yes" | "no" | "unknown"
    confidence: str  # "high" | "medium" | "low"
    evidence: tuple[str, ...]
    architecture_is_distilled: bool

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "architecture_is_distilled": self.architecture_is_distilled,
        }


def instance_defaults(options: dict, preset: str | None, mode: str = "t2i") -> dict:
    """Read the per-architecture defaults the running instance will apply.

    Forge exposes these as `<preset>_<mode>_<field>` options. They beat the
    table copied into presets.py because they come from the instance itself,
    and because any field left out of a request falls back to the API model's
    own defaults, which know nothing about the loaded architecture — that is how
    a generation picked up shift 3.5 when the architecture asks for 3.0.

    A width or height of 0 means "auto", so it is treated as unset.
    """
    if not preset:
        return {}
    prefix = f"{preset}_{mode}_"
    found: dict = {}
    for field_name in ("step", "cfg", "dcfg", "sampler", "scheduler", "width", "height"):
        value = options.get(prefix + field_name)
        if value in (None, "", 0, 0.0):
            continue
        found[field_name] = value
    return found


@dataclass(frozen=True)
class ModuleCheck:
    path: str
    name: str
    exists: bool | None  # None when we cannot see the filesystem at all


@dataclass(frozen=True)
class ModelProfile:
    checkpoint: str | None
    preset: str | None
    lineage: str | None
    turbo: TurboAssessment
    sampler: str | None
    scheduler: str | None
    steps: float | None
    cfg: float | None
    shift: float | None
    width: int | None
    height: int | None
    prompt_style: str
    parameter_source: str
    accelerators: tuple[str, ...]
    accelerator_habit: dict
    modules: tuple[ModuleCheck, ...]
    samples_observed: int
    known_regimes: tuple[dict, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "preset": self.preset,
            "lineage": self.lineage,
            "turbo": self.turbo.as_dict(),
            "recommended": {
                "sampler": self.sampler,
                "scheduler": self.scheduler,
                "steps": self.steps,
                "cfg": self.cfg,
                # Forge maps this onto distilled_cfg_scale, which the UI labels
                # "Shift" or "Distilled CFG Scale" depending on architecture.
                "shift": self.shift,
                "width": self.width,
                "height": self.height,
            },
            "parameter_source": self.parameter_source,
            "prompt_style": self.prompt_style,
            "requires_accelerators": list(self.accelerators),
            "accelerator_habit": self.accelerator_habit,
            "known_regimes": list(self.known_regimes),
            "modules": [
                {"name": m.name, "ok": m.exists, "path": m.path} for m in self.modules
            ],
            "samples_observed": self.samples_observed,
            "warnings": list(self.warnings),
        }


def build_profile(client: ForgeClient, history: HistoryIndex) -> ModelProfile | str:
    """Return a profile, or an error string when the instance is unreachable."""
    options = client.options()
    if not options.ok:
        return options.error or "could not read /sdapi/v1/options"

    data = options.value or {}
    checkpoint = data.get("sd_model_checkpoint")
    preset = data.get("forge_preset")
    warnings: list[str] = []

    modules = _check_modules(client, data, preset)
    missing = [m.name for m in modules if m.exists is False]
    if missing:
        warnings.append(
            f"preset '{preset}' declares modules that do not exist: {', '.join(missing)}"
        )

    regime = history.regime_for(checkpoint) if checkpoint else None
    fallback = defaults_for(preset)
    live = instance_defaults(data, preset)

    # The preset is the architecture talking, and architecture is not a matter
    # of taste: sampler, scheduler and shift come from it whenever it is known.
    # History only adjusts what an accelerator actually changes - step count and
    # guidance - and may override the sampler pair only on strong evidence.
    if not fallback and preset:
        warnings.append(
            f"preset '{preset}' is not in the known architecture table; sampler and scheduler "
            "cannot be validated against the architecture"
        )
    elif not preset:
        warnings.append(
            "the instance reports no active preset, so there is no architecture baseline; "
            "confirm the intended model family before generating"
        )

    # Structural parameters follow the instance first, then the copied table.
    arch_sampler = live.get("sampler") or (fallback.sampler if fallback else None)
    arch_scheduler = live.get("scheduler") or (fallback.scheduler if fallback else None)
    shift = live.get("dcfg")
    if shift is None and fallback:
        shift = fallback.shift
    # Forge records Shift only for engines that consume it, so past generations
    # settle whether this architecture uses the parameter. presets.py lists a
    # shift for krea because the UI shows the field, but krea's engine inherits
    # use_shift = False and discards it - recommending a value there is noise.
    if regime is not None and regime.uses_shift is False and regime.samples >= MIN_CONFIDENT_SAMPLES:
        shift = None
        warnings.append(
            "this architecture ignores shift / distilled CFG: no past generation with this "
            "checkpoint recorded the parameter, so it is not worth setting"
        )
    width = _as_int(live.get("width"))
    height = _as_int(live.get("height"))

    if regime and regime.steps is not None:
        sampler = arch_sampler or regime.sampler
        scheduler = arch_scheduler or regime.scheduler
        if regime.samples >= MIN_CONFIDENT_SAMPLES and regime.sampler and fallback:
            if regime.sampler != fallback.sampler:
                warnings.append(
                    f"{regime.samples} past generations used sampler '{regime.sampler}' instead of "
                    f"the preset's '{fallback.sampler}'; following the observed choice"
                )
                sampler = regime.sampler
                scheduler = regime.scheduler or scheduler
        steps, cfg = regime.steps, regime.cfg
        structural = "instance" if live.get("sampler") else "preset table"
        source = (
            f"{structural} defaults for '{preset}' (sampler, scheduler, shift); "
            f"history ({regime.samples} generations) for steps and CFG"
        )
        accelerators = regime.accelerators
        samples = regime.samples
        if samples < MIN_CONFIDENT_SAMPLES:
            warnings.append(
                f"parameters derived from only {samples} past generation(s); treat as a hint, "
                "not a measurement"
            )
        if accelerators:
            joined = ", ".join(f"<lora:{name}>" for name in accelerators)
            note = (
                f"these parameters were measured WITH {joined} in the prompt — that LoRA is what "
                "makes the low step count work, so include it or the image will come out undercooked"
            )
            if fallback:
                note += (
                    f". Generating without it means the preset values instead: about "
                    f"{fallback.steps} steps at CFG {fallback.cfg:g}"
                )
            warnings.append(note)
    elif live or fallback:
        sampler, scheduler = arch_sampler, arch_scheduler
        steps = _as_float(live.get("step")) or (float(fallback.steps) if fallback else None)
        cfg = _as_float(live.get("cfg"))
        if cfg is None and fallback:
            cfg = fallback.cfg
        origin = "instance" if live else "preset table"
        source = f"{origin} defaults for '{preset}' (no history for this checkpoint)"
        accelerators, samples = (), 0
    else:
        sampler = scheduler = steps = cfg = None
        source = "unknown - no history and no preset match"
        accelerators, samples = (), 0
        warnings.append("no parameter guidance available; ask the operator before generating")

    habit = history.accelerator_habit()
    if habit.get("known") and habit.get("rate") is not None:
        rate = habit["rate"]
        names = ", ".join(item["name"] for item in habit["common"])
        if rate >= ACCELERATOR_HABIT_THRESHOLD:
            warnings.append(
                f"{rate:.0%} of this operator's generations load an accelerator LoRA ({names}). "
                "Assume a distilled workflow: include one and keep steps low at CFG ~1, unless "
                "asked otherwise"
            )
        elif rate <= 1 - ACCELERATOR_HABIT_THRESHOLD:
            warnings.append(
                f"only {rate:.0%} of past generations used an accelerator; assume the "
                "architecture's own step count and guidance unless asked otherwise"
            )

    turbo = assess_turbo(checkpoint, accelerators, fallback, steps, samples)
    if turbo.state == "unknown" and not turbo.architecture_is_distilled:
        warnings.append(
            "cannot tell whether this checkpoint is a distilled/turbo variant. The numbers below "
            "suit the architecture, but a turbo variant would want far fewer steps at CFG ~1 — "
            "ask the operator, or generate one cheap test image before committing to a batch."
        )
    prompt_style = _prompt_style(checkpoint, preset, regime)

    return ModelProfile(
        checkpoint=checkpoint,
        preset=preset,
        lineage=detect_lineage(checkpoint or ""),
        turbo=turbo,
        sampler=sampler,
        scheduler=scheduler,
        steps=steps,
        cfg=cfg,
        shift=shift,
        width=width,
        height=height,
        prompt_style=prompt_style,
        parameter_source=source,
        accelerators=accelerators,
        accelerator_habit=habit,
        modules=modules,
        samples_observed=samples,
        known_regimes=tuple(
            {
                "accelerators": list(item.accelerators),
                "samples": item.samples,
                "steps": item.steps,
                "cfg": item.cfg,
                "sampler": item.sampler,
            }
            for item in (history.regimes_for(checkpoint) if checkpoint else [])
        ),
        warnings=tuple(warnings),
    )


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check_modules(client: ForgeClient, options: dict, preset: str | None) -> tuple[ModuleCheck, ...]:
    key = f"forge_additional_modules_{preset}" if preset else "forge_additional_modules"
    declared = options.get(key) or options.get("forge_additional_modules") or []
    checks: list[ModuleCheck] = []
    for remote in declared:
        local = client.config.localise(str(remote))
        exists = os.path.isfile(local) if local else None
        checks.append(
            ModuleCheck(path=str(remote), name=os.path.basename(str(remote).replace("\\", "/")), exists=exists)
        )
    return tuple(checks)


def assess_turbo(
    checkpoint: str | None,
    accelerators: tuple[str, ...],
    fallback,
    observed_steps: float | None,
    samples: int,
) -> TurboAssessment:
    """Decide whether this checkpoint behaves as a distilled variant.

    Only observations independent of the preset count as evidence. The earlier
    version compared the recommended step count against a fixed ceiling, but on
    a clean install that number *is* the preset, so it merely restated the
    architecture back to itself: klein (4 steps) and zit (9) always read as
    turbo, while an SDXL Turbo with no history never did.
    """
    evidence: list[str] = []
    arch_distilled = bool(fallback and fallback.cfg <= DISTILLED_ARCH_CFG)
    if arch_distilled:
        evidence.append(
            f"architecture guides at CFG {fallback.cfg:g}, so the family is CFG-distilled by design"
        )

    if accelerators:
        evidence.append(f"accelerator LoRA in use: {', '.join(accelerators)}")
        return TurboAssessment("yes", "high", tuple(evidence), arch_distilled)

    # Running far below what the architecture asks for, often enough to be a
    # habit rather than an experiment, is independent evidence.
    if (
        samples >= MIN_CONFIDENT_SAMPLES
        and observed_steps is not None
        and fallback is not None
        and observed_steps <= fallback.steps * TURBO_STEP_RATIO
    ):
        evidence.append(
            f"{samples} past generations ran at {observed_steps:g} steps against a "
            f"preset of {fallback.steps}"
        )
        return TurboAssessment("yes", "high", tuple(evidence), arch_distilled)

    lowered = (checkpoint or "").lower()
    hit = next((hint for hint in TURBO_NAME_HINTS if hint in lowered), None)
    if hit:
        evidence.append(f"checkpoint name contains '{hit}'")
        return TurboAssessment("yes", "medium", tuple(evidence), arch_distilled)

    if arch_distilled:
        # The preset already carries the right low-step, CFG-1 numbers, so the
        # question of an *extra* distilled variant barely matters here.
        evidence.append("no separate evidence about this specific checkpoint")
        return TurboAssessment("unknown", "low", tuple(evidence), True)

    if samples >= MIN_CONFIDENT_SAMPLES and observed_steps is not None:
        evidence.append(
            f"{samples} past generations ran at {observed_steps:g} steps, in line with the preset"
        )
        return TurboAssessment("no", "medium", tuple(evidence), False)

    if samples:
        evidence.append(
            f"only {samples} past generation(s) with this checkpoint — too few to read a pattern"
        )
    else:
        evidence.append("no past generations with this checkpoint")
    evidence.append("no accelerator in use and nothing in the name to go on")
    return TurboAssessment("unknown", "low", tuple(evidence), False)


def _prompt_style(checkpoint: str | None, preset: str | None, regime) -> str:
    lineage = detect_lineage(checkpoint or "")
    if lineage:
        return LINEAGE_PROMPT_STYLE[lineage]
    if regime is not None and regime.samples >= 5:
        ratio = regime.danbooru_ratio
        if ratio >= 0.8:
            return f"danbooru tags ({ratio:.0%} of past generations with this checkpoint)"
        if ratio >= 0.3:
            return f"hybrid: tags and natural language ({ratio:.0%} tag-heavy in past generations)"
        return f"natural language ({1 - ratio:.0%} of past generations were prose)"
    return PROMPT_STYLE.get((preset or "").lower(), "unknown - inspect a past generation first")
