"""What the agent is currently working with, and how to prompt it.

The goal is that an agent never has to be told "this is a turbo anima model, use
11 steps and danbooru tags" — it asks once and the answer is derived from the
live instance plus observed usage.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from . import civitai, identity, modules
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
    checkpoint_tags: tuple[str, ...]
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
    module_health: dict
    samples_observed: int
    known_regimes: tuple[dict, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "checkpoint_tags": list(self.checkpoint_tags),
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
            "module_health": self.module_health,
            "samples_observed": self.samples_observed,
            "warnings": list(self.warnings),
        }


def resolve_dialect(
    client: ForgeClient,
    history: HistoryIndex,
    checkpoint: str | None,
    preset: str | None,
    lora_entries: list | None = None,
) -> identity.DialectResolution:
    """Work out how this checkpoint expects to be prompted.

    Tries, in order: a previously cached answer, an optional CivitAI lookup by
    hash, the prompts actually used with this checkpoint, what the architecture
    implies, and finally the shape of the installed LoRA library.
    """
    sha, declared = _checkpoint_identity(client, checkpoint)
    identifier = sha or (checkpoint or "")

    if not declared and sha and civitai.enabled():
        found = civitai.lookup(sha)
        if found.ok:
            declared = found.base_model

    prompts = [regime_prompt for regime_prompt in history.prompts_for(checkpoint or "")]
    return identity.resolve(
        identifier=identifier,
        architecture=preset,
        declared_base=declared,
        observed_prompts=prompts,
        lora_entries=lora_entries,
    )


def _checkpoint_identity(client: ForgeClient, checkpoint: str | None) -> tuple[str | None, str | None]:
    """The loaded checkpoint's hash, and any base model it declares locally."""
    if not checkpoint:
        return None, None
    listing = client.checkpoints()
    if not listing.ok:
        return None, None

    target = os.path.basename(str(checkpoint).replace("\\", "/")).lower()
    for item in listing.value or []:
        title = str(item.get("title") or "")
        filename = os.path.basename(str(item.get("filename") or "").replace("\\", "/"))
        if target not in title.lower() and target != filename.lower():
            continue
        sha = item.get("sha256") or item.get("hash")
        declared = _sidecar_base_model(client, item.get("filename"))
        return (str(sha) if sha else None), declared
    return None, None


def _sidecar_base_model(client: ForgeClient, remote_path) -> str | None:
    """Read a declared baseModel from a sidecar, when one happens to exist."""
    return _sidecar_metadata(client, remote_path)[0]


def _sidecar_metadata(client: ForgeClient, remote_path) -> tuple[str | None, tuple[str, ...]]:
    """A checkpoint's declared base model and its own tags.

    The tags matter beyond bookkeeping: a dialect can say the architecture does
    not do realism while a specific merge within it does. On one 207-checkpoint
    Anima collection, 23% declared realism tags. Reading them lets the caller
    judge the checkpoint instead of only its family.
    """
    if not remote_path:
        return None, ()
    local = client.config.localise(str(remote_path))
    if not local:
        return None, ()
    stem = os.path.splitext(local)[0]
    base: str | None = None
    tags: tuple[str, ...] = ()
    for suffix in (".json", ".api_info.json"):
        candidate = stem + suffix
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        base = base or data.get("baseModel") or data.get("sd version")
        if not tags:
            found = data.get("modelTags") or []
            if isinstance(found, list):
                tags = tuple(str(tag) for tag in found[:12])
    return (str(base) if base else None), tags


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

    health = _module_health(client, data, preset)
    for problem in health.get("problems", ()):
        warnings.append(problem)

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

    _, checkpoint_tags = _sidecar_metadata(client, _checkpoint_file(client, checkpoint))

    return ModelProfile(
        checkpoint=checkpoint,
        checkpoint_tags=checkpoint_tags,
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
        module_health=health,
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



def preset_for_checkpoint(options: dict, checkpoint: str) -> str | None:
    """Which preset last had this checkpoint selected, per the instance.

    Note what this is *not*: evidence of the checkpoint's architecture. Forge
    stores the last checkpoint chosen under each preset, so if it was ever
    selected while the wrong preset was active, the record says so. A live
    instance was observed with a Krea checkpoint recorded under `anima` for
    exactly that reason. Corroborate before acting on it.
    """
    target = _bare_name(checkpoint)
    if not target:
        return None
    for key, value in options.items():
        if not key.startswith("forge_checkpoint_") or not value:
            continue
        if _bare_name(str(value)) == target:
            return key[len("forge_checkpoint_"):]
    return None


def preset_from_folder(options: dict, checkpoint: str) -> str | None:
    """A preset suggested by the folder a checkpoint sits in.

    Independent of the instance's own bookkeeping, which is the point: it gives
    the registry something to agree or disagree with. Only meaningful when the
    folder name matches a preset the instance knows about.
    """
    parts = [part.strip().lower() for part in str(checkpoint or "").replace("\\", "/").split("/")[:-1]]
    known = {
        key[len("forge_checkpoint_"):]
        for key in options
        if key.startswith("forge_checkpoint_")
    }
    aliases = {"z-image": "zit", "zimage": "zit", "krea 2": "krea", "flux.2": "klein"}
    for part in reversed(parts):
        candidate = aliases.get(part, part)
        if candidate in known:
            return candidate
    return None


def _bare_name(value: str) -> str:
    return os.path.basename(str(value or "").replace("\\", "/")).strip().lower()


def switch_checkpoint(client: ForgeClient, name: str, preset: str | None = None) -> dict:
    """Load a checkpoint, bringing its architecture's modules along with it.

    Setting sd_model_checkpoint alone is not enough. Forge builds its loading
    parameters from `forge_additional_modules`, the *currently* selected VAE and
    text encoder, so a checkpoint from another architecture would load against
    the previous architecture's modules. The UI avoids this by changing preset
    and modules together; this does the same.

    The architecture is inferred from two independent signals — which preset
    last had this checkpoint selected, and the folder it lives in. They are
    only acted on when they agree, because neither is reliable alone: a live
    instance was observed with a Krea checkpoint recorded under `anima`.
    Pass `preset` to state it outright and skip the inference.
    """
    options = client.options()
    if not options.ok:
        return {"ok": False, "error": options.error}

    data = options.value or {}
    current_preset = data.get("forge_preset")
    from_registry = preset_for_checkpoint(data, name)
    from_folder = preset_from_folder(data, name)

    notes: list[str] = []
    if preset:
        target_preset, confidence = preset, "stated by caller"
    elif from_registry and from_folder and from_registry != from_folder:
        target_preset, confidence = None, "conflicting"
        notes.append(
            f"cannot tell which architecture this is: it sits in a '{from_folder}' folder but "
            f"the instance last had it selected under '{from_registry}'. Loading it without "
            f"changing preset, so it keeps the modules of '{current_preset}'. Pass the preset "
            "explicitly, or select it once in the UI, to switch modules with it"
        )
    elif from_registry and from_folder:
        target_preset, confidence = from_registry, "registry and folder agree"
    elif from_registry or from_folder:
        target_preset = from_registry or from_folder
        confidence = "single signal only"
        notes.append(
            f"architecture inferred from a single signal ({'preset registry' if from_registry else 'folder name'}); "
            "pass the preset explicitly if this is wrong"
        )
    else:
        target_preset, confidence = None, "unknown"
        notes.append(
            f"nothing identifies this checkpoint's architecture, so the modules of "
            f"'{current_preset}' stay loaded. If it belongs elsewhere, pass the preset"
        )

    payload: dict[str, object] = {"sd_model_checkpoint": name}
    if target_preset and target_preset != current_preset:
        modules = data.get(f"forge_additional_modules_{target_preset}")
        payload["forge_preset"] = target_preset
        if modules is not None:
            payload["forge_additional_modules"] = modules
        notes.append(
            f"architecture changed from '{current_preset}' to '{target_preset}'; "
            "its VAE and text encoder were selected with it"
        )

    result = client.set_options(payload)
    if not result.ok:
        return {"ok": False, "error": result.error, "attempted": payload}

    return {
        "ok": True,
        "loaded": name,
        "preset": target_preset or current_preset,
        "architecture_confidence": confidence,
        "signals": {"preset_registry": from_registry, "folder": from_folder},
        "applied": {key: value for key, value in payload.items() if key != "sd_model_checkpoint"},
        "notes": notes,
        "warning": "instance-wide change; anyone using the web UI sees it too",
    }



def _module_health(client: ForgeClient, options: dict, preset: str | None) -> dict:
    """A quiet check of the preset's modules against the architecture reference.

    The preset is what Forge will actually load, so it is the operating truth
    and this never overrides it. The reference exists to notice when the preset
    has drifted — Forge records the last selection made under a preset, so
    loading a checkpoint while another preset was active leaves that preset's
    modules behind. Silent unless something looks off.
    """
    if not preset:
        return {"checked": False}

    listing = client.modules()
    if not listing.ok:
        return {"checked": False, "error": listing.error}

    selected = options.get(f"forge_additional_modules_{preset}") or []
    report = modules.audit(preset, list(selected), list(listing.value or []))
    if not report.get("known"):
        return {"checked": False, "reason": "no reference for this architecture"}

    problems: list[str] = []
    for gap in report.get("missing", []):
        candidates = gap.get("candidates_installed") or []
        if candidates:
            problems.append(
                f"{preset} has no {gap['need']} selected, but {candidates[0]} is installed and fits"
            )
        else:
            problems.append(f"{preset} needs a {gap['need']} and none is installed")
    if report.get("unrecognised"):
        names = ", ".join(report["unrecognised"]["modules"])
        problems.append(
            f"{preset} has modules loaded that this architecture does not ask for ({names}) — "
            "either a community build named differently, or left over from another preset"
        )

    return {
        "checked": True,
        "healthy": not problems,
        "problems": problems,
        "loaded": [item["loaded"] for item in report.get("satisfied", [])],
    }


def _checkpoint_file(client: ForgeClient, checkpoint: str | None) -> str | None:
    """The on-disk path Forge reports for a checkpoint title."""
    if not checkpoint:
        return None
    listing = client.checkpoints()
    if not listing.ok:
        return None
    target = os.path.basename(str(checkpoint).replace("\\", "/")).lower()
    for item in listing.value or []:
        title = str(item.get("title") or "").lower()
        filename = str(item.get("filename") or "")
        if target in title or target == os.path.basename(filename.replace("\\", "/")).lower():
            return filename
    return None


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
