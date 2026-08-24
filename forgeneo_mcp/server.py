"""MCP server exposing Forge Neo to an agent.

Design rule: the tools are faithful, not clever. `generate` sends exactly the
prompt it is given and never injects a LoRA on its own — discovery lives in
`loras`, and the decision to use one belongs to the agent that called it.
"""

from __future__ import annotations

import logging
import os
from typing import Any

# httpx logs one INFO line per request. Over stdio that lands in the client's
# MCP log as noise, one entry per API call, so keep it to real problems.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

try:  # SDK 2.x renamed FastMCP to MCPServer; both expose the same decorators
    from mcp.server import MCPServer as _ServerClass
except ImportError:  # pragma: no cover - depends on installed SDK
    from mcp.server.fastmcp import FastMCP as _ServerClass

from . import dialects, downloads, fetcher, identity, modules
from .capabilities import probe
from .client import ForgeClient
from .config import Config
from .generate import build_payload, encode_init_image, resolve_output_dir, run_generation
from .history import HistoryIndex
from .loras import LoraIndex
from .profile import (
    _checkpoint_identity,
    _module_health,
    build_profile,
    resolve_dialect,
    switch_checkpoint,
)

mcp = _ServerClass("forgeneo")

_config = Config.from_env()
_client = ForgeClient(_config)
_history = HistoryIndex(_config.output_dir, limit=_config.history_limit)
_loras = LoraIndex(_client, _history)
_output_dir: str | None = None
_output_resolved = False


def _ensure_output_dir() -> str | None:
    """Resolve the output directory once, lazily, from the live instance."""
    global _output_dir, _output_resolved
    if _output_resolved:
        return _output_dir
    _output_resolved = True
    options = _client.options()
    if options.ok:
        _output_dir = resolve_output_dir(_client, options.value or {})
        if _output_dir and not _history.available:
            _history.set_output_dir(_output_dir)
    return _output_dir


@mcp.tool()
def capabilities() -> dict:
    """Report what this Forge instance offers: routes, counts, and which
    metadata sources are available. Call this first in a session."""
    return probe(_client, _history, _loras, _ensure_output_dir()).as_dict()


@mcp.tool()
def model_profile() -> dict:
    """Describe the currently loaded checkpoint: architecture preset, whether it
    behaves as a turbo/distilled model, the sampling parameters that actually
    worked before, the expected prompt dialect, and whether its VAE and text
    encoder modules exist. Call before writing a prompt for an unfamiliar model."""
    _ensure_output_dir()
    result = build_profile(_client, _history)
    if isinstance(result, str):
        return {"ok": False, "error": result}
    return {"ok": True, **result.as_dict()}


@mcp.tool()
def prompt_dialect(confirm: str = "") -> dict:
    """How the loaded checkpoint expects to be prompted, with its quality tags.

    Returns the dialect (pony / illustrious / animagine / anima / sd15 /
    sdxl_base / natural), the quality prefix and negative baseline it needs, and
    where that conclusion came from. Quality tags are not decoration: an
    Illustrious prompt without them degrades, and a Flux prompt with them
    degrades too.

    When the dialect comes back unknown — `xl` covers Pony, Illustrious and
    stock SDXL, which share tensors and preset — ask the operator, then call
    again with `confirm` set to their answer. It is cached by file hash and
    never asked again."""
    _ensure_output_dir()
    options = _client.options()
    if not options.ok:
        return {"ok": False, "error": options.error}

    data = options.value or {}
    checkpoint = data.get("sd_model_checkpoint")
    preset = data.get("forge_preset")

    if confirm:
        key = confirm.strip().lower()
        if key not in dialects.BY_KEY:
            return {
                "ok": False,
                "error": f"unknown dialect '{confirm}'",
                "choices": sorted(dialects.BY_KEY),
            }
        sha, _ = _checkpoint_identity(_client, checkpoint)
        stored = identity.remember(sha or (checkpoint or ""), key)
        return {
            "ok": True,
            "confirmed": key,
            "cached": stored,
            **dialects.BY_KEY[key].as_dict(),
        }

    resolution = resolve_dialect(_client, _history, checkpoint, preset, _loras.all())
    return {"ok": True, "checkpoint": checkpoint, "architecture": preset, **resolution.as_dict()}


@mcp.tool()
def loras(query: str = "", base_model: str = "", kind: str = "", limit: int = 20, verbose: bool = False) -> dict:
    """Search available LoRAs by name, title, tags, trigger words or description.

    Only call this when the request actually calls for one (a named style,
    character, or concept) — most generations need no LoRA at all. `kind` can be
    "content" or "accelerator"; accelerators change the sampling regime rather
    than the image, so adopting one means adjusting steps and CFG together."""
    _ensure_output_dir()
    limit = max(1, min(int(limit), 50))
    matches = _loras.search(query=query, base_model=base_model or None, kind=kind or None, limit=limit)
    return {
        "ok": _loras.error is None,
        "error": _loras.error,
        "query": query,
        "returned": len(matches),
        "summary": _loras.summary(),
        "results": [entry.as_dict(verbose=verbose) for entry in matches],
    }


@mcp.tool()
def lora_info(name: str) -> dict:
    """Full detail for one LoRA, including description, tags, past usage and a
    ready-to-paste prompt fragment with its trigger words."""
    _ensure_output_dir()
    entry = _loras.get(name)
    if not entry:
        return {"ok": False, "error": f"no LoRA named '{name}'"}
    return {"ok": True, **entry.as_dict(verbose=True)}


@mcp.tool()
def models(action: str = "list", name: str = "", preset: str = "", query: str = "", limit: int = 30) -> dict:
    """List or load checkpoints. action: "list" | "load" | "refresh".

    Loading swaps the model for the whole instance, including any human using
    the web UI at the same time, and takes several seconds — only do it when the
    operator asked for that model. When the target belongs to a different
    architecture, its preset, VAE and text encoder are switched with it, since
    Forge would otherwise load it against whatever modules are selected now. The
    architecture is inferred from two signals and only acted on when they agree;
    pass `preset` to state it outright."""
    action = action.lower().strip()
    if action == "refresh":
        result = _client.post("/sdapi/v1/refresh-checkpoints", {})
        return {"ok": result.ok, "error": result.error}

    if action == "load":
        if not name:
            return {"ok": False, "error": "name is required to load a checkpoint"}
        return switch_checkpoint(_client, name, preset or None)

    if action != "list":
        return {"ok": False, "error": f"unknown action '{action}'"}

    result = _client.checkpoints()
    if not result.ok:
        return {"ok": False, "error": result.error}

    entries = result.value or []
    needle = query.lower().strip()
    filtered = [
        {"title": item.get("title"), "model_name": item.get("model_name")}
        for item in entries
        if not needle or needle in str(item.get("title", "")).lower()
    ]
    return {
        "ok": True,
        "total": len(entries),
        "returned": len(filtered[:limit]),
        "results": filtered[: max(1, min(int(limit), 100))],
    }


@mcp.tool()
def module_check(preset: str = "") -> dict:
    """Check the VAE and text encoders loaded for an architecture against what
    it actually needs, and list installed files that could fill any gap.

    Defaults to the active preset. Worth calling after switching architecture or
    when output looks wrong for no obvious reason: Forge records the last
    selection made under a preset, so loading a checkpoint while another preset
    was active can leave the wrong modules attached. Where the reference does
    not state a VAE, it says so instead of guessing — a wrong VAE degrades
    output without raising an error."""
    options = _client.options()
    if not options.ok:
        return {"ok": False, "error": options.error}

    data = options.value or {}
    arch = (preset or data.get("forge_preset") or "").strip()
    selected = data.get(f"forge_additional_modules_{arch}") or []

    listing = _client.modules()
    if not listing.ok:
        return {"ok": False, "error": listing.error}

    report = modules.audit(arch, list(selected), list(listing.value or []))
    if not report.get("known"):
        return {
            "ok": True,
            "architecture": arch,
            "known": False,
            "detail": "no module requirements recorded for this architecture",
            "currently_selected": [str(name).replace(chr(92), "/").split("/")[-1] for name in selected],
        }
    return {"ok": True, **report}



@mcp.tool()
def module_download(preset: str = "", label: str = "", confirm: bool = False) -> dict:
    """Find, and optionally fetch, a VAE or text encoder the architecture needs.

    Called with no arguments it lists what the active preset is missing and
    where each file comes from, downloading nothing. Downloading requires both a
    `label` naming one entry and `confirm=True`, and the operator has to agree
    first: these are multi-gigabyte files written into their models folder,
    often across a network share.

    Links come from the Forge Classic wiki's Download Models page. Where several
    builds exist — bf16, fp8_scaled, gguf — they are all offered, because which
    to take depends on the operator's hardware, not on a default worth hiding."""
    options = _client.options()
    if not options.ok:
        return {"ok": False, "error": options.error}

    data = options.value or {}
    arch = (preset or data.get("forge_preset") or "").strip()
    available = downloads.for_architecture(arch)
    if not available:
        return {"ok": True, "architecture": arch, "available": [], "detail": "no catalogue entry"}

    root = _forge_models_root(data)

    if not label:
        health = _module_health(_client, data, arch)
        gaps = []
        for problem in health.get("problems", []):
            gaps.append(problem)
        return {
            "ok": True,
            "architecture": arch,
            "current_state": "healthy" if health.get("healthy") else "incomplete",
            "problems": gaps,
            "available": [entry.as_dict() for entry in available],
            "forge_root": root,
            "how_to_download": (
                "ask the operator which build they want, then call again with that label and "
                "confirm=True"
            ),
        }

    chosen = next((entry for entry in available if entry.label.lower() == label.strip().lower()), None)
    if chosen is None:
        return {
            "ok": False,
            "error": f"no entry called '{label}' for {arch}",
            "choices": [entry.label for entry in available],
        }

    if not root:
        return {
            "ok": False,
            "error": (
                "cannot locate the Forge models folder from here. Set FORGE_PATH_MAP, or download "
                f"{chosen.direct_url} into {chosen.target_folder} manually"
            ),
        }

    if chosen.is_directory:
        return {
            "ok": True,
            "fetchable": False,
            "entry": chosen.as_dict(),
            "detail": (
                "this entry links a folder of builds rather than one file, so it cannot be "
                "fetched automatically. Open the page, choose the build that suits the hardware, "
                f"and save it into {chosen.target_folder}"
            ),
        }

    folder = os.path.join(root, *chosen.target_folder.split("/"))
    intent = fetcher.plan(chosen.direct_url, folder, chosen.filename)

    if not confirm:
        return {
            "ok": True,
            "would_download": chosen.as_dict(),
            "plan": intent.as_dict(),
            "confirmed": False,
            "detail": "nothing was downloaded; pass confirm=True once the operator agrees",
        }

    result = fetcher.fetch(chosen.direct_url, folder, chosen.filename)
    return {"architecture": arch, "label": chosen.label, **result}


def _forge_models_root(options: dict) -> str | None:
    """The locally reachable Forge root, derived from a module path.

    Returns the installation root, not the models folder: catalogue entries
    carry their own "models/VAE" style target, so returning the models folder
    here produced models/models/VAE.
    """
    for key, value in options.items():
        if not key.startswith("forge_additional_modules") or not isinstance(value, list):
            continue
        for item in value:
            normalised = str(item).replace(chr(92), "/")
            marker = normalised.lower().find("/models/")
            if marker == -1:
                continue
            local = _client.config.localise(normalised[:marker])
            if local and os.path.isdir(local):
                return local
    return None


@mcp.tool()
def generate(
    prompt: str,
    negative_prompt: str = "",
    steps: int | None = None,
    cfg_scale: float | None = None,
    sampler_name: str = "",
    scheduler: str = "",
    shift: float | None = None,
    width: int = 0,
    height: int = 0,
    seed: int = -1,
    batch_size: int = 1,
    init_image: str = "",
    denoising_strength: float = 0.7,
    use_profile_defaults: bool = True,
) -> dict:
    """Generate an image from an already-written prompt.

    The prompt is sent verbatim: include any `<lora:name:weight>` yourself. With
    use_profile_defaults on, missing sampling parameters are filled from what the
    loaded model actually used before, so leave them unset unless you mean to
    override. That includes `shift` (Forge's distilled_cfg_scale) and the
    dimensions: leaving them at 0 takes the architecture's own values instead of
    a generic default. Returns file paths when the output folder is readable.

    Pass `init_image` (a local file path) to run img2img instead, where
    `denoising_strength` controls how far the result may drift from it: around
    0.3 keeps the composition, 0.75 reinterprets it freely. Edit-style and video
    models expect values close to 1.0."""
    if not prompt.strip():
        return {"ok": False, "error": "prompt is empty"}

    mode = "img2img" if init_image else "txt2img"
    extra: dict[str, Any] = {}
    if init_image:
        encoded, error = encode_init_image(init_image)
        if error:
            return {"ok": False, "error": error}
        extra["init_images"] = [encoded]
        extra["denoising_strength"] = max(0.0, min(float(denoising_strength), 1.0))

    resolved: dict[str, Any] = {
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler_name or None,
        "scheduler": scheduler or None,
        "distilled_cfg_scale": shift,
        "width": width or None,
        "height": height or None,
    }
    applied_from_profile: list[str] = []

    if use_profile_defaults and any(value is None for value in resolved.values()):
        _ensure_output_dir()
        profile = build_profile(_client, _history)
        if not isinstance(profile, str):
            for key, value in (
                ("steps", int(profile.steps) if profile.steps is not None else None),
                ("cfg_scale", profile.cfg),
                ("sampler_name", profile.sampler),
                ("scheduler", profile.scheduler),
                ("distilled_cfg_scale", profile.shift),
                ("width", profile.width),
                ("height", profile.height),
            ):
                if resolved[key] is None and value is not None:
                    resolved[key] = value
                    applied_from_profile.append(key)

    payload = build_payload(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=resolved["steps"],
        cfg_scale=resolved["cfg_scale"],
        sampler_name=resolved["sampler_name"],
        scheduler=resolved["scheduler"],
        distilled_cfg_scale=resolved["distilled_cfg_scale"],
        width=resolved["width"] or 1024,
        height=resolved["height"] or 1024,
        seed=seed,
        batch_size=max(1, int(batch_size)),
        extra=extra or None,
    )

    _ensure_output_dir()
    options = _client.options()
    output_dir = resolve_output_dir(_client, options.value or {}, mode=mode) if options.ok else None
    result = run_generation(_client, payload, output_dir, mode=mode)
    response = result.as_dict()
    response["mode"] = mode
    if applied_from_profile:
        response["defaults_applied"] = applied_from_profile
    return response


@mcp.tool()
def progress(action: str = "status") -> dict:
    """Check or stop the current generation. action: "status" | "interrupt" | "skip"."""
    action = action.lower().strip()
    if action == "interrupt":
        result = _client.interrupt()
        return {"ok": result.ok, "error": result.error}
    if action == "skip":
        result = _client.skip()
        return {"ok": result.ok, "error": result.error}

    result = _client.progress()
    if not result.ok:
        return {"ok": False, "error": result.error}
    data = result.value or {}
    state = data.get("state") or {}
    return {
        "ok": True,
        "progress": round(float(data.get("progress") or 0.0), 3),
        "eta_seconds": round(float(data.get("eta_relative") or 0.0), 1),
        "job": state.get("job"),
        "step": state.get("sampling_step"),
        "total_steps": state.get("sampling_steps"),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
