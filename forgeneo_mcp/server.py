"""MCP server exposing Forge Neo to an agent.

Design rule: the tools are faithful, not clever. `generate` sends exactly the
prompt it is given and never injects a LoRA on its own — discovery lives in
`loras`, and the decision to use one belongs to the agent that called it.
"""

from __future__ import annotations

import logging
from typing import Any

# httpx logs one INFO line per request. Over stdio that lands in the client's
# MCP log as noise, one entry per API call, so keep it to real problems.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

try:  # SDK 2.x renamed FastMCP to MCPServer; both expose the same decorators
    from mcp.server import MCPServer as _ServerClass
except ImportError:  # pragma: no cover - depends on installed SDK
    from mcp.server.fastmcp import FastMCP as _ServerClass

from .capabilities import probe
from .client import ForgeClient
from .config import Config
from .generate import build_payload, encode_init_image, resolve_output_dir, run_generation
from .history import HistoryIndex
from .loras import LoraIndex
from .profile import build_profile

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
def models(action: str = "list", name: str = "", query: str = "", limit: int = 30) -> dict:
    """List or load checkpoints. action: "list" | "load" | "refresh".

    Loading swaps the model for the whole instance, including any human using
    the web UI at the same time, and takes several seconds — only do it when the
    operator asked for that model."""
    action = action.lower().strip()
    if action == "refresh":
        result = _client.post("/sdapi/v1/refresh-checkpoints", {})
        return {"ok": result.ok, "error": result.error}

    if action == "load":
        if not name:
            return {"ok": False, "error": "name is required to load a checkpoint"}
        result = _client.set_options({"sd_model_checkpoint": name})
        if not result.ok:
            return {"ok": False, "error": result.error}
        return {"ok": True, "loaded": name, "note": "instance-wide change; affects the web UI too"}

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
def generate(
    prompt: str,
    negative_prompt: str = "",
    steps: int | None = None,
    cfg_scale: float | None = None,
    sampler_name: str = "",
    scheduler: str = "",
    width: int = 1024,
    height: int = 1024,
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
    override. Returns file paths when the output folder is readable.

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
        width=width,
        height=height,
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
