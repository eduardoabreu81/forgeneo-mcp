"""Environment probe.

Nothing in this bridge assumes a particular installation. The probe reports what
this specific instance offers so the agent can adapt instead of failing: routes
are read from the live OpenAPI document rather than a hardcoded list, because
extensions register their own, and a route being listed does not mean it works
(the development instance served /sdapi/v1/cmd-flags as a 500).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .client import ForgeClient
from .history import HistoryIndex
from .loras import LoraIndex

PROBED_ROUTES = ("/sdapi/v1/txt2img", "/sdapi/v1/img2img", "/sdapi/v1/loras", "/sdapi/v1/options")


@dataclass(frozen=True)
class Capabilities:
    reachable: bool
    url: str
    error: str | None = None
    checkpoint: str | None = None
    preset: str | None = None
    counts: dict | None = None
    routes: dict | None = None
    filesystem: dict | None = None
    loras: dict | None = None
    history: dict | None = None

    def as_dict(self) -> dict:
        if not self.reachable:
            return {"reachable": False, "url": self.url, "error": self.error}
        return {
            "reachable": True,
            "url": self.url,
            "checkpoint": self.checkpoint,
            "preset": self.preset,
            "counts": self.counts,
            "routes": self.routes,
            "filesystem": self.filesystem,
            "loras": self.loras,
            "history": self.history,
        }


def probe(client: ForgeClient, history: HistoryIndex, lora_index: LoraIndex, output_dir: str | None) -> Capabilities:
    options = client.options()
    if not options.ok:
        return Capabilities(reachable=False, url=client.config.url, error=options.error)

    data = options.value or {}
    counts = {
        "checkpoints": _count(client.checkpoints()),
        "loras": _count(client.loras()),
        "samplers": _count(client.samplers()),
        "schedulers": _count(client.schedulers()),
        "modules": _count(client.modules()),
    }

    history.build()
    lora_index.build()

    return Capabilities(
        reachable=True,
        url=client.config.url,
        checkpoint=data.get("sd_model_checkpoint"),
        preset=data.get("forge_preset"),
        counts=counts,
        routes=_routes(client),
        filesystem={
            "output_dir": output_dir,
            "readable": bool(output_dir and os.path.isdir(output_dir)),
            "path_map_entries": len(client.config.path_map),
            "delivery": "file paths" if output_dir else "base64 (no readable output dir)",
        },
        loras=lora_index.summary() if not lora_index.error else {"error": lora_index.error},
        history=_history_report(history, data),
    )


def _history_report(history: HistoryIndex, options: dict) -> dict:
    """History status, plus why it may be empty.

    Both metadata carriers are optional Forge settings. With neither enabled,
    outputs are just pixels and no sampling regime can ever be measured — the
    agent should be told that rather than left to infer it from a zero.
    """
    pnginfo = options.get("enable_pnginfo")
    save_txt = options.get("save_txt")
    report = {
        "available": history.available,
        "metadata_settings": {"enable_pnginfo": pnginfo, "save_txt": save_txt},
        **history.diagnostics(),
        "top_loras": [{"name": name, "uses": uses} for name, uses in history.top_loras(5)],
    }
    if pnginfo is False and save_txt is False:
        report["note"] = (
            "both enable_pnginfo and save_txt are disabled, so generated files carry no "
            "parameters; sampling guidance will fall back to architecture presets"
        )
    elif history.available and history.scanned == 0:
        report["note"] = "output folder is readable but no file carried generation parameters"
    return report


def _count(result) -> int | str:
    if not result.ok:
        return f"unavailable ({result.error})"
    value = result.value
    return len(value) if isinstance(value, (list, dict)) else "unknown"


def _routes(client: ForgeClient) -> dict:
    spec = client.openapi()
    if not spec.ok:
        return {"error": spec.error}
    paths = sorted((spec.value or {}).get("paths", {}))
    sdapi = [path for path in paths if path.startswith("/sdapi/")]
    return {
        "sdapi": len(sdapi),
        "total": len(paths),
        "required_present": {route: route in paths for route in PROBED_ROUTES},
        "extension_namespaces": sorted(
            {path.split("/")[1] for path in paths if path.count("/") > 1 and not path.startswith("/sdapi")}
        )[:12],
    }
