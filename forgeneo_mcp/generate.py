"""Generation requests and result collection.

The API always answers with base64 images. Returning those to an agent is a
context disaster — a single batch can be tens of megabytes — so when the output
directory is reachable we hand back file paths instead, and only fall back to
decoding base64 when it is not.
"""

from __future__ import annotations

import base64
import binascii
import os
import time
from dataclasses import dataclass
from typing import Any

from .client import ApiResult, ForgeClient

MODELS_MARKER = "/models/"
NEW_FILE_GRACE_SECONDS = 2.0
# Forge also drops a sidecar .txt with the infotext next to each image; the
# caller asked for artwork, not for the log.
MEDIA_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".avif", ".jxl", ".mp4", ".mkv")


@dataclass(frozen=True)
class GenerationResult:
    ok: bool
    files: tuple[str, ...] = ()
    delivery: str = "none"
    info: dict | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        payload: dict[str, Any] = {"ok": self.ok, "delivery": self.delivery}
        if self.files:
            payload["files"] = list(self.files)
        if self.info:
            payload["parameters"] = self.info
        if self.error:
            payload["error"] = self.error
        return payload


def resolve_output_dir(client: ForgeClient, options: dict, mode: str = "txt2img") -> str | None:
    """Find a locally readable path for Forge's output folder.

    Explicit configuration wins. Otherwise the folder is derived by combining
    the (usually relative) outdir option with an installation root inferred from
    any absolute module path the instance reports.
    """
    if client.config.output_dir:
        return client.config.output_dir if os.path.isdir(client.config.output_dir) else None

    key = "outdir_img2img_samples" if mode == "img2img" else "outdir_txt2img_samples"
    outdir = (options.get("outdir_samples") or options.get(key) or "").strip()
    if not outdir:
        return None

    normalised = outdir.replace("\\", "/")
    if _is_absolute(normalised):
        local = client.config.localise(normalised)
        return local if local and _usable_target(local) else None

    root = _installation_root(client, options)
    if not root:
        return None
    candidate = os.path.join(root, *normalised.split("/"))
    return candidate if _usable_target(candidate) else None


def _usable_target(path: str) -> bool:
    """Accept a folder Forge has not created yet.

    Output subfolders appear the first time a mode is used: on a machine that
    had only ever run txt2img, outdir_img2img_samples pointed at a directory
    that did not exist, and requiring it up front discarded the result of a
    generation that had already happened. An existing parent is enough to show
    the path mapping is right.
    """
    if os.path.isdir(path):
        return True
    parent = os.path.dirname(path.rstrip("/" + chr(92)))
    return bool(parent) and os.path.isdir(parent)


def _is_absolute(path: str) -> bool:
    return path.startswith("/") or (len(path) > 1 and path[1] == ":")


def _installation_root(client: ForgeClient, options: dict) -> str | None:
    """Infer the Forge root from any absolute path the instance reports."""
    candidates: list[str] = []
    for key, value in options.items():
        if not key.startswith("forge_additional_modules"):
            continue
        if isinstance(value, list):
            candidates.extend(str(item) for item in value)

    for raw in candidates:
        normalised = str(raw).replace("\\", "/")
        marker = normalised.lower().find(MODELS_MARKER)
        if marker == -1:
            continue
        local = client.config.localise(normalised[:marker])
        if local and os.path.isdir(local):
            return local
    return None


def encode_init_image(path: str) -> tuple[str | None, str | None]:
    """Read a local image into base64 for img2img. Returns (data, error)."""
    if not os.path.isfile(path):
        return None, f"init image not found: {path}"
    if not path.lower().endswith(MEDIA_SUFFIXES):
        return None, f"init image is not a recognised image file: {path}"
    try:
        with open(path, "rb") as handle:
            blob = handle.read()
    except OSError as exc:
        return None, f"cannot read init image: {exc}"
    if not blob:
        return None, f"init image is empty: {path}"
    return base64.b64encode(blob).decode("ascii"), None


def build_payload(
    prompt: str,
    negative_prompt: str = "",
    steps: int | None = None,
    cfg_scale: float | None = None,
    sampler_name: str | None = None,
    scheduler: str | None = None,
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    batch_size: int = 1,
    distilled_cfg_scale: float | None = None,
    extra: dict | None = None,
) -> dict:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "batch_size": batch_size,
        "save_images": True,
        "send_images": True,
    }
    if steps is not None:
        payload["steps"] = steps
    if cfg_scale is not None:
        payload["cfg_scale"] = cfg_scale
    if distilled_cfg_scale is not None:
        # Forge feeds this into set_shift(); the UI labels it "Shift" or
        # "Distilled CFG Scale" per architecture. Omitting it does not mean
        # "use the architecture default" - it means the API's own 3.5.
        payload["distilled_cfg_scale"] = distilled_cfg_scale
    if sampler_name:
        payload["sampler_name"] = sampler_name
    if scheduler:
        payload["scheduler"] = scheduler
    if extra:
        payload.update(extra)
    return payload


def run_generation(
    client: ForgeClient,
    payload: dict,
    output_dir: str | None,
    mode: str = "txt2img",
    fallback_dir: str | None = None,
) -> GenerationResult:
    before = _snapshot(output_dir)
    started = time.time()

    result: ApiResult = client.img2img(payload) if mode == "img2img" else client.txt2img(payload)
    if not result.ok:
        return GenerationResult(False, error=result.error)

    data = result.value or {}
    info = _decode_info(data.get("info"))

    if output_dir:
        files = _new_files(output_dir, before, started)
        if files:
            return GenerationResult(True, files=files, delivery="filesystem", info=info)

    images = data.get("images") or []
    if not images:
        return GenerationResult(False, error="Forge returned no images", info=info)
    if not fallback_dir:
        return GenerationResult(
            False,
            error="images were generated but no readable output directory is configured",
            info=info,
        )
    saved = _save_base64(images, fallback_dir)
    if not saved:
        return GenerationResult(False, error="could not decode returned images", info=info)
    return GenerationResult(True, files=saved, delivery="base64", info=info)


def _snapshot(output_dir: str | None) -> set[str]:
    if not output_dir:
        return set()
    found: set[str] = set()
    for root, _, files in os.walk(output_dir):
        for name in files:
            found.add(os.path.join(root, name))
    return found


def _new_files(output_dir: str, before: set[str], started: float) -> tuple[str, ...]:
    fresh: list[tuple[float, str]] = []
    for root, _, files in os.walk(output_dir):
        for name in files:
            if not name.lower().endswith(MEDIA_SUFFIXES):
                continue
            full = os.path.join(root, name)
            if full in before:
                continue
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            if mtime >= started - NEW_FILE_GRACE_SECONDS:
                fresh.append((mtime, full))
    fresh.sort()
    return tuple(_normalise(path) for _, path in fresh)


def _normalise(path: str) -> str:
    """Present one separator style; UNC roots keep their leading pair."""
    unc = path.startswith("//") or path.startswith(chr(92) * 2)
    cleaned = path.replace(chr(92), "/")
    return "//" + cleaned.lstrip("/") if unc else cleaned


def _save_base64(images: list, target_dir: str) -> tuple[str, ...]:
    os.makedirs(target_dir, exist_ok=True)
    stamp = int(time.time())
    saved: list[str] = []
    for index, encoded in enumerate(images):
        if not isinstance(encoded, str):
            continue
        payload = encoded.split(",", 1)[-1]
        try:
            blob = base64.b64decode(payload)
        except (binascii.Error, ValueError):
            continue
        path = os.path.join(target_dir, f"forgeneo-{stamp}-{index}.png")
        try:
            with open(path, "wb") as handle:
                handle.write(blob)
        except OSError:
            continue
        saved.append(path)
    return tuple(saved)


def _decode_info(raw: Any) -> dict | None:
    if isinstance(raw, dict):
        return _trim_info(raw)
    if isinstance(raw, str):
        try:
            import json

            return _trim_info(json.loads(raw))
        except ValueError:
            return None
    return None


def _trim_info(info: dict) -> dict:
    """Keep the fields an agent needs; drop the rest to protect the context."""
    keys = (
        "prompt",
        "negative_prompt",
        "seed",
        "all_seeds",
        "steps",
        "cfg_scale",
        "sampler_name",
        "scheduler",
        "width",
        "height",
        "sd_model_name",
    )
    return {key: info[key] for key in keys if key in info}
