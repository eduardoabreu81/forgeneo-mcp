"""Which VAE and text encoders each architecture needs.

Source: the Forge Classic wiki's Download Models page, which is the reference
for what to fetch and where it goes:
https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models

This table exists because the instance cannot answer the question. Forge stores
`forge_additional_modules_<arch>`, but that records the last selection made
under a preset, not what the architecture requires — loading a checkpoint while
the wrong preset is active overwrites it. Both that field and
`forge_checkpoint_<arch>` have been observed carrying another architecture's
values for exactly that reason.

Requirements are expressed as name patterns, never as exact filenames. A single
architecture is commonly served by several legitimate builds — quantised,
rescaled, or community-improved — and pinning filenames would reject all but
one of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VAE = "vae"
TEXT_ENCODER = "text_encoder"


@dataclass(frozen=True)
class ModuleRequirement:
    kind: str
    label: str
    patterns: tuple[str, ...]
    optional: bool = False

    def matches(self, name: str) -> bool:
        """Substring match, or anchored with "^" where a substring is too loose.

        "ae" is the Flux VAE's actual filename and also sits inside every other
        "*vae*" on disk, so it has to be anchored or it claims all of them.
        """
        lowered = name.lower()
        for pattern in self.patterns:
            if pattern.startswith("^"):
                if lowered.startswith(pattern[1:]):
                    return True
            elif pattern in lowered:
                return True
        return False


@dataclass(frozen=True)
class ArchModules:
    """What an architecture needs loaded alongside its checkpoint."""

    requirements: tuple[ModuleRequirement, ...] = field(default_factory=tuple)
    # Set where the wiki does not state a VAE. Reported as unknown rather than
    # guessed, since a wrong VAE degrades output without erroring.
    vae_unspecified: bool = False
    note: str = ""


# Architectures whose text encoder ships inside the checkpoint; a VAE can be
# supplied but the model loads without one.
_SELF_CONTAINED = ArchModules(note="text encoder is baked into the checkpoint; a VAE is optional")

ARCH_MODULES: dict[str, ArchModules] = {
    "sd": _SELF_CONTAINED,
    "xl": _SELF_CONTAINED,
    "flux": ArchModules(
        requirements=(
            ModuleRequirement(VAE, "Flux VAE (ae)", ("^ae.", "^ae_", "flux_vae", "flux-vae")),
            ModuleRequirement(TEXT_ENCODER, "CLIP-L", ("clip_l", "clip-l")),
            ModuleRequirement(TEXT_ENCODER, "T5-XXL", ("t5xxl", "t5_xxl", "t5-xxl")),
        ),
    ),
    "qwen": ArchModules(
        requirements=(
            ModuleRequirement(VAE, "Qwen-Image VAE", ("qwen_image_vae", "qwenimagevae", "qwen2d_vae")),
            ModuleRequirement(TEXT_ENCODER, "Qwen2.5-VL 7B", ("qwen_2.5_vl", "qwen2.5_vl", "qwen25vl")),
        ),
        note="the GGUF text encoder also needs its mmproj file for img2img",
    ),
    "wan": ArchModules(
        requirements=(
            ModuleRequirement(VAE, "Wan 2.1 VAE", ("wan_2.1_vae", "wan21_vae", "wan_vae")),
            ModuleRequirement(TEXT_ENCODER, "UMT5-XXL", ("umt5", "um_t5")),
        ),
    ),
    "zit": ArchModules(
        requirements=(ModuleRequirement(TEXT_ENCODER, "Qwen3 4B", ("qwen_3_4b", "qwen3_4b", "qwen3-4b")),),
        vae_unspecified=True,
    ),
    "krea": ArchModules(
        requirements=(ModuleRequirement(TEXT_ENCODER, "Qwen3-VL 4B", ("qwen3vl_4b", "qwen3_vl_4b", "qwen3vl")),),
        vae_unspecified=True,
    ),
    "pid": ArchModules(
        requirements=(ModuleRequirement(TEXT_ENCODER, "Gemma 2 2B IT", ("gemma_2_2b", "gemma2_2b", "gemma-2-2b")),),
        vae_unspecified=True,
    ),
    # Anima's own model card names both files it expects.
    "anima": ArchModules(
        requirements=(
            ModuleRequirement(VAE, "Qwen-Image VAE", ("qwen_image_vae", "qwenimagevae", "qwen2d_vae", "vaeanima")),
            ModuleRequirement(TEXT_ENCODER, "Qwen3 0.6B base", ("qwen_3_06b", "qwen3_06b", "qwen_3_0.6b")),
        ),
        note="per the CircleStone Labs model card for Anima",
    ),
}


def requirements_for(arch: str | None) -> ArchModules | None:
    if not arch:
        return None
    return ARCH_MODULES.get(arch.lower())


def classify(filename: str) -> str | None:
    """VAE or text encoder, from the folder Forge keeps it in."""
    lowered = str(filename or "").replace("\\", "/").lower()
    if "/text_encoder" in lowered:
        return TEXT_ENCODER
    if "/vae" in lowered:
        return VAE
    return None


def audit(arch: str | None, selected: list[str], available: list[dict]) -> dict:
    """Compare what a preset has loaded against what its architecture needs.

    `available` is the /sdapi/v1/sd-modules listing. Returns what is satisfied,
    what is missing, and which installed files could fill each gap — candidates
    to offer, not a choice to make: several builds of the same module are all
    valid, and picking between them is the operator's call.
    """
    spec = requirements_for(arch)
    if spec is None:
        return {"known": False, "architecture": arch}

    pool = [
        {"name": str(item.get("model_name") or ""), "kind": classify(str(item.get("filename") or ""))}
        for item in available
        if isinstance(item, dict)
    ]
    selected_names = [str(name).replace("\\", "/").split("/")[-1] for name in selected]

    satisfied, missing = [], []
    for requirement in spec.requirements:
        hit = next((name for name in selected_names if requirement.matches(name)), None)
        if hit:
            satisfied.append({"need": requirement.label, "kind": requirement.kind, "loaded": hit})
            continue
        candidates = [
            item["name"]
            for item in pool
            if item["kind"] == requirement.kind and requirement.matches(item["name"])
        ]
        missing.append(
            {
                "need": requirement.label,
                "kind": requirement.kind,
                "candidates_installed": candidates,
                "action": "select one" if candidates else "download one",
            }
        )

    report = {
        "known": True,
        "architecture": arch,
        "satisfied": satisfied,
        "missing": missing,
        "currently_selected": selected_names,
    }
    if spec.note:
        report["note"] = spec.note
    if spec.vae_unspecified:
        vaes = [item["name"] for item in pool if item["kind"] == VAE]
        report["vae"] = {
            "status": "not specified by the Forge wiki for this architecture",
            "installed_vaes": vaes,
            "advice": "ask the operator which VAE they use here rather than assuming one",
        }
    # With no VAE requirement to compare against, a selected VAE cannot be
    # called wrong - the reference simply does not say what belongs here.
    kinds_specified = {requirement.kind for requirement in spec.requirements}
    selected_kinds = {
        item["name"]: item["kind"]
        for item in pool
        if item["name"] in selected_names
    }
    # A module absent from the listing has no known kind, and guessing one
    # produces false accusations. Only flag what can actually be classified.
    unexpected = [
        name
        for name in selected_names
        if not any(requirement.matches(name) for requirement in spec.requirements)
        and selected_kinds.get(name) in kinds_specified
    ]
    if unexpected and spec.requirements:
        report["unrecognised"] = {
            "modules": unexpected,
            "why": (
                "these match nothing this architecture asks for. Two innocent explanations exist "
                "before assuming a mistake: a community build named outside the usual convention "
                "is still perfectly valid, and matching is by name. The one worth checking is "
                "that Forge records the last selection made under a preset, so loading a "
                "checkpoint while another preset was active leaves that preset's modules here"
            ),
            "check": "confirm with the operator rather than treating this as an error",
        }
    return report
