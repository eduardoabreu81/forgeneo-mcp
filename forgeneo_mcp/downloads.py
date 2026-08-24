"""Where to obtain the VAE and text encoders each architecture needs.

Links come from the Forge Classic wiki's Download Models page, which is the
project's own reference for what to fetch and where it goes:
https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models

Nothing here downloads on its own. Every entry is a suggestion the operator has
to approve: these are multi-gigabyte files landing in someone's models folder,
often over a network share, and the machine may not have room. The tool that
uses this catalogue reports what it would fetch and stops there unless told
otherwise.

Where the wiki offers several builds of the same module, the first entry is the
plain one and the rest are quantised or rescaled variants. Which to take is the
operator's call, not a default worth hiding.
"""

from __future__ import annotations

from dataclasses import dataclass

from .modules import TEXT_ENCODER, VAE

HF_BLOB = "/blob/"
HF_RESOLVE = "/resolve/"
HF_TREE = "/tree/"


@dataclass(frozen=True)
class Download:
    label: str
    kind: str
    url: str
    filename: str
    note: str = ""

    @property
    def is_directory(self) -> bool:
        """Some wiki links point at a folder of builds rather than one file.

        Those cannot be fetched automatically — the operator picks the build
        that suits their hardware from the page.
        """
        return HF_TREE in self.url

    @property
    def direct_url(self) -> str:
        """Hugging Face blob pages are viewers; the file itself is under /resolve/."""
        return self.url.replace(HF_BLOB, HF_RESOLVE, 1)

    @property
    def target_folder(self) -> str:
        return "models/VAE" if self.kind == VAE else "models/text_encoder"

    def as_dict(self) -> dict:
        payload = {
            "label": self.label,
            "kind": self.kind,
            "url": self.direct_url,
            "page": self.url,
            "target_folder": self.target_folder,
        }
        if not self.is_directory:
            payload["filename"] = self.filename
        else:
            payload["fetchable"] = False
            payload["why"] = (
                "this link is a folder of builds, not a single file. Open it, pick the build that "
                f"suits your hardware, and save it into {self.target_folder}"
            )
        if self.note:
            payload["note"] = self.note
        return payload


def _hf(label: str, kind: str, url: str, note: str = "") -> Download:
    return Download(label, kind, url, url.rsplit("/", 1)[-1], note)


_FLUX_AE = _hf(
    "Flux VAE (ae)", VAE,
    "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/blob/main/split_files/vae/ae.safetensors",
)
_QWEN_IMAGE_VAE = (
    _hf(
        "Qwen-Image VAE", VAE,
        "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/blob/main/split_files/vae/qwen_image_vae.safetensors",
    ),
    _hf(
        "Qwen2D VAE (alternative)", VAE,
        "https://huggingface.co/Anzhc/Qwen2D-VAE/blob/main/Qwen2D_VAE.safetensors",
    ),
)

CATALOGUE: dict[str, tuple[Download, ...]] = {
    "sd": (
        _hf(
            "SD1 VAE", VAE,
            "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors",
        ),
    ),
    "xl": (
        _hf(
            "SDXL VAE (fp16-fix)", VAE,
            "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors",
            "shared by every SDXL lineage: Pony, Illustrious/NoobAI, Animagine and stock SDXL",
        ),
    ),
    "lumina": (
        _FLUX_AE,
        _hf(
            "Gemma 2 2B (fp16)", TEXT_ENCODER,
            "https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0/blob/main/Text_Encoder/gemma_2_2b_fp16.safetensors",
        ),
    ),
    "flux": (
        _FLUX_AE,
        _hf(
            "CLIP-L", TEXT_ENCODER,
            "https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors",
        ),
        _hf(
            "T5-XXL (fp16)", TEXT_ENCODER,
            "https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors",
            "fp8_scaled is the lighter option",
        ),
        _hf(
            "T5-XXL (fp8_scaled)", TEXT_ENCODER,
            "https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp8_e4m3fn_scaled.safetensors",
        ),
    ),
    "klein": (
        _FLUX_AE,
        _hf(
            "Flux.2 VAE (Klein 9B)", VAE,
            "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/vae/flux2-vae.safetensors",
        ),
        _hf(
            "Qwen3 4B (Klein 4B)", TEXT_ENCODER,
            "https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/text_encoders/qwen_3_4b.safetensors",
        ),
        _hf(
            "Qwen3 8B (Klein 9B)", TEXT_ENCODER,
            "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b.safetensors",
        ),
        _hf(
            "Qwen3 8B (fp8mixed)", TEXT_ENCODER,
            "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
        ),
    ),
    "zit": (
        _FLUX_AE,
        _hf(
            "Qwen3 4B (bf16)", TEXT_ENCODER,
            "https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/text_encoders/qwen_3_4b.safetensors",
        ),
        _hf(
            "Qwen3 4B (fp8_scaled)", TEXT_ENCODER,
            "https://huggingface.co/jiangchengchengNLP/qwen3-4b-fp8-scaled/blob/main/qwen3_4b_fp8_scaled.safetensors",
        ),
    ),
    "ernie": (
        _hf(
            "Ministral 3 3B", TEXT_ENCODER,
            "https://huggingface.co/Comfy-Org/ERNIE-Image/blob/main/text_encoders/ministral-3-3b.safetensors",
        ),
    ),
    "wan": (
        _hf(
            "Wan 2.1 VAE", VAE,
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors",
        ),
        _hf(
            "UMT5-XXL (fp16)", TEXT_ENCODER,
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/text_encoders/umt5_xxl_fp16.safetensors",
        ),
        _hf(
            "UMT5-XXL (fp8_scaled)", TEXT_ENCODER,
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        ),
    ),
    "qwen": (
        *_QWEN_IMAGE_VAE,
        _hf(
            "Qwen2.5-VL 7B (fp8_scaled)", TEXT_ENCODER,
            "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/blob/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "the GGUF build also needs its mmproj file for img2img",
        ),
    ),
    "anima": (
        *_QWEN_IMAGE_VAE,
        _hf(
            "Qwen3 0.6B base", TEXT_ENCODER,
            "https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/text_encoders/qwen_3_06b_base.safetensors",
        ),
    ),
    "krea": (
        *_QWEN_IMAGE_VAE,
        _hf(
            "Qwen3-VL 4B (bf16 / fp8_scaled)", TEXT_ENCODER,
            "https://huggingface.co/Comfy-Org/Krea-2/tree/main/text_encoders",
            "the build the wiki lists for Krea2; the page holds both bf16 and fp8_scaled",
        ),
        _hf(
            "Qwen3 4B", TEXT_ENCODER,
            "https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/text_encoders/qwen_3_4b.safetensors",
            "also works here, and is a single file that can be fetched directly",
        ),
    ),
    "pid": (
        _hf(
            "Gemma 2 2B IT", TEXT_ENCODER,
            "https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0/blob/main/Text_Encoder/gemma_2_2b_fp16.safetensors",
            "PiD's VAE is listed as Model Dependent, so no VAE is suggested here",
        ),
    ),
}


def for_architecture(arch: str | None) -> tuple[Download, ...]:
    if not arch:
        return ()
    return CATALOGUE.get(arch.lower(), ())


# Words shared by half the catalogue. Matching on them alone would offer the
# Flux VAE for a "Wan 2.1 VAE" gap.
_GENERIC_WORDS = frozenset(
    {"vae", "text", "encoder", "base", "alternative", "the", "and", "for", "fix", "scaled", "mixed"}
)


def matching(arch: str | None, need_label: str) -> tuple[Download, ...]:
    """Downloads that could satisfy a specific gap reported by the auditor.

    Requires a distinctive word to line up, not merely a shared one: offering
    the wrong multi-gigabyte file is worse than offering none.
    """
    words = {
        word
        for word in need_label.lower().replace("(", " ").replace(")", " ").split()
        if len(word) > 2 and word not in _GENERIC_WORDS
    }
    if not words:
        return ()
    hits = [entry for entry in for_architecture(arch) if any(word in entry.label.lower() for word in words)]
    return tuple(hits)
