"""Prompt dialects: how each model lineage expects to be addressed.

Two checkpoints can share an architecture, a preset and a tensor layout and
still want completely different prompts. Pony, Illustrious and stock SDXL are
all `xl` to Forge, yet Pony needs its score ladder, Illustrious needs booru
quality tags, and stock SDXL wants prose. Getting this wrong does not fail
loudly — it just produces consistently worse images.

Quality tags are the sharpest edge: an Illustrious prompt without
`masterpiece, best quality` visibly degrades, and a Flux prompt *with* them
degrades just as much, because the tokens mean nothing to it and dilute the
description.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Dialect:
    key: str
    label: str
    uses_tags: bool
    quality_prefix: tuple[str, ...] = ()
    negative_baseline: tuple[str, ...] = ()
    structure: str = ""
    notes: str = ""
    # Architectures where this dialect is the unambiguous default.
    architectures: tuple[str, ...] = field(default_factory=tuple)

    def prefix_text(self) -> str:
        return ", ".join(self.quality_prefix)

    def negative_text(self) -> str:
        return ", ".join(self.negative_baseline)

    def as_dict(self) -> dict:
        return {
            "dialect": self.key,
            "label": self.label,
            "uses_tags": self.uses_tags,
            "quality_prefix": list(self.quality_prefix),
            "negative_baseline": list(self.negative_baseline),
            "structure": self.structure,
            "notes": self.notes,
        }


PONY = Dialect(
    key="pony",
    label="Pony Diffusion",
    uses_tags=True,
    quality_prefix=("score_9", "score_8_up", "score_7_up"),
    negative_baseline=("score_6", "score_5", "score_4", "worst quality", "low quality"),
    structure="score ladder first, then source_/rating_ if wanted, then subject and booru tags",
    notes=(
        "The score ladder is not optional - without it Pony output collapses in quality. "
        "Optional controls: source_anime / source_cartoon / source_furry / source_pony, "
        "and rating_safe / rating_questionable / rating_explicit."
    ),
)

ILLUSTRIOUS = Dialect(
    key="illustrious",
    label="Illustrious / NoobAI",
    uses_tags=True,
    quality_prefix=("masterpiece", "best quality", "very aesthetic", "absurdres"),
    negative_baseline=(
        "worst quality", "bad quality", "lowres", "jpeg artifacts",
        "signature", "watermark", "username",
    ),
    structure="quality tags first, then subject count (1girl/1boy), character, series, then booru tags",
    notes=(
        "Danbooru vocabulary with underscores kept as trained (long_hair, school_uniform). "
        "NoobAI variants also respond to year_ tags such as year_2023 to steer era."
    ),
)

ANIMAGINE = Dialect(
    key="animagine",
    label="Animagine XL",
    uses_tags=True,
    quality_prefix=("masterpiece", "best quality", "very aesthetic", "absurdres"),
    negative_baseline=("lowres", "bad anatomy", "bad hands", "text", "error", "worst quality"),
    structure="1girl/1boy, character name, series name, then descriptive tags, quality tags last",
    notes="Trained on an ordered template: subject, character, series, rating, quality.",
)

ANIMA = Dialect(
    key="anima",
    label="Anima (hybrid)",
    uses_tags=True,
    quality_prefix=("masterpiece", "best quality"),
    negative_baseline=("worst quality", "low quality", "blurry"),
    structure="booru tags and natural phrases mix freely in the same prompt",
    notes=(
        "Accepts both vocabularies at once, so describe scene and lighting in prose while "
        "keeping booru tags for subject and pose. With a distilled/turbo LoRA loaded, drop "
        "most quality tags - the distillation bakes in a negative prompt and they add little."
    ),
    architectures=("anima",),
)

SD15 = Dialect(
    key="sd15",
    label="SD 1.5 lineage",
    uses_tags=True,
    quality_prefix=("masterpiece", "best quality", "highly detailed"),
    negative_baseline=("worst quality", "low quality", "blurry", "bad anatomy", "extra limbs"),
    structure="comma-separated descriptors, weight with (parentheses) when needed",
    notes="Responds to emphasis syntax and benefits from a substantial negative prompt.",
    architectures=("sd",),
)

SDXL_BASE = Dialect(
    key="sdxl_base",
    label="SDXL base lineage",
    uses_tags=False,
    quality_prefix=("high quality", "detailed"),
    negative_baseline=("low quality", "blurry", "watermark", "text"),
    structure="natural sentence describing subject and scene, then style and camera notes",
    notes="Understands prose; booru tags work poorly unless the merge was tuned for them.",
)

NATURAL = Dialect(
    key="natural",
    label="Natural language",
    uses_tags=False,
    quality_prefix=(),
    negative_baseline=(),
    structure="full descriptive sentences: subject, action, setting, light, then camera and lens",
    notes=(
        "Do not add quality tags. These models were trained on captions, so 'masterpiece' and "
        "'best quality' are dead tokens that dilute the description. Guidance usually sits at "
        "CFG 1, which makes the negative prompt weak or inert - put what you want in the "
        "positive prompt instead."
    ),
    architectures=("flux", "klein", "qwen", "zit", "krea", "lumina", "wan", "ernie", "pid"),
)

ALL: tuple[Dialect, ...] = (PONY, ILLUSTRIOUS, ANIMAGINE, ANIMA, SD15, SDXL_BASE, NATURAL)
BY_KEY: dict[str, Dialect] = {dialect.key: dialect for dialect in ALL}

# Architectures whose lineage cannot be read from the file: same tensors, same
# preset, different training. These are the ones worth asking about.
AMBIGUOUS_ARCHITECTURES = frozenset({"xl"})

# Substrings seen in checkpoint or LoRA base-model declarations. Weak on file
# names (measured at 0.5% coverage on a 216-checkpoint collection) but reliable
# when read from a declared baseModel field.
LINEAGE_MARKERS: tuple[tuple[str, str], ...] = (
    ("illustrious", "illustrious"),
    ("noobai", "illustrious"),
    ("noob", "illustrious"),
    ("animagine", "animagine"),
    ("pony", "pony"),
    ("anima", "anima"),
    ("krea", "natural"),
    ("flux", "natural"),
    ("qwen", "natural"),
    ("z-image", "natural"),
    ("zimage", "natural"),
    ("sdxl", "sdxl_base"),
    ("sd 1.5", "sd15"),
    ("sd1.5", "sd15"),
)


def from_declared_base(value: str | None) -> Dialect | None:
    """Map a declared baseModel string (sidecar, LoRA header, CivitAI) to a dialect."""
    if not value:
        return None
    lowered = str(value).lower()
    for marker, key in LINEAGE_MARKERS:
        if marker in lowered:
            return BY_KEY[key]
    return None


def for_architecture(arch: str | None) -> Dialect | None:
    """The dialect an architecture implies, when it implies one at all."""
    if not arch:
        return None
    lowered = arch.lower()
    if lowered in AMBIGUOUS_ARCHITECTURES:
        return None
    for dialect in ALL:
        if lowered in dialect.architectures:
            return dialect
    return None


def from_observed_prompts(prompts: list[str]) -> Dialect | None:
    """Infer the dialect from prompts that were actually used with a checkpoint."""
    if not prompts:
        return None
    joined = " ".join(prompts).lower()
    if "score_9" in joined or "score_8_up" in joined:
        return PONY
    tagged = sum(1 for prompt in prompts if _looks_tagged(prompt))
    ratio = tagged / len(prompts)
    if ratio >= 0.7:
        return ILLUSTRIOUS if "masterpiece" in joined or "best quality" in joined else ANIMA
    if ratio <= 0.2:
        return NATURAL
    return ANIMA  # mixed vocabulary in the same body of work


def _looks_tagged(prompt: str) -> bool:
    lowered = prompt.lower()
    if any(marker in lowered for marker in ("masterpiece", "best quality", "score_")):
        return True
    return lowered.count("_") >= 2 and lowered.count(",") >= 3
