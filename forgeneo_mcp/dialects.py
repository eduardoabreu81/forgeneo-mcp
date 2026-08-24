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
    # Tag spelling differs between lineages and is easy to get wrong: Anima's
    # documentation asks for spaces, and reserves underscores for score tags.
    tag_style: str = ""
    artist_syntax: str = ""
    weighting: str = ""
    # What the model is not for. Steers the agent to a different checkpoint
    # rather than fighting one that cannot do the job.
    avoid: str = ""
    variants: tuple[str, ...] = field(default_factory=tuple)
    # Architectures where this dialect is the unambiguous default.
    architectures: tuple[str, ...] = field(default_factory=tuple)

    def prefix_text(self) -> str:
        return ", ".join(self.quality_prefix)

    def negative_text(self) -> str:
        return ", ".join(self.negative_baseline)

    def as_dict(self) -> dict:
        payload = {
            "dialect": self.key,
            "label": self.label,
            "uses_tags": self.uses_tags,
            "quality_prefix": list(self.quality_prefix),
            "negative_baseline": list(self.negative_baseline),
            "structure": self.structure,
            "notes": self.notes,
        }
        for name in ("tag_style", "artist_syntax", "weighting", "avoid"):
            value = getattr(self, name)
            if value:
                payload[name] = value
        if self.variants:
            payload["variants"] = list(self.variants)
        return payload


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
    tag_style="lowercase Danbooru vocabulary, comma separated, spaces rather than underscores",
    artist_syntax="artist tags by name, weighted when the style needs reinforcing: '(artist name:1.4)'",
    notes=(
        "Danbooru vocabulary. NoobAI variants also respond to year tags such as 'year 2023' to "
        "steer era. Quality tags matter here: dropping them visibly degrades output."
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
    # Values from CircleStone Labs' own model card, not inferred.
    quality_prefix=("masterpiece", "best quality", "score_7", "safe"),
    negative_baseline=(
        "worst quality", "low quality", "score_1", "score_2", "score_3",
        "artist name", "blurry", "jpeg artifacts", "chromatic aberration",
    ),
    structure=(
        "quality / meta / year / safety tags, then 1girl-1boy-1other, character, series, "
        "artist, then general tags. Order within each group is free."
    ),
    tag_style=(
        "lowercase, spaces instead of underscores ('long hair', not 'long_hair'). Score tags are "
        "the only ones that keep an underscore. Prefer the Gelbooru spelling where it differs "
        "from Danbooru."
    ),
    artist_syntax=(
        "prefix artists with @ ('@big chungus') - without the @ the effect is very weak, and "
        "artist influence dilutes in long prompts, so boost it: '(@artist name:1.5)'"
    ),
    weighting="emphasis works but needs higher weights than SDXL: '(chibi:2)'",
    avoid=(
        "true photorealism on the stock model - the base is an illustration model and its card "
        "says so outright. Merges are a different matter: many are tuned towards semi-realism or "
        "a 2.5D look and reach it convincingly, so judge the loaded checkpoint by its own tags "
        "(realistic, semi-realism, 2.5d, photorealistic, 3d) rather than by the family. For work "
        "that must read as an actual photograph, reach for a photographic model instead. Long "
        "text rendering is weak either way: single words usually work, phrases often do not."
    ),
    notes=(
        "Trained on Danbooru-style tags, natural-language captions, and mixtures of both, so tags "
        "and prose can be interleaved in any order. Two quality systems are accepted and can be "
        "combined: human scores (masterpiece / best quality / good quality / normal quality / "
        "low quality / worst quality) and PonyV7 aesthetic scores (score_9 down to score_1). "
        "Also honours time tags (year 2025, newest / recent / mid / early / old), safety tags "
        "(safe / sensitive / nsfw / explicit) and meta tags (highres, absurdres, official art). "
        "Random tag dropout during training means not every relevant tag is required. Pure "
        "natural language works but wants at least two sentences; very short prompts drift."
    ),
    variants=(
        "Base: unrefined, maximum diversity and style adherence; plain default look without "
        "artist or quality tags. Train LoRAs against this one.",
        "Aesthetic: fine-tuned on curated data with quality tags stripped from captions. Quality "
        "tags are unnecessary, and score_* tags in either prompt push it towards slop.",
        "Turbo: distilled, CFG 1 at 8-12 steps, strong default style and higher stability but "
        "less diversity. Artist tags respond weakly here - use a non-distilled version when "
        "style adherence matters.",
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


BOORU_MARKERS = (
    "masterpiece", "best quality", "score_", "absurdres", "highres",
    "1girl", "1boy", "1other", "solo", "looking at viewer",
)


def _looks_tagged(prompt: str) -> bool:
    """Whether a prompt reads as booru vocabulary rather than prose.

    Underscores are a weak signal on their own: Anima asks for spaces and keeps
    underscores only for score tags, so counting them misses tagged prompts
    entirely. Subject-count and quality vocabulary is the reliable marker.
    """
    lowered = prompt.lower()
    if any(marker in lowered for marker in BOORU_MARKERS):
        return True
    # Many short comma-separated fragments with no sentence structure.
    fragments = [part.strip() for part in lowered.split(",") if part.strip()]
    if len(fragments) < 4:
        return False
    short = sum(1 for fragment in fragments if len(fragment.split()) <= 3)
    return short / len(fragments) >= 0.7
