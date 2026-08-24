"""Dialect resolution has to behave on a clean install: no history, no sidecar,
and a file name that carries the lineage 0.5% of the time."""

import forgeneo_mcp.dialects as dialects
from forgeneo_mcp.identity import DialectResolution, lora_ecosystem, resolve


class FakeLora:
    def __init__(self, base_model):
        self.base_model = base_model


def test_quality_tags_differ_by_lineage():
    assert dialects.PONY.quality_prefix[0] == "score_9"
    assert "masterpiece" in dialects.ILLUSTRIOUS.quality_prefix
    # Natural-language models must get no quality tags at all.
    assert dialects.NATURAL.quality_prefix == ()
    assert dialects.NATURAL.negative_baseline == ()


def test_architecture_implies_dialect_when_unambiguous():
    assert dialects.for_architecture("krea") is dialects.NATURAL
    assert dialects.for_architecture("flux") is dialects.NATURAL
    assert dialects.for_architecture("anima") is dialects.ANIMA


def test_xl_is_ambiguous_and_implies_nothing():
    # Pony, Illustrious and stock SDXL share tensors and preset.
    assert dialects.for_architecture("xl") is None


def test_declared_base_model_maps_to_dialect():
    assert dialects.from_declared_base("Illustrious") is dialects.ILLUSTRIOUS
    assert dialects.from_declared_base("NoobAI XL") is dialects.ILLUSTRIOUS
    assert dialects.from_declared_base("Pony") is dialects.PONY
    assert dialects.from_declared_base("Anima") is dialects.ANIMA
    assert dialects.from_declared_base(None) is None


def test_observed_prompts_detect_pony_by_score_ladder():
    prompts = ["score_9, score_8_up, 1girl, solo"] * 6
    assert dialects.from_observed_prompts(prompts) is dialects.PONY


def test_observed_prompts_detect_prose():
    prompts = ["a photograph of a lighthouse at dusk with storm clouds"] * 6
    assert dialects.from_observed_prompts(prompts) is dialects.NATURAL


def test_lora_ecosystem_needs_a_clear_majority():
    mixed = [FakeLora("Illustrious")] * 3 + [FakeLora("Pony")] * 3
    assert lora_ecosystem(mixed)[0] is None

    leaning = [FakeLora("Illustrious")] * 8 + [FakeLora("Pony")] * 1
    dialect, detail = lora_ecosystem(leaning)
    assert dialect is dialects.ILLUSTRIOUS
    assert "8 of 9" in detail


def test_lora_ecosystem_ignores_tiny_samples():
    assert lora_ecosystem([FakeLora("Illustrious")] * 2)[0] is None


def test_clean_install_on_xl_returns_unknown_with_alternatives():
    result = resolve(identifier="abc123", architecture="xl")
    assert isinstance(result, DialectResolution)
    assert result.known is False
    assert result.confidence == "unknown"
    assert "pony" in result.alternatives and "illustrious" in result.alternatives


def test_clean_install_on_known_architecture_still_answers():
    result = resolve(identifier="abc123", architecture="krea")
    assert result.dialect is dialects.NATURAL
    assert result.source == "architecture"


def test_declared_base_outranks_architecture():
    result = resolve(identifier="x", architecture="xl", declared_base="Pony")
    assert result.dialect is dialects.PONY
    assert result.confidence == "high"


def test_anima_quality_prefix_matches_official_card():
    # CircleStone Labs recommends "masterpiece, best quality, score_7, safe"
    assert dialects.ANIMA.quality_prefix == ("masterpiece", "best quality", "score_7", "safe")
    assert "chromatic aberration" in dialects.ANIMA.negative_baseline
    assert "score_1" in dialects.ANIMA.negative_baseline


def test_anima_knows_it_cannot_do_realism():
    assert "photorealism" in dialects.ANIMA.avoid


def test_anima_tag_style_prefers_spaces():
    assert "spaces instead of underscores" in dialects.ANIMA.tag_style
    assert "@" in dialects.ANIMA.artist_syntax


def test_tagged_detection_survives_space_separated_tags():
    # Anima writes "long hair", not "long_hair" - underscore counting misses it.
    prompt = "masterpiece, best quality, 1girl, solo, long hair, blue eyes, school uniform"
    assert dialects._looks_tagged(prompt) is True


def test_prose_is_not_mistaken_for_tags():
    prompt = (
        "a photograph of a weathered fisherman standing on a harbour dock at first light, "
        "shot on 85mm, natural overcast light"
    )
    assert dialects._looks_tagged(prompt) is False
