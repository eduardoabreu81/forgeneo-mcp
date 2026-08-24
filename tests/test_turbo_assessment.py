"""Turbo assessment must behave sanely on a clean install, where there is no
history at all and the checkpoint carries no usable metadata."""

from forgeneo_mcp.presets import defaults_for
from forgeneo_mcp.profile import assess_turbo

ANIMA = defaults_for("anima")  # 32 steps, CFG 4.0 - not distilled
KLEIN = defaults_for("klein")  # 4 steps, CFG 1.0 - distilled architecture
ZIT = defaults_for("zit")  # 9 steps, CFG 1.0 - distilled architecture


def test_clean_install_unknown_checkpoint_is_unknown_not_no():
    result = assess_turbo("someModel_v1", (), ANIMA, observed_steps=32.0, samples=0)
    assert result.state == "unknown"
    assert result.confidence == "low"
    assert result.architecture_is_distilled is False


def test_distilled_architecture_is_not_reported_as_checkpoint_turbo():
    # The old ceiling rule marked klein and zit turbo purely because their
    # presets are low-step. The architecture flag carries that instead.
    for arch in (KLEIN, ZIT):
        result = assess_turbo("plainName", (), arch, observed_steps=float(arch.steps), samples=0)
        assert result.architecture_is_distilled is True
        assert result.state == "unknown"


def test_name_hint_gives_medium_confidence_without_history():
    result = assess_turbo("magnanima_v10Turbo", (), ANIMA, observed_steps=None, samples=0)
    assert result.state == "yes"
    assert result.confidence == "medium"


def test_accelerator_in_use_is_high_confidence():
    result = assess_turbo("plainName", ("anima-turbo-lora-v0.2",), ANIMA, 32.0, 0)
    assert result.state == "yes"
    assert result.confidence == "high"


def test_history_far_below_preset_is_high_confidence():
    # 11 steps against a 32-step preset, seen repeatedly.
    result = assess_turbo("astrallus_v5", (), ANIMA, observed_steps=11.0, samples=38)
    assert result.state == "yes"
    assert result.confidence == "high"


def test_history_in_line_with_preset_is_a_confident_no():
    result = assess_turbo("plainName", (), ANIMA, observed_steps=30.0, samples=20)
    assert result.state == "no"
    assert result.confidence == "medium"


def test_single_observation_is_not_enough_to_decide():
    result = assess_turbo("plainName", (), ANIMA, observed_steps=10.0, samples=1)
    assert result.state == "unknown"


def test_no_preset_match_still_answers():
    result = assess_turbo("plainName", (), None, observed_steps=None, samples=0)
    assert result.state == "unknown"
    assert result.architecture_is_distilled is False
