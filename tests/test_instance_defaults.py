"""Fields omitted from a request fall back to the API model's own defaults,
which know nothing about the loaded architecture. distilled_cfg_scale defaults
to 3.5 in Forge; the anima preset asks for 3.0. Omitting it silently generated
at the wrong shift, so the profile must read the instance and send it."""

from forgeneo_mcp.generate import build_payload
from forgeneo_mcp.profile import instance_defaults

OPTIONS = {
    "anima_t2i_step": 10.0,
    "anima_t2i_cfg": 1.0,
    "anima_t2i_dcfg": 3.0,
    "anima_t2i_sampler": "ER SDE",
    "anima_t2i_scheduler": "Beta",
    "anima_t2i_width": 832,
    "anima_t2i_height": 1216,
    "anima_i2i_width": 0,
    "anima_i2i_sampler": "ER SDE",
}


def test_reads_per_architecture_defaults():
    live = instance_defaults(OPTIONS, "anima")
    assert live["step"] == 10.0
    assert live["dcfg"] == 3.0
    assert live["sampler"] == "ER SDE"
    assert live["width"] == 832
    assert live["height"] == 1216


def test_zero_dimensions_mean_auto_not_zero():
    live = instance_defaults(OPTIONS, "anima", mode="i2i")
    assert "width" not in live
    assert live["sampler"] == "ER SDE"


def test_unknown_preset_yields_nothing():
    assert instance_defaults(OPTIONS, "flux") == {}
    assert instance_defaults(OPTIONS, None) == {}


def test_payload_sends_distilled_cfg_when_given():
    payload = build_payload("prompt", distilled_cfg_scale=3.0)
    assert payload["distilled_cfg_scale"] == 3.0


def test_payload_omits_distilled_cfg_when_unset():
    # Explicitly not sending 3.5 ourselves; absence is the caller's choice.
    assert "distilled_cfg_scale" not in build_payload("prompt")
