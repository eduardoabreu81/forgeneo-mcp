from forgeneo_mcp.history import _looks_danbooru, _summarise, normalise_checkpoint
from forgeneo_mcp.infotext import parse_infotext


def test_normalise_strips_folder_and_extension():
    # This is the mismatch that made every profile fall back to preset defaults:
    # options reports a path, infotext reports a bare name.
    assert normalise_checkpoint(r"Anima\phoenixAnima_v10.safetensors") == "phoenixanima_v10"
    assert normalise_checkpoint("phoenixAnima_v10") == "phoenixanima_v10"


def test_normalise_handles_forward_slashes_and_hash_suffix():
    assert normalise_checkpoint("sub/dir/model.ckpt") == "model"
    assert normalise_checkpoint("model_v1 [abc123]") == "model_v1"


def test_normalise_is_safe_on_empty_input():
    assert normalise_checkpoint("") == ""


def test_summarise_uses_median_and_display_name():
    entries = [
        parse_infotext(f"a prompt\nSteps: {steps}, CFG scale: 1.5, Sampler: ER SDE, Model: magnanima_v10Turbo")
        for steps in (10, 11, 11, 12)
    ]
    regime = _summarise("magnanima_v10turbo", (), entries)
    assert regime.checkpoint == "magnanima_v10Turbo"  # display name, not the key
    assert regime.steps == 11.0
    assert regime.cfg == 1.5
    assert regime.sampler == "ER SDE"
    assert regime.samples == 4


def test_danbooru_detection():
    assert _looks_danbooru("score_9, score_8_up, 1girl") is True
    assert _looks_danbooru("masterpiece, best quality, solo") is True
    assert _looks_danbooru("long_hair, blue_eyes, standing, smile") is True
    assert _looks_danbooru("a photograph of a mountain at sunrise") is False
