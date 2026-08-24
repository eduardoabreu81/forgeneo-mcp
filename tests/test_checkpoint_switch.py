"""Switching checkpoint has to carry the architecture's modules along.

Forge builds its loading parameters from `forge_additional_modules` — whichever
VAE and text encoder are selected right now. Setting `sd_model_checkpoint`
alone therefore loads a checkpoint from one architecture against another
architecture's modules. The UI never does this because changing preset swaps
both together.
"""

from forgeneo_mcp.client import ApiResult
from forgeneo_mcp.profile import preset_for_checkpoint, switch_checkpoint

OPTIONS = {
    "sd_model_checkpoint": "Anime/animeMix_v10.safetensors",
    "forge_preset": "anima",
    "forge_checkpoint_anima": "Anime/animeMix_v10.safetensors",
    "forge_checkpoint_krea": "Krea/kreaMix_v20.safetensors",
    "forge_additional_modules_anima": ["/models/VAE/anime_vae.safetensors"],
    "forge_additional_modules_krea": ["/models/VAE/krea_vae.safetensors"],
}


class FakeClient:
    def __init__(self, options=None, ok=True):
        self._options = options if options is not None else dict(OPTIONS)
        self._ok = ok
        self.posted: dict | None = None

    def options(self):
        return ApiResult(self._ok, data=self._options, error=None if self._ok else "unreachable")

    def set_options(self, values):
        self.posted = values
        return ApiResult(True, data={})


def test_finds_the_preset_that_owns_a_checkpoint():
    assert preset_for_checkpoint(OPTIONS, "Krea/kreaMix_v20.safetensors") == "krea"
    assert preset_for_checkpoint(OPTIONS, "kreaMix_v20.safetensors") == "krea"
    assert preset_for_checkpoint(OPTIONS, "unknownModel.safetensors") is None


def test_switching_architecture_brings_its_modules():
    client = FakeClient()
    result = switch_checkpoint(client, "Krea/kreaMix_v20.safetensors")

    assert result["ok"] is True
    assert client.posted["sd_model_checkpoint"] == "Krea/kreaMix_v20.safetensors"
    assert client.posted["forge_preset"] == "krea"
    assert client.posted["forge_additional_modules"] == ["/models/VAE/krea_vae.safetensors"]
    assert result["preset"] == "krea"


def test_switching_within_the_same_architecture_leaves_modules_alone():
    client = FakeClient()
    result = switch_checkpoint(client, "Anime/animeMix_v10.safetensors")

    assert client.posted == {"sd_model_checkpoint": "Anime/animeMix_v10.safetensors"}
    assert result["applied"] == {}


def test_unclaimed_checkpoint_warns_instead_of_guessing():
    client = FakeClient()
    result = switch_checkpoint(client, "somethingNew.safetensors")

    assert result["ok"] is True
    assert result["architecture_confidence"] == "unknown"
    assert "forge_preset" not in client.posted
    assert any("nothing identifies this checkpoint" in note for note in result["notes"])


def test_unreachable_instance_reports_rather_than_switching():
    client = FakeClient(ok=False)
    result = switch_checkpoint(client, "whatever.safetensors")

    assert result["ok"] is False
    assert client.posted is None


def test_switch_always_flags_the_instance_wide_effect():
    result = switch_checkpoint(FakeClient(), "Krea/kreaMix_v20.safetensors")
    assert "web UI" in result["warning"]


CONFLICTING = dict(OPTIONS, forge_checkpoint_anima="Krea/museMix_v35.gguf")


def test_conflicting_signals_do_not_switch_preset():
    # Observed on a live instance: a Krea checkpoint recorded under `anima`
    # because it was once selected while that preset was active.
    client = FakeClient(dict(CONFLICTING))
    result = switch_checkpoint(client, "Krea/museMix_v35.gguf")

    assert result["ok"] is True
    assert result["architecture_confidence"] == "conflicting"
    assert "forge_preset" not in client.posted
    assert any("cannot tell which architecture" in note for note in result["notes"])


def test_explicit_preset_overrides_inference():
    client = FakeClient(dict(CONFLICTING))
    result = switch_checkpoint(client, "Krea/museMix_v35.gguf", preset="krea")

    assert result["architecture_confidence"] == "stated by caller"
    assert client.posted["forge_preset"] == "krea"
    assert client.posted["forge_additional_modules"] == ["/models/VAE/krea_vae.safetensors"]


def test_agreeing_signals_are_reported_as_such():
    client = FakeClient()
    result = switch_checkpoint(client, "Krea/kreaMix_v20.safetensors")
    assert result["architecture_confidence"] == "registry and folder agree"
    assert result["signals"] == {"preset_registry": "krea", "folder": "krea"}
