"""The preset is what Forge actually loads, so it is the operating truth. The
reference table exists to notice drift, never to override the preset — Forge
records the last selection made under a preset, so loading a checkpoint while
another was active leaves the wrong modules behind."""

from forgeneo_mcp.client import ApiResult
from forgeneo_mcp.profile import _module_health

MODULES = [
    {"model_name": "qwen_image_vae.safetensors", "filename": "/models/VAE/qwen_image_vae.safetensors"},
    {"model_name": "qwen_3_06b_base.safetensors", "filename": "/models/text_encoder/qwen_3_06b_base.safetensors"},
    {"model_name": "qwen3vl_4b_fp8_scaled.safetensors", "filename": "/models/text_encoder/qwen3vl_4b_fp8_scaled.safetensors"},
]


class FakeClient:
    def __init__(self, ok=True):
        self._ok = ok

    def modules(self):
        return ApiResult(self._ok, data=MODULES, error=None if self._ok else "unreachable")


def test_healthy_preset_reports_no_problems():
    options = {
        "forge_additional_modules_anima": [
            "/models/VAE/qwen_image_vae.safetensors",
            "/models/text_encoder/qwen_3_06b_base.safetensors",
        ]
    }
    health = _module_health(FakeClient(), options, "anima")
    assert health["healthy"] is True
    assert health["problems"] == []
    assert len(health["loaded"]) == 2


def test_drifted_preset_names_the_installed_file_that_fits():
    # A Krea encoder left on the anima preset, the real failure this detects.
    options = {
        "forge_additional_modules_anima": [
            "/models/VAE/qwen_image_vae.safetensors",
            "/models/text_encoder/qwen3vl_4b_fp8_scaled.safetensors",
        ]
    }
    health = _module_health(FakeClient(), options, "anima")
    assert health["healthy"] is False
    assert any("qwen_3_06b_base.safetensors is installed and fits" in p for p in health["problems"])
    assert any("does not ask for" in p for p in health["problems"])


def test_missing_with_nothing_installed_says_so():
    health = _module_health(FakeClient(), {"forge_additional_modules_flux": []}, "flux")
    assert any("none is installed" in problem for problem in health["problems"])


def test_unknown_architecture_is_skipped_quietly():
    health = _module_health(FakeClient(), {}, "nonexistent")
    assert health["checked"] is False


def test_unreachable_listing_does_not_fabricate_a_verdict():
    health = _module_health(FakeClient(ok=False), {}, "anima")
    assert health["checked"] is False
    assert "healthy" not in health


def test_no_preset_means_no_check():
    assert _module_health(FakeClient(), {}, None)["checked"] is False
