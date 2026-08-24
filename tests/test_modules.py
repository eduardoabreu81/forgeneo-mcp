"""Module requirements come from the Forge wiki, because the instance cannot
answer the question: `forge_additional_modules_<arch>` records the last
selection made under a preset, so loading a checkpoint while the wrong preset
is active silently overwrites another architecture's modules."""

from forgeneo_mcp.modules import ARCH_MODULES, audit, classify, requirements_for

AVAILABLE = [
    {"model_name": "ae.safetensors", "filename": "/models/VAE/ae.safetensors"},
    {"model_name": "qwen_image_vae.safetensors", "filename": "/models/VAE/qwen_image_vae.safetensors"},
    {"model_name": "custom_vae_v10.safetensors", "filename": "/models/VAE/Sub/custom_vae_v10.safetensors"},
    {"model_name": "hdrVAE_fp32.safetensors", "filename": "/models/VAE/Anime/hdrVAE_fp32.safetensors"},
    {"model_name": "qwen_3_06b_base.safetensors", "filename": "/models/text_encoder/qwen_3_06b_base.safetensors"},
    {"model_name": "qwen3vl_4b_fp8_scaled.safetensors", "filename": "/models/text_encoder/qwen3vl_4b_fp8_scaled.safetensors"},
]


def test_classifies_by_folder_not_by_name():
    assert classify("/models/VAE/anything.safetensors") == "vae"
    assert classify("/models/text_encoder/anything.safetensors") == "text_encoder"
    assert classify("/models/Stable-diffusion/model.safetensors") is None


def test_flux_vae_pattern_is_anchored():
    # "ae" is the real filename and also sits inside every other "*vae*".
    flux_vae = ARCH_MODULES["flux"].requirements[0]
    assert flux_vae.matches("ae.safetensors") is True
    assert flux_vae.matches("qwen_image_vae.safetensors") is False
    assert flux_vae.matches("custom_vae_v10.safetensors") is False


def test_unclassifiable_module_is_not_accused():
    # A module missing from the listing has no known kind; guessing one
    # would flag a perfectly good VAE as an intruder.
    report = audit("anima", ["some_unlisted_file.safetensors"], AVAILABLE)
    assert "unrecognised" not in report


def test_detects_a_foreign_text_encoder_left_behind():
    # The exact failure this exists for: a Krea encoder stuck on the anima preset,
    # alongside a VAE that does satisfy the requirement.
    report = audit("anima", ["qwen_image_vae.safetensors", "qwen3vl_4b_fp8_scaled.safetensors"], AVAILABLE)
    assert report["unrecognised"]["modules"] == ["qwen3vl_4b_fp8_scaled.safetensors"]
    needs = [item["need"] for item in report["missing"]]
    assert "Qwen3 0.6B base" in needs
    assert [item["loaded"] for item in report["satisfied"]] == ["qwen_image_vae.safetensors"]


def test_unrecognised_is_phrased_as_a_question_not_a_verdict():
    # Name matching cannot tell a community build from a leftover, so the
    # report must not accuse a validly-named-differently VAE.
    report = audit("anima", ["hdrVAE_fp32.safetensors"], AVAILABLE)
    assert "check" in report["unrecognised"]
    assert "community build" in report["unrecognised"]["why"]


def test_missing_module_points_at_an_installed_candidate():
    report = audit("anima", [], AVAILABLE)
    encoder = next(item for item in report["missing"] if item["kind"] == "text_encoder")
    assert encoder["candidates_installed"] == ["qwen_3_06b_base.safetensors"]
    assert encoder["action"] == "select one"


def test_missing_module_with_nothing_installed_says_download():
    report = audit("flux", [], AVAILABLE)
    clip = next(item for item in report["missing"] if item["need"] == "CLIP-L")
    assert clip["candidates_installed"] == []
    assert clip["action"] == "download one"


def test_unspecified_vae_is_reported_not_guessed():
    report = audit("krea", ["qwen_image_vae.safetensors", "qwen3vl_4b_fp8_scaled.safetensors"], AVAILABLE)
    assert "not specified" in report["vae"]["status"]
    # A VAE cannot be called wrong when the reference names none.
    assert "unrecognised" not in report
    assert report["missing"] == []


def test_self_contained_architectures_require_nothing():
    for arch in ("sd", "xl"):
        spec = requirements_for(arch)
        assert spec.requirements == ()
        assert audit(arch, [], AVAILABLE)["missing"] == []


def test_unknown_architecture_says_so():
    assert audit("nonexistent", [], AVAILABLE)["known"] is False
