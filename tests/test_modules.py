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


def test_flux_vae_pattern_is_anchored_but_accepts_rebuilds():
    # "ae" is the real filename and also sits inside every other "*vae*", so it
    # anchors; "flux" anywhere is a strong enough signal for community rebuilds.
    flux_vae = ARCH_MODULES["flux"].requirements[0]
    assert flux_vae.matches("ae.safetensors") is True
    assert flux_vae.matches("ultrafluxVAEImproved_v10.safetensors") is True
    assert flux_vae.matches("qwen_image_vae.safetensors") is False
    assert flux_vae.matches("custom_vae_v10.safetensors") is False


def test_klein_covers_both_sizes():
    # Flux.2-Klein 4B and 9B differ in both companions.
    spec = requirements_for("klein")
    vae, encoder = spec.requirements
    assert vae.matches("ae.safetensors") and vae.matches("flux2-vae.safetensors")
    assert encoder.matches("qwen_3_4b_bf16.safetensors")
    assert encoder.matches("qwen_3_8b_fp8mixed.safetensors")


def test_every_forge_preset_has_requirements():
    from forgeneo_mcp.presets import ARCH_DEFAULTS

    assert set(ARCH_DEFAULTS) <= set(ARCH_MODULES)


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
    assert "Qwen3 0.6B" in needs
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
    # PiD is where the wiki declines to name one: "Model Dependent". The VAE is
    # still required — only its identity is left to the checkpoint.
    report = audit("pid", [], AVAILABLE)
    assert "does not name which one" in report["vae"]["status"]
    assert report["vae"]["installed_vaes"]


def test_krea_vae_comes_from_the_merged_wiki_cell():
    # A first pass read Krea's VAE as unspecified because the source table
    # merges that cell across Qwen-Image, Anima and Krea2.
    report = audit("krea", ["qwen_image_vae.safetensors", "qwen3vl_4b_fp8_scaled.safetensors"], AVAILABLE)
    assert report["missing"] == []
    assert "vae" not in report
    assert len(report["satisfied"]) == 2


def test_sd_and_sdxl_follow_the_reference():
    """The source table lists a VAE for SD1 and SDXL and marks only their text
    encoder N/A. An earlier version called that VAE optional and described it as
    an upgrade — neither claim is in the reference."""
    for arch, pattern in (("sd", "vae-ft-mse"), ("xl", "sdxl-vae")):
        spec = requirements_for(arch)
        assert len(spec.requirements) == 1
        requirement = spec.requirements[0]
        assert requirement.kind == "vae"
        assert requirement.optional is False
        assert requirement.matches(f"{pattern}-something.safetensors")


def test_unknown_architecture_says_so():
    assert audit("nonexistent", [], AVAILABLE)["known"] is False


def test_every_architecture_accounts_for_its_vae():
    """A VAE is what turns latents into pixels, so every architecture needs one.
    Each entry must either name the VAE the reference gives it, or record that
    the reference leaves it to the checkpoint."""
    from forgeneo_mcp.modules import VAE

    for arch, spec in ARCH_MODULES.items():
        requires_vae = any(requirement.kind == VAE for requirement in spec.requirements)
        assert requires_vae or spec.vae_unspecified, (
            f"{arch} neither requires a VAE nor explains where its VAE comes from"
        )


def test_notes_do_not_add_claims_the_reference_lacks():
    # "optional", "upgrade" and "built in" were all inferred, not sourced.
    for arch in ("sd", "xl"):
        note = ARCH_MODULES[arch].note.lower()
        for invented in ("optional", "upgrade", "built-in", "baked"):
            assert invented not in note


def test_unspecified_vae_says_required_not_absent():
    report = audit("pid", [], AVAILABLE)
    assert report["vae"]["status"].startswith("required")


def test_xl_note_covers_every_lineage():
    # Pony, Illustrious and stock SDXL all run under the `xl` preset and share
    # its VAE; they differ in prompt dialect, not in modules.
    note = ARCH_MODULES["xl"].note.lower()
    for lineage in ("pony", "illustrious", "animagine"):
        assert lineage in note
