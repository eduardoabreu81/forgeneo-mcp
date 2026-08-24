from forgeneo_mcp.config import Config, _parse_path_map
from forgeneo_mcp.presets import defaults_for, detect_lineage, looks_like_accelerator


def test_path_map_parsing_normalises_separators():
    pairs = _parse_path_map(r"I:\forge=\\host\I\forge ; C:\x=//host/C/x")
    assert pairs[0][0] == "i:/forge"
    assert pairs[0][1] == "//host/I/forge"
    assert len(pairs) == 2


def test_localise_translates_windows_path():
    config = Config(path_map=_parse_path_map(r"I:\sd-webui-forge-neo=//host/I/sd-webui-forge-neo"))
    local = config.localise(r"I:\sd-webui-forge-neo\models\Lora\a.safetensors")
    assert local == "//host/I/sd-webui-forge-neo/models/Lora/a.safetensors"


def test_localise_returns_none_without_mapping():
    assert Config().localise(r"I:\forge\models\a.safetensors") is None


def test_localise_is_case_insensitive_on_prefix():
    config = Config(path_map=_parse_path_map(r"i:\forge=//host/I/forge"))
    assert config.localise(r"I:\Forge\models\x.safetensors") is not None


def test_accelerator_detection_ignores_substring_false_positives():
    # "hyper-realistic" and "hyperass" must not read as accelerators
    assert looks_like_accelerator("SomeLora", ("hyper-realistic", "character")) is False
    assert looks_like_accelerator("MegaThick", ("hyperass", "curvy")) is False


def test_accelerator_detection_matches_exact_tags():
    assert looks_like_accelerator("whatever", ("assets", "distillation", "dmd2")) is True
    assert looks_like_accelerator("turbo-accel-lora-v1", ()) is True


def test_arch_defaults_known_and_unknown():
    anima = defaults_for("anima")
    assert anima is not None and anima.sampler == "ER SDE"
    assert defaults_for("nonexistent") is None
    assert defaults_for(None) is None


def test_wan_is_the_only_video_arch():
    assert defaults_for("wan").is_video is True
    assert defaults_for("anima").is_video is False


def test_lineage_detection_from_name():
    assert detect_lineage("ponyDiffusionV6XL.safetensors") == "pony"
    assert detect_lineage("waiIllustrious_v14") == "illustrious"
    assert detect_lineage("animeMix_v10Turbo") is None
