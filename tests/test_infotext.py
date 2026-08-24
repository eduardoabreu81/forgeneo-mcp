from forgeneo_mcp.infotext import parse_infotext

SAMPLE = (
    "1girl, solo, <lora:turbo-accel-v2:1> <lora:my_style:0.75> masterpiece\n"
    "Negative prompt: worst quality, blurry\n"
    "Steps: 11, Sampler: ER SDE, Schedule type: Beta, CFG scale: 1.5, Seed: 42, "
    'Size: 1024x1024, Model hash: abc123, Model: animeMix_v10Turbo, Version: f2.0'
)


def test_splits_prompt_and_negative():
    info = parse_infotext(SAMPLE)
    assert info.prompt.startswith("1girl, solo")
    assert info.negative == "worst quality, blurry"


def test_reads_sampling_parameters():
    info = parse_infotext(SAMPLE)
    assert info.steps == 11.0
    assert info.cfg == 1.5
    assert info.sampler == "ER SDE"
    assert info.scheduler == "Beta"
    assert info.checkpoint == "animeMix_v10Turbo"


def test_extracts_loras_with_weights():
    info = parse_infotext(SAMPLE)
    assert info.loras == (("turbo-accel-v2", 1.0), ("my_style", 0.75))


def test_handles_prompt_without_negative():
    info = parse_infotext("a mountain at sunrise\nSteps: 20, CFG scale: 7, Model: foo")
    assert info.prompt == "a mountain at sunrise"
    assert info.negative == ""
    assert info.checkpoint == "foo"


def test_empty_text_is_safe():
    info = parse_infotext("")
    assert info.prompt == ""
    assert info.params == {}
    assert info.loras == ()


def test_prompt_only_text_has_no_params():
    info = parse_infotext("just a prompt with no metadata")
    assert info.prompt == "just a prompt with no metadata"
    assert info.params == {}
