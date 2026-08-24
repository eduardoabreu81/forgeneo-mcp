"""Forge writes Shift / Distilled CFG Scale into infotext only for engines that
consume it, so past generations reveal whether an architecture uses it at all —
without maintaining a table that can disagree with the engine."""

from forgeneo_mcp.history import _summarise
from forgeneo_mcp.infotext import parse_infotext


def _entry(extra=""):
    return parse_infotext(
        "a prompt\nSteps: 10, Sampler: ER SDE, CFG scale: 1.5, "
        f"Model: someModel{extra}"
    )


def test_detects_architecture_that_uses_shift():
    entries = [_entry(", Shift: 3.0") for _ in range(6)]
    regime = _summarise("somemodel", (), entries)
    assert regime.uses_shift is True
    assert regime.shift == 3.0


def test_detects_architecture_that_ignores_shift():
    # krea's engine inherits use_shift = False, so Forge never records it.
    entries = [_entry() for _ in range(6)]
    regime = _summarise("somemodel", (), entries)
    assert regime.uses_shift is False
    assert regime.shift is None


def test_distilled_cfg_scale_counts_as_shift():
    # Flux-style architectures label the same field differently.
    entries = [_entry(", Distilled CFG Scale: 3.5") for _ in range(4)]
    regime = _summarise("somemodel", (), entries)
    assert regime.uses_shift is True
    assert regime.shift == 3.5


def test_shift_is_the_median_of_observations():
    entries = [_entry(", Shift: 3.0"), _entry(", Shift: 3.5"), _entry(", Shift: 3.5")]
    regime = _summarise("somemodel", (), entries)
    assert regime.shift == 3.5
