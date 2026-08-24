"""Regime selection must not let a stray outlier outvote the representative sample."""

from forgeneo_mcp.history import HistoryIndex, _summarise, normalise_checkpoint
from forgeneo_mcp.infotext import parse_infotext


def _entry(steps, cfg, lora=None):
    prompt = "a prompt" if not lora else f"a prompt <lora:{lora}:1>"
    return parse_infotext(
        f"{prompt}\nSteps: {steps}, CFG scale: {cfg}, Sampler: ER SDE, Model: phoenixAnima_v10"
    )


def _index_with_regimes():
    """Mirrors the real instance: 7 generations with an accelerator, 1 without."""
    index = HistoryIndex(None)
    index._built = True  # noqa: SLF001 - test fixture, bypassing the filesystem scan
    key = normalise_checkpoint("phoenixAnima_v10")

    with_turbo = ("Turbo-ANIMA-v2.9",)
    regimes = {
        (key, with_turbo): _summarise(key, with_turbo, [_entry(10, 1.5, "Turbo-ANIMA-v2.9")] * 7),
        (key, ()): _summarise(key, (), [_entry(30, 4.0)]),
    }
    index._regimes = regimes  # noqa: SLF001
    for (_, _), regime in regimes.items():
        index._by_checkpoint[key].append(regime)  # noqa: SLF001
    return index


def test_unspecified_accelerators_picks_the_most_observed_regime():
    index = _index_with_regimes()
    regime = index.regime_for("phoenixAnima_v10")
    assert regime.samples == 7
    assert regime.accelerators == ("Turbo-ANIMA-v2.9",)
    assert regime.steps == 10.0


def test_explicit_empty_tuple_asks_for_no_accelerator():
    index = _index_with_regimes()
    regime = index.regime_for("phoenixAnima_v10", accelerators=())
    assert regime.accelerators == ()
    assert regime.steps == 30.0


def test_explicit_combination_matches_exactly():
    index = _index_with_regimes()
    regime = index.regime_for("phoenixAnima_v10", accelerators=("Turbo-ANIMA-v2.9",))
    assert regime.samples == 7


def test_unknown_combination_falls_back_to_most_observed():
    index = _index_with_regimes()
    regime = index.regime_for("phoenixAnima_v10", accelerators=("never-seen-lora",))
    assert regime.samples == 7


def test_regimes_for_lists_all_sorted_by_samples():
    index = _index_with_regimes()
    regimes = index.regimes_for("phoenixAnima_v10")
    assert [r.samples for r in regimes] == [7, 1]


def test_unknown_checkpoint_returns_nothing():
    assert _index_with_regimes().regime_for("neverSeen") is None
