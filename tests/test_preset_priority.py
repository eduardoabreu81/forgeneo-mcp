"""The preset is the architecture speaking, so sampler/scheduler/shift come from
it. History only adjusts what an accelerator actually changes: steps and CFG."""

from forgeneo_mcp.history import HistoryIndex, _summarise, normalise_checkpoint
from forgeneo_mcp.infotext import parse_infotext


def _entry(steps, cfg, sampler="ER SDE", lora=None):
    prompt = "a prompt" if not lora else f"a prompt <lora:{lora}:1>"
    return parse_infotext(
        f"{prompt}\nSteps: {steps}, CFG scale: {cfg}, Sampler: {sampler}, "
        f"Schedule type: Beta, Model: someModel"
    )


def _index(entries_by_accel):
    index = HistoryIndex(None)
    index._built = True  # noqa: SLF001 - fixture bypasses the filesystem scan
    key = normalise_checkpoint("someModel")
    for accelerators, entries in entries_by_accel.items():
        regime = _summarise(key, accelerators, entries)
        index._regimes[(key, accelerators)] = regime  # noqa: SLF001
        index._by_checkpoint[key].append(regime)  # noqa: SLF001
        for entry in entries:
            for name, weight in entry.loras:
                index._lora_usage[name] += 1  # noqa: SLF001
                index._lora_weights[name].append(weight)  # noqa: SLF001
    return index


def test_habit_is_unknown_without_history():
    habit = HistoryIndex(None).accelerator_habit()
    assert habit["known"] is False
    assert habit["rate"] is None


def test_habit_reports_a_dominant_accelerator():
    index = _index(
        {
            ("turbo-accel-v2",): [_entry(10, 1.5, lora="turbo-accel-v2")] * 9,
            (): [_entry(32, 4.0)],
        }
    )
    habit = index.accelerator_habit()
    assert habit["known"] is True
    assert habit["rate"] == 0.9
    assert habit["common"][0]["name"] == "turbo-accel-v2"
    assert habit["common"][0]["typical_weight"] == 1.0


def test_habit_reports_absence_of_accelerators():
    index = _index({(): [_entry(32, 4.0)] * 10})
    habit = index.accelerator_habit()
    assert habit["rate"] == 0.0
    assert habit["common"] == []


def test_habit_counts_generations_not_regimes():
    index = _index(
        {
            ("acc",): [_entry(10, 1.5, lora="acc")] * 3,
            (): [_entry(32, 4.0)] * 7,
        }
    )
    habit = index.accelerator_habit()
    assert habit["generations"] == 10
    assert habit["rate"] == 0.3
