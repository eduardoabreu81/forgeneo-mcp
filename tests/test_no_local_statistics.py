"""Guard against shipping one environment's measurements as universal advice.

Every number the agent receives should be measured in the environment it is
running against — the operator's own history, their instance's options, their
checkpoint's tags. That principle held everywhere except the text fields, where
a measurement taken on the development machine ("23% of Anima checkpoints
declare realism tags") was written into runtime guidance as though it were a
property of the model family.

This test parses the modules that carry runtime text and fails on statistics
that cannot be true for an arbitrary install. Docstrings are deliberately
exempt: recording the evidence behind a design decision is not the same as
handing that evidence to a caller as fact.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "forgeneo_mcp"

# Keyword arguments whose strings reach the agent through tool output.
RUNTIME_FIELDS = frozenset(
    {"notes", "avoid", "structure", "tag_style", "artist_syntax", "weighting", "detail", "label"}
)

# Percentages, counts of things, and "N of M" phrasings are all environment
# measurements. Version-like and tag-like numbers are not.
STATISTIC_PATTERNS = (
    re.compile(r"\d+\s*%"),
    re.compile(r"\b\d+\s+of\s+\d+\b", re.IGNORECASE),
    # "+" catches the "300+ LoRAs" shape as well as a bare count.
    re.compile(
        r"\b\d+\s*\+?\s*-?\s*(checkpoint|lora|model|generation|image|file)s?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(most|majority|typically|usually)\s+\d+", re.IGNORECASE),
)

ALLOWED = (
    # Literal vocabulary from model documentation, not measurements.
    "year 2023",
    "year 2025",
    "score_1",
    "score_9",
    "two sentences",
    "8-12 steps",
    "CFG 1",
)


def _runtime_strings() -> list[tuple[str, int, str, str]]:
    found: list[tuple[str, int, str, str]] = []
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.keyword) or node.arg not in RUNTIME_FIELDS:
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if isinstance(value, str):
                found.append((path.name, node.value.lineno, node.arg, value))
    return found


def test_runtime_text_carries_no_environment_statistics():
    offenders = []
    for filename, lineno, field, text in _runtime_strings():
        cleaned = text
        for allowed in ALLOWED:
            cleaned = cleaned.replace(allowed, "")
        for pattern in STATISTIC_PATTERNS:
            match = pattern.search(cleaned)
            if match:
                offenders.append(f"{filename}:{lineno} [{field}] -> {match.group(0)!r} in {text[:90]!r}")
                break

    assert not offenders, (
        "Runtime guidance must not carry statistics measured on one installation. "
        "State the qualitative fact and let the bridge count in the environment it runs "
        "against.\n  " + "\n  ".join(offenders)
    )


def test_the_guard_actually_catches_the_original_mistake():
    # The exact wording that shipped, to prove the patterns are not vacuous.
    regression = (
        "true photorealism on the stock model. Community merges are a different matter: on one "
        "207-checkpoint Anima collection, 23% declared realism tags."
    )
    assert any(pattern.search(regression) for pattern in STATISTIC_PATTERNS)


def test_qualitative_replacement_passes():
    replacement = (
        "merges are a different matter: many are tuned towards semi-realism, so judge the loaded "
        "checkpoint by its own tags rather than by the family"
    )
    assert not any(pattern.search(replacement) for pattern in STATISTIC_PATTERNS)


def test_audit_actually_reads_something():
    # A guard that inspects nothing would pass forever.
    assert len(_runtime_strings()) >= 10


# The README is read by more people than any tool output, and the same slip
# reached it twice: example paths from one machine, and a LoRA count from one
# collection. Prose lines are checked; code blocks are not, since a worked
# example needs concrete values to be worth reading.
README = PACKAGE.parent / "README.md"
README_ALLOWED = (
    "3.10",  # Python version
    "7860",  # default port
    "0.1.0",  # project version
    "4n+1",  # Forge's video batch rule
    "3.5",  # a documented Forge default
    "4.40",  # Gradio version
)


def _readme_prose() -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    in_code = False
    for number, line in enumerate(README.read_text(encoding="utf-8").splitlines(), start=1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code and line.strip():
            lines.append((number, line))
    return lines


def test_readme_prose_carries_no_environment_statistics():
    offenders = []
    for number, line in _readme_prose():
        cleaned = line
        for allowed in README_ALLOWED + ALLOWED:
            cleaned = cleaned.replace(allowed, "")
        for pattern in STATISTIC_PATTERNS:
            match = pattern.search(cleaned)
            if match:
                offenders.append(f"README.md:{number} -> {match.group(0)!r} in {line.strip()[:90]!r}")
                break

    assert not offenders, (
        "README prose must not quote counts measured on one installation.\n  "
        + "\n  ".join(offenders)
    )


def test_readme_guard_catches_the_lora_count():
    assert any(pattern.search("so 300+ LoRAs cost nothing") for pattern in STATISTIC_PATTERNS)


def test_readme_guard_reads_the_file():
    assert len(_readme_prose()) >= 40
