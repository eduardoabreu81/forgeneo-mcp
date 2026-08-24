"""Reading and parsing of A1111/Forge generation metadata ("infotext").

PNG parsing is done by hand rather than through Pillow: we only need the tEXt
chunk holding the "parameters" key, which sits near the start of the file. That
keeps the dependency list short and, more importantly, avoids reading whole
images over a network share when indexing thousands of outputs.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PARAMETERS_KEY = b"parameters"
MAX_CHUNK_BYTES = 4 * 1024 * 1024

LORA_PATTERN = re.compile(r"<lora:([^:>]+):([0-9]*\.?[0-9]+)", re.IGNORECASE)
# Key: value pairs on the trailing parameters line, tolerating quoted values.
PARAM_PATTERN = re.compile(r'\s*([\w \-/]+):\s*("(?:[^"]|\\")*"|[^,]*)(?:,|$)')


@dataclass(frozen=True)
class Infotext:
    prompt: str
    negative: str
    params: dict[str, str]
    loras: tuple[tuple[str, float], ...]

    def get_float(self, key: str) -> float | None:
        raw = self.params.get(key)
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    @property
    def checkpoint(self) -> str | None:
        return self.params.get("Model")

    @property
    def steps(self) -> float | None:
        return self.get_float("Steps")

    @property
    def cfg(self) -> float | None:
        return self.get_float("CFG scale")

    @property
    def sampler(self) -> str | None:
        return self.params.get("Sampler")

    @property
    def scheduler(self) -> str | None:
        return self.params.get("Schedule type")


def read_png_parameters(path: str) -> str | None:
    """Return the "parameters" text chunk of a PNG, or None if absent."""
    try:
        with open(path, "rb") as handle:
            if handle.read(8) != PNG_SIGNATURE:
                return None
            while True:
                header = handle.read(8)
                if len(header) < 8:
                    return None
                length, chunk_type = struct.unpack(">I4s", header)
                if chunk_type == b"IDAT" or chunk_type == b"IEND":
                    return None  # pixel data reached; no text chunk present
                if length > MAX_CHUNK_BYTES:
                    return None
                if chunk_type in (b"tEXt", b"iTXt"):
                    body = handle.read(length)
                    text = _decode_text_chunk(chunk_type, body)
                    if text is not None:
                        return text
                else:
                    handle.seek(length, 1)
                handle.seek(4, 1)  # skip CRC
    except (OSError, struct.error):
        return None


def read_generation_metadata(path: str) -> tuple[str | None, str]:
    """Read an image's generation parameters from wherever Forge put them.

    Both carriers are optional and independent settings: `enable_pnginfo`
    embeds a tEXt chunk in the PNG, `save_txt` writes a sibling .txt. Reading
    only the chunk loses the history of anyone who turned that off but kept the
    text files — and formats other than PNG never carry the chunk at all.

    Returns the text and which source supplied it.
    """
    if path.lower().endswith(".png"):
        embedded = read_png_parameters(path)
        if embedded:
            return embedded, "png_chunk"

    sidecar = os.path.splitext(path)[0] + ".txt"
    try:
        with open(sidecar, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read().strip()
    except OSError:
        return None, "none"
    return (text, "txt_sidecar") if text else (None, "none")


def _decode_text_chunk(chunk_type: bytes, body: bytes) -> str | None:
    keyword, _, remainder = body.partition(b"\x00")
    if keyword != PARAMETERS_KEY:
        return None
    if chunk_type == b"tEXt":
        return remainder.decode("utf-8", errors="replace")
    # iTXt: compression flag, compression method, language tag, translated key
    if len(remainder) < 2 or remainder[0] != 0:
        return None  # compressed iTXt is not worth supporting here
    rest = remainder[2:]
    _, _, rest = rest.partition(b"\x00")
    _, _, rest = rest.partition(b"\x00")
    return rest.decode("utf-8", errors="replace")


def parse_infotext(text: str) -> Infotext:
    """Split raw infotext into prompt, negative prompt and parameter pairs."""
    if not text:
        return Infotext("", "", {}, ())

    lines = text.split("\n")
    params_line = ""
    if len(lines) > 1 and _looks_like_params(lines[-1]):
        params_line = lines[-1]
        lines = lines[:-1]

    body = "\n".join(lines)
    prompt, separator, negative = body.partition("Negative prompt:")
    prompt = prompt.strip()
    negative = negative.strip() if separator else ""

    params = {
        key.strip(): value.strip().strip('"')
        for key, value in PARAM_PATTERN.findall(params_line)
        if key.strip()
    }
    loras = tuple(
        (name.strip(), float(weight))
        for name, weight in LORA_PATTERN.findall(prompt)
    )
    return Infotext(prompt=prompt, negative=negative, params=params, loras=loras)


def _looks_like_params(line: str) -> bool:
    return "Steps:" in line or "Sampler:" in line or "CFG scale:" in line
