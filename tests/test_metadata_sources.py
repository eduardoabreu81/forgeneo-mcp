"""enable_pnginfo and save_txt are independent Forge settings; history must
survive any combination of them."""

import struct
import zlib

from forgeneo_mcp.infotext import read_generation_metadata

PARAMS = "a prompt\nSteps: 10, CFG scale: 1.5, Sampler: ER SDE, Model: someModel"


def _png_bytes(text: str | None) -> bytes:
    out = bytearray(b"\x89PNG\r\n\x1a\n")

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    out += chunk(b"IHDR", b"\x00" * 13)
    if text is not None:
        out += chunk(b"tEXt", b"parameters\x00" + text.encode("utf-8"))
    out += chunk(b"IEND", b"")
    return bytes(out)


def test_reads_embedded_chunk_when_present(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(_png_bytes(PARAMS))
    text, source = read_generation_metadata(str(path))
    assert source == "png_chunk"
    assert "Steps: 10" in text


def test_falls_back_to_txt_when_pnginfo_disabled(tmp_path):
    # enable_pnginfo off, save_txt on: the data is only in the sidecar.
    path = tmp_path / "image.png"
    path.write_bytes(_png_bytes(None))
    (tmp_path / "image.txt").write_text(PARAMS, encoding="utf-8")

    text, source = read_generation_metadata(str(path))
    assert source == "txt_sidecar"
    assert "Steps: 10" in text


def test_prefers_embedded_chunk_over_sidecar(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(_png_bytes(PARAMS))
    (tmp_path / "image.txt").write_text("Steps: 99", encoding="utf-8")

    text, source = read_generation_metadata(str(path))
    assert source == "png_chunk"
    assert "Steps: 10" in text


def test_non_png_formats_use_the_sidecar(tmp_path):
    # JPEG and WebP never carry a PNG text chunk.
    path = tmp_path / "image.jpg"
    path.write_bytes(b"not really a jpeg")
    (tmp_path / "image.txt").write_text(PARAMS, encoding="utf-8")

    text, source = read_generation_metadata(str(path))
    assert source == "txt_sidecar"
    assert "Sampler: ER SDE" in text


def test_reports_none_when_both_carriers_are_missing(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(_png_bytes(None))
    text, source = read_generation_metadata(str(path))
    assert text is None
    assert source == "none"


def test_empty_sidecar_counts_as_missing(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(_png_bytes(None))
    (tmp_path / "image.txt").write_text("   ", encoding="utf-8")
    text, source = read_generation_metadata(str(path))
    assert text is None
    assert source == "none"
