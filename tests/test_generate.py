import os
import time

from forgeneo_mcp.generate import _new_files, _normalise, _trim_info, build_payload


def test_payload_omits_unset_parameters():
    payload = build_payload("a prompt")
    assert "steps" not in payload
    assert "cfg_scale" not in payload
    assert "sampler_name" not in payload
    assert payload["save_images"] is True


def test_payload_includes_given_parameters():
    payload = build_payload("a prompt", steps=10, cfg_scale=1.5, sampler_name="ER SDE", scheduler="Beta")
    assert payload["steps"] == 10
    assert payload["cfg_scale"] == 1.5
    assert payload["sampler_name"] == "ER SDE"
    assert payload["scheduler"] == "Beta"


def test_new_files_ignores_infotext_sidecars(tmp_path):
    # Forge writes a .txt next to each image; only the artwork should come back.
    started = time.time()
    image = tmp_path / "00000-model - 832x1216.png"
    sidecar = tmp_path / "00000-model - 832x1216.txt"
    image.write_bytes(b"fake")
    sidecar.write_text("Steps: 10")

    found = _new_files(str(tmp_path), before=set(), started=started)
    assert len(found) == 1
    assert found[0].endswith(".png")


def test_new_files_skips_preexisting(tmp_path):
    started = time.time()
    old = tmp_path / "old.png"
    old.write_bytes(b"fake")
    before = {str(old)}
    new = tmp_path / "new.png"
    new.write_bytes(b"fake")

    found = _new_files(str(tmp_path), before=before, started=started)
    assert len(found) == 1
    assert os.path.basename(found[0]) == "new.png"


def test_normalise_preserves_unc_prefix():
    assert _normalise("//host/I/forge" + chr(92) + "output" + chr(92) + "a.png") == "//host/I/forge/output/a.png"
    assert _normalise("C:" + chr(92) + "forge" + chr(92) + "a.png") == "C:/forge/a.png"


def test_trim_info_drops_noise():
    trimmed = _trim_info({"prompt": "x", "seed": 1, "irrelevant": "y" * 5000})
    assert trimmed == {"prompt": "x", "seed": 1}
