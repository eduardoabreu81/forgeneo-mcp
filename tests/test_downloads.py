"""Downloading is the only thing this bridge writes, and it writes
multi-gigabyte files into someone's models folder. Nothing may start without an
explicit confirmation, and everything that could go wrong is checked first."""

import os

from forgeneo_mcp import downloads, fetcher


def test_blob_pages_become_direct_file_urls():
    # A Hugging Face /blob/ link serves the viewer page, not the file.
    entry = downloads.for_architecture("flux")[0]
    assert "/blob/" in entry.url
    assert "/resolve/" in entry.direct_url
    assert "/blob/" not in entry.direct_url


def test_entries_know_where_they_belong():
    for arch in downloads.CATALOGUE:
        for entry in downloads.for_architecture(arch):
            assert entry.target_folder in ("models/VAE", "models/text_encoder")
            if entry.is_directory:
                # A folder link has no single file, so it advertises no filename.
                assert "filename" not in entry.as_dict()
            else:
                assert entry.filename.endswith(".safetensors")


def test_every_architecture_with_requirements_has_somewhere_to_get_them():
    from forgeneo_mcp.modules import ARCH_MODULES

    for arch, spec in ARCH_MODULES.items():
        if spec.requirements:
            assert downloads.for_architecture(arch), f"{arch} has requirements but no download entry"


def test_matching_finds_an_entry_for_a_reported_gap():
    hits = downloads.matching("flux", "CLIP-L")
    assert hits and hits[0].label == "CLIP-L"
    assert downloads.matching("flux", "Wan 2.1 VAE") == ()


def test_sdxl_entry_names_every_lineage_it_serves():
    entry = downloads.for_architecture("xl")[0]
    assert "pony" in entry.note.lower() and "illustrious" in entry.note.lower()


def test_plan_refuses_when_the_file_is_already_there(tmp_path):
    existing = tmp_path / "clip_l.safetensors"
    existing.write_bytes(b"already here")
    intent = fetcher.plan("https://example.invalid/clip_l.safetensors", str(tmp_path), "clip_l.safetensors")
    assert intent.already_present is True
    assert "already exists" in intent.blocked


def test_plan_refuses_an_unreachable_folder(tmp_path):
    missing = str(tmp_path / "nope" / "deeper")
    intent = fetcher.plan("https://example.invalid/x.safetensors", missing, "x.safetensors")
    assert intent.blocked and "does not exist" in intent.blocked


def test_plan_refuses_when_the_disk_would_be_left_full(tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher, "remote_size", lambda url: 9 * 1024 ** 3)
    monkeypatch.setattr(
        fetcher.shutil, "disk_usage", lambda path: os.terminal_size((0, 0)) and _usage(9 * 1024 ** 3)
    )
    intent = fetcher.plan("https://example.invalid/big.safetensors", str(tmp_path), "big.safetensors")
    assert intent.blocked and "not enough room" in intent.blocked


class _usage:
    def __init__(self, free):
        self.total = free * 2
        self.used = free
        self.free = free


def test_fetch_stops_at_a_blocked_plan(tmp_path):
    existing = tmp_path / "x.safetensors"
    existing.write_bytes(b"present")
    result = fetcher.fetch("https://example.invalid/x.safetensors", str(tmp_path), "x.safetensors")
    assert result["ok"] is False
    # The existing file must survive an attempt that was refused.
    assert existing.read_bytes() == b"present"


def test_partial_downloads_never_look_like_finished_files(tmp_path):
    # A failed transfer leaves nothing behind, not a truncated model.
    result = fetcher.fetch("http://127.0.0.1:1/never.safetensors", str(tmp_path), "never.safetensors")
    assert result["ok"] is False
    assert list(tmp_path.iterdir()) == []


def test_directory_links_are_marked_unfetchable():
    # Some wiki entries point at a folder of builds; those cannot be downloaded
    # automatically because there is no single file to take.
    krea = downloads.for_architecture("krea")
    folder_entry = next(entry for entry in krea if entry.is_directory)
    assert "/tree/" in folder_entry.url
    payload = folder_entry.as_dict()
    assert payload["fetchable"] is False
    assert "pick the build" in payload["why"]


def test_krea_offers_both_valid_encoders():
    # Both the Krea-2 page build and the Z-Image single file work here.
    labels = [entry.label for entry in downloads.for_architecture("krea")]
    assert any("Qwen3-VL 4B" in label for label in labels)
    assert any(label == "Qwen3 4B" for label in labels)


def test_at_least_one_encoder_per_architecture_is_directly_fetchable():
    from forgeneo_mcp.modules import ARCH_MODULES, TEXT_ENCODER

    for arch, spec in ARCH_MODULES.items():
        if not any(requirement.kind == TEXT_ENCODER for requirement in spec.requirements):
            continue
        entries = [
            entry
            for entry in downloads.for_architecture(arch)
            if entry.kind == TEXT_ENCODER and not entry.is_directory
        ]
        assert entries, f"{arch} has no directly fetchable text encoder"
