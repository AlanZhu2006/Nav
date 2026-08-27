import hashlib
import shutil

import pytest

from MemNavData.verify_portable_checksum_manifest import verify_manifest


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_tree(root):
    root.mkdir()
    (root / "checkpoints").mkdir()
    first = root / "checkpoints/member_0.pt"
    second = root / "deployment_manifest.json"
    first.write_bytes(b"checkpoint")
    second.write_text("{}\n", encoding="utf-8")
    manifest = root / "OUTPUTS.sha256"
    manifest.write_text(
        f"{_digest(first)}  checkpoints/member_0.pt\n"
        f"{_digest(second)}  deployment_manifest.json\n",
        encoding="utf-8",
    )
    return manifest


def test_manifest_survives_relocation(tmp_path):
    source = tmp_path / "source"
    manifest = _portable_tree(source)
    assert verify_manifest(manifest)["entry_count"] == 2

    relocated = tmp_path / "unrelated-name"
    shutil.copytree(source, relocated)
    assert verify_manifest(relocated / "OUTPUTS.sha256")["verified"] is True


@pytest.mark.parametrize(
    "bad_name",
    ["/home/cv/member.pt", "../member.pt", "checkpoints/../../member.pt"],
)
def test_manifest_rejects_nonportable_paths(tmp_path, bad_name):
    root = tmp_path / "root"
    manifest = _portable_tree(root)
    manifest.write_text(f"{'0' * 64}  {bad_name}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_manifest(manifest)


def test_manifest_rejects_modified_payload(tmp_path):
    root = tmp_path / "root"
    manifest = _portable_tree(root)
    (root / "deployment_manifest.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_manifest(manifest)
