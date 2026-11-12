"""The release downloader must verify content and reject unsafe archives."""

import hashlib
import io
import tarfile

import pytest

from generalized.download_data import SNAPSHOT_NAME, install_archive, sha256, verify_snapshot


def make_archive(tmp_path, payloads=None, member=None):
    payloads = payloads or {"README.md": b"training annotations\n", "data/rl.jsonl": b'{"source":"temporal"}\n'}
    manifest = "".join(
        hashlib.sha256(content).hexdigest() + "  " + name + "\n" for name, content in payloads.items()
    ).encode()
    archive = tmp_path / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, content in {**payloads, "CHECKSUMS.sha256": manifest}.items():
            info = tarfile.TarInfo(SNAPSHOT_NAME + "/" + name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        if member is not None:
            tar.addfile(member, io.BytesIO(b""))
    return archive, hashlib.sha256(manifest).hexdigest()


def test_install_and_repeat_use_pinned_checksums(tmp_path):
    archive, manifest = make_archive(tmp_path)
    output = tmp_path / "installed"
    result = install_archive(archive, output, sha256(archive), manifest)
    assert result["status"] == "installed"
    assert result["files"] == 3
    assert (output / "data/rl.jsonl").read_text() == '{"source":"temporal"}\n'
    assert install_archive(archive, output, sha256(archive), manifest)["status"] == "already_present"


def test_wrong_archive_hash_leaves_no_output(tmp_path):
    archive, manifest = make_archive(tmp_path)
    output = tmp_path / "installed"
    with pytest.raises(ValueError, match="archive SHA256"):
        install_archive(archive, output, "0" * 64, manifest)
    assert not output.exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", SNAPSHOT_NAME + "/../../escape"])
def test_archive_path_escape_is_rejected(tmp_path, name):
    archive, manifest = make_archive(tmp_path, member=tarfile.TarInfo(name))
    output = tmp_path / "installed"
    with pytest.raises(ValueError, match="Unsafe"):
        install_archive(archive, output, sha256(archive), manifest)
    assert not output.exists()
    assert not (tmp_path / "escape").exists()


def test_archive_symlink_is_rejected(tmp_path):
    link = tarfile.TarInfo(SNAPSHOT_NAME + "/linked")
    link.type, link.linkname = tarfile.SYMTYPE, "../../outside"
    archive, manifest = make_archive(tmp_path, member=link)
    with pytest.raises(ValueError, match="unexpected member"):
        install_archive(archive, tmp_path / "installed", sha256(archive), manifest)


def test_existing_modified_snapshot_is_not_overwritten(tmp_path):
    archive, manifest = make_archive(tmp_path)
    output = tmp_path / "installed"
    install_archive(archive, output, sha256(archive), manifest)
    (output / "README.md").write_text("changed\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        install_archive(archive, output, sha256(archive), manifest)
    assert (output / "README.md").read_text() == "changed\n"


def test_existing_snapshot_rejects_an_unpinned_manifest(tmp_path):
    archive, manifest = make_archive(tmp_path)
    output = tmp_path / "installed"
    install_archive(archive, output, sha256(archive), manifest)
    with pytest.raises(ValueError, match="checksum manifest"):
        verify_snapshot(output, "0" * 64)
