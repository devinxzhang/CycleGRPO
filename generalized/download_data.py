"""Download and verify the versioned Generalized CycleGRPO training annotations."""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath


RELEASE_TAG = "generalized-data-v1"
ASSET_NAME = "generalized-cyclegrpo-training-data-20261112.tar.gz"
SNAPSHOT_NAME = "generalized-cyclegrpo-rl-data"
ASSET_URL = f"https://github.com/devinxzhang/CycleGRPO/releases/download/{RELEASE_TAG}/{ASSET_NAME}"
ASSET_SHA256 = "f369777afd186ccecd9837cf9e319509a9145bb72d684d383e234675fe7703dc"
MANIFEST_SHA256 = "b267b1105b317076d02f1db0b04c96f2b47827735ed84404d5107aa0558f6015"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / SNAPSHOT_NAME


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def safe_parts(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or str(path) != value:
        raise ValueError(f"Unsafe archive/checksum path: {value!r}")
    return path.parts


def verify_snapshot(root, manifest_sha256=MANIFEST_SHA256):
    root = Path(root)
    manifest = root / "CHECKSUMS.sha256"
    if not manifest.is_file() or manifest.is_symlink() or sha256(manifest) != manifest_sha256:
        raise ValueError("Snapshot checksum manifest differs from the pinned release")
    expected = set()
    for line in manifest.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        parts = safe_parts(relative)
        path = root.joinpath(*parts)
        if relative in expected or path.is_symlink() or sha256(path) != digest:
            raise ValueError(f"Snapshot file checksum mismatch: {relative}")
        expected.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest
        and path.suffix != ".pyc"
        and not any(part in {"__pycache__", ".pytest_cache"} for part in path.relative_to(root).parts)
    }
    if actual != expected:
        raise ValueError("Snapshot file inventory differs from the pinned release")
    return len(expected) + 1


def install_archive(archive, output, asset_sha256=ASSET_SHA256, manifest_sha256=MANIFEST_SHA256):
    archive, output = Path(archive), Path(output)
    if sha256(archive) != asset_sha256:
        raise ValueError("Downloaded archive SHA256 differs from the pinned release")
    if output.exists():
        return {"status": "already_present", "path": str(output), "files": verify_snapshot(output, manifest_sha256)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cyclegrpo-extract-", dir=output.parent) as temporary:
        stage = Path(temporary)
        seen = set()
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                parts = safe_parts(member.name)
                if parts[0] != SNAPSHOT_NAME or len(parts) < 2 or not member.isfile() or member.name in seen:
                    raise ValueError("Archive contains an unexpected member, duplicate, link or directory")
                seen.add(member.name)
                target = stage.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError("Archive member has no file payload")
                with source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
        extracted = stage / SNAPSHOT_NAME
        count = verify_snapshot(extracted, manifest_sha256)
        if output.exists():
            raise ValueError("Output appeared during extraction; refusing to overwrite it")
        extracted.rename(output)
    return {"status": "installed", "path": str(output), "files": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded release archive (offline)")
    args = parser.parse_args()
    try:
        if args.output_dir.exists():
            report = {
                "status": "already_present",
                "path": str(args.output_dir),
                "files": verify_snapshot(args.output_dir),
            }
        elif args.archive:
            report = install_archive(args.archive, args.output_dir)
        else:
            args.output_dir.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".cyclegrpo-download-", dir=args.output_dir.parent) as temporary:
                archive = Path(temporary) / ASSET_NAME
                request = urllib.request.Request(ASSET_URL, headers={"User-Agent": "CycleGRPO-data-downloader"})
                with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as destination:
                    shutil.copyfileobj(response, destination)
                report = install_archive(archive, args.output_dir)
        report["release"] = RELEASE_TAG
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, tarfile.TarError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
