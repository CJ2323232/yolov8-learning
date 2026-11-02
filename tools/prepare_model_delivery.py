"""Verify and stage exact original checkpoints; never deserialize or publish them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import pickletools
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = {
    "01_standard", "02_standard_P3_CBAM", "03_P2",
    "04_P2_P3_CBAM", "05_P2_P2_CBAM",
}


class DeliveryError(ValueError):
    """A delivery integrity or destination check failed."""


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def source_path(root: Path, relative: str) -> Path:
    """Only allow portable relative paths which stay inside this project."""
    if not isinstance(relative, str) or not relative:
        raise DeliveryError("Source path must be a nonempty relative string")
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if posix.is_absolute() or windows.drive or "\\" in relative or ".." in posix.parts:
        raise DeliveryError(f"Nonportable source path rejected: {relative!r}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise DeliveryError(f"Source escapes project root: {relative!r}")
    if not resolved.is_file():
        raise DeliveryError(f"Required file is missing: {relative}")
    return resolved


def metadata_summary(path: Path) -> dict:
    """Inspect only pickle instructions; never call pickle.load or torch.load."""
    result = {"method": "zipfile_and_pickletools_without_deserialization", "personal_absolute_path_string_count": None}
    if not zipfile.is_zipfile(path):
        raise DeliveryError("Checkpoint is not a supported ZIP-based PyTorch archive")
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if item.filename.endswith("/data.pkl")]
        if len(members) != 1 or members[0].file_size > 32 * 1024 * 1024:
            raise DeliveryError("Checkpoint metadata is missing, ambiguous, or too large")
        count = 0
        try:
            for _, operand, _ in pickletools.genops(archive.read(members[0])):
                if isinstance(operand, str) and ("/Users/" in operand or re.search(r"[A-Za-z]:\\+Users\\", operand)):
                    count += 1
        except (ValueError, UnicodeError) as error:
            raise DeliveryError("Checkpoint metadata could not be inspected safely") from error
        result["personal_absolute_path_string_count"] = count
    return result


def verify(root: Path, manifest: dict) -> dict:
    """Read and hash all originals before permitting any preparation writes."""
    if manifest.get("status") != "staged_not_published" or manifest.get("published") is not False:
        raise DeliveryError("Manifest must explicitly describe an unpublished staged delivery")
    if manifest.get("release_url") is not None or manifest.get("release_tag") is not None:
        raise DeliveryError("Staged manifest cannot claim a published release")
    entries = manifest.get("models", [])
    if len(entries) != 5 or {entry.get("id") for entry in entries} != EXPECTED_IDS:
        raise DeliveryError("Manifest must contain the five distinct expected experiment IDs")
    assets = set()
    records = []
    for entry in entries:
        name = entry.get("asset_filename", "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*\.pt", name) or name in assets:
            raise DeliveryError("Asset names must be distinct portable .pt filenames")
        assets.add(name)
        if entry.get("download_url") is not None:
            raise DeliveryError("Unpublished checkpoints cannot claim download URLs")
        source = source_path(root, entry.get("source"))
        actual_size = source.stat().st_size
        actual_sha = digest(source)
        if actual_size != entry.get("bytes") or actual_sha != entry.get("sha256"):
            raise DeliveryError(f"Size or SHA-256 mismatch: {entry['source']}")
        configuration = source_path(root, entry.get("configuration"))
        if digest(configuration) != entry.get("configuration_sha256"):
            raise DeliveryError(f"Configuration SHA-256 mismatch: {entry['configuration']}")
        records.append({"id": entry["id"], "source": entry["source"], "asset_filename": name,
                        "bytes": actual_size, "sha256": actual_sha,
                        "checkpoint_metadata": metadata_summary(source)})
    return {"status": "verified_local_originals_not_publication_approval", "models": records,
            "total_weight_bytes": sum(item["bytes"] for item in records),
            "model_execution": False, "network_access": False,
            "publication_blockers": manifest.get("release_gates", {})}


def prepare(root: Path, manifest_path: Path, manifest: dict, output: Path) -> dict:
    """Make independent copies in a new destination, preserving every original."""
    report = verify(root, manifest)
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise DeliveryError("Output already exists; choose a new directory (nothing was overwritten)")
    resolved = output.resolve()
    original_run = (root / "run").resolve()
    if resolved == original_run or resolved.is_relative_to(original_run):
        raise DeliveryError("Delivery output must not be inside the original run directory")
    for entry in manifest["models"]:
        # Also protect any source directory if the inventory is reused elsewhere.
        original_directory = source_path(root, entry["source"]).parent
        if resolved == original_directory or resolved.is_relative_to(original_directory):
            raise DeliveryError("Delivery output must not be inside a source checkpoint directory")
    card = manifest_path.parent / "MODEL_CARD.md"
    if not card.is_file():
        raise DeliveryError("MODEL_CARD.md is missing next to the manifest")
    if not output.parent.is_dir():
        raise DeliveryError("Output parent does not exist; create the intended parent first")
    output.mkdir(exist_ok=False)
    incomplete = output / ".INCOMPLETE"
    incomplete.write_text("Preparation incomplete. Do not publish or treat as verified.\n", encoding="utf-8")
    try:
        for entry in manifest["models"]:
            source = source_path(root, entry["source"])
            target = output / entry["asset_filename"]
            # Independent copies avoid a later staged-file edit mutating original weights.
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            if target.stat().st_size != entry["bytes"] or digest(target) != entry["sha256"]:
                raise DeliveryError(f"Prepared copy failed integrity check: {target.name}")
        # Recheck originals to detect concurrent edits during staging.
        verify(root, manifest)
        (output / "release_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        shutil.copyfile(card, output / "MODEL_CARD.md")
        (output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "NOT_PUBLISHED.txt").write_text(
            "LOCAL REVIEW ONLY — NOT A PUBLISHED RELEASE\n"
            "Dataset/pretraining-derived checkpoint rights and personal metadata review are pending.\n"
            "Project code licensing does not resolve checkpoint or dataset rights.\n"
            "Publication requires separate explicit user authorization; no download URL exists.\n"
            "The exact originals were copied, not sanitized, deserialized, or executed.\n", encoding="utf-8")
        checksums = [f"{digest(path)}  {path.name}" for path in sorted(output.iterdir()) if path.is_file() and path != incomplete]
        (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        incomplete.unlink()
    except Exception:
        # Leave this newly created directory visibly incomplete for inspection; never
        # recursively delete a path which could contain files written by someone else.
        raise
    return dict(report, prepared=True, copy_method="independent_byte_copy", publication=False)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    mode = argument_parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only verification (the default)")
    mode.add_argument("--prepare", action="store_true", help="Copy verified originals into a new local review directory")
    argument_parser.add_argument("--root", type=Path, default=ROOT, help="Project root containing the relative source paths")
    argument_parser.add_argument("--manifest", type=Path, help="Manifest path (default: <root>/models/release_manifest.json)")
    argument_parser.add_argument("--output", type=Path, help="Explicit NEW output directory; required with --prepare")
    return argument_parser


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.prepare and args.output is None:
        argument_parser.error("--prepare requires an explicit --output directory")
    if not args.prepare and args.output is not None:
        argument_parser.error("--output is only valid with --prepare; check mode never writes")
    root = args.root.expanduser().resolve()
    manifest_path = args.manifest or root / "models/release_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = prepare(root, manifest_path, manifest, args.output) if args.prepare else verify(root, manifest)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"Model delivery check failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
