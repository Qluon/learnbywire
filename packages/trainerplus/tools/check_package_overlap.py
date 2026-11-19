"""Read-only wheel ownership check; no extraction, imports or installation.

Compare import-root destinations, treating wheel purelib and platlib as the
same root conservatively. Unsupported wheel installation schemes are rejected.
Installed checks inspect distribution file manifests, not import hooks; this
does not resolve editable source trees or prove runtime import compatibility.
"""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile


class InventoryError(ValueError):
    """The package cannot be compared without guessing its installed paths."""


def _identity(metadata):
    values = {}
    for key in ("Name", "Version"):
        entries = metadata.get_all(key, [])
        if len(entries) != 1 or not entries[0].strip():
            raise InventoryError(f"Distribution must have exactly one nonempty {key}")
        values[key] = entries[0].strip()
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", values["Name"]):
        raise InventoryError("Invalid distribution name")
    if any(ord(char) < 32 for char in values["Version"]):
        raise InventoryError("Invalid distribution version")
    return re.sub(r"[-_.]+", "-", values["Name"]).lower(), values["Version"]


def _wheel_parts(name, *, directory=False):
    # Do not allow ZipFile/PurePath normalization to hide a dangerous name.
    raw = name[:-1] if directory and name.endswith("/") else name
    parts = raw.split("/")
    if (not raw or "\\" in raw or ":" in raw or
            any(ord(char) < 32 for char in raw) or
            any(part in {"", ".", ".."} for part in parts)):
        raise InventoryError(f"Unsafe or ambiguous wheel path: {name!r}")
    return parts


def _destination(parts, *, metadata_dir, directory=False):
    first = parts[0]
    if first.endswith(".dist-info"):
        if first != metadata_dir:
            raise InventoryError("Wheel contains multiple distribution metadata directories")
        return None
    if first.endswith(".data"):
        expected = metadata_dir.removesuffix(".dist-info") + ".data"
        if first != expected:
            raise InventoryError(f"Unexpected wheel data directory: {first}")
        if len(parts) == 1 and directory:
            return None
        if len(parts) < 2 or parts[1] not in {"purelib", "platlib"}:
            raise InventoryError(f"Unsupported wheel .data mapping: {'/'.join(parts)}")
        parts = parts[2:]
        if not parts and directory:
            return None
        if not parts or parts[0].endswith((".dist-info", ".data")):
            raise InventoryError("Ambiguous or empty wheel data destination")
    return None if directory else "/".join(parts)


def wheel_inventory(path):
    """Return normalized file ownership for one wheel without extracting it."""
    path = Path(path).absolute()
    if path.suffix != ".whl":
        raise InventoryError(f"Expected a .whl file: {path}")
    with zipfile.ZipFile(path) as archive:
        entries = []
        archive_names = set()
        for info in archive.infolist():
            if info.orig_filename != info.filename:
                raise InventoryError("Wheel contains a filename truncated by ZIP parsing")
            parts = _wheel_parts(info.filename, directory=info.is_dir())
            canonical = "/".join(parts)
            if canonical in archive_names:
                raise InventoryError(f"Duplicate wheel path: {canonical}")
            archive_names.add(canonical)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}):
                raise InventoryError(f"Unsupported special file in wheel: {info.filename}")
            entries.append((info, parts))
        manifests = [(info, parts[0]) for info, parts in entries
                     if len(parts) == 2 and parts[0].endswith(".dist-info")
                     and parts[1] == "METADATA" and not info.is_dir()]
        if len(manifests) != 1:
            raise InventoryError("Wheel must contain exactly one .dist-info/METADATA")
        manifest, metadata_dir = manifests[0]
        if manifest.file_size > 1024 * 1024:
            raise InventoryError("Wheel METADATA exceeds the 1 MiB inspection limit")
        owner, version = _identity(BytesParser().parsebytes(archive.read(manifest)))
        destinations = set()
        for info, parts in entries:
            destination = _destination(parts, metadata_dir=metadata_dir, directory=info.is_dir())
            if destination is not None:
                if destination in destinations:
                    raise InventoryError(f"Multiple wheel entries install to {destination}")
                destinations.add(destination)
        # A file cannot also be the parent directory of another payload file.
        for destination in destinations:
            if any(str(parent) in destinations for parent in PurePosixPath(destination).parents
                   if str(parent) != "."):
                raise InventoryError(f"Wheel has a file/directory conflict at {destination}")
    return dict(owner=owner, version=version, source=str(path), files=destinations)


def installed_inventories():
    """Read installed RECORD manifests without importing their packages.

    RECORD may legitimately list scripts outside site-packages; these are
    excluded because this check compares import-root ownership only. No installed
    distribution is skipped just because its name/version matches the candidate.
    """
    result = []
    for distribution in importlib.metadata.distributions():
        owner, version = _identity(distribution.metadata)
        records = distribution.files
        if records is None:
            raise InventoryError(f"Installed distribution {owner} has no file manifest")
        root = Path(os.path.abspath(os.fspath(distribution.locate_file(""))))
        files = set()
        for record in records:
            raw = str(record)
            if "\\" in raw or any(ord(char) < 32 for char in raw):
                raise InventoryError(f"Ambiguous installed RECORD path for {owner}: {raw!r}")
            # abspath normalizes legitimate ../bin entries without following
            # package symlinks away from the paths their installer owns.
            absolute = Path(os.path.abspath(os.fspath(distribution.locate_file(record))))
            try:
                relative = absolute.relative_to(root)
            except ValueError:
                continue
            parts = relative.parts
            if not parts or parts[0].endswith((".dist-info", ".egg-info")):
                continue
            files.add(relative.as_posix())
        result.append(dict(owner=owner, version=version,
                           source=f"installed:{root}", files=files))
    return result


def compare(candidate, inventories):
    collisions = []
    checked = []
    for other in inventories:
        identity = {key: other[key] for key in ("owner", "version", "source")}
        checked.append(dict(identity, file_count=len(other["files"])))
        for path in sorted(candidate["files"].intersection(other["files"])):
            collisions.append(dict(identity, path=path))
    collisions.sort(key=lambda item: (item["path"], item["owner"], item["version"], item["source"]))
    return dict(schema_version=1, status="overlap" if collisions else "no_file_overlap",
                candidate={**{key: candidate[key] for key in ("owner", "version", "source")},
                           "file_count": len(candidate["files"])},
                checked=checked, collisions=collisions,
                scope="Import-root file ownership only; not editable-source or runtime compatibility.")


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise InventoryError(message)


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--against", action="append", default=[], type=Path)
    parser.add_argument("--installed", action="store_true")
    try:
        args = parser.parse_args(argv)
        candidate = wheel_inventory(args.candidate)
        others = [wheel_inventory(path) for path in args.against]
        if args.installed:
            others.extend(installed_inventories())
        report = compare(candidate, others)
    except (InventoryError, OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
        print(json.dumps(dict(schema_version=1, status="invalid", error=str(error)), sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["collisions"] else 0


if __name__ == "__main__":
    sys.exit(main())
