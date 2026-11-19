"""Stage one approved, checksum-verified release wheel; never build OptimizerPlus."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def approved_spec(manifest: dict, project: str) -> dict:
    if manifest.get("schema_version") != 1 or manifest.get("repository") != "Qluon/learnbywire":
        raise ValueError("Unexpected release manifest")
    spec = manifest["packages"][project]
    if spec.get("approved_for_pypi") is not True:
        raise ValueError(f"{project} is not approved for PyPI: {spec.get('pending', 'qualification pending')}")
    if Path(spec["filename"]).name != spec["filename"] or not spec["filename"].endswith(".whl"):
        raise ValueError("Invalid wheel filename")
    if not re.fullmatch(r"[a-f0-9]{64}", spec["sha256"]):
        raise ValueError("Invalid wheel checksum")
    return spec


def verify_wheel(path: Path, project: str, spec: dict) -> None:
    if path.name != spec["filename"]:
        raise ValueError("Wheel filename differs from the release manifest")
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError("Wheel checksum differs from the qualified artifact")
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate wheel archive entries")
        for name in names:
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name:
                raise ValueError("Unsafe archive path")
        metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError("Expected one distribution metadata file")
        metadata = BytesParser().parsebytes(wheel.read(metadata_files[0]))
        if normalized(metadata["Name"]) != project or metadata["Version"] != spec["version"]:
            raise ValueError("Distribution name or version differs from the manifest")
        if spec.get("license") and metadata["License-Expression"] != spec["license"]:
            raise ValueError("Unexpected public package licence")
        namespace = spec["namespace"]
        if not any(name.startswith(namespace + "/") for name in names):
            raise ValueError("Expected package namespace is missing")
        for name in names:
            top = name.split("/", 1)[0]
            if top != namespace and not top.endswith((".dist-info", ".libs")):
                raise ValueError(f"Unexpected top-level wheel entry: {top}")
        if project == "learnbywire-optimizerplus":
            forbidden = (".py", ".pyc", ".pyo", ".pyx", ".pxd", ".c", ".cc", ".cpp", ".h", ".cu", ".cuh")
            if any(name.endswith(forbidden) for name in names):
                raise ValueError("OptimizerPlus wheel contains implementation source or bytecode")


def pypi_version(project: str, version: str) -> dict | None:
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{project}/{version}/json",
        headers={"User-Agent": "LearnByWire-release-validation"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def check_registry(manifest: dict, project: str, spec: dict) -> None:
    if pypi_version(project, spec["version"]) is not None:
        raise ValueError("This version already exists on PyPI; compare its files before attempting another upload")
    for dependency in spec.get("requires_published", []):
        required = manifest["packages"][dependency]
        data = pypi_version(dependency, required["version"])
        if data is None:
            raise ValueError(f"Publish {dependency} {required['version']} first")
        matches = [item for item in data["urls"] if item["filename"] == required["filename"]]
        if len(matches) != 1 or matches[0]["digests"]["sha256"] != required["sha256"] or matches[0].get("yanked"):
            raise ValueError(f"Published dependency does not match the approved artifact: {dependency}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "release/pypi-manifest.json")
    parser.add_argument("--wheel-dir", type=Path, help="Use local release wheels instead of downloading a GitHub release")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    spec = approved_spec(manifest, args.project)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be empty")
    check_registry(manifest, args.project, spec)
    with tempfile.TemporaryDirectory(prefix="learnbywire-release-") as directory:
        temporary = Path(directory)
        if args.wheel_dir is None:
            release = json.loads(subprocess.check_output([
                "gh", "release", "view", spec["release_tag"], "--repo", manifest["repository"],
                "--json", "isDraft,tagName",
            ], text=True))
            if release["isDraft"] or release["tagName"] != spec["release_tag"]:
                raise ValueError("The approved GitHub release must be published")
            subprocess.run([
                "gh", "release", "download", spec["release_tag"], "--repo", manifest["repository"],
                "--pattern", spec["filename"], "--dir", str(temporary),
            ], check=True)
            wheel = temporary / spec["filename"]
        else:
            wheel = args.wheel_dir / spec["filename"]
        verify_wheel(wheel, args.project, spec)
        args.output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wheel, args.output / spec["filename"])
    print(json.dumps({"project": args.project, "version": spec["version"], "wheel": spec["filename"],
                      "sha256": spec["sha256"], "verified": True}))


if __name__ == "__main__":
    main()
