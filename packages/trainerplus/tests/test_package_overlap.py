"""Package ownership tests use synthetic wheels and RECORDs, never installs."""
from email.message import Message
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile


TOOL = Path(__file__).resolve().parents[1] / "tools" / "check_package_overlap.py"
SPEC = importlib.util.spec_from_file_location("package_overlap", TOOL)
overlap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlap)


class FakeDistribution:
    def __init__(self, root, name, version, files):
        self.root = root
        self.metadata = Message()
        self.metadata["Name"] = name
        self.metadata["Version"] = version
        self.files = files

    def locate_file(self, path):
        return self.root / str(path)


class PackageOverlapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def wheel(self, name, files, *, version="1.0", label=None):
        stem = f"{name.replace('-', '_')}-{version}"
        path = self.root / f"{label or stem}-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{stem}.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")
            archive.writestr(f"{stem}.dist-info/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\n")
            for item in files:
                archive.writestr(item, "pass\n")
        return path

    def test_legacy_learnbywire_collision_names_owner_and_path(self):
        candidate = overlap.wheel_inventory(self.wheel("learnbywire-trainer", ["learnbywire/__init__.py"]))
        legacy = overlap.wheel_inventory(self.wheel("qluon-optimizerplus", ["learnbywire/__init__.py", "optimizerplus/__init__.py"], version="1.2.2"))
        report = overlap.compare(candidate, [legacy])
        self.assertEqual(report["status"], "overlap")
        self.assertEqual(len(report["collisions"]), 1)
        self.assertEqual(report["collisions"][0]["owner"], "qluon-optimizerplus")
        self.assertEqual(report["collisions"][0]["path"], "learnbywire/__init__.py")

    def test_optimizerplus_only_wheel_is_disjoint(self):
        candidate = overlap.wheel_inventory(self.wheel("learnbywire-trainer", ["learnbywire/__init__.py"]))
        backend = overlap.wheel_inventory(self.wheel("qluon-optimizerplus", ["optimizerplus/__init__.py"]))
        self.assertEqual(overlap.compare(candidate, [backend])["collisions"], [])

    def test_data_purelib_and_platlib_map_to_import_root(self):
        candidate = overlap.wheel_inventory(self.wheel("candidate", ["learnbywire/__init__.py"]))
        for scheme in ("purelib", "platlib"):
            with self.subTest(scheme=scheme):
                backend = overlap.wheel_inventory(self.wheel("backend", [f"backend-1.0.data/{scheme}/learnbywire/__init__.py"], label=scheme))
                self.assertEqual(overlap.compare(candidate, [backend])["collisions"][0]["path"], "learnbywire/__init__.py")

    def test_metadata_is_not_compared_as_package_payload(self):
        left = overlap.wheel_inventory(self.wheel("same-owner", ["left.py"], label="left"))
        right = overlap.wheel_inventory(self.wheel("same-owner", ["right.py"], label="right"))
        self.assertEqual(left["files"], {"left.py"})
        self.assertEqual(overlap.compare(left, [right])["collisions"], [])

    def test_unsafe_or_ambiguous_wheel_paths_are_rejected(self):
        for index, name in enumerate(("../escape.py", "/absolute.py", "a/../b.py", "a//b.py", "a/./b.py", "a\\b.py", "C:/drive.py")):
            with self.subTest(name=name):
                with self.assertRaisesRegex(overlap.InventoryError, "Unsafe or ambiguous"):
                    overlap.wheel_inventory(self.wheel("bad", [name], label=f"unsafe{index}"))

    def test_unsupported_data_schemes_and_foreign_data_roots_are_rejected(self):
        for index, name in enumerate(("bad-1.0.data/scripts/command", "bad-1.0.data/headers/header.h", "different-1.0.data/purelib/a.py")):
            with self.subTest(name=name):
                with self.assertRaises(overlap.InventoryError):
                    overlap.wheel_inventory(self.wheel("bad", [name], label=f"data{index}"))

    def test_duplicate_raw_paths_and_aliased_destinations_are_rejected(self):
        cases = [
            ["pkg.py", "pkg.py"],
            ["pkg.py", "bad-1.0.data/purelib/pkg.py"],
            ["bad-1.0.data/purelib/pkg.py", "bad-1.0.data/platlib/pkg.py"],
            ["pkg", "pkg/child.py"],
        ]
        for index, files in enumerate(cases):
            with self.subTest(files=files), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with self.assertRaises(overlap.InventoryError):
                    overlap.wheel_inventory(self.wheel("bad", files, label=f"duplicate{index}"))

    def test_symlink_entries_are_rejected(self):
        path = self.wheel("bad", [])
        with zipfile.ZipFile(path, "a") as archive:
            info = zipfile.ZipInfo("learnbywire/__init__.py")
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            archive.writestr(info, "../other.py")
        with self.assertRaisesRegex(overlap.InventoryError, "special file"):
            overlap.wheel_inventory(path)

    def test_multiple_metadata_directories_are_rejected(self):
        path = self.wheel("bad", ["other-1.0.dist-info/METADATA"])
        with self.assertRaisesRegex(overlap.InventoryError, "exactly one"):
            overlap.wheel_inventory(path)

    def test_installed_same_owner_version_still_counts(self):
        candidate = overlap.wheel_inventory(self.wheel("learnbywire-trainer", ["learnbywire/__init__.py"]))
        distribution = FakeDistribution(self.root / "site-packages", "learnbywire-trainer", "1.0", ["learnbywire/__init__.py", "learnbywire_trainer-1.0.dist-info/RECORD", "../../../bin/trainer"])
        with patch.object(overlap.importlib.metadata, "distributions", return_value=[distribution]):
            inventories = overlap.installed_inventories()
        self.assertEqual(inventories[0]["files"], {"learnbywire/__init__.py"})
        self.assertEqual(len(overlap.compare(candidate, inventories)["collisions"]), 1)

    def test_installed_distribution_without_manifest_is_rejected(self):
        distribution = FakeDistribution(self.root, "unknown", "1.0", None)
        with patch.object(overlap.importlib.metadata, "distributions", return_value=[distribution]):
            with self.assertRaisesRegex(overlap.InventoryError, "no file manifest"):
                overlap.installed_inventories()

    def test_cli_json_exit_codes(self):
        candidate = self.wheel("candidate", ["learnbywire/__init__.py"])
        legacy = self.wheel("legacy", ["learnbywire/__init__.py"])
        clean = self.wheel("clean", ["optimizerplus/__init__.py"])
        unsafe = self.wheel("unsafe", ["../escape.py"])
        for against, expected, status in ((legacy, 1, "overlap"), (clean, 0, "no_file_overlap"), (unsafe, 2, "invalid")):
            with self.subTest(status=status):
                result = subprocess.run([sys.executable, str(TOOL), "--candidate", str(candidate), "--against", str(against)], capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual(json.loads(result.stdout)["status"], status)


if __name__ == "__main__":
    unittest.main()
