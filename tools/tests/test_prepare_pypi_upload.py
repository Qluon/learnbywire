"""Publication guards: reject changed artifacts, source leaks and wrong dependencies."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

module_spec = importlib.util.spec_from_file_location("prepare_pypi_upload", Path(__file__).resolve().parents[1] / "prepare_pypi_upload.py")
release = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(release)


class PublicationGuards(unittest.TestCase):
    def make_wheel(self, directory, project="learnbywire-trainerplus", extra=None):
        namespace = "optimizerplus" if project.endswith("optimizerplus") else "trainerplus"
        filename = project.replace("-", "_") + "-1.1.1-py3-none-any.whl"
        path = Path(directory) / filename
        info = project.replace("-", "_") + "-1.1.1.dist-info"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{info}/METADATA", f"Metadata-Version: 2.4\nName: {project}\nVersion: 1.1.1\nLicense-Expression: Apache-2.0\n")
            archive.writestr(namespace + "/" + ("core.so" if namespace == "optimizerplus" else "__init__.py"), b"fixture")
            for name, content in (extra or {}).items():
                archive.writestr(name, content)
        spec = {"filename": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "version": "1.1.1", "namespace": namespace, "license": "Apache-2.0", "approved_for_pypi": True}
        return path, spec

    def test_qualified_public_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            path, spec = self.make_wheel(directory)
            release.verify_wheel(path, "learnbywire-trainerplus", spec)

    def test_changed_wheel_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, spec = self.make_wheel(directory)
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                release.verify_wheel(path, "learnbywire-trainerplus", spec)

    def test_optimizerplus_source_rejected_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path, spec = self.make_wheel(directory, "learnbywire-optimizerplus", {"optimizerplus/internal.py": "private implementation"})
            with self.assertRaisesRegex(ValueError, "source or bytecode"):
                release.verify_wheel(path, "learnbywire-optimizerplus", spec)

    def test_foreign_namespace_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, spec = self.make_wheel(directory, extra={"optimizerplus/core.so": b"fixture"})
            with self.assertRaisesRegex(ValueError, "top-level"):
                release.verify_wheel(path, "learnbywire-trainerplus", spec)

    def test_unqualified_package_rejected(self):
        manifest = {"schema_version": 1, "repository": "Qluon/learnbywire", "packages": {"example": {"approved_for_pypi": False}}}
        with self.assertRaisesRegex(ValueError, "not approved"):
            release.approved_spec(manifest, "example")

    def test_existing_version_rejected(self):
        with patch.object(release, "pypi_version", return_value={"urls": []}):
            with self.assertRaisesRegex(ValueError, "already exists"):
                release.check_registry({}, "example", {"version": "1.0"})

    def test_dependency_wrong_hash_rejected(self):
        manifest = {"packages": {"child": {"version": "1.0", "filename": "child.whl", "sha256": "approved"}}}
        data = {"urls": [{"filename": "child.whl", "digests": {"sha256": "different"}}]}
        with patch.object(release, "pypi_version", side_effect=[None, data]):
            with self.assertRaisesRegex(ValueError, "does not match"):
                release.check_registry(manifest, "parent", {"version": "1.0", "requires_published": ["child"]})


if __name__ == "__main__":
    unittest.main()
