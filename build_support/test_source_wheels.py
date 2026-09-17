"""Validate native reuse through the standard setuptools backend."""

import hashlib
import json
import subprocess
import os
import shutil
import tarfile
import sys
from pathlib import Path
from urllib.request import Request
from zipfile import ZipInfo

import pytest
from packaging.tags import sys_tags
from wheel.wheelfile import WheelFile

sys.path.insert(0, str(Path(__file__).parent))
import source_wheels as backend


@pytest.fixture
def info(monkeypatch):
  monkeypatch.setattr(
      backend,
      "CONFIG",
      {
          "distribution": "belfort-example",
          "module": "example",
          "payload": ["example/compiler"],
          "repository": "belfortlabs/example",
          "profile": {"compiler": "opt"},
      },
  )
  monkeypatch.delenv("BELFORT_FORCE_SOURCE", raising=False)
  monkeypatch.delenv("BELFORT_WHEEL_DIR", raising=False)
  monkeypatch.delenv("BELFORT_REQUIRE_WHEEL", raising=False)
  return {
      "repository": "belfortlabs/example",
      "commit": "a" * 40,
      "dirty": False,
  }


@pytest.fixture
def artifact(tmp_path, info):
  filename = f"belfort_example-1.0-{next(sys_tags())}.whl"
  path = tmp_path / filename
  with WheelFile(path, "w") as wheel:
    binary = ZipInfo("example/compiler")
    binary.external_attr = 0o100755 << 16
    wheel.writestr(binary, b"native payload")
    wheel.writestr("example/__init__.py", b"old Python code")
    wheel.writestr(
        "example/toolchain.json",
        json.dumps({**info, "profile": backend.profile()}),
    )
    wheel.writestr(
        "belfort_example-1.0.dist-info/METADATA",
        "Metadata-Version: 2.1\nName: belfort-example\nVersion: 1.0\n",
    )
    wheel.writestr(
        "belfort_example-1.0.dist-info/WHEEL",
        "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag:"
        f" {next(sys_tags())}\n",
    )
  backend.write_manifest(tmp_path)
  return path


def test_restore_keeps_python_sources_and_cached_wheel(
    tmp_path, monkeypatch, info, artifact
):
  monkeypatch.setenv("BELFORT_WHEEL_DIR", str(tmp_path))
  monkeypatch.setattr(
      backend, "published", lambda *_: pytest.fail("Unexpected network access")
  )
  package = tmp_path / "output" / "example"
  package.mkdir(parents=True)
  (package / "__init__.py").write_text("new Python code")
  before = artifact.read_bytes()
  for _ in range(2):
    assert backend.restore_native(package.parent, info)
    assert (package / "compiler").read_bytes() == b"native payload"
    assert (package / "compiler").stat().st_mode & 0o111
    assert (package / "__init__.py").read_text() == "new Python code"
    assert not list(package.parent.glob("*.dist-info"))
    assert artifact.read_bytes() == before


@pytest.mark.parametrize(
    "field,value", [("commit", "b" * 40), ("repository", "other/repo")]
)
def test_rejects_other_source(info, artifact, field, value):
  manifest = json.loads(
      (artifact.parent / backend.manifest_name(info)).read_text()
  )
  manifest[field] = value
  with pytest.raises(ValueError, match="source commit"):
    backend.select(manifest, info)


def test_incompatible_profile_is_a_miss(info, artifact):
  manifest = json.loads(
      (artifact.parent / backend.manifest_name(info)).read_text()
  )
  manifest["wheels"][0]["profile"] = {"compiler": "debug"}
  assert backend.select(manifest, info) is None


def test_incompatible_platform_is_a_miss(info, artifact):
  manifest = json.loads(
      (artifact.parent / backend.manifest_name(info)).read_text()
  )
  manifest["wheels"][0][
      "filename"
  ] = "belfort_example-1.0-cp399-cp399-impossible.whl"
  assert backend.select(manifest, info) is None


def test_corrupt_artifact_fails(info, artifact):
  entry = json.loads(
      (artifact.parent / backend.manifest_name(info)).read_text()
  )["wheels"][0]
  artifact.write_bytes(b"corrupt")
  with pytest.raises(ValueError, match="checksum"):
    backend.verify(artifact, entry, info)


def test_embedded_identity_is_checked(info, artifact):
  entry = {"sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
  with pytest.raises(ValueError, match="provenance"):
    backend.verify(artifact, entry, {**info, "commit": "b" * 40})


@pytest.mark.parametrize("dirty,forced", [(True, False), (False, True)])
def test_custom_source_bypasses_wheels(
    info, tmp_path, monkeypatch, dirty, forced
):
  monkeypatch.setenv("BELFORT_FORCE_SOURCE", "1" if forced else "0")
  monkeypatch.setattr(
      backend, "published", lambda *_: pytest.fail("Unexpected network access")
  )
  assert backend.prebuilt({**info, "dirty": dirty}, tmp_path) is None


def test_missing_wheel_allows_source_unless_required(
    info, tmp_path, monkeypatch
):
  monkeypatch.setattr(backend, "prebuilt", lambda *_: None)
  assert not backend.restore_native(tmp_path, info)
  monkeypatch.setenv("BELFORT_REQUIRE_WHEEL", "1")
  with pytest.raises(RuntimeError, match="No verified wheel"):
    backend.restore_native(tmp_path, info)


def test_redirect_drops_credentials():
  request = Request(
      "https://api.github.com/assets/1",
      headers={"Authorization": "Bearer secret"},
  )
  result = backend.SafeRedirect().redirect_request(
      request, None, 302, "", {}, "https://objects.example/wheel"
  )
  assert result.get_header("Authorization") is None


@pytest.mark.parametrize(
    "filename,dirty", [(".ok", False), ("source.cc", True)]
)
def test_uv_completion_marker_is_not_a_source_change(
    tmp_path, monkeypatch, info, filename, dirty
):
  subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
  subprocess.run(
      [
          "git",
          "-C",
          str(tmp_path),
          "-c",
          "user.name=Test",
          "-c",
          "user.email=test@example.invalid",
          "-c",
          "commit.gpgsign=false",
          "commit",
          "--allow-empty",
          "-qm",
          "test",
      ],
      check=True,
  )
  (tmp_path / filename).touch()
  monkeypatch.setattr(backend, "ROOT", tmp_path)
  assert backend.source()["dirty"] is dirty


@pytest.fixture
def project(tmp_path, info):
  root = tmp_path / "project"
  root.mkdir()
  (root / "build_support").mkdir()
  shutil.copyfile(
      Path(backend.__file__), root / "build_support" / "source_wheels.py"
  )
  (root / "build_support" / "__init__.py").touch()
  (root / "example").mkdir()
  (root / "example" / "__init__.py").write_text("# Current Python sources\n")
  (root / "build-source.json").write_text(json.dumps(info))
  (root / "MANIFEST.in").write_text(
      "recursive-include build_support *.py\ninclude build-source.json\n"
  )
  (root / "pyproject.toml").write_text("""[build-system]
requires = ["setuptools", "setuptools-scm", "wheel", "packaging"]
build-backend = "setuptools.build_meta"
[project]
name = "belfort-example"
dynamic = ["version"]
[tool.setuptools_scm]
[tool.source-wheel]
distribution = "belfort-example"
module = "example"
repository = "belfortlabs/example"
payload = ["example/compiler"]
profile = { compiler = "opt" }
""")
  (root / "setup.py").write_text("""from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel
from packaging.tags import sys_tags
from build_support import source_wheels
class Native(build_ext):
    def run(self):
        info = source_wheels.source()
        if not source_wheels.restore_native(Path(self.build_lib), info):
            raise AssertionError("This test must never invoke a native compiler")
class Wheel(bdist_wheel):
    def get_tag(self):
        return tuple(str(next(sys_tags())).split("-"))
source_wheels.configure_git_version()
setup(packages=["example"], ext_modules=[Extension("example._native", [])],
      cmdclass={"build_ext": Native, "bdist_wheel": Wheel, "sdist": source_wheels.SourceDistribution})
""")
  return root


def run_hook(root, hook, destination, **environment):
  destination.mkdir(parents=True, exist_ok=True)
  env = {
      key: value
      for key, value in os.environ.items()
      if not key.startswith(("BELFORT_", "SETUPTOOLS_SCM_"))
  }
  result = subprocess.run(
      [
          sys.executable,
          "-I",
          "-c",
          (
              "from setuptools import build_meta;"
              f" print(build_meta.{hook}({str(destination)!r}))"
          ),
      ],
      cwd=root,
      env={**env, **environment},
      capture_output=True,
      text=True,
  )
  assert result.returncode == 0, result.stdout + result.stderr
  return destination / result.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_standard_backend_reuses_wheels_on_repeated_builds(
    project, artifact, tmp_path, version
):
  # With version 1.0, the cached input and the output are the same file.
  for _ in range(2):
    output = run_hook(
        project,
        "build_wheel",
        artifact.parent,
        SETUPTOOLS_SCM_PRETEND_VERSION=version,
        BELFORT_WHEEL_DIR=str(artifact.parent),
        BELFORT_REQUIRE_WHEEL="1",
    )
    with WheelFile(output) as wheel:
      assert wheel.read("example/compiler") == b"native payload"
      assert wheel.read("example/__init__.py") == b"# Current Python sources\n"
      metadata = wheel.read(
          f"belfort_example-{version}.dist-info/METADATA"
      ).decode()
      assert f"Version: {version}\n" in metadata
      for name in wheel.namelist():
        wheel.read(name)  # Verify setuptools' RECORD hashes.
    backend.write_manifest(artifact.parent)


def test_release_sdist_preserves_version_without_override(project, tmp_path):
  archive = run_hook(
      project,
      "build_sdist",
      tmp_path / "dist",
      SETUPTOOLS_SCM_PRETEND_VERSION="20260916",
  )
  unpacked = tmp_path / "unpacked"
  with tarfile.open(archive) as source_archive:
    source_archive.extractall(unpacked, filter="data")
  root = next(unpacked.iterdir())
  metadata = run_hook(
      root, "prepare_metadata_for_build_wheel", tmp_path / "metadata"
  )
  assert "Version: 20260916\n" in (metadata / "METADATA").read_text()
  assert (
      json.loads((root / "build-source.json").read_text())["commit"] == "a" * 40
  )
  assert (project / "build-source.json").read_bytes() == (
      root / "build-source.json"
  ).read_bytes()
