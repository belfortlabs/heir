"""Validate source identity, binary reuse, and the PEP 517 metadata contract."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import Request
from zipfile import ZipFile

import pytest
from packaging.tags import sys_tags
from wheel.wheelfile import WheelFile

sys.path.insert(0, str(Path(__file__).parent))
import wheel_backend as backend


@pytest.fixture
def info(monkeypatch):
  monkeypatch.setattr(
      backend,
      "CONFIG",
      {
          "distribution": "belfort-example",
          "module": "example",
          "repository": "belfortlabs/example",
          "profile": {"compiler": "opt"},
      },
  )
  monkeypatch.delenv("BELFORT_FORCE_SOURCE", raising=False)
  monkeypatch.delenv("BELFORT_WHEEL_DIR", raising=False)
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
    wheel.writestr("example/compiler", b"native payload")
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


def test_exact_commit_reuses_payload_without_compilation(
    tmp_path, monkeypatch, info, artifact
):
  monkeypatch.setenv("BELFORT_WHEEL_DIR", str(tmp_path))
  monkeypatch.setattr(backend, "source", lambda: info)

  def forbidden(*args, **kwargs):
    pytest.fail(
        "A cached wheel must not trigger a native build or network access"
    )

  monkeypatch.setattr(backend.delegate, "build_wheel", forbidden)
  monkeypatch.setattr(backend, "published", forbidden)
  metadata = tmp_path / "belfort_example-1.1.dev0.dist-info"
  metadata.mkdir()
  expected = "Metadata-Version: 2.1\nName: belfort-example\nVersion: 1.1.dev0\n"
  (metadata / "METADATA").write_text(expected)
  (metadata / "top_level.txt").write_text("example\n")
  destination = tmp_path / "output"
  result = backend.build_wheel(
      str(destination), metadata_directory=str(metadata)
  )
  assert result.startswith("belfort_example-1.1.dev0-")
  with WheelFile(destination / result) as wheel:
    assert wheel.read("example/compiler") == b"native payload"
    assert wheel.read(metadata.name + "/METADATA").decode() == expected
    assert wheel.read(metadata.name + "/top_level.txt") == b"example\n"
    assert wheel.read(metadata.name + "/WHEEL")
    assert not any(
        name.startswith("belfort_example-1.0.dist-info")
        for name in wheel.namelist()
    )
    # WheelFile validates RECORD hashes when the entries are read.
    for name in wheel.namelist():
      wheel.read(name)


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


def test_missing_wheel_uses_source_backend(
    info, tmp_path, artifact, monkeypatch
):
  monkeypatch.setattr(backend, "source", lambda: info)
  monkeypatch.setattr(backend, "prebuilt", lambda *_: None)
  calls = []

  def build(directory, config_settings, metadata_directory):
    calls.append(config_settings)
    (Path(directory) / artifact.name).write_bytes(artifact.read_bytes())
    return artifact.name

  monkeypatch.setattr(backend.delegate, "build_wheel", build)
  name = backend.build_wheel(str(tmp_path / "output"), {"option": "value"})
  assert calls == [{"option": "value"}]
  with ZipFile(tmp_path / "output" / name) as wheel:
    assert (
        json.loads(wheel.read("example/toolchain.json"))["commit"]
        == info["commit"]
    )


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


@pytest.mark.parametrize("release", [None, "20260916"])
def test_source_version_is_stable_and_preserves_release_override(
    monkeypatch, info, release
):
  import os

  monkeypatch.setitem(backend.CONFIG, "scm-version", True)
  monkeypatch.setattr(backend, "source", lambda: info)
  key = "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BELFORT_EXAMPLE"
  monkeypatch.delenv(key, raising=False)
  monkeypatch.delenv("SETUPTOOLS_SCM_PRETEND_VERSION", raising=False)
  if release:
    monkeypatch.setenv("SETUPTOOLS_SCM_PRETEND_VERSION", release)

  @backend.source_version
  def metadata():
    return os.environ.get("SETUPTOOLS_SCM_PRETEND_VERSION") or os.environ[key]

  assert metadata() == (release or "0.0.0+g" + info["commit"])
  assert key not in os.environ
