"""Reuse verified native files from source wheels in setuptools build_ext."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zipfile import ZipFile

try:
  import tomllib
except ModuleNotFoundError:
  import tomli as tomllib
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"][
    "source-wheel"
]


def source() -> dict:
  try:
    top = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
    if Path(top).resolve() != ROOT:
      raise ValueError("not the package checkout")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    # uv writes this completion marker into its cached Git checkouts.
    # It is not a source change and must not disable wheel reuse.
    dirty = any(line != "?? .ok" for line in changes)
    return {
        "repository": CONFIG["repository"],
        "commit": commit,
        "dirty": dirty,
    }
  except (OSError, subprocess.CalledProcessError, ValueError):
    path = ROOT / "build-source.json"
    if not path.exists():
      raise RuntimeError(
          "A Git checkout or build-source.json is required"
      ) from None
    return json.loads(path.read_text())


def profile() -> dict:
  result = dict(CONFIG["profile"])
  # CUDA wheels must match an explicit build contract, even on consumers
  # which have no nvcc. The CI image supplies this same toolkit version.
  if "cuda" in result:
    result["cuda"] = os.environ.get("CYCLOPS_CUDA_VERSION", result["cuda"])
    result["architectures"] = os.environ.get(
        "CYCLOPS_CUDA_ARCHITECTURES", result["architectures"]
    )
  return result


class SafeRedirect(HTTPRedirectHandler):

  def redirect_request(self, request, fp, code, msg, headers, newurl):
    redirected = super().redirect_request(
        request, fp, code, msg, headers, newurl
    )
    if redirected is not None:
      # GitHub assets redirect to object storage on a different host.
      redirected.remove_header("Authorization")
    return redirected


def download(url: str, *, binary: bool = False) -> bytes:
  if not url.startswith("https://"):
    raise ValueError("Wheel downloads require HTTPS")
  headers = {
      "Accept": (
          "application/octet-stream"
          if binary
          else "application/vnd.github+json"
      )
  }
  token = os.environ.get("TOOLCHAIN_GITHUB_TOKEN")
  if token and url.startswith("https://api.github.com/"):
    headers["Authorization"] = f"Bearer {token}"
  with build_opener(SafeRedirect()).open(
      Request(url, headers=headers), timeout=60
  ) as response:
    return response.read()


def manifest_name(info: dict) -> str:
  return f"{CONFIG['distribution']}-{info['commit']}.json"


def published(info: dict) -> tuple[dict, dict[str, str]] | None:
  url = f"https://api.github.com/repos/{CONFIG['repository']}/releases/tags/wheels-{info['commit']}"
  try:
    release = json.loads(download(url))
    assets = {asset["name"]: asset["url"] for asset in release["assets"]}
    name = manifest_name(info)
    if name not in assets:
      return None
    return json.loads(download(assets[name], binary=True)), assets
  except HTTPError as exc:
    if exc.code == 404:
      return None
    if exc.code < 500:
      raise
    warnings.warn(f"Wheel service unavailable ({exc.code}); build the source")
  except (URLError, TimeoutError) as exc:
    warnings.warn(
        f"Wheel service unavailable ({type(exc).__name__}); build the source"
    )
  return None


def select(manifest: dict, info: dict) -> dict | None:
  if manifest.get("schema") != 1:
    raise ValueError("Unsupported wheel manifest schema")
  if any(manifest.get(key) != info[key] for key in ("repository", "commit")):
    raise ValueError("Wheel manifest does not match the source commit")
  supported = list(sys_tags())
  candidates = []
  for entry in manifest["wheels"]:
    name, _, _, tags = parse_wheel_filename(entry["filename"])
    if (
        name != canonicalize_name(CONFIG["distribution"])
        or entry["profile"] != profile()
    ):
      continue
    ranks = [supported.index(tag) for tag in tags if tag in supported]
    if ranks:
      candidates.append((min(ranks), entry["filename"], entry))
  return min(candidates)[2] if candidates else None


def verify(path: Path, entry: dict, info: dict) -> None:
  if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
    raise ValueError(f"Wheel checksum mismatch: {path.name}")
  with ZipFile(path) as wheel:
    provenance = json.loads(wheel.read(f"{CONFIG['module']}/toolchain.json"))
  expected = {**info, "profile": profile()}
  if provenance != expected:
    raise ValueError(f"Wheel provenance mismatch: {path.name}")


def prebuilt(info: dict, directory: Path) -> Path | None:
  if info["dirty"] or os.environ.get("BELFORT_FORCE_SOURCE") == "1":
    return None
  if not re.fullmatch(r"[0-9a-f]{40}", info["commit"]):
    raise ValueError("Expected a full source commit")
  local = os.environ.get("BELFORT_WHEEL_DIR")
  if local:
    manifest_path = Path(local) / manifest_name(info)
    if manifest_path.exists():
      entry = select(json.loads(manifest_path.read_text()), info)
      if entry:
        path = Path(local) / entry["filename"]
        verify(path, entry, info)
        return path
  remote = published(info)
  if remote is None:
    return None
  manifest, assets = remote
  entry = select(manifest, info)
  if entry is None:
    return None
  path = directory / entry["filename"]
  path.write_bytes(download(assets[path.name], binary=True))
  verify(path, entry, info)
  return path


def configure_git_version() -> None:
  """Keep Git versions stable; let setuptools-scm read release PKG-INFO."""
  key = "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_" + CONFIG[
      "distribution"
  ].upper().replace("-", "_")
  if (
      (ROOT / "PKG-INFO").is_file()
      or key in os.environ
      or "SETUPTOOLS_SCM_PRETEND_VERSION" in os.environ
  ):
    return
  info = source()
  os.environ[key] = (
      "0.0.0+g" + info["commit"] + (".dirty" if info["dirty"] else "")
  )


class SourceDistribution(sdist):

  def make_release_tree(self, base_dir, files):
    info = source()
    super().make_release_tree(base_dir, files)
    path = Path(base_dir) / "build-source.json"
    # setuptools can hardlink source files into the release tree.
    path.unlink(missing_ok=True)
    path.write_text(json.dumps(info))


def record_source(build_lib: Path, info: dict) -> None:
  path = build_lib / CONFIG["module"] / "toolchain.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps({**info, "profile": profile()}, sort_keys=True))


def restore_native(build_lib: Path, info: dict) -> bool:
  """Restore only native files; setuptools owns all Python package metadata."""
  with tempfile.TemporaryDirectory() as temporary:
    candidate = prebuilt(info, Path(temporary))
    if candidate is None:
      if os.environ.get("BELFORT_REQUIRE_WHEEL") == "1":
        raise RuntimeError(
            "No verified wheel matches the source commit and build profile"
        )
      return False
    with ZipFile(candidate) as wheel:
      for member in wheel.infolist():
        name = PurePosixPath(member.filename)
        if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
          raise ValueError(f"Unsafe wheel member: {member.filename}")
        if member.is_dir() or not any(
            member.filename == prefix
            or member.filename.startswith(prefix + "/")
            for prefix in CONFIG["payload"]
        ):
          continue
        target = build_lib / member.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(wheel.read(member))
        target.chmod((member.external_attr >> 16) & 0o777 or 0o644)
    record_source(build_lib, info)
    return True


def write_manifest(directory: Path) -> Path:
  entries = []
  identity = None
  for path in sorted(directory.glob("*.whl")):
    name, _, _, _ = parse_wheel_filename(path.name)
    if name != canonicalize_name(CONFIG["distribution"]):
      continue
    with ZipFile(path) as wheel:
      info = json.loads(wheel.read(f"{CONFIG['module']}/toolchain.json"))
    if info["dirty"]:
      raise ValueError("Cannot publish a wheel from a dirty checkout")
    current = {key: info[key] for key in ("repository", "commit")}
    if identity is not None and current != identity:
      raise ValueError("Wheels have different source commits")
    identity = current
    entries.append({
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "profile": info["profile"],
    })
  if not entries:
    raise ValueError("No wheels found")
  output = directory / manifest_name(identity)
  output.write_text(
      json.dumps({"schema": 1, **identity, "wheels": entries}, indent=2) + "\n"
  )
  return output


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Create a source wheel manifest")
  parser.add_argument("directory", type=Path)
  print(write_manifest(parser.parse_args().directory))
