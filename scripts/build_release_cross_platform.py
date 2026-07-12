#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Build and validate a complete gigatoken PyPI release from macOS.

This produces abi3 wheels for CPython 3.10+ on macOS, manylinux2014,
and Windows, for both x86-64 and ARM64, plus a source distribution.

Prerequisites:
  * uv, Rust/rustup, and Docker
  * Homebrew LLVM 19 or newer (`brew install llvm`) for Windows cross-builds
  * A running Docker-compatible container runtime

The default is build-only. Pass --publish only after reviewing dist/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
MATURIN_VERSION = "1.14.1"
MATURIN_IMAGE = f"ghcr.io/pyo3/maturin:v{MATURIN_VERSION}"
PYTHON_VERSIONS = ("3.10", "3.11", "3.12", "3.13", "3.14")


def run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    print(f"\n+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def output(command: list[str], *, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(command, cwd=ROOT, env=env, text=True).strip()


def require_command(command: str, hint: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(f"missing required command {command!r}; {hint}")


def project_version() -> str:
    cargo_toml = (ROOT / "Cargo.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', cargo_toml, re.MULTILINE)
    if match is None:
        raise SystemExit("could not read package version from Cargo.toml")
    return match.group(1)


def maturin() -> list[str]:
    return ["uvx", "--from", f"maturin=={MATURIN_VERSION}", "maturin"]


def clean_dist() -> None:
    DIST.mkdir(exist_ok=True)
    for pattern in ("gigatoken-*.whl", "gigatoken-*.tar.gz"):
        for artifact in DIST.glob(pattern):
            artifact.unlink()


def clang_version(executable: Path) -> int | None:
    try:
        version_text = output([str(executable), "--version"])
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"clang version (\d+)", version_text)
    return int(match.group(1)) if match else None


def windows_build_env() -> dict[str, str]:
    candidates = [
        Path("/opt/homebrew/opt/llvm/bin/clang-cl"),
        Path("/usr/local/opt/llvm/bin/clang-cl"),
    ]
    path_clang = shutil.which("clang-cl")
    if path_clang:
        candidates.append(Path(path_clang))

    for candidate in candidates:
        if candidate.is_file() and (clang_version(candidate) or 0) >= 19:
            env = os.environ.copy()
            env["PATH"] = f"{candidate.parent}{os.pathsep}{env['PATH']}"
            return env

    raise SystemExit("Windows cross-builds require clang-cl 19 or newer. Install it with `brew install llvm`.")


def build_macos_wheels() -> None:
    targets = {
        "x86_64-apple-darwin": "10.12",
        "aarch64-apple-darwin": "11.0",
    }
    run(["rustup", "target", "add", *targets])
    for target, deployment_target in targets.items():
        env = os.environ.copy()
        env["MACOSX_DEPLOYMENT_TARGET"] = deployment_target
        run(
            [
                *maturin(),
                "build",
                "--release",
                "--locked",
                "--compatibility",
                "pypi",
                "--target",
                target,
                "--out",
                str(DIST),
            ],
            env=env,
        )


def build_windows_wheels() -> None:
    targets = ("x86_64-pc-windows-msvc", "aarch64-pc-windows-msvc")
    run(["rustup", "target", "add", *targets])
    env = windows_build_env()
    for target in targets:
        run(
            [
                *maturin(),
                "build",
                "--release",
                "--locked",
                "--compatibility",
                "pypi",
                "--target",
                target,
                "--out",
                str(DIST),
            ],
            env=env,
        )


def build_linux_wheel(architecture: str, target: str) -> None:
    docker_platform = {"x86_64": "linux/amd64", "aarch64": "linux/arm64"}[architecture]
    container_command = (
        f"rustup toolchain install nightly --profile minimal && maturin build --release --locked --target {target} --manylinux 2014 --target-dir /tmp/gigatoken-target --out /out"
    )
    run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            docker_platform,
            "--entrypoint",
            "bash",
            "-v",
            f"{ROOT}:/io:ro",
            "-v",
            f"{DIST}:/out",
            "-w",
            "/io",
            MATURIN_IMAGE,
            "-lc",
            container_command,
        ]
    )


def build_linux_wheels() -> None:
    run(["docker", "info"])
    build_linux_wheel("x86_64", "x86_64-unknown-linux-gnu")
    build_linux_wheel("aarch64", "aarch64-unknown-linux-gnu")


def build_sdist() -> None:
    run([*maturin(), "sdist", "--out", str(DIST)])


def expected_artifacts(version: str) -> list[Path]:
    names = [
        f"gigatoken-{version}-cp310-abi3-macosx_10_12_x86_64.whl",
        f"gigatoken-{version}-cp310-abi3-macosx_11_0_arm64.whl",
        f"gigatoken-{version}-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        f"gigatoken-{version}-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
        f"gigatoken-{version}-cp310-abi3-win_amd64.whl",
        f"gigatoken-{version}-cp310-abi3-win_arm64.whl",
        f"gigatoken-{version}.tar.gz",
    ]
    artifacts = [DIST / name for name in names]
    missing = [path.name for path in artifacts if not path.is_file()]
    if missing:
        raise SystemExit(f"release build is incomplete; missing: {', '.join(missing)}")
    return artifacts


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def smoke_test_wheel(wheel: Path) -> None:
    smoke_code = (
        "import importlib.metadata, gigatoken; "
        "assert importlib.metadata.version('gigatoken') == VERSION; "
        "tokens = list(gigatoken.pretokenizer(b'Hello, world!')); "
        "assert b''.join(tokens) == b'Hello, world!'"
    )
    version = project_version()
    smoke_code = f"VERSION = {version!r}; {smoke_code}"

    with tempfile.TemporaryDirectory(prefix="gigatoken-wheel-test-") as temp:
        temp_path = Path(temp)
        for python_version in PYTHON_VERSIONS:
            venv = temp_path / python_version.replace(".", "")
            run(["uv", "venv", "--python", python_version, str(venv)])
            run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(venv_python(venv)),
                    "--no-deps",
                    str(wheel),
                ]
            )
            env = os.environ.copy()
            env["VIRTUAL_ENV"] = str(venv)
            env["PATH"] = f"{venv_python(venv).parent}{os.pathsep}{env['PATH']}"
            run(
                [
                    "uv",
                    "run",
                    "--active",
                    "--no-project",
                    "python",
                    "-c",
                    smoke_code,
                ],
                cwd=temp_path,
                env=env,
            )


def smoke_test_sdist(sdist: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="gigatoken-sdist-test-") as temp:
        temp_path = Path(temp)
        venv = temp_path / "venv"
        run(["uv", "venv", "--python", "3.14", str(venv)])
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv_python(venv)),
                "--no-deps",
                str(sdist),
            ]
        )
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = f"{venv_python(venv).parent}{os.pathsep}{env['PATH']}"
        run(
            [
                "uv",
                "run",
                "--active",
                "--no-project",
                "python",
                "-c",
                "import gigatoken; assert b''.join(gigatoken.pretokenizer(b'sdist works')) == b'sdist works'",
            ],
            cwd=temp_path,
            env=env,
        )


def validate(artifacts: list[Path], *, skip_smoke_tests: bool) -> None:
    run(["uvx", "--from", "twine", "twine", "check", *map(str, artifacts)])
    run(["uv", "publish", "--dry-run", *map(str, artifacts)])

    if not skip_smoke_tests:
        machine = platform.machine().lower()
        native_tag = "arm64" if machine in {"arm64", "aarch64"} else "x86_64"
        native_wheel = next(path for path in artifacts if path.suffix == ".whl" and "macosx" in path.name and native_tag in path.name)
        smoke_test_wheel(native_wheel)
        smoke_test_sdist(next(path for path in artifacts if path.name.endswith(".tar.gz")))


def check_pypi_version(version: str, *, required: bool) -> None:
    url = "https://pypi.org/pypi/gigatoken/json"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            releases = json.load(response).get("releases", {})
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        if required:
            raise SystemExit(f"could not verify the version against PyPI: {error}")
        print(f"warning: could not check existing PyPI releases: {error}")
        return
    if version in releases:
        raise SystemExit(f"gigatoken {version} already exists on PyPI")


def print_hashes(artifacts: list[Path]) -> None:
    print("\nRelease artifacts:")
    for artifact in artifacts:
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        print(f"{digest}  {artifact.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="upload the validated artifacts to PyPI",
    )
    parser.add_argument(
        "--skip-smoke-tests",
        action="store_true",
        help="skip wheel tests on Python 3.10-3.14 and the sdist install test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sys.platform != "darwin":
        raise SystemExit("the complete release build currently requires macOS so both macOS architectures can be produced locally")

    require_command("uv", "install uv from https://docs.astral.sh/uv/")
    version = project_version()
    check_pypi_version(version, required=args.publish)

    if args.publish:
        artifacts = expected_artifacts(version)
        validate(artifacts, skip_smoke_tests=args.skip_smoke_tests)
        print_hashes(artifacts)
        run(["uv", "publish", *map(str, artifacts)])
        return

    require_command("cargo", "install Rust with rustup")
    require_command("rustup", "install Rust with rustup")
    require_command("docker", "install and start Docker Desktop or OrbStack")

    clean_dist()
    run(["cargo", "check", "--lib", "--locked"])
    build_macos_wheels()
    build_windows_wheels()
    build_linux_wheels()
    build_sdist()

    artifacts = expected_artifacts(version)
    validate(artifacts, skip_smoke_tests=args.skip_smoke_tests)
    print_hashes(artifacts)

    print("\nBuild complete. Review dist/, then publish with:")
    print("  uv run scripts/build_release.py --publish")


if __name__ == "__main__":
    main()
