#!/usr/bin/env python3
"""Cross-platform local validation for DataQualityEnv."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = (
    ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
)
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
PORT = 8010
OPENENV = shutil.which("openenv")
DOCKER = shutil.which("docker")

if OPENENV is None:
    candidate_name = "openenv.exe" if os.name == "nt" else "openenv"
    candidate = Path(PYTHON).with_name(candidate_name)
    if candidate.exists():
        OPENENV = str(candidate)


def run_step(label: str, command: list[str], cwd: Path | None = None) -> None:
    print(f"[RUN] {label}: {' '.join(command)}")
    subprocess.run(command, cwd=cwd or ROOT, check=True)


def wait_for_healthcheck(url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}")


def smoke_test_server() -> None:
    command = [
        PYTHON,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
    ]
    print(f"[RUN] server smoke test: {' '.join(command)}")
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_healthcheck(f"http://127.0.0.1:{PORT}/health")
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/tasks", timeout=5) as response:
            tasks = json.load(response)
        if len(tasks) < 3:
            raise RuntimeError("Expected at least 3 tasks from /tasks")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def maybe_run_docker_build() -> None:
    if DOCKER is None:
        print("[SKIP] Docker CLI not found; skipping local docker build.")
        return

    try:
        subprocess.run(
            [DOCKER, "info"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        print("[SKIP] Docker daemon is not reachable; skipping local docker build.")
        return

    run_step("docker build", [DOCKER, "build", ".", "-t", "data-quality-env:preflight"])


def main() -> None:
    if OPENENV is None:
        raise RuntimeError("Could not find the 'openenv' CLI in the active environment.")

    run_step("openenv validate", [OPENENV, "validate"])
    run_step("unit tests", [PYTHON, "-m", "unittest", "discover", "-s", "tests", "-v"])
    run_step("baseline inference", [PYTHON, "inference.py"])
    smoke_test_server()
    maybe_run_docker_build()
    print("[OK] Local preflight checks passed.")


if __name__ == "__main__":
    main()
