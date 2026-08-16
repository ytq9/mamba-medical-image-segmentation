#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


DEFAULT_PACKAGES = [
    "causal-conv1d>=1.4.0",
    "mamba-ssm>=2.2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure mamba_ssm is importable in the current Python environment.")
    parser.add_argument("--check-only", action="store_true", help="Only check importability; do not install anything.")
    parser.add_argument("--force", action="store_true", help="Run pip install even if mamba_ssm already imports.")
    parser.add_argument(
        "--package",
        action="append",
        default=None,
        help="Override package spec. Can be repeated. Defaults to causal-conv1d and mamba-ssm.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ok, message = probe_mamba_ssm()
    if ok and not args.force:
        print(f"mamba_ssm already available: {message}")
        return
    if args.check_only:
        raise SystemExit(f"mamba_ssm unavailable: {message}")

    packages = args.package or DEFAULT_PACKAGES
    command = [sys.executable, "-m", "pip", "install", "--no-build-isolation", *packages]
    print("Installing Mamba dependencies:")
    print(" ".join(command))
    subprocess.check_call(command)

    ok, message = probe_mamba_ssm()
    if not ok:
        raise SystemExit(f"mamba_ssm installation finished, but import still fails: {message}")
    print(f"mamba_ssm ready: {message}")


def probe_mamba_ssm() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        from mamba_ssm import Mamba  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - report exact import/build failure to the user
        return False, f"{type(exc).__name__}: {exc}"
    return True, "from mamba_ssm import Mamba succeeded"


if __name__ == "__main__":
    main()
