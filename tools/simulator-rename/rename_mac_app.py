#!/usr/bin/env python3
"""Rename a macOS Donkey simulator bundle to the "Donkey Sim" spelling.

Renames ``donkey_sim.app`` (or the unique bundle below a parent directory) to
``Donkey Sim.app``, renames the inner ``Contents/MacOS/donkey_sim`` executable
and updates ``CFBundleName`` / ``CFBundleExecutable`` in ``Info.plist``.  The
bundle identifier is deliberately left untouched.  Renaming invalidates the
code signature, so the bundle is re-signed ad-hoc by default; an unsigned
bundle is ignored by some macOS lookups (appearance variants, asset catalogs).

Run this on the extracted app before zipping the release asset.
"""
import argparse
import plistlib
import subprocess
from pathlib import Path

NEW_NAME = "Donkey Sim"
OLD_NAME = "donkey_sim"


def run(command):
    subprocess.run([str(part) for part in command], check=True)


def resolve_bundle(app: Path) -> Path:
    if app.name in (f"{OLD_NAME}.app", f"{NEW_NAME}.app"):
        return app
    matches = sorted(app.glob(f"{OLD_NAME}.app")) + sorted(app.glob(f"{NEW_NAME}.app"))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {OLD_NAME}.app under {app}, found {len(matches)}")
    return matches[0]


def rename_inner_executable(app: Path) -> None:
    macos_dir = app / "Contents" / "MacOS"
    old = macos_dir / OLD_NAME
    new = macos_dir / NEW_NAME
    if new.exists():
        _ensure_executable(new)
        return
    if not old.is_file():
        raise SystemExit(f"no {OLD_NAME} or {NEW_NAME} executable inside {macos_dir}")
    old.rename(new)
    _ensure_executable(new)


def _ensure_executable(path: Path) -> None:
    # Archives may not carry the executable bit for the player binary; without it
    # LaunchServices refuses to spawn the bundle.
    path.chmod(path.stat().st_mode | 0o111)


def patch_info_plist(app: Path) -> None:
    plist_path = app / "Contents" / "Info.plist"
    with open(plist_path, "rb") as stream:
        info = plistlib.load(stream)
    if info.get("CFBundleExecutable") not in (OLD_NAME, NEW_NAME):
        raise SystemExit(f"unexpected CFBundleExecutable: {info.get('CFBundleExecutable')!r}")
    info["CFBundleName"] = NEW_NAME
    info["CFBundleExecutable"] = NEW_NAME
    with open(plist_path, "wb") as stream:
        plistlib.dump(info, stream, sort_keys=False)


def rename_bundle(app: Path) -> Path:
    if app.name == f"{NEW_NAME}.app":
        return app
    renamed = app.with_name(f"{NEW_NAME}.app")
    if renamed.exists():
        raise SystemExit(f"target already exists: {renamed}")
    app.rename(renamed)
    return renamed


def resign(app: Path) -> None:
    run(["codesign", "--force", "--deep", "-s", "-", app])
    run(["codesign", "--verify", "--strict", app])


def main():
    parser = argparse.ArgumentParser(description="Rename a donkey_sim.app bundle to Donkey Sim.app.")
    parser.add_argument("app", type=Path, help="path to the bundle, or a directory containing it")
    parser.add_argument("--skip-sign", action="store_true", help="skip the ad-hoc re-signing step")
    args = parser.parse_args()

    if not args.app.is_dir():
        raise SystemExit(f"not a directory: {args.app}")
    app = resolve_bundle(args.app)

    rename_inner_executable(app)
    patch_info_plist(app)
    app = rename_bundle(app)
    if not args.skip_sign:
        resign(app)
    print(f"renamed bundle: {app}")


if __name__ == "__main__":
    main()
