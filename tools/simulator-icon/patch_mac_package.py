#!/usr/bin/env python3
"""Patch a DonkeySimMac release zip with the app-icon appearance feature.

Takes an already-button-patched macOS release zip and produces a new zip
carrying, on top of everything already inside:

  - Assembly-CSharp.dll — Cecil-injected so CarRecovery.Update also calls
    DonkeySimUpdate.SimulatorAppIconHook.Install (and re-anchors the update
    button hook alongside). The injected call is the activation path: a
    loose DLL's [RuntimeInitializeOnLoadMethod] does NOT fire in a built
    player, and the shipped 2022.3.49 base never runs the historical
    TcpMenuHandler.Start anchor — CarRecovery.Update is verified to execute
    every frame from first boot; both Install methods are idempotent.
  - Contents/Resources/Data/Managed/SimulatorAppIconHook.dll — the runtime
    Dock-tile switcher + in-menu Auto/Light/Dark setting (compiled here with
    the Unity editor's mcs against the package's own UnityEngine modules).
  - Contents/Resources/Data/ScriptingAssemblies.json — the hook DLL
    registered (names + types:16), mirroring the button hook entry.
  - Contents/Resources/icon-day.png / icon-night.png — the light artwork
    (byte-identical to the shipped icon) and the dark variant (same subject
    on pure #000000).
  - Contents/Resources/Assets.car + CFBundleIconName — the dual-appearance
    bundle catalog (actool-compiled from tools/simulator-icon/AppIcon.icon)
    so Finder/Launchpad/Dock follow the system appearance.
  - ad-hoc re-seal — macOS ignores the appearance catalog on unsigned
    bundles (verified live), and the additions invalidate the previous seal.

Usage:
  patch_mac_package.py <source.zip> <output.zip> [--mono-root DIR]

--mono-root defaults to the Unity 2020.3.49f1 editor's MonoBleedingEdge
(mcs output runs fine on the 2022.3 player the simulator ships with).
"""
import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

HOOK_DLL_NAME = "SimulatorAppIconHook.dll"
TOOL_DIR = Path(__file__).resolve().parent
UPDATE_BUTTON_DIR = TOOL_DIR.parent / "simulator-update-button"
ICON_HOOK_SOURCE = UPDATE_BUTTON_DIR / "SimulatorAppIconHook.cs"
BUTTON_HOOK_SOURCE = UPDATE_BUTTON_DIR / "SimulatorUpdateButtonHook.cs"
PATCHER_SOURCE = UPDATE_BUTTON_DIR / "PatchAssembly.cs"
APPLY_ICON_SCRIPT = TOOL_DIR / "apply-dark-icon.sh"
DEFAULT_MONO_ROOT = Path(
    "/Applications/Unity/Hub/Editor/2020.3.49f1/Unity.app/Contents/MonoBleedingEdge"
)
COMPILE_REFERENCES = [
    "mscorlib.dll",
    "System.dll",
    "netstandard.dll",
    "UnityEngine.CoreModule.dll",
    "UnityEngine.IMGUIModule.dll",
    "UnityEngine.TextRenderingModule.dll",
    "UnityEngine.UI.dll",
    "UnityEngine.UIModule.dll",
    "UnityEngine.InputLegacyModule.dll",
]


def run(command, **kwargs):
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def compile_hook_dll(mono_root, managed_dir, source, output_dll):
    mcs = mono_root / "bin" / "mcs"
    if not mcs.is_file():
        raise RuntimeError(f"mcs not found under {mono_root} (pass --mono-root)")
    missing = [name for name in COMPILE_REFERENCES if not (managed_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"missing managed references: {missing}")
    run(
        [
            mcs,
            "-target:library",
            "-nostdlib",
            "-noconfig",
            "-sdk:4.5",
            f"-out:{output_dll}",
            *[f"-r:{managed_dir / name}" for name in COMPILE_REFERENCES],
            source,
        ]
    )


def compile_patcher(mono_root, work):
    mcs = mono_root / "bin" / "mcs"
    # The Unity-bundled lib/mono/unity/Mono.Cecil.dll is an ancient stripped
    # build without ReaderParameters; the GAC copies are complete. Pick the
    # highest GAC version available (0.10 verified; 0.11 preferred).
    gac_root = mono_root / "lib/mono/gac/Mono.Cecil"
    cecil_candidates = []
    if gac_root.is_dir():
        def version_key(path):
            try:
                return tuple(int(p) for p in path.parent.name.split("__")[0].split("."))
            except ValueError:
                return (0,)
        cecil_candidates = sorted(gac_root.glob("*/Mono.Cecil.dll"), key=version_key)
    if not cecil_candidates:
        raise RuntimeError(f"no GAC Mono.Cecil under {gac_root}")
    cecil = cecil_candidates[-1]
    patcher_exe = work / "PatchAssembly.exe"
    run(
        [
            mcs,
            "-target:exe",
            "-nostdlib",
            "-noconfig",
            "-sdk:4.5",
            f"-out:{patcher_exe}",
            f"-r:{mono_root / 'lib/mono/4.5/mscorlib.dll'}",
            f"-r:{cecil}",
            f"-r:{mono_root / 'lib/mono/4.5/System.Core.dll'}",
            PATCHER_SOURCE,
        ]
    )
    return mono_root / "bin" / "mono", patcher_exe


def extract_preserving_metadata(archive, work):
    """zipfile.extractall drops the exec bit; restore modes from the zip."""
    archive.extractall(work)
    for info in archive.infolist():
        mode = (info.external_attr >> 16) & 0o777
        if mode and not info.is_dir():
            target = work / info.filename
            if target.is_file():
                target.chmod(mode)


def patch_app_directory(app, work, mono_root):
    resources = app / "Contents" / "Resources"
    data = resources / "Data"
    managed = data / "Managed"
    if not managed.is_dir():
        raise RuntimeError(f"Managed directory not found: {managed}")

    manifest_path = data / "ScriptingAssemblies.json"
    manifest = json.loads(manifest_path.read_text())
    if HOOK_DLL_NAME in manifest["names"]:
        raise RuntimeError("source zip already carries the app icon hook")
    if len(manifest["names"]) != len(manifest["types"]):
        raise RuntimeError("unexpected ScriptingAssemblies.json shape")

    icon_hook_dll = work / HOOK_DLL_NAME
    button_hook_dll = work / "SimulatorUpdateButtonHook.dll"
    compile_hook_dll(mono_root, managed, ICON_HOOK_SOURCE, icon_hook_dll)
    if "SimulatorUpdateButtonHook.dll" in manifest["names"]:
        # The published button DLL is already inside; recompile a copy purely
        # so the patcher can resolve the (already-injected) button Install.
        compile_hook_dll(mono_root, managed, BUTTON_HOOK_SOURCE, button_hook_dll)
    mono, patcher_exe = compile_patcher(mono_root, work)

    patched_assembly = work / "Assembly-CSharp.patched.dll"
    run(
        [
            mono,
            patcher_exe,
            managed / "Assembly-CSharp.dll",
            patched_assembly,
            f"{icon_hook_dll}=DonkeySimUpdate.SimulatorAppIconHook::CarRecovery.Update",
            f"{button_hook_dll}=DonkeySimUpdate.SimulatorUpdateButtonHook::CarRecovery.Update",
        ]
    )
    patched_bytes = patched_assembly.read_bytes()
    if b"SimulatorAppIconHook" not in patched_bytes:
        raise RuntimeError("patched Assembly-CSharp.dll does not reference the icon hook")
    (managed / "Assembly-CSharp.dll").write_bytes(patched_bytes)

    shutil.copyfile(icon_hook_dll, managed / HOOK_DLL_NAME)
    manifest["names"].append(HOOK_DLL_NAME)
    manifest["types"].append(16)
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))

    for name in ("icon-day.png", "icon-night.png"):
        source = ICON_HOOK_SOURCE.parent.parent / "simulator-icon" / "AppIcon.icon" / "Assets" / name
        if not source.is_file():
            raise RuntimeError(f"icon artwork missing: {source}")
        shutil.copyfile(source, resources / name)

    run(["zsh", APPLY_ICON_SCRIPT, app])

    run(["codesign", "--force", "--deep", "--sign", "-", app])
    run(["codesign", "--verify", "--deep", "--strict", app])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mono-root", type=Path, default=DEFAULT_MONO_ROOT)
    args = parser.parse_args()

    source = args.source.absolute()
    output = args.output.absolute()
    if not source.is_file():
        raise RuntimeError(f"source zip not found: {source}")
    for required in (ICON_HOOK_SOURCE, BUTTON_HOOK_SOURCE, PATCHER_SOURCE):
        if not required.is_file():
            raise RuntimeError(f"source not found: {required}")

    with tempfile.TemporaryDirectory(prefix="donkey-icon-patch-") as temporary:
        work = Path(temporary)
        with zipfile.ZipFile(source) as archive:
            roots = {name.split("/")[0] for name in archive.namelist()}
            if roots != {"donkey_sim.app"}:
                raise RuntimeError(f"unexpected zip roots: {sorted(roots)}")
            extract_preserving_metadata(archive, work)
        app = work / "donkey_sim.app"
        patch_app_directory(app, work, args.mono_root)

        output.parent.mkdir(parents=True, exist_ok=True)
        # ditto preserves symlinks/permissions the way macOS player bundles need.
        run(["ditto", "-c", "-k", "--keepParent", "--sequesterRsrc", app, output])

    with zipfile.ZipFile(output) as verify:
        names = set(verify.namelist())
    for expected in (
        "donkey_sim.app/Contents/Resources/Data/Managed/SimulatorAppIconHook.dll",
        "donkey_sim.app/Contents/Resources/icon-day.png",
        "donkey_sim.app/Contents/Resources/icon-night.png",
        "donkey_sim.app/Contents/Resources/Assets.car",
        "donkey_sim.app/Contents/_CodeSignature/CodeResources",
    ):
        if expected not in names:
            raise RuntimeError(f"patched zip missing {expected}")
    print(f"patched package written: {output}")


if __name__ == "__main__":
    main()
