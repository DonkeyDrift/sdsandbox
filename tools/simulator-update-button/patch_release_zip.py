#!/usr/bin/env python3
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

HOOK_NAME = "SimulatorUpdateButtonHook.dll"

# Release archives used to ship `donkey_sim*` names; repacked outputs use the
# `Donkey Sim` spelling.  Unity requires the executable and its `<name>_Data`
# folder to match, so they are always renamed as a pair.
ENTRY_RENAMES = (
    ("donkey_sim.exe", "Donkey Sim.exe"),
    ("donkey_sim.x86_64", "Donkey Sim.x86_64"),
    ("donkey_sim.app", "Donkey Sim.app"),
    ("/MacOS/donkey_sim", "/MacOS/Donkey Sim"),
    ("donkey_sim_Data", "Donkey Sim_Data"),
)


def renamed_entry(filename):
    for old, new in ENTRY_RENAMES:
        filename = filename.replace(old, new)
    return filename


def run(command, **kwargs):
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_tools(work, mono_root, managed_dir, hook_source, patcher_source):
    mcs = mono_root / "bin" / "mcs"
    mono = mono_root / "bin" / "mono"
    cecil = None
    gac_root = mono_root / "lib/mono/gac/Mono.Cecil"

    def cecil_version_key(path):
        try:
            return tuple(int(part) for part in path.parent.name.split("__")[0].split("."))
        except ValueError:
            return (0,)

    gac_cecil = sorted(gac_root.glob("*/*.dll"), key=cecil_version_key, reverse=True)
    if gac_cecil:
        cecil = gac_cecil[0]
    if cecil is None:
        cecil = mono_root / "lib/mono/unity/Mono.Cecil.dll"
    hook_dll = work / HOOK_NAME
    references = [
        "mscorlib.dll",
        "System.dll",
        "netstandard.dll",
        "UnityEngine.CoreModule.dll",
        "UnityEngine.IMGUIModule.dll",
        "UnityEngine.TextRenderingModule.dll",
    ]
    missing = [name for name in references if not (managed_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"missing managed references: {missing}")

    run(
        [
            mcs,
            "-target:library",
            "-nostdlib",
            "-noconfig",
            "-sdk:4.5",
            f"-out:{hook_dll}",
            *[f"-r:{managed_dir / name}" for name in references],
            hook_source,
        ]
    )

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
            f"-r:{managed_dir / 'netstandard.dll'}",
            f"-r:{mono_root / 'lib/mono/4.5/System.Core.dll'}",
            patcher_source,
        ]
    )
    # mono resolves dependencies next to the executable first; keep Cecil reachable
    # for mono roots whose global assembly cache has no 0.11.x build.
    shutil.copy(cecil, work / "Mono.Cecil.dll")
    return mono, hook_dll, patcher_exe


def copy_zip_entry(source, info, destination, replacement=None):
    data = source.read(info.filename) if replacement is None else replacement
    cloned = zipfile.ZipInfo(renamed_entry(info.filename), info.date_time)
    cloned.compress_type = zipfile.ZIP_DEFLATED
    cloned.create_system = info.create_system
    cloned.external_attr = info.external_attr
    cloned.internal_attr = info.internal_attr
    cloned.comment = info.comment
    destination.writestr(cloned, data)


def patch_zip(source_path, output_path, hook_source, patcher_source, mono_root):
    source_path = source_path.absolute()
    output_path = output_path.absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="donkey-release-patch-") as temporary:
        work = Path(temporary)
        managed_dir = work / "managed"
        managed_dir.mkdir()

        with zipfile.ZipFile(source_path) as source:
            assembly_entries = [
                info for info in source.infolist() if info.filename.endswith("/Managed/Assembly-CSharp.dll")
            ]
            if len(assembly_entries) != 1:
                raise RuntimeError("expected exactly one Assembly-CSharp.dll")
            data_prefix = assembly_entries[0].filename.rsplit("/Managed/", 1)[0] + "/"
            manifest_entries = [
                info for info in source.infolist() if info.filename == data_prefix + "ScriptingAssemblies.json"
            ]
            if len(manifest_entries) != 1:
                raise RuntimeError("expected exactly one ScriptingAssemblies.json")

            assembly_info = assembly_entries[0]
            manifest_info = manifest_entries[0]
            managed_prefix = assembly_info.filename.rsplit("/", 1)[0] + "/"
            renamed_entries = sorted(
                (info.filename, renamed_entry(info.filename))
                for info in source.infolist()
                if renamed_entry(info.filename) != info.filename
            )

            for info in source.infolist():
                if info.filename.startswith(managed_prefix) and not info.is_dir():
                    (managed_dir / info.filename.rsplit("/", 1)[-1]).write_bytes(source.read(info.filename))

            original_assembly = managed_dir / "Assembly-CSharp.dll"
            manifest = json.loads(source.read(manifest_info.filename))
            if HOOK_NAME in manifest["names"]:
                raise RuntimeError("source release already contains the update button hook")

            mono, hook_dll, patcher_exe = compile_tools(
                work,
                mono_root,
                managed_dir,
                hook_source.absolute(),
                patcher_source.absolute(),
            )
            patched_assembly = work / "Assembly-CSharp.patched.dll"
            run(
                [
                    mono,
                    patcher_exe,
                    original_assembly,
                    hook_dll,
                    patched_assembly,
                ]
            )

            patched_assembly_bytes = patched_assembly.read_bytes()
            if b"SimulatorUpdateButtonHook" not in patched_assembly_bytes:
                raise RuntimeError("patched Assembly-CSharp.dll does not reference the update hook")

            manifest["names"].append(HOOK_NAME)
            manifest["types"].append(16)
            manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
            hook_bytes = hook_dll.read_bytes()

            temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
            with zipfile.ZipFile(temporary_output, "w", zipfile.ZIP_DEFLATED) as destination:
                for info in source.infolist():
                    if info.filename == assembly_info.filename:
                        copy_zip_entry(source, info, destination, patched_assembly_bytes)
                    elif info.filename == manifest_info.filename:
                        copy_zip_entry(source, info, destination, manifest_bytes)
                    else:
                        copy_zip_entry(source, info, destination)

                hook_info = zipfile.ZipInfo(
                    renamed_entry(f"{data_prefix}Managed/{HOOK_NAME}"),
                    assembly_info.date_time,
                )
                hook_info.compress_type = zipfile.ZIP_DEFLATED
                hook_info.create_system = 3
                hook_info.external_attr = 0o100644 << 16
                destination.writestr(hook_info, hook_bytes)

            with zipfile.ZipFile(temporary_output) as verification:
                bad_file = verification.testzip()
                if bad_file is not None:
                    raise RuntimeError(f"output zip verification failed: {bad_file}")
                names = set(verification.namelist())
                expected = {
                    renamed_entry(assembly_info.filename),
                    renamed_entry(manifest_info.filename),
                    renamed_entry(f"{data_prefix}Managed/{HOOK_NAME}"),
                }
                if not expected.issubset(names):
                    raise RuntimeError("output zip is missing patched entries")

            temporary_output.replace(output_path)

    print(f"patched {output_path} ({output_path.stat().st_size} bytes)")
    if renamed_entries:
        print(f"renamed {len(renamed_entries)} entries to Donkey Sim naming, e.g.:")
        for old_name, new_name in renamed_entries[:5]:
            print(f"  {old_name} -> {new_name}")
    print(f"sha256 {sha256(output_path)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_zip", type=Path)
    parser.add_argument("output_zip", type=Path)
    parser.add_argument("--hook-source", type=Path, required=True)
    parser.add_argument("--patcher-source", type=Path, required=True)
    parser.add_argument(
        "--mono-root",
        type=Path,
        required=True,
        help="Unity MonoBleedingEdge directory containing bin/mcs, bin/mono and Mono.Cecil",
    )
    args = parser.parse_args()
    patch_zip(
        args.source_zip,
        args.output_zip,
        args.hook_source,
        args.patcher_source,
        args.mono_root.absolute(),
    )


if __name__ == "__main__":
    main()
