from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = ROOT / "Trilinos/build/packages/rol/example/PDE-OPT/dynamic/navier-stokes"
DEFAULT_SOURCE = ROOT / "tools/rol_navier_stokes_probe.cpp"
DEFAULT_OUTPUT = DEFAULT_BUILD_DIR / "rol_navier_stokes_probe.exe"


def _read_make_var(path: Path, name: str) -> list[str]:
    prefix = f"{name} = "
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return shlex.split(line[len(prefix) :])
    return []


def _compile(build_dir: Path, source: Path, output: Path) -> Path:
    flags_make = (
        build_dir
        / "CMakeFiles/ROL_example_PDE-OPT_dynamic_navier-stokes_example_01.dir/flags.make"
    )
    obj = output.with_suffix(".o")
    command = [
        "/usr/bin/c++",
        *_read_make_var(flags_make, "CXX_DEFINES"),
        *_read_make_var(flags_make, "CXX_INCLUDES"),
        "-I" + str(ROOT / "Trilinos/packages/rol/example/PDE-OPT/dynamic/navier-stokes"),
        *_read_make_var(flags_make, "CXX_FLAGS"),
        "-c",
        str(source),
        "-o",
        str(obj),
    ]
    subprocess.run(command, cwd=build_dir, check=True)
    return obj


def _link(build_dir: Path, obj: Path, output: Path) -> None:
    link_file = (
        build_dir
        / "CMakeFiles/ROL_example_PDE-OPT_dynamic_navier-stokes_example_01.dir/link.txt"
    )
    command = shlex.split(link_file.read_text(encoding="utf-8").strip())
    original_obj = "CMakeFiles/ROL_example_PDE-OPT_dynamic_navier-stokes_example_01.dir/example_01.cpp.o"
    command = [str(obj) if item == original_obj else item for item in command]
    for i, item in enumerate(command[:-1]):
        if item == "-o":
            command[i + 1] = str(output)
            break
    subprocess.run(command, cwd=build_dir, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the local ROL Navier-Stokes oracle probe.")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    build_dir = args.build_dir.expanduser().resolve()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    obj = _compile(build_dir, source, output)
    _link(build_dir, obj, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
