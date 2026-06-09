from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROL_EXAMPLE = ROOT / "rol/example/PDE-OPT/dynamic/navier-stokes"
if not ROL_EXAMPLE.exists():
    ROL_EXAMPLE = ROOT / "Trilinos/packages/rol/example/PDE-OPT/dynamic/navier-stokes"

XML = ROL_EXAMPLE / "input.xml"
MESH = ROL_EXAMPLE / "channel.txt"
