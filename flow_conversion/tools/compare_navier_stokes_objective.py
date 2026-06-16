from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from navier_stokes_objective import NavierStokesReducedObjective
from navier_stokes_objective.config import read_config


DEFAULT_EXAMPLE_DIR = ROOT / "rol/example/PDE-OPT/dynamic/navier-stokes"
if not DEFAULT_EXAMPLE_DIR.exists():
    DEFAULT_EXAMPLE_DIR = ROOT / "Trilinos/packages/rol/example/PDE-OPT/dynamic/navier-stokes"
DEFAULT_XML = DEFAULT_EXAMPLE_DIR / "input.xml"


def compute_python_result(
    *,
    xml_path: str | Path = DEFAULT_XML,
    mesh_path: str | Path | None = None,
    backend: str = "fe",
    time_steps: int | None = None,
    end_time: float | None = None,
    theta: float | None = None,
    fe_cell_ids: np.ndarray | None = None,
    control: np.ndarray | None = None,
    direction: np.ndarray | None = None,
    linear_solver: str = "scipy",
    klu2_executable: str | Path | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"backend": backend, "linear_solver": linear_solver}
    if klu2_executable is not None:
        kwargs["klu2_executable"] = klu2_executable
    if time_steps is not None:
        kwargs["time_steps"] = int(time_steps)
    if end_time is not None:
        kwargs["end_time"] = float(end_time)
    if theta is not None:
        kwargs["theta"] = float(theta)
    if fe_cell_ids is not None:
        kwargs["fe_cell_ids"] = np.asarray(fe_cell_ids, dtype=np.int64)
    obj = NavierStokesReducedObjective.from_xml(xml_path, mesh_path=mesh_path, **kwargs)
    z = default_control(obj.num_controls) if control is None else _as_vector(control, obj.num_controls, "control")
    v = default_direction(obj.num_controls) if direction is None else _as_vector(direction, obj.num_controls, "direction")
    value = obj.value(z)
    gradient = obj.gradient(z)
    hess_vec = obj.hess_vec(z, v)
    return {
        "metadata": {
            "backend": backend,
            "xml_path": str(Path(xml_path).expanduser()),
            "mesh_path": None if mesh_path is None else str(Path(mesh_path).expanduser()),
            "time_steps": obj.config.nt,
            "end_time": obj.config.end_time,
            "theta": obj.config.theta,
            "fe_cell_ids": None if fe_cell_ids is None else np.asarray(fe_cell_ids, dtype=np.int64).tolist(),
            "linear_solver": linear_solver,
            "klu2_executable": None if klu2_executable is None else str(Path(klu2_executable).expanduser()),
        },
        "control": z.tolist(),
        "direction": v.tolist(),
        "value": float(value),
        "gradient": gradient.tolist(),
        "hess_vec": hess_vec.tolist(),
    }


def compare_results(
    python_result: dict[str, Any],
    reference_result: dict[str, Any],
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[bool, dict[str, Any]]:
    value_py = float(python_result["value"])
    value_ref = float(reference_result["value"])
    gradient_py = np.asarray(python_result["gradient"], dtype=float)
    gradient_ref = np.asarray(reference_result["gradient"], dtype=float)
    hess_py = np.asarray(python_result["hess_vec"], dtype=float)
    hess_ref = np.asarray(reference_result["hess_vec"], dtype=float)

    errors = {
        "value_abs": abs(value_py - value_ref),
        "value_rel": _relative_error(value_py, value_ref),
        "gradient_linf_abs": _linf_error(gradient_py, gradient_ref),
        "gradient_linf_rel": _relative_linf_error(gradient_py, gradient_ref),
        "hess_vec_linf_abs": _linf_error(hess_py, hess_ref),
        "hess_vec_linf_rel": _relative_linf_error(hess_py, hess_ref),
    }
    ok = (
        np.isclose(value_py, value_ref, rtol=rtol, atol=atol)
        and np.allclose(gradient_py, gradient_ref, rtol=rtol, atol=atol)
        and np.allclose(hess_py, hess_ref, rtol=rtol, atol=atol)
    )
    return bool(ok), errors


def default_control(num_controls: int) -> np.ndarray:
    z = np.zeros(num_controls)
    if num_controls > 1:
        z[1:] = np.linspace(0.015, -0.025, num_controls - 1)
    return z


def default_direction(num_controls: int) -> np.ndarray:
    v = np.zeros(num_controls)
    if num_controls > 1:
        idx = np.arange(1, num_controls, dtype=float)
        v[1:] = 0.1 * np.cos(idx)
    return v


def load_reference(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_reference_command(command: list[str], cwd: str | Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=None if cwd is None else Path(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for line in reversed(completed.stdout.splitlines()):
        text = line.strip()
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Reference command must print one JSON object with value, gradient, and hess_vec fields"
        ) from exc


def _parse_vector(spec: str | None, length: int, name: str) -> np.ndarray | None:
    if spec is None:
        return None
    text = Path(spec[1:]).expanduser().read_text(encoding="utf-8") if spec.startswith("@") else spec
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        values = [float(item) for item in text.replace("\n", ",").split(",") if item.strip()]
    return _as_vector(np.asarray(values, dtype=float), length, name)


def _parse_cell_ids(spec: str | None) -> np.ndarray | None:
    if spec is None:
        return None
    text = Path(spec[1:]).expanduser().read_text(encoding="utf-8") if spec.startswith("@") else spec
    if not text.strip():
        return None
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        values = [int(item) for item in text.replace("\n", ",").split(",") if item.strip()]
    return np.asarray(values, dtype=np.int64)


def _as_vector(values: np.ndarray, length: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}, got {arr.shape}")
    return arr.copy()


def _linf_error(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _relative_error(a: float, b: float) -> float:
    return float(abs(a - b) / max(1.0, abs(b)))


def _relative_linf_error(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    denom = max(1.0, float(np.max(np.abs(b))) if b.size else 0.0)
    return float(_linf_error(a, b) / denom)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the Python Navier-Stokes objective against a ROL reference JSON result.",
    )
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="Path to the ROL input.xml file.")
    parser.add_argument("--mesh", default=None, help="Optional mesh path override.")
    parser.add_argument("--backend", choices=("graph", "fe"), default="fe", help="Python backend to evaluate.")
    parser.add_argument("--linear-solver", choices=("scipy", "klu2"), default="scipy", help="FE sparse solver backend.")
    parser.add_argument("--klu2-executable", default=None, help="Path to tools/klu2_sparse_solve.cpp helper executable.")
    parser.add_argument("--time-steps", type=int, default=None, help="Override XML time-step count.")
    parser.add_argument("--end-time", type=float, default=None, help="Override XML end time.")
    parser.add_argument("--theta", type=float, default=None, help="Override XML theta.")
    parser.add_argument("--fe-cell-ids", default=None, help="Comma-separated FE cell ids, JSON list, or @file.")
    parser.add_argument("--control", default=None, help="Comma-separated control vector, JSON list, or @file.")
    parser.add_argument("--direction", default=None, help="Comma-separated HVP direction, JSON list, or @file.")
    parser.add_argument("--reference-json", default=None, help="JSON file from a C++/ROL reference probe.")
    parser.add_argument(
        "--reference-command",
        nargs=argparse.REMAINDER,
        help="Command that prints a reference JSON object. Must be the final CLI option.",
    )
    parser.add_argument(
        "--reference-cwd",
        default=None,
        help="Working directory for --reference-command. Defaults to the repository root.",
    )
    parser.add_argument("--rtol", type=float, default=1e-8, help="Relative comparison tolerance.")
    parser.add_argument("--atol", type=float, default=1e-10, help="Absolute comparison tolerance.")
    parser.add_argument("--output-json", default=None, help="Optional path for the Python result JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    fe_cell_ids = _parse_cell_ids(args.fe_cell_ids)
    config = read_config(args.xml, time_steps=args.time_steps, end_time=args.end_time, theta=args.theta)
    control = _parse_vector(args.control, config.nt, "control")
    direction = _parse_vector(args.direction, config.nt, "direction")
    python_result = compute_python_result(
        xml_path=args.xml,
        mesh_path=args.mesh,
        backend=args.backend,
        time_steps=args.time_steps,
        end_time=args.end_time,
        theta=args.theta,
        fe_cell_ids=fe_cell_ids,
        control=control,
        direction=direction,
        linear_solver=args.linear_solver,
        klu2_executable=args.klu2_executable,
    )
    if args.output_json is not None:
        Path(args.output_json).expanduser().write_text(json.dumps(python_result, indent=2) + "\n", encoding="utf-8")

    reference: dict[str, Any] | None = None
    if args.reference_json is not None:
        reference = load_reference(args.reference_json)
    if args.reference_command:
        reference = run_reference_command(args.reference_command, cwd=args.reference_cwd or ROOT)

    if reference is None:
        print(json.dumps(python_result, indent=2))
        print(
            "\nNo reference was provided. Pass --reference-json or --reference-command to compare against C++/ROL.",
            file=sys.stderr,
        )
        return 0

    ok, errors = compare_results(python_result, reference, rtol=args.rtol, atol=args.atol)
    print(
        json.dumps(
            {
                "passed": ok,
                "rtol": args.rtol,
                "atol": args.atol,
                "errors": errors,
            },
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
