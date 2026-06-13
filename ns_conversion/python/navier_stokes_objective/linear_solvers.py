from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
from typing import Protocol

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


class SparseLinearSolver(Protocol):
    def solve(
        self,
        matrix: sparse.spmatrix,
        rhs: np.ndarray,
        *,
        transpose: bool = False,
        global_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve a sparse linear system."""


@dataclass(frozen=True)
class ScipySparseLinearSolver:
    """Sparse direct solver backed by SciPy/SuperLU."""

    def solve(
        self,
        matrix: sparse.spmatrix,
        rhs: np.ndarray,
        *,
        transpose: bool = False,
        global_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        del global_ids
        mat = matrix.T if transpose else matrix
        return np.asarray(spla.spsolve(mat.tocsc(), np.asarray(rhs, dtype=float)), dtype=float)


@dataclass(frozen=True)
class KLU2SparseLinearSolver:
    """Sparse direct solver backed by Trilinos Amesos2/KLU2.

    The project does not have PyTrilinos bindings in the conda environment, so
    this adapter delegates one linear solve at a time to a small local executable
    linked against the same Trilinos build as the ROL oracle.
    """

    executable: Path | str | None = None

    def __post_init__(self) -> None:
        path = _default_klu2_executable() if self.executable is None else Path(self.executable).expanduser()
        object.__setattr__(self, "executable", path)

    def solve(
        self,
        matrix: sparse.spmatrix,
        rhs: np.ndarray,
        *,
        transpose: bool = False,
        global_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        exe = Path(self.executable)
        if not exe.exists():
            raise FileNotFoundError(
                f"KLU2 sparse-solve helper was not found at {exe}. "
                "Build it with tools/build_rol_navier_stokes_probe.py using tools/klu2_sparse_solve.cpp."
            )

        rhs_arr = np.asarray(rhs, dtype=float)
        if rhs_arr.ndim != 1:
            raise ValueError(f"rhs must be one-dimensional, got shape {rhs_arr.shape}")
        mat = matrix.tocsr()
        mat.sort_indices()
        mat.sum_duplicates()
        if not np.all(np.isfinite(mat.data)):
            raise ValueError("matrix contains nonfinite entries")
        if not np.all(np.isfinite(rhs_arr)):
            raise ValueError("rhs contains nonfinite entries")
        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"matrix must be square, got shape {mat.shape}")
        if mat.shape[0] != rhs_arr.size:
            raise ValueError(f"rhs length {rhs_arr.size} does not match matrix shape {mat.shape}")
        gids = None if global_ids is None else np.asarray(global_ids, dtype=np.int64)
        if gids is not None and gids.shape != rhs_arr.shape:
            raise ValueError(f"global_ids shape {gids.shape} does not match rhs shape {rhs_arr.shape}")

        with tempfile.TemporaryDirectory(prefix="klu2_sparse_solve_") as tmp:
            input_path = Path(tmp) / "problem.txt"
            _write_sparse_problem(input_path, mat, rhs_arr, global_ids=gids)
            command = [str(exe), str(input_path)]
            if transpose:
                command.append("--transpose")
            completed = subprocess.run(
                command,
                cwd=exe.parent,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if completed.returncode != 0:
                problem_copy = _preserve_failed_problem(input_path)
                raise RuntimeError(
                    _format_failed_solve_message(
                        command=command,
                        returncode=completed.returncode,
                        stdout=completed.stdout,
                        stderr=completed.stderr,
                        problem_copy=problem_copy,
                    )
                )

        try:
            payload = _load_last_json_object(completed.stdout)
        except RuntimeError as err:
            raise RuntimeError(
                "KLU2 helper finished successfully but did not return a JSON solution.\n"
                f"stdout:\n{_trim_output(completed.stdout)}\n"
                f"stderr:\n{_trim_output(completed.stderr)}"
            ) from err
        solution = np.asarray(payload["solution"], dtype=float)
        if solution.shape != rhs_arr.shape:
            raise RuntimeError(f"KLU2 helper returned solution shape {solution.shape}; expected {rhs_arr.shape}")
        return solution


def build_sparse_linear_solver(
    solver: str | SparseLinearSolver,
    *,
    klu2_executable: Path | str | None = None,
) -> SparseLinearSolver:
    if isinstance(solver, str):
        key = solver.lower()
        if key == "scipy":
            return ScipySparseLinearSolver()
        if key == "klu2":
            return KLU2SparseLinearSolver(klu2_executable)
        raise ValueError(f"Unknown sparse linear solver {solver!r}; expected 'scipy' or 'klu2'")
    return solver


def _write_sparse_problem(
    path: Path,
    matrix: sparse.csr_matrix,
    rhs: np.ndarray,
    *,
    global_ids: np.ndarray | None = None,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        has_global_ids = global_ids is not None
        handle.write(f"{matrix.shape[0]} {matrix.nnz} {int(has_global_ids)}\n")
        if has_global_ids:
            handle.write(" ".join(str(int(gid)) for gid in global_ids))
            handle.write("\n")
        for row in range(matrix.shape[0]):
            start = matrix.indptr[row]
            stop = matrix.indptr[row + 1]
            cols = matrix.indices[start:stop]
            vals = matrix.data[start:stop]
            for col, val in zip(cols, vals):
                handle.write(f"{row} {int(col)} {float(val):.17e}\n")
        for value in rhs:
            handle.write(f"{float(value):.17e}\n")


def _load_last_json_object(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
    raise RuntimeError("KLU2 helper did not print a JSON solution object")


def _format_failed_solve_message(
    *,
    command: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
    problem_copy: Path | None,
) -> str:
    if returncode < 0:
        signum = -returncode
        try:
            status = f"signal {signal.Signals(signum).name} ({signum})"
        except ValueError:
            status = f"signal {signum}"
    else:
        status = f"exit code {returncode}"
    message = [
        f"KLU2 sparse-solve helper failed with {status}.",
        f"command: {' '.join(command)}",
    ]
    if problem_copy is not None:
        message.append(f"sparse problem copy: {problem_copy}")
    else:
        message.append(
            "Set NAVIER_STOKES_KLU2_DEBUG_DIR to preserve the sparse problem file for replay."
        )
    message.append(f"stdout:\n{_trim_output(stdout)}")
    message.append(f"stderr:\n{_trim_output(stderr)}")
    return "\n".join(message)


def _preserve_failed_problem(input_path: Path) -> Path | None:
    debug_dir = os.environ.get("NAVIER_STOKES_KLU2_DEBUG_DIR")
    if not debug_dir:
        return None
    target_dir = Path(debug_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{input_path.parent.name}_{input_path.name}"
    shutil.copy2(input_path, target)
    return target


def _trim_output(text: str, *, limit: int = 4000) -> str:
    if not text:
        return "<empty>"
    if len(text) <= limit:
        return text.rstrip()
    return "...<truncated>...\n" + text[-limit:].rstrip()


def _default_klu2_executable() -> Path:
    override = os.environ.get("NAVIER_STOKES_KLU2_SOLVER")
    if override:
        return Path(override).expanduser()
    root = Path(__file__).resolve().parents[2]
    return root / "Trilinos/build/packages/rol/example/PDE-OPT/dynamic/navier-stokes/klu2_sparse_solve.exe"
