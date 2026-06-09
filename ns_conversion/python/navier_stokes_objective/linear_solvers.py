from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
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
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        payload = _load_last_json_object(completed.stdout)
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


def _default_klu2_executable() -> Path:
    override = os.environ.get("NAVIER_STOKES_KLU2_SOLVER")
    if override:
        return Path(override).expanduser()
    root = Path(__file__).resolve().parents[2]
    return root / "Trilinos/build/packages/rol/example/PDE-OPT/dynamic/navier-stokes/klu2_sparse_solve.exe"
