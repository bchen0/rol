from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from navier_stokes_objective.linear_solvers import (
    KLU2SparseLinearSolver,
    ScipySparseLinearSolver,
    build_sparse_linear_solver,
)


KLU2_EXE = ROOT / "Trilinos/build/packages/rol/example/PDE-OPT/dynamic/navier-stokes/klu2_sparse_solve.exe"


def test_scipy_sparse_solver_handles_transpose() -> None:
    matrix = sparse.csr_matrix([[3.0, 1.0], [2.0, 4.0]])
    rhs = np.array([7.0, 10.0])
    solver = ScipySparseLinearSolver()

    assert np.allclose(solver.solve(matrix, rhs), np.linalg.solve(matrix.toarray(), rhs))
    assert np.allclose(solver.solve(matrix, rhs, transpose=True), np.linalg.solve(matrix.toarray().T, rhs))


def test_build_sparse_linear_solver_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown sparse linear solver"):
        build_sparse_linear_solver("not-a-solver")


@pytest.mark.skipif(not KLU2_EXE.exists(), reason="KLU2 sparse-solve helper is not built")
def test_klu2_sparse_solver_matches_scipy_on_nonsingular_system() -> None:
    matrix = sparse.csr_matrix([[3.0, 1.0], [2.0, 4.0]])
    rhs = np.array([7.0, 10.0])
    scipy_solver = ScipySparseLinearSolver()
    klu2_solver = KLU2SparseLinearSolver(KLU2_EXE)

    assert np.allclose(klu2_solver.solve(matrix, rhs), scipy_solver.solve(matrix, rhs), rtol=1e-14, atol=1e-14)
    assert np.allclose(
        klu2_solver.solve(matrix, rhs, transpose=True),
        scipy_solver.solve(matrix, rhs, transpose=True),
        rtol=1e-14,
        atol=1e-14,
    )
