from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from navier_stokes_objective.fe import TaylorHoodSpace
from navier_stokes_objective.fem_assembler import TaylorHoodAssembler
from navier_stokes_objective.mesh import read_channel_mesh


from navier_stokes_paths import MESH


def test_global_sparse_assembler_jacobian_matches_finite_difference() -> None:
    mesh = read_channel_mesh(MESH)
    space = TaylorHoodSpace.from_mesh(mesh, cell_ids=np.arange(4))
    assembler = TaylorHoodAssembler.from_space(space, reynolds_number=200.0)
    rng = np.random.default_rng(21)
    state = rng.normal(scale=0.05, size=assembler.state_size)
    direction = rng.normal(scale=0.05, size=assembler.state_size)

    matvec = assembler.assemble_jacobian(state) @ direction
    direct = assembler.apply_jacobian(state, direction)
    assert np.allclose(matvec, direct)

    eps = 1e-6
    fd = (assembler.assemble_residual(state + eps * direction) - assembler.assemble_residual(state - eps * direction)) / (
        2.0 * eps
    )
    assert np.allclose(matvec, fd, rtol=2e-6, atol=2e-8)
