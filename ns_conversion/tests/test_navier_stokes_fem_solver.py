from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from navier_stokes_objective.config import read_config
from navier_stokes_objective.fe import TaylorHoodSpace
from navier_stokes_objective.fem_assembler import TaylorHoodAssembler
from navier_stokes_objective.fem_dynamic import DynamicNavierStokesFEConstraint
from navier_stokes_objective.mesh import read_channel_mesh


from navier_stokes_paths import MESH, XML


def test_dynamic_fe_newton_solver_reduces_residual() -> None:
    config = read_config(XML, time_steps=20, end_time=0.05)
    mesh = read_channel_mesh(MESH)
    cylinder_cells = np.asarray(sorted({c for side in mesh.side_sets[4] for c in side})[:8])
    space = TaylorHoodSpace.from_mesh(mesh, cell_ids=cylinder_cells)
    assembler = TaylorHoodAssembler.from_space(space, reynolds_number=config.reynolds_number)
    constraint = DynamicNavierStokesFEConstraint.from_config(assembler, config)

    u_old = constraint.potential_flow_state()
    z = 0.1
    initial = u_old.copy()
    constraint.apply_boundary_values(initial, z)
    before = constraint.residual_norm(constraint.residual(u_old, initial, z))
    u_new = constraint.solve(u_old, z, initial_guess=initial, absolute_tol=1e-9, relative_tol=1e-9, max_iter=12)
    residual = constraint.residual(u_old, u_new, z)
    after = constraint.residual_norm(residual)

    assert after < 1e-9
    assert after < 1e-4 * before
