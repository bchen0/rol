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


def test_dynamic_fe_constraint_jacobians_match_finite_difference() -> None:
    config = read_config(XML, time_steps=5, end_time=0.1)
    mesh = read_channel_mesh(MESH)
    cylinder_cells = np.asarray(sorted({c for side in mesh.side_sets[4] for c in side})[:5])
    space = TaylorHoodSpace.from_mesh(mesh, cell_ids=cylinder_cells)
    assembler = TaylorHoodAssembler.from_space(space, reynolds_number=config.reynolds_number)
    constraint = DynamicNavierStokesFEConstraint.from_config(assembler, config)
    rng = np.random.default_rng(31)
    u_old = rng.normal(scale=0.02, size=constraint.state_size)
    u_new = rng.normal(scale=0.02, size=constraint.state_size)
    v_old = rng.normal(scale=0.02, size=constraint.state_size)
    v_new = rng.normal(scale=0.02, size=constraint.state_size)
    z = 0.3
    dz = -0.7
    eps = 1e-6

    j_old = constraint.jacobian_uold(u_old, u_new, z) @ v_old
    fd_old = (constraint.residual(u_old + eps * v_old, u_new, z) - constraint.residual(u_old - eps * v_old, u_new, z)) / (
        2.0 * eps
    )
    assert np.allclose(j_old, fd_old, rtol=3e-6, atol=3e-8)

    j_new = constraint.jacobian_unew(u_old, u_new, z) @ v_new
    fd_new = (constraint.residual(u_old, u_new + eps * v_new, z) - constraint.residual(u_old, u_new - eps * v_new, z)) / (
        2.0 * eps
    )
    assert np.allclose(j_new, fd_new, rtol=3e-6, atol=3e-8)

    j_z = constraint.jacobian_z(u_old, u_new, z) * dz
    fd_z = (constraint.residual(u_old, u_new, z + eps * dz) - constraint.residual(u_old, u_new, z - eps * dz)) / (
        2.0 * eps
    )
    assert np.allclose(j_z, fd_z, rtol=3e-6, atol=3e-8)
