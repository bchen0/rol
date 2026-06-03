from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from navier_stokes_objective.fe import TaylorHoodSpace, q1_basis, q2_basis, tensor_product_gauss
from navier_stokes_objective.fem_pde import NavierStokesLocalResidual
from navier_stokes_objective.mesh import read_channel_mesh


MESH = ROOT / "rol/example/PDE-OPT/dynamic/navier-stokes/channel.txt"


def test_q2_q1_basis_partition_of_unity() -> None:
    points, weights = tensor_product_gauss(4)
    q2_values, q2_grads = q2_basis(points)
    q1_values, q1_grads = q1_basis(points)

    assert points.shape == (9, 2)
    assert np.isclose(np.sum(weights), 4.0)
    assert np.allclose(np.sum(q2_values, axis=0), 1.0)
    assert np.allclose(np.sum(q1_values, axis=0), 1.0)
    assert np.allclose(np.sum(q2_grads, axis=0), 0.0)
    assert np.allclose(np.sum(q1_grads, axis=0), 0.0)


def test_taylor_hood_space_maps_and_positive_jacobians() -> None:
    mesh = read_channel_mesh(MESH)
    space = TaylorHoodSpace.from_mesh(mesh, cell_ids=np.arange(8))

    assert space.velocity_dofs.shape == (8, 9)
    assert space.pressure_dofs.shape == (8, 4)
    assert space.num_pressure_dofs == mesh.num_nodes
    assert space.num_velocity_dofs > mesh.num_nodes + mesh.num_cells
    assert np.all(space.det_jacobian > 0.0)
    assert np.allclose(np.sum(space.q2_values, axis=0), 1.0)


def test_local_navier_stokes_jacobian_matches_finite_difference() -> None:
    mesh = read_channel_mesh(MESH)
    space = TaylorHoodSpace.from_mesh(mesh, cell_ids=np.arange(3))
    pde = NavierStokesLocalResidual(space, reynolds_number=200.0)
    rng = np.random.default_rng(11)
    state = rng.normal(scale=0.2, size=(space.num_cells, space.local_state_size))
    direction = rng.normal(scale=0.2, size=state.shape)

    analytic = pde.apply_jacobian(state, direction)
    eps = 1e-6
    fd = (pde.residual(state + eps * direction) - pde.residual(state - eps * direction)) / (2.0 * eps)
    assert np.allclose(analytic, fd, rtol=2e-6, atol=2e-8)
