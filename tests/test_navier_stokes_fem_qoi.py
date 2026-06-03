from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from navier_stokes_objective.config import read_config
from navier_stokes_objective.fe import TaylorHoodSpace
from navier_stokes_objective.fem_assembler import TaylorHoodAssembler
from navier_stokes_objective.fem_qoi import NavierStokesFEObjective
from navier_stokes_objective.mesh import read_channel_mesh


XML = ROOT / "rol/example/PDE-OPT/dynamic/navier-stokes/input.xml"
MESH = ROOT / "rol/example/PDE-OPT/dynamic/navier-stokes/channel.txt"


def test_fe_state_qoi_gradients_match_finite_difference() -> None:
    config = read_config(XML, time_steps=5, end_time=0.1)
    mesh = read_channel_mesh(MESH)
    cells = np.asarray(sorted({c for side in mesh.side_sets[1] for c in side})[:5])
    space = TaylorHoodSpace.from_mesh(mesh, cell_ids=cells)
    assembler = TaylorHoodAssembler.from_space(space, reynolds_number=config.reynolds_number)
    qoi = NavierStokesFEObjective(assembler, config)
    rng = np.random.default_rng(41)
    state = rng.normal(scale=0.05, size=assembler.state_size)
    direction = rng.normal(scale=0.05, size=assembler.state_size)

    eps = 1e-6
    for objective_type in ("Dissipation", "Tracking"):
        grad = qoi.state_gradient(state, objective_type)
        fd = (qoi.state_value(state + eps * direction, objective_type) - qoi.state_value(state - eps * direction, objective_type)) / (
            2.0 * eps
        )
        assert np.isclose(float(np.dot(grad, direction)), fd, rtol=2e-6, atol=2e-8)


def test_fe_downstream_and_integrated_gradients_match_finite_difference() -> None:
    config = read_config(XML, time_steps=5, end_time=0.1)
    mesh = read_channel_mesh(MESH)
    cells = np.asarray(sorted({c for side in mesh.side_sets[1] for c in side})[:5])
    space = TaylorHoodSpace.from_mesh(mesh, cell_ids=cells)
    assembler = TaylorHoodAssembler.from_space(space, reynolds_number=config.reynolds_number)
    qoi = NavierStokesFEObjective(assembler, config)
    rng = np.random.default_rng(43)
    state = rng.normal(scale=0.05, size=assembler.state_size)
    direction = rng.normal(scale=0.05, size=assembler.state_size)

    eps = 1e-6
    grad = qoi.downstream_power_gradient(state)
    fd = (qoi.downstream_power_value(state + eps * direction) - qoi.downstream_power_value(state - eps * direction)) / (
        2.0 * eps
    )
    assert np.isclose(float(np.dot(grad, direction)), fd, rtol=2e-6, atol=2e-8)

    z = 0.3
    dz = -0.4
    grad_state = qoi.integrated_state_gradient(state)
    grad_z = qoi.integrated_control_gradient(z)
    fd_int = (
        qoi.integrated_value(state + eps * direction, z + eps * dz)
        - qoi.integrated_value(state - eps * direction, z - eps * dz)
    ) / (2.0 * eps)
    assert np.isclose(float(np.dot(grad_state, direction) + grad_z * dz), fd_int, rtol=2e-6, atol=2e-8)
