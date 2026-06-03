from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from navier_stokes_objective import NavierStokesReducedObjective
from navier_stokes_objective.mesh import read_channel_mesh


XML = ROOT / "rol/example/PDE-OPT/dynamic/navier-stokes/input.xml"
MESH = ROOT / "rol/example/PDE-OPT/dynamic/navier-stokes/channel.txt"


def test_fe_backend_reduced_value_gradient_and_hvp() -> None:
    mesh = read_channel_mesh(MESH)
    cylinder_cells = np.asarray(sorted({c for side in mesh.side_sets[4] for c in side})[:1])
    obj = NavierStokesReducedObjective.from_xml(
        XML,
        backend="fe",
        time_steps=2,
        end_time=0.01,
        fe_cell_ids=cylinder_cells,
        newton_absolute_tol=1e-9,
        newton_relative_tol=1e-9,
        newton_max_iter=12,
    )
    z = np.array([0.0, 0.04])
    direction = np.array([0.0, -0.2])

    value = obj.value(z)
    grad = obj.gradient(z)
    assert np.isfinite(value)
    assert grad.shape == z.shape
    assert grad[0] == 0.0

    eps = 1e-5
    fd = (obj.value(z + eps * direction) - obj.value(z - eps * direction)) / (2.0 * eps)
    assert np.isclose(fd, float(np.dot(grad, direction)), rtol=2e-3, atol=1e-6)

    hv = obj.hess_vec(z, np.zeros_like(direction))
    assert hv.shape == z.shape
    assert np.array_equal(hv, np.zeros_like(z))
