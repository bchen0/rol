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


def test_mesh_parser_counts_and_sidesets() -> None:
    mesh = read_channel_mesh(MESH)
    assert mesh.nodes.shape == (2762, 2)
    assert mesh.cells.shape == (2672, 4)
    assert len(mesh.side_sets) == 5
    assert sum(len(side) for side in mesh.side_sets[4]) == 74
    assert mesh.stiffness.shape == (2762, 2762)


def test_value_gradient_and_hvp_checks() -> None:
    obj = NavierStokesReducedObjective.from_xml(XML, time_steps=6, end_time=0.15)
    z = np.linspace(0.0, 0.2, obj.num_controls)
    v = np.array([0.0, 0.4, -0.1, 0.25, -0.35, 0.15])

    f0 = obj.value(z)
    g = obj.gradient(z)
    assert np.isfinite(f0)
    assert g.shape == z.shape
    assert g[0] == 0.0

    eps = 1e-6
    fd = (obj.value(z + eps * v) - obj.value(z - eps * v)) / (2.0 * eps)
    assert np.isclose(fd, float(np.dot(g, v)), rtol=1e-4, atol=1e-6)

    hv = obj.hess_vec(z, v)
    assert hv.shape == z.shape
    assert hv[0] == 0.0
    grad_fd = (obj.gradient(z + eps * v) - obj.gradient(z - eps * v)) / (2.0 * eps)
    assert np.allclose(hv, grad_fd, rtol=5e-3, atol=5e-5)
