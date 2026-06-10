from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from navier_stokes_objective import NavierStokesReducedObjective
from navier_stokes_objective.mesh import read_channel_mesh


from navier_stokes_paths import MESH, XML


def test_mesh_parser_counts_and_sidesets() -> None:
    mesh = read_channel_mesh(MESH)
    assert mesh.nodes.shape == (2762, 2)
    assert mesh.cells.shape == (2672, 4)
    assert len(mesh.side_sets) == 5
    assert sum(len(side) for side in mesh.side_sets[4]) == 74
    assert mesh.stiffness.shape == (2762, 2762)


def test_fe_initial_condition_loads_rol_matrix_market_vector(tmp_path: Path) -> None:
    probe = NavierStokesReducedObjective.from_xml(XML, backend="fe", time_steps=2, end_time=0.01)
    state_size = probe.state_size
    rol_values = np.arange(state_size, dtype=float) + 0.125
    initial_condition = tmp_path / "initial_condition_Re200.txt"
    map_file = tmp_path / "map_initial_condition_Re200.txt"
    _write_matrix_market_vector(initial_condition, rol_values)
    map_values = np.column_stack([np.arange(state_size, dtype=np.int64), np.zeros(state_size, dtype=np.int64)]).ravel()
    _write_matrix_market_vector(map_file, map_values, integer=True)

    obj = NavierStokesReducedObjective.from_xml(
        XML,
        backend="fe",
        time_steps=2,
        end_time=0.01,
        initial_condition_path=initial_condition,
    )

    expected = np.empty(state_size)
    n = obj.mesh.num_nodes
    e = len(obj.fe_space.edge_to_id)
    v = obj.fe_constraint.num_velocity_dofs
    node_end = 3 * n
    edge_end = node_end + 2 * e
    expected[:n] = rol_values[:node_end:3]
    expected[v : v + n] = rol_values[1:node_end:3]
    expected[2 * v : 2 * v + n] = rol_values[2:node_end:3]
    expected[n : n + e] = rol_values[node_end:edge_end:2]
    expected[v + n : v + n + e] = rol_values[node_end + 1 : edge_end:2]
    expected[n + e : v] = rol_values[edge_end::2]
    expected[v + n + e : 2 * v] = rol_values[edge_end + 1 :: 2]
    assert np.array_equal(obj._initial_state, expected)


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


def _write_matrix_market_vector(path: Path, values: np.ndarray, *, integer: bool = False) -> None:
    data_type = "integer" if integer else "real"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"%%MatrixMarket matrix array {data_type} general\n")
        handle.write(f"{values.size} 1\n")
        for value in values:
            handle.write(f"{int(value)}\n" if integer else f"{float(value):.17e}\n")
