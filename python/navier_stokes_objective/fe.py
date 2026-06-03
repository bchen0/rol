from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .mesh import ChannelMesh


_LOCAL_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
_Q2_NODES = (
    (-1.0, -1.0),
    (1.0, -1.0),
    (1.0, 1.0),
    (-1.0, 1.0),
    (0.0, -1.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (0.0, 0.0),
)


def tensor_product_gauss(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Return tensor-product Gauss points for a requested exactness degree."""
    n_1d = max(1, (degree + 2) // 2)
    x, w = np.polynomial.legendre.leggauss(n_1d)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    wx, wy = np.meshgrid(w, w, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    weights = (wx * wy).ravel()
    return pts, weights


def q1_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xi = points[:, 0]
    eta = points[:, 1]
    values = np.vstack(
        [
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
        ]
    )
    grads = np.empty((4, points.shape[0], 2))
    grads[0, :, 0] = -0.25 * (1.0 - eta)
    grads[0, :, 1] = -0.25 * (1.0 - xi)
    grads[1, :, 0] = 0.25 * (1.0 - eta)
    grads[1, :, 1] = -0.25 * (1.0 + xi)
    grads[2, :, 0] = 0.25 * (1.0 + eta)
    grads[2, :, 1] = 0.25 * (1.0 + xi)
    grads[3, :, 0] = -0.25 * (1.0 + eta)
    grads[3, :, 1] = 0.25 * (1.0 - xi)
    return values, grads


def q2_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xi = points[:, 0]
    eta = points[:, 1]
    lx = [
        0.5 * xi * (xi - 1.0),
        1.0 - xi * xi,
        0.5 * xi * (xi + 1.0),
    ]
    ly = [
        0.5 * eta * (eta - 1.0),
        1.0 - eta * eta,
        0.5 * eta * (eta + 1.0),
    ]
    dlx = [xi - 0.5, -2.0 * xi, xi + 0.5]
    dly = [eta - 0.5, -2.0 * eta, eta + 0.5]
    # Tensor-product index pairs matching ROL's topology order:
    # vertices, edge midpoints, then cell center.
    pairs = ((0, 0), (2, 0), (2, 2), (0, 2), (1, 0), (2, 1), (1, 2), (0, 1), (1, 1))
    values = np.empty((9, points.shape[0]))
    grads = np.empty((9, points.shape[0], 2))
    for i, (ix, iy) in enumerate(pairs):
        values[i] = lx[ix] * ly[iy]
        grads[i, :, 0] = dlx[ix] * ly[iy]
        grads[i, :, 1] = lx[ix] * dly[iy]
    return values, grads


@dataclass(frozen=True)
class TaylorHoodSpace:
    mesh: ChannelMesh
    cell_ids: np.ndarray
    velocity_dofs: np.ndarray
    pressure_dofs: np.ndarray
    num_velocity_dofs: int
    num_pressure_dofs: int
    edge_to_id: dict[tuple[int, int], int]
    cub_points: np.ndarray
    cub_weights: np.ndarray
    q2_values: np.ndarray
    q1_values: np.ndarray
    q2_grads: np.ndarray
    q1_grads: np.ndarray
    q2_weighted_values: np.ndarray
    q1_weighted_values: np.ndarray
    q2_weighted_grads: np.ndarray
    q1_weighted_grads: np.ndarray
    det_jacobian: np.ndarray
    physical_points: np.ndarray

    @classmethod
    def from_mesh(
        cls,
        mesh: ChannelMesh,
        *,
        cell_ids: np.ndarray | None = None,
        cubature_degree: int = 4,
    ) -> "TaylorHoodSpace":
        selected = np.arange(mesh.num_cells, dtype=np.int64) if cell_ids is None else np.asarray(cell_ids, dtype=np.int64)
        edge_to_id = _build_edge_ids(mesh)
        velocity_dofs = _velocity_dofs(mesh, selected, edge_to_id)
        pressure_dofs = mesh.cells[selected].copy()
        points, weights = tensor_product_gauss(cubature_degree)
        q2_val_ref, q2_grad_ref = q2_basis(points)
        q1_val_ref, q1_grad_ref = q1_basis(points)
        (
            q2_grads,
            q1_grads,
            det_j,
            physical_points,
        ) = _map_to_physical(mesh.nodes[mesh.cells[selected]], points, q2_grad_ref, q1_grad_ref, q1_val_ref)
        weighted = det_j * weights[None, :]
        return cls(
            mesh=mesh,
            cell_ids=selected,
            velocity_dofs=velocity_dofs,
            pressure_dofs=pressure_dofs,
            num_velocity_dofs=mesh.num_nodes + len(edge_to_id) + mesh.num_cells,
            num_pressure_dofs=mesh.num_nodes,
            edge_to_id=edge_to_id,
            cub_points=points,
            cub_weights=weights,
            q2_values=q2_val_ref,
            q1_values=q1_val_ref,
            q2_grads=q2_grads,
            q1_grads=q1_grads,
            q2_weighted_values=q2_val_ref[None, :, :] * weighted[:, None, :],
            q1_weighted_values=q1_val_ref[None, :, :] * weighted[:, None, :],
            q2_weighted_grads=q2_grads * weighted[:, None, :, None],
            q1_weighted_grads=q1_grads * weighted[:, None, :, None],
            det_jacobian=det_j,
            physical_points=physical_points,
        )

    @property
    def num_cells(self) -> int:
        return int(self.cell_ids.size)

    @property
    def local_state_size(self) -> int:
        return 22

    def split_local_state(self, local_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(local_state, dtype=float)
        if arr.shape != (self.num_cells, self.local_state_size):
            raise ValueError(f"Expected local state shape {(self.num_cells, self.local_state_size)}, got {arr.shape}")
        return arr[:, :9], arr[:, 9:18], arr[:, 18:]

    def combine_local_state(self, ux: np.ndarray, uy: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return np.concatenate([ux, uy, pressure], axis=1)

    def evaluate_value(self, coeff: np.ndarray, family: Literal["q1", "q2"]) -> np.ndarray:
        values = self.q2_values if family == "q2" else self.q1_values
        return np.einsum("cf,fp->cp", coeff, values)

    def evaluate_gradient(self, coeff: np.ndarray, family: Literal["q1", "q2"]) -> np.ndarray:
        grads = self.q2_grads if family == "q2" else self.q1_grads
        return np.einsum("cf,cfpd->cpd", coeff, grads)


def _build_edge_ids(mesh: ChannelMesh) -> dict[tuple[int, int], int]:
    edge_to_id: dict[tuple[int, int], int] = {}
    for cell in mesh.cells:
        for i, j in _LOCAL_EDGES:
            a = int(cell[i])
            b = int(cell[j])
            if a > b:
                a, b = b, a
            if (a, b) not in edge_to_id:
                edge_to_id[(a, b)] = len(edge_to_id)
    return edge_to_id


def _velocity_dofs(mesh: ChannelMesh, cell_ids: np.ndarray, edge_to_id: dict[tuple[int, int], int]) -> np.ndarray:
    out = np.empty((cell_ids.size, 9), dtype=np.int64)
    num_edges = len(edge_to_id)
    for row, cell_id in enumerate(cell_ids):
        cell = mesh.cells[int(cell_id)]
        out[row, :4] = cell
        for edge_local_id, (i, j) in enumerate(_LOCAL_EDGES):
            a = int(cell[i])
            b = int(cell[j])
            if a > b:
                a, b = b, a
            out[row, 4 + edge_local_id] = mesh.num_nodes + edge_to_id[(a, b)]
        out[row, 8] = mesh.num_nodes + num_edges + int(cell_id)
    return out


def _map_to_physical(
    cell_nodes: np.ndarray,
    points: np.ndarray,
    q2_grad_ref: np.ndarray,
    q1_grad_ref: np.ndarray,
    q1_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_cells = cell_nodes.shape[0]
    num_points = points.shape[0]
    jac = np.einsum("cva,vpd->cpad", cell_nodes, q1_grad_ref)
    det_j = np.linalg.det(jac)
    if np.any(det_j <= 0.0):
        raise ValueError("Encountered a non-positive cell Jacobian determinant")
    inv_j = np.linalg.inv(jac)
    q2_grads = np.einsum("fpd,cpde->cfpe", q2_grad_ref, inv_j)
    q1_grads = np.einsum("fpd,cpde->cfpe", q1_grad_ref, inv_j)
    physical = np.einsum("vf,cvd->cfd", q1_values, cell_nodes)
    assert physical.shape == (num_cells, num_points, 2)
    return q2_grads, q1_grads, det_j, physical
