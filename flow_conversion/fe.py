from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import DarcyMesh


_P1_REF_NODES = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
_P1_GRAD_REF = np.asarray([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)


def triangle_cubature(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Return reference-triangle cubature points and weights.

    The weights integrate over the reference triangle with vertices
    ``(0, 0), (1, 0), (0, 1)``, so they sum to ``0.5``.
    """
    if degree == -1:
        return _P1_REF_NODES.copy(), np.full(3, 1.0 / 6.0)
    if degree <= 1:
        return np.asarray([[1.0 / 3.0, 1.0 / 3.0]]), np.asarray([0.5])
    if degree <= 2:
        pts = np.asarray([[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]])
        return pts, np.full(3, 1.0 / 6.0)

    # Dunavant degree-6 rule.  Published weights sum to 1 on the unit-area
    # barycentric triangle, so halve them for this reference cell.
    a1 = 0.249286745170910
    b1 = 0.501426509658179
    w1 = 0.116786275726379
    a2 = 0.063089014491502
    b2 = 0.873821971016996
    w2 = 0.050844906370207
    a3 = 0.310352451033785
    b3 = 0.636502499121399
    c3 = 0.053145049844816
    w3 = 0.082851075618374
    bary = [
        (b1, a1, a1),
        (a1, b1, a1),
        (a1, a1, b1),
        (b2, a2, a2),
        (a2, b2, a2),
        (a2, a2, b2),
        (a3, b3, c3),
        (a3, c3, b3),
        (b3, a3, c3),
        (b3, c3, a3),
        (c3, a3, b3),
        (c3, b3, a3),
    ]
    weights = np.asarray([w1, w1, w1, w2, w2, w2, w3, w3, w3, w3, w3, w3]) * 0.5
    pts = np.asarray([[l1, l2] for _, l1, l2 in bary], dtype=float)
    return pts, weights


def line_cubature(degree: int) -> tuple[np.ndarray, np.ndarray]:
    n = max(1, (degree + 2) // 2)
    return np.polynomial.legendre.leggauss(n)


def p1_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xi = points[:, 0]
    eta = points[:, 1]
    values = np.vstack([1.0 - xi - eta, xi, eta])
    grads = np.broadcast_to(_P1_GRAD_REF[:, None, :], (3, points.shape[0], 2)).copy()
    return values, grads


def p0_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.ones((1, points.shape[0]))
    grads = np.zeros((1, points.shape[0], 2))
    return values, grads


def side_reference_points(local_side: int, t: np.ndarray) -> np.ndarray:
    if local_side == 0:
        return np.column_stack([(1.0 + t) * 0.5, np.zeros_like(t)])
    if local_side == 1:
        return np.column_stack([(1.0 - t) * 0.5, (1.0 + t) * 0.5])
    if local_side == 2:
        return np.column_stack([np.zeros_like(t), (1.0 - t) * 0.5])
    raise ValueError(f"Invalid triangle local side {local_side}")


@dataclass(frozen=True)
class TriP1Space:
    mesh: DarcyMesh
    cubature_degree: int
    cub_points_ref: np.ndarray
    cub_weights_ref: np.ndarray
    basis_values: np.ndarray
    basis_grads: np.ndarray
    physical_points: np.ndarray
    physical_weights: np.ndarray
    cell_dofs: np.ndarray
    cell_jacobians: np.ndarray

    @classmethod
    def from_mesh(cls, mesh: DarcyMesh, cubature_degree: int) -> "TriP1Space":
        if mesh.element_type != "TRI":
            raise NotImplementedError("Only TRI elements are implemented for the filtered Darcy port")
        points, weights = triangle_cubature(cubature_degree)
        values_ref, grad_ref = p1_basis(points)
        cell_nodes = mesh.nodes[mesh.cells]
        jac = np.empty((mesh.num_cells, 2, 2))
        jac[:, :, 0] = cell_nodes[:, 1] - cell_nodes[:, 0]
        jac[:, :, 1] = cell_nodes[:, 2] - cell_nodes[:, 0]
        det = np.linalg.det(jac)
        if np.any(det <= 0.0):
            raise ValueError("Encountered a non-positive triangle Jacobian determinant")
        inv_jac = np.linalg.inv(jac)
        grads = np.einsum("fpd,cde->cfpe", grad_ref, inv_jac)
        physical = (
            cell_nodes[:, 0, None, :]
            + points[None, :, 0, None] * (cell_nodes[:, 1, None, :] - cell_nodes[:, 0, None, :])
            + points[None, :, 1, None] * (cell_nodes[:, 2, None, :] - cell_nodes[:, 0, None, :])
        )
        physical_weights = det[:, None] * weights[None, :]
        return cls(
            mesh=mesh,
            cubature_degree=cubature_degree,
            cub_points_ref=points,
            cub_weights_ref=weights,
            basis_values=values_ref,
            basis_grads=grads,
            physical_points=physical,
            physical_weights=physical_weights,
            cell_dofs=mesh.cells.copy(),
            cell_jacobians=jac,
        )

    @property
    def num_dofs(self) -> int:
        return self.mesh.num_nodes

    @property
    def num_cells(self) -> int:
        return self.mesh.num_cells

    @property
    def num_points(self) -> int:
        return int(self.cub_points_ref.shape[0])

    def local_coefficients(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=float)
        if arr.shape != (self.num_dofs,):
            raise ValueError(f"Expected vector of shape {(self.num_dofs,)}, got {arr.shape}")
        return arr[self.cell_dofs]

    def evaluate_value(self, vector: np.ndarray) -> np.ndarray:
        coeff = self.local_coefficients(vector)
        return np.einsum("cf,fp->cp", coeff, self.basis_values)

    def evaluate_gradient(self, vector: np.ndarray) -> np.ndarray:
        coeff = self.local_coefficients(vector)
        return np.einsum("cf,cfpd->cpd", coeff, self.basis_grads)


@dataclass(frozen=True)
class TriP0Space:
    mesh: DarcyMesh
    cubature_degree: int
    basis_values: np.ndarray
    cell_dofs: np.ndarray

    @classmethod
    def from_mesh(cls, mesh: DarcyMesh, cubature_degree: int) -> "TriP0Space":
        points, _ = triangle_cubature(cubature_degree)
        values, _ = p0_basis(points)
        return cls(
            mesh=mesh,
            cubature_degree=cubature_degree,
            basis_values=values,
            cell_dofs=np.arange(mesh.num_cells, dtype=np.int64)[:, None],
        )

    @property
    def num_dofs(self) -> int:
        return self.mesh.num_cells

    def evaluate_value(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=float)
        if arr.shape != (self.num_dofs,):
            raise ValueError(f"Expected vector of shape {(self.num_dofs,)}, got {arr.shape}")
        return arr[:, None] * self.basis_values[0][None, :]
