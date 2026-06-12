from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .config import FilteredDarcyConfig, read_config
from .fe import TriP0Space, TriP1Space, line_cubature, p1_basis, side_reference_points
from .mesh import DarcyMesh, read_darcy_mesh
from .permeability import Permeability


PATM = 101.325


@dataclass
class _StateData:
    filtered: np.ndarray
    theta: np.ndarray | None
    operator: sparse.csr_matrix
    rhs: np.ndarray
    state: np.ndarray
    control_jacobian: sparse.csr_matrix
    adjoint: np.ndarray


class FilteredDarcyObjective:
    """Python port of the ROL filtered Darcy objective ``fobj``.

    The implemented branch matches the supplied ``filteredDarcy/input.xml``:
    triangular cells, P1 pressure, P1 filtered control, P0 density, and the
    optional two-entry constant-velocity parameter block.
    """

    def __init__(
        self,
        input_xml: str | Path,
        mesh_path: str | Path | None = None,
        use_param_var: bool | None = None,
    ) -> None:
        self.config = read_config(input_xml, use_param_var=use_param_var)
        mesh_file = Path(mesh_path).expanduser() if mesh_path is not None else Path(self.config.mesh_file)
        if not mesh_file.is_absolute():
            mesh_file = self.config.xml_path.parent / mesh_file
        if not mesh_file.exists():
            raise FileNotFoundError(
                f"Mesh file {mesh_file} does not exist. Pass mesh_path explicitly if it was generated elsewhere."
            )
        self.mesh = read_darcy_mesh(mesh_file)
        self._check_supported_branch()

        self.pressure = TriP1Space.from_mesh(self.mesh, self.config.cubature_degree)
        self.control = self.pressure
        self.filter_space = TriP1Space.from_mesh(self.mesh, self.config.filter_cubature_degree)
        self.density = TriP0Space.from_mesh(self.mesh, self.config.filter_cubature_degree)
        self.permeability = Permeability.from_config(self.config)

        self._outflow_rows = self._build_outflow_rows()
        self._inlet_loads = self._build_inlet_loads()
        self._target, self._weight = self._build_target_and_weight()
        self._filter_matrix, self._filter_mass = self._assemble_filter()
        self._filter_solver = spla.splu(self._filter_matrix.tocsc())

    @classmethod
    def from_xml(
        cls,
        input_xml: str | Path,
        mesh_path: str | Path | None = None,
        **kwargs: Any,
    ) -> "FilteredDarcyObjective":
        return cls(input_xml, mesh_path=mesh_path, **kwargs)

    @property
    def density_size(self) -> int:
        return self.density.num_dofs

    @property
    def parameter_size(self) -> int:
        return 2 if self.config.use_param_var else 0

    @property
    def size(self) -> int:
        return self.density_size + self.parameter_size

    def initial_guess(self, density_value: float = 0.5) -> np.ndarray:
        density = np.full(self.density_size, float(density_value))
        if self.parameter_size:
            params = np.full(self.parameter_size, float(density_value))
            return self.pack(density, params)
        return self.pack(density)

    def pack(self, density: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        rho = np.asarray(density, dtype=float).reshape(-1)
        if rho.shape != (self.density_size,):
            raise ValueError(f"Expected density shape {(self.density_size,)}, got {rho.shape}")
        if self.parameter_size == 0:
            if params is not None and np.asarray(params).size:
                raise ValueError("This objective was constructed without a parameter block")
            return rho.copy()
        if params is None:
            theta = np.zeros(self.parameter_size)
        else:
            theta = np.asarray(params, dtype=float).reshape(-1)
        if theta.shape != (self.parameter_size,):
            raise ValueError(f"Expected parameter shape {(self.parameter_size,)}, got {theta.shape}")
        return np.concatenate([rho, theta])

    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.shape != (self.size,):
            raise ValueError(f"Expected flat vector of shape {(self.size,)}, got {arr.shape}")
        rho = arr[: self.density_size].copy()
        if self.parameter_size:
            return rho, arr[self.density_size :].copy()
        return rho, None

    def value(self, x: np.ndarray) -> float:
        rho, theta = self.unpack(x)
        filtered = self.apply_filter(rho)
        operator, rhs = self._assemble_darcy_operator(filtered)
        state = spla.spsolve(operator, rhs)
        return float(self._qoi_value(state, filtered, theta))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        rho, theta = self.unpack(x)
        data = self._state_data(rho, theta)
        g_f, g_theta = self._reduced_gradient(data)
        g_rho = self.apply_filter_transpose(g_f)
        if self.parameter_size:
            return np.concatenate([g_rho, g_theta])
        return g_rho

    def hess_vec(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        rho, theta = self.unpack(x)
        v_rho, v_theta = self.unpack(v)
        data = self._state_data(rho, theta)
        v_filtered = self.apply_filter(v_rho)
        hv_f, hv_theta = self._reduced_hess_vec(data, v_filtered, v_theta)
        hv_rho = self.apply_filter_transpose(hv_f)
        if self.parameter_size:
            return np.concatenate([hv_rho, hv_theta])
        return hv_rho

    def apply_filter(self, density: np.ndarray) -> np.ndarray:
        rho = np.asarray(density, dtype=float).reshape(-1)
        if rho.shape != (self.density_size,):
            raise ValueError(f"Expected density shape {(self.density_size,)}, got {rho.shape}")
        return np.asarray(self._filter_solver.solve(self._filter_mass @ rho), dtype=float)

    def apply_filter_transpose(self, vector: np.ndarray) -> np.ndarray:
        y = np.asarray(vector, dtype=float).reshape(-1)
        if y.shape != (self.control.num_dofs,):
            raise ValueError(f"Expected filtered-control vector shape {(self.control.num_dofs,)}, got {y.shape}")
        solved = self._filter_solver.solve(y, trans="T")
        return np.asarray(self._filter_mass.T @ solved, dtype=float)

    def solve_state(self, x: np.ndarray) -> np.ndarray:
        rho, theta = self.unpack(x)
        del theta
        filtered = self.apply_filter(rho)
        operator, rhs = self._assemble_darcy_operator(filtered)
        return np.asarray(spla.spsolve(operator, rhs), dtype=float)

    def _check_supported_branch(self) -> None:
        cfg = self.config
        if self.mesh.element_type != "TRI" or cfg.element_type != "TRI":
            raise NotImplementedError("The Python filtered Darcy port currently supports Element Type=TRI only")
        if cfg.pressure_basis_degree != 1 or cfg.filter_basis_degree != 1:
            raise NotImplementedError("Only P1 pressure and P1 filtered-control bases are implemented")
        if cfg.density_basis_degree != 0:
            raise NotImplementedError("Only P0 density controls are implemented")

    def _build_outflow_rows(self) -> list[set[int]]:
        rows: list[set[int]] = [set() for _ in range(self.mesh.num_cells)]
        if len(self.mesh.side_sets) <= 2:
            return rows
        for local_side, cells in enumerate(self.mesh.side_sets[2]):
            for cell in cells:
                rows[int(cell)].update(self.mesh.local_sides[local_side])
        return rows

    def _build_inlet_loads(self) -> np.ndarray:
        loads = np.zeros((self.mesh.num_cells, 3))
        if len(self.mesh.side_sets) == 0:
            return loads
        t, w = line_cubature(self.config.boundary_cubature_degree)
        for local_side, cells in enumerate(self.mesh.side_sets[0]):
            ref_pts = side_reference_points(local_side, t)
            values, _ = p1_basis(ref_pts)
            loc_a, loc_b = self.mesh.local_sides[local_side]
            for cell in cells:
                cell_i = int(cell)
                a = self.mesh.nodes[self.mesh.cells[cell_i, loc_a]]
                b = self.mesh.nodes[self.mesh.cells[cell_i, loc_b]]
                length = float(np.linalg.norm(b - a))
                phys = 0.5 * ((1.0 - t)[:, None] * a[None, :] + (1.0 + t)[:, None] * b[None, :])
                weighted = 0.5 * length * w * phys[:, 0]
                loads[cell_i] += -self.config.inlet_velocity * np.einsum("p,ip->i", weighted, values)
        return loads

    def _build_target_and_weight(self) -> tuple[np.ndarray, np.ndarray]:
        points = self.pressure.physical_points
        r = points[:, :, 0]
        z = points[:, :, 1]
        target = np.zeros(points.shape)
        rad = self.config.diffuser_radius
        if self.config.target_type == 1:
            denom = rad * rad - z * z
            target[:, :, 0] = -r * z / (denom * denom)
            target[:, :, 1] = 1.0 / denom
        elif self.config.target_type == 2:
            rv = self.config.rvec
            zv = self.config.zvec
            for i in range(5):
                mask = (z >= zv[i]) & (z < zv[i + 1])
                slope = (rv[i + 1] - rv[i]) / (zv[i + 1] - zv[i])
                radius = rv[i] + slope * (z - zv[i])
                target[:, :, 0] = np.where(mask, r * slope / np.power(radius, 3), target[:, :, 0])
                target[:, :, 1] = np.where(mask, 1.0 / np.power(radius, 2), target[:, :, 1])
        else:
            raise NotImplementedError("Only Target Type 1 and 2 are implemented")

        base = self._base_weight(points, target)
        x_scale = 0.0 if self.config.only_axial else self.config.radial_tracking_scale
        y_scale = self.config.axial_tracking_scale
        if self.config.normalized_misfit and not self.config.only_axial:
            norm2 = target[:, :, 0] ** 2 + target[:, :, 1] ** 2
            scale = np.where(base != 0.0, 1.0 / norm2, 1.0)
            weight_r = r * base * scale
            weight_z = r * base * scale
        else:
            weight_r = r * x_scale * base
            weight_z = r * y_scale * base
        weight = np.stack([weight_r, weight_z], axis=2)
        return target, weight

    def _base_weight(self, points: np.ndarray, target: np.ndarray) -> np.ndarray:
        z = points[:, :, 1]
        if self.config.optimize_domain_fraction:
            inside = np.abs(z) <= self.config.integration_domain_fraction * self.config.diffuser_top
            if self.config.invert_domain_fraction:
                inside = ~inside
            return inside.astype(float)
        if self.config.polynomial_weight:
            p = self.config.target_weighting_power
            top = self.config.diffuser_top
            out = np.empty_like(z)
            pos = z > 0.0
            out[pos] = np.power(z[pos] - top, p) / np.power(-top, p)
            out[~pos] = np.power(z[~pos] + top, p) / np.power(top, p)
            return out
        if self.config.target_velocity_weight:
            norm2 = target[:, :, 0] ** 2 + target[:, :, 1] ** 2
            return 1.0 / norm2
        return np.ones_like(z)

    def _assemble_filter(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        rows_a: list[int] = []
        cols_a: list[int] = []
        data_a: list[float] = []
        rows_m: list[int] = []
        cols_m: list[int] = []
        data_m: list[float] = []
        eps2 = self.config.filter_radius * self.config.filter_radius
        N = self.filter_space.basis_values
        for c in range(self.mesh.num_cells):
            dofs = self.filter_space.cell_dofs[c]
            weights = self.filter_space.physical_weights[c] * self.filter_space.physical_points[c, :, 0]
            grads = self.filter_space.basis_grads[c]
            local_a = eps2 * np.einsum("p,ipd,jpd->ij", weights, grads, grads)
            local_a += np.einsum("p,ip,jp->ij", weights, N, N)
            local_m = np.einsum("p,ip->i", weights, N)
            _append_local_matrix(rows_a, cols_a, data_a, dofs, dofs, local_a)
            for i, row in enumerate(dofs):
                rows_m.append(int(row))
                cols_m.append(c)
                data_m.append(float(local_m[i]))
        shape_a = (self.filter_space.num_dofs, self.filter_space.num_dofs)
        shape_m = (self.filter_space.num_dofs, self.density.num_dofs)
        return (
            sparse.coo_matrix((data_a, (rows_a, cols_a)), shape=shape_a).tocsr(),
            sparse.coo_matrix((data_m, (rows_m, cols_m)), shape=shape_m).tocsr(),
        )

    def _assemble_darcy_operator(self, filtered: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray]:
        z_val = self.control.evaluate_value(filtered)
        alpha = self.permeability.compute(z_val, self.pressure.physical_points, 0)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        q = np.zeros(self.pressure.num_dofs)
        for c in range(self.mesh.num_cells):
            dofs = self.pressure.cell_dofs[c]
            weights = self.pressure.physical_weights[c] * self.pressure.physical_points[c, :, 0] * alpha[c]
            grads = self.pressure.basis_grads[c]
            local = np.einsum("p,ipd,jpd->ij", weights, grads, grads)
            local_q = self._inlet_loads[c].copy()
            for row in self._outflow_rows[c]:
                local[row, :] = 0.0
                local[row, row] = 1.0
                local_q[row] = -PATM
            _append_local_matrix(rows, cols, data, dofs, dofs, local)
            for i, row in enumerate(dofs):
                q[int(row)] += local_q[i]
        operator = sparse.coo_matrix(
            (data, (rows, cols)), shape=(self.pressure.num_dofs, self.pressure.num_dofs)
        ).tocsr()
        return operator, -q

    def _assemble_control_jacobian(self, state: np.ndarray, filtered: np.ndarray) -> sparse.csr_matrix:
        grad_u = self.pressure.evaluate_gradient(state)
        z_val = self.control.evaluate_value(filtered)
        alpha1 = self.permeability.compute(z_val, self.pressure.physical_points, 1)
        N = self.control.basis_values
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for c in range(self.mesh.num_cells):
            dofs = self.pressure.cell_dofs[c]
            weights = self.pressure.physical_weights[c] * self.pressure.physical_points[c, :, 0] * alpha1[c]
            grads = self.pressure.basis_grads[c]
            grad_dot = np.einsum("pd,ipd->ip", grad_u[c], grads)
            local = np.einsum("p,ip,jp->ij", weights, grad_dot, N)
            for row in self._outflow_rows[c]:
                local[row, :] = 0.0
            _append_local_matrix(rows, cols, data, dofs, dofs, local)
        return sparse.coo_matrix(
            (data, (rows, cols)), shape=(self.pressure.num_dofs, self.control.num_dofs)
        ).tocsr()

    def _state_data(self, rho: np.ndarray, theta: np.ndarray | None) -> _StateData:
        filtered = self.apply_filter(rho)
        operator, rhs = self._assemble_darcy_operator(filtered)
        state = np.asarray(spla.spsolve(operator, rhs), dtype=float)
        j2 = self._assemble_control_jacobian(state, filtered)
        grad_u = self._qoi_gradient_state(state, filtered, theta)
        adjoint = -np.asarray(spla.spsolve(operator.T.tocsr(), grad_u), dtype=float)
        return _StateData(filtered, theta, operator, rhs, state, j2, adjoint)

    def _reduced_gradient(self, data: _StateData) -> tuple[np.ndarray, np.ndarray]:
        grad_f = self._qoi_gradient_control(data.state, data.filtered, data.theta)
        grad_f = grad_f + data.control_jacobian.T @ data.adjoint
        if self.parameter_size:
            grad_theta = self._qoi_gradient_parameter(data.state, data.filtered, data.theta)
        else:
            grad_theta = np.zeros(0)
        return np.asarray(grad_f, dtype=float), grad_theta

    def _reduced_hess_vec(
        self,
        data: _StateData,
        v_filtered: np.ndarray,
        v_theta: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        state_sens_rhs = -(data.control_jacobian @ v_filtered)
        state_sens = np.asarray(spla.spsolve(data.operator, state_sens_rhs), dtype=float)

        adj_rhs = self._qoi_hess_state(data.state, data.filtered, data.theta, state_sens, v_filtered, v_theta)
        adj_rhs = adj_rhs + self._constraint_h21_apply(data.adjoint, data.filtered, v_filtered)
        adj_sens = -np.asarray(spla.spsolve(data.operator.T.tocsr(), adj_rhs), dtype=float)

        hv_f = data.control_jacobian.T @ adj_sens
        hv_f = hv_f + self._qoi_hess_control(data.state, data.filtered, data.theta, state_sens, v_filtered, v_theta)
        hv_f = hv_f + self._constraint_h12_apply(data.adjoint, data.filtered, state_sens)
        hv_f = hv_f + self._constraint_h22_apply(data.adjoint, data.state, data.filtered, v_filtered)

        if self.parameter_size:
            hv_theta = self._qoi_hess_parameter(data.state, data.filtered, data.theta, state_sens, v_filtered, v_theta)
        else:
            hv_theta = np.zeros(0)
        return np.asarray(hv_f, dtype=float), hv_theta

    def _theta0(self, theta: np.ndarray | None) -> float:
        if theta is None:
            return self.config.target_axial_velocity
        return float(theta[0])

    def _qoi_common(self, state: np.ndarray, filtered: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        grad_u = self.pressure.evaluate_gradient(state)
        z_val = self.control.evaluate_value(filtered)
        alpha = self.permeability.compute(z_val, self.pressure.physical_points, 0)
        alpha1 = self.permeability.compute(z_val, self.pressure.physical_points, 1)
        return grad_u, z_val, alpha, alpha1

    def _qoi_value(self, state: np.ndarray, filtered: np.ndarray, theta: np.ndarray | None) -> float:
        grad_u, _, alpha, _ = self._qoi_common(state, filtered)
        vel = alpha[:, :, None] * grad_u + self._theta0(theta) * self._target
        integrand = np.sum(self._weight * vel * vel, axis=2)
        return 0.5 * float(np.sum(self.pressure.physical_weights * integrand))

    def _qoi_gradient_state(self, state: np.ndarray, filtered: np.ndarray, theta: np.ndarray | None) -> np.ndarray:
        grad_u, _, alpha, _ = self._qoi_common(state, filtered)
        vel = alpha[:, :, None] * grad_u + self._theta0(theta) * self._target
        awvel = self._weight * vel * alpha[:, :, None]
        return self._assemble_state_vector_from_grad_integrand(awvel)

    def _qoi_gradient_control(self, state: np.ndarray, filtered: np.ndarray, theta: np.ndarray | None) -> np.ndarray:
        grad_u, _, alpha, alpha1 = self._qoi_common(state, filtered)
        vel = alpha[:, :, None] * grad_u + self._theta0(theta) * self._target
        deriv = np.sum(self._weight * vel * alpha1[:, :, None] * grad_u, axis=2)
        return self._assemble_control_vector_from_value_integrand(deriv)

    def _qoi_gradient_parameter(self, state: np.ndarray, filtered: np.ndarray, theta: np.ndarray | None) -> np.ndarray:
        grad_u, _, alpha, _ = self._qoi_common(state, filtered)
        vel = alpha[:, :, None] * grad_u + self._theta0(theta) * self._target
        out = np.zeros(self.parameter_size)
        out[0] = float(np.sum(self.pressure.physical_weights * np.sum(self._weight * vel * self._target, axis=2)))
        return out

    def _qoi_hess_state(
        self,
        state: np.ndarray,
        filtered: np.ndarray,
        theta: np.ndarray | None,
        state_dir: np.ndarray,
        control_dir: np.ndarray,
        theta_dir: np.ndarray | None,
    ) -> np.ndarray:
        grad_u, _, alpha, alpha1 = self._qoi_common(state, filtered)
        grad_s = self.pressure.evaluate_gradient(state_dir)
        val_v = self.control.evaluate_value(control_dir)
        vel = alpha[:, :, None] * grad_u + self._theta0(theta) * self._target
        integrand = self._weight * alpha[:, :, None] * alpha[:, :, None] * grad_s
        integrand += self._weight * (
            (2.0 * alpha[:, :, None] * grad_u + self._theta0(theta) * self._target)
            * alpha1[:, :, None]
            * val_v[:, :, None]
        )
        if theta_dir is not None and theta_dir.size:
            integrand += self._weight * alpha[:, :, None] * self._target * float(theta_dir[0])
        del vel
        return self._assemble_state_vector_from_grad_integrand(integrand)

    def _qoi_hess_control(
        self,
        state: np.ndarray,
        filtered: np.ndarray,
        theta: np.ndarray | None,
        state_dir: np.ndarray,
        control_dir: np.ndarray,
        theta_dir: np.ndarray | None,
    ) -> np.ndarray:
        grad_u = self.pressure.evaluate_gradient(state)
        grad_s = self.pressure.evaluate_gradient(state_dir)
        z_val = self.control.evaluate_value(filtered)
        val_v = self.control.evaluate_value(control_dir)
        alpha = self.permeability.compute(z_val, self.pressure.physical_points, 0)
        alpha1 = self.permeability.compute(z_val, self.pressure.physical_points, 1)
        alpha2 = self.permeability.compute(z_val, self.pressure.physical_points, 2)
        vel = alpha[:, :, None] * grad_u + self._theta0(theta) * self._target
        deriv = np.sum(self._weight * vel * alpha1[:, :, None] * grad_s, axis=2)
        deriv += np.sum(self._weight * alpha1[:, :, None] * grad_u * alpha[:, :, None] * grad_s, axis=2)
        deriv += np.sum(self._weight * vel * alpha2[:, :, None] * val_v[:, :, None] * grad_u, axis=2)
        deriv += np.sum(self._weight * alpha1[:, :, None] * grad_u * alpha1[:, :, None] * val_v[:, :, None] * grad_u, axis=2)
        if theta_dir is not None and theta_dir.size:
            deriv += np.sum(
                self._weight * alpha1[:, :, None] * grad_u * self._target * float(theta_dir[0]),
                axis=2,
            )
        return self._assemble_control_vector_from_value_integrand(deriv)

    def _qoi_hess_parameter(
        self,
        state: np.ndarray,
        filtered: np.ndarray,
        theta: np.ndarray | None,
        state_dir: np.ndarray,
        control_dir: np.ndarray,
        theta_dir: np.ndarray | None,
    ) -> np.ndarray:
        grad_u = self.pressure.evaluate_gradient(state)
        grad_s = self.pressure.evaluate_gradient(state_dir)
        z_val = self.control.evaluate_value(filtered)
        val_v = self.control.evaluate_value(control_dir)
        alpha = self.permeability.compute(z_val, self.pressure.physical_points, 0)
        alpha1 = self.permeability.compute(z_val, self.pressure.physical_points, 1)
        out = np.zeros(self.parameter_size)
        h0 = np.sum(self._weight * alpha[:, :, None] * grad_s * self._target, axis=2)
        h0 += np.sum(self._weight * alpha1[:, :, None] * grad_u * val_v[:, :, None] * self._target, axis=2)
        if theta_dir is not None and theta_dir.size:
            h0 += np.sum(self._weight * self._target * self._target * float(theta_dir[0]), axis=2)
        out[0] = float(np.sum(self.pressure.physical_weights * h0))
        del theta
        return out

    def _constraint_h21_apply(self, adjoint: np.ndarray, filtered: np.ndarray, control_dir: np.ndarray) -> np.ndarray:
        grad_l = self._state_gradient_with_dirichlet_zero(adjoint)
        val_v = self.control.evaluate_value(control_dir)
        z_val = self.control.evaluate_value(filtered)
        alpha1 = self.permeability.compute(z_val, self.pressure.physical_points, 1)
        integrand = alpha1[:, :, None] * val_v[:, :, None] * grad_l
        return self._assemble_state_vector_from_grad_integrand(integrand, include_radius=True)

    def _constraint_h12_apply(self, adjoint: np.ndarray, filtered: np.ndarray, state_dir: np.ndarray) -> np.ndarray:
        grad_l = self._state_gradient_with_dirichlet_zero(adjoint)
        grad_s = self.pressure.evaluate_gradient(state_dir)
        z_val = self.control.evaluate_value(filtered)
        alpha1 = self.permeability.compute(z_val, self.pressure.physical_points, 1)
        deriv = alpha1 * np.sum(grad_l * grad_s, axis=2)
        return self._assemble_control_vector_from_value_integrand(deriv, include_radius=True)

    def _constraint_h22_apply(
        self,
        adjoint: np.ndarray,
        state: np.ndarray,
        filtered: np.ndarray,
        control_dir: np.ndarray,
    ) -> np.ndarray:
        grad_l = self._state_gradient_with_dirichlet_zero(adjoint)
        grad_u = self.pressure.evaluate_gradient(state)
        val_v = self.control.evaluate_value(control_dir)
        z_val = self.control.evaluate_value(filtered)
        alpha2 = self.permeability.compute(z_val, self.pressure.physical_points, 2)
        deriv = alpha2 * val_v * np.sum(grad_u * grad_l, axis=2)
        return self._assemble_control_vector_from_value_integrand(deriv, include_radius=True)

    def _state_gradient_with_dirichlet_zero(self, vector: np.ndarray) -> np.ndarray:
        coeff = self.pressure.local_coefficients(vector).copy()
        for c, rows in enumerate(self._outflow_rows):
            for row in rows:
                coeff[c, row] = 0.0
        return np.einsum("cf,cfpd->cpd", coeff, self.pressure.basis_grads)

    def _assemble_state_vector_from_grad_integrand(
        self,
        integrand: np.ndarray,
        *,
        include_radius: bool = False,
    ) -> np.ndarray:
        out = np.zeros(self.pressure.num_dofs)
        weights = self.pressure.physical_weights
        if include_radius:
            weights = weights * self.pressure.physical_points[:, :, 0]
        local = np.einsum("cp,cpd,cfpd->cf", weights, integrand, self.pressure.basis_grads)
        for c in range(self.mesh.num_cells):
            np.add.at(out, self.pressure.cell_dofs[c], local[c])
        return out

    def _assemble_control_vector_from_value_integrand(
        self,
        integrand: np.ndarray,
        *,
        include_radius: bool = False,
    ) -> np.ndarray:
        out = np.zeros(self.control.num_dofs)
        weights = self.pressure.physical_weights
        if include_radius:
            weights = weights * self.pressure.physical_points[:, :, 0]
        local = np.einsum("cp,cp,fp->cf", weights, integrand, self.control.basis_values)
        for c in range(self.mesh.num_cells):
            np.add.at(out, self.control.cell_dofs[c], local[c])
        return out


def _append_local_matrix(
    rows: list[int],
    cols: list[int],
    data: list[float],
    row_dofs: np.ndarray,
    col_dofs: np.ndarray,
    local: np.ndarray,
) -> None:
    for i, row in enumerate(row_dofs):
        for j, col in enumerate(col_dofs):
            value = float(local[i, j])
            if value != 0.0:
                rows.append(int(row))
                cols.append(int(col))
                data.append(value)
