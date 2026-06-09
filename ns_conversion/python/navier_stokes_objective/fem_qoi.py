from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import NavierStokesConfig
from .fe import q2_basis
from .fem_assembler import TaylorHoodAssembler


StateObjectiveType = Literal["Dissipation", "Tracking"]


@dataclass(frozen=True)
class NavierStokesFEObjective:
    assembler: TaylorHoodAssembler
    config: NavierStokesConfig

    @property
    def num_velocity_dofs(self) -> int:
        return self.assembler.num_velocity_dofs

    @property
    def state_size(self) -> int:
        return self.assembler.state_size

    @property
    def viscosity(self) -> float:
        return self.config.viscosity

    def state_value(self, state: np.ndarray, objective_type: str) -> float:
        ux, uy, _ = self._split_global(state)
        local_ux = ux[self.assembler.space.velocity_dofs]
        local_uy = uy[self.assembler.space.velocity_dofs]
        if objective_type == "Dissipation":
            return self._dissipation_value(local_ux, local_uy)
        if objective_type == "Tracking":
            return self._tracking_value(local_ux, local_uy)
        raise NotImplementedError(f"FE QoI {objective_type!r} is not implemented")

    def state_gradient(self, state: np.ndarray, objective_type: str) -> np.ndarray:
        ux, uy, _ = self._split_global(state)
        local_ux = ux[self.assembler.space.velocity_dofs]
        local_uy = uy[self.assembler.space.velocity_dofs]
        if objective_type == "Dissipation":
            gx, gy = self._dissipation_gradient(local_ux, local_uy)
        elif objective_type == "Tracking":
            gx, gy = self._tracking_gradient(local_ux, local_uy)
        else:
            raise NotImplementedError(f"FE QoI {objective_type!r} is not implemented")
        return self._scatter_velocity_gradient(gx, gy)

    def integrated_value(self, state: np.ndarray, z: float) -> float:
        value = self.config.state_cost * self.state_value(state, self.config.integrated_objective_type)
        value += self.config.state_boundary_cost * self.downstream_power_value(state)
        value += 0.5 * self.config.control_cost * float(z) ** 2
        return float(value)

    def integrated_state_gradient(self, state: np.ndarray) -> np.ndarray:
        grad = self.config.state_cost * self.state_gradient(state, self.config.integrated_objective_type)
        grad += self.config.state_boundary_cost * self.downstream_power_gradient(state)
        return grad

    def integrated_control_gradient(self, z: float) -> float:
        return self.config.control_cost * float(z)

    def final_value(self, state: np.ndarray) -> float:
        return self.config.final_time_state_cost * self.state_value(state, self.config.final_time_objective_type)

    def final_state_gradient(self, state: np.ndarray) -> np.ndarray:
        return self.config.final_time_state_cost * self.state_gradient(state, self.config.final_time_objective_type)

    def downstream_power_value(self, state: np.ndarray) -> float:
        ux, uy, _ = self._split_global(state)
        local_ux = ux[self.assembler.space.velocity_dofs]
        local_uy = uy[self.assembler.space.velocity_dofs]
        total = 0.0
        for row, values, weights in self.downstream_boundary_quadrature:
            ux_val = local_ux[row] @ values
            uy_val = local_uy[row] @ values
            dx = ux_val - 1.0
            density = 0.5 * (dx * dx + uy_val * uy_val) * ux_val
            total += float(np.dot(weights, density))
        return total

    def downstream_power_gradient(self, state: np.ndarray) -> np.ndarray:
        ux, uy, _ = self._split_global(state)
        local_ux = ux[self.assembler.space.velocity_dofs]
        local_uy = uy[self.assembler.space.velocity_dofs]
        gx_local = np.zeros_like(local_ux)
        gy_local = np.zeros_like(local_uy)
        for row, values, weights in self.downstream_boundary_quadrature:
            ux_val = local_ux[row] @ values
            uy_val = local_uy[row] @ values
            dx = ux_val - 1.0
            gx_val = dx * ux_val + 0.5 * (dx * dx + uy_val * uy_val)
            gy_val = ux_val * uy_val
            gx_local[row] += np.einsum("p,fp->f", weights * gx_val, values)
            gy_local[row] += np.einsum("p,fp->f", weights * gy_val, values)
        return self._scatter_velocity_gradient(gx_local, gy_local)

    @property
    def downstream_boundary_quadrature(self) -> tuple[tuple[int, np.ndarray, np.ndarray], ...]:
        if not hasattr(self, "_downstream_boundary_quadrature"):
            object.__setattr__(
                self,
                "_downstream_boundary_quadrature",
                _q2_boundary_quadrature(self.assembler.space, sideset=1, degree=self.config.boundary_cubature_degree),
            )
        return self._downstream_boundary_quadrature

    def _dissipation_value(self, local_ux: np.ndarray, local_uy: np.ndarray) -> float:
        sp = self.assembler.space
        grad_ux = sp.evaluate_gradient(local_ux, "q2")
        grad_uy = sp.evaluate_gradient(local_uy, "q2")
        sigma00 = 2.0 * grad_ux[:, :, 0]
        sigma11 = 2.0 * grad_uy[:, :, 1]
        sigma01 = grad_ux[:, :, 1] + grad_uy[:, :, 0]
        weighted = sp.det_jacobian * sp.cub_weights[None, :]
        total = np.sum(weighted * (sigma00 * sigma00 + sigma11 * sigma11 + 2.0 * sigma01 * sigma01))
        return float(0.5 * self.viscosity * total)

    def _dissipation_gradient(self, local_ux: np.ndarray, local_uy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sp = self.assembler.space
        grad_ux = sp.evaluate_gradient(local_ux, "q2")
        grad_uy = sp.evaluate_gradient(local_uy, "q2")
        sigma00 = 2.0 * grad_ux[:, :, 0]
        sigma11 = 2.0 * grad_uy[:, :, 1]
        sigma01 = grad_ux[:, :, 1] + grad_uy[:, :, 0]
        gx = self.viscosity * (
            2.0 * np.einsum("cp,cfp->cf", sigma00, sp.q2_weighted_grads[:, :, :, 0])
            + 2.0 * np.einsum("cp,cfp->cf", sigma01, sp.q2_weighted_grads[:, :, :, 1])
        )
        gy = self.viscosity * (
            2.0 * np.einsum("cp,cfp->cf", sigma11, sp.q2_weighted_grads[:, :, :, 1])
            + 2.0 * np.einsum("cp,cfp->cf", sigma01, sp.q2_weighted_grads[:, :, :, 0])
        )
        return gx, gy

    def _tracking_value(self, local_ux: np.ndarray, local_uy: np.ndarray) -> float:
        sp = self.assembler.space
        ux_val = sp.evaluate_value(local_ux, "q2")
        uy_val = sp.evaluate_value(local_uy, "q2")
        target_x, target_y = self._tracking_target()
        dx = ux_val - target_x
        dy = uy_val - target_y
        weighted = sp.det_jacobian * sp.cub_weights[None, :]
        return float(0.5 * np.sum(weighted * (dx * dx + dy * dy)))

    def _tracking_gradient(self, local_ux: np.ndarray, local_uy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sp = self.assembler.space
        ux_val = sp.evaluate_value(local_ux, "q2")
        uy_val = sp.evaluate_value(local_uy, "q2")
        target_x, target_y = self._tracking_target()
        gx = np.einsum("cp,cfp->cf", ux_val - target_x, sp.q2_weighted_values)
        gy = np.einsum("cp,cfp->cf", uy_val - target_y, sp.q2_weighted_values)
        return gx, gy

    def _tracking_target(self) -> tuple[np.ndarray, np.ndarray]:
        y = self.assembler.space.physical_points[:, :, 1]
        if self.config.use_parabolic_inflow:
            target_x = (2.0 + y) * (2.0 - y)
        else:
            target_x = np.ones_like(y)
        return target_x, np.zeros_like(y)

    def _scatter_velocity_gradient(self, local_gx: np.ndarray, local_gy: np.ndarray) -> np.ndarray:
        v = self.num_velocity_dofs
        grad = np.zeros(self.state_size)
        np.add.at(grad, self.assembler.space.velocity_dofs.ravel(), local_gx.ravel())
        np.add.at(grad, v + self.assembler.space.velocity_dofs.ravel(), local_gy.ravel())
        return grad

    def _split_global(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(state, dtype=float)
        if arr.shape != (self.state_size,):
            raise ValueError(f"Expected state shape {(self.state_size,)}, got {arr.shape}")
        v = self.num_velocity_dofs
        return arr[:v], arr[v : 2 * v], arr[2 * v :]

    def _combine_global(self, ux: np.ndarray, uy: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return np.concatenate([ux, uy, pressure])


def _line_gauss(degree: int) -> tuple[np.ndarray, np.ndarray]:
    n_1d = max(1, (int(degree) + 2) // 2)
    return np.polynomial.legendre.leggauss(n_1d)


def _side_reference_points(side: int, t: np.ndarray) -> np.ndarray:
    if side == 0:
        return np.column_stack([t, -np.ones_like(t)])
    if side == 1:
        return np.column_stack([np.ones_like(t), t])
    if side == 2:
        return np.column_stack([t, np.ones_like(t)])
    if side == 3:
        return np.column_stack([-np.ones_like(t), t])
    raise ValueError(f"Unexpected quadrilateral side id {side}")


def _q2_boundary_quadrature(space, sideset: int, degree: int) -> tuple[tuple[int, np.ndarray, np.ndarray], ...]:
    points, line_weights = _line_gauss(degree)
    values_by_side = {
        side: q2_basis(_side_reference_points(side, points))[0]
        for side in range(4)
    }
    cell_to_row = {int(cell_id): row for row, cell_id in enumerate(space.cell_ids)}
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    for side, cell_ids in enumerate(space.mesh.side_sets[sideset]):
        values = values_by_side[side]
        loc_a, loc_b = ((0, 1), (1, 2), (2, 3), (3, 0))[side]
        for cell_id in cell_ids:
            row = cell_to_row.get(int(cell_id))
            if row is None:
                continue
            cell = space.mesh.cells[int(cell_id)]
            a = int(cell[loc_a])
            b = int(cell[loc_b])
            length = float(np.linalg.norm(space.mesh.nodes[a] - space.mesh.nodes[b]))
            out.append((row, values, 0.5 * length * line_weights))
    return tuple(out)
