from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import NavierStokesConfig
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
        weights = self.downstream_weights
        if not np.any(weights):
            return 0.0
        ux, uy, _ = self._split_global(state)
        dx = ux - 1.0
        density = 0.5 * (dx * dx + uy * uy) * ux
        return float(np.dot(weights, density))

    def downstream_power_gradient(self, state: np.ndarray) -> np.ndarray:
        weights = self.downstream_weights
        if not np.any(weights):
            return np.zeros(self.state_size)
        ux, uy, _ = self._split_global(state)
        dx = ux - 1.0
        gx = weights * (dx * ux + 0.5 * (dx * dx + uy * uy))
        gy = weights * ux * uy
        return self._combine_global(gx, gy, np.zeros(self.assembler.num_pressure_dofs))

    @property
    def downstream_weights(self) -> np.ndarray:
        if not hasattr(self, "_downstream_weights"):
            object.__setattr__(self, "_downstream_weights", _q2_boundary_weights(self.assembler.space, sideset=1))
        return self._downstream_weights

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


def _q2_boundary_weights(space, sideset: int) -> np.ndarray:
    weights = np.zeros(space.num_velocity_dofs)
    active = set(int(i) for i in space.velocity_dofs.ravel())
    for a, b in space.mesh.side_edges[sideset]:
        lo, hi = (int(a), int(b)) if a < b else (int(b), int(a))
        edge_dof = space.mesh.num_nodes + space.edge_to_id[(lo, hi)]
        if edge_dof not in active:
            continue
        length = float(np.linalg.norm(space.mesh.nodes[int(a)] - space.mesh.nodes[int(b)]))
        if int(a) in active:
            weights[int(a)] += length / 6.0
        weights[edge_dof] += 4.0 * length / 6.0
        if int(b) in active:
            weights[int(b)] += length / 6.0
    return weights
