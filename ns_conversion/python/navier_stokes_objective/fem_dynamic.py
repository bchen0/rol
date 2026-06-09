from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .config import NavierStokesConfig
from .fem_assembler import TaylorHoodAssembler
from .linear_solvers import SparseLinearSolver, build_sparse_linear_solver


@dataclass(frozen=True)
class DynamicNavierStokesFEConstraint:
    assembler: TaylorHoodAssembler
    dt: float
    theta: float
    use_parabolic_inflow: bool
    cylinder_center_x: float
    cylinder_center_y: float
    cylinder_radius: float
    pin_pressure: bool = False
    linear_solver: SparseLinearSolver | str = "scipy"

    @classmethod
    def from_config(
        cls,
        assembler: TaylorHoodAssembler,
        config: NavierStokesConfig,
        *,
        pin_pressure: bool = False,
        linear_solver: SparseLinearSolver | str = "scipy",
        klu2_executable: str | None = None,
    ) -> "DynamicNavierStokesFEConstraint":
        if config.use_nonpenetrating_walls:
            raise NotImplementedError("Non-penetrating wall boundary rows are not implemented in the FE dynamic slice")
        return cls(
            assembler=assembler,
            dt=config.dt,
            theta=config.theta,
            use_parabolic_inflow=config.use_parabolic_inflow,
            cylinder_center_x=config.cylinder_center_x,
            cylinder_center_y=config.cylinder_center_y,
            cylinder_radius=config.cylinder_radius,
            pin_pressure=pin_pressure,
            linear_solver=build_sparse_linear_solver(linear_solver, klu2_executable=klu2_executable),
        )

    @property
    def state_size(self) -> int:
        return self.assembler.state_size

    @property
    def num_velocity_dofs(self) -> int:
        return self.assembler.num_velocity_dofs

    def residual(self, u_old: np.ndarray, u_new: np.ndarray, z: float) -> np.ndarray:
        old = self._as_state(u_old)
        new = self._as_state(u_new)
        local_old = self.assembler.gather(old)
        local_new = self.assembler.gather(new)
        local_res = self.dt * (
            (1.0 - self.theta) * self.assembler.pde.residual(local_old)
            + self.theta * self.assembler.pde.residual(local_new)
        )
        delta_x = local_new[:, :9] - local_old[:, :9]
        delta_y = local_new[:, 9:18] - local_old[:, 9:18]
        local_res[:, :9] += np.einsum("cfg,cg->cf", self.local_velocity_mass, delta_x)
        local_res[:, 9:18] += np.einsum("cfg,cg->cf", self.local_velocity_mass, delta_y)
        self._apply_local_boundary_residual(local_res, local_new, float(z))

        res = np.zeros(self.state_size)
        np.add.at(res, self.assembler.local_global_dofs.ravel(), local_res.ravel())
        return res

    def jacobian_uold(self, u_old: np.ndarray, u_new: np.ndarray, z: float) -> sparse.csr_matrix:
        del u_new, z
        old = self._as_state(u_old)
        local_jac = self.dt * (1.0 - self.theta) * self.assembler.pde.jacobian(self.assembler.gather(old))
        local_jac[:, :9, :9] -= self.local_velocity_mass
        local_jac[:, 9:18, 9:18] -= self.local_velocity_mass
        self._zero_local_boundary_rows(local_jac)
        return self._assemble_local_jacobian(local_jac)

    def jacobian_unew(self, u_old: np.ndarray, u_new: np.ndarray, z: float) -> sparse.csr_matrix:
        del u_old, z
        new = self._as_state(u_new)
        local_jac = self.dt * self.theta * self.assembler.pde.jacobian(self.assembler.gather(new))
        local_jac[:, :9, :9] += self.local_velocity_mass
        local_jac[:, 9:18, 9:18] += self.local_velocity_mass
        self._identity_local_boundary_rows(local_jac)
        return self._assemble_local_jacobian(local_jac)

    def jacobian_z(self, u_old: np.ndarray, u_new: np.ndarray, z: float) -> np.ndarray:
        del u_old, u_new, z
        data = self.boundary_data
        local = np.zeros((self.assembler.space.num_cells, self.assembler.space.local_state_size))
        for entry in data.local_rows:
            if entry.sideset != 4:
                continue
            global_dofs = self.assembler.space.velocity_dofs[entry.cell_row, entry.local_dofs]
            local[entry.cell_row, entry.local_dofs] = -data.control_x[global_dofs]
            local[entry.cell_row, 9 + entry.local_dofs] = -data.control_y[global_dofs]
        out = np.zeros(self.state_size)
        np.add.at(out, self.assembler.local_global_dofs.ravel(), local.ravel())
        return out

    def solve(
        self,
        u_old: np.ndarray,
        z: float,
        *,
        initial_guess: np.ndarray | None = None,
        absolute_tol: float = 1e-12,
        relative_tol: float = 0.0,
        step_tol: float = 0.0,
        max_iter: int = 20,
    ) -> np.ndarray:
        old = self._as_state(u_old)
        new = old.copy() if initial_guess is None else self._as_state(initial_guess).copy()

        r0 = None
        for _ in range(max_iter):
            residual = self.residual(old, new, z)
            norm = self.residual_norm(residual)
            if r0 is None:
                r0 = max(norm, 1.0)
            if norm <= absolute_tol or norm / r0 <= relative_tol:
                return new

            jac = self.jacobian_unew(old, new, z)
            jac_active, rhs_active, active = self._active_newton_system(jac, -residual)
            step_active = self.linear_solver.solve(jac_active, rhs_active, global_ids=active)
            step = np.zeros_like(new)
            step[active] = step_active
            step_norm = float(np.linalg.norm(step))
            if step_tol > 0.0 and step_norm <= step_tol * (1.0 + float(np.linalg.norm(new))):
                return new

            alpha = 1.0
            accepted = False
            while alpha >= 1e-6:
                trial = new + alpha * step
                if self.residual_norm(self.residual(old, trial, z)) < norm:
                    new = trial
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                new = new + step

        final_norm = self.residual_norm(self.residual(old, new, z))
        raise RuntimeError(f"FE dynamic Newton solve did not converge in {max_iter} iterations; residual norm {final_norm:.3e}")

    def solve_linearized_unew(
        self,
        rhs: np.ndarray,
        u_old: np.ndarray,
        u_new: np.ndarray,
        z: float,
        *,
        transpose: bool = False,
    ) -> np.ndarray:
        """Solve the active linearization with respect to ``u_new``."""
        rhs_arr = self._as_state(rhs)
        jac = self.jacobian_unew(u_old, u_new, z)
        jac_active, rhs_active, active = self._active_newton_system(jac, rhs_arr)
        sol_active = self.linear_solver.solve(jac_active, rhs_active, transpose=transpose, global_ids=active)
        out = np.zeros_like(rhs_arr)
        out[active] = sol_active
        return out

    def solve_adjoint_unew(self, rhs: np.ndarray, u_old: np.ndarray, u_new: np.ndarray, z: float) -> np.ndarray:
        """Solve ``J_unew.T lambda = rhs`` on the active FE system."""
        return self.solve_linearized_unew(rhs, u_old, u_new, z, transpose=True)

    @property
    def velocity_mass(self) -> sparse.csr_matrix:
        if not hasattr(self, "_velocity_mass"):
            object.__setattr__(self, "_velocity_mass", self.assembler.assemble_velocity_mass())
        return self._velocity_mass

    @property
    def local_velocity_mass(self) -> np.ndarray:
        if not hasattr(self, "_local_velocity_mass"):
            local_mass = np.einsum("gp,cfp->cfg", self.assembler.space.q2_values, self.assembler.space.q2_weighted_values)
            object.__setattr__(self, "_local_velocity_mass", local_mass)
        return self._local_velocity_mass

    def residual_norm(self, residual: np.ndarray) -> float:
        active = self.active_dofs
        values = residual[active]
        pinned_positions = np.flatnonzero(active == self.pressure_pin_dof) if self.pin_pressure else np.asarray([])
        if pinned_positions.size:
            values = values.copy()
            values[int(pinned_positions[0])] = 0.0
        return float(np.linalg.norm(values))

    def apply_boundary_values(self, state: np.ndarray, z: float) -> None:
        arr = self._as_state(state)
        data = self.boundary_data
        v = self.num_velocity_dofs
        dofs = data.boundary_dofs
        arr[dofs] = data.base_x[dofs] + float(z) * data.control_x[dofs]
        arr[v + dofs] = data.base_y[dofs] + float(z) * data.control_y[dofs]

    def potential_flow_state(self) -> np.ndarray:
        data = self.boundary_data
        coords_v = data.coords
        x = coords_v[:, 0] - self.cylinder_center_x
        y = coords_v[:, 1] - self.cylinder_center_y
        rad = np.sqrt(x * x + y * y)
        arg = np.arctan2(y, x)
        ratio = (self.cylinder_radius / rad) ** 2
        radial = (1.0 - ratio) * np.cos(arg)
        angular = -(1.0 + ratio) * np.sin(arg)
        ux = radial * np.cos(arg) - angular * np.sin(arg)
        uy = radial * np.sin(arg) + angular * np.cos(arg)

        space = self.assembler.space
        coords_p = space.mesh.nodes
        xp = coords_p[:, 0] - self.cylinder_center_x
        yp = coords_p[:, 1] - self.cylinder_center_y
        rp = np.sqrt(xp * xp + yp * yp)
        argp = np.arctan2(yp, xp)
        ratiop = (self.cylinder_radius / rp) ** 2
        radialp = (1.0 - ratiop) * np.cos(argp)
        angularp = -(1.0 + ratiop) * np.sin(argp)
        vx = radialp * np.cos(argp) - angularp * np.sin(argp)
        vy = radialp * np.sin(argp) + angularp * np.cos(argp)
        pressure = 0.5 * (1.0 - (vx * vx + vy * vy))

        state = np.concatenate([ux, uy, pressure])
        return state

    def _as_state(self, state: np.ndarray) -> np.ndarray:
        arr = np.asarray(state, dtype=float)
        if arr.shape != (self.state_size,):
            raise ValueError(f"Expected state shape {(self.state_size,)}, got {arr.shape}")
        return arr

    def _apply_local_boundary_residual(self, local_res: np.ndarray, local_new: np.ndarray, z: float) -> None:
        data = self.boundary_data
        for entry in data.local_rows:
            global_dofs = self.assembler.space.velocity_dofs[entry.cell_row, entry.local_dofs]
            local_res[entry.cell_row, entry.local_dofs] = (
                local_new[entry.cell_row, entry.local_dofs]
                - data.base_x[global_dofs]
                - z * data.control_x[global_dofs]
            )
            local_res[entry.cell_row, 9 + entry.local_dofs] = (
                local_new[entry.cell_row, 9 + entry.local_dofs]
                - data.base_y[global_dofs]
                - z * data.control_y[global_dofs]
            )

    def _zero_local_boundary_rows(self, local_jac: np.ndarray) -> None:
        for entry in self.boundary_data.local_rows:
            local_jac[entry.cell_row, entry.local_dofs, :] = 0.0
            local_jac[entry.cell_row, 9 + entry.local_dofs, :] = 0.0

    def _identity_local_boundary_rows(self, local_jac: np.ndarray) -> None:
        self._zero_local_boundary_rows(local_jac)
        for entry in self.boundary_data.local_rows:
            local_jac[entry.cell_row, entry.local_dofs, entry.local_dofs] = 1.0
            rows = 9 + entry.local_dofs
            local_jac[entry.cell_row, rows, rows] = 1.0

    def _assemble_local_jacobian(self, local_jac: np.ndarray) -> sparse.csr_matrix:
        dofs = self.assembler.local_global_dofs
        rows = np.broadcast_to(dofs[:, :, None], local_jac.shape).ravel()
        cols = np.broadcast_to(dofs[:, None, :], local_jac.shape).ravel()
        return sparse.coo_matrix((local_jac.ravel(), (rows, cols)), shape=(self.state_size, self.state_size)).tocsr()

    def _active_newton_system(self, jac: sparse.csr_matrix, rhs: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
        active = self.active_dofs
        sub = jac[active][:, active].tolil()
        sub_rhs = rhs[active].copy()
        if not self.pin_pressure:
            return sub.tocsc(), sub_rhs, active
        pinned_positions = np.flatnonzero(active == self.pressure_pin_dof)
        if pinned_positions.size == 0:
            raise RuntimeError("Cannot pin pressure gauge: active FE system has no pressure dofs")
        pinned = int(pinned_positions[0])
        sub.rows[pinned] = [pinned]
        sub.data[pinned] = [1.0]
        sub_rhs[pinned] = 0.0
        return sub.tocsr(), sub_rhs, active

    @property
    def active_dofs(self) -> np.ndarray:
        if not hasattr(self, "_active_dofs"):
            data = self.boundary_data
            v = self.num_velocity_dofs
            active_set = set(
                int(i)
                for i in np.concatenate(
                    [
                        self.assembler.local_global_dofs.ravel(),
                        data.boundary_dofs,
                        v + data.boundary_dofs,
                    ]
                )
            )
            space = self.assembler.space
            num_nodes = space.mesh.num_nodes
            num_edges = len(space.edge_to_id)
            ordered: list[int] = []
            for i in range(num_nodes):
                ordered.extend(dof for dof in (i, v + i, 2 * v + i) if dof in active_set)
            for i in range(num_edges):
                base = num_nodes + i
                ordered.extend(dof for dof in (base, v + base) if dof in active_set)
            for cell_id in range(space.mesh.num_cells):
                base = num_nodes + num_edges + cell_id
                ordered.extend(dof for dof in (base, v + base) if dof in active_set)
            active = np.asarray(ordered, dtype=np.int64)
            if active.size != len(active_set):
                active = np.unique(
                    np.concatenate(
                        [
                            self.assembler.local_global_dofs.ravel(),
                            data.boundary_dofs,
                            v + data.boundary_dofs,
                        ]
                    )
                ).astype(np.int64)
            object.__setattr__(self, "_active_dofs", active)
        return self._active_dofs

    @property
    def pressure_pin_dof(self) -> int:
        return 2 * self.num_velocity_dofs + int(np.min(self.assembler.space.pressure_dofs))

    @property
    def boundary_data(self) -> "VelocityBoundaryData":
        if not hasattr(self, "_boundary_data"):
            object.__setattr__(self, "_boundary_data", VelocityBoundaryData.build(self))
        return self._boundary_data


@dataclass(frozen=True)
class LocalBoundaryRow:
    sideset: int
    cell_row: int
    local_side: int
    local_dofs: np.ndarray


@dataclass(frozen=True)
class VelocityBoundaryData:
    coords: np.ndarray
    boundary_dofs: np.ndarray
    cylinder_dofs: np.ndarray
    base_x: np.ndarray
    base_y: np.ndarray
    control_x: np.ndarray
    control_y: np.ndarray
    local_rows: tuple[LocalBoundaryRow, ...]

    @classmethod
    def build(cls, constraint: DynamicNavierStokesFEConstraint) -> "VelocityBoundaryData":
        space = constraint.assembler.space
        mesh = space.mesh
        coords = velocity_dof_coordinates(space)
        boundary: set[int] = set()
        cylinder: set[int] = set()
        base_x = np.zeros(space.num_velocity_dofs)
        base_y = np.zeros(space.num_velocity_dofs)
        control_x = np.zeros(space.num_velocity_dofs)
        control_y = np.zeros(space.num_velocity_dofs)
        local_rows: list[LocalBoundaryRow] = []

        for sideset in (0, 2, 3, 4):
            sideset_dofs: set[int] = set()
            for cell_row, local_side, local_dofs in _sideset_local_boundary_rows(space, sideset):
                side_dofs = space.velocity_dofs[cell_row, local_dofs]
                boundary.update(int(dof) for dof in side_dofs)
                sideset_dofs.update(int(dof) for dof in side_dofs)
                local_rows.append(
                    LocalBoundaryRow(
                        sideset=sideset,
                        cell_row=cell_row,
                        local_side=local_side,
                        local_dofs=local_dofs,
                    )
                )
            side_dofs = np.asarray(sorted(sideset_dofs), dtype=np.int64)
            if side_dofs.size == 0:
                continue
            y = coords[side_dofs, 1]
            if sideset == 4:
                cylinder.update(side_dofs)
                control_x[side_dofs] = (coords[side_dofs, 1] - constraint.cylinder_center_y) / constraint.cylinder_radius
                control_y[side_dofs] = (constraint.cylinder_center_x - coords[side_dofs, 0]) / constraint.cylinder_radius
            else:
                base_x[side_dofs] = (2.0 + y) * (2.0 - y) if constraint.use_parabolic_inflow else 1.0
                base_y[side_dofs] = 0.0
        return cls(
            coords=coords,
            boundary_dofs=np.asarray(sorted(boundary), dtype=np.int64),
            cylinder_dofs=np.asarray(sorted(cylinder), dtype=np.int64),
            base_x=base_x,
            base_y=base_y,
            control_x=control_x,
            control_y=control_y,
            local_rows=tuple(local_rows),
        )


def velocity_dof_coordinates(space) -> np.ndarray:
    mesh = space.mesh
    coords = np.zeros((space.num_velocity_dofs, 2))
    coords[: mesh.num_nodes] = mesh.nodes
    num_edges = len(space.edge_to_id)
    for (a, b), edge_id in space.edge_to_id.items():
        coords[mesh.num_nodes + edge_id] = 0.5 * (mesh.nodes[a] + mesh.nodes[b])
    centers = np.mean(mesh.nodes[mesh.cells], axis=1)
    coords[mesh.num_nodes + num_edges : mesh.num_nodes + num_edges + mesh.num_cells] = centers
    return coords


_LOCAL_SIDE_Q2_DOFS = (
    np.asarray([0, 1, 4], dtype=np.int64),
    np.asarray([1, 2, 5], dtype=np.int64),
    np.asarray([2, 3, 6], dtype=np.int64),
    np.asarray([3, 0, 7], dtype=np.int64),
)


def _sideset_local_boundary_rows(space, sideset: int) -> tuple[tuple[int, int, np.ndarray], ...]:
    cell_to_row = {int(cell_id): row for row, cell_id in enumerate(space.cell_ids)}
    rows: list[tuple[int, int, np.ndarray]] = []
    for local_side, cell_ids in enumerate(space.mesh.side_sets[sideset]):
        local_dofs = _LOCAL_SIDE_Q2_DOFS[local_side]
        for cell_id in cell_ids:
            cell_row = cell_to_row.get(int(cell_id))
            if cell_row is not None:
                rows.append((cell_row, local_side, local_dofs))
    return tuple(rows)
