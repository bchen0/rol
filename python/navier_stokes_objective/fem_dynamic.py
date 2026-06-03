from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .config import NavierStokesConfig
from .fem_assembler import TaylorHoodAssembler


@dataclass(frozen=True)
class DynamicNavierStokesFEConstraint:
    assembler: TaylorHoodAssembler
    dt: float
    theta: float
    use_parabolic_inflow: bool
    cylinder_center_x: float
    cylinder_center_y: float
    cylinder_radius: float

    @classmethod
    def from_config(cls, assembler: TaylorHoodAssembler, config: NavierStokesConfig) -> "DynamicNavierStokesFEConstraint":
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
        res = self.dt * ((1.0 - self.theta) * self.assembler.assemble_residual(old) + self.theta * self.assembler.assemble_residual(new))
        mass = self.velocity_mass
        v = self.num_velocity_dofs
        res[:v] += mass @ (new[:v] - old[:v])
        res[v : 2 * v] += mass @ (new[v : 2 * v] - old[v : 2 * v])
        self._overwrite_boundary_residual(res, new, float(z))
        return res

    def jacobian_uold(self, u_old: np.ndarray, u_new: np.ndarray, z: float) -> sparse.csr_matrix:
        del u_new, z
        old = self._as_state(u_old)
        jac = self.dt * (1.0 - self.theta) * self.assembler.assemble_jacobian(old)
        jac = (jac - self.velocity_mass_state_block).tolil()
        self._zero_boundary_rows(jac)
        return jac.tocsr()

    def jacobian_unew(self, u_old: np.ndarray, u_new: np.ndarray, z: float) -> sparse.csr_matrix:
        del u_old, z
        new = self._as_state(u_new)
        jac = self.dt * self.theta * self.assembler.assemble_jacobian(new)
        jac = (jac + self.velocity_mass_state_block).tolil()
        self._identity_boundary_rows(jac)
        return jac.tocsr()

    def jacobian_z(self, u_old: np.ndarray, u_new: np.ndarray, z: float) -> np.ndarray:
        del u_old, u_new, z
        out = np.zeros(self.state_size)
        data = self.boundary_data
        v = self.num_velocity_dofs
        out[data.cylinder_dofs] = -data.control_x[data.cylinder_dofs]
        out[v + data.cylinder_dofs] = -data.control_y[data.cylinder_dofs]
        return out

    def solve(
        self,
        u_old: np.ndarray,
        z: float,
        *,
        initial_guess: np.ndarray | None = None,
        absolute_tol: float = 1e-10,
        relative_tol: float = 1e-10,
        step_tol: float = 1e-11,
        max_iter: int = 20,
    ) -> np.ndarray:
        old = self._as_state(u_old)
        new = old.copy() if initial_guess is None else self._as_state(initial_guess).copy()
        self.apply_boundary_values(new, z)

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
            step_active = spla.spsolve(jac_active, rhs_active)
            step = np.zeros_like(new)
            step[active] = step_active
            step_norm = float(np.linalg.norm(step))
            if step_norm <= step_tol * (1.0 + float(np.linalg.norm(new))):
                return new

            alpha = 1.0
            accepted = False
            while alpha >= 1e-6:
                trial = new + alpha * step
                self.apply_boundary_values(trial, z)
                if self.residual_norm(self.residual(old, trial, z)) < norm:
                    new = trial
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                new = new + step
                self.apply_boundary_values(new, z)

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
        """Solve the pinned active linearization with respect to ``u_new``."""
        rhs_arr = self._as_state(rhs)
        jac = self.jacobian_unew(u_old, u_new, z)
        jac_active, rhs_active, active = self._active_newton_system(jac, rhs_arr)
        matrix = jac_active.T if transpose else jac_active
        sol_active = spla.spsolve(matrix, rhs_active)
        out = np.zeros_like(rhs_arr)
        out[active] = sol_active
        return out

    def solve_adjoint_unew(self, rhs: np.ndarray, u_old: np.ndarray, u_new: np.ndarray, z: float) -> np.ndarray:
        """Solve ``J_unew.T lambda = rhs`` on the pinned active FE system."""
        return self.solve_linearized_unew(rhs, u_old, u_new, z, transpose=True)

    @property
    def velocity_mass(self) -> sparse.csr_matrix:
        if not hasattr(self, "_velocity_mass"):
            object.__setattr__(self, "_velocity_mass", self.assembler.assemble_velocity_mass())
        return self._velocity_mass

    @property
    def velocity_mass_state_block(self) -> sparse.csr_matrix:
        if not hasattr(self, "_velocity_mass_state_block"):
            pressure_zero = sparse.csr_matrix((self.assembler.num_pressure_dofs, self.assembler.num_pressure_dofs))
            block = sparse.block_diag(
                (self.velocity_mass, self.velocity_mass, pressure_zero),
                format="csr",
            )
            object.__setattr__(self, "_velocity_mass_state_block", block)
        return self._velocity_mass_state_block

    def residual_norm(self, residual: np.ndarray) -> float:
        active = self.active_dofs
        values = residual[active]
        pinned_positions = np.flatnonzero(active == self.pressure_pin_dof)
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
        rad = np.maximum(rad, self.cylinder_radius * (1.0 + 1e-8))
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
        rp = np.maximum(rp, self.cylinder_radius * (1.0 + 1e-8))
        argp = np.arctan2(yp, xp)
        ratiop = (self.cylinder_radius / rp) ** 2
        radialp = (1.0 - ratiop) * np.cos(argp)
        angularp = -(1.0 + ratiop) * np.sin(argp)
        vx = radialp * np.cos(argp) - angularp * np.sin(argp)
        vy = radialp * np.sin(argp) + angularp * np.cos(argp)
        pressure = 0.5 * (1.0 - (vx * vx + vy * vy))

        state = np.concatenate([ux, uy, pressure])
        self.apply_boundary_values(state, 0.0)
        return state

    def _as_state(self, state: np.ndarray) -> np.ndarray:
        arr = np.asarray(state, dtype=float)
        if arr.shape != (self.state_size,):
            raise ValueError(f"Expected state shape {(self.state_size,)}, got {arr.shape}")
        return arr

    def _overwrite_boundary_residual(self, residual: np.ndarray, new: np.ndarray, z: float) -> None:
        data = self.boundary_data
        v = self.num_velocity_dofs
        dofs = data.boundary_dofs
        residual[dofs] = new[dofs] - data.base_x[dofs] - z * data.control_x[dofs]
        residual[v + dofs] = new[v + dofs] - data.base_y[dofs] - z * data.control_y[dofs]

    def _zero_boundary_rows(self, mat: sparse.lil_matrix) -> None:
        data = self.boundary_data
        v = self.num_velocity_dofs
        for row in np.concatenate([data.boundary_dofs, v + data.boundary_dofs]):
            mat.rows[int(row)] = []
            mat.data[int(row)] = []

    def _identity_boundary_rows(self, mat: sparse.lil_matrix) -> None:
        data = self.boundary_data
        v = self.num_velocity_dofs
        for row in np.concatenate([data.boundary_dofs, v + data.boundary_dofs]):
            idx = int(row)
            mat.rows[idx] = [idx]
            mat.data[idx] = [1.0]

    def _active_newton_system(self, jac: sparse.csr_matrix, rhs: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
        active = self.active_dofs
        sub = jac[active][:, active].tolil()
        sub_rhs = rhs[active].copy()
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
class VelocityBoundaryData:
    coords: np.ndarray
    boundary_dofs: np.ndarray
    cylinder_dofs: np.ndarray
    base_x: np.ndarray
    base_y: np.ndarray
    control_x: np.ndarray
    control_y: np.ndarray

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

        active_velocity_dofs = set(int(i) for i in space.velocity_dofs.ravel())
        for sideset in (0, 2, 3, 4):
            side_dofs = _sideset_velocity_dofs(space, sideset, active_velocity_dofs)
            if side_dofs.size == 0:
                continue
            boundary.update(side_dofs)
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


def _sideset_velocity_dofs(space, sideset: int, active_velocity_dofs: set[int]) -> np.ndarray:
    mesh = space.mesh
    dofs: set[int] = set()
    for a, b in mesh.side_edges[sideset]:
        if int(a) in active_velocity_dofs:
            dofs.add(int(a))
        if int(b) in active_velocity_dofs:
            dofs.add(int(b))
        lo, hi = (int(a), int(b)) if a < b else (int(b), int(a))
        edge_dof = mesh.num_nodes + space.edge_to_id[(lo, hi)]
        if edge_dof in active_velocity_dofs:
            dofs.add(edge_dof)
    return np.asarray(sorted(dofs), dtype=np.int64)
