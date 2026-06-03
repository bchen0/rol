from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .config import NavierStokesConfig, read_config
from .fe import TaylorHoodSpace
from .fem_assembler import TaylorHoodAssembler
from .fem_dynamic import DynamicNavierStokesFEConstraint
from .fem_qoi import NavierStokesFEObjective
from .mesh import ChannelMesh, read_channel_mesh


@dataclass
class _StateCache:
    key: bytes
    states: np.ndarray


class NavierStokesReducedObjective:
    """Reduced dynamic objective for the ROL channel-flow example.

    This class mirrors the object constructed by
    ``rol/example/PDE-OPT/dynamic/navier-stokes/objective.cpp``.  It parses the
    same XML, uses the same scalar parametric rotation-control convention, and
    follows ROL's dynamic-control indexing: control block 0 is unused and
    intervals 1 through ``nt - 1`` are evaluated.

    By default, velocity is represented on the channel mesh vertices and
    advanced by an implicit graph diffusion model with Dirichlet wall/cylinder
    controls.  Passing ``backend="fe"`` switches to the Q2/Q1 Taylor-Hood
    finite-element backend implemented in this package.
    """

    def __init__(
        self,
        config: NavierStokesConfig,
        mesh: ChannelMesh,
        *,
        backend: str = "graph",
        fe_cell_ids: Any | None = None,
        cubature_degree: int = 4,
        newton_absolute_tol: float = 1e-10,
        newton_relative_tol: float = 1e-10,
        newton_step_tol: float = 1e-11,
        newton_max_iter: int = 20,
        hess_vec_difference_step: float = 1e-5,
    ) -> None:
        if not config.use_parametric_control:
            raise NotImplementedError("Only the default scalar parametric-control branch is implemented")
        if config.use_nonpenetrating_walls:
            raise NotImplementedError("Use Non-Penetrating Walls=true is not implemented in the Python model")
        if config.nt < 2:
            raise ValueError("Number of time steps must be at least 2")

        self.backend = backend.lower()
        self.config = config
        self.mesh = mesh
        self.hess_vec_difference_step = float(hess_vec_difference_step)
        self.num_controls = config.nt
        self.control_shape = (config.nt,)
        self.newton_absolute_tol = float(newton_absolute_tol)
        self.newton_relative_tol = float(newton_relative_tol)
        self.newton_step_tol = float(newton_step_tol)
        self.newton_max_iter = int(newton_max_iter)

        self._state_cache: _StateCache | None = None
        if self.backend == "graph":
            self.state_size = 2 * mesh.num_nodes
            self._build_boundary_data()
            self._build_dynamics()
            self._initial_state = self._build_initial_state()
        elif self.backend == "fe":
            self._build_fe_backend(fe_cell_ids, int(cubature_degree))
        else:
            raise ValueError(f"Unknown backend {backend!r}; expected 'graph' or 'fe'")

    @classmethod
    def from_xml(
        cls,
        xml_path: str | Path,
        mesh_path: str | Path | None = None,
        initial_condition_path: str | Path | None = None,
        **kwargs: Any,
    ) -> "NavierStokesReducedObjective":
        """Create the Python objective from the ROL XML input file.

        Extra keyword arguments ``time_steps``, ``end_time``, and ``theta`` are
        accepted for small derivative checks without editing the XML.  Pass
        ``backend="fe"`` and, optionally, ``fe_cell_ids`` to use the
        Taylor-Hood finite-element backend.  The ``initial_condition_path``
        argument is accepted for API parity; both backends currently use the
        same potential-flow fallback as their initial state.
        """
        config_keys = {"time_steps", "end_time", "theta"}
        config_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in config_keys}
        init_keys = {
            "backend",
            "fe_cell_ids",
            "cubature_degree",
            "newton_absolute_tol",
            "newton_relative_tol",
            "newton_step_tol",
            "newton_max_iter",
            "hess_vec_difference_step",
        }
        init_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in init_keys}
        config = read_config(xml_path, **config_kwargs)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")

        mesh_file = Path(mesh_path).expanduser() if mesh_path is not None else Path(config.mesh_file)
        if not mesh_file.is_absolute():
            mesh_file = config.xml_path.parent / mesh_file
        mesh = read_channel_mesh(mesh_file)
        return cls(config, mesh, **init_kwargs)

    def value(self, z: Any) -> float:
        controls = self._as_control(z)
        if self.backend == "fe":
            return self._value_fe(controls)
        states = self.solve_state(controls)
        total = 0.0
        dt = self.config.dt
        theta = self.config.theta
        for k in range(1, self.config.nt):
            u_old = states[k - 1]
            u_new = states[k]
            zk = controls[k]
            total += dt * (
                (1.0 - theta) * self._integrated_value(u_old, zk)
                + theta * self._integrated_value(u_new, zk)
            )
            if k == self.config.nt - 1:
                total += self.config.final_time_state_cost * self._state_value(
                    u_new, self.config.final_time_objective_type
                )
        return float(total)

    def gradient(self, z: Any) -> np.ndarray:
        controls = self._as_control(z)
        if self.backend == "fe":
            return self._gradient_fe(controls)
        states = self.solve_state(controls)
        grad = np.zeros_like(controls)
        adj = np.zeros(self.state_size)
        dt = self.config.dt
        theta = self.config.theta

        for k in range(self.config.nt - 1, 0, -1):
            u_old = states[k - 1]
            u_new = states[k]
            zk = controls[k]

            grad_u_old = dt * (1.0 - theta) * self._integrated_grad(u_old)
            grad_u_new = dt * theta * self._integrated_grad(u_new)
            grad_z = dt * self.config.control_cost * zk
            if k == self.config.nt - 1:
                grad_u_new = grad_u_new + self.config.final_time_state_cost * self._state_grad(
                    u_new, self.config.final_time_objective_type
                )

            mu = adj + grad_u_new
            grad[k] = grad_z + self._transition_z_transpose(mu)
            adj = grad_u_old + self._transition_uold_transpose(mu)

        grad[0] = 0.0
        return grad

    def hess_vec(self, z: Any, v: Any) -> np.ndarray:
        controls = self._as_control(z)
        direction = self._as_control(v)
        norm_v = float(np.linalg.norm(direction))
        if norm_v == 0.0:
            return np.zeros_like(controls)
        step = self.hess_vec_difference_step * (1.0 + float(np.linalg.norm(controls))) / norm_v
        hv = (self.gradient(controls + step * direction) - self.gradient(controls - step * direction)) / (2.0 * step)
        hv[0] = 0.0
        return hv

    def solve_state(self, z: Any) -> np.ndarray:
        controls = self._as_control(z)
        key = controls.tobytes()
        if self._state_cache is not None and self._state_cache.key == key:
            return self._state_cache.states.copy()

        states = np.zeros((self.config.nt, self.state_size))
        states[0] = self._initial_state
        if self.backend == "fe":
            for k in range(1, self.config.nt):
                initial_guess = states[k - 1].copy()
                self.fe_constraint.apply_boundary_values(initial_guess, controls[k])
                states[k] = self.fe_constraint.solve(
                    states[k - 1],
                    controls[k],
                    initial_guess=initial_guess,
                    absolute_tol=self.newton_absolute_tol,
                    relative_tol=self.newton_relative_tol,
                    step_tol=self.newton_step_tol,
                    max_iter=self.newton_max_iter,
                )
        else:
            for k in range(1, self.config.nt):
                states[k] = self._advance_state(states[k - 1], controls[k])

        self._state_cache = _StateCache(key=key, states=states.copy())
        return states

    def initial_control(self) -> np.ndarray:
        """Return the sinusoidal scalar-control initial guess from the XML."""
        guess = self.config.raw.get("Problem", {}).get("Initial Guess", {})
        amp = float(guess.get("Amplitude", 6.0))
        st = float(guess.get("Strouhal Number", 0.74))
        phase = float(guess.get("Phase Shift", 0.0))
        z = np.zeros(self.config.nt)
        for k in range(1, self.config.nt):
            t_mid = (k + 0.5) * self.config.dt
            z[k] = amp * np.sin(2.0 * np.pi * st * t_mid + phase)
        return z

    def _as_control(self, z: Any) -> np.ndarray:
        arr = np.asarray(z, dtype=float)
        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        elif arr.ndim != 1:
            raise ValueError(f"Expected control shape ({self.config.nt},) or ({self.config.nt}, 1); got {arr.shape}")
        if arr.size != self.config.nt:
            raise ValueError(f"Expected {self.config.nt} control blocks with block 0 unused; got {arr.size}")
        return np.array(arr, dtype=float, copy=True)

    def _build_fe_backend(self, fe_cell_ids: Any | None, cubature_degree: int) -> None:
        cell_ids = None if fe_cell_ids is None else np.asarray(fe_cell_ids, dtype=np.int64)
        if cell_ids is not None and cell_ids.size == 0:
            raise ValueError("FE backend requires at least one selected cell")
        self.fe_space = TaylorHoodSpace.from_mesh(self.mesh, cell_ids=cell_ids, cubature_degree=cubature_degree)
        self.fe_assembler = TaylorHoodAssembler.from_space(
            self.fe_space,
            reynolds_number=self.config.reynolds_number,
        )
        self.fe_constraint = DynamicNavierStokesFEConstraint.from_config(self.fe_assembler, self.config)
        self.fe_objective = NavierStokesFEObjective(self.fe_assembler, self.config)
        self.state_size = self.fe_assembler.state_size
        self._initial_state = self.fe_constraint.potential_flow_state()

    def _value_fe(self, controls: np.ndarray) -> float:
        states = self.solve_state(controls)
        total = 0.0
        dt = self.config.dt
        theta = self.config.theta
        for k in range(1, self.config.nt):
            u_old = states[k - 1]
            u_new = states[k]
            zk = controls[k]
            total += dt * (
                (1.0 - theta) * self.fe_objective.integrated_value(u_old, zk)
                + theta * self.fe_objective.integrated_value(u_new, zk)
            )
            if k == self.config.nt - 1:
                total += self.fe_objective.final_value(u_new)
        return float(total)

    def _gradient_fe(self, controls: np.ndarray) -> np.ndarray:
        states = self.solve_state(controls)
        grad = np.zeros_like(controls)
        adj = np.zeros(self.state_size)
        dt = self.config.dt
        theta = self.config.theta

        for k in range(self.config.nt - 1, 0, -1):
            u_old = states[k - 1]
            u_new = states[k]
            zk = controls[k]

            grad_u_old = dt * (1.0 - theta) * self.fe_objective.integrated_state_gradient(u_old)
            grad_u_new = dt * theta * self.fe_objective.integrated_state_gradient(u_new)
            grad_z = dt * self.fe_objective.integrated_control_gradient(zk)
            if k == self.config.nt - 1:
                grad_u_new = grad_u_new + self.fe_objective.final_state_gradient(u_new)

            lam = self.fe_constraint.solve_adjoint_unew(adj + grad_u_new, u_old, u_new, zk)
            grad[k] = grad_z - float(np.dot(self.fe_constraint.jacobian_z(u_old, u_new, zk), lam))
            adj = grad_u_old - self.fe_constraint.jacobian_uold(u_old, u_new, zk).T @ lam

        grad[0] = 0.0
        return grad

    def _build_boundary_data(self) -> None:
        n = self.mesh.num_nodes
        fixed_nodes: set[int] = set()
        for sideset in (0, 2, 3, 4):
            fixed_nodes.update(int(i) for i in self.mesh.sideset_nodes(sideset))
        self.boundary_nodes = np.asarray(sorted(fixed_nodes), dtype=np.int64)
        is_boundary = np.zeros(n, dtype=bool)
        is_boundary[self.boundary_nodes] = True
        self.free_nodes = np.flatnonzero(~is_boundary).astype(np.int64)

        base_x = np.zeros(n)
        base_y = np.zeros(n)
        control_x = np.zeros(n)
        control_y = np.zeros(n)
        coords = self.mesh.nodes

        for sideset in (0, 2, 3):
            nodes = self.mesh.sideset_nodes(sideset)
            y = coords[nodes, 1]
            base_x[nodes] = (2.0 + y) * (2.0 - y) if self.config.use_parabolic_inflow else 1.0
            base_y[nodes] = 0.0

        cyl = self.mesh.sideset_nodes(4)
        x = coords[cyl, 0]
        y = coords[cyl, 1]
        r = self.config.cylinder_radius
        control_x[cyl] = (y - self.config.cylinder_center_y) / r
        control_y[cyl] = (self.config.cylinder_center_x - x) / r
        base_x[cyl] = 0.0
        base_y[cyl] = 0.0

        self.base_boundary_x = base_x
        self.base_boundary_y = base_y
        self.control_boundary_x = control_x
        self.control_boundary_y = control_y
        self.downstream_weights = self.mesh.sideset_node_weights(1)
        self.mass = np.maximum(self.mesh.node_mass, np.finfo(float).eps)
        self.target_x = (2.0 + coords[:, 1]) * (2.0 - coords[:, 1]) if self.config.use_parabolic_inflow else np.ones(n)
        self.target_y = np.zeros(n)

    def _build_dynamics(self) -> None:
        f = self.free_nodes
        b = self.boundary_nodes
        K = self.mesh.stiffness.tocsr()
        self.K = K
        self.K_ff = K[f][:, f].tocsr()
        self.K_fb = K[f][:, b].tocsr()
        self.K_f_all = K[f, :].tocsr()
        self.M_f = self.mass[f]
        A = sparse.diags(self.M_f) + self.config.dt * self.config.theta * self.config.viscosity * self.K_ff
        self._solve_A = spla.factorized(A.tocsc())
        self._solve_AT = spla.factorized(A.T.tocsc())

        bx = self.control_boundary_x[b]
        by = self.control_boundary_y[b]
        scale = -self.config.dt * self.config.theta * self.config.viscosity
        self._dz_rhs_x = scale * (self.K_fb @ bx)
        self._dz_rhs_y = scale * (self.K_fb @ by)
        self._dz_free_x = self._solve_A(self._dz_rhs_x)
        self._dz_free_y = self._solve_A(self._dz_rhs_y)

    def _build_initial_state(self) -> np.ndarray:
        x = self.mesh.nodes[:, 0] - self.config.cylinder_center_x
        y = self.mesh.nodes[:, 1] - self.config.cylinder_center_y
        r = np.sqrt(x * x + y * y)
        r = np.maximum(r, self.config.cylinder_radius * (1.0 + 1e-8))
        arg = np.arctan2(y, x)
        ratio = (self.config.cylinder_radius / r) ** 2
        radial = (1.0 - ratio) * np.cos(arg)
        angular = -(1.0 + ratio) * np.sin(arg)
        ux = radial * np.cos(arg) - angular * np.sin(arg)
        uy = radial * np.sin(arg) + angular * np.cos(arg)
        full = self._compose_state(ux, uy)
        full[self.boundary_nodes] = self.base_boundary_x[self.boundary_nodes]
        full[self.mesh.num_nodes + self.boundary_nodes] = self.base_boundary_y[self.boundary_nodes]
        return full

    def _advance_state(self, u_old: np.ndarray, z_k: float) -> np.ndarray:
        n = self.mesh.num_nodes
        ux_old = u_old[:n]
        uy_old = u_old[n:]
        bx = self.base_boundary_x + z_k * self.control_boundary_x
        by = self.base_boundary_y + z_k * self.control_boundary_y
        ux_new = np.empty(n)
        uy_new = np.empty(n)
        ux_new[self.boundary_nodes] = bx[self.boundary_nodes]
        uy_new[self.boundary_nodes] = by[self.boundary_nodes]
        ux_new[self.free_nodes] = self._advance_component(ux_old, bx)
        uy_new[self.free_nodes] = self._advance_component(uy_old, by)
        return self._compose_state(ux_new, uy_new)

    def _advance_component(self, old: np.ndarray, boundary_values: np.ndarray) -> np.ndarray:
        f = self.free_nodes
        b = self.boundary_nodes
        rhs = self.M_f * old[f]
        if self.config.theta != 1.0:
            rhs -= self.config.dt * (1.0 - self.config.theta) * self.config.viscosity * (self.K_f_all @ old)
        rhs -= self.config.dt * self.config.theta * self.config.viscosity * (self.K_fb @ boundary_values[b])
        return self._solve_A(rhs)

    def _transition_uold_transpose(self, q: np.ndarray) -> np.ndarray:
        n = self.mesh.num_nodes
        out_x = self._transition_uold_transpose_component(q[:n])
        out_y = self._transition_uold_transpose_component(q[n:])
        return self._compose_state(out_x, out_y)

    def _transition_uold_transpose_component(self, q: np.ndarray) -> np.ndarray:
        y = self._solve_AT(q[self.free_nodes])
        out = np.zeros(self.mesh.num_nodes)
        out[self.free_nodes] += self.M_f * y
        if self.config.theta != 1.0:
            out -= self.config.dt * (1.0 - self.config.theta) * self.config.viscosity * (self.K_f_all.T @ y)
        return out

    def _transition_z_transpose(self, q: np.ndarray) -> float:
        n = self.mesh.num_nodes
        b = self.boundary_nodes
        qx = q[:n]
        qy = q[n:]
        yx = self._solve_AT(qx[self.free_nodes])
        yy = self._solve_AT(qy[self.free_nodes])
        val = float(np.dot(qx[b], self.control_boundary_x[b]) + np.dot(qy[b], self.control_boundary_y[b]))
        val += float(np.dot(yx, self._dz_rhs_x) + np.dot(yy, self._dz_rhs_y))
        return val

    def _integrated_value(self, u: np.ndarray, z_k: float) -> float:
        val = self.config.state_cost * self._state_value(u, self.config.integrated_objective_type)
        val += self.config.state_boundary_cost * self._downstream_power_value(u)
        val += 0.5 * self.config.control_cost * z_k * z_k
        return float(val)

    def _integrated_grad(self, u: np.ndarray) -> np.ndarray:
        grad = self.config.state_cost * self._state_grad(u, self.config.integrated_objective_type)
        grad = grad + self.config.state_boundary_cost * self._downstream_power_grad(u)
        return grad

    def _state_value(self, u: np.ndarray, objective_type: str) -> float:
        n = self.mesh.num_nodes
        ux = u[:n]
        uy = u[n:]
        if objective_type == "Dissipation":
            return 0.5 * self.config.viscosity * (float(ux @ (self.K @ ux)) + float(uy @ (self.K @ uy)))
        if objective_type == "Tracking":
            dx = ux - self.target_x
            dy = uy - self.target_y
            return 0.5 * float(np.dot(self.mass, dx * dx + dy * dy))
        raise NotImplementedError(f"Objective type {objective_type!r} is not implemented in the Python graph model")

    def _state_grad(self, u: np.ndarray, objective_type: str) -> np.ndarray:
        n = self.mesh.num_nodes
        ux = u[:n]
        uy = u[n:]
        if objective_type == "Dissipation":
            return self._compose_state(self.config.viscosity * (self.K @ ux), self.config.viscosity * (self.K @ uy))
        if objective_type == "Tracking":
            return self._compose_state(self.mass * (ux - self.target_x), self.mass * (uy - self.target_y))
        raise NotImplementedError(f"Objective type {objective_type!r} is not implemented in the Python graph model")

    def _downstream_power_value(self, u: np.ndarray) -> float:
        n = self.mesh.num_nodes
        ux = u[:n]
        uy = u[n:]
        dx = ux - 1.0
        density = 0.5 * (dx * dx + uy * uy) * ux
        return float(np.dot(self.downstream_weights, density))

    def _downstream_power_grad(self, u: np.ndarray) -> np.ndarray:
        n = self.mesh.num_nodes
        ux = u[:n]
        uy = u[n:]
        dx = ux - 1.0
        gx = self.downstream_weights * (dx * ux + 0.5 * (dx * dx + uy * uy))
        gy = self.downstream_weights * ux * uy
        return self._compose_state(gx, gy)

    def _compose_state(self, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(ux, dtype=float), np.asarray(uy, dtype=float)])
