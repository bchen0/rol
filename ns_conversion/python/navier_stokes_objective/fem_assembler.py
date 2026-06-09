from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .fe import TaylorHoodSpace
from .fem_pde import NavierStokesLocalResidual


@dataclass(frozen=True)
class TaylorHoodAssembler:
    """Global sparse assembler for the Q2/Q1 local Navier-Stokes residual."""

    space: TaylorHoodSpace
    pde: NavierStokesLocalResidual

    @classmethod
    def from_space(cls, space: TaylorHoodSpace, *, reynolds_number: float) -> "TaylorHoodAssembler":
        return cls(space=space, pde=NavierStokesLocalResidual(space, reynolds_number))

    @property
    def num_velocity_dofs(self) -> int:
        return self.space.num_velocity_dofs

    @property
    def num_pressure_dofs(self) -> int:
        return self.space.num_pressure_dofs

    @property
    def state_size(self) -> int:
        return 2 * self.num_velocity_dofs + self.num_pressure_dofs

    @property
    def local_global_dofs(self) -> np.ndarray:
        v = self.num_velocity_dofs
        return np.concatenate(
            [
                self.space.velocity_dofs,
                v + self.space.velocity_dofs,
                2 * v + self.space.pressure_dofs,
            ],
            axis=1,
        )

    def gather(self, state: np.ndarray) -> np.ndarray:
        arr = np.asarray(state, dtype=float)
        if arr.shape != (self.state_size,):
            raise ValueError(f"Expected state shape {(self.state_size,)}, got {arr.shape}")
        return arr[self.local_global_dofs]

    def assemble_residual(self, state: np.ndarray) -> np.ndarray:
        local = self.gather(state)
        local_residual = self.pde.residual(local)
        residual = np.zeros(self.state_size)
        np.add.at(residual, self.local_global_dofs.ravel(), local_residual.ravel())
        return residual

    def assemble_jacobian(self, state: np.ndarray) -> sparse.csr_matrix:
        local = self.gather(state)
        local_jac = self.pde.jacobian(local)
        dofs = self.local_global_dofs
        rows = np.broadcast_to(dofs[:, :, None], local_jac.shape).ravel()
        cols = np.broadcast_to(dofs[:, None, :], local_jac.shape).ravel()
        data = local_jac.ravel()
        return sparse.coo_matrix((data, (rows, cols)), shape=(self.state_size, self.state_size)).tocsr()

    def apply_jacobian(self, state: np.ndarray, direction: np.ndarray) -> np.ndarray:
        local = self.gather(state)
        local_dir = self.gather(direction)
        local_jv = self.pde.apply_jacobian(local, local_dir)
        out = np.zeros(self.state_size)
        np.add.at(out, self.local_global_dofs.ravel(), local_jv.ravel())
        return out

    def assemble_velocity_mass(self) -> sparse.csr_matrix:
        local_mass = np.einsum("gp,cfp->cfg", self.space.q2_values, self.space.q2_weighted_values)
        dofs = self.space.velocity_dofs
        rows = np.broadcast_to(dofs[:, :, None], local_mass.shape).ravel()
        cols = np.broadcast_to(dofs[:, None, :], local_mass.shape).ravel()
        data = local_mass.ravel()
        return sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(self.num_velocity_dofs, self.num_velocity_dofs),
        ).tocsr()
