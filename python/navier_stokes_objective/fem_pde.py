from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fe import TaylorHoodSpace


@dataclass(frozen=True)
class NavierStokesLocalResidual:
    space: TaylorHoodSpace
    reynolds_number: float

    @property
    def viscosity(self) -> float:
        return 1.0 / self.reynolds_number

    def residual(self, local_state: np.ndarray) -> np.ndarray:
        ux, uy, pressure = self.space.split_local_state(local_state)
        rx, ry, rp = self.residual_components(ux, uy, pressure)
        return self.space.combine_local_state(rx, ry, rp)

    def jacobian(self, local_state: np.ndarray) -> np.ndarray:
        ux, uy, pressure = self.space.split_local_state(local_state)
        return self.jacobian_components(ux, uy, pressure)

    def apply_jacobian(self, local_state: np.ndarray, direction: np.ndarray) -> np.ndarray:
        jac = self.jacobian(local_state)
        return np.einsum("cij,cj->ci", jac, direction)

    def residual_components(
        self,
        ux_coeff: np.ndarray,
        uy_coeff: np.ndarray,
        pressure_coeff: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sp = self.space
        ux_val = sp.evaluate_value(ux_coeff, "q2")
        uy_val = sp.evaluate_value(uy_coeff, "q2")
        pressure_val = sp.evaluate_value(pressure_coeff, "q1")
        grad_ux = sp.evaluate_gradient(ux_coeff, "q2")
        grad_uy = sp.evaluate_gradient(uy_coeff, "q2")

        conv_x = ux_val * grad_ux[:, :, 0] + uy_val * grad_ux[:, :, 1]
        conv_y = ux_val * grad_uy[:, :, 0] + uy_val * grad_uy[:, :, 1]
        div_u = grad_ux[:, :, 0] + grad_uy[:, :, 1]

        rx = self.viscosity * np.einsum("cpd,cfpd->cf", grad_ux, sp.q2_weighted_grads)
        rx += np.einsum("cp,cfp->cf", conv_x, sp.q2_weighted_values)
        rx -= np.einsum("cp,cfp->cf", pressure_val, sp.q2_weighted_grads[:, :, :, 0])

        ry = self.viscosity * np.einsum("cpd,cfpd->cf", grad_uy, sp.q2_weighted_grads)
        ry += np.einsum("cp,cfp->cf", conv_y, sp.q2_weighted_values)
        ry -= np.einsum("cp,cfp->cf", pressure_val, sp.q2_weighted_grads[:, :, :, 1])

        rp = -np.einsum("cp,cfp->cf", div_u, sp.q1_weighted_values)
        return rx, ry, rp

    def jacobian_components(
        self,
        ux_coeff: np.ndarray,
        uy_coeff: np.ndarray,
        pressure_coeff: np.ndarray,
    ) -> np.ndarray:
        del pressure_coeff
        sp = self.space
        c = sp.num_cells
        jac = np.zeros((c, sp.local_state_size, sp.local_state_size))

        ux_val = sp.evaluate_value(ux_coeff, "q2")
        uy_val = sp.evaluate_value(uy_coeff, "q2")
        grad_ux = sp.evaluate_gradient(ux_coeff, "q2")
        grad_uy = sp.evaluate_gradient(uy_coeff, "q2")

        q2 = sp.q2_values
        q1 = sp.q1_values
        grad_q2 = sp.q2_grads
        wq2 = sp.q2_weighted_values
        wq1 = sp.q1_weighted_values
        wgrad_q2 = sp.q2_weighted_grads

        diffusion = self.viscosity * np.einsum("cfpd,cgpd->cfg", grad_q2, wgrad_q2)
        adv_trial = ux_val[:, None, :] * grad_q2[:, :, :, 0] + uy_val[:, None, :] * grad_q2[:, :, :, 1]
        adv = np.einsum("cgp,cfp->cfg", adv_trial, wq2)

        d_ux_dx = np.einsum("gp,cp,cfp->cfg", q2, grad_ux[:, :, 0], wq2)
        d_ux_dy = np.einsum("gp,cp,cfp->cfg", q2, grad_ux[:, :, 1], wq2)
        d_uy_dx = np.einsum("gp,cp,cfp->cfg", q2, grad_uy[:, :, 0], wq2)
        d_uy_dy = np.einsum("gp,cp,cfp->cfg", q2, grad_uy[:, :, 1], wq2)

        jac[:, 0:9, 0:9] = diffusion + adv + d_ux_dx
        jac[:, 0:9, 9:18] = d_ux_dy
        jac[:, 9:18, 0:9] = d_uy_dx
        jac[:, 9:18, 9:18] = diffusion + adv + d_uy_dy

        jac[:, 0:9, 18:22] = -np.einsum("cfp,gp->cfg", wgrad_q2[:, :, :, 0], q1)
        jac[:, 9:18, 18:22] = -np.einsum("cfp,gp->cfg", wgrad_q2[:, :, :, 1], q1)
        jac[:, 18:22, 0:9] = -np.einsum("cfp,cgp->cfg", wq1, grad_q2[:, :, :, 0])
        jac[:, 18:22, 9:18] = -np.einsum("cfp,cgp->cfg", wq1, grad_q2[:, :, :, 1])
        return jac
