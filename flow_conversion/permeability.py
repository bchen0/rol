from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FilteredDarcyConfig


@dataclass(frozen=True)
class Permeability:
    diffuser_radius: float
    viscosity: float
    min_perm: float
    max_perm: float
    parametrization_type: int

    @classmethod
    def from_config(cls, config: FilteredDarcyConfig) -> "Permeability":
        radius = config.diffuser_radius + (10.0 if config.use_darcy_flow else 0.0)
        return cls(
            diffuser_radius=radius,
            viscosity=config.dynamic_viscosity,
            min_perm=config.minimum_permeability,
            max_perm=config.maximum_permeability,
            parametrization_type=config.parametrization_type,
        )

    def compute(self, z: np.ndarray, points: np.ndarray, deriv: int = 0) -> np.ndarray:
        z_arr = np.asarray(z, dtype=float)
        mask = np.linalg.norm(points, axis=-1) <= self.diffuser_radius + np.sqrt(np.finfo(float).eps)
        if self.parametrization_type == 1:
            a = self.min_perm / self.viscosity
            b = np.log(self.max_perm / self.min_perm)
            base = a * np.exp(b * z_arr)
            if deriv == 0:
                out = base
            elif deriv == 1:
                out = b * base
            elif deriv == 2:
                out = b * b * base
            else:
                raise ValueError("deriv must be 0, 1, or 2")
        else:
            a = self.min_perm / self.viscosity
            b = self.max_perm / self.viscosity
            if deriv == 0:
                out = a + z_arr * (b - a)
            elif deriv == 1:
                out = np.full_like(z_arr, b - a)
            elif deriv == 2:
                out = np.zeros_like(z_arr)
            else:
                raise ValueError("deriv must be 0, 1, or 2")
        return np.where(mask, out, 0.0)
