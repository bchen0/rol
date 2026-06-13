from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import NavierStokesConfig, read_config


class L1DynamicObjective:
    """Nonsmooth control objective used as ``nobj`` in ``example_02.cpp``.

    This mirrors ``L1_Dyn_Objective`` from ``l1penaltydynamic.hpp``: the value is
    a beta-weighted trapezoidal-in-time L1 norm plus an indicator for the control
    bounds, and ``prox`` is soft-thresholding followed by projection onto those
    bounds.
    """

    def __init__(
        self,
        nt: int,
        end_time: float,
        *,
        theta: float = 1.0,
        beta: float = 1e-2,
        lower_bound: float | np.ndarray = -1.0,
        upper_bound: float | np.ndarray = 1.0,
        control_dimension: int = 1,
        time_step_widths: np.ndarray | None = None,
    ) -> None:
        self.nt = int(nt)
        if self.nt < 2:
            raise ValueError("Number of time steps must be at least 2")
        self.end_time = float(end_time)
        self.theta = float(theta)
        self.beta = float(beta)
        self.control_dimension = int(control_dimension)
        if self.control_dimension < 1:
            raise ValueError("control_dimension must be positive")

        if time_step_widths is None:
            widths = np.full(self.nt, self.end_time / float(self.nt), dtype=float)
        else:
            widths = np.asarray(time_step_widths, dtype=float)
            if widths.shape != (self.nt,):
                raise ValueError(f"Expected time_step_widths shape {(self.nt,)}, got {widths.shape}")
        if np.any(widths <= 0.0):
            raise ValueError("time-step widths must be positive")
        self.time_step_widths = widths

        self.lower_bound = np.asarray(lower_bound, dtype=float)
        self.upper_bound = np.asarray(upper_bound, dtype=float)
        lb = self._broadcast_bound(self.lower_bound, "lower_bound")
        ub = self._broadcast_bound(self.upper_bound, "upper_bound")
        if np.any(lb > ub):
            raise ValueError("lower_bound must be less than or equal to upper_bound")

    @classmethod
    def from_xml(
        cls,
        xml_path: str | Path,
        *,
        time_steps: int | None = None,
        end_time: float | None = None,
        theta: float | None = None,
        l1_control_cost: float | None = None,
        lower_bound: float | np.ndarray | None = None,
        upper_bound: float | np.ndarray | None = None,
        control_dimension: int = 1,
    ) -> L1DynamicObjective:
        """Construct the nonsmooth objective from the ROL XML input.

        The C++ class reads ``Theta`` from
        ``Reduced Dynamic Objective / Time Discretization`` and falls back to
        ``1.0`` when that nested list is absent.  ``input_02.xml`` is in that
        fallback case, so this method deliberately mirrors it instead of using
        the top-level time-discretization theta unless ``theta`` is supplied.
        """

        config = read_config(xml_path, time_steps=time_steps, end_time=end_time)
        l1_theta = float(theta) if theta is not None else _reduced_dynamic_theta(config)
        return cls.from_config(
            config,
            theta=l1_theta,
            beta=_l1_control_cost(config) if l1_control_cost is None else float(l1_control_cost),
            lower_bound=_lower_control_bound(config) if lower_bound is None else lower_bound,
            upper_bound=_upper_control_bound(config) if upper_bound is None else upper_bound,
            control_dimension=control_dimension,
        )

    @classmethod
    def from_config(
        cls,
        config: NavierStokesConfig,
        *,
        theta: float | None = None,
        beta: float | None = None,
        lower_bound: float | np.ndarray | None = None,
        upper_bound: float | np.ndarray | None = None,
        control_dimension: int = 1,
    ) -> L1DynamicObjective:
        return cls(
            config.nt,
            config.end_time,
            theta=_reduced_dynamic_theta(config) if theta is None else float(theta),
            beta=_l1_control_cost(config) if beta is None else float(beta),
            lower_bound=_lower_control_bound(config) if lower_bound is None else lower_bound,
            upper_bound=_upper_control_bound(config) if upper_bound is None else upper_bound,
            control_dimension=control_dimension,
        )

    @property
    def control_shape(self) -> tuple[int, ...]:
        if self.control_dimension == 1:
            return (self.nt,)
        return (self.nt, self.control_dimension)

    def value(self, z: Any) -> float:
        controls, _ = self._as_control(z)
        lower = self._broadcast_bound(self.lower_bound, "lower_bound")
        upper = self._broadcast_bound(self.upper_bound, "upper_bound")
        if np.any(controls < lower) or np.any(controls > upper):
            return float("inf")
        return float(np.sum(self._l1_weights()[:, None] * np.abs(controls)))

    def prox(self, z: Any, step: float) -> np.ndarray:
        """Return ``prox_{step * nobj}(z)``.

        This is the same operation as the C++ method: componentwise
        soft-thresholding with the time-dependent L1 parameter, then clipping to
        the lower and upper control bounds.
        """

        step = float(step)
        if step < 0.0:
            raise ValueError("step must be nonnegative")
        controls, squeezed = self._as_control(z)
        threshold = step * self._l1_weights()[:, None]
        prox = np.sign(controls) * np.maximum(np.abs(controls) - threshold, 0.0)
        prox = np.minimum(
            self._broadcast_bound(self.upper_bound, "upper_bound"),
            np.maximum(self._broadcast_bound(self.lower_bound, "lower_bound"), prox),
        )
        return prox[:, 0] if squeezed else prox

    def _as_control(self, z: Any) -> tuple[np.ndarray, bool]:
        arr = np.asarray(z, dtype=float)
        if self.control_dimension == 1 and arr.shape == (self.nt,):
            return arr.reshape(self.nt, 1), True
        expected = (self.nt, self.control_dimension)
        if arr.shape != expected:
            raise ValueError(f"Expected control shape {self.control_shape} or {expected}, got {arr.shape}")
        return arr, False

    def _broadcast_bound(self, bound: np.ndarray, name: str) -> np.ndarray:
        target = (self.nt, self.control_dimension)
        if bound.shape == () or bound.shape == (1,):
            return np.broadcast_to(bound, target)
        if self.control_dimension == 1 and bound.shape == (self.nt,):
            return bound.reshape(self.nt, 1)
        if bound.shape == (self.control_dimension,):
            return np.broadcast_to(bound.reshape(1, self.control_dimension), target)
        if bound.shape == target:
            return bound
        raise ValueError(f"Cannot broadcast {name} shape {bound.shape} to {target}")

    def _l1_weights(self) -> np.ndarray:
        widths = self.time_step_widths
        coeff = np.zeros(self.nt, dtype=float)
        coeff[0] = widths[1] * (1.0 - self.theta)
        if self.nt > 2:
            coeff[1:-1] = self.theta * widths[1:-1] + (1.0 - self.theta) * widths[2:]
        coeff[-1] = self.theta * widths[-1]
        return self.beta * coeff


def _reduced_dynamic_theta(config: NavierStokesConfig) -> float:
    reduced = config.raw.get("Reduced Dynamic Objective", {})
    if not isinstance(reduced, dict):
        return 1.0
    time_discretization = reduced.get("Time Discretization", {})
    if not isinstance(time_discretization, dict):
        return 1.0
    return float(time_discretization.get("Theta", 1.0))


def _l1_control_cost(config: NavierStokesConfig) -> float:
    return float(getattr(config, "l1_control_cost", _problem_value(config, "L1 Control Cost", 1e-2)))


def _lower_control_bound(config: NavierStokesConfig) -> float:
    return float(getattr(config, "lower_control_bound", _problem_value(config, "Lower Control Bound", -1.0)))


def _upper_control_bound(config: NavierStokesConfig) -> float:
    return float(getattr(config, "upper_control_bound", _problem_value(config, "Upper Control Bound", 1.0)))


def _problem_value(config: NavierStokesConfig, name: str, default: float) -> float:
    raw = getattr(config, "raw", {})
    if not isinstance(raw, dict):
        return default
    problem = raw.get("Problem", {})
    if not isinstance(problem, dict):
        return default
    return float(problem.get(name, default))
