import math
import sys

import torch

from pyrol.getTypeName import getTypeName


BoundConstraintBase = getTypeName("BoundConstraint")

if BoundConstraintBase is None:
    raise ImportError(
        "TorchBoundConstraint requires generated PyROL bindings that expose "
        "ROL::BoundConstraint. Rebuild pyrol.so after the PyROL_ETI.hpp "
        "update."
    )


def _flat_tensor(vector):
    if isinstance(vector, torch.Tensor):
        return vector.view(-1)

    flat = getattr(vector, "flat", None)
    if isinstance(flat, torch.Tensor):
        return flat.view(-1)

    tensor = getattr(vector, "tensor", None)
    if isinstance(tensor, torch.Tensor):
        return tensor.view(-1)

    torch_object = getattr(vector, "torch_object", None)
    if isinstance(torch_object, torch.Tensor):
        return torch_object.view(-1)

    raise TypeError(
        "Expected a torch.Tensor or a torch-backed PyROL vector with "
        "a tensor, flat, or torch_object tensor attribute."
    )


def _require_floating_tensor(tensor):
    if not tensor.is_floating_point():
        raise TypeError("Torch bound vectors must use a floating-point dtype.")


def _require_torch_vector(vector):
    if isinstance(vector, torch.Tensor):
        raise TypeError(
            "TorchBoundConstraint bounds must be PyROL torch vectors, such "
            "as TensorVector or TensorDictVector, not raw torch.Tensor objects."
        )
    _require_floating_tensor(_flat_tensor(vector))
    for name in ("clone", "dimension", "zero", "plus"):
        if not hasattr(vector, name):
            raise TypeError(
                "Torch bound vectors must implement the PyROL Vector "
                f"method {name!r}."
            )


def _set_vector(target, source):
    if hasattr(target, "set"):
        target.set(source)
    else:
        target.zero()
        target.plus(source)


def _as_bound_vector(value):
    _require_torch_vector(value)
    vector = value.clone()
    _set_vector(vector, value)
    return vector


class TorchBoundConstraint(BoundConstraintBase):
    """Torch implementation of ROL::StdBoundConstraint semantics."""

    def __init__(
        self,
        lower_or_upper,
        upper=None,
        *,
        isLower=False,
        scale=1.0,
        feasTol=None,
    ):
        super().__init__()
        self.deactivate()

        if isinstance(upper, bool):
            isLower = upper
            upper = None

        self.scale = float(scale)
        self.feasTol = (
            math.sqrt(sys.float_info.epsilon)
            if feasTol is None
            else float(feasTol)
        )

        bound = _as_bound_vector(lower_or_upper)
        self.dim = bound.dimension()

        if upper is None:
            self._lower = bound.clone()
            self._upper = bound.clone()
            if isLower:
                _set_vector(self._lower, bound)
                self._upper.setScalar(math.inf)
                self.activateLower()
            else:
                self._lower.setScalar(-math.inf)
                _set_vector(self._upper, bound)
                self.activateUpper()
            self.min_diff = math.inf
        else:
            self._lower = bound
            self._upper = _as_bound_vector(upper)
            self._check_bounds_compatible()
            self.activate()
            diff = self._upper_data() - self._lower_data()
            self.min_diff = 0.5 * diff.min().item()

    def getLowerBound(self):
        return self._lower

    def getUpperBound(self):
        return self._upper

    def project(self, x):
        with torch.no_grad():
            x_data = self._checked_data(x)
            if self.isLowerActivated():
                x_data.copy_(torch.maximum(x_data, self._lower_data()))
            if self.isUpperActivated():
                x_data.copy_(torch.minimum(x_data, self._upper_data()))

    def projectInterior(self, x):
        with torch.no_grad():
            x_data = self._checked_data(x)
            eps = self.feasTol
            tol = 100.0 * sys.float_info.epsilon

            if self.isLowerActivated():
                lower = self._lower_data()
                val = torch.where(
                    lower < -tol,
                    (1.0 - eps) * lower,
                    torch.where(lower > tol, (1.0 + eps) * lower, lower + eps),
                )
                val = torch.minimum(lower + eps * self.min_diff, val)
                x_data.copy_(torch.maximum(x_data, val))

            if self.isUpperActivated():
                upper = self._upper_data()
                val = torch.where(
                    upper < -tol,
                    (1.0 + eps) * upper,
                    torch.where(upper > tol, (1.0 - eps) * upper, upper - eps),
                )
                val = torch.maximum(upper - eps * self.min_diff, val)
                x_data.copy_(torch.minimum(x_data, val))

    def pruneUpperActive(self, v, *args):
        if not self.isUpperActivated():
            return

        with torch.no_grad():
            v_data = self._checked_data(v)
            upper = self._upper_data()

            if len(args) == 1:
                x, eps = args[0], 0.0
                mask = self._checked_data(x) >= upper - self._active_eps(eps)
            elif len(args) == 2:
                x, eps = args
                mask = self._checked_data(x) >= upper - self._active_eps(eps)
            elif len(args) == 4:
                g, x, xeps, geps = args
                mask = (
                    (self._checked_data(x) >= upper - self._active_eps(xeps))
                    & (self._checked_data(g) < -float(geps))
                )
            else:
                raise TypeError("Unexpected pruneUpperActive argument count.")

            v_data.masked_fill_(mask, 0.0)

    def pruneLowerActive(self, v, *args):
        if not self.isLowerActivated():
            return

        with torch.no_grad():
            v_data = self._checked_data(v)
            lower = self._lower_data()

            if len(args) == 1:
                x, eps = args[0], 0.0
                mask = self._checked_data(x) <= lower + self._active_eps(eps)
            elif len(args) == 2:
                x, eps = args
                mask = self._checked_data(x) <= lower + self._active_eps(eps)
            elif len(args) == 4:
                g, x, xeps, geps = args
                mask = (
                    (self._checked_data(x) <= lower + self._active_eps(xeps))
                    & (self._checked_data(g) > float(geps))
                )
            else:
                raise TypeError("Unexpected pruneLowerActive argument count.")

            v_data.masked_fill_(mask, 0.0)

    def isFeasible(self, x):
        with torch.no_grad():
            x_data = self._checked_data(x)
            lower_feasible = True
            upper_feasible = True

            if self.isLowerActivated():
                lower_feasible = torch.all(x_data >= self._lower_data()).item()
            if self.isUpperActivated():
                upper_feasible = torch.all(x_data <= self._upper_data()).item()

            return bool(lower_feasible and upper_feasible)

    def applyInverseScalingFunction(self, dv, v, x, g):
        with torch.no_grad():
            scaling = self._build_scaling_tensor(x, g)
            self._checked_data(dv).copy_(self._checked_data(v) / scaling)

    def applyScalingFunctionJacobian(self, dv, v, x, g):
        with torch.no_grad():
            x_data = self._checked_data(x)
            g_data = self._checked_data(g)
            v_data = self._checked_data(v)

            scaling = self._build_scaling_tensor(x, g)
            c = self._build_c()
            indicator = scaling < c

            d1prime = torch.sign(g_data)
            zero_grad = d1prime == 0
            lodiff = x_data - self._lower_data()
            updiff = self._upper_data() - x_data
            zero_grad_prime = torch.where(
                updiff < lodiff,
                torch.full_like(d1prime, -1.0),
                torch.ones_like(d1prime),
            )
            d1prime = torch.where(zero_grad, zero_grad_prime, d1prime)

            value = d1prime * g_data * v_data
            self._checked_data(dv).copy_(
                torch.where(indicator, value, torch.zeros_like(value))
            )

    def _lower_data(self):
        return _flat_tensor(self._lower)

    def _upper_data(self):
        return _flat_tensor(self._upper)

    def _check_bounds_compatible(self):
        lower = self._lower_data()
        upper = self._upper_data()
        if lower.numel() != upper.numel():
            raise ValueError("Lower and upper torch bounds have different dimensions.")
        if lower.dtype != upper.dtype:
            raise TypeError("Lower and upper torch bounds have different dtypes.")
        if lower.device != upper.device:
            raise ValueError("Lower and upper torch bounds are on different devices.")

    def _checked_data(self, vector):
        data = _flat_tensor(vector)
        if data.numel() != self.dim:
            raise ValueError("Torch vector dimension does not match the bounds.")
        if data.dtype != self._lower_data().dtype:
            raise TypeError("Torch vector dtype does not match the bounds.")
        if data.device != self._lower_data().device:
            raise ValueError("Torch vector device does not match the bounds.")
        return data

    def _active_eps(self, eps):
        return min(self.scale * float(eps), self.min_diff)

    def _build_c(self):
        return torch.minimum(
            0.5 * (self._upper_data() - self._lower_data()),
            torch.ones_like(self._lower_data()),
        )

    def _build_scaling_tensor(self, x, g):
        x_data = self._checked_data(x)
        g_data = self._checked_data(g)

        lower = self._lower_data()
        upper = self._upper_data()
        lodiff = x_data - lower
        updiff = upper - x_data
        c = self._build_c()

        active_lower = (-g_data > lodiff) & (lodiff <= updiff)
        active_upper = (g_data > updiff) & (updiff <= lodiff)
        binding_value = torch.minimum(torch.abs(g_data), c)
        interior_value = torch.minimum(torch.minimum(lodiff, updiff), c)

        return torch.where(active_lower | active_upper, binding_value, interior_value)


TorchBounds = TorchBoundConstraint

__all__ = ["TorchBoundConstraint", "TorchBounds"]
