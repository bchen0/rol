"""Axisymmetric flow-opt objective facades for PyROL."""

import os

import numpy as np

from ._flow_opt_brinkman import _BrinkmanObjective
from ._flow_opt_darcy import _DarcyObjective
from ._flow_opt_filtered_darcy import _FilteredDarcyObjective


class _FlowOptObjective:
    _impl_type = None

    def __init__(self, xml_path):
        self._impl = self._impl_type(os.fspath(xml_path))

    @property
    def comm_rank(self):
        return self._impl.comm_rank

    @property
    def comm_size(self):
        return self._impl.comm_size

    @property
    def local_field_size(self):
        return self._impl.local_field_size

    @property
    def global_field_size(self):
        return self._impl.global_field_size

    @property
    def parameter_size(self):
        return self._impl.parameter_size

    @property
    def local_size(self):
        return self._impl.local_size

    @property
    def uses_parameter_control(self):
        return self._impl.uses_parameter_control

    def value(self, z, tol=1e-8):
        return self._impl.value(z, tol)

    def gradient(self, z, tol=1e-8):
        return self._impl.gradient(z, tol)

    def hess_vec(self, v, z, tol=1e-8):
        return self._impl.hess_vec(v, z, tol)

    def value_and_gradient(self, z, tol=1e-8):
        return self._impl.value_and_gradient(z, tol)

    def gradient_dot(self, z, v, tol=1e-8):
        return self._impl.gradient_dot(z, v, tol)

    def hess_vec_dot(self, v, z, tol=1e-8):
        return self._impl.hess_vec_dot(v, z, tol)

    def default_control(self):
        return np.full(self.local_size, 0.5, dtype=float)

    initial_control = default_control


class DarcyObjective(_FlowOptObjective):
    """Reduced objective from flow-opt/axisymmetric/models/darcy."""

    _impl_type = _DarcyObjective


class BrinkmanObjective(_FlowOptObjective):
    """Reduced objective from flow-opt/axisymmetric/models/brinkman."""

    _impl_type = _BrinkmanObjective


class FilteredDarcyObjective(_FlowOptObjective):
    """Filtered reduced objective from flow-opt/axisymmetric/models/filteredDarcy."""

    _impl_type = _FilteredDarcyObjective
