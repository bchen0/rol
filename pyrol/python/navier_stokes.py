"""Dynamic Navier-Stokes objective facade for PyROL."""

import os

from ._navier_stokes import _NavierStokesObjective


class NavierStokesObjective:
    """Reduced dynamic Navier-Stokes objective backed by Trilinos."""

    def __init__(self, xml_path, cache_dir=None, spinup_time=None):
        xml_path = os.fspath(xml_path)
        cache_arg = "" if cache_dir is None else os.fspath(cache_dir)
        spinup_arg = -1.0 if spinup_time is None else float(spinup_time)
        self._impl = _NavierStokesObjective(xml_path, cache_arg, spinup_arg)

    @property
    def num_steps(self):
        return self._impl.num_steps

    @property
    def num_controls(self):
        return self._impl.num_controls

    @property
    def comm_rank(self):
        return self._impl.comm_rank

    @property
    def comm_size(self):
        return self._impl.comm_size

    def value(self, z, tol=1e-8):
        return self._impl.value(z, tol)

    def gradient(self, z, tol=1e-8):
        return self._impl.gradient(z, tol)

    def hess_vec(self, v, z, tol=1e-8):
        return self._impl.hess_vec(v, z, tol)

    def value_and_gradient(self, z, tol=1e-8):
        return self._impl.value_and_gradient(z, tol)
